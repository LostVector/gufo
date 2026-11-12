#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "src/families/qwen/dflash/hip/executor.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

namespace {
namespace qfn = gufo::models::qwen38_flash_next;
namespace cache = gufo::cache;
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
struct Guard : cache::MutationGuard {
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
};
struct Saved {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded,
          "component transfer failed");
}
Saved Capture(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
              gufo::hip::TransferPool& pool) {
  Saved saved{adapter.Positions(slot), {}};
  for (const auto& component : adapter.Components()) {
    const auto rows = saved.positions[component.id.value - 1].valid_rows;
    auto& bytes = saved.bytes.emplace_back(component.row_bytes
                                               ? rows * component.row_bytes
                                               : component.state_bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + 97);
        auto stream = pool.TryAcquire();
        Require(bool(stream), "no capture stream");
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *stream));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count =
            std::min<std::size_t>((1 << 20) + 13, bytes.size() - offset);
        auto stream = pool.TryAcquire();
        Require(bool(stream), "no private capture stream");
        Wait(adapter.CapturePrivatePiece(
            slot, component.id, offset, std::span(bytes).subspan(offset, count),
            *stream));
        offset += count;
      }
    }
  }
  return saved;
}
void Load(qfn::ContinuationAdapter& adapter, cache::Slot& slot,
          const Saved& saved, gufo::hip::TransferPool& pool,
          bool omit = false) {
  adapter.BeginRestore(slot, saved.positions);
  Require(!slot.IsValid(), "incomplete restore exposed executable state");
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const auto& bytes = saved.bytes[i];
    if (component.row_bytes) {
      for (cache::Rows end = saved.positions[i].valid_rows; end;) {
        const auto first = end > 97 ? end - 97 : 0;
        auto stream = pool.TryAcquire();
        Require(bool(stream), "no row load stream");
        Wait(adapter.CopyRowsIn(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *stream));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > (1 << 20) + 13 ? end - ((1 << 20) + 13) : 0;
        const auto count =
            end - first -
            (omit && i + 1 == adapter.Components().size() && first == 0 ? 1
                                                                        : 0);
        auto stream = pool.TryAcquire();
        Require(bool(stream), "no private load stream");
        Wait(adapter.LoadPrivatePiece(slot, component.id, first,
                                      std::span(bytes).subspan(first, count),
                                      *stream));
        end = first;
      }
    }
  }
}
std::vector<std::int32_t> Generate(qfn::Session& session, unsigned mode,
                                   std::string& error) {
  gufo::sampling::SamplingConfig config;
  config.seed = 517;
  if (mode) {
    config.temperature = 0.8F;
    config.top_k = mode == 1 ? 40 : 0;
    config.top_p = 0.95F;
    config.repeat_penalty = 1.1F;
  }
  gufo::sampling::SamplerState sampler(config);
  std::vector<std::int32_t> tokens;
  while (tokens.size() < 32) {
    qfn::Session::DecodeResult step;
    Require(
        session.DecodeDFlashStep(std::min<std::size_t>(8, 32 - tokens.size()),
                                 sampler, &step, &error, false),
        error);
    Require(!step.tokens.empty(), "component-backed DFlash made no progress");
    tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
  }
  return tokens;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc == 1) {
    std::fprintf(stderr, "SKIP: external model paths required\n");
    return 77;
  }
  try {
    Require(argc == 3 || (argc == 4 && std::string(argv[3]) == "adaptive"),
            "pass target first-shard and draft GGUF paths");
    std::string error;
    auto model = qfn::Model::Load(
        argv[1],
        {.max_context = 8192,
         .dflash_model_path = argv[2],
         .max_draft_tokens = 7,
         .dflash_policy = argc == 4
                              ? gufo::speculative::DFlashDraftPolicy::kAdaptive
                              : gufo::speculative::DFlashDraftPolicy::kFixed},
        &error);
    Require(bool(model), error);
    auto draft = model->LoadDFlashDraft(argv[2], &error);
    Require(bool(draft), error);
    gufo::hip::TransferPool pool(2);
    qfn::ContinuationAdapter adapter(model,
                                     gufo::core::SessionMode::kAutoregressive,
                                     8192, {'d', 'f', 'l', '2'}, draft);
    Require(adapter.PrivateScratchBytes() ==
                adapter.Components().back().state_bytes,
            "external component capacity is unaccounted");
    Guard ga, gb;
    auto source = adapter.CreateSlot(ga), destination = adapter.CreateSlot(gb);
    auto& a = adapter.GetSession(*source);
    auto& b = adapter.GetSession(*destination);
    Require(a.DFlashEnabled() && b.DFlashEnabled(),
            "component slot omitted draft attachment");
    const auto prompt = model->Tokenize(
        "Write a short Python function that adds two integers.\n");
    Require(adapter.Invalidate(*source) && adapter.Invalidate(*destination),
            "boundary setup invalidation failed");
    std::string text;
    for (unsigned i = 0; i < 500; ++i)
      text += "A small blue bird rests on a tree.\n";
    const auto long_prompt =
        model->Tokenize(text + "Summarize the scene in one sentence:");
    Saved earlier, boundary;
    const std::array<std::uint32_t, 2> boundaries{
        static_cast<std::uint32_t>(long_prompt.size() - 17),
        static_cast<std::uint32_t>(long_prompt.size() - 5)};
    std::vector<std::size_t> captured;
    adapter.PrefillThrough(
        *source, long_prompt, boundaries, [&](std::size_t i) {
          captured.push_back(i);
          (i == 0 ? earlier : boundary) = Capture(adapter, *source, pool);
        });
    Require(captured == std::vector<std::size_t>{0, 1},
            "external draft checkpoint order or count changed");
    const auto completed = Capture(adapter, *source, pool);
    Load(adapter, *destination, completed, pool);
    Require(adapter.Validate(*destination, completed.positions),
            "capture after boundary reused stale draft state");
    Load(adapter, *destination, boundary, pool);
    Require(adapter.Validate(*destination, boundary.positions),
            "component boundary validation failed");
    Require(b.Sync(long_prompt, &error), error);
    const auto expected = Generate(a, 0, error);
    Require(expected == Generate(b, 0, error),
            "component boundary+tail changed DFlash output");
    Load(adapter, *destination, earlier, pool);
    Require(adapter.Validate(*destination, earlier.positions),
            "earlier external draft checkpoint validation failed");
    Require(b.Sync(long_prompt, &error), error);
    Require(expected == Generate(b, 0, error),
            "earlier external draft checkpoint+tail changed output");
    for (unsigned mode = 0; mode < 3; ++mode) {
      Require(adapter.Invalidate(*source) && adapter.Invalidate(*destination),
              "slot invalidation failed");
      Require(a.Sync(prompt, &error), error);
      const auto saved = Capture(adapter, *source, pool);
      const auto expected = Generate(a, mode, error);
      Require(a.AllocatedBytes() <= model->DFlashSessionBytes(*draft, 8192),
              "DFlash state exceeds declared request memory");
      Load(adapter, *destination, saved, pool);
      Require(adapter.Validate(*destination, saved.positions),
              "component validation failed");
      Require(Generate(b, mode, error) == expected,
              "component restore changed DFlash tokens");
      const auto later_a = Capture(adapter, *source, pool),
                 later_b = Capture(adapter, *destination, pool);
      Require(later_a.bytes.back() == later_b.bytes.back(),
              "component draft state differs after replay");
      const auto learned = Generate(a, mode, error);
      Load(adapter, *destination, later_a, pool);
      Require(adapter.Validate(*destination, later_a.positions),
              "learned component state invalid");
      Require(Generate(b, mode, error) == learned,
              "learned component policy changed replay");
      Require(adapter.Invalidate(*destination), "slot reuse failed");
      auto corrupt = saved;
      corrupt.bytes.back()[sizeof(std::uint64_t)] ^= std::byte{0x80};
      Load(adapter, *destination, corrupt, pool);
      Require(!adapter.Validate(*destination, corrupt.positions) &&
                  !destination->IsValid(),
              "corrupt draft component became executable");
      Require(adapter.Invalidate(*destination), "corrupt slot recovery failed");
      Load(adapter, *destination, saved, pool, true);
      Require(!adapter.Validate(*destination, saved.positions),
              "incomplete private draft coverage accepted");
      Require(adapter.Invalidate(*destination), "partial slot recovery failed");
      Load(adapter, *destination, saved, pool);
      Require(adapter.Validate(*destination, saved.positions),
              "restore after failure failed");
      Require(Generate(b, mode, error) == expected,
              "failed restore poisoned later request");
      std::printf(
          "component mode=%u tokens=32 exact=1 private_pieces=1 reverse_load=1 "
          "corrupt_recovery=1 partial_recovery=1\n",
          mode);
      std::fflush(stdout);
    }
    std::printf(
        "PASS: DFlash component capture/restore including 5009-token boundary, "
        "sampled replay and failure recovery\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
