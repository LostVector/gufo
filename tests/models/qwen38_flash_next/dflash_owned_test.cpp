#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/families/qwen/dflash/hip/executor.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace {
namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;
void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Output {
  std::vector<std::int32_t> tokens;
  std::vector<std::uint64_t> rng;
  bool operator==(const Output&) const = default;
};
Output Generate(qfn::Session& session, sampling::SamplingConfig config,
                std::string& error) {
  sampling::SamplerState sampler(config);
  Output result;
  while (result.tokens.size() < 32) {
    qfn::Session::DecodeResult step;
    Require(session.DecodeDFlashStep(
                std::min<std::size_t>(8, 32 - result.tokens.size()), sampler,
                &step, &error, false),
            error);
    Require(!step.tokens.empty(), "owned DFlash made no progress");
    result.tokens.insert(result.tokens.end(), step.tokens.begin(),
                         step.tokens.end());
    result.rng.push_back(sampler.rng_state());
  }
  return result;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc == 1) {
    std::fprintf(stderr, "SKIP: external target/draft paths required\n");
    return 77;
  }
  try {
    Require(argc == 3 || (argc == 4 && std::string(argv[3]) == "adaptive"),
            "pass target first-shard and draft GGUF paths");
    std::weak_ptr<qfn::Model> owner;
    {
      std::string error;
      auto model = qfn::Model::Load(
          argv[1],
          {.max_context = 8192,
           .dflash_model_path = argv[2],
           .max_draft_tokens = 7,
           .dflash_policy =
               argc == 4 ? gufo::speculative::DFlashDraftPolicy::kAdaptive
                         : gufo::speculative::DFlashDraftPolicy::kFixed},
          &error);
      Require(bool(model), error);
      owner = model;
      auto draft = model->LoadDFlashDraft(argv[2], &error);
      Require(bool(draft), error);
      // The public shared factory must reject borrowed half tables before
      // packing weights or launching GPU work. Tiny host tables deliberately
      // differ from the artifact geometry, so removal of the donor check
      // fails this test safely at the later dimension check.
      auto draft_reader = gufo::core::GgufReader::OpenFile(argv[2], &error);
      Require(bool(draft_reader), error);
      std::shared_ptr<const gufo::core::GgufReader> reader(
          std::move(draft_reader));
      std::array<float, 32 * 32> host_table{};
      for (bool half_embedding : {false, true}) {
        gufo::hip::DFlashTarget synthetic;
        synthetic.config.hidden_size = 32;
        synthetic.config.vocab_size = 32;
        synthetic.token_embedding = {host_table.data(),
                                     half_embedding
                                         ? gufo::core::GgmlType::kF16
                                         : gufo::core::GgmlType::kF32,
                                     host_table.size(), sizeof(host_table)};
        synthetic.output = {host_table.data(),
                            half_embedding ? gufo::core::GgmlType::kF32
                                           : gufo::core::GgmlType::kF16,
                            host_table.size(), sizeof(host_table)};
        synthetic.tokenizer = &model->tokenizer();
        synthetic.owner = model;
        Require(
            !gufo::hip::QwenDFlashGpuModel::Create(reader, synthetic, &error),
            "shared factory accepted an F16 donor");
        Require(error.find(half_embedding ? "token embedding" : "output") !=
                        std::string::npos &&
                    error.find("F16") != std::string::npos,
                "shared factory did not reject the F16 donor before packing");
        error.clear();
      }
      auto ar = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     8192, &error);
      auto session = model->CreateSession(
          gufo::core::SessionMode::kAutoregressive, 8192, &error, draft);
      Require(bool(ar) && bool(session) && session->DFlashEnabled(), error);
      Require(session->AllocatedBytes() > ar->AllocatedBytes(),
              "owned draft memory is unaccounted");
      Require(!session->EnableDFlash(draft, &error),
              "duplicate draft attachment accepted");
      error.clear();
      const auto prompt = model->Tokenize(
          "Write a short Python function that adds two integers.\n");
      Require(ar->Sync(prompt, &error) && session->Sync(prompt, &error), error);
      auto target_only = ar->SaveSnapshot(&error);
      Require(bool(target_only), error);
      for (unsigned mode = 0; mode < 3; ++mode) {
        session->Reset();
        Require(session->Sync(prompt, &error), error);
        auto saved = session->SaveSnapshot(&error);
        Require(bool(saved), error);
        Require(saved->SizeBytes() == session->SnapshotBytes(),
                "snapshot size omits draft payload");
        std::vector<std::uint8_t> payload(saved->SizeBytes());
        Require(saved->CopyTo(payload), "owned snapshot serialization failed");
        sampling::SamplingConfig config;
        config.seed = 513;
        if (mode) {
          config.temperature = 0.8F;
          config.top_k = mode == 1 ? 40 : 0;
          config.top_p = 0.95F;
          config.seed = 513;
          config.repeat_penalty = 1.1F;
        }
        const auto first = Generate(*session, config, error);
        if (mode == 0) {
          sampling::SamplerState greedy;
          std::vector<std::int32_t> expected;
          while (expected.size() < 32) {
            qfn::Session::DecodeResult step;
            Require(ar->DecodeStep(1, greedy, &step, &error, false), error);
            expected.insert(expected.end(), step.tokens.begin(),
                            step.tokens.end());
          }
          Require(first.tokens == expected,
                  "owned greedy output differs from AR");
        }
        auto learned = session->SaveSnapshot(&error);
        Require(bool(learned), error);
        if (argc == 4) {
          const std::array<std::uint8_t, 8> policy_magic{'Q', 'F', 'N', 'D',
                                                         'F', 'L', '0', '1'};
          const auto bytes = learned->bytes();
          auto at = std::search(bytes.begin(), bytes.end(),
                                policy_magic.begin(), policy_magic.end());
          Require(at != bytes.end(), "adaptive snapshot lacks policy header");
          float mean;
          std::memcpy(&mean, &*at + 24, sizeof(mean));
          Require(mean != 3.0F,
                  "adaptive snapshot did not capture learned acceptance state");
        }
        const auto learned_output = Generate(*session, config, error);
        Require(session->RestoreSnapshot(*learned, &error), error);
        Require(learned_output == Generate(*session, config, error),
                "learned policy snapshot did not preserve tokens/cycle RNG");
        session->Reset();
        Require(session->RestoreSnapshot(*saved, &error), error);
        const auto second = Generate(*session, config, error);
        if (!(first == second)) {
          std::fprintf(stderr,
                       "replay mode=%u first_cycles=%zu second_cycles=%zu "
                       "tokens_equal=%u\n",
                       mode, first.rng.size(), second.rng.size(),
                       first.tokens == second.tokens);
          for (std::size_t i = 0; i < first.tokens.size(); ++i)
            std::fprintf(stderr, "token[%zu]=%d/%d\n", i, first.tokens[i],
                         second.tokens[i]);
        }
        Require(first == second, "owned borrowed snapshot replay differs");
        auto fork = model->CreateSession(
            gufo::core::SessionMode::kAutoregressive, 8192, &error, draft);
        Require(bool(fork), error);
        Require(!fork->RestoreSnapshot(*target_only, &error) &&
                    fork->Position() == 0,
                "target-only snapshot restored into external-draft session");
        error.clear();
        Require(!ar->RestoreSnapshot(payload, &error),
                "external snapshot restored into plain AR");
        error.clear();
        const std::array<std::uint8_t, 8> magic{'G', 'D', 'F', 'G',
                                                'P', 'U', '0', '1'};
        auto corrupted = payload;
        auto found = std::search(corrupted.begin(), corrupted.end(),
                                 magic.begin(), magic.end());
        Require(found != corrupted.end(),
                "owned snapshot lacks versioned draft payload");
        *found ^= 0x80;
        Require(
            !fork->RestoreSnapshot(corrupted, &error) && fork->Position() == 0,
            "malformed draft snapshot changed target");
        error.clear();
        Require(fork->RestoreSnapshot(payload, &error), error);
        Require(first == Generate(*fork, config, error),
                "owned serialized fork replay differs");
        auto small = model->CreateSession(
            gufo::core::SessionMode::kAutoregressive, 1024, &error, draft);
        Require(bool(small), error);
        Require(
            !small->RestoreSnapshot(payload, &error) && small->Position() == 0,
            "incompatible draft context restored or changed target");
        error.clear();
        std::printf(
            "owned mode=%u tokens=%zu borrowed=1 serialized_fork=1 "
            "incompatible=1 corrupt=1\n",
            mode, first.tokens.size());
        std::fflush(stdout);
      }
      std::string long_prompt;
      for (unsigned i = 0; i < 500; ++i)
        long_prompt += "A small blue bird rests on a tree.\n";
      const auto tokens =
          model->Tokenize(long_prompt + "Summarize the scene in one sentence:");
      session->Reset();
      ar->Reset();
      std::unique_ptr<qfn::SessionSnapshot> checkpoint;
      const auto boundary = tokens.size() - 5;
      Require(session->SyncThrough(tokens, boundary, &checkpoint, &error),
              error);
      Require(bool(checkpoint) && session->Position() == tokens.size(),
              "owned prefill checkpoint failed");
      auto restored = model->CreateSession(
          gufo::core::SessionMode::kAutoregressive, 8192, &error, draft);
      Require(bool(restored), error);
      Require(restored->RestoreSnapshot(*checkpoint, &error) &&
                  restored->Position() == boundary,
              error);
      Require(restored->Sync(tokens, &error), error);
      Require(ar->Sync(tokens, &error), error);
      const auto actual = Generate(*restored, {}, error);
      sampling::SamplerState greedy;
      std::vector<std::int32_t> expected;
      while (expected.size() < 32) {
        qfn::Session::DecodeResult step;
        Require(ar->DecodeStep(1, greedy, &step, &error, false), error);
        expected.insert(expected.end(), step.tokens.begin(), step.tokens.end());
      }
      Require(actual.tokens == expected,
              "owned boundary restore+tail differs from AR");
      std::printf(
          "owned long_rows=%zu boundary=%zu restore_and_tail=1 exact_ar=1\n",
          tokens.size(), boundary);
      std::fflush(stdout);
    }
    Require(owner.expired(),
            "target/draft/session ownership forms a reference cycle");
    std::printf(
        "PASS: owned DFlash sessions, coherent persistent snapshots, "
        "checkpoint boundary and no reference cycle\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
