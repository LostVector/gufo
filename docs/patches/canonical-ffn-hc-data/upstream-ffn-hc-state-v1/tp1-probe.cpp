#include "src/core/crypto/sha256.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::models::qwen38_flash_next { void FfnHcProbePlan(unsigned); }
namespace q = gufo::models::qwen38_flash_next;
namespace gufo::models::qwen38_flash_next::rocm {
void FfnHcProbeReset(bool);
std::uint64_t FfnHcProbeHeads(unsigned);
std::uint64_t FfnHcProbeProduced(unsigned);
} // namespace gufo::models::qwen38_flash_next::rocm
void Need(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::string Scores(const q::Session &session) {
  auto scores = session.Logits();
  Need(scores.size() == 248320 &&
           std::all_of(scores.begin(), scores.end(),
                       [](float x) { return std::isfinite(x); }),
       "complete finite target scores");
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t *>(scores.data()),
       scores.size_bytes()});
}
struct Record {
  std::vector<std::int32_t> tokens;
  std::string scores;
  std::uint32_t position;
  std::uint64_t rng, drafted, accepted;
  bool operator==(const Record &) const = default;
};
int main(int argc, char **argv) {
  try {
    Need(argc == 4, "target mtp output");
    std::string error;
    q::ModelOptions opt;
    opt.max_context = 65536;
    opt.decode_concurrency = 2;
    opt.max_draft_tokens = 7;
    opt.mtp_model_path = argv[2];
    auto model = q::Model::Load(argv[1], opt, &error);
    Need(bool(model), error);
    std::array<std::unique_ptr<q::Session>, 4> sessions;
    for (auto &s : sessions) {
      s = model->CreateSession(gufo::core::SessionMode::kSpeculative, 65536,
                               &error);
      Need(bool(s), error);
    }
    gufo::ReasoningOptions reason;
    reason.enabled = false;
    auto chat = gufo::tokenization::ResolveQwenChatOptions(reason);
    chat.max_output_bytes = gufo::tokenization::RenderedPromptBoundBytes(65536);
    std::array<std::vector<std::int32_t>, 2> short_prefix;
    for (unsigned actor = 0; actor < 2; ++actor) {
      std::string text = "Implement a robust timestamp parser and explain the "
                         "edge cases. Independent actor " +
                         std::to_string(actor);
      const std::array messages{gufo::tokenization::ChatMessage{
          gufo::tokenization::ChatRole::kUser, text}};
      auto ids = gufo::tokenization::QwenChatTemplate::RenderAndTokenize(
          model->tokenizer(), messages, chat, &error);
      Need(bool(ids), error);
      short_prefix[actor] = {ids->begin(), ids->end()};
    }
    auto pattern = model->Tokenize("Independent contexts retain sampling state "
                                   "during distributed inference. ");
    Need(!pattern.empty(), "pattern");
    std::ofstream output(argv[3]);
    Need(bool(output), "output");
    unsigned records = 0, gathers = 0, state_pairs = 0;
    for (unsigned depth : {0U, 32768U})
      for (bool sampled : {false, true}) {
        std::cout << "START depth=" << depth << " sampled=" << sampled
                  << std::endl;
        for (unsigned actor = 0; actor < 2; ++actor) {
          auto prefix = short_prefix[actor];
          if (depth) {
            auto suffix = prefix;
            prefix.resize(depth - suffix.size());
            for (unsigned i = 0; i < prefix.size(); ++i)
              prefix[i] = pattern[i % pattern.size()];
            prefix.insert(prefix.end(), suffix.begin(), suffix.end());
            Need(prefix.size() == depth, "actual32K");
          }
          sessions[actor]->Reset();
          Need(sessions[actor]->Sync(prefix, &error), error);
        }
        gufo::sampling::SamplingConfig ca{};
        ca.temperature = sampled ? .7F : 0.F;
        ca.top_k = 40;
        ca.top_p = .9F;
        ca.seed = 73;
        auto cb = ca;
        cb.seed = 91;
        std::array<gufo::sampling::SamplerState, 2> rng{
            gufo::sampling::SamplerState(ca), gufo::sampling::SamplerState(cb)};
        auto Run = [&](std::array<q::Session *, 2> actors, auto &random,
                       const std::string &phase, unsigned count, bool enabled) {
          std::vector<Record> result;
          for (unsigned step = 0; step < count; ++step) {
            q::rocm::FfnHcProbeReset(enabled);
            q::FfnHcProbePlan(step % 2 ? 0U : 1U);
            std::array<q::Session::DecodeResult, 2> decoded;
            std::array<q::Session::BatchOutcome, 2> outcomes;
            const std::array before{actors[0]->Statistics(),
                                    actors[1]->Statistics()};
            // The test-only engine hook supplies equal predeclared greedy
            // plans (1/0); sampled planning is unchanged. Record actual packed
            // rows, rather than assuming the fork's per-request caps.
            const std::array<q::Session::DecodeRequest, 2> requests{
                {{actors[0], 16, &random[0], &decoded[0], false, &outcomes[0]},
                 {actors[1], 16, &random[1], &decoded[1], false, &outcomes[1]}}};
            Need(q::Session::DecodeBatch(requests, &error), error);
            unsigned hc = 0, gc = 0;
            for (unsigned n = 0; n <= 64; ++n) {
              hc += q::rocm::FfnHcProbeHeads(n);
              gc += q::rocm::FfnHcProbeProduced(n);
            }
            Need(hc == 1, "one actual target verification batch");
            unsigned rows = 0;
            for (unsigned n = 1; n <= 64; ++n)
              if (q::rocm::FfnHcProbeHeads(n)) rows = n;
            const bool eligible = enabled && rows > 0 && rows <= 8;
            Need(rows > 0 && (eligible ?
                     (q::rocm::FfnHcProbeProduced(rows) > 0 &&
                      gc == q::rocm::FfnHcProbeProduced(rows)) : gc == 0),
                 "actual batch shape and exclusive FFN HC route/fallback activation");
            gathers += unsigned(enabled);
            for (unsigned actor = 0; actor < 2; ++actor) {
              Need(outcomes[actor].completed && !decoded[actor].tokens.empty(),
                   "both actors progress");
              auto after = actors[actor]->Statistics();
              Record x{decoded[actor].tokens,
                       Scores(*actors[actor]),
                       actors[actor]->Position(),
                       random[actor].rng_state(),
                       after.drafted - before[actor].drafted,
                       after.accepted - before[actor].accepted};
              result.push_back(x);
              output << "{\"depth\":" << depth
                     << ",\"sampled\":" << unsigned(sampled) << ",\"phase\":\""
                     << phase << "\",\"step\":" << step
                     << ",\"actor\":" << actor << ",\"rows\":" << rows
                     << ",\"fused_ffn_hc\":" << unsigned(enabled)
                     << ",\"position\":" << x.position << ",\"rng\":" << x.rng
                     << ",\"drafted\":" << x.drafted
                     << ",\"accepted\":" << x.accepted
                     << ",\"scores_sha256\":\"" << x.scores
                     << "\",\"tokens\":[";
              for (unsigned j = 0; j < x.tokens.size(); ++j) {
                if (j)
                  output << ',';
                output << x.tokens[j];
              }
              output << "]}\n";
              output.flush();
              ++records;
            }
          }
          return result;
        };
        const std::array<q::Session *, 2> original{sessions[0].get(),
                                                   sessions[1].get()},
            other{sessions[2].get(), sessions[3].get()};
        Run(original, rng, "warm", 3, true);
        std::array<std::unique_ptr<q::SessionSnapshot>, 2> borrowed, reference;
        std::array<std::vector<std::uint8_t>, 2> payload;
        std::array<unsigned, 2> base_position;
        for (unsigned actor = 0; actor < 2; ++actor) {
          borrowed[actor] = original[actor]->SaveSnapshot(
              &error, q::Session::SnapshotMode::kBorrowed);
          Need(bool(borrowed[actor]) &&
                   original[actor]->OwnsSnapshot(*borrowed[actor]),
               "borrowed ownership");
          auto bytes = borrowed[actor]->bytes();
          payload[actor] = {bytes.begin(), bytes.end()};
          Need(payload[actor].size() == borrowed[actor]->SizeBytes() &&
                   !payload[actor].empty(),
               "complete borrowed payload");
          base_position[actor] = original[actor]->Position();
          std::string file = std::string(argv[3]) + "." +
                             std::to_string(depth) + "-" +
                             std::to_string(sampled) + "-" +
                             std::to_string(actor) + ".snapshot.bin";
          std::ofstream f(file, std::ios::binary);
          f.write(reinterpret_cast<const char *>(payload[actor].data()),
                  payload[actor].size());
          Need(bool(f), "snapshot file write");
        }
        const auto base_rng = rng;
        auto expected = Run(original, rng, "full-reference", 2, false);
        for (unsigned actor = 0; actor < 2; ++actor) {
          reference[actor] = original[actor]->SaveSnapshot(
              &error, q::Session::SnapshotMode::kMaterialized);
          Need(bool(reference[actor]), error);
        }
        for (const std::string phase : {"borrowed-same", "borrowed-cross",
                                        "serialized-cross", "file-cross"}) {
          auto actors = phase == "borrowed-same" ? original : other;
          for (unsigned actor = 0; actor < 2; ++actor) {
            if (phase == "borrowed-same" || phase == "borrowed-cross")
              Need(actors[actor]->RestoreSnapshot(*borrowed[actor], &error),
                   error);
            else if (phase == "serialized-cross")
              Need(actors[actor]->RestoreSnapshot(payload[actor], &error),
                   error);
            else {
              std::string file = std::string(argv[3]) + "." +
                                 std::to_string(depth) + "-" +
                                 std::to_string(sampled) + "-" +
                                 std::to_string(actor) + ".snapshot.bin";
              std::ifstream f(file, std::ios::binary);
              std::vector<std::uint8_t> restored(
                  (std::istreambuf_iterator<char>(f)), {});
              Need(restored == payload[actor], "file bytes roundtrip");
              Need(actors[actor]->RestoreSnapshot(restored, &error), error);
            }
            Need(actors[actor]->Position() == base_position[actor],
                 "restored position");
          }
          auto state = base_rng;
          Need(Run(actors, state, phase, 2, true) == expected,
               "full tokens/scores/position/RNG/work replay: " + phase);
          for (unsigned actor = 0; actor < 2; ++actor) {
            auto end = actors[actor]->SaveSnapshot(
                &error, q::Session::SnapshotMode::kMaterialized);
            Need(bool(end), error);
            auto a = reference[actor]->bytes(), b = end->bytes();
            Need(a.size() == b.size() &&
                     std::equal(a.begin(), a.end(), b.begin()),
                 "complete materialized post-replay state: " + phase);
            std::cout << "STATE depth=" << depth << " sampled=" << sampled
                      << " phase=" << phase << " actor=" << actor
                      << " bytes=" << b.size() << " sha256="
                      << gufo::crypto::Sha256Hex(
                             {reinterpret_cast<const std::uint8_t *>(b.data()), b.size()})
                      << std::endl;
            ++state_pairs;
          }
          std::cout << "PASS depth=" << depth << " sampled=" << sampled
                    << " phase=" << phase << std::endl;
        }
      }
    Need(records == 104 && gathers == 44 && state_pairs == 32,
         "exact total coverage");
    std::cout << "PASS records=104 actual_ffn_hc_enabled_batches=44 full_state_pairs=32 "
                 "borrowed_same_cross_serialized_file_tokens_logits_RNG_work_"
                 "exact=true\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
