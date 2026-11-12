#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/families/qwen/dflash/hip/executor.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/dflash_features.hpp"

namespace {
namespace qfn = gufo::models::qwen38_flash_next;
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}
struct Features {
  explicit Features(std::size_t size) {
    Hip(hipMalloc(&device, size * sizeof(float)));
  }
  ~Features() {
    if (device)
      (void)hipFree(device);
  }
  std::vector<float> Read(std::size_t size) const {
    std::vector<float> result(size);
    Hip(hipMemcpy(result.data(), device, size * sizeof(float),
                  hipMemcpyDeviceToHost));
    Require(std::all_of(result.begin(), result.end(),
                        [](float x) { return std::isfinite(x); }),
            "target features are non-finite");
    return result;
  }
  float* device{nullptr};
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      std::fprintf(stderr,
                   "SKIP: external target and draft model paths required\n");
      return 77;
    }
    Require(argc == 3,
            "pass target first-shard and FlashNext DFlash2 GGUF paths");
    constexpr unsigned context = 8192, output_limit = 32;
    std::string error;
    auto model = qfn::Model::Load(argv[1],
                                  {.max_context = context,
                                   .dflash_model_path = argv[2],
                                   .max_draft_tokens = 7},
                                  &error);
    Require(bool(model), error);
    auto draft_model = model->LoadDFlashDraft(argv[2], &error);
    Require(bool(draft_model), error);
    const auto& cfg = draft_model->GetDFlashConfig();
    Require(
        cfg.target_layer_ids == std::vector<std::uint32_t>({5, 15, 25, 35, 45}),
        "FlashNext completed-layer feature taps mismatch");
    Require(draft_model->GetConfig().hidden_size == 2560 &&
                draft_model->GetConfig().head_dim == 128 && cfg.num_layers == 5,
            "FlashNext DFlash checkpoint topology mismatch");
    const auto width =
        cfg.target_layer_ids.size() * draft_model->GetConfig().hidden_size;
    std::size_t all_proposed = 0, all_accepted = 0;
    unsigned case_index = 0;
    struct Case {
      std::string prompt;
      unsigned drafts;
      bool corrupt_first;
      bool top_one{false};
    };
    std::string long_prompt;
    for (unsigned i = 0; i < 500; ++i)
      long_prompt += "A small blue bird rests on a tree.\n";
    long_prompt += "Summarize the scene in one sentence:";
    for (const auto& test :
         {Case{"The capital of France is", 3, false},
          Case{"Write a short Python function that adds two integers.\n", 3,
               false},
          Case{"The first five prime numbers are", 3, false},
          Case{"The capital of France is", 1, false},
          Case{"The capital of France is", 7, false},
          Case{"The capital of France is", 7, true},
          Case{long_prompt, 3, false},
          Case{"The capital of France is", 3, false, true}}) {
      Features features(model->PrefillThroughCapacity() * width);
      auto ar = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     context, &error);
      auto target = model->CreateSession(
          gufo::core::SessionMode::kAutoregressive, context, &error);
      Require(bool(ar) && bool(target), error);
      auto draft = gufo::hip::QwenDFlashGpuExecutor::Create(draft_model,
                                                            context, &error);
      Require(bool(draft), error);
      std::size_t consumed_rows = 0, consumed_passes = 0;
      target->SetTargetFeatureCapture(
          [layers = cfg.target_layer_ids,
           hidden = draft_model->GetConfig().hidden_size,
           streams = model->config().hc_count,
           destination = features.device](unsigned layer, const float* residual,
                                          unsigned rows, hipStream_t stream) {
            const auto found = std::find(layers.begin(), layers.end(), layer);
            if (found != layers.end())
              qfn::rocm::CaptureDFlashFeatures(residual, destination, rows,
                                               hidden, streams, layers.size(),
                                               found - layers.begin(), stream);
          },
          [&](unsigned position, unsigned rows, std::string* failure) {
            if (position == 0) {
              draft->Reset();
              consumed_rows = 0;
            }
            Require(position == consumed_rows,
                    "committed feature stream is discontinuous");
            auto committed = features.Read(rows * width);
            if (!draft->InjectTargetContext(committed, position, rows)) {
              if (failure)
                *failure = "committed target-feature injection failed";
              return false;
            }
            consumed_rows += rows;
            ++consumed_passes;
            return true;
          });
      const auto tokens = model->Tokenize(test.prompt);
      std::printf("case=%u prompt_rows=%zu pass_capacity=%u\n", case_index,
                  tokens.size(), model->PrefillThroughCapacity());
      std::fflush(stdout);
      Require(ar->Sync(tokens, &error), error);
      gufo::sampling::SamplingConfig sampling;
      if (test.top_one) {
        sampling.temperature = 0.8F;
        sampling.top_k = 1;
      }
      gufo::sampling::SamplerState ar_sampler(sampling);
      std::vector<std::int32_t> expected;
      while (expected.size() < output_limit) {
        qfn::Session::DecodeResult step;
        Require(ar->DecodeStep(1, ar_sampler, &step, &error), error);
        expected.insert(expected.end(), step.tokens.begin(), step.tokens.end());
        if (step.stop)
          break;
      }
      Require(!expected.empty(), "AR control returned no tokens");
      if (test.prompt == long_prompt) {
        Require(target->Sync(std::span(tokens).first(17), &error), error);
        Require(consumed_rows == 17, "initial prefix features were lost");
        target->Reset();
        Require(target->Position() == 0, "target reset kept the old frontier");
        Require(target->Sync(std::span(tokens).first(31), &error), error);
        Require(consumed_rows == 31, "reset did not restart draft context");
      }
      Require(target->Sync(tokens, &error), error);
      Require(consumed_rows == tokens.size(), "prefill feature rows were lost");
      const auto prefill_passes = consumed_passes;
      Require(target->Sync(tokens, &error), error);
      Require(consumed_passes == prefill_passes,
              "unchanged prompt duplicated target features");
      if (test.prompt == long_prompt)
        Require(prefill_passes > 1,
                "long prompt did not exercise chunked prefill");
      gufo::sampling::SamplerState sampler(sampling);
      std::vector<std::int32_t> actual;
      std::size_t proposed = 0, accepted = 0, cycles = 0;
      while (actual.size() < expected.size()) {
        auto peek = sampler;
        const auto anchor =
            static_cast<std::int32_t>(peek.Sample(target->Logits()));
        Require(!model->IsStopToken(anchor),
                "DFlash target stopped before AR control");
        const auto base = target->Position();
        const auto remaining = expected.size() - actual.size();
        const auto count = std::min<std::size_t>(test.drafts, remaining - 1);
        const bool forced_reject = test.corrupt_first && cycles == 0;
        qfn::Session::DecodeResult step;
        if (forced_reject) {
          auto proposals = draft->ForwardBlock(anchor, base, count);
          std::vector<std::int32_t> chain{anchor};
          chain.insert(chain.end(), proposals.begin(), proposals.end());
          Require(chain.size() > 1, "forced rejection needs a proposal");
          chain[1] = (expected[1] + 1) % model->VocabSize();
          Require(target->VerifyGreedyDraft(chain, sampler, &step, &error),
                  error);
        } else {
          Require(target->DecodeDFlashStep(*draft, count + 1, sampler, &step,
                                           &error),
                  error);
        }
        Require(!step.tokens.empty(), "DFlash verification made no progress");
        if (forced_reject)
          Require(step.tokens.size() == 1,
                  "deliberately wrong proposal was accepted");
        actual.insert(actual.end(), step.tokens.begin(), step.tokens.end());
        Require(std::equal(actual.begin(), actual.end(), expected.begin()),
                "DFlash greedy output differs from AR control");
        proposed += count;
        accepted += step.tokens.size() - 1;
        ++cycles;
        if (step.stop)
          break;
        Require(consumed_rows == target->Position(),
                "committed feature frontier mismatch");
      }
      Require(actual == expected, "DFlash final greedy output differs from AR");
      all_proposed += proposed;
      all_accepted += accepted;
      std::printf(
          "case=%u tokens=%zu cycles=%zu proposed=%zu accepted=%zu width=%u "
          "forced_reject=%u top_one=%u prefill_rows=%zu prefill_passes=%zu "
          "exact=1\n",
          case_index++, actual.size(), cycles, proposed, accepted, test.drafts,
          test.corrupt_first, test.top_one, tokens.size(), prefill_passes);
      std::fflush(stdout);
    }
    Require(all_proposed > 0 && all_accepted > 0,
            "DFlash proposals were never accepted");
    std::printf(
        "PASS: FlashNext native DFlash2 greedy prototype; proposed=%zu "
        "accepted=%zu\n",
        all_proposed, all_accepted);
    return 0;
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "%s\n", exception.what());
    return 1;
  }
}
