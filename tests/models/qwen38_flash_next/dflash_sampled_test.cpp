#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/families/qwen/dflash/hip/executor.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/dflash_features.hpp"

namespace {
namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;
void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
void Hip(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}
struct Buffer {
  explicit Buffer(std::size_t count) {
    Hip(hipMalloc(&ptr, count * sizeof(float)));
  }
  ~Buffer() { (void)hipFree(ptr); }
  float* ptr{};
};
struct Replay {
  std::vector<std::int32_t> tokens;
  std::vector<std::uint64_t> rng;
  bool operator==(const Replay&) const = default;
};
}  // namespace
int main(int argc, char** argv) {
  if (argc == 1) {
    std::fprintf(stderr, "SKIP: external model paths required\n");
    return 77;
  }
  try {
    Require(argc == 3, "pass target first-shard and DFlash2 GGUF paths");
    constexpr unsigned context = 1024, limit = 32;
    std::string error;
    auto oversized = qfn::Model::Load(argv[1],
                                      {.max_context = context,
                                       .dflash_model_path = argv[2],
                                       .max_draft_tokens = 8},
                                      &error);
    Require(!oversized && error.find("at most seven") != std::string::npos,
            "external drafter bypassed target verification width limit");
    error.clear();
    auto model = qfn::Model::Load(argv[1],
                                  {.max_context = context,
                                   .dflash_model_path = argv[2],
                                   .max_draft_tokens = 3},
                                  &error);
    Require(bool(model), error);
    auto draft_model = model->LoadDFlashDraft(argv[2], &error);
    Require(bool(draft_model), error);
    const auto& config = draft_model->GetDFlashConfig();
    const auto hidden = draft_model->GetConfig().hidden_size;
    const auto width = config.target_layer_ids.size() * hidden;
    std::size_t accepted = 0, rejected = 0;
    for (unsigned mode = 0; mode < 2; ++mode) {
      Replay first;
      for (unsigned repeat = 0; repeat < 3; ++repeat) {
        Buffer features(model->PrefillThroughCapacity() * width);
        auto ar = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       context, &error);
        auto target = model->CreateSession(
            gufo::core::SessionMode::kAutoregressive, context, &error);
        Require(bool(ar) && bool(target), error);
        auto draft = gufo::hip::QwenDFlashGpuExecutor::Create(draft_model,
                                                              context, &error);
        Require(bool(draft), error);
        target->SetTargetFeatureCapture(
            [&, destination = features.ptr](unsigned layer,
                                            const float* residual,
                                            unsigned rows, hipStream_t stream) {
              const auto tap = std::find(config.target_layer_ids.begin(),
                                         config.target_layer_ids.end(), layer);
              if (tap != config.target_layer_ids.end())
                qfn::rocm::CaptureDFlashFeatures(
                    residual, destination, rows, hidden,
                    model->config().hc_count, config.target_layer_ids.size(),
                    tap - config.target_layer_ids.begin(), stream);
            },
            [&](unsigned position, unsigned rows, std::string*) {
              if (position == 0)
                draft->Reset();
              std::vector<float> values(rows * width);
              Hip(hipMemcpy(values.data(), features.ptr,
                            values.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
              Require(std::all_of(values.begin(), values.end(),
                                  [](float v) { return std::isfinite(v); }),
                      "nonfinite target feature");
              return draft->InjectTargetContext(values, position, rows);
            });
        const auto prompt = model->Tokenize(
            "Write a short Python function that adds two integers.\n");
        Require(ar->Sync(prompt, &error) && target->Sync(prompt, &error),
                error);
        sampling::SamplingConfig cfg;
        cfg.temperature = 0.8F;
        cfg.top_k = mode == 0 ? 40 : 0;
        cfg.top_p = 0.95F;
        cfg.repeat_penalty = 1.1F;
        cfg.frequency_penalty = 0.03F;
        cfg.presence_penalty = 0.07F;
        cfg.seed = 513;
        sampling::SamplerState sampler(cfg), teacher(cfg);
        Replay replay;
        bool restored = false, checking_restore = false;
        std::vector<std::int32_t> restored_tokens;
        sampling::SamplerState::DrawState restored_draw{};
        while (replay.tokens.size() < limit) {
          if (repeat == 2) {
            qfn::Session::DecodeResult step;
            Require(target->DecodeDFlashStep(
                        *draft,
                        std::min<std::size_t>(4, limit - replay.tokens.size()),
                        sampler, &step, &error, false),
                    error);
            Require(!step.tokens.empty(),
                    "native DFlash cycle made no progress");
            replay.tokens.insert(replay.tokens.end(), step.tokens.begin(),
                                 step.tokens.end());
            replay.rng.push_back(sampler.rng_state());
            continue;
          }
          const bool checkpoint = !restored && replay.tokens.size() >= 12;
          std::unique_ptr<qfn::SessionSnapshot> saved_target, saved_ar;
          std::unique_ptr<gufo::hip::QwenDFlashGpuSnapshot> saved_draft;
          auto saved_sampler = sampler, saved_teacher = teacher;
          if (checkpoint) {
            saved_target = target->SaveSnapshot(
                &error, qfn::Session::SnapshotMode::kMaterialized);
            saved_ar = ar->SaveSnapshot(
                &error, qfn::Session::SnapshotMode::kMaterialized);
            saved_draft = draft->SaveSnapshot();
            Require(bool(saved_target) && bool(saved_ar) && bool(saved_draft),
                    error);
          }
          const auto anchor = sampler.Sample(target->Logits());
          Require(anchor == teacher.Sample(ar->Logits()),
                  "sampled anchor differs from scalar teacher");
          const auto count =
              std::min<std::size_t>(3, limit - replay.tokens.size() - 1);
          std::vector<float> uniforms(count);
          if (count) {
            auto rng = sampling::NextRandom(sampler.mutable_rng_state());
            Require(rng == sampling::NextRandom(teacher.mutable_rng_state()),
                    "draft RNG seed differs");
            for (auto& draw : uniforms)
              draw = static_cast<float>(sampling::Uniform(&rng));
          }
          auto proposal_sampler = sampler.WithoutConstraint();
          proposal_sampler.Accept(anchor);
          gufo::speculative::DraftProposal proposal;
          proposal.start_pos = target->Position();
          proposal.candidates_per_token = config.selector_top_k;
          proposal.tokens = draft->ForwardBlock(
              anchor, target->Position(), count, cfg.temperature, uniforms,
              nullptr, &proposal.candidate_ids,
              &proposal.candidate_probabilities, {}, &proposal_sampler);
          if (replay.tokens.empty() && count > 1) {
            auto malformed = proposal;
            malformed.candidate_probabilities.back() =
                std::numeric_limits<float>::quiet_NaN();
            const auto before = sampler.SaveDrawState();
            const auto position = target->Position();
            const std::vector<float> logits(target->Logits().begin(),
                                            target->Logits().end());
            qfn::Session::DecodeResult invalid;
            Require(!target->VerifySampledDraft(anchor, malformed, sampler,
                                                &invalid, &error, false),
                    "malformed later proposal was accepted");
            Require(position == target->Position() &&
                        std::equal(logits.begin(), logits.end(),
                                   target->Logits().begin()),
                    "invalid proposal changed target");
            Require(before.rng == sampler.SaveDrawState().rng &&
                        before.pending == sampler.SaveDrawState().pending,
                    "invalid proposal changed sampler");
            error.clear();
            malformed = proposal;
            malformed.tokens.clear();
            malformed.candidate_probabilities.clear();
            Require(!target->VerifySampledDraft(anchor, malformed, sampler,
                                                &invalid, &error, false),
                    "empty proposal with orphaned candidates was accepted");
            Require(position == target->Position() &&
                        std::equal(logits.begin(), logits.end(),
                                   target->Logits().begin()) &&
                        before.rng == sampler.SaveDrawState().rng &&
                        before.pending == sampler.SaveDrawState().pending,
                    "orphaned candidates changed target or sampler");
            error.clear();
          }
          qfn::Session::DecodeResult step;
          Require(target->VerifySampledDraft(anchor, proposal, sampler, &step,
                                             &error, false),
                  error);
          std::vector<std::int32_t> expected{static_cast<std::int32_t>(anchor)};
          teacher.Accept(anchor);
          Require(ar->Evaluate(anchor, &error), error);
          for (std::size_t row = 0; row < proposal.tokens.size(); ++row) {
            const auto ids = std::span(proposal.candidate_ids)
                                 .subspan(row * config.selector_top_k,
                                          config.selector_top_k);
            const auto q = std::span(proposal.candidate_probabilities)
                               .subspan(row * config.selector_top_k,
                                        config.selector_top_k);
            double selected = 0;
            for (std::size_t i = 0; i < ids.size(); ++i)
              if (ids[i] == proposal.tokens[row])
                selected += q[i];
            const auto p = teacher.Distribution(ar->Logits());
            if (teacher.Uniform() * selected >=
                p.probability(proposal.tokens[row])) {
              teacher.DeferSample(teacher.SampleResidual(ar->Logits(), ids, q));
              ++rejected;
              break;
            }
            const auto token = proposal.tokens[row];
            teacher.Accept(token);
            Require(ar->Evaluate(token, &error), error);
            expected.push_back(token);
            ++accepted;
          }
          Require(step.tokens == expected,
                  "sampled verification differs from scalar target teacher");
          Require(sampler.SaveDrawState().rng == teacher.SaveDrawState().rng &&
                      sampler.SaveDrawState().pending ==
                          teacher.SaveDrawState().pending,
                  "residual/RNG differs from teacher");
          Require(
              std::equal(sampler.history().begin(), sampler.history().end(),
                         teacher.history().begin(), teacher.history().end()),
              "sampled accepted history differs");
          if (checking_restore) {
            Require(
                step.tokens == restored_tokens &&
                    sampler.SaveDrawState().rng == restored_draw.rng &&
                    sampler.SaveDrawState().pending == restored_draw.pending,
                "coherent target/draft/sampler snapshot replay differs");
            checking_restore = false;
          }
          if (checkpoint) {
            restored_tokens = step.tokens;
            restored_draw = sampler.SaveDrawState();
            Require(target->RestoreSnapshot(*saved_target, &error) &&
                        ar->RestoreSnapshot(*saved_ar, &error),
                    error);
            draft->RestoreSnapshot(*saved_draft);
            sampler = saved_sampler;
            teacher = saved_teacher;
            restored = true;
            checking_restore = true;
            continue;
          }
          replay.tokens.insert(replay.tokens.end(), step.tokens.begin(),
                               step.tokens.end());
          replay.rng.push_back(sampler.rng_state());
        }
        Require((repeat == 2 || restored) && !checking_restore,
                "snapshot replay was not exercised");
        sampling::SamplerState terminal(cfg);
        terminal.DeferSample(model->EosToken());
        const auto stop = terminal.Sample(target->Logits());
        Require(model->IsStopToken(stop), "deferred stop anchor is not EOS");
        gufo::speculative::DraftProposal empty;
        empty.start_pos = target->Position();
        const auto position = target->Position();
        qfn::Session::DecodeResult stopped;
        Require(target->VerifySampledDraft(stop, empty, terminal, &stopped,
                                           &error, true),
                error);
        Require(stopped.stop && stopped.tokens.empty() &&
                    target->Position() == position &&
                    terminal.history().empty(),
                "stop anchor changed target or accepted history");
        terminal.DeferSample(stop);
        const auto before_stop = terminal.rng_state();
        Require(target->DecodeDFlashStep(*draft, 4, terminal, &stopped, &error,
                                         true),
                error);
        Require(stopped.stop && stopped.tokens.empty() &&
                    target->Position() == position &&
                    terminal.history().empty() &&
                    terminal.rng_state() == before_stop,
                "native DFlash cycle did not preserve deferred EOS semantics");
        if (repeat == 0)
          first = replay;
        else
          Require(first == replay,
                  "sampled process-local replay changed tokens or RNG");
        std::printf(
            "sampled mode=%u repeat=%u tokens=%zu exact_scalar_teacher=1 "
            "seeded_replay=1 snapshot_replay=%u deferred_eos=1\n",
            mode, repeat, replay.tokens.size(), repeat != 2);
        std::fflush(stdout);
      }
    }
    Require(accepted > 0 && rejected > 0,
            "sampled test did not exercise acceptance and residual rejection");
    std::printf("PASS: sampled DFlash2 accepted=%zu rejected=%zu\n", accepted,
                rejected);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
