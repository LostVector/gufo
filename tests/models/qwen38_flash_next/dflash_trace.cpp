#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/families/qwen/dflash/hip/executor.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/dflash_features.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"

namespace {
namespace qfn = gufo::models::qwen38_flash_next;
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}
struct Buffer {
  explicit Buffer(std::size_t size) {
    Hip(hipMalloc(&data, size * sizeof(float)));
  }
  ~Buffer() { (void)hipFree(data); }
  float* data{};
};
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 4 || argc == 6 || argc == 7,
            "pass target first-shard, draft GGUF, NEW trace directory "
            "[prefix-token-count draft-count [rejection|presence|features]]");
    const auto prefix_value = argc >= 6 ? std::stoul(argv[4]) : 24UL;
    const auto draft_value = argc >= 6 ? std::stoul(argv[5]) : 3UL;
    Require(argc != 7 || std::string_view(argv[6]) == "rejection" ||
                std::string_view(argv[6]) == "presence" ||
                std::string_view(argv[6]) == "features",
            "unknown trace scenario");
    Require(prefix_value > 0 && prefix_value <= 8000 && draft_value > 0 &&
                draft_value <= 7,
            "bounded trace requires 1..8000 prompt rows and 1..7 drafts");
    const auto prefix_rows = static_cast<unsigned>(prefix_value);
    const auto draft_count = static_cast<unsigned>(draft_value);
    Require(argc != 7 || std::string_view(argv[6]) != "features" ||
                prefix_rows == 1,
            "scalar feature qualification is bounded to one token");
    const auto context = std::max(128U, prefix_rows + 16);
    std::filesystem::path directory(argv[3]);
    Require(std::filesystem::create_directories(directory),
            "trace directory exists");
    const gufo::hip::DFlashTrace write = [&](std::string_view name,
                                             std::span<const float> values) {
      Require(std::all_of(values.begin(), values.end(),
                          [](float value) { return std::isfinite(value); }),
              "nonfinite trace stage " + std::string(name));
      std::ofstream file(directory / (std::string(name) + ".f32"),
                         std::ios::binary);
      file.write(reinterpret_cast<const char*>(values.data()),
                 values.size_bytes());
      Require(file.good(), "write trace stage " + std::string(name));
    };
    std::string error;
    auto model = qfn::Model::Load(argv[1],
                                  {.max_context = context,
                                   .dflash_model_path = argv[2],
                                   .max_draft_tokens = 7},
                                  &error);
    Require(bool(model), error);
    auto draft_model = model->LoadDFlashDraft(argv[2], &error);
    Require(bool(draft_model), error);
    auto target = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       context, &error);
    auto draft =
        gufo::hip::QwenDFlashGpuExecutor::Create(draft_model, context, &error);
    Require(bool(target) && bool(draft), error);
    const auto& cfg = draft_model->GetDFlashConfig();
    const auto hidden = draft_model->GetConfig().hidden_size;
    const auto width = hidden * cfg.target_layer_ids.size();
    Buffer features(model->PrefillThroughCapacity() * width);
    std::vector<float> captured;
    target->SetTargetFeatureCapture(
        [&](unsigned layer, const float* residual, unsigned rows,
            hipStream_t stream) {
          const auto tap = std::find(cfg.target_layer_ids.begin(),
                                     cfg.target_layer_ids.end(), layer);
          if (tap != cfg.target_layer_ids.end())
            qfn::rocm::CaptureDFlashFeatures(
                residual, features.data, rows, hidden, model->config().hc_count,
                cfg.target_layer_ids.size(), tap - cfg.target_layer_ids.begin(),
                stream);
        },
        [&](unsigned position, unsigned rows, std::string*) {
          Require(position * width == captured.size(),
                  "discontinuous target capture");
          const auto before = captured.size();
          captured.resize(before + rows * width);
          Hip(hipMemcpy(captured.data() + before, features.data,
                        rows * width * sizeof(float), hipMemcpyDeviceToHost));
          return true;
        });
    const std::string sentence =
        "Virtual memory gives each process its own address space. The "
        "operating "
        "system maps virtual pages to physical memory and handles page "
        "faults. ";
    std::string prompt = sentence;
    auto tokens = model->Tokenize(prompt);
    while (tokens.size() < prefix_rows) {
      prompt += sentence;
      tokens = model->Tokenize(prompt);
    }
    tokens.resize(prefix_rows);
    Require(target->Sync(tokens, &error), error);
    gufo::sampling::SamplingConfig sampling;
    sampling.temperature = 0;
    sampling.seed = 1;
    gufo::sampling::SamplerState sampler(sampling);
    auto anchor = sampler.Sample(target->Logits());
    write("target_features", captured);
    Require(draft->InjectTargetContext(captured, 0, tokens.size(), write),
            "inject target context");
    const std::array<float, 7> draws{0.13F, 0.37F, 0.71F, 0.21F,
                                     0.59F, 0.83F, 0.43F};
    const auto uniforms = std::span(draws).first(draft_count);
    float presence_penalty = 0;
    const auto capture = [&] {
      write("target_features", captured);
      gufo::sampling::SamplingConfig policy_config;
      policy_config.temperature = 1;
      policy_config.presence_penalty = presence_penalty;
      gufo::sampling::SamplerState policy(policy_config);
      for (auto token : tokens)
        policy.Accept(token);
      policy.Accept(anchor);
      std::vector<float> probabilities, confidences;
      std::vector<gufo::tokenization::TokenId> candidates;
      const auto proposed = draft->ForwardBlock(
          anchor, tokens.size(), draft_count, 1.0F, uniforms, &confidences,
          &candidates, &probabilities, write,
          presence_penalty ? &policy : nullptr);
      write("probabilities", probabilities);
      write("selected_probabilities", confidences);
      write("uniforms", uniforms);
      auto metadata = gufo::json::Value::object();
      const auto array = [&](const char* key, const auto& values) {
        metadata[key] = gufo::json::Value::array();
        for (auto value : values)
          metadata[key].push_back(static_cast<double>(value));
      };
      array("prompt_tokens", tokens);
      array("target_layer_ids", cfg.target_layer_ids);
      array("proposed", proposed);
      array("candidates", candidates);
      metadata["anchor"] = static_cast<double>(anchor);
      metadata["position"] = tokens.size();
      metadata["draft_count"] = proposed.size();
      metadata["temperature"] = 1.0;
      metadata["presence_penalty"] = presence_penalty;
      auto history = tokens;
      history.push_back(anchor);
      array("penalty_history", history);
      metadata["block_activation_dtype"] = "fp32";
      metadata["head_type"] =
          static_cast<double>(draft_model->GetWeights().output.type);
      if (argc == 7 && std::string_view(argv[6]) == "presence") {
        auto teacher = model->CreateSession(
            gufo::core::SessionMode::kAutoregressive, context, &error);
        Require(bool(teacher), error);
        auto conditioned = tokens;
        conditioned.push_back(anchor);
        for (unsigned row = 0; row < draft_count; ++row) {
          Require(teacher->Sync(conditioned, &error), error);
          write("target_logits." + std::to_string(row), teacher->Logits());
          conditioned.push_back(proposed[row]);
        }
        metadata["target_sampling"] = gufo::json::Value::object();
        metadata["target_sampling"]["temperature"] = 1.0;
        metadata["target_sampling"]["top_p"] = 0.95;
        metadata["target_sampling"]["top_k"] = 20;
      }
      std::ofstream file(directory / "trace.json");
      file << metadata.dump() << '\n';
      Require(file.good(), "write trace manifest");
      return std::pair{proposed, probabilities};
    };
    const auto initial = capture();
    if (argc == 7 && std::string_view(argv[6]) == "features") {
      auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
      Require(bool(reader), error);
      auto weights = qfn::ModelWeights::Bind(*reader, &error);
      Require(bool(weights), error);
      std::unique_ptr<qfn::NgramTable> ngram;
      if (weights->config.ple_layer >= 0) {
        const auto& table = weights->ple_table;
        ngram = qfn::NgramTable::Open(
            reader->GetMappedRegions()[table.shard].file_descriptor,
            table.file_offset, table.rows, weights->config.ple_head_dim,
            table.type, &error);
        Require(bool(ngram), error);
      }
      for (auto storage : {qfn::ReferenceModel::Storage::kFloat32,
                           qfn::ReferenceModel::Storage::kDecode}) {
        qfn::ReferenceModel reference(*weights, ngram.get(), context, storage);
        std::vector<float> expected(width);
        Require(reference.Step(
                    tokens.front(), {}, {}, &error,
                    [&](unsigned layer, std::span<const float> residual) {
                      auto tap = std::find(cfg.target_layer_ids.begin(),
                                           cfg.target_layer_ids.end(), layer);
                      if (tap == cfg.target_layer_ids.end())
                        return;
                      const auto index = tap - cfg.target_layer_ids.begin();
                      for (unsigned channel = 0; channel < hidden; ++channel) {
                        double mean = 0;
                        for (unsigned stream = 0;
                             stream < model->config().hc_count; ++stream)
                          mean += residual[stream * hidden + channel];
                        expected[index * hidden + channel] =
                            mean / model->config().hc_count;
                      }
                    }),
                error);
        write(storage == qfn::ReferenceModel::Storage::kFloat32
                  ? "target_features.scalar_f32"
                  : "target_features.scalar_decode",
              expected);
      }
    }
    if (argc == 7 && std::string_view(argv[6]) != "features") {
      const auto initial_directory = directory;
      const auto branch = [&](const std::filesystem::path& previous,
                              std::string_view name) {
        directory = initial_directory / name;
        Require(std::filesystem::create_directory(directory),
                "trace branch exists");
        for (const auto& file : std::filesystem::directory_iterator(previous)) {
          if (file.is_regular_file() &&
              file.path().filename().string().starts_with("context."))
            std::filesystem::copy_file(file.path(),
                                       directory / file.path().filename());
        }
      };
      if (std::string_view(argv[6]) == "rejection") {
        const auto before = tokens.size();
        std::vector<std::int32_t> chain{
            static_cast<std::int32_t>(anchor),
            static_cast<std::int32_t>(cfg.mask_token_id)};
        chain.insert(chain.end(), initial.first.begin() + 1,
                     initial.first.end());
        qfn::Session::DecodeResult rejected;
        Require(
            target->VerifyGreedyDraft(chain, sampler, &rejected, &error, false),
            error);
        Require(
            rejected.tokens.size() == 1 && target->Position() == before + 1 &&
                captured.size() == (before + 1) * width,
            "trace did not force first-proposal rejection and one-row commit");
        tokens.push_back(anchor);
        anchor = sampler.Sample(target->Logits());
        branch(initial_directory, "after-rejection");
        Require(
            draft->InjectTargetContext(
                std::span(captured).subspan(before * width), before, 1, write),
            "inject committed anchor");
        const auto incremental = capture();
        auto snapshot = draft->SaveSnapshot();
        std::vector<std::uint8_t> payload(snapshot->PersistentPayloadBytes());
        Require(snapshot->SerializePersistent(payload) == payload.size(),
                "serialize draft snapshot");
        auto restored = gufo::hip::QwenDFlashGpuExecutor::Create(
            draft_model, context, &error);
        Require(bool(restored), error);
        restored->RestorePersistentSnapshot(payload);
        draft = std::move(restored);
        const auto previous = directory;
        branch(previous, "after-restore");
        Require(capture() == incremental,
                "persistent draft restore changed proposals or q");
      }
      const auto previous = directory;
      branch(previous, "with-presence-penalty");
      presence_penalty = 1.5F;
      const auto penalized = capture();
      if (std::string_view(argv[6]) == "presence")
        Require(
            penalized.second != initial.second,
            "presence-policy trace did not change any proposal probabilities");
    }
    std::puts("PASS: real Flash Next feature/draft trace captured");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
