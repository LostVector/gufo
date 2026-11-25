#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/sampling.hpp"
#include "src/families/qwen/kernels/ops/token.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

struct DeviceDelete {
  void operator()(void* pointer) const { (void)hipFree(pointer); }
};

template<typename T>
auto Allocate(std::size_t count) {
  T* pointer = nullptr;
  CheckHip(hipMalloc(&pointer, count * sizeof(T)), "allocate");
  return std::unique_ptr<T, DeviceDelete>(pointer);
}

void CheckPenaltyArgmax(std::uint32_t vocab) {
  namespace sampling = gufo::sampling;
  // Ragged vocabularies end at the allocation boundary, with every MTP width.
  for (unsigned rows = 1; rows <= 7; ++rows) {
    for (unsigned mode = 0; mode < 5; ++mode) {
      sampling::SamplingConfig config{
          .seed = 73,
          .repeat_penalty = mode == 1 ? 0.7F : 1.3F,
          .repeat_last_n = mode == 2 ? 0U : 3U,
          .frequency_penalty = mode == 1 ? -0.4F : 0.2F,
          .presence_penalty = mode == 1 ? -1.5F : 1.5F};
      if (mode >= 3) {
        config.repeat_penalty = 1;
        config.frequency_penalty = 0;
        config.presence_penalty = 1.0e-8F;
      }
      sampling::SamplerState sampler(config,
                                     std::vector<sampling::TokenId>{0, 1, 0});
      sampler.Accept(0);
      sampler.Accept(vocab - 1);
      std::vector<float> logits(std::size_t(rows) * vocab);
      std::vector<sampling::TokenPenalty> counts;
      q::GreedyPenaltyRows batch{};
      std::vector<int> expected(rows, -1);
      for (unsigned row = 0; row < rows; ++row) {
        auto values =
            std::span(logits).subspan(std::size_t(row) * vocab, vocab);
        for (unsigned i = 0; i < vocab; ++i)
          values[i] =
              float(int((i * 7919U + row * 31U) % 65521U) - 32768) / 32768;
        values[0] = values[vocab - 1] = mode == 2 ? -1.e35F : 1.0F;
        if (mode == 2)
          std::fill(values.begin(), values.end(), -1.e35F);
        if (mode >= 3) {
          std::fill(values.begin(), values.end(), -100.0F);
          // FP32 penalty arithmetic incorrectly picks token zero here.
          values[0] = values[1] = 1.0F;
          values[vocab - 1] = std::numeric_limits<float>::infinity();
        }
        if (mode == 4)
          std::fill(values.begin(), values.end(),
                    std::numeric_limits<float>::quiet_NaN());
        const auto penalties = sampler.penalties();
        counts.insert(counts.end(), penalties.begin(), penalties.end());
        batch.offsets[row + 1] = counts.size();
        try {
          expected[row] = sampler.Sample(values);
        } catch (const std::runtime_error&) {
          if (mode != 4)
            throw;
        }
        sampler.Accept((row + 2) % vocab);
      }
      auto device_logits = Allocate<float>(logits.size());
      auto device_counts = Allocate<sampling::TokenPenalty>(counts.size());
      auto partial =
          Allocate<q::PenaltyArgmaxCandidate>(rows * q::kArgmaxParts);
      auto output = Allocate<q::ArgmaxCandidate>(rows);
      CheckHip(hipMemcpy(device_logits.get(), logits.data(),
                         logits.size() * sizeof(float), hipMemcpyHostToDevice),
               "penalty logits");
      CheckHip(
          hipMemcpy(device_counts.get(), counts.data(),
                    counts.size() * sizeof(counts[0]), hipMemcpyHostToDevice),
          "penalty histories");
      batch.penalties = device_counts.get();
      q::PenalizedArgmax(device_logits.get(), batch, config.repeat_penalty,
                         config.frequency_penalty, config.presence_penalty,
                         partial.get(), output.get(), rows, vocab, nullptr);
      std::vector<q::ArgmaxCandidate> actual(rows);
      CheckHip(hipMemcpy(actual.data(), output.get(), rows * sizeof(actual[0]),
                         hipMemcpyDeviceToHost),
               "penalty predictions");
      for (unsigned row = 0; row < rows; ++row) {
        if (actual[row].index != expected[row] ||
            (expected[row] < 0 && std::isfinite(actual[row].value)))
          throw std::runtime_error("penalty argmax differs from CPU: vocab=" +
                                   std::to_string(vocab) +
                                   " row=" + std::to_string(row) +
                                   " mode=" + std::to_string(mode));
      }
      std::vector<float> retained(logits.size());
      CheckHip(
          hipMemcpy(retained.data(), device_logits.get(),
                    retained.size() * sizeof(float), hipMemcpyDeviceToHost),
          "retained logits");
      if (std::memcmp(retained.data(), logits.data(),
                      logits.size() * sizeof(float)))
        throw std::runtime_error("penalty selection modified frontier logits");
    }
  }
  std::cout << "penalty argmax CPU exact: vocab=" << vocab << " widths=1..7\n";
}

void CheckArgmax(std::uint32_t vocab) {
  constexpr std::uint32_t rows = 10;
  constexpr std::int32_t guard = -1234567;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> logits(static_cast<std::size_t>(rows) * vocab);
  for (std::uint32_t t = 0; t < rows; ++t) {
    float* row = logits.data() + static_cast<std::size_t>(t) * vocab;
    for (std::uint32_t i = 0; i < vocab; ++i) {
      row[i] = -static_cast<float>((i * 7919U + t * 31U) % 65521U);
    }
    switch (t) {
      case 0:
        row[vocab - 1] = 10.0F;
        break;
      case 1:
        row[0] = row[vocab - 1] = 10.0F;
        break;
      case 2:
        row[vocab / 3] = row[vocab - 1] = inf;
        break;
      case 3:
        std::fill_n(row, vocab, -inf);
        break;
      case 4:
        row[0] = nan;
        break;
      case 5:
        row[vocab - 1] = nan;
        row[vocab / 2] = 10.0F;
        break;
      case 6:
        std::fill_n(row, vocab, nan);
        break;
      case 7:
        for (std::uint32_t i = 0; i < vocab; ++i)
          row[i] = i % 2 ? 0.0F : -0.0F;
        break;
      case 8:
        row[vocab / 2] = row[vocab - 1] = 10.0F;
        break;
      default:
        break;
    }
  }
  auto device_logits = Allocate<float>(logits.size());
  auto scratch = Allocate<q::ArgmaxCandidate>(rows * q::kArgmaxParts);
  auto output = Allocate<std::int32_t>(rows + 2);
  const auto greedy_scratch_size = 2 * rows * ((vocab - 1) / 4096 + 1);
  auto greedy_scratch = Allocate<float>(greedy_scratch_size);
  auto greedy_ids = Allocate<std::uint32_t>(rows);
  auto greedy_output = Allocate<q::ArgmaxCandidate>(rows + 2);
  std::vector<q::ArgmaxCandidate> predictions(rows + 2, {-1234567.0F, guard});
  std::vector<std::int32_t> actual(rows + 2, guard);
  CheckHip(hipMemcpy(output.get(), actual.data(),
                     actual.size() * sizeof(actual[0]), hipMemcpyHostToDevice),
           "initialize output");
  CheckHip(hipMemcpy(greedy_output.get(), predictions.data(),
                     predictions.size() * sizeof(predictions[0]),
                     hipMemcpyHostToDevice),
           "initialize greedy output");
  // Reuse the captured selection with changed logits, as the draft executor
  // does.
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  CheckHip(hipStreamCreate(&stream), "create stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "capture");
  q::Argmax(device_logits.get(), scratch.get(), output.get() + 1, rows, vocab,
            stream);
  gufo::hip::LaunchBatchedGPUArgmax(
      device_logits.get(), greedy_ids.get(), rows, vocab,
      {greedy_scratch.get(), greedy_scratch_size}, stream);
  q::GatherArgmaxCandidates(device_logits.get(), greedy_ids.get(),
                            greedy_output.get() + 1, rows, vocab, stream);
  CheckHip(hipStreamEndCapture(stream, &graph), "finish capture");
  CheckHip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
           "instantiate");
  for (int replay = 0; replay < 2; ++replay) {
    CheckHip(hipMemcpyAsync(device_logits.get(), logits.data(),
                            logits.size() * sizeof(float),
                            hipMemcpyHostToDevice, stream),
             "upload logits");
    CheckHip(hipGraphLaunch(executable, stream), "select");
    CheckHip(hipMemcpyAsync(actual.data(), output.get(),
                            actual.size() * sizeof(actual[0]),
                            hipMemcpyDeviceToHost, stream),
             "download tokens");
    CheckHip(hipMemcpyAsync(predictions.data(), greedy_output.get(),
                            predictions.size() * sizeof(predictions[0]),
                            hipMemcpyDeviceToHost, stream),
             "download greedy predictions");
    CheckHip(hipStreamSynchronize(stream), "synchronize");
    if (actual.front() != guard || actual.back() != guard) {
      throw std::runtime_error("argmax overwrote output guard");
    }
    if (predictions.front().index != guard ||
        predictions.back().index != guard ||
        predictions.front().value != -1234567.0F ||
        predictions.back().value != -1234567.0F)
      throw std::runtime_error("greedy argmax overwrote output guard");
    for (std::uint32_t t = 0; t < rows; ++t) {
      const float* row = logits.data() + static_cast<std::size_t>(t) * vocab;
      const auto expected =
          static_cast<std::int32_t>(std::max_element(row, row + vocab) - row);
      if (actual[t + 1] != expected) {
        throw std::runtime_error(
            "argmax mismatch: vocab=" + std::to_string(vocab) + " row=" +
            std::to_string(t) + " expected=" + std::to_string(expected) +
            " actual=" + std::to_string(actual[t + 1]));
      }
      const auto& prediction = predictions[t + 1];
      gufo::sampling::SamplerState sampler;
      std::uint32_t greedy = 0;
      bool finite = true;
      try {
        greedy = sampler.Sample({row, vocab});
      } catch (const std::runtime_error&) {
        finite = false;
      }
      if (!finite) {
        if (std::isfinite(prediction.value))
          throw std::runtime_error("greedy argmax accepted nonfinite row");
      } else if (prediction.index != static_cast<std::int32_t>(greedy) ||
                 std::bit_cast<std::uint32_t>(prediction.value) !=
                     std::bit_cast<std::uint32_t>(row[greedy])) {
        throw std::runtime_error(
            "greedy argmax disagrees with sampler: vocab=" +
            std::to_string(vocab) + " row=" + std::to_string(t));
      }
    }
    std::reverse(logits.begin(), logits.end());
  }
  CheckHip(hipGraphExecDestroy(executable), "destroy executable");
  CheckHip(hipGraphDestroy(graph), "destroy graph");
  CheckHip(hipStreamDestroy(stream), "destroy stream");
  std::cout << "MTP argmax vocab=" << vocab
            << ": CPU oracle and graph replay passed\n";
}

void CheckCandidates(std::uint32_t vocab) {
  constexpr auto kCandidates = gufo::models::qwen38_flash_next::kMtpCandidates;
  const auto workspace_size = q::MtpCandidateWorkspaceSize(vocab);
  auto device_ids = Allocate<std::uint32_t>(workspace_size + 2);
  auto scratch_ids = Allocate<std::uint32_t>(workspace_size + 2);
  // Match the executor's packed result: selection consumes the input IDs
  // before the final scores overwrite unused selection workspace.
  auto* device_scores =
      reinterpret_cast<float*>(device_ids.get() + 1 + kCandidates);
  auto device_logits = Allocate<float>(vocab);
  std::vector<float> logits(vocab);
  std::vector<std::uint32_t> expected(vocab);
  const auto count = std::min<std::size_t>(vocab, kCandidates);
  const auto result_count = std::min<std::size_t>(vocab, kCandidates);
  std::vector<float> scores(result_count);
  std::vector<std::uint32_t> ids(workspace_size + 2, UINT32_MAX);
  CheckHip(hipMemcpy(device_ids.get(), ids.data(), ids.size() * 4,
                     hipMemcpyHostToDevice),
           "candidate ID guards");
  CheckHip(hipMemcpy(scratch_ids.get(), ids.data(), ids.size() * 4,
                     hipMemcpyHostToDevice),
           "candidate scratch guards");
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  CheckHip(hipStreamCreate(&stream), "candidate stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
           "candidate capture");
  q::MtpTopCandidates(device_logits.get(), device_ids.get() + 1,
                      scratch_ids.get() + 1, device_scores, vocab, stream);
  CheckHip(hipStreamEndCapture(stream, &graph), "candidate capture end");
  CheckHip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
           "candidate graph");

  const auto score = [&](std::uint32_t id) {
    return std::isfinite(logits[id]) ? logits[id]
                                     : -std::numeric_limits<float>::infinity();
  };
  const auto better = [&](auto a, auto b) {
    return score(a) == score(b) ? a < b : score(a) > score(b);
  };
  for (unsigned mode = 0; mode < 4; ++mode) {
    for (std::uint32_t i = 0; i < vocab; ++i) {
      logits[i] = mode == 0   ? static_cast<float>((i * 7919U) % 257)
                  : mode == 1 ? (i % 2 ? 0.0F : -0.0F)
                  : mode == 2 ? std::numeric_limits<float>::quiet_NaN()
                              : (i >= vocab - count ? 10.0F : -100.0F);
    }
    if (mode == 0) {
      logits[0] = std::numeric_limits<float>::infinity();
      logits[vocab - 1] = std::numeric_limits<float>::quiet_NaN();
    }
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(expected.begin(), expected.end(), better);
    CheckHip(hipMemcpyAsync(device_logits.get(), logits.data(), vocab * 4,
                            hipMemcpyHostToDevice, stream),
             "candidate upload");
    CheckHip(hipGraphLaunch(executable, stream), "candidate replay");
    CheckHip(hipMemcpyAsync(ids.data(), device_ids.get(), ids.size() * 4,
                            hipMemcpyDeviceToHost, stream),
             "candidate IDs");
    CheckHip(hipStreamSynchronize(stream), "candidate synchronization");
    if (ids.front() != UINT32_MAX || ids.back() != UINT32_MAX)
      throw std::runtime_error("MTP candidate ID guard changed");
    for (std::size_t i = 0; i < count; ++i)
      if (ids[i + 1] != expected[i])
        throw std::runtime_error("MTP candidate IDs disagree with CPU");

    CheckHip(hipMemcpyAsync(scores.data(), device_scores, scores.size() * 4,
                            hipMemcpyDeviceToHost, stream),
             "candidate scores");
    CheckHip(hipStreamSynchronize(stream), "rescoring synchronization");
    if (ids.front() != UINT32_MAX || ids.back() != UINT32_MAX)
      throw std::runtime_error("MTP candidate output guard changed");
    for (std::size_t i = 0; i < result_count; ++i) {
      if (ids[i + 1] != expected[i] ||
          std::bit_cast<std::uint32_t>(scores[i]) !=
              std::bit_cast<std::uint32_t>(score(expected[i])))
        throw std::runtime_error(
            "MTP candidate scores disagree with CPU: mode=" +
            std::to_string(mode) + " rank=" + std::to_string(i) + " ID=" +
            std::to_string(ids[i + 1]) + "/" + std::to_string(expected[i]) +
            " score=" + std::to_string(scores[i]) + "/" +
            std::to_string(score(expected[i])));
    }
    CheckHip(hipMemcpy(ids.data(), scratch_ids.get(), ids.size() * 4,
                       hipMemcpyDeviceToHost),
             "candidate scratch download");
    if (ids.front() != UINT32_MAX || ids.back() != UINT32_MAX)
      throw std::runtime_error("MTP candidate scratch guard changed");
  }

  CheckHip(hipGraphExecDestroy(executable), "candidate graph destroy");
  CheckHip(hipGraphDestroy(graph), "candidate source graph destroy");
  CheckHip(hipStreamDestroy(stream), "candidate stream destroy");
  std::cout << "MTP candidates vocab=" << vocab
            << ": exact ordering, ties, guards and graph replay passed\n";
}

void CheckRemapDraftIds() {
  constexpr std::uint32_t prefix = 8, table_rows = 4;
  constexpr std::int32_t beyond = 0x7EADBEEF;
  // The table's rows (one empty slot), then memory the table does not own.
  std::vector<std::int32_t> table{100, -1, 102, 103};
  table.resize(table_rows + 60, beyond);
  const std::vector<std::uint32_t> input{
      UINT32_MAX - 1, 0,          prefix - 1,  prefix,    prefix + 1,
      prefix + 3,     prefix + 4, prefix + 63, INT32_MAX, UINT32_MAX - 1};
  const std::vector<std::uint32_t> expected{
      UINT32_MAX - 1, 0,          prefix - 1, 100,           UINT32_MAX, 103,
      UINT32_MAX,     UINT32_MAX, UINT32_MAX, UINT32_MAX - 1};
  auto device_table = Allocate<std::int32_t>(table.size());
  auto device_ids = Allocate<std::uint32_t>(input.size());
  CheckHip(hipMemcpy(device_table.get(), table.data(), table.size() * 4,
                     hipMemcpyHostToDevice),
           "draft table upload");
  CheckHip(hipMemcpy(device_ids.get(), input.data(), input.size() * 4,
                     hipMemcpyHostToDevice),
           "draft ID upload");
  // The first and last IDs are guards outside n.
  q::RemapDraftIds(device_ids.get() + 1,
                   static_cast<std::uint32_t>(input.size() - 2), prefix,
                   device_table.get(), table_rows, nullptr);
  std::vector<std::uint32_t> actual(input.size());
  CheckHip(hipMemcpy(actual.data(), device_ids.get(), actual.size() * 4,
                     hipMemcpyDeviceToHost),
           "draft ID download");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i] != expected[i])
      throw std::runtime_error(
          "draft ID remap: input=" + std::to_string(input[i]) + " expected=" +
          std::to_string(expected[i]) + " actual=" + std::to_string(actual[i]));
  std::cout << "Draft ID remap: table rows, empty slots and rows past the "
               "table passed\n";
}

// Each rank's half of a split vocabulary and the summed exchange rows must
// select what the full argmax selects, including ties across the halves and
// NaN or infinite logits.
void CheckSplitArgmax(std::uint32_t vocab) {
  constexpr std::uint32_t width = 2560;
  const std::uint32_t half = vocab / 2;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto device_logits = Allocate<float>(vocab);
  auto scratch = Allocate<q::ArgmaxCandidate>(q::kArgmaxParts);
  auto exchange = Allocate<float>(width);
  auto output = Allocate<std::int32_t>(1);
  for (std::uint32_t t = 0; t < 10; ++t) {
    std::vector<float> row(vocab);
    for (std::uint32_t i = 0; i < vocab; ++i)
      row[i] = -static_cast<float>((i * 7919U + t * 31U) % 65521U);
    switch (t) {
      case 0:
        row[half] = row[vocab - 1] = 10.0F;  // Tie inside the second half.
        break;
      case 1:
        row[half - 1] = row[half] = 10.0F;  // Tie across the halves.
        break;
      case 2:
        row[half] = inf;
        break;
      case 3:
        std::fill(row.begin(), row.end(), -inf);
        break;
      case 4:
        row[0] = nan;
        row[half] = inf;
        break;
      case 5:
        std::fill(row.begin(), row.begin() + half, nan);
        row[vocab - 1] = 10.0F;
        row[0] = 1.0F;
        break;
      case 6:
        std::fill(row.begin(), row.end(), nan);
        break;
      case 7:
        for (std::uint32_t i = 0; i < vocab; ++i)
          row[i] = i % 2 ? 0.0F : -0.0F;
        break;
      case 8:
        std::fill(row.begin() + half, row.end(), -inf);
        break;
      default:
        break;
    }
    CheckHip(hipMemcpy(device_logits.get(), row.data(), vocab * sizeof(float),
                       hipMemcpyHostToDevice),
             "upload split logits");
    std::vector<float> sum(width, 0.0F);
    for (std::uint32_t part = 0; part < 2; ++part) {
      std::vector<float> rank(width, 1234.0F);
      CheckHip(hipMemcpy(exchange.get(), rank.data(), width * sizeof(float),
                         hipMemcpyHostToDevice),
               "poison exchange");
      q::SplitArgmaxPart(device_logits.get() + part * half, scratch.get(),
                         exchange.get(), width, half, part * half, part,
                         nullptr);
      CheckHip(hipMemcpy(rank.data(), exchange.get(), width * sizeof(float),
                         hipMemcpyDeviceToHost),
               "download exchange");
      for (std::uint32_t i = 0; i < width; ++i) {
        if (i / 2 != part && rank[i] != 0.0F)
          throw std::runtime_error("split argmax left a foreign slot set");
        sum[i] += rank[i];
      }
    }
    CheckHip(hipMemcpy(exchange.get(), sum.data(), width * sizeof(float),
                       hipMemcpyHostToDevice),
             "upload exchange sum");
    q::SplitArgmaxFinish(exchange.get(), output.get(), nullptr);
    std::int32_t actual = -1;
    CheckHip(
        hipMemcpy(&actual, output.get(), sizeof(actual), hipMemcpyDeviceToHost),
        "download split argmax");
    const auto expected = static_cast<std::int32_t>(
        std::max_element(row.begin(), row.end()) - row.begin());
    if (actual != expected) {
      throw std::runtime_error(
          "split argmax mismatch: vocab=" + std::to_string(vocab) + " row=" +
          std::to_string(t) + " expected=" + std::to_string(expected) +
          " actual=" + std::to_string(actual));
    }
  }
  std::cout << "split argmax vocab=" << vocab
            << ": ties, infinities and NaN match the full argmax\n";
}

}  // namespace

// A split head must return the same winner and a whole-vocabulary probability,
// including cases in which either rank has no finite logits. Poison both the
// probability and surrounding words so token-only writes cannot pass.
void CheckSplitConfidence(std::uint32_t vocab) {
  constexpr std::uint32_t width = 2560;
  constexpr std::int32_t guard = 0x12345678;
  constexpr std::int32_t poison = 0x7FC12345;
  const std::uint32_t half = vocab / 2;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto logits = Allocate<float>(vocab);
  auto scratch = Allocate<q::ArgmaxCandidate>(q::kArgmaxParts);
  auto exchange = Allocate<float>(width);
  auto out = Allocate<std::int32_t>(4);
  auto full = Allocate<std::int32_t>(4);
  constexpr std::array gates{
      gufo::models::qwen38_flash_next::kMtpGateProbability,
      gufo::models::qwen38_flash_next::kMtpSampledGateProbability,
      gufo::models::qwen38_flash_next::kMtpBatchDropProbability};
  constexpr auto cases = 12U + 2U * static_cast<std::uint32_t>(gates.size());
  double worst = 0;
  for (std::uint32_t test = 0; test < cases; ++test) {
    std::vector<float> row(vocab);
    for (std::uint32_t i = 0; i < vocab; ++i)
      row[i] = static_cast<float>(
                   static_cast<int>((i * 7919U + test * 31U) % 257U) - 128) /
               8;
    switch (test) {
      case 0:
        row[half - 1] = row[half] = 40;
        break;
      case 1:
        std::fill(row.begin(), row.end(), 0);
        break;
      case 2:
        std::fill(row.begin(), row.end(), -inf);
        break;
      case 3:
        row[0] = nan;
        row[half] = 100;
        break;
      case 4:
        row[half] = inf;
        break;
      case 5:
        std::fill(row.begin(), row.end(), nan);
        break;
      case 6:
      case 7:
        std::fill(row.begin(), row.end(), -inf);
        row[test == 6 ? half - 1 : half] = 6;
        break;
      case 8:
        for (float& x : row)
          x += 10000;
        break;
      case 9:
        for (float& x : row)
          x -= 10000;
        break;
      default: {
        // Keep the original 0.9 cases and bracket every actual policy gate.
        // The offset exceeds the unchanged numerical error contract below.
        const double center = test < 12 ? .9 : gates[(test - 12) / 2];
        const double p = center + (test % 2 == 0 ? -1e-5 : 1e-5);
        std::fill(row.begin(), row.end(), -inf);
        row[0] = static_cast<float>(std::log((1 - p) / p));
        row[half] = 0;
        break;
      }
    }
    const auto winner = static_cast<std::int32_t>(
        std::max_element(row.begin(), row.end()) - row.begin());
    double want = 0;
    if (std::isfinite(row[winner])) {
      double sum = 0;
      for (float value : row)
        if (std::isfinite(value))
          sum += std::exp(static_cast<double>(value) - row[winner]);
      if (sum > 0)
        want = 1 / sum;
    }
    CheckHip(hipMemcpy(logits.get(), row.data(), vocab * sizeof(float),
                       hipMemcpyHostToDevice),
             "confidence logits");
    std::vector<float> sum(width, 0);
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
      std::vector<float> payload(width, 1234);
      CheckHip(hipMemcpy(exchange.get(), payload.data(), width * sizeof(float),
                         hipMemcpyHostToDevice),
               "poison confidence exchange");
      q::SplitArgmaxPart(logits.get() + rank * half, scratch.get(),
                         exchange.get(), width, half, rank * half, rank,
                         nullptr);
      q::SplitArgmaxConfidencePart(logits.get() + rank * half, exchange.get(),
                                   half, rank, nullptr);
      CheckHip(hipMemcpy(payload.data(), exchange.get(), width * sizeof(float),
                         hipMemcpyDeviceToHost),
               "confidence exchange download");
      for (std::uint32_t i = 0; i < width; ++i) {
        if (i / 2 != rank && i != 4 + rank && payload[i] != 0)
          throw std::runtime_error("confidence wrote a foreign exchange slot");
        sum[i] += payload[i];
      }
    }
    std::int32_t initial[4]{guard, -1, poison, guard};
    CheckHip(
        hipMemcpy(out.get(), initial, sizeof(initial), hipMemcpyHostToDevice),
        "poison split confidence output");
    CheckHip(
        hipMemcpy(full.get(), initial, sizeof(initial), hipMemcpyHostToDevice),
        "poison full confidence output");
    CheckHip(hipMemcpy(exchange.get(), sum.data(), width * sizeof(float),
                       hipMemcpyHostToDevice),
             "summed confidence exchange");
    q::SplitArgmaxConfidenceFinish(exchange.get(), out.get() + 1, nullptr);
    q::Argmax(logits.get(), scratch.get(), full.get() + 1, 1, vocab, nullptr);
    q::ArgmaxConfidence(logits.get(), full.get() + 1, vocab, nullptr);
    std::int32_t actual[4], whole[4];
    CheckHip(
        hipMemcpy(actual, out.get(), sizeof(actual), hipMemcpyDeviceToHost),
        "split confidence result");
    CheckHip(hipMemcpy(whole, full.get(), sizeof(whole), hipMemcpyDeviceToHost),
             "whole confidence result");
    for (const auto* result : {actual, whole}) {
      const float probability = std::bit_cast<float>(result[2]);
      const double error = std::abs(static_cast<double>(probability) - want);
      // Two ppm for ordinary probabilities; an absolute 2e-11 floor at
      // very small probabilities accommodates independent F32 reductions.
      const double tolerance = 2e-6 * std::max(want, 1e-5);
      if (result[0] != guard || result[3] != guard || result[1] != winner ||
          !std::isfinite(probability) || error > tolerance)
        throw std::runtime_error("split/full confidence oracle failed: vocab=" +
                                 std::to_string(vocab) +
                                 " case=" + std::to_string(test) +
                                 " expected=" + std::to_string(want) +
                                 " actual=" + std::to_string(probability));
      if (test >= 12) {
        const float gate = gates[(test - 12) / 2];
        if ((probability < gate) != (want < gate))
          throw std::runtime_error(
              "split/full confidence changed policy gate: vocab=" +
              std::to_string(vocab) + " case=" + std::to_string(test));
      }
      worst = std::max(worst, error);
    }
  }
  std::cout << "split/full confidence vocab=" << vocab << ": " << cases
            << " cases, worst absolute error " << worst << '\n';
}

// Independent global ordering plus the original full GPU selector. Inputs
// cross both partition boundaries; output is in-place and peer data immutable.
void CheckCandidateMerge(std::uint32_t vocab) {
  const auto started = std::chrono::steady_clock::now();
  constexpr unsigned keep = gufo::models::qwen38_flash_next::kMtpCandidates;
  constexpr std::uint32_t guard = 0xfeed1234U;
  const auto workspace = q::MtpCandidateWorkspaceSize(vocab);
  auto logits = Allocate<float>(vocab);
  auto full = Allocate<std::uint32_t>(workspace);
  auto scratch = Allocate<std::uint32_t>(workspace);
  auto left = Allocate<std::uint32_t>(workspace);
  auto right = Allocate<std::uint32_t>(workspace);
  auto output = Allocate<std::uint32_t>(2 * keep + 2);
  std::vector<float> input(vocab);
  std::vector<std::uint32_t> expected(vocab), reference(2 * keep),
      actual(2 * keep + 2);
  std::vector<std::uint32_t> peer_before(2 * keep), peer_after(2 * keep);
  const auto select = [&](const float* values, std::uint32_t* ids, unsigned n) {
    q::MtpTopCandidates(values, ids, scratch.get(),
                        reinterpret_cast<float*>(ids + keep), n, nullptr);
  };
  for (unsigned mode = 0; mode < 6; ++mode) {
    for (unsigned i = 0; i < vocab; ++i) {
      const auto bits = (i + 1) * 0x9e3779b9U;
      input[i] =
          mode == 0   ? float(bits % 17)
          : mode == 1 ? std::bit_cast<float>(i % 2 ? 0x80000000U : 0U)
          : mode == 2 ? std::bit_cast<float>(i % 3 == 0   ? 0x7fc12345U
                                             : i % 3 == 1 ? 0x7f800000U
                                                          : 0xff800000U)
          : mode == 3 ? (i >= vocab / 2 ? 1.0F : -1.0F)
          : mode == 4
              ? (i == 0 || i == vocab / 2 || i == vocab - 1 ? 1.0F : -INFINITY)
              : std::bit_cast<float>(bits);
    }
    const auto score = [&](unsigned i) {
      return std::isfinite(input[i]) ? input[i] : -INFINITY;
    };
    std::iota(expected.begin(), expected.end(), 0);
    std::partial_sort(expected.begin(), expected.begin() + keep, expected.end(),
                      [&](auto a, auto b) {
                        return score(a) == score(b) ? a < b
                                                    : score(a) > score(b);
                      });
    CheckHip(hipMemcpy(logits.get(), input.data(), vocab * sizeof(float),
                       hipMemcpyHostToDevice),
             "merge logits upload");
    select(logits.get(), full.get(), vocab);
    select(logits.get(), left.get(), vocab / 2);
    select(logits.get() + vocab / 2, right.get(), vocab / 2);
    CheckHip(hipMemcpy(reference.data(), full.get(), reference.size() * 4,
                       hipMemcpyDeviceToHost),
             "full merge reference");
    for (unsigned rank = 0; rank < 2; ++rank) {
      const auto* local = rank == 0 ? left.get() : right.get();
      const auto* peer = rank == 0 ? right.get() : left.get();
      CheckHip(hipMemcpy(peer_before.data(), peer, peer_before.size() * 4,
                         hipMemcpyDeviceToHost),
               "merge peer before");
      std::fill(actual.begin(), actual.end(), guard);
      CheckHip(hipMemcpy(output.get(), actual.data(), actual.size() * 4,
                         hipMemcpyHostToDevice),
               "merge output guards");
      CheckHip(hipMemcpy(output.get() + 1, local, 2 * keep * 4,
                         hipMemcpyDeviceToDevice),
               "merge local input");
      q::MergeMtpCandidates(output.get() + 1, peer, vocab / 2, rank, nullptr);
      CheckHip(hipMemcpy(actual.data(), output.get(), actual.size() * 4,
                         hipMemcpyDeviceToHost),
               "merged output");
      CheckHip(hipMemcpy(peer_after.data(), peer, peer_after.size() * 4,
                         hipMemcpyDeviceToHost),
               "merge peer after");
      if (actual.front() != guard || actual.back() != guard ||
          peer_before != peer_after)
        throw std::runtime_error("MTP merge guard/peer mutation");
      for (unsigned i = 0; i < keep; ++i) {
        const auto bits = std::bit_cast<std::uint32_t>(score(expected[i]));
        if (actual[i + 1] != expected[i] || actual[keep + i + 1] != bits ||
            actual[i + 1] != reference[i] ||
            actual[keep + i + 1] != reference[keep + i])
          throw std::runtime_error(
              "MTP merge oracle mismatch: vocab=" + std::to_string(vocab) +
              " mode=" + std::to_string(mode) +
              " rank=" + std::to_string(rank));
      }
    }
  }
  // Bad shape/rank/pointers fail before any kernel is submitted.
  unsigned rejected = 0;
  for (auto half : {0U, 63U, UINT32_MAX}) {
    try {
      q::MergeMtpCandidates(output.get(), left.get(), half, 0, nullptr);
    } catch (const std::invalid_argument&) {
      ++rejected;
    }
  }
  try {
    q::MergeMtpCandidates(output.get(), left.get(), vocab / 2, 2, nullptr);
  } catch (const std::invalid_argument&) {
    ++rejected;
  }
  try {
    q::MergeMtpCandidates(nullptr, left.get(), vocab / 2, 0, nullptr);
  } catch (const std::invalid_argument&) {
    ++rejected;
  }
  try {
    q::MergeMtpCandidates(output.get(), nullptr, vocab / 2, 0, nullptr);
  } catch (const std::invalid_argument&) {
    ++rejected;
  }
  if (rejected != 6)
    throw std::runtime_error("MTP merge argument guards");
  std::cout << "candidate merge vocab=" << vocab
            << ": six patterns/both ranks, "
            << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - started)
                   .count()
            << " ms\n";
}

// ggml quantize_row_q4_K_ref on dequantized Q8_0 values, in plain IEEE
// single precision (no contraction): the order QuantizeQ8_0ToQ4K keeps on
// the device.
int HostNearest(float value) {
  return static_cast<int>(std::nearbyint(value));
}

float HostQkx2(const float* x, const float* w, float* the_min) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  float min = x[0], max = x[0], sum_w = w[0], sum_x = sum_w * x[0];
  for (int i = 1; i < 32; ++i) {
    min = std::min(min, x[i]);
    max = std::max(max, x[i]);
    sum_w += w[i];
    sum_x += w[i] * x[i];
  }
  if (min > 0)
    min = 0;
  if (max == min) {
    *the_min = -min;
    return 0;
  }
  float iscale = 15.0F / (max - min);
  float scale = 1 / iscale;
  float best = 0;
  for (int i = 0; i < 32; ++i) {
    const int l = std::clamp(HostNearest(iscale * (x[i] - min)), 0, 15);
    const float diff = scale * static_cast<float>(l) + min - x[i];
    best += w[i] * (diff * diff);
  }
  for (int step = 0; step <= 20; ++step) {
    iscale = (-1.0F + 0.1F * static_cast<float>(step) + 15.0F) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(
          std::clamp(HostNearest(iscale * (x[i] - min)), 0, 15));
      sum_l += w[i] * l;
      sum_l2 += w[i] * l * l;
      sum_xl += w[i] * l * x[i];
    }
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det <= 0)
      continue;
    float this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
    float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
    if (this_min > 0) {
      this_min = 0;
      this_scale = sum_xl / sum_l2;
    }
    float error = 0;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(
          std::clamp(HostNearest(iscale * (x[i] - min)), 0, 15));
      const float diff = this_scale * l + this_min - x[i];
      error += w[i] * (diff * diff);
    }
    if (error < best) {
      best = error;
      scale = this_scale;
      min = this_min;
    }
  }
  *the_min = -min;
  return scale;
}

void HostQ4K(const float* x, std::uint8_t* out) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  float scales[8], mins[8], max_scale = 0, max_min = 0;
  for (int j = 0; j < 8; ++j) {
    const float* v = x + 32 * j;
    float sum_x2 = 0;
    for (int i = 0; i < 32; ++i)
      sum_x2 += v[i] * v[i];
    const float av_x = std::sqrt(sum_x2 / 32);
    float w[32];
    for (int i = 0; i < 32; ++i)
      w[i] = av_x + std::fabs(v[i]);
    scales[j] = HostQkx2(v, w, &mins[j]);
    max_scale = std::max(max_scale, scales[j]);
    max_min = std::max(max_min, mins[j]);
  }
  const float inv_scale = max_scale > 0 ? 63.0F / max_scale : 0.0F;
  const float inv_min = max_min > 0 ? 63.0F / max_min : 0.0F;
  std::uint8_t* packed = out + 4;
  std::fill_n(packed, 12, 0);
  std::uint8_t sc[8], m[8];
  for (int j = 0; j < 8; ++j) {
    sc[j] = std::min<std::uint8_t>(
        63, static_cast<std::uint8_t>(HostNearest(inv_scale * scales[j])));
    m[j] = std::min<std::uint8_t>(
        63, static_cast<std::uint8_t>(HostNearest(inv_min * mins[j])));
    if (j < 4) {
      packed[j] = sc[j];
      packed[j + 4] = m[j];
    } else {
      packed[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
      packed[j - 4] |= (sc[j] >> 4) << 6;
      packed[j] |= (m[j] >> 4) << 6;
    }
  }
  const auto d = static_cast<_Float16>(max_scale / 63.0F);
  const auto dmin = static_cast<_Float16>(max_min / 63.0F);
  std::memcpy(out, &d, 2);
  std::memcpy(out + 2, &dmin, 2);
  std::uint8_t codes[256]{};
  for (int j = 0; j < 8; ++j) {
    const float scale = static_cast<float>(d) * static_cast<float>(sc[j]);
    const float shift = static_cast<float>(dmin) * static_cast<float>(m[j]);
    for (int i = 0; scale != 0 && i < 32; ++i)
      codes[32 * j + i] = static_cast<std::uint8_t>(
          std::clamp(HostNearest((x[32 * j + i] + shift) / scale), 0, 15));
  }
  for (int byte = 0; byte < 128; ++byte)
    out[16 + byte] = codes[64 * (byte / 32) + byte % 32] |
                     (codes[64 * (byte / 32) + 32 + byte % 32] << 4);
}
// The draft head's Q4_K copy: device bytes equal the host reference, rows
// are close to their Q8_0 values, and a row's projection is the same at
// every batch width.
void CheckQ4KDraftHead() {
  constexpr std::uint32_t kRows = 96, kCols = 2560, kBlocks = kCols / 32;
  std::vector<std::uint8_t> q8(std::size_t{kRows} * kBlocks * 34);
  std::vector<float> values(std::size_t{kRows} * kCols);
  std::uint32_t state = 0x1234567U;
  const auto next = [&] {
    state = state * 1664525U + 1013904223U;
    return state >> 8;
  };
  for (std::size_t b = 0; b < std::size_t{kRows} * kBlocks; ++b) {
    // Mixed scales, so some sub-blocks round to a zero 6-bit scale; a few
    // constant, positive-only, negative-only and zero blocks.
    const float scale = (b % 7 == 3 ? 1.0e-5F : 1.0e-3F) *
                        static_cast<float>(1 + next() % 1000);
    const auto d = static_cast<_Float16>(scale);
    std::memcpy(&q8[b * 34], &d, 2);
    for (int i = 0; i < 32; ++i) {
      int code = static_cast<int>(next() % 255) - 127;
      if (b % 29 == 5)
        code = 17;
      else if (b % 31 == 7)
        code = std::abs(code);
      else if (b % 37 == 11)
        code = -std::abs(code);
      else if (b % 41 == 13)
        code = 0;
      q8[b * 34 + 2 + i] =
          static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
      values[b * 32 + i] = static_cast<float>(static_cast<std::int8_t>(code)) *
                           static_cast<float>(d);
    }
  }
  const std::size_t bytes = std::size_t{kRows} * (kCols / 256) * 144;
  auto device_q8 = Allocate<std::uint8_t>(q8.size());
  auto device_q4 = Allocate<std::uint8_t>(bytes);
  CheckHip(
      hipMemcpy(device_q8.get(), q8.data(), q8.size(), hipMemcpyHostToDevice),
      "Q8_0 rows");
  q::QuantizeQ8_0ToQ4K(device_q8.get(), device_q4.get(), kRows, kCols, nullptr);
  std::vector<std::uint8_t> actual(bytes);
  CheckHip(
      hipMemcpy(actual.data(), device_q4.get(), bytes, hipMemcpyDeviceToHost),
      "Q4_K rows");
  std::vector<std::uint8_t> expected(bytes);
  for (std::size_t sb = 0; sb < bytes / 144; ++sb)
    HostQ4K(values.data() + sb * 256, expected.data() + sb * 144);
  for (std::size_t sb = 0; sb < bytes / 144; ++sb) {
    if (std::memcmp(&actual[sb * 144], &expected[sb * 144], 144) == 0)
      continue;
    std::string detail;
    for (int i = 0; i < 144; ++i)
      if (actual[sb * 144 + i] != expected[sb * 144 + i])
        detail += " [" + std::to_string(i) + "] " +
                  std::to_string(actual[sb * 144 + i]) + "/" +
                  std::to_string(expected[sb * 144 + i]);
    throw std::runtime_error(
        "Q4_K superblock " + std::to_string(sb) +
        " differs from the reference (device/host):" + detail.substr(0, 400));
  }
  // Reconstruction error relative to the Q8_0 values.
  double error = 0, norm = 0;
  for (std::size_t sb = 0; sb < bytes / 144; ++sb) {
    const std::uint8_t* block = actual.data() + sb * 144;
    _Float16 d, dmin;
    std::memcpy(&d, block, 2);
    std::memcpy(&dmin, block + 2, 2);
    const std::uint8_t* packed = block + 4;
    for (int j = 0; j < 8; ++j) {
      const int sc = j < 4
                         ? packed[j] & 63
                         : (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4);
      const int m = j < 4 ? packed[j + 4] & 63
                          : (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
      for (int i = 0; i < 32; ++i) {
        const int byte = 32 * (j / 2) + i;
        const int code = (block[16 + byte] >> (4 * (j % 2))) & 0xF;
        const double value = double(d) * sc * code - double(dmin) * m;
        const double reference = values[sb * 256 + 32 * j + i];
        error += (value - reference) * (value - reference);
        norm += reference * reference;
      }
    }
  }
  if (!(std::sqrt(error / norm) < 0.1))
    throw std::runtime_error("Q4_K draft head reconstruction error");
  // One projection of eight rows against eight projections of one row.
  constexpr int kTokens = 8;
  std::vector<float> x(std::size_t{kTokens} * kCols);
  for (auto& v : x)
    v = static_cast<float>(static_cast<int>(next() % 2001) - 1000) / 997.0F;
  auto device_x = Allocate<float>(x.size());
  auto ids = Allocate<std::int32_t>(kTokens);
  auto batch = Allocate<float>(std::size_t{kTokens} * kRows);
  auto single = Allocate<float>(std::size_t{kTokens} * kRows);
  CheckHip(hipMemcpy(device_x.get(), x.data(), x.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "draft head input");
  CheckHip(hipMemset(ids.get(), 0, kTokens * sizeof(std::int32_t)),
           "draft head ids");
  if (qfn_mmq_moe_vec(12, device_q4.get(), device_x.get(), ids.get(),
                      batch.get(), kRows, kCols, kTokens, 1, 1, nullptr) != 0)
    throw std::runtime_error("batched Q4_K draft head projection");
  for (int t = 0; t < kTokens; ++t)
    if (qfn_mmq_moe_vec(12, device_q4.get(), device_x.get() + t * kCols,
                        ids.get(), single.get() + t * kRows, kRows, kCols, 1, 1,
                        1, nullptr) != 0)
      throw std::runtime_error("single-row Q4_K draft head projection");
  std::vector<float> batched(std::size_t{kTokens} * kRows);
  std::vector<float> rows(batched.size());
  CheckHip(hipMemcpy(batched.data(), batch.get(),
                     batched.size() * sizeof(float), hipMemcpyDeviceToHost),
           "batched draft logits");
  CheckHip(hipMemcpy(rows.data(), single.get(), rows.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "single-row draft logits");
  if (std::memcmp(batched.data(), rows.data(), rows.size() * sizeof(float)))
    throw std::runtime_error("Q4_K draft logits depend on the batch width");
  // Executor::Dense projects the draft head with the dense kernel at 1..8
  // rows; every width must round each row like the routed single-row
  // projection, or a seeded sampled draft depends on concurrent sessions.
  const std::size_t q8_row = qfn_mmq_q8_1_bytes(kTokens, kCols) / kTokens;
  auto device_q8_1 = Allocate<std::uint8_t>(q8_row * kTokens);
  auto dense = Allocate<float>(std::size_t{kTokens} * kRows);
  if (qfn_mmq_quantize_q8_1(device_x.get(), device_q8_1.get(), kTokens, kCols,
                            nullptr) != 0)
    throw std::runtime_error("draft head input quantization");
  std::vector<float> got(std::size_t{kTokens} * kRows);
  for (int width = 1; width <= kTokens; ++width)
    for (int start = 0; start + width <= kTokens; start += width) {
      if (qfn_mmq_q4_K_dense_vec_preq(
              device_q4.get(), device_q8_1.get() + start * q8_row, dense.get(),
              kRows, width, kCols, nullptr) != 0)
        throw std::runtime_error("dense Q4_K draft head projection");
      const std::size_t count = std::size_t(width) * kRows;
      CheckHip(hipMemcpy(got.data(), dense.get(), count * sizeof(float),
                         hipMemcpyDeviceToHost),
               "dense draft logits");
      if (std::memcmp(got.data(), rows.data() + start * kRows,
                      count * sizeof(float)) != 0)
        throw std::runtime_error("dense Q4_K draft logits of width " +
                                 std::to_string(width) +
                                 " differ from the single-row projection");
    }
  std::cout << "Q4_K draft head: reference bytes, reconstruction and "
               "batch-width invariance passed\n";
}

int main() {
  try {
    for (unsigned vocab : {128U, 130U, 99328U, 248320U})
      CheckCandidateMerge(vocab);

    CheckQ4KDraftHead();
    CheckRemapDraftIds();
    CheckPenaltyArgmax(257);
    CheckPenaltyArgmax(248320);
    CheckArgmax(1);
    CheckArgmax(257);
    CheckArgmax(248320);
    CheckSplitArgmax(2);
    CheckSplitArgmax(256);
    CheckSplitArgmax(248320);
    for (std::uint32_t vocab : {2U, 10U, 256U, 99328U, 248320U})
      CheckSplitConfidence(vocab);
    CheckHip(gufo::hip::PrepareBatchedGPUArgmax(), "prepare MTP argmax");
    CheckArgmax(248320);
    for (const unsigned vocab :
         {1, 63, 64, 65, 255, 256, 257, 1024, 1025, 16385, 248320}) {
      CheckCandidates(vocab);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
