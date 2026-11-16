#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
struct ExecutorTestAccess {
  static bool PackedDenseBatch(const DeviceTensor& w, const float* x, float* y,
                               std::uint32_t n, std::string* error) {
    // This packed-only branch needs the stream and qfn context, not a loaded
    // model. Exercise the actual batching adapter without a model download.
    Executor executor;
    return executor.DenseBatch(w, x, y, n, error);
  }
};
}  // namespace gufo::models::qwen38_flash_next::rocm

namespace {
using gufo::core::GgmlType;
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Buffer {
  void* data{};
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
std::uint32_t Next(std::uint32_t& seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}
std::vector<std::uint8_t> Weights(GgmlType type, int rows, int k, int experts) {
  const auto row_bytes = gufo::quant::QuantizedRowBytes(type, k);
  const auto block = gufo::quant::QuantizedBlockElements(type);
  const auto block_bytes = gufo::quant::QuantizedRowBytes(type, block);
  std::vector<std::uint8_t> packed(row_bytes * rows * experts);
  std::uint32_t seed = 0x73495U + static_cast<unsigned>(type);
  for (auto& v : packed)
    v = static_cast<std::uint8_t>(Next(seed));
  // Powers of two and exact integer activations isolate packing/dot logic
  // from activation quantization error. Q6_K stores its scale at the end.
  for (std::size_t i = 0; i < packed.size(); i += block_bytes) {
    const std::uint16_t d = 0x2000;  // 1/128
    const auto scale_offset = type == GgmlType::kQ6_K ? block_bytes - 2 : 0;
    std::memcpy(packed.data() + i + scale_offset, &d, 2);
    if (type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
        type == GgmlType::kQ5_1)
      std::memcpy(packed.data() + i + 2, &d, 2);
  }
  return packed;
}
void Run(GgmlType type, int rows, int k, int tokens, bool shared,
         bool tiled = false) {
  constexpr int experts = 16, used = 10, guard = 16;
  constexpr float poison = 123456.0F;
  const auto packed = Weights(type, rows, k, experts);
  const auto row_bytes = gufo::quant::QuantizedRowBytes(type, k);
  std::vector<float> decoded(std::size_t(experts) * rows * k);
  for (int i = 0; i < experts * rows; ++i)
    if (!gufo::quant::DequantizeRow(type, packed.data() + i * row_bytes,
                                    decoded.data() + std::size_t(i) * k, k))
      throw std::runtime_error("CPU oracle rejected weight row");
  std::vector<float> x(std::size_t(tokens) * k);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = i % 32 == 0 ? 127.0F : static_cast<float>(int(i * 13 % 15) - 7);
  std::vector<std::int32_t> ids(tokens * used);
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < used; ++j)
      ids[t * used + j] = j == 1 ? -1 : (j + (shared ? 0 : t)) % experts;
  for (int t = 0; t < tokens; ++t) {
    ids[t * used + 2] = ids[t * used];  // duplicate slots must both be written
    ids[t * used + used - 1] = experts - 1;
  }
  const std::size_t count = std::size_t(tokens) * used * rows;
  std::vector<float> output(count + 2 * guard, poison);
  Buffer w(packed.size() + (tiled ? 4096 : 0)), input(x.size() * sizeof(float)),
      indices(ids.size() * sizeof(std::int32_t));
  if (tiled)
    Check(hipMemset(w.data, 0, packed.size() + 4096));
  Buffer result(output.size() * sizeof(float)),
      scalar(output.size() * sizeof(float));
  Check(hipMemcpy(w.data, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Check(hipMemcpy(input.data, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  Check(hipMemcpy(indices.data, ids.data(), ids.size() * 4,
                  hipMemcpyHostToDevice));
  Check(hipMemcpy(result.data, output.data(), output.size() * 4,
                  hipMemcpyHostToDevice));
  Check(hipMemcpy(scalar.data, output.data(), output.size() * 4,
                  hipMemcpyHostToDevice));
  auto* dst = static_cast<float*>(result.data) + guard;
  const int rc =
      tiled ? qfn_mmq_moe_raw(static_cast<int>(type), w.data,
                              static_cast<float*>(input.data),
                              static_cast<std::int32_t*>(indices.data), dst,
                              rows, k, tokens, experts, used, nullptr)
            : qfn_mmq_moe_vec(static_cast<int>(type), w.data,
                              static_cast<float*>(input.data),
                              static_cast<std::int32_t*>(indices.data), dst,
                              rows, k, tokens, experts, used, nullptr);
  if (rc)
    throw std::runtime_error("batched vector projection failed");
  // Individual native launches are a grouping check, not the independent
  // numerical oracle below. Keep both so one cannot conceal the other's gap.
  for (int t = 0; t < tokens; ++t)
    if (qfn_mmq_moe_vec(static_cast<int>(type), w.data,
                        static_cast<float*>(input.data) + std::size_t(t) * k,
                        static_cast<std::int32_t*>(indices.data) + t * used,
                        static_cast<float*>(scalar.data) + guard +
                            std::size_t(t) * used * rows,
                        rows, k, 1, experts, used, nullptr))
      throw std::runtime_error("scalar vector projection failed");
  Check(hipStreamSynchronize(nullptr));
  Check(hipMemcpy(output.data(), result.data, output.size() * 4,
                  hipMemcpyDeviceToHost));
  std::vector<float> scalar_output(output.size());
  Check(hipMemcpy(scalar_output.data(), scalar.data, output.size() * 4,
                  hipMemcpyDeviceToHost));
  if (std::memcmp(output.data(), scalar_output.data(), output.size() * 4)) {
    for (std::size_t i = 0; i < output.size(); ++i) {
      if (std::bit_cast<std::uint32_t>(output[i]) !=
          std::bit_cast<std::uint32_t>(scalar_output[i]))
        throw std::runtime_error(
            std::string(gufo::core::ToString(type)) +
            " grouping mismatch index=" + std::to_string(i) +
            " batched=" + std::to_string(output[i]) +
            " scalar=" + std::to_string(scalar_output[i]));
    }
  }
  if (!std::all_of(output.begin(), output.begin() + guard,
                   [](float v) { return v == poison; }) ||
      !std::all_of(output.end() - guard, output.end(),
                   [](float v) { return v == poison; }))
    throw std::runtime_error("output guard overwritten");
  double max_abs = 0;
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < used; ++j)
      for (int r = 0; r < rows; ++r) {
        const int expert = ids[t * used + j];
        double reference = 0;
        if (expert >= 0)
          for (int c = 0; c < k; ++c)
            reference +=
                double(decoded[(std::size_t(expert) * rows + r) * k + c]) *
                x[std::size_t(t) * k + c];
        const float actual =
            output[guard + (std::size_t(t) * used + j) * rows + r];
        const double error = std::abs(double(actual) - reference);
        max_abs = std::max(max_abs, error);
        // These exact power-of-two fixtures have integral Q8 codes and
        // representable FP32 partial sums. No numerical tolerance is needed or
        // widened here.
        if (!std::isfinite(actual) || double(actual) != reference)
          throw std::runtime_error(std::string(gufo::core::ToString(type)) +
                                   " oracle mismatch row=" + std::to_string(r) +
                                   " expected=" + std::to_string(reference) +
                                   " actual=" + std::to_string(actual));
      }
  std::cout << (tiled ? "tiled:" : "") << gufo::core::ToString(type) << ','
            << rows << ',' << k << ',' << tokens << ',' << shared << ','
            << count << ',' << max_abs << '\n';
}

void Dense(GgmlType type, int k, int tokens, bool force_matrix = false) {
  constexpr int rows = 65, guard = 16;
  constexpr float poison = 123456.0F;
  auto packed = Weights(type, rows, k, 1);
  const auto row_bytes = gufo::quant::QuantizedRowBytes(type, k);
  std::vector<float> decoded(std::size_t(rows) * k);
  for (int r = 0; r < rows; ++r)
    if (!gufo::quant::DequantizeRow(type, packed.data() + r * row_bytes,
                                    decoded.data() + std::size_t(r) * k, k))
      throw std::runtime_error("dense CPU decode failed");
  std::vector<float> x(std::size_t(tokens) * k);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = i % 32 == 0 ? 127.0F : static_cast<float>(int(i * 13 % 15) - 7);
  std::vector<float> output(std::size_t(rows) * tokens + 2 * guard, poison);
  // Match production upload padding: tiled rows whose K is only aligned
  // to 32 may read past a packed row against zero activation padding.
  Buffer w(packed.size() + 4096), input(x.size() * 4),
      result(output.size() * 4);
  Check(hipMemset(w.data, 0, packed.size() + 4096));
  Check(hipMemcpy(w.data, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Check(hipMemcpy(input.data, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  Check(hipMemcpy(result.data, output.data(), output.size() * 4,
                  hipMemcpyHostToDevice));
  if (qfn_mmq_dense(static_cast<int>(type), w.data,
                    static_cast<float*>(input.data),
                    static_cast<float*>(result.data) + guard, rows, tokens, k,
                    nullptr, force_matrix))
    throw std::runtime_error("packed dense projection failed");
  Check(hipStreamSynchronize(nullptr));
  Check(hipMemcpy(output.data(), result.data, output.size() * 4,
                  hipMemcpyDeviceToHost));
  for (int i = 0; i < guard; ++i)
    if (output[i] != poison || output[output.size() - 1 - i] != poison)
      throw std::runtime_error("dense output guard changed");
  double max_abs = 0;
  for (int t = 0; t < tokens; ++t)
    for (int r = 0; r < rows; ++r) {
      double expected = 0;
      for (int c = 0; c < k; ++c)
        expected +=
            double(decoded[std::size_t(r) * k + c]) * x[std::size_t(t) * k + c];
      const float actual = output[guard + std::size_t(t) * rows + r];
      max_abs = std::max(max_abs, std::abs(double(actual) - expected));
      if (!std::isfinite(actual) || double(actual) != expected)
        throw std::runtime_error(
            std::string(gufo::core::ToString(type)) +
            " dense mismatch tokens=" + std::to_string(tokens) + " row=" +
            std::to_string(r) + " expected=" + std::to_string(expected) +
            " actual=" + std::to_string(actual));
    }
  std::cout << (force_matrix ? "dense-prefill:" : "dense:")
            << gufo::core::ToString(type) << ',' << rows << ',' << k << ','
            << tokens << ",0," << rows * tokens << ',' << max_abs << '\n';
}
// Nonintegral activations expose the vector/tiled scale-precision change
// that exact-integer packing fixtures deliberately exclude. A short prefill
// tail must reproduce the corresponding rows of a wide prefill call.
void DensePrefillTails(GgmlType type) {
  constexpr int rows = 65, k = 2560, wide = 33;
  auto packed = Weights(type, rows, k, 1);
  std::vector<float> x(wide * k);
  for (int t = 0; t < wide; ++t)
    for (int c = 0; c < k; ++c)
      x[t * k + c] = static_cast<float>((c * 13) % 257 - 128) / 13.0F;
  Buffer w(packed.size() + 4096), input(x.size() * sizeof(float)),
      output(wide * rows * sizeof(float));
  Check(hipMemset(w.data, 0, packed.size() + 4096));
  Check(hipMemcpy(w.data, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Check(hipMemcpy(input.data, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  const auto project = [&](int n, bool matrix) {
    if (qfn_mmq_dense(
            static_cast<int>(type), w.data, static_cast<float*>(input.data),
            static_cast<float*>(output.data), rows, n, k, nullptr, matrix))
      throw std::runtime_error("prefill-tail projection failed");
    Check(hipStreamSynchronize(nullptr));
    std::vector<float> result(n * rows);
    Check(hipMemcpy(result.data(), output.data, result.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    return result;
  };
  const auto reference = project(wide, true);
  for (int n : {1, 2, 8}) {
    const auto actual = project(n, true);
    if (std::memcmp(reference.data(), actual.data(),
                    actual.size() * sizeof(float)))
      throw std::runtime_error(std::string(gufo::core::ToString(type)) +
                               " short prefill changes matrix arithmetic");
    std::cout << "prefill-tail-parity:" << gufo::core::ToString(type) << ','
              << rows << ',' << k << ',' << n << ",0," << n * rows << ",0\n";
  }
  if (type == GgmlType::kQ6_K || type == GgmlType::kIQ4_NL ||
      type == GgmlType::kIQ4_XS) {
    const auto vector = project(1, false);
    if (!std::memcmp(reference.data(), vector.data(),
                     vector.size() * sizeof(float)))
      throw std::runtime_error(
          "tail regression fixture fails to expose the old precision switch");
  }
}

// Real mixed-format projections have unequal block scales. Fast-math may
// reassociate the MMQ scale/min terms differently for different tile widths.
// Repeat one nonintegral input so batch position cannot alter its answer.
void DensePrefillWidths(GgmlType type) {
  constexpr int rows = 65, k = 2560, wide = 1122;
  auto packed = Weights(type, rows, k, 1);
  const auto block = gufo::quant::QuantizedBlockElements(type);
  const auto bytes = gufo::quant::QuantizedRowBytes(type, block);
  for (std::size_t off = 0; off < packed.size(); off += bytes) {
    const std::uint16_t d = 0x1800 + (off / bytes * 37) % 1024;
    const auto scale = type == GgmlType::kQ6_K ? bytes - 2 : 0;
    std::memcpy(packed.data() + off + scale, &d, 2);
    if (type == GgmlType::kQ4_K || type == GgmlType::kQ5_K) {
      const std::uint16_t dm = 0x1c00 + (off / bytes * 53) % 1024;
      std::memcpy(packed.data() + off + 2, &dm, 2);
    }
  }
  std::vector<float> x(wide * k);
  for (int t = 0; t < wide; ++t)
    for (int c = 0; c < k; ++c)
      x[t * k + c] = static_cast<float>((c * 13) % 257 - 128) / 73.0F;
  Buffer w(packed.size() + 4096), input(x.size() * sizeof(float)),
      output(wide * rows * sizeof(float));
  Check(hipMemset(w.data, 0, packed.size() + 4096));
  Check(hipMemcpy(w.data, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Check(hipMemcpy(input.data, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  std::vector<float> reference;
  for (int n : {wide, 170, 1, 8}) {
    if (qfn_mmq_dense(
            static_cast<int>(type), w.data, static_cast<float*>(input.data),
            static_cast<float*>(output.data), rows, n, k, nullptr, true))
      throw std::runtime_error("prefill-width projection failed");
    Check(hipStreamSynchronize(nullptr));
    std::vector<float> actual(rows);
    Check(
        hipMemcpy(actual.data(),
                  static_cast<float*>(output.data) + std::size_t(n - 1) * rows,
                  rows * sizeof(float), hipMemcpyDeviceToHost));
    if (reference.empty())
      reference = actual;
    if (std::memcmp(reference.data(), actual.data(), rows * sizeof(float)))
      throw std::runtime_error(
          std::string(gufo::core::ToString(type)) +
          " variable-scale projection changed with batch width=" +
          std::to_string(n));
    std::cout << "prefill-width-parity:" << gufo::core::ToString(type) << ','
              << rows << ',' << k << ',' << n << ",0," << rows << ",0\n";
  }
}

void RoutedPrefillWidths(GgmlType type) {
  constexpr int rows = 65, k = 2560, wide = 1122, experts = 16, used = 10;
  auto packed = Weights(type, rows, k, experts);
  const auto block = gufo::quant::QuantizedBlockElements(type);
  const auto bytes = gufo::quant::QuantizedRowBytes(type, block);
  for (std::size_t off = 0; off < packed.size(); off += bytes) {
    const std::uint16_t d = 0x1800 + (off / bytes * 37) % 1024;
    const auto scale = type == GgmlType::kQ6_K ? bytes - 2 : 0;
    std::memcpy(packed.data() + off + scale, &d, 2);
  }
  std::vector<float> x(wide * k);
  std::vector<std::int32_t> ids(wide * used);
  for (int t = 0; t < wide; ++t) {
    for (int c = 0; c < k; ++c)
      x[t * k + c] = static_cast<float>((c * 13) % 257 - 128) / 73.0F;
    for (int j = 0; j < used; ++j)
      ids[t * used + j] = j == 1 ? -1 : j == 2 ? 0 : j;
  }
  Buffer w(packed.size() + 4096), input(x.size() * sizeof(float)),
      indices(ids.size() * sizeof(std::int32_t)),
      output(wide * used * rows * sizeof(float));
  Check(hipMemset(w.data, 0, packed.size() + 4096));
  Check(hipMemcpy(w.data, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Check(hipMemcpy(input.data, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  Check(hipMemcpy(indices.data, ids.data(), ids.size() * sizeof(std::int32_t),
                  hipMemcpyHostToDevice));
  struct ResetHints {
    ~ResetHints() {
      qfn_mmq_set_routed_tile_cols(0);
      qfn_mmq_set_routed_max_expert_rows(0);
    }
  } reset;
  std::vector<float> reference;
  for (int n : {wide, 170, 9, 1}) {
    qfn_mmq_set_routed_tile_cols(n == wide ? 64 : 16);
    qfn_mmq_set_routed_max_expert_rows(2 * n);  // expert0 occupies two slots
    if (qfn_mmq_moe_raw(static_cast<int>(type), w.data,
                        static_cast<float*>(input.data),
                        static_cast<std::int32_t*>(indices.data),
                        static_cast<float*>(output.data), rows, k, n, experts,
                        used, nullptr))
      throw std::runtime_error("routed prefill-width projection failed");
    Check(hipStreamSynchronize(nullptr));
    std::vector<float> actual(used * rows);
    Check(hipMemcpy(
        actual.data(),
        static_cast<float*>(output.data) + std::size_t(n - 1) * used * rows,
        actual.size() * sizeof(float), hipMemcpyDeviceToHost));
    if (reference.empty())
      reference = actual;
    if (std::memcmp(reference.data(), actual.data(),
                    actual.size() * sizeof(float)))
      throw std::runtime_error(
          std::string(gufo::core::ToString(type)) +
          " routed projection changed with batch/tile width=" +
          std::to_string(n));
    std::cout << "routed-width-parity:" << gufo::core::ToString(type) << ','
              << rows << ',' << k << ',' << n << ",0," << actual.size()
              << ",0\n";
  }
}

void PackedDecodeBatch(GgmlType type, int k, int tokens) {
  namespace gpu = gufo::models::qwen38_flash_next::rocm;
  constexpr int rows = 65, guard = 16;
  constexpr float poison = 123456.0F;
  auto packed = Weights(type, rows, k, 1);
  const auto block = gufo::quant::QuantizedBlockElements(type);
  const auto bytes = gufo::quant::QuantizedRowBytes(type, block);
  for (std::size_t off = 0; off < packed.size(); off += bytes) {
    const std::uint16_t scale = 0x1800 + (off / bytes * 37) % 1024;
    const auto where = type == GgmlType::kQ6_K ? bytes - 2 : 0;
    std::memcpy(packed.data() + off + where, &scale, 2);
    if (type == GgmlType::kQ4_K || type == GgmlType::kQ5_K) {
      const std::uint16_t minimum = 0x1c00 + (off / bytes * 53) % 1024;
      std::memcpy(packed.data() + off + 2, &minimum, 2);
    }
  }
  std::vector<float> x(std::size_t(tokens) * k);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<float>(int(i * 13 % 257) - 128) / 73.0F;
  std::vector<float> output(tokens * rows + 2 * guard, poison),
      reference(output);
  Buffer w(packed.size() + 4096), input(x.size() * sizeof(float)),
      result(output.size() * sizeof(float)),
      scalar(output.size() * sizeof(float));
  Check(hipMemset(w.data, 0, packed.size() + 4096));
  Check(hipMemcpy(w.data, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Check(hipMemcpy(input.data, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  Check(hipMemcpy(result.data, output.data(), output.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  Check(hipMemcpy(scalar.data, reference.data(),
                  reference.size() * sizeof(float), hipMemcpyHostToDevice));
  const gpu::DeviceTensor tensor{.data = w.data,
                                 .type = type,
                                 .cols = static_cast<std::uint32_t>(k),
                                 .rows = rows};
  std::string error;
  if (!gpu::ExecutorTestAccess::PackedDenseBatch(
          tensor, static_cast<float*>(input.data),
          static_cast<float*>(result.data) + guard, tokens, &error))
    throw std::runtime_error(error);
  for (int t = 0; t < tokens; ++t)
    if (qfn_mmq_dense(static_cast<int>(type), w.data,
                      static_cast<float*>(input.data) + std::size_t(t) * k,
                      static_cast<float*>(scalar.data) + guard + t * rows, rows,
                      1, k, nullptr, false))
      throw std::runtime_error("packed scalar decode failed");
  Check(hipStreamSynchronize(nullptr));
  Check(hipMemcpy(output.data(), result.data, output.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  Check(hipMemcpy(reference.data(), scalar.data,
                  reference.size() * sizeof(float), hipMemcpyDeviceToHost));
  if (std::memcmp(output.data(), reference.data(),
                  output.size() * sizeof(float)))
    throw std::runtime_error(
        std::string(gufo::core::ToString(type)) +
        " wide decode changed vector arithmetic or guards");
  for (int i = 0; i < guard; ++i)
    if (output[i] != poison || output[output.size() - 1 - i] != poison)
      throw std::runtime_error("batched packed decode guard changed");
  std::cout << "decode-batch:" << gufo::core::ToString(type) << ',' << rows
            << ',' << k << ',' << tokens << ",0," << tokens * rows << ",0\n";
}

void Embedding(int streams) {
  namespace rocm = gufo::models::qwen38_flash_next::rocm;
  constexpr int rows = 17, k = 2560, guard = 16;
  constexpr float poison = 123456.0F;
  const auto type = GgmlType::kIQ4_XS;
  const auto packed = Weights(type, rows, k, 1);
  std::vector<float> decoded(std::size_t(rows) * k);
  const auto row_bytes = gufo::quant::QuantizedRowBytes(type, k);
  for (int r = 0; r < rows; ++r)
    if (!gufo::quant::DequantizeRow(type, packed.data() + r * row_bytes,
                                    decoded.data() + std::size_t(r) * k, k))
      throw std::runtime_error("embedding CPU decode failed");
  const std::array<std::int32_t, 4> ids{0, 16, 3, 16};
  const auto count = ids.size() * streams * k;
  std::vector<float> output(count + 2 * guard, poison);
  Buffer w(packed.size()), indices(sizeof(ids)), result(output.size() * 4);
  Check(hipMemcpy(w.data, packed.data(), packed.size(), hipMemcpyHostToDevice));
  Check(
      hipMemcpy(indices.data, ids.data(), sizeof(ids), hipMemcpyHostToDevice));
  Check(hipMemcpy(result.data, output.data(), output.size() * 4,
                  hipMemcpyHostToDevice));
  rocm::EmbedTokens(w.data, rocm::WeightType::kIQ4_XS,
                    static_cast<std::int32_t*>(indices.data),
                    static_cast<float*>(result.data) + guard, ids.size(), k,
                    streams, nullptr);
  Check(hipStreamSynchronize(nullptr));
  Check(hipMemcpy(output.data(), result.data, output.size() * 4,
                  hipMemcpyDeviceToHost));
  for (int i = 0; i < guard; ++i)
    if (output[i] != poison || output[output.size() - 1 - i] != poison)
      throw std::runtime_error("embedding output guard changed");
  for (std::size_t t = 0; t < ids.size(); ++t)
    for (int st = 0; st < streams; ++st)
      for (int c = 0; c < k; ++c)
        if (output[guard + (t * streams + st) * k + c] !=
            decoded[std::size_t(ids[t]) * k + c])
          throw std::runtime_error("IQ4_XS embedding mismatch");
  std::cout << "embed:IQ4_XS," << rows << ',' << k << ',' << ids.size() << ','
            << streams << ',' << count << ",0\n";
}

}  // namespace
int main(int argc, char** argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    Check(hipSetDevice(0));
    if (qfn_mmq_init(0))
      throw std::runtime_error("MMQ initialization failed");
    std::cout << "format,rows,k,tokens,shared,values,max_abs_error\n";
    const bool dense_only =
        argc == 2 && std::string(argv[1]) == "--dense-embedding-only";
    if (!dense_only) {
      for (auto type : {GgmlType::kQ2_0, GgmlType::kQ6_K, GgmlType::kIQ2_S,
                        GgmlType::kIQ3_XXS, GgmlType::kIQ3_S, GgmlType::kIQ4_NL,
                        GgmlType::kIQ4_XS})
        for (int width : {1, 2, 7, 9, 33})
          for (bool shared : {false, true}) {
            Run(type, 65, 2560, width, shared);
            if (width == 2)
              Run(type, 320, 2560, width, shared);
          }
      for (auto type : {GgmlType::kQ2_0, GgmlType::kIQ4_NL})
        for (int k : {320, 640})
          for (int width : {1, 7, 33})
            Run(type, 65, k, width, false);
    }
    for (auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                      GgmlType::kIQ4_NL, GgmlType::kIQ4_XS})
      for (int n : {1, 2, 8, 9, 17, 33})
        Dense(type, 2560, n);
    for (int n : {1, 9, 33})
      Dense(GgmlType::kIQ4_NL, 320, n);
    // Prefill tails below the vector threshold keep matrix arithmetic.
    for (auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                      GgmlType::kIQ4_NL, GgmlType::kIQ4_XS})
      for (int n : {1, 2, 8})
        Dense(type, 2560, n, true);
    for (auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                      GgmlType::kIQ4_NL, GgmlType::kIQ4_XS})
      DensePrefillTails(type);
    for (auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                      GgmlType::kIQ4_NL, GgmlType::kIQ4_XS})
      DensePrefillWidths(type);
    for (auto type : {GgmlType::kQ2_0, GgmlType::kQ6_K, GgmlType::kIQ2_S,
                      GgmlType::kIQ3_XXS, GgmlType::kIQ3_S, GgmlType::kIQ4_NL,
                      GgmlType::kIQ4_XS})
      RoutedPrefillWidths(type);
    for (auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                      GgmlType::kIQ4_NL, GgmlType::kIQ4_XS})
      for (int n : {17, 64})
        PackedDecodeBatch(type, 2560, n);
    for (auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                      GgmlType::kIQ4_NL, GgmlType::kIQ4_XS})
      for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 17, 64})
        PackedDecodeBatch(type, 2560, n);
    for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 17, 64}) {
      PackedDecodeBatch(GgmlType::kQ6_K, 6144, n);
      PackedDecodeBatch(GgmlType::kIQ4_NL, 640, n);
    }
    PackedDecodeBatch(GgmlType::kIQ4_NL, 320, 63);
    for (int streams : {1, 4})
      Embedding(streams);
    for (auto type : {GgmlType::kQ2_0, GgmlType::kQ6_K, GgmlType::kIQ2_S,
                      GgmlType::kIQ3_XXS, GgmlType::kIQ3_S, GgmlType::kIQ4_NL,
                      GgmlType::kIQ4_XS}) {
      Run(type, 65, 2560, 33, false, true);
      Run(type, 320, 2560, 17, true, true);
    }
    for (auto type : {GgmlType::kQ2_0, GgmlType::kIQ4_NL})
      for (int k : {320, 640})
        Run(type, 65, k, 33, false, true);
    for (auto type :
         {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ5_1, GgmlType::kQ8_0})
      Run(type, 65, 2560, 33, false, true);

    std::cout << "PASS packed GSQ vector/dense/embedding; "
                 "numerical/grouping/guards\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
