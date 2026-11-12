#include "src/models/qwen38_flash_next/kernels/rocm/dflash_features.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
__global__ void CaptureMean(const float* residual, float* features,
                            std::size_t elements, std::uint32_t hidden,
                            std::uint32_t streams, std::uint32_t taps,
                            std::uint32_t tap) {
  const std::size_t index = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const auto row = index / hidden;
  const auto channel = index % hidden;
  float sum = 0.0F;
  for (std::uint32_t part = 0; part < streams; ++part) {
    sum = __fadd_rn(sum, residual[(row * streams + part) * hidden + channel]);
  }
  features[(row * taps + tap) * hidden + channel] =
      __fmul_rn(sum, __fdiv_rn(1.0F, static_cast<float>(streams)));
}
}  // namespace

void CaptureDFlashFeatures(const float* residual, float* features,
                           std::uint32_t tokens, std::uint32_t hidden,
                           std::uint32_t streams, std::uint32_t taps,
                           std::uint32_t tap, hipStream_t stream) {
  if (!residual || !features || !tokens || !hidden || !streams || !taps ||
      tap >= taps) {
    throw std::invalid_argument("invalid DFlash feature capture layout");
  }
  const std::size_t elements = std::size_t{tokens} * hidden;
  const auto max = std::numeric_limits<std::size_t>::max() / sizeof(float);
  if (elements > max / streams || elements > max / taps ||
      (elements + 255) / 256 > std::numeric_limits<unsigned>::max()) {
    throw std::length_error("DFlash feature capture layout overflows");
  }
  CaptureMean<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                stream>>>(residual, features, elements, hidden, streams, taps,
                          tap);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
