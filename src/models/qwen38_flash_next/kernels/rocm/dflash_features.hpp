#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {

// Input is the raw post-layer residual [token][HC stream][hidden], before
// any following mixer normalization. Destination is [token][tap][hidden].
// Writes only the requested tap; all other feature slots remain untouched.
// The caller owns both buffers through completion. Submission is asynchronous
// and may be captured in a HIP graph with stable buffer addresses.
void CaptureDFlashFeatures(const float* residual, float* features,
                           std::uint32_t tokens, std::uint32_t hidden,
                           std::uint32_t streams, std::uint32_t taps,
                           std::uint32_t tap, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm
