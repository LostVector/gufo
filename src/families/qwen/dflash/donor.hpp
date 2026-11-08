#pragma once

#include <limits>
#include <string>
#include <string_view>

#include "src/families/qwen/hip_common/tensor.hpp"

namespace gufo::speculative {

/// Borrowed target tables bypass the draft's packing/conversion path. Admit
/// only formats understood by both the embedding and block-GEMM consumers.
/// In particular, F16 is neither F32 nor a packed quantized format.
inline std::string DFlashDonorError(const models::QwenTensorRef& table,
                                    std::size_t hidden, std::size_t vocab,
                                    std::string_view role) {
  const auto error = [&](std::string_view detail) {
    return "DFlash target " + std::string(role) + " " + std::string(detail);
  };
  switch (table.type) {
    case core::GgmlType::kF32:
    case core::GgmlType::kBF16:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kIQ3_S:
      break;
    default:
      return error("format " + std::string(core::ToString(table.type)) +
                   " is unsupported by DFlash donor kernels");
  }
  if (!hidden || !vocab ||
      hidden > std::numeric_limits<std::size_t>::max() / vocab ||
      table.empty() || table.num_elements != hidden * vocab ||
      quant::EncodedSizeBytes(table.type, hidden) == 0 ||
      !table.FitsAvailableStorage()) {
    return error("has invalid shape or insufficient storage");
  }
  return {};
}

}  // namespace gufo::speculative
