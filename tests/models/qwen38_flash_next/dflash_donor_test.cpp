#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "src/families/qwen/dflash/donor.hpp"

namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    using gufo::core::GgmlType;
    using gufo::models::QwenTensorRef;
    using gufo::speculative::DFlashDonorError;
    constexpr std::size_t hidden = 256, vocab = 2;
    std::array<float, hidden * vocab> storage{};
    for (const auto type :
         {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0, GgmlType::kQ3_K,
          GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kIQ4_NL,
          GgmlType::kIQ4_XS, GgmlType::kIQ3_S}) {
      const QwenTensorRef table{storage.data(), type, hidden * vocab,
                                sizeof(storage)};
      Require(DFlashDonorError(table, hidden, vocab, "output").empty(),
              "supported donor was rejected");
      auto bad = table;
      bad.available_bytes = table.EncodedSizeBytes() - 1;
      Require(!DFlashDonorError(bad, hidden, vocab, "output").empty(),
              "truncated donor storage was accepted");
      bad = table;
      --bad.num_elements;
      Require(!DFlashDonorError(bad, hidden, vocab, "output").empty(),
              "wrong donor element count was accepted");
    }
    for (const auto type :
         {GgmlType::kF16, GgmlType::kQ8_K, static_cast<GgmlType>(999)}) {
      const QwenTensorRef table{storage.data(), type, hidden * vocab};
      for (const char* role : {"token embedding", "output"}) {
        const auto error = DFlashDonorError(table, hidden, vocab, role);
        Require(error.find(role) != std::string::npos &&
                    error.find("unsupported") != std::string::npos,
                "unsupported donor did not fail with its role");
      }
    }
    const QwenTensorRef good{storage.data(), GgmlType::kF32, hidden * vocab};
    Require(!DFlashDonorError(good, 0, vocab, "output").empty() &&
                !DFlashDonorError(good, hidden, 0, "output").empty() &&
                !DFlashDonorError(good, std::numeric_limits<std::size_t>::max(),
                                  vocab, "output")
                     .empty() &&
                !DFlashDonorError({}, hidden, vocab, "output").empty(),
            "invalid donor geometry was accepted");
    const QwenTensorRef packed{storage.data(), GgmlType::kQ4_K, hidden * vocab};
    Require(!DFlashDonorError(packed, hidden / 2, vocab * 2, "output").empty(),
            "misaligned quantized donor rows were accepted");
    std::puts("PASS: DFlash donor formats, exact storage, shape and overflow");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
