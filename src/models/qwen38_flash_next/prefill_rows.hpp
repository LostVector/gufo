#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_ROWS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_ROWS_HPP_

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gufo::models::qwen38_flash_next {
struct PrefillRows {
  std::uint32_t chunk;
  std::uint32_t through;
};

// Keep the calibrated local chunk when it fits. A TP exchange must fit the
// whole trunk batch, including its optional short tail, in one transport
// window. Smaller windows use whole power-of-two chunks with room for that
// tail.
constexpr PrefillRows PlanPrefillRows(
    std::uint32_t context, std::uint32_t hidden,
    std::size_t exchange_bytes = std::numeric_limits<std::size_t>::max()) {
  constexpr std::uint32_t chunk = 4096, tail = 128;
  if (hidden == 0)
    return {};
  const auto limit = static_cast<std::uint32_t>(std::min<std::size_t>(
      context, exchange_bytes / (std::size_t{hidden} * sizeof(float))));
  const auto desired = std::min(context, chunk + tail);
  if (limit >= desired)
    return {std::min(context, chunk), desired};
  const auto rows = limit > tail ? std::bit_floor(limit - tail) : limit;
  return {rows, std::min(limit, rows + tail)};
}
}  // namespace gufo::models::qwen38_flash_next
#endif
