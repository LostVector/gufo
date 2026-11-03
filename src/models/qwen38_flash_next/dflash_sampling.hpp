#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace gufo::models::qwen38_flash_next {

struct DFlashProbabilityRow {
  std::span<const sampling::TokenId> ids;
  std::span<const float> probabilities;
  double selected_probability;
};

// Match the existing generic DFlash verifier's F32-selector contract. Unlike
// native MTP's integer-mass proposal, these probabilities are rounded F32.
// Validate every row before executing or mutating the target, including the
// rows after a potential early rejection. Do not renormalize the exported q.
inline std::vector<DFlashProbabilityRow> DFlashProbabilityRows(
    const speculative::DraftProposal& proposal, std::size_t vocab) {
  const auto count = proposal.tokens.size();
  const auto width = proposal.candidates_per_token;
  if (count == 0 || width == 0 ||
      width > proposal.candidate_ids.size() / count ||
      proposal.candidate_ids.size() != count * width ||
      proposal.candidate_probabilities.size() != proposal.candidate_ids.size())
    throw std::invalid_argument("malformed DFlash sampled proposal");
  std::vector<DFlashProbabilityRow> rows;
  rows.reserve(count);
  for (std::size_t row = 0; row < count; ++row) {
    auto ids = std::span(proposal.candidate_ids).subspan(row * width, width);
    auto probabilities =
        std::span(proposal.candidate_probabilities).subspan(row * width, width);
    double sum = 0, selected = 0;
    for (std::size_t i = 0; i < width; ++i) {
      if (ids[i] >= vocab || !std::isfinite(probabilities[i]) ||
          probabilities[i] < 0 || probabilities[i] > 1 ||
          std::find(ids.begin(), ids.begin() + i, ids[i]) != ids.begin() + i)
        throw std::invalid_argument("invalid DFlash sampled candidate");
      sum += probabilities[i];
      if (ids[i] == proposal.tokens[row])
        selected += probabilities[i];
    }
    if (std::abs(sum - 1.0) > 1e-5 || !(selected > 0))
      throw std::invalid_argument("unnormalized DFlash sampled proposal");
    rows.push_back({ids, probabilities, selected});
  }
  return rows;
}

}  // namespace gufo::models::qwen38_flash_next
