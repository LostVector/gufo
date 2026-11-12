#pragma once

#include <algorithm>
#include <cmath>
#include <gufo/ext/draft_backend.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace gufo::models::qwen38_flash_next {

struct DFlashProbabilityRow {
  std::span<const sampling::TokenId> ids;
  std::span<const float> probabilities;
  double selected_probability;
};

// Lookup replaces only a trailing suffix. Retained neural rows therefore keep
// the exact predecessor history under which their q probabilities were drawn.
inline void ApplyDFlashLookup(speculative::DraftProposal& proposal,
                              std::span<const std::int32_t> chain,
                              std::size_t copied, bool sampled) {
  if (copied == 0 || chain.size() <= copied ||
      std::ranges::any_of(chain, [](auto token) { return token < 0; }))
    throw std::invalid_argument("invalid DFlash lookup chain");
  const auto neural = chain.size() - 1 - copied;
  if (neural > proposal.tokens.size() ||
      !std::equal(chain.begin() + 1, chain.begin() + 1 + neural,
                  proposal.tokens.begin()))
    throw std::invalid_argument("DFlash lookup changed a neural prefix");
  const auto count = chain.size() - 1;
  const auto width = proposal.candidates_per_token;
  if (neural && sampled &&
      (width == 0 ||
       proposal.tokens.size() > proposal.candidate_ids.size() / width ||
       proposal.tokens.size() * width != proposal.candidate_ids.size() ||
       proposal.candidate_ids.size() !=
           proposal.candidate_probabilities.size() ||
       count > proposal.candidate_ids.max_size() / width ||
       count > proposal.candidate_probabilities.max_size() / width))
    throw std::invalid_argument("DFlash lookup candidate rows are incomplete");
  const auto old_count = proposal.tokens.size();
  proposal.tokens.assign(chain.begin() + 1, chain.end());
  proposal.from_lookup = neural == 0;
  if (!sampled) {
    proposal.candidates_per_token = 0;
    proposal.candidate_ids.clear();
    proposal.candidate_probabilities.clear();
  } else if (neural == 0) {
    proposal.candidates_per_token = 1;
    proposal.candidate_ids = proposal.tokens;
    proposal.candidate_probabilities.assign(count, 1.0F);
  } else {
    proposal.candidate_ids.resize(count * width);
    proposal.candidate_probabilities.resize(count * width);
    // Copies can fill the target budget beyond the neural block. Reuse a
    // valid ID set to pad those rows; only the copied token has nonzero q.
    for (auto row = old_count; row < count; ++row)
      std::copy_n(proposal.candidate_ids.begin(), width,
                  proposal.candidate_ids.begin() + row * width);
    for (auto row = neural; row < count; ++row) {
      auto ids = std::span(proposal.candidate_ids).subspan(row * width, width);
      auto probabilities = std::span(proposal.candidate_probabilities)
                               .subspan(row * width, width);
      auto found = std::find(ids.begin(), ids.end(), proposal.tokens[row]);
      if (found == ids.end()) {
        found = ids.begin();
        *found = proposal.tokens[row];
      }
      std::fill(probabilities.begin(), probabilities.end(), 0.0F);
      probabilities[found - ids.begin()] = 1.0F;
    }
  }
}

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
