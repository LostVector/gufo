#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>

#include "src/models/qwen38_flash_next/dflash_sampling.hpp"

namespace {
namespace qfn = gufo::models::qwen38_flash_next;
using gufo::speculative::DraftProposal;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
DraftProposal Neural() {
  return {.tokens = {2, 3, 4},
          .candidate_ids = {1, 2, 3, 4, 0, 3, 5, 7, 0, 2, 4, 8},
          .candidate_probabilities = {.1F, .4F, .3F, .2F, .2F, .3F, .1F, .4F,
                                      .1F, .1F, .7F, .1F},
          .candidates_per_token = 4,
          .start_pos = 43};
}
auto Fields(const DraftProposal& p) {
  return std::tie(p.tokens, p.candidate_ids, p.candidate_probabilities,
                  p.candidates_per_token, p.start_pos, p.from_lookup);
}
}  // namespace

int main() {
  try {
    const std::array<std::int32_t, 4> mixed{9, 2, 3, 6};
    auto proposal = Neural();
    const auto original = proposal;
    qfn::ApplyDFlashLookup(proposal, mixed, 2, true);
    auto rows = qfn::DFlashProbabilityRows(proposal, 16);
    Require(
        proposal.tokens == std::vector<gufo::sampling::TokenId>({2, 3, 6}) &&
            !proposal.from_lookup && proposal.start_pos == 43,
        "mixed lookup changed proposal identity");
    Require(std::equal(rows[0].ids.begin(), rows[0].ids.end(),
                       original.candidate_ids.begin()) &&
                std::equal(rows[0].probabilities.begin(),
                           rows[0].probabilities.end(),
                           original.candidate_probabilities.begin()),
            "lookup changed the retained neural q row");
    Require(
        rows[1].selected_probability == 1 && rows[2].selected_probability == 1,
        "existing and new copy candidates are not point masses");
    // Check the complete rejection identity independently for a toy target.
    // In particular, a rejected copy must not survive in the residual mass.
    for (std::size_t row = 1; row < rows.size(); ++row) {
      std::array<double, 16> q{};
      for (std::size_t j = 0; j < rows[row].ids.size(); ++j)
        q[rows[row].ids[j]] = rows[row].probabilities[j];
      const double accepted = double(proposal.tokens[row] + 1) / 136;
      for (unsigned token = 0; token < 16; ++token) {
        const double p = double(token + 1) / 136;
        const double residual = std::max(0.0, p - q[token]) / (1 - accepted);
        const double output = std::min(p, q[token]) + (1 - accepted) * residual;
        Require(std::abs(output - p) < 1e-15,
                "point-mass rejection does not preserve target probability");
        Require(q[token] == 0 || residual == 0,
                "rejected copy remains in residual support");
      }
    }
    auto extended = Neural();
    const std::array<std::int32_t, 6> longer{9, 2, 3, 6, 7, 8};
    qfn::ApplyDFlashLookup(extended, longer, 4, true);
    const auto extended_rows = qfn::DFlashProbabilityRows(extended, 16);
    Require(extended_rows.size() == 5 &&
                extended_rows[3].selected_probability == 1 &&
                extended_rows[4].selected_probability == 1,
            "copy budget cannot extend past the neural block");
    DraftProposal copied;
    const std::array<std::int32_t, 4> chain{9, 7, 8, 7};
    qfn::ApplyDFlashLookup(copied, chain, 3, true);
    rows = qfn::DFlashProbabilityRows(copied, 16);
    Require(copied.from_lookup && copied.candidates_per_token == 1 &&
                rows.size() == 3,
            "copy-only proposal does not use compact point-mass rows");
    for (const auto& row : rows)
      Require(row.selected_probability == 1, "copy-only q is not one");
    auto greedy = Neural();
    qfn::ApplyDFlashLookup(greedy, mixed, 2, false);
    Require(greedy.tokens == proposal.tokens && greedy.candidate_ids.empty() &&
                greedy.candidate_probabilities.empty() &&
                greedy.candidates_per_token == 0,
            "greedy copy retained stale sampling rows");
    for (unsigned failure = 0; failure < 6; ++failure) {
      auto invalid = Neural();
      auto bad_chain = mixed;
      std::size_t count = 2;
      if (failure == 0)
        count = 0;
      if (failure == 1)
        count = bad_chain.size();
      if (failure == 2)
        bad_chain.back() = -1;
      if (failure == 3)
        bad_chain[1] = 8;
      if (failure == 4)
        invalid.candidate_probabilities.pop_back();
      if (failure == 5)
        invalid.candidates_per_token = 0;
      const auto before = invalid;
      bool rejected = false;
      try {
        qfn::ApplyDFlashLookup(invalid, bad_chain, count, true);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      Require(rejected && Fields(invalid) == Fields(before),
              "malformed lookup mutated or escaped validation");
    }
    std::cout << "DFlash pure/mixed lookup q and rejection identity passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
