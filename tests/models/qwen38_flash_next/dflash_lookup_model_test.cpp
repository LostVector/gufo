#include <algorithm>
#include <bit>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/dflash_sampling.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace {
namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;
void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Output {
  std::vector<std::int32_t> tokens;
  std::vector<std::uint64_t> rng;
  std::vector<std::uint32_t> logit_bits;
  std::uint64_t copied{0}, accepted{0};
  bool operator==(const Output&) const = default;
};
Output Finish(qfn::Session& session, sampling::SamplerState sampler,
              std::string& error) {
  const auto before = session.Statistics();
  Output output;
  bool stopped = false;
  for (unsigned cycle = 0; cycle < 128; ++cycle) {
    qfn::Session::DecodeResult step;
    Require(session.DecodeDFlashStep(8, sampler, &step, &error), error);
    output.tokens.insert(output.tokens.end(), step.tokens.begin(),
                         step.tokens.end());
    output.rng.push_back(sampler.rng_state());
    if (step.stop) {
      stopped = true;
      break;
    }
    Require(!step.tokens.empty(), "lookup cycle made no progress");
  }
  Require(stopped, "constrained lookup did not stop");
  output.copied = session.Statistics().lookup - before.lookup;
  output.accepted =
      session.Statistics().lookup_accepted - before.lookup_accepted;
  // A single allowed grammar token can hide incorrect model state. Compare
  // the complete unmasked target logits after actual continuation execution.
  for (const auto value : session.Logits())
    output.logit_bits.push_back(std::bit_cast<std::uint32_t>(value));
  return output;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    std::fprintf(stderr, "SKIP: external target/draft paths required\n");
    return 77;
  }
  try {
    Require(argc == 3, "pass target first-shard and draft GGUF paths");
    std::string error;
    auto model = qfn::Model::Load(
        argv[1],
        {.max_context = 512,
         .dflash_model_path = argv[2],
         .max_draft_tokens = 7,
         .dflash_policy = gufo::speculative::DFlashDraftPolicy::kFixed},
        &error);
    Require(bool(model), error);
    auto draft = model->LoadDFlashDraft(argv[2], &error);
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, 512, &error, draft);
    Require(bool(draft) && bool(session), error);
    // Independent grammar pieces force a known copyable history. Actual token
    // IDs, weights, target forward, lookup matching, draft KV, rollback and
    // snapshot formats stay real; this is not a model-quality assertion.
    constexpr std::int32_t quote = 500, first_letter = 1000;
    std::vector<std::int32_t> expected{501, quote, 503, quote, 504, quote};
    for (unsigned repeat = 0; repeat < 2; ++repeat)
      for (unsigned letter = 0; letter < 26; ++letter)
        expected.push_back(first_letter + letter);
    expected.push_back(quote);
    expected.push_back(502);
    std::vector<std::int32_t> prompt{17};
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      prompt.insert(prompt.end(), expected.begin(), expected.end());
      prompt.push_back(18 + repeat);
    }
    auto constraint = std::make_shared<sampling::TokenConstraint>();
    constraint->grammar = sampling::JsonConstraint::Compile(
        gufo::json::parse(
            R"({"type":"object","properties":{"x":{"type":"string","const":"ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ"}},"required":["x"],"additionalProperties":false})"),
        true);
    constraint->vocabulary = std::make_shared<sampling::ConstraintVocabulary>(
        model->VocabSize(), [&](std::uint32_t token) {
          std::string piece = "!";
          if (token == quote)
            piece = "\"";
          else if (token == 501)
            piece = "{";
          else if (token == 502)
            piece = "}";
          else if (token == 503)
            piece = "x";
          else if (token == 504)
            piece = ":";
          else if (token >= first_letter && token < first_letter + 26)
            piece.assign(1, 'A' + token - first_letter);
          return sampling::ConstraintVocabulary::Piece{
              .text = model->IsStopToken(token) ? "" : piece,
              .stop = model->IsStopToken(token)};
        });
    sampling::SamplingConfig config;
    config.seed = 513;
    config.constraint = constraint;
    Require(session->Sync(prompt, &error), error);
    sampling::SamplerState sampler(config);
    std::vector<std::int32_t> prefix;
    for (unsigned cycle = 0; session->Statistics().lookup_accepted == 0;
         ++cycle) {
      Require(cycle < 48, "native DFlash lookup never accepted a copy");
      qfn::Session::DecodeResult step;
      // A one-proposal neural plan must not restrict the independent copy
      // budget: the loaded target still has space to verify seven copies.
      Require(session->DecodeDFlashStep(8, sampler, &step, &error, true, 1),
              error);
      Require(!step.stop && !step.tokens.empty(), "lookup did not activate");
      if (cycle == 0)
        Require(session->Statistics().lookup == 0,
                "first request step copied a prompt wrapper");
      prefix.insert(prefix.end(), step.tokens.begin(), step.tokens.end());
    }
    auto saved = session->SaveSnapshot(
        &error, qfn::Session::SnapshotMode::kMaterialized);
    Require(bool(saved), error);
    Require(session->Statistics().lookup > 1,
            "lookup was restricted to the one-token neural plan");
    std::vector<std::uint8_t> bytes(saved->SizeBytes());
    Require(saved->CopyTo(bytes),
            "active lookup snapshot serialization failed");
    const auto first = Finish(*session, sampler, error);
    auto complete = prefix;
    complete.insert(complete.end(), first.tokens.begin(), first.tokens.end());
    Require(complete == expected && first.accepted > 0,
            "lookup changed forced target output or never resumed");
    Require(session->RestoreSnapshot(*saved, &error), error);
    Require(Finish(*session, sampler, error) == first,
            "active lookup object restore changed tokens/RNG/copy counts");
    session->Reset();
    Require(session->RestoreSnapshot(bytes, &error), error);
    Require(Finish(*session, sampler, error) == first,
            "active lookup serialized restore changed tokens/RNG/copy counts");
    session->ResetDraftPolicy();
    const auto before = session->Statistics().lookup;
    sampling::SamplerState fresh(config);
    qfn::Session::DecodeResult step;
    Require(session->DecodeDFlashStep(8, fresh, &step, &error), error);
    Require(session->Statistics().lookup == before,
            "new request retained an active copy cursor");
    // Positive sampled-copy coverage: the independent literal grammar makes
    // each copied target token certain, so every point mass must be accepted.
    // The scalar-teacher suite separately exercises real neural/mixed q and
    // rejected copies, with and without compact target candidate rows.
    session->Reset();
    Require(session->Sync(prompt, &error), error);
    auto sampled_config = config;
    sampled_config.temperature = 0.8F;
    sampled_config.top_k = 40;
    sampling::SamplerState sampled(sampled_config);
    const auto accepted_before = session->Statistics().accepted;
    std::size_t position = 0;
    while (position < expected.size()) {
      const auto anchor = sampled.Sample(session->Logits());
      Require(anchor == expected[position], "sampled literal anchor differs");
      const auto count =
          std::min<std::size_t>(7, expected.size() - position - 1);
      gufo::speculative::DraftProposal copied;
      copied.start_pos = session->Position();
      if (count) {
        const auto chain = std::span(expected).subspan(position, count + 1);
        qfn::ApplyDFlashLookup(copied, chain, count, true);
      }
      qfn::Session::DecodeResult verified;
      Require(session->VerifySampledDraft(anchor, copied, sampled, &verified,
                                          &error),
              error);
      Require(!verified.stop && verified.tokens.size() == count + 1 &&
                  std::equal(verified.tokens.begin(), verified.tokens.end(),
                             expected.begin() + position),
              "certain sampled copy was rejected or changed");
      position += verified.tokens.size();
    }
    const auto sampled_copies =
        session->Statistics().accepted - accepted_before;
    Require(sampled_copies > 0, "sampled point-mass acceptance unexercised");
    qfn::Session::DecodeResult terminal;
    Require(session->DecodeDFlashStep(8, sampled, &terminal, &error) &&
                terminal.stop && terminal.tokens.empty(),
            "sampled copy did not preserve EOS/draft-context coherence");
    // A full context stops normally even though its remaining physical
    // verification budget is zero. It is not a zero-token caller request.
    session->Reset();
    auto full_prompt = prompt;
    full_prompt.resize(512, prompt.back());
    Require(session->Sync(full_prompt, &error), error);
    sampling::SamplerState at_end(config);
    const auto end_rng = at_end.rng_state();
    qfn::Session::DecodeResult full;
    Require(session->DecodeDFlashStep(8, at_end, &full, &error) && full.stop &&
                full.tokens.empty() && at_end.rng_state() == end_rng &&
                session->Position() == 512,
            "full context became an invalid zero-token request");
    std::printf(
        "PASS: native DFlash lookup, %zu exact forced tokens, "
        "continued copied=%llu accepted=%llu, object/serialized "
        "restore, EOS and new-request gate; sampled accepted=%llu\n",
        complete.size(), static_cast<unsigned long long>(first.copied),
        static_cast<unsigned long long>(first.accepted),
        static_cast<unsigned long long>(sampled_copies));
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "%s\n", exception.what());
    return 1;
  }
}
