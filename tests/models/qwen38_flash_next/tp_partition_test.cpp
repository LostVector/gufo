#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#include "src/models/qwen38_flash_next/prefill_rows.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::models::qwen38_flash_next::ModelWeights;
using gufo::models::qwen38_flash_next::TensorRef;
using gufo::models::qwen38_flash_next::distributed::PlanRoutedBytes;
using gufo::models::qwen38_flash_next::distributed::TpPartition;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

void RequireError(const std::optional<TpPartition>& result,
                  const char* message) {
  Require(!result.has_value(), message);
}

ModelWeights OneLayer(TensorRef gate, TensorRef up, TensorRef down) {
  ModelWeights weights;
  weights.config.expert_ff = 640;
  weights.layers.resize(1);
  weights.layers[0].ffn_gate_exps = gate;
  weights.layers[0].ffn_up_exps = up;
  weights.layers[0].ffn_down_exps = down;
  return weights;
}

}  // namespace

int main() {
  using gufo::models::qwen38_flash_next::PlanPrefillRows;
  const auto local = PlanPrefillRows(65536, 2560);
  Require(local.chunk == 4096 && local.through == 4224,
          "preserve upstream local chunk and short tail");
  const auto native = PlanPrefillRows(65536, 2560, (48U << 20) - 4096);
  Require(native.chunk == 4096 && native.through == 4224,
          "native window caps a complete trunk batch including its tail");
  for (const auto context : {1U, 127U, 128U, 4096U, 16384U, 65536U}) {
    for (std::size_t bytes = 0; bytes < (50U << 20); bytes += 7919) {
      const auto rows = PlanPrefillRows(context, 2560, bytes);
      Require(rows.chunk <= rows.through && rows.through <= context &&
                  std::size_t{rows.through} * 2560 * sizeof(float) <= bytes,
              "chunk and tail always fit context and exchange bounds");
      Require(rows.chunk > 0 || bytes < 2560 * sizeof(float),
              "every window fitting one token makes progress");
    }
  }
  Require(PlanPrefillRows(65536, 0).through == 0,
          "invalid hidden size cannot divide by zero");
  std::string error;
  const auto single = TpPartition::Create(640, 0, 1, &error);
  Require(single.has_value(), error.c_str());
  Require(!single->distributed() && single->ff_begin == 0 &&
              single->ff_count == 640,
          "one rank holds every intermediate unit");

  const auto first = TpPartition::Create(640, 0, 2, &error);
  const auto second = TpPartition::Create(640, 1, 2, &error);
  Require(first.has_value() && second.has_value(), error.c_str());
  Require(first->ff_begin == 0 && first->ff_count == 320,
          "rank zero holds the first half of every expert");
  Require(second->ff_begin == 320 && second->ff_count == 320,
          "rank one holds the second half of every expert");

  // The model's shapes: 2560-wide Q4_K gate/up rows, 640 per expert; down
  // rows 640 wide, in Q5_1 or Q8_0 (32-value blocks), 2560 per expert.
  const void* data = reinterpret_cast<const void*>(0x1000);
  const TensorRef gate{data, GgmlType::kQ4_K, 2560, 640, 4, 0, 0, "gate"};
  const TensorRef down_q5{data, GgmlType::kQ5_1, 640, 2560, 4, 0, 0, "down"};
  const TensorRef down_q8{data, GgmlType::kQ8_0, 640, 2560, 4, 0, 0, "down"};
  const std::size_t gate_row = 10 * 144;  // ten Q4_K super-blocks
  for (const auto& [down, block_bytes] :
       {std::pair{down_q5, std::size_t{24}},
        std::pair{down_q8, std::size_t{34}}}) {
    const auto plan =
        PlanRoutedBytes(OneLayer(gate, gate, down), *second, nullptr, &error);
    Require(plan.has_value(), error.c_str());
    const std::size_t full =
        2 * gate_row * 640 * 4 + block_bytes * 20 * 2560 * 4;
    Require(plan->full_encoded_bytes == full, "full routed bytes");
    Require(plan->local_encoded_bytes ==
                2 * gate_row * 320 * 4 + block_bytes * 10 * 2560 * 4,
            "a rank's share is half of every routed tensor");
  }

  // A down share must be whole quantization blocks: 16 of 32 values is not.
  const TensorRef narrow_down{data, GgmlType::kQ8_0, 32, 2560, 4, 0, 0, "down"};
  const TensorRef narrow_gate{data, GgmlType::kQ4_K, 2560, 32, 4, 0, 0, "gate"};
  auto narrow = OneLayer(narrow_gate, narrow_gate, narrow_down);
  narrow.config.expert_ff = 32;
  const auto narrow_split = TpPartition::Create(32, 0, 2, &error);
  Require(narrow_split.has_value(), error.c_str());
  Require(!PlanRoutedBytes(narrow, *narrow_split, nullptr, &error).has_value(),
          "reject a share that splits a quantization block");

  Require(
      !PlanRoutedBytes(OneLayer(gate, gate, down_q8),
                       *TpPartition::Create(320, 0, 2, &error), nullptr, &error)
           .has_value(),
      "reject a split that does not match the model");
  auto invalid = *first;
  invalid.ff_begin = 1;
  Require(!invalid.Valid(), "reject a misplaced share");
  RequireError(TpPartition::Create(641, 0, 2, &error),
               "reject an uneven intermediate size");
  RequireError(TpPartition::Create(0, 0, 1, &error),
               "reject an empty intermediate size");
  RequireError(TpPartition::Create(640, 2, 2, &error),
               "reject an out-of-range rank");

  // Mixed GSQ-RCO projections keep their formats. The GDN output gathers
  // three disjoint 1024-column spans; attention gathers 3072 columns.
  ModelWeights mixed;
  auto& c = mixed.config;
  c.expert_ff = 640;
  c.hidden_size = 2560;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ssm_head_dim = 128;
  c.num_heads = 24;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  mixed.layers.resize(2);
  const auto tensor = [&](GgmlType type, unsigned cols, unsigned rows) {
    return TensorRef{data, type, cols, rows, 1, 0, 0, "mixed"};
  };
  auto& g = mixed.layers[0];
  g.linear = true;
  g.ssm_qkv = tensor(GgmlType::kQ6_K, 2560, 10240);
  g.ssm_gate = tensor(GgmlType::kIQ4_XS, 2560, 6144);
  g.ssm_out = tensor(GgmlType::kQ6_K, 6144, 2560);
  g.ssm_alpha = g.ssm_beta = tensor(GgmlType::kBF16, 2560, 48);
  g.ssm_conv1d = tensor(GgmlType::kF32, 4, 10240);
  g.ssm_dt = g.ssm_a = tensor(GgmlType::kF32, 48, 1);
  auto& a = mixed.layers[1];
  a.linear = false;
  a.attn_q = tensor(GgmlType::kQ4_K, 2560, 12288);
  a.attn_k = tensor(GgmlType::kIQ4_XS, 2560, 512);
  a.attn_v = tensor(GgmlType::kQ5_K, 2560, 512);
  a.attn_out = tensor(GgmlType::kQ6_K, 6144, 2560);
  using gufo::models::qwen38_flash_next::distributed::MixerSplitFits;
  for (const auto split : {*first, *second})
    Require(MixerSplitFits(mixed, split, &error), error.c_str());
  {
    auto bad = mixed;
    bad.config.ssm_num_k_heads = 2;
    bad.config.ssm_num_v_heads = 6;
    Require(!MixerSplitFits(bad, *first, &error),
            "reject 128-column Q6_K GDN share");
    bad = mixed;
    bad.config.num_heads = 2;
    bad.config.head_dim = 128;
    Require(!MixerSplitFits(bad, *first, &error),
            "reject 128-column Q6_K attention share");
    bad = mixed;
    bad.config.ssm_num_k_heads = 0;
    Require(!MixerSplitFits(bad, *first, &error),
            "reject zero GDN key heads before division");
    bad = mixed;
    bad.layers[0].ssm_beta.type = GgmlType::kF32;
    Require(!MixerSplitFits(bad, *first, &error),
            "reject mixed alpha/beta stack formats");
    bad = mixed;
    bad.layers[0].ssm_qkv.type = GgmlType::kQ2_0;
    Require(!MixerSplitFits(bad, *first, &error),
            "reject a format without a dense compute path");
    bad = mixed;
    bad.layers[1].attn_v.data = nullptr;
    Require(!MixerSplitFits(bad, *first, &error),
            "reject missing attention projection");
    bad = mixed;
    bad.layers[0].ssm_out.cols = 256;
    Require(!MixerSplitFits(bad, *first, &error),
            "reject GDN column gather beyond a row");
    auto invalid_split = *first;
    invalid_split.world_size = 0;
    Require(!MixerSplitFits(mixed, invalid_split, &error),
            "reject invalid split before division");
  }
  {
    auto legacy = mixed;
    for (auto& l : legacy.layers) {
      for (auto* t : {&l.ssm_qkv, &l.ssm_gate, &l.ssm_out, &l.attn_q, &l.attn_k,
                      &l.attn_v, &l.attn_out})
        if (!t->empty())
          t->type = GgmlType::kQ8_0;
      l.ssm_alpha.type = l.ssm_beta.type = GgmlType::kF32;
    }
    for (const auto split : {*first, *second})
      Require(MixerSplitFits(legacy, split, &error),
              "preserve legacy Q8 mixer splitting");
  }
  {
    const auto low_gate = tensor(GgmlType::kIQ2_S, 2560, 640);
    auto lg = low_gate;
    lg.experts = 4;
    const TensorRef low_down{data, GgmlType::kQ2_0, 640, 2560, 4, 0,
                             0,    "Q2_0 down"};
    const auto plan =
        PlanRoutedBytes(OneLayer(lg, lg, low_down), *second, nullptr, &error);
    Require(plan.has_value(), error.c_str());
    Require(
        plan->local_encoded_bytes == 2 * 10 * 82 * 320 * 4 + 90 * 2560 * 4,
        "IQ2_S rows and Q2_0 64-value column shares keep exact encoded bytes");
  }

  std::puts("PASS: TP=2 intermediate split and Q4_K/Q5_1/Q8_0 shares");
  return 0;
}
