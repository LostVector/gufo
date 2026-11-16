#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"

#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <limits>
#include <utility>

namespace gufo::models::qwen38_flash_next::distributed {
namespace {

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

/// Bytes of `elements` consecutive values of one row, or zero when the count
/// is not a whole number of quantization blocks.
std::size_t SpanBytes(core::GgmlType type, std::size_t elements) {
  switch (type) {
    case core::GgmlType::kF32:
      return elements * 4;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      return elements * 2;
    default:
      return gufo::quant::QuantizedRowBytes(type, elements);
  }
}

bool CheckedMulAdd(std::size_t* total, std::size_t a, std::size_t b,
                   std::size_t c, std::string* error_msg) {
  if (a != 0 &&
      (b > std::numeric_limits<std::size_t>::max() / a ||
       (b != 0 && c > std::numeric_limits<std::size_t>::max() / (a * b)))) {
    SetError(error_msg, "TP routed weight plan overflows");
    return false;
  }
  const std::size_t value = a * b * c;
  if (*total > std::numeric_limits<std::size_t>::max() - value) {
    SetError(error_msg, "TP routed weight plan overflows");
    return false;
  }
  *total += value;
  return true;
}

/// A gate or up projection: `ff_count` of each expert's `expert_ff` rows.
bool AddRowShare(const TensorRef& t, const TpPartition& partition,
                 RoutedWeightPlan* plan, std::string* error_msg) {
  const std::size_t row_bytes = t.RowBytes();
  if (t.empty() || row_bytes == 0 || t.rows != partition.expert_ff) {
    SetError(error_msg, "TP routed tensor " + std::string(t.name) +
                            " does not match the intermediate split");
    return false;
  }
  return CheckedMulAdd(&plan->full_encoded_bytes, row_bytes, t.rows, t.experts,
                       error_msg) &&
         CheckedMulAdd(&plan->local_encoded_bytes, row_bytes,
                       partition.ff_count, t.experts, error_msg);
}

/// A down projection: `ff_count` of each row's `expert_ff` columns.
bool AddColumnShare(const TensorRef& t, const TpPartition& partition,
                    RoutedWeightPlan* plan, std::string* error_msg) {
  const std::size_t row_bytes = t.RowBytes();
  const std::size_t share_bytes = SpanBytes(t.type, partition.ff_count);
  if (t.empty() || row_bytes == 0 || t.cols != partition.expert_ff) {
    SetError(error_msg, "TP routed tensor " + std::string(t.name) +
                            " does not match the intermediate split");
    return false;
  }
  if (share_bytes == 0) {
    SetError(error_msg,
             "TP share of " + std::string(t.name) +
                 " does not fall on a quantization-block boundary: " +
                 std::to_string(partition.ff_count) +
                 " columns must be a multiple of the quantization "
                 "block size (32 for Q8_0 and Q5_1, 256 for Q4_K "
                 "and Q5_K)");
    return false;
  }
  return CheckedMulAdd(&plan->full_encoded_bytes, row_bytes, t.rows, t.experts,
                       error_msg) &&
         CheckedMulAdd(&plan->local_encoded_bytes, share_bytes, t.rows,
                       t.experts, error_msg);
}

bool AddLayerRouted(const LayerWeights& layer, const TpPartition& partition,
                    RoutedWeightPlan* plan, std::string* error_msg) {
  return AddRowShare(layer.ffn_gate_exps, partition, plan, error_msg) &&
         AddRowShare(layer.ffn_up_exps, partition, plan, error_msg) &&
         AddColumnShare(layer.ffn_down_exps, partition, plan, error_msg);
}

}  // namespace

std::optional<RoutedWeightPlan> PlanRoutedBytes(const ModelWeights& weights,
                                                const TpPartition& partition,
                                                const MtpWeights* mtp_weights,
                                                std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  if (!partition.Valid() || partition.expert_ff != weights.config.expert_ff ||
      (mtp_weights != nullptr &&
       mtp_weights->config.expert_ff != partition.expert_ff)) {
    SetError(error_msg,
             "TP routed weight plan geometry does not match the model");
    return std::nullopt;
  }
  RoutedWeightPlan plan{};
  for (const auto& layer : weights.layers) {
    if (!AddLayerRouted(layer, partition, &plan, error_msg)) {
      return std::nullopt;
    }
  }
  if (mtp_weights != nullptr &&
      !AddLayerRouted(mtp_weights->block, partition, &plan, error_msg)) {
    return std::nullopt;
  }
  return plan;
}

std::optional<TpPartition> TpPartition::Create(std::uint32_t expert_ff,
                                               std::uint32_t rank,
                                               std::uint32_t world_size,
                                               std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  if (expert_ff == 0 || world_size == 0 || rank >= world_size ||
      expert_ff % world_size != 0) {
    SetError(error_msg,
             "TP expert split requires a nonzero, evenly divisible expert "
             "intermediate size and a valid rank");
    return std::nullopt;
  }
  TpPartition result;
  result.rank = rank;
  result.world_size = world_size;
  result.expert_ff = expert_ff;
  result.ff_count = expert_ff / world_size;
  result.ff_begin = result.ff_count * rank;
  return result;
}

bool MixerSplitFits(const ModelWeights& weights, const TpPartition& partition,
                    std::string* error) {
  const auto& c = weights.config;
  const auto world = partition.world_size;
  const auto fail = [&](std::string message) {
    SetError(error, std::move(message));
    return false;
  };
  if (!partition.Valid() || c.ssm_num_k_heads == 0 ||
      c.ssm_num_k_heads % world != 0 ||
      c.ssm_num_v_heads % c.ssm_num_k_heads != 0 || c.num_kv_heads == 0 ||
      c.num_kv_heads % world != 0 || c.num_heads % c.num_kv_heads != 0 ||
      c.ssm_head_dim == 0 || c.ssm_head_dim % 32 != 0 || c.head_dim == 0 ||
      c.head_dim % 32 != 0)
    return fail("head counts or dimensions");
  const auto projection = [](const TensorRef& t) {
    if (t.empty() || t.experts != 1 || t.RowBytes() == 0)
      return false;
    switch (t.type) {
      case core::GgmlType::kF32:
      case core::GgmlType::kF16:
      case core::GgmlType::kBF16:
      case core::GgmlType::kQ8_0:
      case core::GgmlType::kQ4_K:
      case core::GgmlType::kQ5_K:
      case core::GgmlType::kQ6_K:
      case core::GgmlType::kIQ4_NL:
      case core::GgmlType::kIQ4_XS:
        return true;
      default:
        return false;
    }
  };
  const auto columns = [](const TensorRef& t, std::uint64_t first,
                          std::uint64_t count) {
    return count != 0 && first <= t.cols && count <= t.cols - first &&
           (first == 0 || SpanBytes(t.type, first) != 0) &&
           SpanBytes(t.type, count) != 0;
  };
  const std::uint64_t keys = c.ssm_num_k_heads / world;
  const std::uint64_t first_key = keys * partition.rank;
  const std::uint64_t groups = c.ssm_num_v_heads / c.ssm_num_k_heads;
  const std::uint64_t queries = c.num_heads / world;
  for (const auto& l : weights.layers) {
    if (l.linear) {
      if (!projection(l.ssm_qkv) || !projection(l.ssm_gate) ||
          !projection(l.ssm_out) || l.ssm_qkv.cols != l.ssm_gate.cols ||
          (l.ssm_alpha.type != core::GgmlType::kF32 &&
           l.ssm_alpha.type != core::GgmlType::kQ8_0 &&
           l.ssm_alpha.type != core::GgmlType::kBF16) ||
          l.ssm_beta.type != l.ssm_alpha.type ||
          l.ssm_beta.cols != l.ssm_alpha.cols ||
          l.ssm_conv1d.type != core::GgmlType::kF32 ||
          l.ssm_dt.type != core::GgmlType::kF32 ||
          l.ssm_a.type != core::GgmlType::kF32)
        return fail("projection types of a GDN layer");
      for (std::uint64_t group = 0; group < groups; ++group)
        if (!columns(l.ssm_out,
                     (group * c.ssm_num_k_heads + first_key) * c.ssm_head_dim,
                     keys * c.ssm_head_dim))
          return fail("GDN output share splits a quantization block");
    } else {
      if (!projection(l.attn_q) || !projection(l.attn_k) ||
          !projection(l.attn_v) || !projection(l.attn_out) ||
          l.attn_q.cols != l.attn_k.cols || l.attn_q.cols != l.attn_v.cols)
        return fail("projection types of an attention layer");
      if (!columns(l.attn_out, queries * partition.rank * c.head_dim,
                   queries * c.head_dim))
        return fail("attention output share splits a quantization block");
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::distributed
