#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
void ValidatePleOwner(const std::uint32_t* local, const std::uint32_t* peer,
                      std::uint32_t* output, std::uint32_t rows,
                      std::uint32_t values, bool owner, std::uint32_t* failed,
                      const std::uint32_t* timed_out, hipStream_t stream);
namespace {
struct GatherState {
  std::shared_ptr<Communicator> communicator;
  std::uint32_t heads;
  std::uint32_t embedding_dim;
  std::uint32_t max_tokens;
  std::uint32_t* host{nullptr};
  std::uint32_t* device{nullptr};
  std::uint64_t owner_rows{0};
  std::uint64_t remote_rows{0};

  GatherState(std::shared_ptr<Communicator> c, std::uint32_t h, std::uint32_t e,
              std::uint32_t requested_tokens)
      : communicator(std::move(c)), heads(h), embedding_dim(e) {
    if (heads == 0 || embedding_dim == 0 || communicator->world_size() != 2 ||
        requested_tokens < 2)
      throw std::runtime_error("invalid cooperative PLE configuration");
    const auto words = communicator->MaxPartialBytes() / sizeof(std::uint32_t);
    const auto per_token = std::size_t{heads} + embedding_dim;
    const auto capacity = words > 4 ? (words - 4) / per_token : 0;
    max_tokens = static_cast<std::uint32_t>(
        std::min<std::size_t>(requested_tokens, capacity));
    if (max_tokens < 2)
      throw std::runtime_error("cooperative PLE exchange window is too small");
    const std::size_t bytes =
        (16 + 4 + std::size_t{max_tokens} * per_token) * 4;
    if (hipHostMalloc(reinterpret_cast<void**>(&host), bytes,
                      hipHostMallocMapped | hipHostMallocCoherent) !=
        hipSuccess)
      throw std::runtime_error("cooperative PLE staging allocation failed");
    if (hipHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0) !=
        hipSuccess) {
      (void)hipHostFree(host);
      host = nullptr;
      throw std::runtime_error("cooperative PLE staging alias failed");
    }
    std::memset(host, 0, bytes);
  }
  ~GatherState() {
    std::fprintf(
        stderr, "cooperative_ple rank=%u owner_rows=%llu remote_rows=%llu\n",
        communicator->rank(), static_cast<unsigned long long>(owner_rows),
        static_cast<unsigned long long>(remote_rows));
    if (host != nullptr)
      (void)hipHostFree(host);
  }
};
}  // namespace

Executor::PleGather Executor::TwoRankPleGather(
    std::shared_ptr<Communicator> communicator, std::uint32_t heads,
    std::uint32_t embedding_dim, std::uint32_t max_tokens) {
  auto state = std::make_shared<GatherState>(std::move(communicator), heads,
                                             embedding_dim, max_tokens);
  const bool owner = state->communicator->rank() == 1;
  PleGather result;
  result.local_owner = owner;
  result.max_tokens = state->max_tokens;
  result.exchange = [state, owner](
                        std::span<const std::uint32_t> rows, float* host_output,
                        const float* device_output, std::size_t values,
                        bool local_ok, hipStream_t stream, std::string* error) {
    if (rows.empty() ||
        rows.size() > std::size_t{state->max_tokens} * state->heads ||
        rows.size() % state->heads != 0 ||
        values != rows.size() / state->heads * state->embedding_dim) {
      if (error)
        *error = "invalid cooperative PLE row/embedding shape";
      return false;
    }
    auto* packet = state->host + 16;
    packet[0] = 0x504c4531u;
    packet[1] = 1;
    packet[2] = static_cast<std::uint32_t>(rows.size());
    packet[3] = owner ? (local_ok ? 1u : 0u) : 2u;
    std::memcpy(packet + 4, rows.data(), rows.size_bytes());
    const std::size_t metadata = 4 + rows.size();
    if (owner && local_ok) {
      std::memcpy(packet + metadata, host_output, values * 4);
      state->owner_rows += rows.size();
    } else {
      std::memset(packet + metadata, 0, values * 4);
      if (!owner)
        state->remote_rows += rows.size();
    }
    const std::size_t bytes = (metadata + values) * 4;
    Communicator::QueuedPartial queued;
    if (!state->communicator->QueuePartial(bytes, &queued, error))
      return false;
    StagePartial(reinterpret_cast<const float*>(state->device + 16),
                 queued.send, bytes / 4, queued.arrivals, queued.ready,
                 queued.value, stream);
    WaitValue(queued.arrived, queued.value, queued.wait_ticks, queued.timed_out,
              stream);
    ValidatePleOwner(
        state->device + 16, reinterpret_cast<const std::uint32_t*>(queued.peer),
        reinterpret_cast<std::uint32_t*>(const_cast<float*>(device_output)),
        static_cast<std::uint32_t>(rows.size()),
        static_cast<std::uint32_t>(values), owner, state->device,
        queued.timed_out, stream);
    return true;
  };
  result.status = [state](std::string* error) {
    if (std::atomic_ref<std::uint32_t>(*state->host)
            .load(std::memory_order_acquire) != 0) {
      if (error)
        *error = "cooperative PLE metadata/read failure";
      return false;
    }
    return true;
  };
  return result;
}
}  // namespace gufo::models::qwen38_flash_next::rocm
