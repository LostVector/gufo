// Real two-process/native-transport correctness probe, not a latency benchmark.
// Usage: probe RANK BOOTSTRAP_HOST PORT RDMA_DEVICE good|rows|read
#include <hip/hip_runtime.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;

void Hip(hipError_t e) {
  if (e != hipSuccess)
    throw std::runtime_error(hipGetErrorString(e));
}

int main(int argc, char** argv) {
  if (argc != 6)
    return 2;
  const auto rank = static_cast<std::uint32_t>(std::stoul(argv[1]));
  const std::string mode = argv[5];
  if (rank > 1 || (mode != "good" && mode != "rows" && mode != "read"))
    return 2;
  try {
    Hip(hipSetDevice(0));
    std::string error;
    q::IbrverbsConfig config;
    config.rank = rank;
    config.bootstrap_host = argv[2];
    config.bootstrap_port = static_cast<std::uint16_t>(std::stoul(argv[3]));
    config.rdma_device = argv[4];
    auto comm = q::CreateIbrverbsCommunicator(config, &error);
    if (!comm || !comm->SupportsPleOwner() ||
        !comm->BeginOperation(7001, &error))
      throw std::runtime_error(error.empty() ? "PLE negotiation failed"
                                             : error);
    hipStream_t stream = nullptr;
    Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    {
      constexpr std::uint32_t heads = 2, dim = 37, capacity = 64;
      auto gather = q::Executor::TwoRankPleGather(comm, heads, dim, capacity);
      float* device = nullptr;
      Hip(hipMalloc(&device, capacity * dim * sizeof(float)));
      // Capacity, zero, ordinary/speculative boundary and fallback contracts.
      if (gather.CanGather(0, true) || !gather.CanGather(32, false) ||
          gather.CanGather(33, false) || !gather.CanGather(64, true) ||
          gather.CanGather(65, true))
        throw std::runtime_error("PLE eligibility contract failed");
      constexpr std::array shapes{1U, 2U, 8U, 9U, 32U, 33U, 64U};
      constexpr std::array bits{0U,          0x80000000U, 1U,
                                0x7f800000U, 0xff800000U, 0x7fc12345U,
                                0x3f812345U};
      std::size_t completed = 0;
      for (unsigned repeat = 0; repeat < (mode == "good" ? 20U : 1U);
           ++repeat) {
        for (const auto tokens : shapes) {
          std::vector<std::uint32_t> rows(tokens * heads);
          std::vector<float> expected(tokens * dim), input(tokens * dim),
              actual(tokens * dim);
          for (std::size_t i = 0; i < rows.size(); ++i)
            rows[i] = static_cast<std::uint32_t>(i * 19 + repeat * 3);
          for (std::size_t i = 0; i < expected.size(); ++i) {
            expected[i] =
                std::bit_cast<float>(bits[(i + repeat) % bits.size()]);
            input[i] = rank == 1 ? expected[i] : -123.0F;
          }
          Hip(hipMemcpyAsync(device, input.data(), input.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
          if (mode == "rows" && rank == 0)
            rows.back() ^= 1;
          const bool local_ok = !(mode == "read" && rank == 1);
          if (!gather.exchange(rows, input.data(), device, input.size(),
                               local_ok, stream, &error)) {
            comm->Abort(error);
            throw std::runtime_error(error);
          }
          Hip(hipStreamSynchronize(stream));
          const bool valid = gather.status(&error);
          if (mode != "good") {
            if (valid)
              throw std::runtime_error("invalid payload was accepted");
            comm->Abort("expected negative PLE probe termination");
            Hip(hipFree(device));
            Hip(hipStreamDestroy(stream));
            std::printf("PASS rank=%u mode=%s rejected invalid payload\n", rank,
                        mode.c_str());
            return 0;
          }
          if (!comm->CheckQueued(&error) || !valid)
            throw std::runtime_error(error);
          Hip(hipMemcpy(actual.data(), device, actual.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
          for (std::size_t i = 0; i < actual.size(); ++i)
            if (std::bit_cast<std::uint32_t>(actual[i]) !=
                std::bit_cast<std::uint32_t>(expected[i]))
              throw std::runtime_error("PLE output bits changed");
          ++completed;
        }
      }
      if (!comm->EndOperation(7001, &error))
        throw std::runtime_error(error);
      Hip(hipFree(device));
      std::printf("PASS rank=%u exchanges=%zu exact payload bits and reuse\n",
                  rank, completed);
    }
    Hip(hipStreamDestroy(stream));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL rank=%u %s\n", rank, e.what());
    return 1;
  }
}
