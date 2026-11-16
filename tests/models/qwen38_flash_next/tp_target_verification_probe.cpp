#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"
namespace q = gufo::models::qwen38_flash_next::rocm;
void need(bool b, const std::string& m) {
  if (!b)
    throw std::runtime_error(m);
}
void hip(hipError_t e) {
  need(e == hipSuccess, hipGetErrorString(e));
}
struct Buffer {
  void* p{};
  explicit Buffer(std::size_t n) {
    hip(hipMalloc(&p, std::max<std::size_t>(n, 4)));
  }
  ~Buffer() {
    if (p)
      hipFree(p);
  }
  template<class T>
  T* as() const {
    return static_cast<T*>(p);
  }
};
int main(int argc, char** argv) {
  try {
    need(argc == 6, "probe rank host device mode output");
    unsigned rank = std::stoul(argv[1]);
    std::string mode = argv[4];
    need(rank < 2, "rank");
    hip(hipSetDevice(0));
    q::IbrverbsConfig cfg;
    cfg.rank = rank;
    cfg.bootstrap_host = argv[2];
    cfg.rdma_device = argv[3];
    cfg.bootstrap_port = 29769;
    std::string error;
    auto comm = q::CreateIbrverbsCommunicator(cfg, &error);
    need(bool(comm), error);
    need(comm->SupportsCompactTargetVerification(),
         "new host-window feature absent");
    std::ofstream out(argv[5]);
    need(bool(out), "output");
    hipStream_t stream;
    hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto candidates = q::Executor::TwoRankTargetVerificationCandidates(comm);
    constexpr unsigned vocab = 248320, half = vocab / 2;
    auto materialize = q::Executor::TwoRankTargetVerificationRow(comm, vocab);
    std::vector<unsigned> shapes = mode == "matching"
                                       ? std::vector<unsigned>{2, 3, 7, 8}
                                       : std::vector<unsigned>{3};
    unsigned scope = 45000;
    for (auto rows : shapes) {
      unsigned vr = rows - 1;
      std::vector<float> full(std::size_t(rows) * vocab),
          local(std::size_t(rows) * half);
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned i = 0; i < vocab; ++i) {
          auto x = (i * 7919U + r * 71U) % 65521U;
          float value = float(int(x) - 32768) / 4096.F;
          if (i == 0)
            value = -0.F;
          if (i == half)
            value = 0.F;
          full[std::size_t(r) * vocab + i] = value;
          if (i >= rank * half && i < (rank + 1) * half)
            local[std::size_t(r) * half + i - rank * half] = value;
        }
      std::vector<gufo::sampling::TokenPenalty> penalties;
      q::GreedyPenaltyRows batch{};
      for (unsigned r = 0; r < vr; ++r) {
        for (auto token : {0U, half - 1, half, vocab - 1})
          penalties.push_back({token, r + 1, 1});
        batch.offsets[r + 1] = penalties.size();
      }
      unsigned count = vr * 64 + penalties.size(),
               words = vr * 128 + penalties.size();
      Buffer gpuFull(full.size() * 4), park(local.size() * 4),
          gpuPen(penalties.size() * sizeof(penalties[0])),
          ids(vr * q::MtpCandidateWorkspaceSize(vocab) * 4),
          scratch(vr * q::MtpCandidateWorkspaceSize(vocab) * 4),
          pack(words * 4),
          result(count * sizeof(gufo::sampling::LogitCandidate)),
          reference(count * sizeof(gufo::sampling::LogitCandidate)),
          expanded(full.size() * 4);
      hip(hipMemcpy(gpuFull.p, full.data(), full.size() * 4,
                    hipMemcpyHostToDevice));
      hip(hipMemcpy(park.p, local.data(), local.size() * 4,
                    hipMemcpyHostToDevice));
      hip(hipMemcpy(gpuPen.p, penalties.data(),
                    penalties.size() * sizeof(penalties[0]),
                    hipMemcpyHostToDevice));
      batch.penalties = gpuPen.as<gufo::sampling::TokenPenalty>();
      need(comm->BeginOperation(++scope, &error), error);
      q::PartitionVerificationCandidates(
          park.as<float>(), batch, ids.as<std::uint32_t>(),
          scratch.as<std::uint32_t>(), pack.as<std::uint32_t>(), vr, vocab,
          rank, stream);
      if (mode == "row-mismatch")
        need(materialize(expanded.as<float>(), park.as<float>() + rank * half,
                         rank, rows, stream, &error),
             error);
      else
        need(candidates(
                 pack.as<std::uint32_t>(), batch,
                 result.as<gufo::sampling::LogitCandidate>(), vr, vocab,
                 0xC100000000000000ULL | rows |
                     (mode == "signature-mismatch" ? std::uint64_t(rank) << 16
                                                   : 0),
                 stream, &error),
             error);
      hip(hipStreamSynchronize(stream));
      bool accepted = comm->CheckQueued(&error);
      if (mode != "matching") {
        need(!accepted, "semantic mismatch accepted");
        need(error.find("identity or size mismatch") != std::string::npos ||
                 error.find("peer") != std::string::npos ||
                 error.find("aborted") != std::string::npos ||
                 error.find("closed") != std::string::npos,
             "unexpected rejection: " + error);
        out << "REJECT " << error << '\n';
        out.flush();
        comm->Abort("expected mismatch shutdown");
        break;
      }
      need(accepted, error);
      q::VerificationCandidates(
          gpuFull.as<float>(), batch, ids.as<std::uint32_t>(),
          scratch.as<std::uint32_t>(),
          reference.as<gufo::sampling::LogitCandidate>(), vr, vocab, stream);
      hip(hipStreamSynchronize(stream));
      std::vector<gufo::sampling::LogitCandidate> a(count), b(count);
      hip(hipMemcpy(a.data(), result.p, count * sizeof(a[0]),
                    hipMemcpyDeviceToHost));
      hip(hipMemcpy(b.data(), reference.p, count * sizeof(b[0]),
                    hipMemcpyDeviceToHost));
      for (unsigned i = 0; i < count; ++i)
        need(a[i].token == b[i].token &&
                 std::bit_cast<std::uint32_t>(a[i].value) ==
                     std::bit_cast<std::uint32_t>(b[i].value),
             "transport candidate bit mismatch");
      for (unsigned r = rows; r-- > 0;) {
        need(materialize(expanded.as<float>() + std::size_t(r) * vocab,
                         park.as<float>() + std::size_t(r) * half, r, rows,
                         stream, &error),
             error);
        hip(hipStreamSynchronize(stream));
        need(comm->CheckQueued(&error), error);
      }
      std::vector<float> actual(full.size());
      hip(hipMemcpy(actual.data(), expanded.p, actual.size() * 4,
                    hipMemcpyDeviceToHost));
      for (unsigned i = 0; i < full.size(); ++i)
        need(std::bit_cast<std::uint32_t>(full[i]) ==
                 std::bit_cast<std::uint32_t>(actual[i]),
             "full materialized row bits differ");
      hip(hipMemcpy(actual.data(), park.p, local.size() * 4,
                    hipMemcpyDeviceToHost));
      for (unsigned i = 0; i < local.size(); ++i)
        need(std::bit_cast<std::uint32_t>(local[i]) ==
                 std::bit_cast<std::uint32_t>(actual[i]),
             "parked source changed");
      need(comm->EndOperation(scope, &error), error);
      out << "PASS rows=" << rows
          << " compact candidates and all reverse-order full rows exact\n";
      out.flush();
    }
    hip(hipStreamSynchronize(stream));
    hip(hipStreamDestroy(stream));
    std::cout << "PASS " << mode << " rank=" << rank << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
