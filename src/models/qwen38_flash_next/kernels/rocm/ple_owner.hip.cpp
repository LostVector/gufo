#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
__global__ void ValidatePleOwnerKernel(const std::uint32_t* local,
                                       const std::uint32_t* peer,
                                       std::uint32_t* output,
                                       std::uint32_t rows, std::uint32_t values,
                                       bool owner, std::uint32_t* failed,
                                       const std::uint32_t* timed_out) {
  __shared__ bool valid;
  if (threadIdx.x == 0) {
    bool ok = *timed_out == 0 && local[0] == 0x504c4531u &&
              peer[0] == local[0] && local[1] == 1 && peer[1] == 1 &&
              local[2] == rows && peer[2] == rows &&
              local[3] == (owner ? 1u : 2u) && peer[3] == (owner ? 2u : 1u);
    for (std::uint32_t i = 0; i < rows; ++i)
      ok = ok && local[4 + i] == peer[4 + i];
    valid = ok;
    if (!ok)
      __hip_atomic_store(failed, 1u, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
  if (!valid || owner)
    return;
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < values)
    output[i] = peer[4 + rows + i];
}
}  // namespace

void ValidatePleOwner(const std::uint32_t* local, const std::uint32_t* peer,
                      std::uint32_t* output, std::uint32_t rows,
                      std::uint32_t values, bool owner, std::uint32_t* failed,
                      const std::uint32_t* timed_out, hipStream_t stream) {
  ValidatePleOwnerKernel<<<(values + 255) / 256, 256, 0, stream>>>(
      local, peer, output, rows, values, owner, failed, timed_out);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
