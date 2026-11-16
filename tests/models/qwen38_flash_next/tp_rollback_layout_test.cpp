// Actual wide-buffer allocation and exact-size GPU copies, with no model
// weights or mock collective. Full and split-head geometry must agree with
// the recurrent state owned by one rank.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
void RequireLayout(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
void CheckLayout(hipError_t result) {
  RequireLayout(result == hipSuccess, hipGetErrorString(result));
}
struct DeviceModelTestAccess {
  static std::unique_ptr<DeviceModel> Geometry(unsigned world) {
    auto model = std::unique_ptr<DeviceModel>(new DeviceModel());
    auto& c = model->config_;
    c.num_layers = 4;
    c.full_attention_interval = 4;
    c.ssm_num_k_heads = 16;
    c.ssm_num_v_heads = 48;
    c.ssm_head_dim = 128;
    c.ssm_conv_kernel = 4;
    c.ple_layer = -1;
    model->tp_world_size_ = world;
    model->trunk_heads_.ssm_k = 16 / world;
    model->trunk_heads_.ssm_v = 48 / world;
    model->trunk_heads_.ssm_dim = 128;
    return model;
  }
};
struct ExecutorTestAccess {
  static void WideRankLayout(unsigned world) {
    auto model = DeviceModelTestAccess::Geometry(world);
    Executor executor;
    executor.model_ = model.get();
    std::string error;
    const bool allocated = executor.EnsureWideSegments(1, &error);
    RequireLayout(allocated, error.c_str());
    const auto& segment = executor.wide_segments_.front();
    // Independent full-machine fixture: three recurrent layers. Rank-local
    // heads halve every recurrent region for TP2; there is no PLE region.
    const std::size_t conv = 3 * (2 * (16 / world) + 48 / world) * 128;
    const std::size_t state = (48 / world) * 128 * 128;
    const std::size_t delta = (16 / world) * 128 + (48 / world) * 130;
    constexpr std::size_t snapshots = 7;
    const auto per_layer = conv * (snapshots + 1) + 2 * state + 6 * delta;
    RequireLayout(executor.WideSegmentFloats() == 3 * per_layer,
                  "wide allocation charges another rank's state");
    float* source = nullptr;
    CheckLayout(hipMalloc(&source, conv * sizeof(float)));
    try {
      CheckLayout(hipMemset(source, 0, conv * sizeof(float)));
      for (unsigned il = 0; il < 3; ++il) {
        RequireLayout(segment.state_start[il] - segment.conv_start[il] ==
                          static_cast<std::ptrdiff_t>(conv),
                      "convolution region is not rank-local");
        RequireLayout(segment.conv[il].rows[0] - segment.state_start[il] ==
                          static_cast<std::ptrdiff_t>(state),
                      "starting state region is not rank-local");
        for (unsigned row = 0; row < snapshots; ++row) {
          RequireLayout(
              segment.state[il].rows[row] - segment.conv[il].rows[row] ==
                  static_cast<std::ptrdiff_t>(conv),
              "rollback convolution region has wrong width");
          if (row + 1 < snapshots)
            RequireLayout(
                segment.conv[il].rows[row + 1] - segment.state[il].rows[row] ==
                    static_cast<std::ptrdiff_t>(row == 0 ? state : delta),
                "rollback state region has wrong head count");
        }
        CheckLayout(hipMemcpy(segment.conv_start[il], source,
                              conv * sizeof(float), hipMemcpyDeviceToDevice));
      }
    } catch (...) {
      (void)hipFree(source);
      throw;
    }
    CheckLayout(hipFree(source));
    std::cout << "PASS wide recurrent layout world=" << world
              << " conv_floats=" << conv << " state_floats=" << state << '\n';
  }
};
}  // namespace gufo::models::qwen38_flash_next::rocm

int main() {
  try {
    gufo::models::qwen38_flash_next::rocm::ExecutorTestAccess::WideRankLayout(
        1);
    gufo::models::qwen38_flash_next::rocm::ExecutorTestAccess::WideRankLayout(
        2);
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
