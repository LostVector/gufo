#include "src/models/qwen38_flash_next/kernels/rocm/dflash_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Shape(unsigned rows, unsigned hidden, unsigned streams) {
  constexpr unsigned taps = 5, tap = 2;
  constexpr float sentinel = 123456.0F;
  std::vector<float> input(std::size_t{rows} * streams * hidden);
  std::vector<float> output(std::size_t{rows} * taps * hidden, sentinel);
  float *device_input = nullptr, *device_output = nullptr;
  hipStream_t stream;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  Check(hipMalloc(&device_input, input.size() * sizeof(float)));
  Check(hipMalloc(&device_output, output.size() * sizeof(float)));
  // Each graph replay must read the new residual, not a cached host value.
  hipGraph_t graph;
  hipGraphExec_t executable;
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  gufo::models::qwen38_flash_next::rocm::CaptureDFlashFeatures(
      device_input, device_output, rows, hidden, streams, taps, tap, stream);
  Check(hipStreamEndCapture(stream, &graph));
  Check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (unsigned replay = 0; replay != 2; ++replay) {
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = (static_cast<int>((i * 37 + replay * 11) % 257) - 128) / 16.0F;
    std::fill(output.begin(), output.end(), sentinel);
    Check(hipMemcpyAsync(device_input, input.data(),
                         input.size() * sizeof(float), hipMemcpyHostToDevice,
                         stream));
    Check(hipMemcpyAsync(device_output, output.data(),
                         output.size() * sizeof(float), hipMemcpyHostToDevice,
                         stream));
    Check(hipGraphLaunch(executable, stream));
    Check(hipMemcpyAsync(output.data(), device_output,
                         output.size() * sizeof(float), hipMemcpyDeviceToHost,
                         stream));
    Check(hipStreamSynchronize(stream));
    for (unsigned row = 0; row < rows; ++row) {
      for (unsigned channel = 0; channel < hidden; ++channel) {
        double reference = 0;
        for (unsigned part = 0; part < streams; ++part)
          reference +=
              input[(std::size_t{row} * streams + part) * hidden + channel];
        reference /= streams;
        for (unsigned slot = 0; slot < taps; ++slot) {
          const float actual =
              output[(std::size_t{row} * taps + slot) * hidden + channel];
          if ((slot != tap && actual != sentinel) ||
              (slot == tap && std::abs(actual - reference) > 2e-6))
            throw std::runtime_error(
                "raw HC mean or feature-slot ownership mismatch");
        }
      }
    }
  }
  Check(hipGraphExecDestroy(executable));
  Check(hipGraphDestroy(graph));
  Check(hipFree(device_output));
  Check(hipFree(device_input));
  Check(hipStreamDestroy(stream));
}
}  // namespace

int main() {
  try {
    for (unsigned rows : {1U, 7U, 37U})
      for (unsigned hidden : {1U, 17U, 2560U})
        for (unsigned streams : {1U, 3U, 4U})
          Shape(rows, hidden, streams);
    std::puts(
        "PASS: raw HC means, partial rows/channels, five-tap layout and two "
        "graph replays across27shapes");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
