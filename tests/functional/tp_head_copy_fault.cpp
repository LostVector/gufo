#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <link.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace {
std::atomic<bool> injected{false};
std::uintptr_t image_base = 0;
int image(struct dl_phdr_info* p, size_t, void*) {
  if (!p->dlpi_name || !*p->dlpi_name) {
    image_base = p->dlpi_addr;
    return 1;
  }
  return 0;
}
bool match(std::uintptr_t caller, size_t dpitch, size_t spitch, size_t width,
           size_t height, hipMemcpyKind kind) {
  if (dpitch != 993280 || spitch != 496640 || width != 496640 || height != 2 ||
      kind != hipMemcpyDeviceToDevice || injected.load())
    return false;
  const char* arm = std::getenv("STRIX_COPY_ARM");
  const char* lo = std::getenv("STRIX_COPY_BEGIN");
  const char* hi = std::getenv("STRIX_COPY_END");
  const char* mode = std::getenv("STRIX_COPY_MODE");
  if (!arm || !lo || !hi || !mode ||
      (std::strcmp(mode, "copy-error") && std::strcmp(mode, "peer-exit")) ||
      access(arm, F_OK))
    return false;
  dl_iterate_phdr(image, nullptr);
  if (!(image_base + std::strtoull(lo, nullptr, 0) <= caller &&
        caller < image_base + std::strtoull(hi, nullptr, 0)))
    return false;
  bool expected = false;
  return injected.compare_exchange_strong(expected, true);
}
void log(std::uintptr_t caller, const char* mode) {
  const char* path = std::getenv("STRIX_COPY_LOG");
  if (!path)
    return;
  int fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0)
    return;
  char line[768];
  int n = std::snprintf(
      line, sizeof(line),
      "{\"event\":\"%s\",\"pid\":%d,\"tid\":%ld,\"caller_ip\":%llu,\"image_"
      "base\":%llu,\"width\":496640,\"height\":2,\"source_pitch\":496640,"
      "\"destination_pitch\":993280,\"error\":%d}\n",
      mode, getpid(), syscall(SYS_gettid), (unsigned long long)caller,
      (unsigned long long)image_base, (int)hipErrorInvalidValue);
  if (n > 0 && n < (int)sizeof(line)) {
    const auto written = write(fd, line, n);
    (void)written;
  }
  close(fd);
}
}  // namespace
extern "C" hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch,
                                       const void* src, size_t spitch,
                                       size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream) {
  auto caller = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));
  if (match(caller, dpitch, spitch, width, height, kind)) {
    const char* mode = std::getenv("STRIX_COPY_MODE");
    log(caller, mode);
    if (!std::strcmp(mode, "peer-exit"))
      _exit(86);
    return hipErrorInvalidValue;
  }
  using Fn = hipError_t (*)(void*, size_t, const void*, size_t, size_t, size_t,
                            hipMemcpyKind, hipStream_t);
  auto next = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipMemcpy2DAsync"));
  return next ? next(dst, dpitch, src, spitch, width, height, kind, stream)
              : hipErrorNotInitialized;
}
