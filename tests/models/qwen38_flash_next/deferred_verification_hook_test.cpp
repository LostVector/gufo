
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
using __half = std::uint16_t;
constexpr int hipMemcpyDeviceToDevice = 1;
static std::string events;
static bool prefill_phase = true;
int hipStreamSynchronize(void*) {
  events += 'D';
  return 0;
}
int hipMalloc(void** p, std::size_t n) {
  *p = std::malloc(n);
  return *p ? 0 : 1;
}
int hipMemcpyAsync(void* d, const void* s, std::size_t n, int, void*) {
  std::memcpy(d, s, n);
  return 0;
}
void AssignError(std::string* e, const char* m) {
  if (e)
    *e = m;
}
void need(bool v, const char* m) {
  if (!v)
    throw std::runtime_error(m);
}
struct PrefillPhase {
  bool old;
  PrefillPhase(bool p) : old(prefill_phase) { prefill_phase = p; }
  ~PrefillPhase() { prefill_phase = old; }
};
unsigned HcInjectParts(unsigned) {
  return 1;
}
std::size_t Q8TiledBytes(std::size_t n, unsigned d) {
  return n * d * 4;
}
struct Session {};
struct Fixture {
  static constexpr unsigned kHookParkedRows = 4;
  struct Config {
    unsigned hidden_size = 4, hc_count = 2;
    unsigned HcDim() const { return 8; }
  } cfg;
  struct Scratch {
    void *res, *inject, *xn_half, *xn_q8t, *xn, *block_out, *hc_defer_w;
  } s_;
  std::array<std::array<unsigned char, 256>, 7> buffers;
  void* inject_fused_ = nullptr;
  bool xn_half_ = false, moe_pending_ = true, defer_residual_ = true,
       res_deferred_ = true, shexp_half_ready_ = true;
  unsigned inject_parts_ = 3, q8_slot_ = 4, xn_q8_ready_ = 5,
           mixed_q8_ready_ = 6, pending_combine_ = 7;
  unsigned routed_max_rows_ = 8, routed_n_tiles_ = 9, routed_wide_tiles_ = 10,
           routed_tile_rows_ = 11;
  int routed_tile_cols_ = 12;
  const float* selected_logits_ = reinterpret_cast<float*>(0x1000);
  unsigned wide_rows_ = 0;
  unsigned tokens[2] = {7, 8};
  std::span<const std::int32_t> prefetch_next_;
  bool prefetch_deferred_ = false;
  void* hook_park_ = nullptr;
  void* stream_ = nullptr;
  std::function<void()> hook_;
  unsigned hooks = 0, aborts = 0, prefetches = 0, yarns = 0;
  Session outer, child;
  const Session* partial_verification_owner_ = nullptr;
  unsigned partial_verification_rows_ = 0,
           partial_verification_materialized_ = 0;
  struct Options {
    std::function<void(const std::string&)> abort;
  } options_;
  Fixture() {
    for (auto& b : buffers)
      b.fill(0x33);
    s_ = {buffers[0].data(), buffers[1].data(), buffers[2].data(),
          buffers[3].data(), buffers[4].data(), buffers[5].data(),
          buffers[6].data()};
    prefetch_next_ = {reinterpret_cast<const std::int32_t*>(tokens), 2};
    options_.abort = [this](const std::string&) {
      ++aborts;
      events += 'A';
    };
  }
  ~Fixture() { std::free(hook_park_); }
  const Config& config() const { return cfg; }
  bool Check(int r, const char*, std::string*) const { return r == 0; }
  void UseScratch(Scratch s) { s_ = s; }
  void UseYarn(const Session& s) {
    need(&s == &outer, "wrong resumed session");
    ++yarns;
  }
  void PrefetchPle(const Session& s, std::span<const std::int32_t> next) {
    need(&s == &outer && next.size() == 2 && next[0] == 7 && next[1] == 8,
         "wrong resumed prefetch");
    ++prefetches;
  }
  bool RunLayerHook(Session&, std::string*);
  bool FailPartialVerification(std::string*);
  void ClearPartialVerification();
};
// Production method is const because its scratch/bookkeeping fields are
// mutable. Keep fake field declarations concise by applying const to Fixture
// through a const forwarding shim: all extracted definitions below use nonconst
// Fixture.

#include "methods.inc"
int main() {
  try {
    unsigned cases = 0;
    for (bool half : {false, true})
      for (bool deferred : {false, true})
        for (bool prefetch : {false, true}) {
          Fixture f;
          std::string error;
          prefill_phase = true;
          f.xn_half_ = half;
          f.res_deferred_ = deferred;
          f.prefetch_deferred_ = prefetch;
          f.hook_ = [&] {
            ++f.hooks;
            need(!prefill_phase, "nested decode ran as prefill");
            for (auto& b : f.buffers)
              b.fill(0xa5);
            f.partial_verification_owner_ = &f.child;
            f.partial_verification_rows_ = 3;
            f.partial_verification_materialized_ = 1;
            f.selected_logits_ = reinterpret_cast<float*>(0x2000);
            f.wide_rows_ = 7;
            f.inject_parts_ = 99;
            f.routed_n_tiles_ = 99;
          };
          need(f.RunLayerHook(f.outer, &error), "hook failed");
          need(f.partial_verification_owner_ == nullptr &&
                   f.partial_verification_rows_ == 0 &&
                   f.partial_verification_materialized_ == 0,
               "nested metadata leaked into parent");
          need(f.selected_logits_ == reinterpret_cast<float*>(0x1000) &&
                   f.wide_rows_ == 0 && f.inject_parts_ == 3 &&
                   f.routed_n_tiles_ == 9,
               "original dispatch not restored");
          need(prefill_phase && f.hooks == 1 && f.aborts == 0 && f.yarns == 1 &&
                   f.prefetches == unsigned(prefetch),
               "phase/hook/prefetch mismatch");
          auto restored = [&](unsigned b, unsigned n) {
            for (unsigned i = 0; i < n; ++i)
              need(f.buffers[b][i] == 0x33, "parked bytes not restored");
          };
          restored(0, 128);
          restored(1, 32);
          if (half) {
            restored(2, 64);
            restored(3, 128);
          } else
            restored(4, 128);
          if (deferred) {
            restored(5, 64);
            restored(6, 32);
          }
          ++cases;
        }
    {
      Fixture f;
      std::string error;
      events.clear();
      f.partial_verification_owner_ = &f.outer;
      f.partial_verification_rows_ = 3;
      f.hook_ = [&] { ++f.hooks; };
      need(!f.RunLayerHook(f.outer, &error), "live deferred row yielded");
      need(f.hooks == 0 && f.aborts == 1 && events == "AD",
           "live guard did not abort before drain");
      ++cases;
    }
    std::cout << "PASS " << cases
              << " actual RunLayerHook host-state cases (GPU APIs simulated)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
