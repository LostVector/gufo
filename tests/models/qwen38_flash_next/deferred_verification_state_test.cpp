
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static std::string order;
int hipStreamSynchronize(void*) {
  order += 'D';
  return 0;
}
void AssignError(std::string* e, const char* m) {
  if (e)
    *e = m;
}
void need(bool b, const char* m) {
  if (!b)
    throw std::runtime_error(m);
}
struct Session {};
struct Fixture {
  struct Options {
    std::function<void(const std::string&)> abort;
    std::function<bool(float*, const float*, std::uint32_t, std::uint32_t,
                       void*, std::string*)>
        compact_target_row;
  } options_;
  Session owner, foreign;
  struct Config {
    unsigned vocab_size = 8;
  } config_;
  struct Scratch {
    float* logits;
  } s_;
  std::array<float, 32> out, park, peer, expected;
  float* partial_verification_logits_;
  mutable const Session* partial_verification_owner_;
  mutable unsigned partial_verification_rows_ = 4,
                   partial_verification_materialized_ = 0;
  const float* selected_logits_ = nullptr;
  unsigned wide_rows_ = 0;
  void* stream_ = nullptr;
  unsigned calls = 0, aborts = 0, fail_row = 99;
  bool throw_row = false;
  Fixture(unsigned rank)
      : partial_verification_logits_(park.data()),
        partial_verification_owner_(&owner) {
    out.fill(-123456);
    expected.fill(-98765);
    park.fill(-9999);
    peer.fill(-8888);
    for (unsigned row = 0; row < 4; ++row)
      for (unsigned i = 0; i < 4; ++i) {
        park[row * 4 + i] = 100 * row + i;
        peer[row * 4 + i] = 1000 + 100 * row + i;
        expected[row * 8 + rank * 4 + i] = park[row * 4 + i];
        expected[row * 8 + (1 - rank) * 4 + i] = peer[row * 4 + i];
      }
    options_.abort = [this](const std::string&) {
      ++aborts;
      order += 'A';
    };
    options_.compact_target_row = [this, rank](float* dst, const float* src,
                                               unsigned row, unsigned total,
                                               void*, std::string* e) {
      ++calls;
      need(row < 4 && total == 4, "row tag/count mismatch");
      need(src == park.data() + row * 4, "wrong parked row/stride");
      need(dst == out.data() + row * 8, "wrong full output row/stride");
      if (row == fail_row) {
        if (e)
          *e = "injected callback failure";
        if (throw_row)
          throw std::runtime_error("injected");
        return false;
      }
      std::copy_n(src, 4, dst + rank * 4);
      std::copy_n(peer.data() + row * 4, 4, dst + (1 - rank) * 4);
      return true;
    };
    s_.logits = out.data();
  }
  const Config& config() const { return config_; }
  bool FailPartialVerification(std::string*) const;
  void ClearPartialVerification() const;
  bool EnsureFullVerificationRows(std::uint32_t, std::uint32_t, std::string*,
                                  const Session* = nullptr) const;
};
#include "methods.inc"
int main() {
  try {
    unsigned cases = 0;
    for (unsigned rank : {0U, 1U}) {
      {
        Fixture f(rank);
        std::string e;
        auto old = f.park;
        need(f.EnsureFullVerificationRows(3, 1, &e, &f.owner), "row3");
        need(f.EnsureFullVerificationRows(1, 2, &e, &f.owner), "rows1/2");
        need(f.EnsureFullVerificationRows(0, 1, &e, &f.owner), "row0");
        need(f.EnsureFullVerificationRows(0, 4, &e, &f.owner), "repeat span");
        need(f.calls == 4 && f.partial_verification_materialized_ == 15,
             "reuse mask");
        need(f.out == f.expected && f.park == old,
             "cross-row overwrite/output");
        ++cases;
      }
      for (unsigned bad : {0U, 1U, 2U}) {
        Fixture f(rank);
        std::string e;
        order.clear();
        unsigned row = bad == 0   ? 4
                       : bad == 1 ? 0
                                  : 3,
                 count = bad == 1   ? 0
                         : bad == 2 ? 2
                                    : 1;
        need(!f.EnsureFullVerificationRows(row, count, &e, &f.owner),
             "invalid range accepted");
        need(f.calls == 0 && f.partial_verification_materialized_ == 0 &&
                 f.aborts == 1 && order == "AD",
             "invalid range cleanup");
        ++cases;
      }
      for (unsigned bad : {0U, 1U, 2U, 3U, 4U}) {
        Fixture f(rank);
        std::string e;
        order.clear();
        const Session* owner = &f.owner;
        if (bad == 0)
          owner = &f.foreign;
        if (bad == 1)
          f.selected_logits_ = f.park.data();
        if (bad == 2)
          f.wide_rows_ = 3;
        if (bad == 3)
          f.options_.compact_target_row = {};
        if (bad == 4)
          f.partial_verification_logits_ = nullptr;
        need(!f.EnsureFullVerificationRows(0, 1, &e, owner),
             "invalid owner/view/callback accepted");
        need(f.calls == 0 && f.aborts == 1 && order == "AD",
             "invalid view cleanup");
        ++cases;
      }
      for (bool throws : {false, true}) {
        Fixture f(rank);
        std::string e;
        order.clear();
        f.fail_row = 1;
        f.throw_row = throws;
        bool failed = false;
        try {
          failed = !f.EnsureFullVerificationRows(0, 3, &e, &f.owner);
        } catch (const std::runtime_error&) {
          failed = throws;
        }
        need(
            failed && f.calls == 2 && f.partial_verification_materialized_ == 1,
            "failed row advanced or continued");
        need(f.aborts == 1 && order == "AD", "callback abort/drain order");
        ++cases;
      }
      {
        Fixture f(rank);
        std::string e;
        need(f.EnsureFullVerificationRows(2, 1, &e, &f.owner), "pre-reset");
        f.ClearPartialVerification();
        need(f.partial_verification_owner_ == nullptr &&
                 f.partial_verification_rows_ == 0 &&
                 f.partial_verification_materialized_ == 0,
             "reset fields");
        need(f.EnsureFullVerificationRows(99, 99, &e),
             "ordinary route changed");
        need(f.calls == 1 && f.aborts == 0, "cleared view queued work");
        ++cases;
      }
    }
    std::cout << "PASS " << cases << " source-extracted state cases\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
