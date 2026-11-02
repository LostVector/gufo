#include "src/cli/serve/tp_disk_store.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/fs.hpp"

namespace gufo::server {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'G', 'U', 'F', 'O',
                                                'T', 'P', '2', 'S'};
constexpr std::uint32_t kFormat = 1;
// magic, format, payload version, key, payload bytes, identity and payload
// SHA-256.
constexpr std::size_t kHeaderBytes = 8 + 4 + 4 + 8 + 8 + 32 + 32;

using Digest = std::array<std::uint8_t, 32>;

class Fd {
public:
  explicit Fd(platform::PlatformFile fd = platform::kInvalidFile) : fd_(fd) {}
  ~Fd() {
    if (platform::FileValid(fd_))
      platform::CloseFile(fd_);
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  [[nodiscard]] platform::PlatformFile get() const { return fd_; }

private:
  platform::PlatformFile fd_;
};

void Put(std::uint8_t* out, std::uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i)
    out[i] = static_cast<std::uint8_t>(value >> (8 * i));
}

std::uint64_t Get(const std::uint8_t* in, int bytes) {
  std::uint64_t value = 0;
  for (int i = 0; i < bytes; ++i)
    value |= std::uint64_t{in[i]} << (8 * i);
  return value;
}

std::string Name(std::uint64_t key, const char* suffix) {
  char name[40];
  std::snprintf(name, sizeof(name), "tp2-%016llx%s",
                static_cast<unsigned long long>(key), suffix);
  return name;
}

bool WriteAll(platform::PlatformFile fd, const std::uint8_t* data,
              std::size_t bytes) {
  while (bytes > 0) {
    std::error_code ec;
    const auto written = platform::Write(fd, data, bytes, ec);
    if (ec == std::errc::interrupted)
      continue;
    if (ec || written == 0)
      return false;
    data += written;
    bytes -= static_cast<std::size_t>(written);
  }
  return true;
}

bool ReadAll(platform::PlatformFile fd, std::uint8_t* data, std::size_t bytes) {
  while (bytes > 0) {
    std::error_code ec;
    const auto got = platform::Read(fd, data, bytes, ec);
    if (ec == std::errc::interrupted)
      continue;
    if (ec || got == 0)
      return false;
    data += got;
    bytes -= static_cast<std::size_t>(got);
  }
  return true;
}

}  // namespace

struct TpDiskStore::Impl {
  struct Job {
    std::uint64_t key;
    std::shared_ptr<const TextModelRunner> runner;
    std::shared_ptr<const TextRunnerSnapshot> snapshot;
    std::size_t bytes;
  };
  struct Entry {
    std::size_t bytes{0};
    std::uint64_t used{0};
  };

  Impl(Options options, std::vector<std::uint8_t> identity,
       std::uint32_t payload_version)
      : options(std::move(options)),
        identity_digest(crypto::Sha256Hasher{}.Finish()),
        payload_version(payload_version) {
    crypto::Sha256Hasher hasher;
    hasher.Update(identity);
    identity_digest = hasher.Finish();
    if (this->options.directory.empty() || this->options.capacity_bytes == 0)
      throw std::invalid_argument("TP disk cache needs a directory and size");
    std::filesystem::create_directories(this->options.directory);
    std::filesystem::permissions(this->options.directory,
                                 std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace);
    std::error_code ec;
    if (!directory.Open(this->options.directory, ec))
      throw std::system_error(ec, "TP disk cache directory");
    Index();
    worker = std::thread([this] { Work(); });
  }

  ~Impl() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    changed.notify_all();
    worker.join();
  }

  /// Indexes the entries a previous run left, oldest use first, and removes
  /// unfinished writes.
  void Index() {
    std::vector<std::pair<std::filesystem::file_time_type, std::uint64_t>>
        found;
    for (const auto& file :
         std::filesystem::directory_iterator(options.directory)) {
      const auto name = file.path().filename().string();
      unsigned long long key = 0;
      char suffix[8] = {};
      if (std::sscanf(name.c_str(), "tp2-%16llx.%5s", &key, suffix) != 2 ||
          !file.is_regular_file())
        continue;
      if (std::strcmp(suffix, "tmp") == 0) {
        std::error_code ignored;
        (void)platform::Unlink(directory, name, ignored);
      } else if (std::strcmp(suffix, "snap") == 0 && key != 0) {
        entries[key].bytes = file.file_size();
        retained += entries[key].bytes;
        found.emplace_back(file.last_write_time(), key);
      }
    }
    std::ranges::sort(found);
    for (const auto& [time, key] : found)
      entries[key].used = ++clock;
    Evict();
  }

  /// Under `mutex`: removes the least recently used entries beyond capacity.
  void Evict() {
    while (retained > options.capacity_bytes && !entries.empty()) {
      const auto oldest = std::ranges::min_element(
          entries, {}, [](const auto& entry) { return entry.second.used; });
      Remove(oldest->first);
    }
  }

  /// Under `mutex`.
  void Remove(std::uint64_t key) {
    const auto found = entries.find(key);
    if (found == entries.end())
      return;
    retained -= found->second.bytes;
    entries.erase(found);
    std::error_code ignored;
    (void)platform::Unlink(directory, Name(key, ".snap"), ignored);
  }

  void Work() {
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex);
      changed.wait(lock, [this] { return stopping || !jobs.empty(); });
      if (jobs.empty())
        return;  // stopping, and every queued write is done
      Job job = std::move(jobs.front());
      jobs.pop_front();
      writing = true;
      lock.unlock();
      const bool written = Write(job);
      job.snapshot.reset();
      lock.lock();
      writing = false;
      queued -= job.bytes;
      if (written) {
        // The rename replaced any earlier file under this key: forget its
        // bytes without unlinking the new one.
        if (const auto previous = entries.find(job.key);
            previous != entries.end())
          retained -= previous->second.bytes;
        entries[job.key] = {.bytes = kHeaderBytes + job.bytes, .used = ++clock};
        retained += kHeaderBytes + job.bytes;
        Evict();
      }
      changed.notify_all();
    }
  }

  /// Writes a temporary file and publishes it under its key. Runs without
  /// `mutex`; a failure leaves no file.
  bool Write(const Job& job) {
    const auto temporary = Name(job.key, ".tmp");
    std::error_code ec;
    Fd fd(platform::OpenAt(directory, temporary,
                           platform::FileMode::kOpenOrCreate, ec));
    if (!platform::FileValid(fd.get()))
      return false;
    if (!platform::Truncate(fd.get(), 0, ec)) {
      (void)platform::Unlink(directory, temporary, ec);
      return false;
    }
    std::array<std::uint8_t, kHeaderBytes> header{};
    bool valid = WriteAll(fd.get(), header.data(), header.size());
    crypto::Sha256Hasher hasher;
    std::size_t streamed = 0;
    try {
      job.runner->StreamPersistentSnapshot(
          *job.snapshot, [&](std::span<const std::uint8_t> bytes) {
            if (!valid)
              return;
            hasher.Update(bytes);
            streamed += bytes.size();
            valid = WriteAll(fd.get(), bytes.data(), bytes.size());
          });
    } catch (...) {
      valid = false;
    }
    valid = valid && streamed == job.bytes;
    if (valid) {
      std::copy(kMagic.begin(), kMagic.end(), header.begin());
      Put(header.data() + 8, kFormat, 4);
      Put(header.data() + 12, payload_version, 4);
      Put(header.data() + 16, job.key, 8);
      Put(header.data() + 24, streamed, 8);
      std::copy(identity_digest.begin(), identity_digest.end(),
                header.begin() + 32);
      const Digest payload = hasher.Finish();
      std::copy(payload.begin(), payload.end(), header.begin() + 64);
      valid = platform::SeekTo(fd.get(), 0, ec) &&
              WriteAll(fd.get(), header.data(), header.size()) &&
              platform::SyncFile(fd.get(), ec);
      // The image is durable now. Drop its clean pages, as rank 0's disk cache
      // does, so multi-GB halves don't accumulate in the page cache.
      if (valid)
        platform::Advise(fd.get(), platform::AdviseHint::kDontNeed, 0, 0);
      valid = valid && platform::Rename(directory, temporary, directory,
                                        Name(job.key, ".snap"), ec);
      valid = valid && platform::SyncDirectory(directory, ec);
    }
    if (!valid)
      (void)platform::Unlink(directory, temporary, ec);
    return valid;
  }

  void Restore(std::uint64_t key, const TextModelRunner& runner,
               TextRunnerState& state) {
    const auto name = Name(key, ".snap");
    std::vector<std::uint8_t> file;
    {
      const std::lock_guard<std::mutex> lock(mutex);
      if (!entries.contains(key))
        throw std::runtime_error("TP disk cache has no entry for this key");
      std::error_code ec;
      Fd fd(platform::OpenAt(directory, name, platform::FileMode::kRead, ec));
      platform::FileInfo info{};
      if (!platform::FileValid(fd.get()) ||
          !platform::Stat(fd.get(), &info, ec) || !info.regular ||
          info.size < kHeaderBytes) {
        Remove(key);
        throw std::runtime_error("TP disk cache entry is unreadable");
      }
      file.resize(static_cast<std::size_t>(info.size));
      if (!ReadAll(fd.get(), file.data(), file.size())) {
        Remove(key);
        throw std::runtime_error("TP disk cache entry read failed");
      }
      // Restore uses the in-memory copy; drop the page-cache one.
      platform::Advise(fd.get(), platform::AdviseHint::kDontNeed, 0, 0);
      entries[key].used = ++clock;
    }
    const auto payload =
        std::span<const std::uint8_t>(file).subspan(kHeaderBytes);
    crypto::Sha256Hasher hasher;
    hasher.Update(payload);
    const Digest payload_digest = hasher.Finish();
    const bool valid = std::equal(kMagic.begin(), kMagic.end(), file.begin()) &&
                       Get(file.data() + 8, 4) == kFormat &&
                       Get(file.data() + 12, 4) == payload_version &&
                       Get(file.data() + 16, 8) == key &&
                       Get(file.data() + 24, 8) == payload.size() &&
                       std::equal(identity_digest.begin(),
                                  identity_digest.end(), file.begin() + 32) &&
                       std::equal(payload_digest.begin(), payload_digest.end(),
                                  file.begin() + 64);
    if (!valid) {
      const std::lock_guard<std::mutex> lock(mutex);
      Remove(key);
      throw std::runtime_error("TP disk cache entry is corrupt or foreign");
    }
    (void)platform::TouchAt(directory, name);
    runner.RestorePersistentSnapshot(state, payload);
  }

  Options options;
  Digest identity_digest;
  std::uint32_t payload_version;
  platform::Directory directory;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::deque<Job> jobs;
  std::map<std::uint64_t, Entry> entries;
  std::size_t retained{0};
  std::size_t queued{0};
  std::uint64_t clock{0};
  bool writing{false};
  bool stopping{false};
  std::thread worker;
};

TpDiskStore::TpDiskStore(Options options, std::vector<std::uint8_t> identity,
                         std::uint32_t payload_version)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(identity),
                                   payload_version)) {}

TpDiskStore::~TpDiskStore() = default;

bool TpDiskStore::PersistAsync(
    std::uint64_t key, std::shared_ptr<const TextModelRunner> runner,
    std::shared_ptr<const TextRunnerSnapshot> snapshot) {
  if (key == 0 || runner == nullptr || snapshot == nullptr)
    return false;
  const auto bytes = runner->PersistentSnapshotPayloadBytes(*snapshot);
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping ||
        bytes + kHeaderBytes > impl_->options.capacity_bytes ||
        bytes > impl_->options.staging_capacity_bytes - impl_->queued)
      return false;
    impl_->queued += bytes;
    impl_->jobs.push_back({.key = key,
                           .runner = std::move(runner),
                           .snapshot = std::move(snapshot),
                           .bytes = bytes});
  }
  impl_->changed.notify_all();
  return true;
}

void TpDiskStore::Restore(std::uint64_t key, const TextModelRunner& runner,
                          TextRunnerState& state) {
  impl_->Restore(key, runner, state);
}

void TpDiskStore::Flush() {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  impl_->changed.wait(
      lock, [this] { return impl_->jobs.empty() && !impl_->writing; });
}

std::size_t TpDiskStore::entry_count() const {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->entries.size();
}

std::size_t TpDiskStore::retained_bytes() const {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->retained;
}

}  // namespace gufo::server
