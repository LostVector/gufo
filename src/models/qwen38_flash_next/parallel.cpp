#include "src/models/qwen38_flash_next/runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <utility>

#ifdef GUFO_ENABLE_TP2_RDMA
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"
#endif

namespace gufo::models::qwen38_flash_next {

#ifdef GUFO_ENABLE_TP2_RDMA
namespace {

class ParallelRuntime final : public ext::TextParallelRuntime {
public:
  explicit ParallelRuntime(std::shared_ptr<rocm::Communicator> communicator)
      : communicator_(std::move(communicator)) {}

  std::shared_ptr<ext::TextModelRunner> CreateRunner(
      const ext::TextRunnerOptions& options, std::string* error) override {
    return CreateQwenFlashNextTextRunner(options, communicator_, error,
                                         &uses_yarn_);
  }

  std::uint32_t Rank() const noexcept override { return communicator_->rank(); }
  std::uint32_t WorldSize() const noexcept override {
    return communicator_->world_size();
  }
  bool UsesYarn() const noexcept override { return uses_yarn_; }
  std::string Describe() const override { return communicator_->Describe(); }
  bool BeginOperation(std::uint64_t index, std::string* error) override {
    return communicator_->BeginOperation(index, error);
  }
  bool EndOperation(std::uint64_t index, std::string* error) override {
    return communicator_->EndOperation(index, error);
  }

private:
  std::shared_ptr<rocm::Communicator> communicator_;
  bool uses_yarn_{false};
};

}  // namespace
#endif

std::shared_ptr<ext::TextParallelRuntime> CreateQwenFlashNextParallelRuntime(
    const ext::TextParallelTransportOptions& options, std::string* error) {
#ifdef GUFO_ENABLE_TP2_RDMA
  const rocm::IbrverbsConfig config{
      .rank = options.rank,
      .world_size = options.world_size,
      .bootstrap_host = options.bootstrap_host,
      .bootstrap_port = options.bootstrap_port,
      .device_index = options.device_index,
      .rdma_device = options.rdma_device,
      .rdma_port = options.rdma_port,
      .gid_index = options.gid_index,
  };
  auto communicator = rocm::CreateIbrverbsCommunicator(config, error);
  if (!communicator)
    return {};
  return std::make_shared<ParallelRuntime>(std::move(communicator));
#else
  (void)options;
  if (error)
    *error = "this Flash-Next extension has no TP2 RDMA support";
  return {};
#endif
}

}  // namespace gufo::models::qwen38_flash_next

#endif
