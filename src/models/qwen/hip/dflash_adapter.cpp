#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#if defined(ENGINE_ENABLE_HIP)
namespace gufo::hip {
namespace {
DFlashTarget TargetView(std::shared_ptr<const QwenGpuModel> model) {
  if (!model)
    return {};
  return {model->GetConfig(), model->GetWeights().token_embd,
          model->GetWeights().output, &model->GetTokenizer(), std::move(model)};
}
}  // namespace
std::shared_ptr<const QwenDFlashGpuModel> QwenDFlashGpuModel::Create(
    std::shared_ptr<const core::GgufReader> reader,
    std::shared_ptr<const QwenGpuModel> target, std::string* error) {
  return Create(std::move(reader), TargetView(std::move(target)), error);
}
std::unique_ptr<QwenDFlashGpuDraftBackend>
QwenDFlashGpuDraftBackend::CreateFromGguf(
    std::string_view path, std::shared_ptr<const QwenGpuModel> target,
    QwenDFlashGpuDraftConfig config, std::string* error) {
  return CreateFromGguf(path, TargetView(std::move(target)), config, error);
}
}  // namespace gufo::hip
#endif
