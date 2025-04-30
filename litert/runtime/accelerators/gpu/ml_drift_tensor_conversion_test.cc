#include "litert/runtime/accelerators/gpu/ml_drift_tensor_conversion.h"

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/c/litert_any.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/test/matchers.h"
#include "tflite/kernels/test_util.h"

namespace {

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};

TEST(TensorConversionTest, CreateTensor) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan";
#endif
  // ML Drift setup.
  if (!ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }

  ml_drift::cl::Environment ml_drift_env;
  ASSERT_OK(ml_drift::cl::CreateEnvironment(&ml_drift_env));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny context_id,
      litert::ToLiteRtAny(std::any(
          reinterpret_cast<int64_t>(ml_drift_env.context().context()))));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny queue_id,
      litert::ToLiteRtAny(
          std::any(reinterpret_cast<int64_t>(ml_drift_env.queue()->queue()))));

  // Create litert Environment using ml_drift environment. This ensures that we
  // are using the same OpenCL device, context and command queue.
  const std::array<LiteRtEnvOption, 2> environment_options = {
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClContext,
          /*.value=*/context_id,
      },
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClCommandQueue,
          /*.value=*/queue_id,
      },
  };
  LiteRtGpuGlobalEnvironmentCreate(environment_options.size(),
                                   environment_options.data());

  // Create litert::TensorBuffer.
  const litert::RankedTensorType kTensorType(::kTensorType);
  size_t bytes = sizeof(kTensorData);
  auto tensor_buffer_expected = litert::TensorBuffer::CreateManaged(
      kLiteRtTensorBufferTypeOpenClBuffer, kTensorType, bytes);
  ASSERT_TRUE(tensor_buffer_expected);

  // Convert litert::TensorBuffer to ml_drift::cl::Tensor.
  ASSERT_OK_AND_ASSIGN(auto tensor, ml_drift::ConvertToTensor(
                                        *tensor_buffer_expected, ml_drift_env));

  // Test upload and download of tensor data.

  // Fill descriptor with data.
  ml_drift::TensorDescriptor descriptor_with_data = tensor.GetDescriptor();
  descriptor_with_data.UploadData<float>(&kTensorData[0]);
  // Upload descriptor with data to Open CL memory.
  ASSERT_OK(
      tensor.UploadDescriptorData(descriptor_with_data, ml_drift_env.queue()));

  // Create new tensor with same shape as the original tensor.
  ml_drift::TensorFloat32 new_tensor;
  const ml_drift::BHWC shape = ml_drift::BHWC(
      tensor.Batch(), tensor.Height(), tensor.Width(), tensor.Channels());
  new_tensor.shape = shape;
  new_tensor.data.resize(new_tensor.shape.DimensionsProduct());

  ml_drift::TensorDescriptor new_descriptor;
  // Downloads from Open CL memory to descriptor.
  ASSERT_OK(tensor.ToDescriptor(&new_descriptor, ml_drift_env.queue()));
  // Downloads from descriptor to tensor.
  new_descriptor.DownloadData(&new_tensor);
  EXPECT_THAT(new_tensor.data,
              testing::Pointwise(tflite::FloatingPointEq(), kTensorData));
}

}  // namespace
