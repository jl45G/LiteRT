#include "litert/runtime/accelerators/gpu/ml_drift_tensor_conversion.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace ml_drift {

ml_drift::DataType ConvertFrom(litert::ElementType element_type) {
  switch (element_type) {
    case litert::ElementType::None:
      return ml_drift::DataType::UNKNOWN;
    case litert::ElementType::Bool:
      return ml_drift::DataType::BOOL;
    case litert::ElementType::Int4:
      return ml_drift::DataType::INT4;
    case litert::ElementType::Int8:
      return ml_drift::DataType::INT8;
    case litert::ElementType::Int16:
      return ml_drift::DataType::INT16;
    case litert::ElementType::Int32:
      return ml_drift::DataType::INT32;
    case litert::ElementType::Int64:
      return ml_drift::DataType::INT64;
    case litert::ElementType::UInt8:
      return ml_drift::DataType::UINT8;
    case litert::ElementType::UInt16:
      return ml_drift::DataType::UINT16;
    case litert::ElementType::UInt32:
      return ml_drift::DataType::UINT32;
    case litert::ElementType::UInt64:
      return ml_drift::DataType::UINT64;
    case litert::ElementType::Float16:
      return ml_drift::DataType::FLOAT16;
    case litert::ElementType::BFloat16:
      return ml_drift::DataType::BFLOAT16;
    case litert::ElementType::Float32:
      return ml_drift::DataType::FLOAT32;
    case litert::ElementType::Float64:
      return ml_drift::DataType::FLOAT64;
    // Cannot be converted to ML Drift data type.
    case litert::ElementType::Complex64:
    case litert::ElementType::Complex128:
    case litert::ElementType::TfResource:
    case litert::ElementType::TfString:
    case litert::ElementType::TfVariant:
      return ml_drift::DataType::UNKNOWN;
      break;
  }
}

absl::StatusOr<ml_drift::cl::Tensor> ConvertToTensor(
    const litert::TensorBuffer& tensor_buffer,
    const ml_drift::cl::Environment& env) {
  auto tensor_buffer_type_expected = tensor_buffer.BufferType();
  if (!tensor_buffer_type_expected) {
    return absl::InvalidArgumentError("Failed to get tensor buffer type.");
  }
  if (*tensor_buffer_type_expected != kLiteRtTensorBufferTypeOpenClBuffer) {
    return absl::InvalidArgumentError(
        "Invalid tensor buffer type, expected OpenCL buffer.");
  }

  auto cl_mem_expected = tensor_buffer.GetOpenClMemory();
  if (!cl_mem_expected) {
    return absl::InvalidArgumentError("Failed to get OpenCL memory.");
  }

  auto ranked_tensor_type_expected = tensor_buffer.TensorType();
  if (!ranked_tensor_type_expected) {
    return absl::InvalidArgumentError("Failed to get tensor type.");
  }
  ml_drift::DataType data_type =
      ConvertFrom(ranked_tensor_type_expected->ElementType());
  if (data_type == ml_drift::DataType::UNKNOWN) {
    return absl::InvalidArgumentError("Unsupported ml_drift data type.");
  }

  auto storage_type =
      ml_drift::cl::GetFastestStorageType(env.device().GetInfo());
  // TODO: b/391346692 - Expand litert::Layout to be compatible with ML Drift.
  // We assume that the tensor is BHWC for now.
  ml_drift::Layout layout = ml_drift::Layout::BHWC;

  auto c = ranked_tensor_type_expected->Layout().Dimensions()[0];

  auto tensor_descriptor =
      ml_drift::TensorDescriptor(data_type, storage_type, layout);
  tensor_descriptor.SetBHWCShape(BHWC(1, 1, 1, c));
  ml_drift::cl::Tensor tensor;
  RETURN_IF_ERROR(ml_drift::cl::CreateTensorShared(
      env.context(), *cl_mem_expected, tensor_descriptor, &tensor));
  return tensor;
}

}  // namespace ml_drift
