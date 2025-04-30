#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ML_DRIFT_TENSOR_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ML_DRIFT_TENSOR_CONVERSION_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "litert/cc/litert_tensor_buffer.h"

namespace ml_drift {

// Converts a LiteRT TensorBuffer to an ML Drift Tensor.
absl::StatusOr<ml_drift::cl::Tensor> ConvertToTensor(
    const litert::TensorBuffer& tensor_buffer,
    const ml_drift::cl::Environment& env);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ML_DRIFT_TENSOR_CONVERSION_H_
