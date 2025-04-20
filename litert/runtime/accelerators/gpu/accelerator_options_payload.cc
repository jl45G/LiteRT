#include "litert/runtime/accelerators/gpu/accelerator_options_payload.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

using ::litert::ErrorStatusBuilder;

extern "C" {

struct LiteRtGpuOptionsPayloadT {
  // Increment the minor version every time a field is added.
  static constexpr const absl::string_view kIdentifier = "ml_drift_payload";

  bool enable_constant_tensor_sharing = false;
  bool enable_infinite_float_capping = false;
  bool benchmark_mode = false;
  // Added in version 1.2.0.
  bool allow_src_quantized_fc_conv_ops = false;
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
};

}  // extern "C"

namespace litert::ml_drift {
namespace {

litert::Expected<LiteRtGpuOptionsPayloadT*> GetPayload(
    LiteRtOpaqueOptions options) {
  const char* identifier = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  LITERT_RETURN_IF_ERROR(identifier == LiteRtGpuOptionsPayloadT::kIdentifier,
                         ErrorStatusBuilder::InvalidArgument())
      << "Payload stored in accelerator options is incompatible. Got "
      << identifier << ", expected " << LiteRtGpuOptionsPayloadT::kIdentifier
      << ".";

  LiteRtGpuOptionsPayloadT* payload;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsData(options, reinterpret_cast<void**>(&payload)));
  return payload;
}

}  // namespace
}  // namespace litert::ml_drift

LiteRtStatus LiteRtCreateGpuOptions(LiteRtOpaqueOptions* options) {
  auto payload = std::make_unique<LiteRtGpuOptionsPayloadT>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGpuOptionsPayloadT::kIdentifier.data(), payload.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtGpuOptionsPayloadT*>(payload);
      },
      options));
  payload.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsConstantTensorSharing(
    LiteRtOpaqueOptions gpu_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::ml_drift::GetPayload(gpu_options));
  payload->enable_constant_tensor_sharing = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsInfiniteFloatCapping(
  LiteRtOpaqueOptions gpu_options, bool enable) {
LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                        litert::ml_drift::GetPayload(gpu_options));
payload->enable_infinite_float_capping = enable;
return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsBenchmarkMode(LiteRtOpaqueOptions gpu_options,
  bool enable) {
LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
litert::ml_drift::GetPayload(gpu_options));
payload->benchmark_mode = enable;
return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    LiteRtOpaqueOptions gpu_accelerator_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(
      LiteRtGpuOptionsPayloadT * payload,
      litert::ml_drift::GetPayload(gpu_accelerator_options));
  payload->allow_src_quantized_fc_conv_ops = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegatePrecision precision) {
  LITERT_ASSIGN_OR_RETURN(
      LiteRtGpuOptionsPayloadT * payload,
      litert::ml_drift::GetPayload(gpu_accelerator_options));
  payload->precision = precision;
  return kLiteRtStatusOk;
}

const char* LiteRtGetGpuOptionsPayloadIdentifier() {
  return LiteRtGpuOptionsPayloadT::kIdentifier.data();
}

LiteRtStatus LiteRtGetGpuOptionsConstantTensorSharing(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->enable_constant_tensor_sharing;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->enable_infinite_float_capping;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsBenchmarkMode(bool* enabled,
                                              LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->benchmark_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->allow_src_quantized_fc_conv_ops;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(precision, ErrorStatusBuilder::InvalidArgument())
      << "`precision` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *precision = payload->precision;
  return kLiteRtStatusOk;
}
