#include <memory>
#include <utility>

#include "third_party/odml/infra/ml_drift_delegate/ml_drift_cl.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_cl_litert.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_delegate.h"
#include "litert/c/litert_accelerator.h"
#include "litert/c/litert_accelerator_registration.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/accelerators/gpu/accelerator_options.h"
#include "litert/runtime/accelerators/gpu/accelerator_options_payload.h"

namespace {
MlDriftDelegatePrecision GetMlDriftPrecision(
    LiteRtDelegatePrecision precision) {
  switch (precision) {
    case kLiteRtDelegatePrecisionDefault:
      return kDefault;
    case kLiteRtDelegatePrecisionFp16:
      return kFp16;
    case kLiteRtDelegatePrecisionFp32:
      return kFp32;
  }
}
}  // namespace

// Accelerator implementation for the LiteRT GPU OpenCL accelerator.
class GpuOpenClAccelerator {
 public:
  static std::unique_ptr<GpuOpenClAccelerator> Create() {
    auto accelerator = std::make_unique<GpuOpenClAccelerator>();
    accelerator->hardware_support_ = kLiteRtHwAcceleratorGpu;
    return accelerator;
  }

  static void Destroy(void* accelerator) {
    GpuOpenClAccelerator* instance =
        reinterpret_cast<GpuOpenClAccelerator*>(accelerator);
    delete instance;
  }

  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    static const char* lrt_name = "GPU OpenCL";
    *name = lrt_name;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    static constexpr const LiteRtApiVersion lrt_version = {
        /*major=*/1,
        /*minor=*/0,
        /*patch=*/0,
    };
    *version = lrt_version;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetHardwareSupport(
      LiteRtAccelerator accelerator,
      LiteRtHwAcceleratorSet* supported_hardware) {
    static LiteRtHwAcceleratorSet hardware_support = kLiteRtHwAcceleratorGpu;
    *supported_hardware = hardware_support;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus IsTfLiteDelegateResponsibleForJitCompilation(
      LiteRtAcceleratorT* accelerator, bool* does_jit_compilation) {
    LITERT_RETURN_IF_ERROR(does_jit_compilation,
                           litert::ErrorStatusBuilder::InvalidArgument())
        << "`does_jit_compilation` pointer is null.";
    *does_jit_compilation = true;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus CreateDelegate(LiteRtAccelerator accelerator,
                                     LiteRtOpaqueOptions options,
                                     void** delegate) {
    LiteRtGpuOptionsPayload gpu_compilation_data = nullptr;
    LiteRtStatus options_data_status = LiteRtFindOpaqueOptionsData(
        options, litert::ml_drift::GpuOptions::GetPayloadIdentifier(),
        reinterpret_cast<void**>(&gpu_compilation_data));
    switch (options_data_status) {
      case kLiteRtStatusOk:
        break;
      case kLiteRtStatusErrorNotFound:
        gpu_compilation_data = nullptr;
        break;
      default:
        return options_data_status;
    }

    litert::ml_drift::MlDriftClDelegateOptionsPtr gpu_delegate_options =
        litert::ml_drift::MlDriftClDelegateDefaultOptionsPtr();

    if (gpu_compilation_data != nullptr) {
      LiteRtGetGpuOptionsConstantTensorSharing(
          &gpu_delegate_options->enable_constant_tensors_sharing,
          gpu_compilation_data);
      LiteRtGetGpuOptionsInfiniteFloatCapping(
          &gpu_delegate_options->enable_infinite_float_capping,
          gpu_compilation_data);
      LiteRtGetGpuOptionsBenchmarkMode(
          &gpu_delegate_options->litert_benchmark_mode, gpu_compilation_data);
          LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
            &gpu_delegate_options->allow_src_quantized_fc_conv_ops,
            gpu_compilation_data);
        LiteRtDelegatePrecision litert_delegate_precision =
            kLiteRtDelegatePrecisionDefault;
        LiteRtGetGpuAcceleratorCompilationOptionsPrecision(
            &litert_delegate_precision, gpu_compilation_data);
        gpu_delegate_options->precision =
            GetMlDriftPrecision(litert_delegate_precision);
    }

    litert::TfLiteDelegatePtr gpu_delegate =
        litert::ml_drift::CreateMlDriftClDelegate(
            std::move(gpu_delegate_options));
    *delegate = gpu_delegate.release();
    return kLiteRtStatusOk;
  }

  static void DestroyDelegate(void* delegate) {
    LiteRtDeleteMlDriftClDelegate(reinterpret_cast<TfLiteDelegate*>(delegate));
  }

  LiteRtHwAcceleratorSet hardware_support_;
};

// Discovery C function for the GPU OpenCL accelerator by LiteRT.
// This function is called by the LiteRT environment constructor and the
// function name is looked up by dlsym().
extern "C" LiteRtStatus LiteRtRegisterAcceleratorGpuOpenCl(
    LiteRtEnvironment env) {
  auto gpu_opencl_accelerator = GpuOpenClAccelerator::Create();
  LiteRtAccelerator accelerator;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetName(accelerator, GpuOpenClAccelerator::GetName));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetVersion(
      accelerator, GpuOpenClAccelerator::GetVersion));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetHardwareSupport(
      accelerator, GpuOpenClAccelerator::GetHardwareSupport));
  LITERT_RETURN_IF_ERROR(LiteRtSetDelegateFunction(
      accelerator, GpuOpenClAccelerator::CreateDelegate,
      GpuOpenClAccelerator::DestroyDelegate));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation(
          accelerator,
          GpuOpenClAccelerator::IsTfLiteDelegateResponsibleForJitCompilation));
  LITERT_RETURN_IF_ERROR(LiteRtRegisterAccelerator(
      env, accelerator, gpu_opencl_accelerator.release(),
      GpuOpenClAccelerator::Destroy));
  return kLiteRtStatusOk;
}

#if defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)

// Function pointer defined in auto_registration.cc.
extern "C" LiteRtStatus (*LiteRtRegisterStaticLinkedAcceleratorGpu)(
    LiteRtEnvironmentT& environment);

namespace {

class StaticGpuAcceleratorInitializer {
 public:
  StaticGpuAcceleratorInitializer() {
    LiteRtRegisterStaticLinkedAcceleratorGpu =
        [](LiteRtEnvironmentT& environment) -> LiteRtStatus {
      return LiteRtRegisterAcceleratorGpuOpenCl(&environment);
    };
  }
};

StaticGpuAcceleratorInitializer g_initializer;

}  // namespace

#endif  // defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
