// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <utility>

#include "third_party/odml/infra/ml_drift_delegate/ml_drift_cl.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_cl_litert.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_delegate.h"
#include "litert/c/litert_accelerator.h"
#include "litert/c/litert_accelerator_registration.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/accelerator_options.h"

namespace litert::ml_drift {
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

LiteRtGpuOptionsPayload GetGpuOptionsPayload(Options& options) {
  auto opaque_options = options.GetOpaqueOptions();
  if (!opaque_options || opaque_options->Get() == nullptr) {
    return nullptr;
  }
  LiteRtGpuOptionsPayload gpu_compilation_data = nullptr;
  const auto stat = LiteRtFindOpaqueOptionsData(
      opaque_options->Get(), GpuOptions::GetPayloadIdentifier(),
      reinterpret_cast<void**>(&gpu_compilation_data));
  return (stat == kLiteRtStatusOk) ? gpu_compilation_data : nullptr;
}

}  // namespace
}  // namespace litert::ml_drift

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
                                     LiteRtOptions options, void** delegate) {
    litert::Options cc_options(options, litert::OwnHandle::kNo);
    auto* gpu_compilation_data =
        litert::ml_drift::GetGpuOptionsPayload(cc_options);

    litert::ml_drift::MlDriftClDelegateOptionsPtr gpu_delegate_options =
        litert::ml_drift::MlDriftClDelegateDefaultOptionsPtr();

    if (gpu_compilation_data != nullptr) {
      LITERT_LOG(LITERT_VERBOSE, "User provided gpu options found.");
      LiteRtGetGpuOptionsConstantTensorSharing(
          &gpu_delegate_options->enable_constant_tensors_sharing,
          gpu_compilation_data);

      LiteRtGetGpuOptionsInfiniteFloatCapping(
          &gpu_delegate_options->enable_infinite_float_capping,
          gpu_compilation_data);

      LiteRtGetGpuOptionsBenchmarkMode(
          &gpu_delegate_options->litert_benchmark_mode, gpu_compilation_data);

      LiteRtGetGpuOptionsNoImmutableExternalTensorsMode(
          &gpu_delegate_options->litert_no_immutable_external_tensors_mode,
          gpu_compilation_data);

      LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &gpu_delegate_options->allow_src_quantized_fc_conv_ops,
          gpu_compilation_data);

      LiteRtDelegatePrecision litert_delegate_precision =
          kLiteRtDelegatePrecisionDefault;

      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(
          &litert_delegate_precision, gpu_compilation_data);

      gpu_delegate_options->precision =
          ::litert::ml_drift::GetMlDriftPrecision(litert_delegate_precision);

      LiteRtDelegateBufferStorageType litert_delegate_buffer_storage_type =
          kLiteRtDelegateBufferStorageTypeDefault;

      LiteRtGetGpuAcceleratorCompilationOptionsBufferStorageType(
          &litert_delegate_buffer_storage_type, gpu_compilation_data);

      gpu_delegate_options->use_buffer_storage_type =
          litert_delegate_buffer_storage_type ==
          kLiteRtDelegateBufferStorageTypeBuffer;

      LiteRtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
          &gpu_delegate_options->prefer_texture_weights, gpu_compilation_data);

      LiteRtGetGpuAcceleratorCompilationOptionsSerializationDir(
          &gpu_delegate_options->serialization_dir, gpu_compilation_data);

      LiteRtGetGpuAcceleratorCompilationOptionsModelCacheKey(
          &gpu_delegate_options->model_token, gpu_compilation_data);

      LiteRtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
          &gpu_delegate_options->serialize_program_cache, gpu_compilation_data);

      LiteRtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &gpu_delegate_options->serialize_external_tensors,
          gpu_compilation_data);
    }

    LiteRtEnvironment env;
    LITERT_RETURN_IF_ERROR(LiteRtGetAcceleratorEnvironment(accelerator, &env));
    litert::TfLiteDelegatePtr gpu_delegate =
        litert::ml_drift::CreateMlDriftClDelegate(
            std::move(gpu_delegate_options), env);
    *delegate = gpu_delegate.release();
    if (*delegate == nullptr) {
      return kLiteRtStatusErrorRuntimeFailure;
    }

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
// TODO: Fix this function to use a pointer type, there are no references
// in C.
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
