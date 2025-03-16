#include "litert/kotlin/src/main/jni/litert_compiled_model_jni.h"

#include <jni.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/kotlin/src/main/jni/litert_jni_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_compilation_options.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_model.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"  // from @org_tensorflow

namespace {

// Creates a CompiledModel from the given handles.
// The handles are not owned by the returned CompiledModel.
litert::CompiledModel CreateCompileModel(jlong compiled_model_handle,
                                         jlong model_handle) {
  auto c_model = reinterpret_cast<LiteRtModel>(model_handle);
  ABSL_CHECK(c_model != nullptr);
  auto c_compiled_model =
      reinterpret_cast<LiteRtCompiledModel>(compiled_model_handle);
  ABSL_CHECK(c_compiled_model != nullptr);
  return litert::CompiledModel(c_model, c_compiled_model, false);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreate(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong env_handle,
                                                          jlong model_handle,
                                                          jintArray options) {
  int options_size = env->GetArrayLength(options);
  auto options_array = env->GetIntArrayElements(options, nullptr);
  LiteRtHwAcceleratorSet accelerators = kLiteRtHwAcceleratorNone;
  for (int i = 0; i < options_size; ++i) {
    switch (options_array[i]) {
      case litert::jni::kAccelatorNone:
        break;
      case litert::jni::kAccelatorCpu:
        accelerators |= kLiteRtHwAcceleratorCpu;
        break;
      case litert::jni::kAccelatorGpu:
        accelerators |= kLiteRtHwAcceleratorGpu;
        break;
      case litert::jni::kAccelatorNpu:
        accelerators |= kLiteRtHwAcceleratorNpu;
        break;
      default:
        LITERT_LOG(LITERT_ERROR, "Unsupported accelerator: %d.",
                   options_array[i]);
    }
  }
  env->ReleaseIntArrayElements(options, options_array, 0);

  LiteRtCompilationOptions compilation_options = nullptr;
  auto status = LiteRtCreateCompilationOptions(&compilation_options);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compilation options.");
    return 0;
  }
  status = LiteRtSetCompilationOptionsHardwareAccelerators(compilation_options,
                                                           accelerators);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to set hardware accelerators.");
    return 0;
  }

  auto litert_env = reinterpret_cast<LiteRtEnvironment>(env_handle);
  ABSL_CHECK(litert_env != nullptr);
  auto model = reinterpret_cast<LiteRtModel>(model_handle);
  ABSL_CHECK(model != nullptr);
  LiteRtCompiledModel compiled_model = nullptr;
  status = LiteRtCreateCompiledModel(
      litert_env, model, std::move(compilation_options), &compiled_model);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compiled model.");
    return 0;
  }
  return reinterpret_cast<jlong>(compiled_model);
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring input_name) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto ss = env->GetStringUTFChars(signature, nullptr);
  auto signature_str =
      absl::string_view(ss, env->GetStringUTFLength(signature));
  auto ins = env->GetStringUTFChars(input_name, nullptr);
  auto input_name_str =
      absl::string_view(ins, env->GetStringUTFLength(input_name));
  auto tensor_buffer =
      compiled_model.CreateInputBuffer(signature_str, input_name_str);
  env->ReleaseStringUTFChars(signature, ss);
  env->ReleaseStringUTFChars(input_name, ins);
  if (!tensor_buffer) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input buffer.");
    return 0;
  }
  return reinterpret_cast<jlong>(std::move(tensor_buffer->Release()));
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring output_name) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto ss = env->GetStringUTFChars(signature, nullptr);
  auto signature_str =
      absl::string_view(ss, env->GetStringUTFLength(signature));
  auto ons = env->GetStringUTFChars(output_name, nullptr);
  auto output_name_str =
      absl::string_view(ons, env->GetStringUTFLength(output_name));
  auto tensor_buffer =
      compiled_model.CreateOutputBuffer(signature_str, output_name_str);
  env->ReleaseStringUTFChars(signature, ss);
  env->ReleaseStringUTFChars(output_name, ons);
  if (!tensor_buffer) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output buffer.");
    return 0;
  }
  return reinterpret_cast<jlong>(std::move(tensor_buffer->Release()));
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto tensor_buffers = compiled_model.CreateInputBuffers(signature_index);
  if (!tensor_buffers) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input buffers.");
    return nullptr;
  }
  std::vector<jlong> input_tensor_buffers;
  input_tensor_buffers.reserve(tensor_buffers->size());
  for (int i = 0; i < tensor_buffers->size(); ++i) {
    input_tensor_buffers.push_back(
        reinterpret_cast<jlong>(std::move(tensor_buffers->at(i).Release())));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          input_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto tensor_buffers = compiled_model.CreateOutputBuffers(signature_index);
  if (!tensor_buffers) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output buffers.");
    return nullptr;
  }
  std::vector<jlong> output_tensor_buffers;
  output_tensor_buffers.reserve(tensor_buffers->size());
  for (int i = 0; i < tensor_buffers->size(); ++i) {
    output_tensor_buffers.push_back(
        reinterpret_cast<jlong>(std::move(tensor_buffers->at(i).Release())));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          output_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_CompiledModel_nativeRun(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index, jlongArray input_buffers, jlongArray output_buffers) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto num_inputs = env->GetArrayLength(input_buffers);
  auto inputs = env->GetLongArrayElements(input_buffers, nullptr);
  std::vector<litert::TensorBuffer> input_buffer_vector;
  input_buffer_vector.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto litert_tensor_buffer = reinterpret_cast<LiteRtTensorBuffer>(inputs[i]);
    input_buffer_vector.push_back(
        litert::TensorBuffer(litert_tensor_buffer, false));
  }
  env->ReleaseLongArrayElements(input_buffers, inputs, 0);

  auto num_outputs = env->GetArrayLength(output_buffers);
  auto outputs = env->GetLongArrayElements(output_buffers, nullptr);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto litert_tensor_buffer =
        reinterpret_cast<LiteRtTensorBuffer>(outputs[i]);
    output_buffer_vector.push_back(
        litert::TensorBuffer(litert_tensor_buffer, false));
  }
  env->ReleaseLongArrayElements(output_buffers, outputs, 0);
  auto result = compiled_model.Run(signature_index, input_buffer_vector,
                                   output_buffer_vector);
  if (!result) {
    // TODO(niuchl): throw an exception.
  }
}

JNIEXPORT jboolean JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeRunAsync(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index, jlongArray input_buffers, jlongArray output_buffers) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  // Process input buffers
  const int num_inputs = env->GetArrayLength(input_buffers);
  jlong* inputs = env->GetLongArrayElements(input_buffers, nullptr);
  std::vector<litert::TensorBuffer> input_buffer_vector;
  input_buffer_vector.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(inputs[i]);
    input_buffer_vector.emplace_back(c_buffer, /*owned=*/false);
  }
  env->ReleaseLongArrayElements(input_buffers, inputs, 0);

  // Process output buffers
  const int num_outputs = env->GetArrayLength(output_buffers);
  jlong* outputs = env->GetLongArrayElements(output_buffers, nullptr);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(outputs[i]);
    output_buffer_vector.emplace_back(c_buffer, /*owned=*/false);
  }
  env->ReleaseLongArrayElements(output_buffers, outputs, 0);

  // Execute model asynchronously if supported
  bool async_executed = false;
  auto result = compiled_model.RunAsync(
      static_cast<size_t>(signature_index), input_buffer_vector,
      output_buffer_vector, /*OUT*/ async_executed);

  if (!result) {
    LITERT_LOG(LITERT_ERROR, "RunAsync() failed: %s",
               result.Error().Message().c_str());
    return JNI_FALSE;
  }
  if (!async_executed) {
    LITERT_LOG(LITERT_WARNING, "RunAsync() executed synchronously");
  }
  return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeDestroy(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  LiteRtDestroyCompiledModel(reinterpret_cast<LiteRtCompiledModel>(handle));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
