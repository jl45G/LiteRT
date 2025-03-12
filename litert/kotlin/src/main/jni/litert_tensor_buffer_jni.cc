
#include "litert/kotlin/src/main/jni/litert_tensor_buffer_jni.h"

// (NEW) Our helper for shape conversion

#include <jni.h>
#include <vector>
#include <cstring>   // for std::memcpy
#include <algorithm> // for std::min

#ifdef __ANDROID__
// For older NDK or if you are building for < 29, guard usage of AHardwareBuffer
#include <android/hardware_buffer.h>
#endif

#include <GLES3/gl3.h> // Or <GLES2/gl2.h> if needed

#include "tensorflow/lite/experimental/litert/c/litert_logging.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"  // from @org_tensorflow

// ----------------------------------------------------------------------
// HELPER: Check if a ByteBuffer is direct by calling GetDirectBufferAddress
// ----------------------------------------------------------------------
static bool IsDirectBuffer(JNIEnv* env, jobject buffer) {
  return (env->GetDirectBufferAddress(buffer) != nullptr);
}

// ----------------------------------------------------------------------
// HELPER: map "elementTypeCode" -> litert::ElementType
// ----------------------------------------------------------------------
static litert::ElementType ConvertElementTypeCode(int code) {
  // Example partial mapping:
  switch (code) {
    case 0: return litert::ElementType::Float32;
    case 1: return litert::ElementType::Int32;
    case 2: return litert::ElementType::Int8;
    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported element type code: %d", code);
      return litert::ElementType::None;
  }
}

static litert::Dimensions ConvertToInlinedVector(const std::vector<int32_t>& src) {
  litert::Dimensions out;
  out.reserve(src.size());
  for (auto val : src) {
    out.push_back(val);
  }
  return out;
}

// ----------------------------------------------------------------------
// HELPER: read Java int[] => std::vector<int32_t>
static std::vector<int32_t> BuildStdVectorFromJIntArray(JNIEnv* env, jintArray jdims) {
  if (!jdims) {
    return {};
  }
  jsize rank = env->GetArrayLength(jdims);
  std::vector<int32_t> dims(rank);
  jint* raw = env->GetIntArrayElements(jdims, nullptr);
  for (int i = 0; i < rank; i++) {
    dims[i] = static_cast<int32_t>(raw[i]);
  }
  env->ReleaseIntArrayElements(jdims, raw, JNI_ABORT);
  return dims;
}

// ----------------------------------------------------------------------
// HELPER: Build a litert::RankedTensorType if needed
// Suppose the constructor is: RankedTensorType(ElementType, Layout&&).
// So we do a simple helper to pass std::move(layout).
static litert::RankedTensorType MakeRankedTensorType(
    litert::ElementType elemType,
    litert::Layout&& layout) {
  return litert::RankedTensorType(elemType, std::move(layout));
}

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// (A) Existing JNI methods: array-based read/write/destroy
////////////////////////////////////////////////////////////////////////////////

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(
    JNIEnv* env, jclass /*clazz*/, jlong handle, jintArray input) {
  auto input_array = env->GetIntArrayElements(input, nullptr);
  jsize num_elements = env->GetArrayLength(input);
  absl::Span<const jint> input_span(input_array, num_elements);

  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(handle);
  litert::TensorBuffer tensor_buffer(c_buffer, /*owned=*/false);

  auto write_result = tensor_buffer.Write<jint>(input_span);
  env->ReleaseIntArrayElements(input, input_array, JNI_ABORT);

  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteInt: Failed: %s",
               write_result.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFloat(
    JNIEnv* env, jclass /*clazz*/, jlong handle, jfloatArray input) {
  auto input_array = env->GetFloatArrayElements(input, nullptr);
  jsize num_elements = env->GetArrayLength(input);
  absl::Span<const jfloat> input_span(input_array, num_elements);

  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(handle);
  litert::TensorBuffer tensor_buffer(c_buffer, /*owned=*/false);

  auto write_result = tensor_buffer.Write<jfloat>(input_span);
  env->ReleaseFloatArrayElements(input, input_array, JNI_ABORT);

  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFloat: Failed: %s",
               write_result.Error().Message().c_str());
  }
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt(
    JNIEnv* env, jclass /*clazz*/, jlong handle) {
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(handle);
  litert::TensorBuffer tensor_buffer(c_buffer, /*owned=*/false);

  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "nativeReadInt: No TensorType: %s",
               tensor_type.Error().Message().c_str());
    return nullptr;
  }

  auto num_elems = tensor_type->Layout().NumElements();
  if (!num_elems.has_value()) {
    LITERT_LOG(LITERT_ERROR, "nativeReadInt: unknown or dynamic shape.");
    return nullptr;
  }

  auto lock_and_addr =
      litert::TensorBufferScopedLock::Create<const int>(tensor_buffer);
  if (!lock_and_addr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadInt: lock failed => %s",
               lock_and_addr.Error().Message().c_str());
    return nullptr;
  }

  jintArray result = env->NewIntArray(num_elems.value());
  env->SetIntArrayRegion(result, 0, num_elems.value(),
                         lock_and_addr->second);
  return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(
    JNIEnv* env, jclass /*clazz*/, jlong handle) {
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(handle);
  litert::TensorBuffer tensor_buffer(c_buffer, /*owned=*/false);

  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "nativeReadFloat: No TensorType: %s",
               tensor_type.Error().Message().c_str());
    return nullptr;
  }

  auto num_elems = tensor_type->Layout().NumElements();
  if (!num_elems.has_value()) {
    LITERT_LOG(LITERT_ERROR, "nativeReadFloat: dynamic shape? Not supported.");
    return nullptr;
  }

  auto lock_and_addr =
      litert::TensorBufferScopedLock::Create<const float>(tensor_buffer);
  if (!lock_and_addr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadFloat: lock failed => %s",
               lock_and_addr.Error().Message().c_str());
    return nullptr;
  }

  jfloatArray result = env->NewFloatArray(num_elems.value());
  env->SetFloatArrayRegion(result, 0, num_elems.value(),
                           lock_and_addr->second);
  return result;
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(
    JNIEnv* /*env*/, jclass /*clazz*/, jlong handle) {
  LiteRtDestroyTensorBuffer(reinterpret_cast<LiteRtTensorBuffer>(handle));
}

////////////////////////////////////////////////////////////////////////////////
// (B) Zero-copy JNI
////////////////////////////////////////////////////////////////////////////////

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromDirectBuffer(
    JNIEnv* env,
    jclass /*clazz*/,
    jint element_type_code,
    jintArray dimensions,
    jobject direct_buffer,
    jlong size_in_bytes) {

  // 1) Convert element type
  litert::ElementType elemType = ConvertElementTypeCode(element_type_code);
  if (elemType == litert::ElementType::None) {
    return 0;
  }

  // 2) Build dims from JNI
  auto dims = BuildStdVectorFromJIntArray(env, dimensions);
  // Convert to InlinedVector:
  litert::Dimensions inlinedDims = ConvertToInlinedVector(dims);
  litert::Layout layout(std::move(inlinedDims), litert::Strides());

  // 3) Check direct buffer
  if (!IsDirectBuffer(env, direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateFromDirectBuffer: not direct");
    return 0;
  }
  void* bufferAddr = env->GetDirectBufferAddress(direct_buffer);
  if (!bufferAddr) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateFromDirectBuffer: address fail");
    return 0;
  }

  // 4) Build RankedTensorType
  litert::RankedTensorType rtype(elemType, std::move(layout));

  // 5) Create from host memory
  auto tb_or = litert::TensorBuffer::CreateFromHostMemory(
      rtype, bufferAddr, static_cast<size_t>(size_in_bytes));
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromHostMemory fail: %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFromDirect(
    JNIEnv* env, jclass /*clazz*/,
    jlong tensor_buffer_handle,
    jobject src_direct_buffer,
    jlong size_in_bytes) {

  if (!tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: buffer=0");
    return;
  }
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle);
  litert::TensorBuffer tb(c_buffer, /*owned=*/false);

  if (!IsDirectBuffer(env, src_direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "src_direct_buffer not direct => no zero copy");
    return;
  }
  void* srcAddr = env->GetDirectBufferAddress(src_direct_buffer);
  if (!srcAddr) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: getAddress fail");
    return;
  }

  auto lockOr = tb.Lock();
  if (!lockOr) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: Lock fail => %s",
               lockOr.Error().Message().c_str());
    return;
  }
  void* dstAddr = *lockOr;

  auto tb_size = tb.Size();
  if (!tb_size) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: Size fail => %s",
               tb_size.Error().Message().c_str());
    tb.Unlock();
    return;
  }

  size_t n = std::min(static_cast<size_t>(*tb_size),
                      static_cast<size_t>(size_in_bytes));
  std::memcpy(dstAddr, srcAddr, n);
  tb.Unlock();
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadToDirect(
    JNIEnv* env, jclass /*clazz*/,
    jlong tensor_buffer_handle,
    jobject dst_direct_buffer,
    jlong size_in_bytes) {

  if (!tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: buffer=0");
    return;
  }
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle);
  litert::TensorBuffer tb(c_buffer, /*owned=*/false);

  if (!IsDirectBuffer(env, dst_direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "dst_direct_buffer not direct => no zero copy");
    return;
  }
  void* dstAddr = env->GetDirectBufferAddress(dst_direct_buffer);
  if (!dstAddr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: getAddress fail");
    return;
  }

  auto lockOr = tb.Lock();
  if (!lockOr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: Lock fail => %s",
               lockOr.Error().Message().c_str());
    return;
  }
  void* srcAddr = *lockOr;

  auto tb_size = tb.Size();
  if (!tb_size) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: Size fail => %s",
               tb_size.Error().Message().c_str());
    tb.Unlock();
    return;
  }

  size_t n = std::min(static_cast<size_t>(*tb_size),
                      static_cast<size_t>(size_in_bytes));
  std::memcpy(dstAddr, srcAddr, n);
  tb.Unlock();
}

////////////////////////////////////////////////////////////////////////////////
// (C) EVENT-RELATED
////////////////////////////////////////////////////////////////////////////////

JNIEXPORT jboolean JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeHasEvent(
    JNIEnv* /*env*/, jclass /*clazz*/, jlong buffer_handle) {
  if (!buffer_handle) {
    return JNI_FALSE;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          /*owned=*/false);
  return tb.HasEvent() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetEvent(
    JNIEnv* /*env*/, jclass /*clazz*/, jlong buffer_handle) {
  if (!buffer_handle) {
    return 0;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          /*owned=*/false);
  auto event_or = tb.GetEvent();
  if (!event_or) {
    LITERT_LOG(LITERT_ERROR, "nativeGetEvent: fail => %s",
               event_or.Error().Message().c_str());
    return 0;
  }
  // Return raw pointer for non-owned usage:
  return reinterpret_cast<jlong>(event_or->Get());
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeSetEvent(
    JNIEnv* /*env*/, jclass /*clazz*/,
    jlong buffer_handle, jlong event_handle) {
  if (!buffer_handle || !event_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeSetEvent: invalid handle(s)");
    return;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle), false);
  litert::Event ev(reinterpret_cast<LiteRtEvent>(event_handle), /*owned=*/true);
  auto st = tb.SetEvent(std::move(ev));
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeSetEvent: fail => %s",
               st.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeClearEvent(
    JNIEnv* /*env*/, jclass /*clazz*/, jlong buffer_handle) {
  if (!buffer_handle) {
    return;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle), false);
  auto st = tb.ClearEvent();
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeClearEvent: fail => %s",
               st.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWaitOnEvent(
    JNIEnv* /*env*/, jclass /*clazz*/,
    jlong buffer_handle, jlong timeout_ms) {
  if (!buffer_handle) {
    return;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle), false);
  auto event_or = tb.GetEvent();
  if (!event_or) {
    LITERT_LOG(LITERT_ERROR, "nativeWaitOnEvent: fail => %s",
               event_or.Error().Message().c_str());
    return;
  }
  auto wait_st = event_or->Wait(static_cast<int64_t>(timeout_ms));
  if (!wait_st) {
    LITERT_LOG(LITERT_ERROR, "nativeWaitOnEvent: Wait fail => %s",
               wait_st.Error().Message().c_str());
  }
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_AlignedBufferUtils_nativeGetDirectBufferAddress(
    JNIEnv* env, jclass /*clazz*/, jobject buffer) {
  void* raw_ptr = env->GetDirectBufferAddress(buffer);
  return reinterpret_cast<jlong>(raw_ptr);
}

// (C) AHardwareBuffer Interop
#if __ANDROID_API__ >= 29
#include <android/hardware_buffer_jni.h>
#endif

// For createFromAhwb
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromAhwb(
    JNIEnv* env, jclass /*clazz*/,
    jint element_type_code,
    jintArray dimensions,
    jobject hardware_buffer,
    jlong ahwb_offset) {

  // 1) Convert element type
  litert::ElementType elemTy = ConvertElementTypeCode(element_type_code);
  if (elemTy == litert::ElementType::None) {
    return 0;
  }

  // 2) Convert dims
  auto dimsVec = BuildStdVectorFromJIntArray(env, dimensions);
  litert::Dimensions inlinedDims = ConvertToInlinedVector(dimsVec);
  litert::Strides empty_strides;
  litert::Layout layout(std::move(inlinedDims), std::move(empty_strides));

  litert::RankedTensorType rtype(elemTy, std::move(layout));

#if __ANDROID_API__ >= 29
  // Actually create from AHWB
  AHardwareBuffer* c_ahwb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer);
  if (!c_ahwb) {
    LITERT_LOG(LITERT_ERROR, "AHardwareBuffer_fromHardwareBuffer returned null");
    return 0;
  }
  auto tb_or = litert::TensorBuffer::CreateFromAhwb(rtype, c_ahwb, (size_t)ahwb_offset);
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromAhwb fail => %s", tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
#else
  LITERT_LOG(LITERT_ERROR, "nativeCreateFromAhwb: not supported < API 29");
  return 0;
#endif
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetAhwb(
    JNIEnv* env, jclass /*clazz*/, jlong tensor_buffer_handle) {
#if __ANDROID_API__ >= 29
  if (!tensor_buffer_handle) {
    return nullptr;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle), false);
  auto ahwb_or = tb.GetAhwb();
  if (!ahwb_or) {
    LITERT_LOG(LITERT_ERROR, "nativeGetAhwb fail => %s", ahwb_or.Error().Message().c_str());
    return nullptr;
  }
  AHardwareBuffer* c_ahwb = *ahwb_or;
  return AHardwareBuffer_toHardwareBuffer(env, c_ahwb);
#else
  LITERT_LOG(LITERT_ERROR, "nativeGetAhwb: < 29 not supported");
  return nullptr;
#endif
}

// (D) Possibly other GL createFromGlTexture or createFromGlBuffer code ...
// omitted for brevity, but follows the same pattern.

#ifdef __cplusplus
}
#endif
