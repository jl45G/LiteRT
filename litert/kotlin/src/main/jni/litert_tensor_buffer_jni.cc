#include "litert/kotlin/src/main/jni/litert_tensor_buffer_jni.h"

// ----------------------------------------------------------------------
// 1) Standard includes
// ----------------------------------------------------------------------
#include <jni.h>

#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include <cstring>
#include <vector>
#include <algorithm>

// ----------------------------------------------------------------------
// 2) LiteRT includes
// ----------------------------------------------------------------------
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"  // from @org_tensorflow

// ---------------------------------------------------------------------------------
// 3) HELPER: Check if a ByteBuffer is "direct" by calling GetDirectBufferAddress
// ---------------------------------------------------------------------------------
static bool IsDirectBuffer(JNIEnv* env, jobject buffer) {
  // If this returns non-null, the buffer is direct.
  void* addr = env->GetDirectBufferAddress(buffer);
  return (addr != nullptr);
}

// ----------------------------------------------------------------------
// 3) Helper: map "elementTypeCode" -> litert::ElementType
// ----------------------------------------------------------------------
static litert::ElementType ConvertElementTypeCode(int code) {
  // Example partial mapping:
  switch (code) {
    case 0: return litert::ElementType::Float32; // 0 => float32
    case 1: return litert::ElementType::Int32;   // 1 => int32
    case 2: return litert::ElementType::Int8;    // etc...
    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported element type code: %d", code);
      return litert::ElementType::None;
  }
}

// Helper for building litert::Dimensions from a std::vector<int32_t>.
static litert::Dimensions ConvertToDimensions(const std::vector<int32_t>& in) {
  litert::Dimensions dims;
  dims.reserve(in.size());
  for (auto val : in) {
    dims.push_back(val);
  }
  return dims;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

////////////////////////////////////////////////////////////////////////////////
// (A) EXISTING JNI METHODS (array-based read/write/destroy)
////////////////////////////////////////////////////////////////////////////////

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(
    JNIEnv* env, jclass clazz, jlong handle, jintArray input) {
  auto input_array = env->GetIntArrayElements(input, nullptr);
  jsize num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

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
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray input) {
  auto input_array = env->GetFloatArrayElements(input, nullptr);
  jsize num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

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
    JNIEnv* env, jclass clazz, jlong handle) {
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
  jintArray result = env->NewIntArray(num_elems.value());
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetIntArrayRegion(result, 0, num_elems.value(),
                         lock_and_addr->second);
  return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(
    JNIEnv* env, jclass clazz, jlong handle) {
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
    LITERT_LOG(LITERT_ERROR, "nativeReadFloat: unknown or dynamic shape.");
    return nullptr;
  }

  auto lock_and_addr =
      litert::TensorBufferScopedLock::Create<const float>(tensor_buffer);
  jfloatArray result = env->NewFloatArray(num_elems.value());
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetFloatArrayRegion(result, 0, num_elems.value(),
                           lock_and_addr->second);
  return result;
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
  LiteRtDestroyTensorBuffer(reinterpret_cast<LiteRtTensorBuffer>(handle));
}

////////////////////////////////////////////////////////////////////////////////
// (B) ZERO-COPY JNI METHODS
////////////////////////////////////////////////////////////////////////////////

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromDirectBuffer(
    JNIEnv* env,
    jclass clazz,
    jint element_type_code,
    jintArray dimensions,
    jobject direct_buffer,
    jlong size_in_bytes) {

  // 1) Convert element type
  litert::ElementType elemType = ConvertElementTypeCode(element_type_code);
  if (elemType == litert::ElementType::None) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateFromDirectBuffer: invalid code=%d",
               element_type_code);
    return 0;
  }

  // 2) Convert shape
  jsize rank = env->GetArrayLength(dimensions);
  std::vector<int32_t> dimsVec(rank);
  {
    jint* rawDims = env->GetIntArrayElements(dimensions, nullptr);
    for (int i = 0; i < rank; i++) {
      dimsVec[i] = static_cast<int32_t>(rawDims[i]);
    }
    env->ReleaseIntArrayElements(dimensions, rawDims, JNI_ABORT);
  }

  // 3) Check direct buffer
  if (!IsDirectBuffer(env,direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "ByteBuffer is not direct; cannot do zero copy.");
    return 0;
  }
  void* bufferAddr = env->GetDirectBufferAddress(direct_buffer);
  if (!bufferAddr) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateFromDirectBuffer: getDirectBufferAddress failed.");
    return 0;
  }

  // 4) Build the RankedTensorType
  litert::Dimensions dims = ConvertToDimensions(dimsVec);
  litert::Layout layout(std::move(dims), /*strides=*/{});
  litert::RankedTensorType rtype(elemType, std::move(layout));

  // 5) Create from host memory
  auto tb_or = litert::TensorBuffer::CreateFromHostMemory(
      rtype,
      bufferAddr,
      static_cast<size_t>(size_in_bytes));
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromHostMemory failed: %s",
               tb_or.Error().Message().c_str());
    return 0;
  }

  // 6) Return handle
  LiteRtTensorBuffer handle = tb_or->Release();
  return reinterpret_cast<jlong>(handle);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFromDirect(
    JNIEnv* env,
    jclass clazz,
    jlong tensor_buffer_handle,
    jobject src_direct_buffer,
    jlong size_in_bytes) {

  if (!tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: buffer handle=0");
    return;
  }
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle);
  litert::TensorBuffer tb(c_buffer, /*owned=*/false);

  if (!IsDirectBuffer(env, src_direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "src_direct_buffer not direct => can't zero copy");
    return;
  }
  void* srcAddr = env->GetDirectBufferAddress(src_direct_buffer);
  if (!srcAddr) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: getDirectBufferAddress failed.");
    return;
  }

  // Lock
  auto lockOr = tb.Lock();
  if (!lockOr) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: Lock() failed: %s",
               lockOr.Error().Message().c_str());
    return;
  }
  void* dstAddr = *lockOr;

  auto tb_size = tb.Size();
  if (!tb_size) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: tb.Size() failed: %s",
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
    JNIEnv* env,
    jclass clazz,
    jlong tensor_buffer_handle,
    jobject dst_direct_buffer,
    jlong size_in_bytes) {

  if (!tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: buffer handle=0");
    return;
  }
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle);
  litert::TensorBuffer tb(c_buffer, /*owned=*/false);

  if (!IsDirectBuffer(env, dst_direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "dst_direct_buffer not direct => can't zero copy");
    return;
  }
  void* dstAddr = env->GetDirectBufferAddress(dst_direct_buffer);
  if (!dstAddr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: getDirectBufferAddress failed.");
    return;
  }

  // Lock
  auto lockOr = tb.Lock();
  if (!lockOr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: Lock() failed: %s",
               lockOr.Error().Message().c_str());
    return;
  }
  void* srcAddr = *lockOr;

  auto tb_size = tb.Size();
  if (!tb_size) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: tb.Size() failed: %s",
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
// (C) EVENT-RELATED METHODS
////////////////////////////////////////////////////////////////////////////////

JNIEXPORT jboolean JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeHasEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle) {
  if (!buffer_handle) {
    return JNI_FALSE;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          /*owned=*/false);
  return tb.HasEvent() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle) {
  if (!buffer_handle) {
    return 0;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          /*owned=*/false);
  auto event_or = tb.GetEvent();
  if (!event_or) {
    LITERT_LOG(LITERT_ERROR, "nativeGetEvent: failed => %s",
               event_or.Error().Message().c_str());
    return 0;
  }
  // The returned Event is a "NonOwnedHandle" from the buffer. We'll pass
  // the raw pointer back. Usually not owned. If you want an Owned copy,
  // you'd do something else. For now, just do:
  return reinterpret_cast<jlong>(event_or->Get());
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeSetEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle, jlong event_handle) {
  if (!buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeSetEvent: buffer_handle=0");
    return;
  }
  if (!event_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeSetEvent: event_handle=0");
    return;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          /*owned=*/false);

  litert::Event ev(reinterpret_cast<LiteRtEvent>(event_handle),
                   /*owned=*/true); // pass ownership to the buffer
  auto st = tb.SetEvent(std::move(ev));
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeSetEvent: SetEvent failed => %s",
               st.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeClearEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle) {
  if (!buffer_handle) {
    return;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          /*owned=*/false);
  auto st = tb.ClearEvent();
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeClearEvent: ClearEvent failed => %s",
               st.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWaitOnEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle, jlong timeout_ms) {
  if (!buffer_handle) {
    return;
  }
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          /*owned=*/false);

  auto event_or = tb.GetEvent();
  if (!event_or) {
    LITERT_LOG(LITERT_ERROR, "nativeWaitOnEvent: GetEvent failed => %s",
               event_or.Error().Message().c_str());
    return;
  }
  // pass the desired ms, -1 => indefinite wait
  auto wait_st = event_or->Wait(static_cast<int64_t>(timeout_ms));
  if (!wait_st) {
    LITERT_LOG(LITERT_ERROR, "nativeWaitOnEvent: Wait() failed => %s",
               wait_st.Error().Message().c_str());
  }
}

JNIEXPORT jlong JNICALL
    Java_com_google_ai_edge_litert_AlignedBufferUtils_nativeGetDirectBufferAddress(
    JNIEnv* env, jclass clazz, jobject buffer) {
// Attempt to retrieve direct pointer
void* raw_ptr = env->GetDirectBufferAddress(buffer);
// Return as jlong
return reinterpret_cast<jlong>(raw_ptr);
}

#ifdef __cplusplus
}  // extern "C"
#endif
