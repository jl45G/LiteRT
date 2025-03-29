#include "litert/kotlin/src/main/jni/litert_tensor_buffer_jni.h"

#include <jni.h>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_logging.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"
#include "absl/types/span.h"  // from @com_google_absl
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>
#include <vector>


#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#endif

#ifdef __APPLE__
// For iOS you might handle differently
#endif

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static bool IsDirectBuffer(JNIEnv* env, jobject buffer) {
  return (env->GetDirectBufferAddress(buffer) != nullptr);
}

// Maps the integer code from Java to a litert::ElementType
static litert::ElementType ConvertElementTypeCode(int code) {
  switch (code) {
    case 0:
      return litert::ElementType::Float32;
    case 1:
      return litert::ElementType::Int32;
    case 2:
      return litert::ElementType::Int8;
    // Add more if needed
    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported element type code: %d", code);
      return litert::ElementType::None;
  }
}
}  // namespace

// Convert Java int[] -> std::vector<int32_t>
static std::vector<int32_t> JIntArrayToVector(JNIEnv* env, jintArray jarr) {
  if (!jarr) return {};
  jsize length = env->GetArrayLength(jarr);
  std::vector<int32_t> result(length);
  jint* raw = env->GetIntArrayElements(jarr, nullptr);
  for (int i = 0; i < length; i++) {
    result[i] = raw[i];
  }
  env->ReleaseIntArrayElements(jarr, raw, JNI_ABORT);
  return result;
}

// Helper to build litert::RankedTensorType
static litert::RankedTensorType MakeRankedTensorType(
    litert::ElementType elemType, const std::vector<int32_t>& dims) {
  litert::Dimensions inlinedDims;
  inlinedDims.reserve(dims.size());
  for (int d : dims) {
    inlinedDims.push_back(d);
  }
  // Strides empty => row-major default
  litert::Layout layout(std::move(inlinedDims), litert::Strides());
  return litert::RankedTensorType(elemType, std::move(layout));
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jintArray input) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto write_result = tensor_buffer.Write<jint>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer.");
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFloat(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray input) {
  AUTO_CLEANUP_JNI_FLOAT_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto write_result = tensor_buffer.Write<jfloat>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer.");
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt8(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jbyteArray input) {
  AUTO_CLEANUP_JNI_BYTE_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto write_result = tensor_buffer.Write<jbyte>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer.");
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteBoolean(
    JNIEnv* env, jclass clazz, jlong handle, jbooleanArray input) {
  AUTO_CLEANUP_JNI_BOOLEAN_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto write_result = tensor_buffer.Write<jboolean>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer.");
  }
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    return nullptr;
  }
  auto lock_and_addr =
      litert::TensorBufferScopedLock::Create<const int>(tensor_buffer);
  jintArray result = env->NewIntArray(num_elements.value());
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetIntArrayRegion(result, 0, num_elements.value(),
                         lock_and_addr->second);
  return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    return nullptr;
  }

  auto lock_and_addr =
      litert::TensorBufferScopedLock::Create<const float>(tensor_buffer);
  jfloatArray result = env->NewFloatArray(num_elements.value());
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetFloatArrayRegion(result, 0, num_elements.value(),
                           lock_and_addr->second);
  return result;
}

JNIEXPORT jbyteArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt8(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    return nullptr;
  }

  auto lock_and_addr =
      litert::TensorBufferScopedLock::Create<const jbyte>(tensor_buffer);
  jbyteArray result = env->NewByteArray(num_elements.value());
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetByteArrayRegion(result, 0, num_elements.value(),
                          lock_and_addr->second);
  return result;
}

JNIEXPORT jbooleanArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadBoolean(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer = litert::TensorBuffer(tb, false);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    return nullptr;
  }

  auto lock_and_addr =
      litert::TensorBufferScopedLock::Create<const jboolean>(tensor_buffer);
  jbooleanArray result = env->NewBooleanArray(num_elements.value());
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetBooleanArrayRegion(result, 0, num_elements.value(),
                             lock_and_addr->second);
  return result;
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  LiteRtDestroyTensorBuffer(reinterpret_cast<LiteRtTensorBuffer>(handle));
}

// Zero-copy buffer management methods

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromDirectBuffer(
    JNIEnv* env, jclass, jint element_type_code, jintArray dimensions,
    jobject direct_buffer, jlong size_in_bytes) {
  litert::ElementType et = ConvertElementTypeCode(element_type_code);
  if (et == litert::ElementType::None) {
    return 0;
  }
  std::vector<int32_t> dims = JIntArrayToVector(env, dimensions);
  auto rtype = MakeRankedTensorType(et, dims);

  if (!IsDirectBuffer(env, direct_buffer)) {
    LITERT_LOG(LITERT_ERROR,
               "nativeCreateFromDirectBuffer: not a direct buffer!");
    return 0;
  }
  void* addr = env->GetDirectBufferAddress(direct_buffer);
  if (!addr) {
    LITERT_LOG(LITERT_ERROR,
               "nativeCreateFromDirectBuffer: could not get address!");
    return 0;
  }

  auto tb_or = litert::TensorBuffer::CreateFromHostMemory(
      rtype, addr, static_cast<size_t>(size_in_bytes));
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromHostMemory fail => %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFromDirect(
    JNIEnv* env, jclass, jlong tensor_buffer_handle, jobject src_direct_buffer,
    jlong size_in_bytes) {
  if (!tensor_buffer_handle) return;
  litert::TensorBuffer tb(
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle), false);

  if (!IsDirectBuffer(env, src_direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: src buffer not direct");
    return;
  }
  void* src_addr = env->GetDirectBufferAddress(src_direct_buffer);
  if (!src_addr) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: getAddress fail");
    return;
  }

  auto lockOr = tb.Lock();
  if (!lockOr) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: lock fail => %s",
               lockOr.Error().Message().c_str());
    return;
  }
  void* dst = *lockOr;
  auto tb_size = tb.Size();
  if (!tb_size) {
    tb.Unlock();
    LITERT_LOG(LITERT_ERROR, "nativeWriteFromDirect: get size fail => %s",
               tb_size.Error().Message().c_str());
    return;
  }

  size_t n = std::min(*tb_size, static_cast<size_t>(size_in_bytes));
  std::memcpy(dst, src_addr, n);
  tb.Unlock();
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadToDirect(
    JNIEnv* env, jclass, jlong tensor_buffer_handle, jobject dst_direct_buffer,
    jlong size_in_bytes) {
  if (!tensor_buffer_handle) return;
  litert::TensorBuffer tb(
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle), false);

  if (!IsDirectBuffer(env, dst_direct_buffer)) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: dst buffer not direct");
    return;
  }
  void* dst_addr = env->GetDirectBufferAddress(dst_direct_buffer);
  if (!dst_addr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: getAddress fail");
    return;
  }

  auto lockOr = tb.Lock();
  if (!lockOr) {
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: lock fail => %s",
               lockOr.Error().Message().c_str());
    return;
  }
  void* src = *lockOr;
  auto tb_size = tb.Size();
  if (!tb_size) {
    tb.Unlock();
    LITERT_LOG(LITERT_ERROR, "nativeReadToDirect: get size fail => %s",
               tb_size.Error().Message().c_str());
    return;
  }

  size_t n = std::min(*tb_size, static_cast<size_t>(size_in_bytes));
  std::memcpy(dst_addr, src, n);
  tb.Unlock();
}

// Event management methods

JNIEXPORT jboolean JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeHasEvent(
    JNIEnv* env, jclass, jlong buffer_handle) {
  if (!buffer_handle) return JNI_FALSE;
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          false);
  return tb.HasEvent() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetEvent(
    JNIEnv* env, jclass, jlong buffer_handle) {
  if (!buffer_handle) return 0;
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          false);
  auto ev_or = tb.GetEvent();
  if (!ev_or) {
    LITERT_LOG(LITERT_ERROR, "nativeGetEvent fail => %s",
               ev_or.Error().Message().c_str());
    return 0;
  }
  // Returns a non-owned handle to the event
  return reinterpret_cast<jlong>(ev_or->Get());
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeSetEvent(JNIEnv* env, jclass,
                                                           jlong buffer_handle,
                                                           jlong event_handle) {
  if (!buffer_handle || !event_handle) return;
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          false);
  litert::Event ev(reinterpret_cast<LiteRtEvent>(event_handle), true /*owned*/);
  auto st = tb.SetEvent(std::move(ev));
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeSetEvent fail => %s",
               st.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeClearEvent(
    JNIEnv* env, jclass, jlong buffer_handle) {
  if (!buffer_handle) return;
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          false);
  auto st = tb.ClearEvent();
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeClearEvent fail => %s",
               st.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWaitOnEvent(
    JNIEnv* env, jclass, jlong buffer_handle, jlong timeout_ms) {
  if (!buffer_handle) return;
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(buffer_handle),
                          false);
  auto ev_or = tb.GetEvent();
  if (!ev_or) {
    LITERT_LOG(LITERT_ERROR, "nativeWaitOnEvent: no event => %s",
               ev_or.Error().Message().c_str());
    return;
  }
  auto w = ev_or->Wait(static_cast<int64_t>(timeout_ms));
  if (!w) {
    LITERT_LOG(LITERT_ERROR, "nativeWaitOnEvent: wait fail => %s",
               w.Error().Message().c_str());
  }
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_AlignedBufferUtils_nativeGetDirectBufferAddress(
    JNIEnv* env, jclass, jobject buffer) {
  void* addr = env->GetDirectBufferAddress(buffer);
  return reinterpret_cast<jlong>(addr);
}

// AHardwareBuffer integration methods

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromAhwb(
    JNIEnv* env, jclass, jint element_type_code, jintArray dimensions,
    jobject hardware_buffer, jlong ahwb_offset) {
#if __ANDROID_API__ >= 29
  litert::ElementType et = ConvertElementTypeCode(element_type_code);
  if (et == litert::ElementType::None) {
    return 0;
  }
  std::vector<int32_t> dims = JIntArrayToVector(env, dimensions);
  auto rtype = MakeRankedTensorType(et, dims);

  AHardwareBuffer* c_ahwb =
      AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer);
  if (!c_ahwb) {
    LITERT_LOG(LITERT_ERROR,
               "nativeCreateFromAhwb: AHardwareBuffer_fromHardwareBuffer "
               "returned null");
    return 0;
  }
  auto tb_or = litert::TensorBuffer::CreateFromAhwb(
      rtype, c_ahwb, static_cast<size_t>(ahwb_offset));
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromAhwb fail => %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
#else
  LITERT_LOG(LITERT_ERROR,
             "nativeCreateFromAhwb: Android API < 29 not supported");
  return 0;
#endif
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetAhwb(
    JNIEnv* env, jclass, jlong tensor_buffer_handle) {
#if __ANDROID_API__ >= 29
  if (!tensor_buffer_handle) return nullptr;
  litert::TensorBuffer tb(
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle), false);
  auto ahwb_or = tb.GetAhwb();
  if (!ahwb_or) {
    LITERT_LOG(LITERT_ERROR, "nativeGetAhwb: fail => %s",
               ahwb_or.Error().Message().c_str());
    return nullptr;
  }
  AHardwareBuffer* c_ahwb = *ahwb_or;
  if (!c_ahwb) return nullptr;

  jobject jhwbuf = AHardwareBuffer_toHardwareBuffer(env, c_ahwb);
  return jhwbuf;
#else
  LITERT_LOG(LITERT_ERROR, "nativeGetAhwb: Android API < 29 not supported");
  return nullptr;
#endif
}

// OpenGL texture integration methods

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromGlTexture(
    JNIEnv* env, jclass, jint element_type_code, jintArray dimensions,
    jint glTarget, jint glId, jint glFormat, jlong sizeBytes, jint layer) {
#if LITERT_HAS_OPENGL_SUPPORT
  litert::ElementType et = ConvertElementTypeCode(element_type_code);
  if (et == litert::ElementType::None) return 0;
  std::vector<int32_t> dims = JIntArrayToVector(env, dimensions);
  auto rtype = MakeRankedTensorType(et, dims);

  auto tb_or = litert::TensorBuffer::CreateFromGlTexture(
      rtype, static_cast<GLenum>(glTarget), static_cast<GLuint>(glId),
      static_cast<GLenum>(glFormat), static_cast<size_t>(sizeBytes),
      static_cast<GLint>(layer));
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromGlTexture fail => %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
#else
  LITERT_LOG(LITERT_ERROR,
             "nativeCreateFromGlTexture: OpenGL support not available");
  return 0;
#endif
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetGlTexture(
    JNIEnv* env, jclass, jlong tensor_buffer_handle) {
#if LITERT_HAS_OPENGL_SUPPORT
  if (!tensor_buffer_handle) return nullptr;
  litert::TensorBuffer tb(
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle), false);
  auto gl_or = tb.GetGlTexture();
  if (!gl_or) {
    LITERT_LOG(LITERT_ERROR, "nativeGetGlTexture fail => %s",
               gl_or.Error().Message().c_str());
    return nullptr;
  }

  jintArray result = env->NewIntArray(6);
  jint vals[6];
  vals[0] = static_cast<jint>(gl_or->target);
  vals[1] = static_cast<jint>(gl_or->id);
  vals[2] = static_cast<jint>(gl_or->format);
  // Split 64-bit size into two 32-bit values for JNI compatibility
  uint64_t sb = static_cast<uint64_t>(gl_or->size_bytes);
  vals[3] = static_cast<jint>(sb & 0xFFFFFFFFul);
  vals[4] = static_cast<jint>((sb >> 32) & 0xFFFFFFFFul);
  vals[5] = static_cast<jint>(gl_or->layer);
  env->SetIntArrayRegion(result, 0, 6, vals);
  return result;
#else
  LITERT_LOG(LITERT_ERROR, "nativeGetGlTexture: OpenGL support not available");
  return nullptr;
#endif
}

// OpenGL buffer integration methods

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromGlBuffer(
    JNIEnv* env, jclass, jint element_type_code, jintArray dimensions,
    jint glTarget, jint glId, jlong sizeBytes, jlong offset) {
#if LITERT_HAS_OPENGL_SUPPORT
  litert::ElementType et = ConvertElementTypeCode(element_type_code);
  if (et == litert::ElementType::None) return 0;
  std::vector<int32_t> dims = JIntArrayToVector(env, dimensions);
  auto rtype = MakeRankedTensorType(et, dims);

  auto tb_or = litert::TensorBuffer::CreateFromGlBuffer(
      rtype, static_cast<GLenum>(glTarget), static_cast<GLuint>(glId),
      static_cast<size_t>(sizeBytes), static_cast<size_t>(offset));
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromGlBuffer fail => %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
#else
  LITERT_LOG(LITERT_ERROR,
             "nativeCreateFromGlBuffer: OpenGL support not available");
  return 0;
#endif
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetGlBuffer(
    JNIEnv* env, jclass, jlong tensor_buffer_handle) {
#if LITERT_HAS_OPENGL_SUPPORT
  if (!tensor_buffer_handle) return nullptr;
  litert::TensorBuffer tb(
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle), false);
  auto glbuf_or = tb.GetGlBuffer();
  if (!glbuf_or) {
    LITERT_LOG(LITERT_ERROR, "nativeGetGlBuffer fail => %s",
               glbuf_or.Error().Message().c_str());
    return nullptr;
  }
  // Return array with buffer parameters: [target, id, size_bytes, offset, 0, 0]
  // Last two elements reserved for future use
  jlongArray result = env->NewLongArray(6);
  jlong vals[6];
  vals[0] = static_cast<jlong>(glbuf_or->target);
  vals[1] = static_cast<jlong>(glbuf_or->id);
  vals[2] = static_cast<jlong>(glbuf_or->size_bytes);
  vals[3] = static_cast<jlong>(glbuf_or->offset);
  vals[4] = 0;  // Reserved for future use
  vals[5] = 0;  // Reserved for future use
  env->SetLongArrayRegion(result, 0, 6, vals);
  return result;
#else
  LITERT_LOG(LITERT_ERROR, "nativeGetGlBuffer: OpenGL support not available");
  return nullptr;
#endif
}

// TensorBufferScopedLock implementation

// Structure to track tensor buffer handle and locked pointer.
struct JniTensorScopedLock {
  LiteRtTensorBuffer c_tensor_buffer = nullptr;
  void* locked_ptr = nullptr;
};

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBufferScopedLock_nativeCreateScopedLock(
    JNIEnv* env, jclass /*clazz*/, jlong tensor_buffer_handle) {
  if (!tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateScopedLock: invalid buffer handle=0");
    return 0;
  }

  // Create non-owned wrapper for the tensor buffer
  litert::TensorBuffer tb(
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle),
      /*owned=*/false);

  // Lock the tensor buffer for memory access
  auto lockResult = tb.Lock();
  if (!lockResult) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateScopedLock: Lock() failed => %s",
               lockResult.Error().Message().c_str());
    return 0;
  }

  // Allocate tracking structure for the lock
  auto* scopedLock = new (std::nothrow) JniTensorScopedLock;
  if (!scopedLock) {
    LITERT_LOG(LITERT_ERROR,
               "nativeCreateScopedLock: Failed to allocate lock structure");
    tb.Unlock();
    return 0;
  }

  scopedLock->c_tensor_buffer =
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle);
  scopedLock->locked_ptr = *lockResult;

  return reinterpret_cast<jlong>(scopedLock);
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBufferScopedLock_nativeGetLockedPointer(
    JNIEnv* env, jclass /*clazz*/, jlong scoped_lock_handle) {
  if (!scoped_lock_handle) {
    return 0;
  }
  auto* lockObj = reinterpret_cast<JniTensorScopedLock*>(scoped_lock_handle);
  return reinterpret_cast<jlong>(lockObj->locked_ptr);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBufferScopedLock_nativeDestroyScopedLock(
    JNIEnv* env, jclass /*clazz*/, jlong scoped_lock_handle) {
  if (!scoped_lock_handle) return;
  auto* lockObj = reinterpret_cast<JniTensorScopedLock*>(scoped_lock_handle);
  if (!lockObj->c_tensor_buffer) {
    delete lockObj;
    return;
  }

  // Unlock the tensor buffer
  litert::TensorBuffer tb(lockObj->c_tensor_buffer, /*owned=*/false);
  auto st = tb.Unlock();
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeDestroyScopedLock: unlock failed => %s",
               st.Error().Message().c_str());
  }

  delete lockObj;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
