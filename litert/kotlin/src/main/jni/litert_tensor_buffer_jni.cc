#include "litert/kotlin/src/main/jni/litert_tensor_buffer_jni.h"

#include <jni.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "litert/c/litert_logging.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_tensor_buffer.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#endif

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

////////////////////////////////////////////////////////////////////////////////
// (A) Existing JNI methods (read/write arrays)
////////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(JNIEnv* env, jclass,
                                                           jlong handle,
                                                           jintArray input) {
  if (!handle) return;
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(handle);
  litert::TensorBuffer tb(c_buffer, false);

  jsize num_elements = env->GetArrayLength(input);
  jint* data_ptr = env->GetIntArrayElements(input, nullptr);

  absl::Span<const jint> data_span(data_ptr, num_elements);
  auto result = tb.Write<jint>(data_span);

  env->ReleaseIntArrayElements(input, data_ptr, JNI_ABORT);

  if (!result) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteInt failed => %s",
               result.Error().Message().c_str());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFloat(
    JNIEnv* env, jclass, jlong handle, jfloatArray input) {
  if (!handle) return;
  auto c_buffer = reinterpret_cast<LiteRtTensorBuffer>(handle);
  litert::TensorBuffer tb(c_buffer, false);

  jsize num_elems = env->GetArrayLength(input);
  jfloat* data_ptr = env->GetFloatArrayElements(input, nullptr);

  absl::Span<const jfloat> data_span(data_ptr, num_elems);
  auto result = tb.Write<jfloat>(data_span);

  env->ReleaseFloatArrayElements(input, data_ptr, JNI_ABORT);
  if (!result) {
    LITERT_LOG(LITERT_ERROR, "nativeWriteFloat failed => %s",
               result.Error().Message().c_str());
  }
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt(JNIEnv* env, jclass,
                                                          jlong handle) {
  if (!handle) return nullptr;
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(handle), false);

  auto type_or = tb.TensorType();
  if (!type_or) {
    LITERT_LOG(LITERT_ERROR, "nativeReadInt: no tensor type => %s",
               type_or.Error().Message().c_str());
    return nullptr;
  }
  auto num_elems = type_or->Layout().NumElements();
  if (!num_elems.has_value()) {
    LITERT_LOG(LITERT_ERROR, "nativeReadInt: dynamic shape not supported");
    return nullptr;
  }

  std::vector<int> temp(num_elems.value());
  auto read_res = tb.Read<int>(absl::MakeSpan(temp));
  if (!read_res) {
    LITERT_LOG(LITERT_ERROR, "nativeReadInt: read fail => %s",
               read_res.Error().Message().c_str());
    return nullptr;
  }

  jintArray result = env->NewIntArray(num_elems.value());
  env->SetIntArrayRegion(result, 0, num_elems.value(), temp.data());
  return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(JNIEnv* env, jclass,
                                                            jlong handle) {
  if (!handle) return nullptr;
  litert::TensorBuffer tb(reinterpret_cast<LiteRtTensorBuffer>(handle), false);

  auto type_or = tb.TensorType();
  if (!type_or) {
    LITERT_LOG(LITERT_ERROR, "nativeReadFloat: no tensor type => %s",
               type_or.Error().Message().c_str());
    return nullptr;
  }
  auto num_elems = type_or->Layout().NumElements();
  if (!num_elems.has_value()) {
    LITERT_LOG(LITERT_ERROR, "nativeReadFloat: dynamic shape not supported");
    return nullptr;
  }

  std::vector<float> temp(num_elems.value());
  auto read_res = tb.Read<float>(absl::MakeSpan(temp));
  if (!read_res) {
    LITERT_LOG(LITERT_ERROR, "nativeReadFloat: read fail => %s",
               read_res.Error().Message().c_str());
    return nullptr;
  }

  jfloatArray result = env->NewFloatArray(num_elems.value());
  env->SetFloatArrayRegion(result, 0, num_elems.value(), temp.data());
  return result;
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(JNIEnv* env, jclass,
                                                          jlong handle) {
  if (handle) {
    LiteRtDestroyTensorBuffer(reinterpret_cast<LiteRtTensorBuffer>(handle));
  }
}

////////////////////////////////////////////////////////////////////////////////
// (B) Zero-copy JNI
////////////////////////////////////////////////////////////////////////////////
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
      rtype, addr, (size_t)size_in_bytes);
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

  size_t n = std::min(*tb_size, (size_t)size_in_bytes);
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

  size_t n = std::min(*tb_size, (size_t)size_in_bytes);
  std::memcpy(dst_addr, src, n);
  tb.Unlock();
}

////////////////////////////////////////////////////////////////////////////////
// (C) EVENT-RELATED
////////////////////////////////////////////////////////////////////////////////
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
  // This returns a non-owned handle
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
  auto w = ev_or->Wait((int64_t)timeout_ms);
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

////////////////////////////////////////////////////////////////////////////////
// (D) AHardwareBuffer Interop
////////////////////////////////////////////////////////////////////////////////
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
  auto tb_or =
      litert::TensorBuffer::CreateFromAhwb(rtype, c_ahwb, (size_t)ahwb_offset);
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromAhwb fail => %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
#else
  LITERT_LOG(LITERT_ERROR, "nativeCreateFromAhwb: < 29 not supported");
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
  LITERT_LOG(LITERT_ERROR, "nativeGetAhwb: < 29 not supported");
  return nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// (E) GL TEXTURE
////////////////////////////////////////////////////////////////////////////////
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
      rtype, (GLenum)glTarget, (GLuint)glId, (GLenum)glFormat,
      (size_t)sizeBytes, (GLint)layer);
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromGlTexture fail => %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
#else
  LITERT_LOG(LITERT_ERROR, "nativeCreateFromGlTexture: no OpenGL support");
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
  // We'll pack [target, id, format, sizeBytes (low32?), sizeBytes(hi32?),
  // layer] into an int array If you only want the first few, adapt as needed.
  jintArray result = env->NewIntArray(6);
  jint vals[6];
  vals[0] = (jint)gl_or->target;
  vals[1] = (jint)gl_or->id;
  vals[2] = (jint)gl_or->format;
  // We can only pass 32-bit so let's do a simple split if needed
  uint64_t sb = (uint64_t)(gl_or->size_bytes);
  vals[3] = (jint)(sb & 0xFFFFFFFFul);
  vals[4] = (jint)((sb >> 32) & 0xFFFFFFFFul);
  vals[5] = (jint)gl_or->layer;
  env->SetIntArrayRegion(result, 0, 6, vals);
  return result;
#else
  LITERT_LOG(LITERT_ERROR, "nativeGetGlTexture: no OpenGL support");
  return nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// (F) GL BUFFER
////////////////////////////////////////////////////////////////////////////////
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
      rtype, (GLenum)glTarget, (GLuint)glId, (size_t)sizeBytes, (size_t)offset);
  if (!tb_or) {
    LITERT_LOG(LITERT_ERROR, "CreateFromGlBuffer fail => %s",
               tb_or.Error().Message().c_str());
    return 0;
  }
  return reinterpret_cast<jlong>(tb_or->Release());
#else
  LITERT_LOG(LITERT_ERROR, "nativeCreateFromGlBuffer: no OpenGL support");
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
  // We'll pack [target, id, sizeBytesLow, sizeBytesHigh, offsetLow, offsetHigh]
  // into a jlongArray (or int?) Here let's just do jlongArray of length 6 so we
  // can store 64-bit fields directly.
  jlongArray result = env->NewLongArray(6);
  jlong vals[6];
  vals[0] = (jlong)(glbuf_or->target);  // though it's typically 32-bit
  vals[1] = (jlong)(glbuf_or->id);
  vals[2] = (jlong)(glbuf_or->size_bytes);
  vals[3] = (jlong)(glbuf_or->offset);
  // The last two can be zeros if you want, or use them for something else
  vals[4] = 0;
  vals[5] = 0;
  env->SetLongArrayRegion(result, 0, 6, vals);
  return result;
#else
  LITERT_LOG(LITERT_ERROR, "nativeGetGlBuffer: no OpenGL support");
  return nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// TENSOR SCOPED LOCK IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

// We store a small struct to remember the buffer handle & the locked pointer.
struct JniTensorScopedLock {
  LiteRtTensorBuffer c_tensor_buffer = nullptr;
  void* locked_ptr = nullptr;
};

// 1) CREATE SCOPED LOCK
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBufferScopedLock_nativeCreateScopedLock(
    JNIEnv* env, jclass /*clazz*/, jlong tensor_buffer_handle) {
  if (!tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateScopedLock: invalid buffer handle=0");
    return 0;
  }

  // Wrap it in the C++ convenience class (non‐owned).
  litert::TensorBuffer tb(
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle),
      /*owned=*/false);
  // Attempt to lock
  auto lockResult = tb.Lock();
  if (!lockResult) {
    LITERT_LOG(LITERT_ERROR, "nativeCreateScopedLock: Lock() failed => %s",
               lockResult.Error().Message().c_str());
    return 0;
  }

  // Allocate a small struct on the heap to store the info.
  auto* scopedLock = new (std::nothrow) JniTensorScopedLock;
  if (!scopedLock) {
    LITERT_LOG(LITERT_ERROR,
               "nativeCreateScopedLock: OOM allocating lock struct");
    // Must unlock if we can’t store it
    tb.Unlock();
    return 0;
  }
  scopedLock->c_tensor_buffer =
      reinterpret_cast<LiteRtTensorBuffer>(tensor_buffer_handle);
  scopedLock->locked_ptr = *lockResult;
  // Return the pointer to this struct as a jlong
  return reinterpret_cast<jlong>(scopedLock);
}

// 2) GET LOCKED POINTER
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBufferScopedLock_nativeGetLockedPointer(
    JNIEnv* env, jclass /*clazz*/, jlong scoped_lock_handle) {
  if (!scoped_lock_handle) {
    return 0;
  }
  auto* lockObj = reinterpret_cast<JniTensorScopedLock*>(scoped_lock_handle);
  // Return the pointer as a jlong
  return reinterpret_cast<jlong>(lockObj->locked_ptr);
}

// 3) DESTROY SCOPED LOCK
JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBufferScopedLock_nativeDestroyScopedLock(
    JNIEnv* env, jclass /*clazz*/, jlong scoped_lock_handle) {
  if (!scoped_lock_handle) return;
  auto* lockObj = reinterpret_cast<JniTensorScopedLock*>(scoped_lock_handle);
  if (!lockObj->c_tensor_buffer) {
    // Nothing to unlock
    delete lockObj;
    return;
  }

  // Actually call Unlock on the underlying buffer
  litert::TensorBuffer tb(lockObj->c_tensor_buffer, /*owned=*/false);
  auto st = tb.Unlock();
  if (!st) {
    LITERT_LOG(LITERT_ERROR, "nativeDestroyScopedLock: unlock failed => %s",
               st.Error().Message().c_str());
  }
  // free
  delete lockObj;
}

#ifdef __cplusplus
}
#endif
