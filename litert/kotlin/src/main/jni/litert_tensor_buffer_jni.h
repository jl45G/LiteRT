#ifndef THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

///////////////////////////////////////////////////////////////////////////////
// EXISTING METHODS (array-based write/read/destroy)
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(
    JNIEnv* env, jclass clazz, jlong handle, jintArray input);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFloat(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray input);

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt(
    JNIEnv* env, jclass clazz, jlong handle);

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(
    JNIEnv* env, jclass clazz, jlong handle);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle);

///////////////////////////////////////////////////////////////////////////////
// ZERO-COPY METHODS (Create from direct ByteBuffer, read/write direct)
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromDirectBuffer(
    JNIEnv* env,
    jclass clazz,
    jint element_type_code,  // e.g. 0=Float32, 1=Int32, etc.
    jintArray dimensions,    // shape
    jobject direct_buffer,   // ByteBuffer.allocateDirect()
    jlong size_in_bytes);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFromDirect(
    JNIEnv* env,
    jclass clazz,
    jlong tensor_buffer_handle,
    jobject src_direct_buffer,
    jlong size_in_bytes);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadToDirect(
    JNIEnv* env,
    jclass clazz,
    jlong tensor_buffer_handle,
    jobject dst_direct_buffer,
    jlong size_in_bytes);

///////////////////////////////////////////////////////////////////////////////
// EVENT-RELATED METHODS
///////////////////////////////////////////////////////////////////////////////
JNIEXPORT jboolean JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeHasEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle);

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeSetEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle, jlong event_handle);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeClearEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWaitOnEvent(
    JNIEnv* env, jclass clazz, jlong buffer_handle, jlong timeout_ms);

JNIEXPORT jlong JNICALL
    Java_com_google_ai_edge_litert_AlignedBufferUtils_nativeGetDirectBufferAddress(
    JNIEnv* env, jclass clazz, jobject buffer);
// Existing methods omitted for brevity...

// =========================
// (A) AHardwareBuffer Interop
// =========================
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromAhwb(
    JNIEnv* env,
    jclass clazz,
    jint element_type_code,
    jintArray dimensions,
    jobject hardware_buffer,  // android.hardware.HardwareBuffer
    jlong ahwb_offset);

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetAhwb(
    JNIEnv* env,
    jclass clazz,
    jlong tensor_buffer_handle);

// =========================
// (B) GL Texture Interop
// =========================

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromGlTexture(
    JNIEnv* env,
    jclass clazz,
    jint element_type_code,
    jintArray dimensions,
    jint glTarget,
    jint glId,
    jint glFormat,
    jlong sizeBytes,
    jint layer);

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetGlTexture(
    JNIEnv* env,
    jclass clazz,
    jlong tensor_buffer_handle);

// =========================
// (C) GL Buffer Interop
// =========================
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeCreateFromGlBuffer(
    JNIEnv* env,
    jclass clazz,
    jint element_type_code,
    jintArray dimensions,
    jint glTarget,
    jint glId,
    jlong sizeBytes,
    jlong offset);

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeGetGlBuffer(
    JNIEnv* env,
    jclass clazz,
    jlong tensor_buffer_handle);

// The rest of your existing JNI function declarations...
// e.g. nativeWriteInt, nativeReadFloat, etc.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_
