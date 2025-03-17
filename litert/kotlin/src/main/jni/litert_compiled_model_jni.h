#ifndef THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_COMPILED_MODEL_JNI_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_COMPILED_MODEL_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreate(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong env_handle,
                                                          jlong model_handle,
                                                          jintArray options);

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring input_name);

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring output_name);

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index);

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index);

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_CompiledModel_nativeRun(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index, jlongArray input_buffers, jlongArray output_buffers);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeDestroy(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle);
// Executes the model asynchronously if supported by the runtime.
// Returns JNI_TRUE if execution is performed asynchronously,
// JNI_FALSE if the runtime falls back to synchronous execution.
JNIEXPORT jboolean JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeRunAsync(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index, jlongArray input_buffers, jlongArray output_buffers);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_COMPILED_MODEL_JNI_H_
