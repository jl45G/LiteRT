#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Event native methods
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateManaged(JNIEnv* env, jclass clazz,
                                                          jint type);

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromSyncFenceFd(JNIEnv* env,
                                                                  jclass clazz,
                                                                  jint fd,
                                                                  jboolean owns_fd);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_Event_nativeSignal(JNIEnv* env, jclass clazz,
                                                   jlong handle);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_Event_nativeDestroy(JNIEnv* env, jclass clazz,
                                                    jlong handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_