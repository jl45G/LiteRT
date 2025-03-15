#ifndef THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an Event from a Linux sync fence fd.
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromSyncFenceFd(
    JNIEnv* env, jclass clazz, jint sync_fence_fd, jboolean owns_fd);

// Creates an Event from an OpenCL event handle. The event handle is typically
// of type cl_event, passed as jlong from the Java side.
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromOpenClEvent(
    JNIEnv* env, jclass clazz, jlong cl_event_handle);

// Creates a "managed" event. (Internally uses LiteRtCreateManagedEvent.)
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateManaged(
    JNIEnv* env, jclass clazz, jint event_type);  // e.g. 0 for UNKNOWN, etc.

// Returns the sync fence fd from the event, or -1 if not applicable.
JNIEXPORT jint JNICALL
Java_com_google_ai_edge_litert_Event_nativeGetSyncFenceFd(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong event_handle);

// Returns the underlying cl_event as a jlong, or 0 if not applicable
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeGetOpenClEvent(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong event_handle);

// Wait on the event with the given timeout (ms). -1 = indefinite.
JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Event_nativeWait(
    JNIEnv* env, jclass clazz, jlong event_handle, jlong timeout_ms);

// Returns the event type, e.g. 0 for unknown, 1 for sync_fence, 2 for OpenCL,
// etc.
JNIEXPORT jint JNICALL Java_com_google_ai_edge_litert_Event_nativeGetType(
    JNIEnv* env, jclass clazz, jlong event_handle);

// Destroy the event. This frees the underlying LiteRtEvent resource.
JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Event_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong event_handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_
