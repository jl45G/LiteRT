#include "litert/kotlin/src/main/jni/litert_event_jni.h"

#include <jni.h>

#include "litert/c/litert_event.h"
#include "litert/c/litert_event_type.h"
#include "litert/cc/litert_event.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateManaged(JNIEnv* env, jclass clazz,
                                                          jint type) {
  LiteRtEventType event_type;
  switch (type) {
    case 0:  // NONE
      event_type = kLiteRtEventTypeNone;
      break;
    case 1:  // SYNC_FENCE_FD
      event_type = kLiteRtEventTypeSyncFenceFd;
      break;
    case 2:  // OPEN_CL
      event_type = kLiteRtEventTypeOpenCl;
      break;
    default:
      return 0;  // Invalid type
  }
  
  auto event = litert::Event::CreateManaged(event_type);
  if (!event) {
    return 0;
  }
  
  // Release the event to avoid double deletion
  return reinterpret_cast<jlong>(event->Release());
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromSyncFenceFd(JNIEnv* env,
                                                                  jclass clazz,
                                                                  jint fd,
                                                                  jboolean owns_fd) {
  auto event = litert::Event::CreateFromSyncFenceFd(fd, owns_fd);
  if (!event) {
    return 0;
  }
  
  // Release the event to avoid double deletion
  return reinterpret_cast<jlong>(event->Release());
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_Event_nativeSignal(JNIEnv* env, jclass clazz,
                                                   jlong handle) {
  auto event = reinterpret_cast<LiteRtEvent>(handle);
  LiteRtEventSignal(event);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_Event_nativeDestroy(JNIEnv* env, jclass clazz,
                                                    jlong handle) {
  auto event = reinterpret_cast<LiteRtEvent>(handle);
  LiteRtEventDestroy(event);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus