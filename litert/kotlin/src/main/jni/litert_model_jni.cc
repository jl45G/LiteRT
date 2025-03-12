#include "litert/kotlin/src/main/jni/litert_model_jni.h"

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <cstdint>

#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"  // from @org_tensorflow
#endif  // __ANDROID__

#include <jni.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"  // from @org_tensorflow
#include "tensorflow/lite/experimental/litert/c/litert_model.h"  // from @org_tensorflow

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef __ANDROID__
JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadAsset(
    JNIEnv* env, jclass clazz, jobject asset_manager, jstring asset_name) {
  auto am = AAssetManager_fromJava(env, asset_manager);
  auto asset_name_str = env->GetStringUTFChars(asset_name, nullptr);
  auto g_model_asset =
      AAssetManager_open(am, asset_name_str, AASSET_MODE_BUFFER);

  auto buffer = litert::OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(AAsset_getBuffer(g_model_asset)),
      AAsset_getLength(g_model_asset));
  env->ReleaseStringUTFChars(asset_name, asset_name_str);
  AAsset_close(g_model_asset);

  LiteRtModel model = nullptr;
  auto status =
      LiteRtCreateModelFromBuffer(buffer.Data(), buffer.Size(), &model);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create model from asset.");
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}
#endif  // __ANDROID__

JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadFile(
    JNIEnv* env, jclass clazz, jstring file_path) {
  auto file_path_str = env->GetStringUTFChars(file_path, nullptr);
  LiteRtModel model = nullptr;
  auto status = LiteRtCreateModelFromFile(file_path_str, &model);
  env->ReleaseStringUTFChars(file_path, file_path_str);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create model from file.");
    return 0;
  }
  return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Model_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
  LiteRtDestroyModel(reinterpret_cast<LiteRtModel>(handle));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
