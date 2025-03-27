#!/bin/bash

blaze build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //litert/kotlin/sample_apps/dummy_sample:dummy_sample_qualcomm
adb install -r \
  blaze-bin/litert/kotlin/sample_apps/dummy_sample/dummy_sample_qualcomm.apk

adb shell am start -a android.intent.action.MAIN \
  -n org.tensorflow.tflite.experimental.litert.sample/.MainActivity \
  --ez "use_npu_accelerator" true
