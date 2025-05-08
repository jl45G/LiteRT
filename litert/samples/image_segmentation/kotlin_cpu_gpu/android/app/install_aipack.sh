#!/bin/bash

source gbash.sh || exit 1
set -e  # Exit immediately if a command exits with a non-zero status

DEFINE_string device_id "" "ADB device when multiple devices are attached."
DEFINE_string device_group "other" \
  "Device group for CPU/GPU/NPU selection among 'other', 'qti_v73', " \
  "'qti_v75', 'qti_v79'"
DEFINE_bool use_gradle true "Whether to use gradle to build the app."

gbash::init_google "$@"

tmp_dir=$(mktemp -d)

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

APP_ROOT="third_party/odml/litert/litert/samples/image_segmentation/kotlin_cpu_gpu/android"

if [[ "$FLAGS_use_gradle" == 1 ]]; then
  echo "Building LiteRT Shlib AAR..."
  blaze build -c opt --config=android_arm64 --android_ndk_min_sdk_version=26 \
    //third_party/odml/litert/litert/kotlin:litert_kotlin_api_aar
  mkdir -p $APP_ROOT/app/libs
  cp -f blaze-genfiles/third_party/odml/litert/litert/kotlin/litert_kotlin_api_aar.aar \
    $APP_ROOT/app/libs

  echo "Fetching NPU feature modules..."
  sh third_party/odml/litert/litert/google/npu_runtime_libraries/create_zip_archive.sh \
    --output_dir=/tmp/
  NPU_FEATURE_MODULES_ZIP_FILE=litert_npu_runtime_libraries.zip
  unzip -o -d $APP_ROOT /tmp/$NPU_FEATURE_MODULES_ZIP_FILE \
    "qnn_runtime_v73/*" "qnn_runtime_v75/*" "qnn_runtime_v79/*" "runtime_strings/*"
  rm /tmp/$NPU_FEATURE_MODULES_ZIP_FILE

  if [[ "$FLAGS_device_group" == "other" ]]; then
    USE_NPU=""
  else
    USE_NPU="-PuseNpu=true"
  fi

  echo "Building App Bundle with Gradle..."
  (cd $APP_ROOT; ./gradlew bundle $USE_NPU)
  AAB_PATH="$APP_ROOT/app/build/outputs/bundle/release/app-release.aab"
else
  echo "Building App Bundle with Blaze..."
  blaze build -c opt --config=android_arm64 --android_ndk_min_sdk_version=26 \
    $APP_ROOT/app:image_segmentation_aab
  AAB_PATH="blaze-bin/$APP_ROOT/app/image_segmentation_aab_unsigned.aab"
fi

bundletool='java -jar /google/bin/releases/bundletool/public/bundletool-all.jar'

echo "Building App Bundle APKs..."
$bundletool build-apks \
  --bundle=$AAB_PATH \
  --output="$tmp_dir/image_segmentation.apks" \
  --local-testing \
  --overwrite \
  --ks=tools/android/debug_keystore \
  --ks-pass=pass:android \
  --ks-key-alias=androiddebugkey

echo "Installing App Bundle APKs..."
device_id_arg=""
if [[ -n "$FLAGS_device_id" ]]; then
  device_id_arg="--device-id=$FLAGS_device_id"
fi
$bundletool install-apks --apks="$tmp_dir/image_segmentation.apks" \
  $device_id_arg --device-groups=$FLAGS_device_group
