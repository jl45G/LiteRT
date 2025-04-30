#!/bin/bash

source gbash.sh || exit 1
set -e  # Exit immediately if a command exits with a non-zero status

DEFINE_string device_id "" "ADB device when multiple devices are attached."
DEFINE_string device_group "other" \
  "Device group for CPU/GPU/NPU selection among 'other', 'qti_v73', " \
  "'qti_v75', 'qti_v79'"

gbash::init_google "$@"

tmp_dir=$(mktemp -d)

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

blaze build --config=android_arm64 --android_ndk_min_sdk_version=34 \
  //third_party/odml/litert/litert/samples/image_segmentation/kotlin_cpu_gpu/android/app:image_segmentation_aab

bundletool='java -jar /google/bin/releases/bundletool/public/bundletool-all.jar'

$bundletool build-apks \
  --bundle=blaze-bin/third_party/odml/litert/litert/samples/image_segmentation/kotlin_cpu_gpu/android/app/image_segmentation_aab_unsigned.aab \
  --output="$tmp_dir/image_segmentation.apks" \
  --local-testing \
  --overwrite \
  --ks=tools/android/debug_keystore \
  --ks-pass=pass:android \
  --ks-key-alias=androiddebugkey

device_id_arg=""
if [[ -n "$FLAGS_device_id" ]]; then
  device_id_arg="--device-id=$FLAGS_device_id"
fi
$bundletool install-apks --apks="$tmp_dir/image_segmentation.apks" \
  $device_id_arg --device-groups=$FLAGS_device_group
