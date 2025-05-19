# Copyright 2025 Google LLC.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

source gbash.sh || exit 1
set -e  # Exit immediately if a command exits with a non-zero status

DEFINE_string device_id "" "ADB device when multiple devices are attached."
DEFINE_string device_group "other" \
  "Device group for CPU/GPU/NPU selection among 'other', 'Qualcomm_SM8750', "\
  "'Qualcomm_SM8650', 'Qualcomm_SM8550'"
DEFINE_bool use_gradle true "Whether to use gradle to build the app."

gbash::init_google "$@"

tmp_dir=$(mktemp -d)

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
APP_ROOT="$SCRIPT_DIR/../android"

if [[ "$FLAGS_use_gradle" == 1 ]]; then
  echo "Building LiteRT Shlib AAR..."
  blaze --blazerc=/dev/null build -c opt --config=android_arm64 --android_ndk_min_sdk_version=26 \
    //third_party/odml/litert/litert/kotlin:litert
  mkdir -p $APP_ROOT/app/libs
  cp -f blaze-genfiles/third_party/odml/litert/litert/kotlin/litert.aar \
    $APP_ROOT/app/libs

  echo "Setting up AI packs..."
  cp -R $SCRIPT_DIR/ai_pack/ $APP_ROOT/
  mv $APP_ROOT/ai_pack/device_targeting_configuration.xml $APP_ROOT/app/

  echo "Setting up NPU runtime library feature modules..."
  sh third_party/odml/litert/litert/google/npu_runtime_libraries/create_zip_archive.sh \
    --output_dir="$tmp_dir"
  RUNTIME_LIBS_DIR="$APP_ROOT/litert_npu_runtime_libraries"
  mkdir -p $RUNTIME_LIBS_DIR
  unzip -o $tmp_dir/litert_npu_runtime_libraries.zip -d $RUNTIME_LIBS_DIR > /dev/null
  sh $RUNTIME_LIBS_DIR/fetch_qualcomm_library.sh

  echo "Building App Bundle with Gradle..."
  (cd $APP_ROOT; ./gradlew bundle)
  AAB_PATH="$APP_ROOT/app/build/outputs/bundle/release/app-release.aab"
else
  echo "Building App Bundle with Blaze..."
  blaze --blazerc=/dev/null build -c opt --config=android_arm64 --android_ndk_min_sdk_version=26 \
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
