#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

source gbash.sh || exit

DEFINE_string output_dir --required "" "Output directory to store the zip archive."

google3_dir="$(gbash::get_google3_dir)"
src_dir_name="$(gbash::get_absolute_caller_dir)"
qnn_versions=(69 73 75 79)
jni_arm64_dir="src/main/jni/arm64-v8a"

dest_dir_name=$(gbash::make_temp_dir "litert_npu_runtime_libraries")
gbash::remove_directory_on_exit "${dest_dir_name}"

blaze_build() {
  blaze --blazerc=/dev/null build -c opt --config=android_arm64 --android_ndk_min_sdk_version=26 "$@"
}

main() {
  echo "Compiling Qualcomm dispatch API ..."
  blaze_build //third_party/odml/litert/litert/vendors/qualcomm/dispatch:dispatch_api_so

  cp -rf ${src_dir_name}/*runtime* "${dest_dir_name}/"
  cp -rf ${src_dir_name}/fetch_qualcomm_library.sh "${dest_dir_name}/"

  for version in "${qnn_versions[@]}"; do
    echo "Copying libraries to ${dest_dir_name}/qualcomm_runtime_v${version}/${jni_arm64_dir}/"
    mkdir -p "${dest_dir_name}/qualcomm_runtime_v${version}/${jni_arm64_dir}/"

    # libLiteRtDispatch_Qualcomm.so
    cp -rf ${google3_dir}/blaze-bin/third_party/odml/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so \
      "${dest_dir_name}/qualcomm_runtime_v${version}/${jni_arm64_dir}/"
  done

  echo "Compiling MTK dispatch API ..."
  blaze_build //third_party/odml/litert/litert/vendors/mediatek/dispatch:dispatch_api_so

  echo "Copying libraries to ${dest_dir_name}/mediatek_runtime/${jni_arm64_dir}/"
  mkdir -p "${dest_dir_name}/mediatek_runtime/${jni_arm64_dir}/"
  cp -rf ${google3_dir}/blaze-bin/third_party/odml/litert/litert/vendors/mediatek/dispatch/libLiteRtDispatch_Mediatek.so \
    "${dest_dir_name}/mediatek_runtime/${jni_arm64_dir}/"

  echo "Creating zip archive ..."
  pushd "${dest_dir_name}"
  zip -r ${FLAGS_output_dir}/litert_npu_runtime_libraries.zip ./ > /dev/null
  popd

  echo "Done, and the zip archive is available at ${FLAGS_output_dir}/litert_npu_runtime_libraries.zip"
}

gbash::main "$@"
