#!/bin/bash

BLAZE_ARGS=(
  -c opt

  # Note: includes x86_64 so we can test the artifacts with emulators.
  --android_platforms=//buildenv/platforms/android:arm64-v8a,//buildenv/platforms/android:x86_64
  --android_ndk_min_sdk_version=26
)

rabbit --blazerc=/dev/null --verifiable mpm \
  "${BLAZE_ARGS[@]}" \
  //third_party/odml/litert/litert/google/release/android:mpm
