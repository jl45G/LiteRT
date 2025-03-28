# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//devtools/deviceinfra/api/builddefs/device:rules.bzl", "satellite_lab_device")
load("//devtools/mobileharness/api/builddef/ait:ait.bzl", "android_logcat_decorator")
load("//mobile/build:build_defs.bzl", "android_jni_library")
load(
    "//tools/build_defs/android:rules.bzl",
    "android_binary",
    "android_instrumentation_test",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//third_party/odml/litert:__subpackages__",
        "//litert:__subpackages__",
    ],
)

cc_library(
    name = "litert_jni_common",
    hdrs = ["src/main/jni/litert_jni_common.h"],
    deps = [
        "@com_google_absl//absl/cleanup",  # buildcleaner: keep
    ],
)

cc_library(
    name = "litert_environment_jni",
    srcs = ["src/main/jni/litert_environment_jni.cc"],
    hdrs = ["src/main/jni/litert_environment_jni.h"],
    deps = [
        ":litert_jni_common",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_environment",
    ] + select({
        "//tools/cc_target_os:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

cc_library(
    name = "litert_model_jni",
    srcs = ["src/main/jni/litert_model_jni.cc"],
    hdrs = ["src/main/jni/litert_model_jni.h"],
    deps = [
        ":litert_jni_common",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_buffer_ref",
    ] + select({
        "//tools/cc_target_os:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

cc_library(
    name = "litert_compiled_model_jni",
    srcs = ["src/main/jni/litert_compiled_model_jni.cc"],
    hdrs = ["src/main/jni/litert_compiled_model_jni.h"],
    deps = [
        ":litert_jni_common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_tensor_buffer",
    ] + select({
        "//tools/cc_target_os:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

cc_library(
    name = "litert_tensor_buffer_jni",
    srcs = ["src/main/jni/litert_tensor_buffer_jni.cc"],
    hdrs = ["src/main/jni/litert_tensor_buffer_jni.h"],
    deps = [
        ":litert_jni_common",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_tensor_buffer",
        "@opencl_headers",
    ] + select({
        "//tools/cc_target_os:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

android_jni_library(
    name = "litert_jni",
    binary_name = "liblitert_jni.so",
    force_android = True,
    linkopts = ["-landroid"],
    deps = [
        ":litert_compiled_model_jni",
        ":litert_environment_jni",
        ":litert_event_jni",
        ":litert_model_jni",
        ":litert_tensor_buffer_jni",
    ],
)

cc_library(
    name = "litert_event_jni",
    srcs = ["src/main/jni/litert_event_jni.cc"],
    hdrs = ["src/main/jni/litert_event_jni.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/experimental/litert/c:litert_common",
        "@org_tensorflow//tensorflow/lite/experimental/litert/c:litert_event",
        "@org_tensorflow//tensorflow/lite/experimental/litert/c:litert_event_type",
        "@org_tensorflow//tensorflow/lite/experimental/litert/c:litert_logging",
        "@org_tensorflow//tensorflow/lite/experimental/litert/cc:litert_event",
    ] + select({
        "//tools/cc_target_os:android": [],
        "//conditions:default": ["//third_party/java/jdk:jni"],
    }),
)

android_library(
    name = "environment",
    srcs = ["src/main/kotlin/com/google/ai/edge/litert/Environment.kt"],
    deps = [
        ":accelerator_provider",
    ],
)

android_library(
    name = "model_config",
    srcs = ["src/main/kotlin/com/google/ai/edge/litert/acceleration/ModelConfig.kt"],
    deps = [
        ":environment",
    ],
)

android_library(
    name = "accelerator_provider",
    srcs = ["src/main/kotlin/com/google/ai/edge/litert/acceleration/AcceleratorProvider.kt"],
)

android_library(
    name = "litert_kotlin_api",
    srcs = [
        "src/main/kotlin/com/google/ai/edge/litert/AlignedBufferUtils.kt",
        "src/main/kotlin/com/google/ai/edge/litert/LiteRt.kt",
    ],
    exports_manifest = 1,
    manifest = "src/main/AndroidManifest.xml",
    visibility = ["//visibility:public"],
    exports = [
        ":accelerator_provider",
        ":environment",
        ":model_config",
    ],
    deps = [
        ":environment",
        ":litert_jni",
        ":model_config",
        "//third_party/odml/infra/ml_drift_delegate/litert:ml_drift_cl_accelerator_shared_lib",  # buildcleaner: keep
        "//litert/c:litert_runtime_c_api_shared_lib",  # buildcleaner: keep
    ],
)

# Tests

android_binary(
    name = "litert_dummy_app_for_test",
    testonly = True,
    manifest = "src/main/AndroidManifest.xml",
    deps = [":litert_kotlin_api"],
)

android_library(
    name = "litert_kotlin_api_test_lib",
    testonly = True,
    srcs = ["src/test/kotlin/com/google/ai/edge/litert/LiteRtTest.kt"],
    deps = [
        ":litert_kotlin_api",
        "//third_party/android/androidx_test/core",
        "//third_party/android/androidx_test/ext/junit",
        "//third_party/android/androidx_test/runner/android_junit_runner",
        "//third_party/java/junit:junit-android",
        "//third_party/java/truth:truth-android",
    ],
)

android_binary(
    name = "litert_test_app",
    testonly = True,
    assets = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    assets_dir = "testdata",
    instruments = ":litert_dummy_app_for_test",
    manifest = "src/test/AndroidManifest-test.xml",
    deps = [
        ":litert_kotlin_api_test_lib",
    ],
)

android_instrumentation_test(
    name = "litert_kotlin_api_emulator_test",
    target_device = "//tools/mobile/devices/android/generic_phone:android_21_x86",
    test_app = ":litert_test_app",
    test_class = "com.google.ai.edge.litert.LiteRtTest",
)

android_library(
    name = "litert_kotlin_api_mh_test_lib",
    testonly = True,
    srcs = ["src/test/kotlin/com/google/ai/edge/litert/LiteRtMhTest.kt"],
    deps = [
        ":litert_kotlin_api",
        "//third_party/android/androidx_test/core",
        "//third_party/android/androidx_test/ext/junit",
        "//third_party/android/androidx_test/runner/android_junit_runner",
        "//third_party/java/junit:junit-android",
        "//third_party/java/truth:truth-android",
    ],
)

android_binary(
    name = "litert_mh_test_app",
    testonly = True,
    assets = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    assets_dir = "testdata",
    instruments = ":litert_dummy_app_for_test",
    manifest = "src/test/AndroidManifest-test.xml",
    deps = [
        ":litert_kotlin_api_mh_test_lib",
    ],
)

# go/omnilab-ait-doc#android-logcat-decorator
android_logcat_decorator(
    name = "logcat_decorator",
    log_filter_specs = ["tflite:V"],
    log_to_file = True,
)

satellite_lab_device(
    name = "odml_test_lab_device",
    dimensions = {
        "label": "odml-test",
    },
    type = "AndroidRealDevice",
)

# blaze test --config=android_arm64 --test_output=streamed --notest_loasd --nocache_test_results \
#   //litert/kotlin:litert_kotlin_api_mh_test
android_instrumentation_test(
    name = "litert_kotlin_api_mh_test",
    timeout = "long",
    args = [
        "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
        "--run_as=odml-team",
    ],
    tags = [
        "local",
        "notap",
    ],
    target_device = ":odml_test_lab_device",
    test_app = ":litert_mh_test_app",
    test_class = "com.google.ai.edge.litert.LiteRtMhTest",
    utp_host_plugins = [
        ":logcat_decorator",
    ],
)

# MH tests for NPU
android_library(
    name = "litert_kotlin_api_npu_mh_test_lib",
    testonly = True,
    srcs = ["src/test/kotlin/com/google/ai/edge/litert/LiteRtNpuMhTest.kt"],
    deps = [
        ":litert_kotlin_api",
        "//third_party/android/androidx_test/core",
        "//third_party/android/androidx_test/ext/junit",
        "//third_party/android/androidx_test/runner/android_junit_runner",
        "//third_party/java/junit:junit-android",
        "//third_party/java/truth:truth-android",
    ],
)

android_binary(
    name = "litert_npu_mh_test_app",
    testonly = True,
    instruments = ":litert_dummy_app_for_test",
    manifest = "src/test/AndroidManifest-test.xml",
    deps = [
        ":litert_kotlin_api_npu_mh_test_lib",
        "//litert/kotlin/src/test/npu_models:qualcomm_v75",
        "//litert/vendors/qualcomm/google:qualcomm_npu_runtime_v75",
    ],
)

satellite_lab_device(
    name = "odml_test_lab_qualcomm_device",
    dimensions = {
        "label": "odml-test",
    },
    models = ["sm-s921u"],  # Samsung S24
    type = "AndroidRealDevice",
)

# blaze test --config=android_arm64 --test_output=streamed --notest_loasd --nocache_test_results \
#   //litert/kotlin:litert_kotlin_api_npu_mh_test
android_instrumentation_test(
    name = "litert_kotlin_api_npu_mh_test",
    timeout = "long",
    args = [
        "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
        "--run_as=odml-team",
    ],
    tags = [
        "local",
        "notap",
    ],
    target_device = ":odml_test_lab_qualcomm_device",
    test_app = ":litert_npu_mh_test_app",
    test_class = "com.google.ai.edge.litert.LiteRtNpuMhTest",
    utp_host_plugins = [
        ":logcat_decorator",
    ],
)
