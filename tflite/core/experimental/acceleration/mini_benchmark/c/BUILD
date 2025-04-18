# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
load("//tflite/experimental/acceleration/mini_benchmark:build_defs.bzl", "cc_library_with_forced_in_process_benchmark_variant")
load("//tflite/experimental/acceleration/mini_benchmark:special_rules.bzl", "libjpeg_handle_deps", "minibenchmark_visibility_allowlist")

default_visibility_group = [
    "//tflite/experimental/acceleration/mini_benchmark:__subpackages__",
] + minibenchmark_visibility_allowlist()

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = default_visibility_group,
    licenses = ["notice"],
)

cc_library_with_forced_in_process_benchmark_variant(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    in_process_deps = [
        "//tflite/experimental/acceleration/mini_benchmark:blocking_validator_runner",
    ],
    deps = [
        ":c_api_types",
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/acceleration/configuration/c:delegate_plugin",
        "//tflite/experimental/acceleration/mini_benchmark:benchmark_result_evaluator",
        "//tflite/experimental/acceleration/mini_benchmark:status_codes",
        "//tflite/experimental/acceleration/mini_benchmark:validator_runner_entrypoint",
        "//tflite/experimental/acceleration/mini_benchmark:validator_runner_options",
        "@flatbuffers",
    ],
)

cc_test(
    name = "c_api_test",
    srcs = ["c_api_test.cc"],
    deps = [
        ":c_api",
        "//tflite/acceleration/configuration:configuration_fbs",
        "//tflite/experimental/acceleration/mini_benchmark:embedded_mobilenet_model",
        "//tflite/experimental/acceleration/mini_benchmark:embedded_mobilenet_validation_model",
        "//tflite/experimental/acceleration/mini_benchmark:embedded_simple_addition_model",
        "//tflite/experimental/acceleration/mini_benchmark:mini_benchmark_test_helper",
        "//tflite/experimental/acceleration/mini_benchmark:status_codes",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@flatbuffers//:runtime_cc",
    ] + libjpeg_handle_deps(),
)

cc_library(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    visibility = ["//visibility:private"],
)
