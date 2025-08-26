# Copyright 2025 Google LLC.
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

load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    deps = [
        ":buffer",
        ":datatypes",
        ":graph",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "arithmetic",
    srcs = [
        "arithmetic.cc",
        "arithmetic_tflite.cc",
    ],
    hdrs = [
        "arithmetic.h",
        "arithmetic_graph.h",
    ],
    deps = [
        ":buffer",
        ":datatypes",
        ":graph",
        ":graph_tflite",
        ":shape",
        ":tensor",
        "//litert/cc:litert_macros",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":arithmetic",
        ":buffer",
        ":datatypes",
        ":graph",
        ":tensor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":arithmetic",
        ":buffer",
        ":datatypes",
        ":graph",
        ":tensor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "datatypes",
    hdrs = ["datatypes.h"],
)

alias(
    name = "macros",
    actual = "//litert/cc:litert_macros",
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    deps = [
        ":buffer_cast",
        ":datatypes",
        "//litert/c:litert_common",
        "//litert/cc:litert_expected",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    deps = [
        ":buffer",
        ":buffer_cast",
        ":datatypes",
        ":matchers",
        "//litert/test:matchers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer_cast",
    hdrs = ["buffer_cast.h"],
)

cc_test(
    name = "buffer_cast_test",
    srcs = ["buffer_cast_test.cc"],
    deps = [
        ":buffer",
        ":buffer_cast",
        ":datatypes",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "buffer_cast_no_rtti_test",
    srcs = ["buffer_cast_test.cc"],
    defines = ["LITERT_TENSOR_USE_RTTI=0"],
    deps = [
        ":buffer",
        ":buffer_cast",
        ":datatypes",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "graph_tflite",
    hdrs = ["graph_tflite.h"],
    visibility = ["//third_party/odml/litert/tensor/examples:__pkg__"],
    deps = [
        ":graph",
        "//tflite/converter/schema:schema_fbs",
        "@com_google_absl//absl/status:statusor",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    deps = [
        ":buffer",
        ":datatypes",
        ":macros",
        "//litert/c:litert_logging",
        "//tflite/c:c_api_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "graph_test",
    srcs = ["graph_test.cc"],
    deps = [
        ":buffer",
        ":datatypes",
        ":graph",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "//testing/base/public:gunit_for_library_testonly",
    ],
)

cc_library(
    name = "tflite_flatbuffer_conversion",
    srcs = ["tflite_flatbuffer_conversion.cc"],
    hdrs = ["tflite_flatbuffer_conversion.h"],
    deps = [
        ":buffer",
        ":datatypes",
        ":graph",
        ":graph_tflite",
        ":macros",
        ":tensor",
        "//tflite:framework_stable",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs_with_mutable",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "tflite_flatbuffer_conversion_test",
    srcs = ["tflite_flatbuffer_conversion_test.cc"],
    deps = [
        ":arithmetic",
        ":buffer",
        ":datatypes",
        ":matchers",
        ":tensor",
        ":tflite_flatbuffer_conversion",
        "//tflite:framework",
        "//tflite:test_util",
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core:framework",
        "//tflite/core/kernels:builtin_ops",
        "//tflite/schema:schema_fbs_with_mutable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
