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
# ==============================================================================

# Wasm binding for the LiterT tensor library.

load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
load("//third_party/bazel_rules/rules_wasm/wasm:defs.bzl", "wasm_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "litert_tensor_wasm_cc",
    srcs = ["wasm_api.cc"],
    linkopts = [
        "-sEXPORT_ALL=1",
    ],
    # doesn't build without emscripten support, is tested by the Wasm build
    # rule, see go/wasm-cc-binary-tags
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tflite:framework",
        "//third_party/emscripten:embind",
        "//third_party/odml/litert/tensor",
        "//third_party/odml/litert/tensor:arithmetic",
        "//third_party/odml/litert/tensor:buffer",
        "//third_party/odml/litert/tensor:datatypes",
        "//third_party/odml/litert/tensor:tflite_flatbuffer_conversion",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# To build:
# blaze build --config=wasm --features=emscripten_wasm_threads //third_party/odml/litert/tensor/wasm:litert_tensor_wasm
wasm_cc_binary(
    name = "litert_tensor_wasm",
    cc_target = ":litert_tensor_wasm_cc",
)

filegroup(
    name = "litert_tensor_wasm_files",
    srcs = [
        "litert_tensor_wasm/litert_tensor_wasm.js",
        "litert_tensor_wasm/litert_tensor_wasm.wasm",
        "//third_party/odml/litert/tensor/wasm/demo:demo_files",
    ],
)
