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

load("//devtools/python/blaze:strict.bzl", "py_strict_library", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.google.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "compiled_model_wrapper_lib",
    srcs = ["compiled_model_wrapper.cc"],
    hdrs = [
        "compiled_model_wrapper.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_common",
        "//litert/c:litert_tensor_buffer",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/core:framework",
    ],
)

pybind_extension(
    name = "_pywrap_litert_compiled_model_wrapper",
    srcs = [
        "compiled_model_wrapper_pybind11.cc",
    ],
    hdrs = ["compiled_model_wrapper.h"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "_pywrap_litert_compiled_model_wrapper.pyi",
    ],
    deps = [
        ":compiled_model_wrapper_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

py_strict_library(
    name = "compiled_model",
    srcs = [
        "compiled_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//litert/python/compiled_model_wrapper:_pywrap_litert_compiled_model_wrapper",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "compiled_model_test",
    srcs = ["compiled_model_test.py"],
    data = [
        "testdata/simple_model_float.tflite",
        "testdata/simple_model_int.tflite",
    ],
    # Static linking is required because this loads a cc_binary as a shared
    # library, which would otherwise create ODR violations.
    # copybara:uncomment linking_mode = "static",
    tags = [
        "no_oss",  # TODO(b/190842754): Enable test in OSS.
    ],
    deps = [
        ":compiled_model",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
    ],
)

OUT_DIR = "$(RULEDIR)"

CONVERTER = "@org_tensorflow//tensorflow/compiler/mlir/lite:tf_tfl_translate"

CMD = """
for mlir_file in $(SRCS); do
    $(location {converter}) --input-mlir $$mlir_file --o={out_dir}/testdata/$$(basename $$mlir_file .mlir).tflite
done
""".format(
    converter = CONVERTER,
    out_dir = OUT_DIR,
)

genrule(
    name = "mlir_test_data",
    srcs = glob(["testdata/*.mlir"]),
    outs = [s.removesuffix(".mlir") + ".tflite" for s in glob(["testdata/*.mlir"])],
    cmd = CMD,
    tools = [CONVERTER],
)
