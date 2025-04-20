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

load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_binary")

pytype_strict_binary(
    name = "image_segmentation",
    srcs = ["image_segmentation.py"],
    data = [
        "testdata/deeplab_v3.tflite",
        "testdata/image.jpg",
        "testdata/selfie_multiclass_256x256.tflite",
    ],
    # Static linking is required because this loads a cc_binary as a shared
    # library, which would otherwise create ODR violations.
    # copybara:uncomment linking_mode = "static",
    deps = [
        "//litert/python/litert_wrapper/compiled_model_wrapper:compiled_model",
        "//litert/python/litert_wrapper/tensor_buffer_wrapper:tensor_buffer",
        "//third_party/py/PIL:pil",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/tensorflow",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow/python/platform:resource_loader",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)
