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

load("//litert/build_common:litert_build_defs.bzl", "copy_file", "export_lrt_runtime_only_linkopt", "export_lrt_runtime_only_script")

# Experimental libraries & runtimes -- DO NOT use or depend on these.
package(
    default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"],
    default_visibility = ["//litert:litert_internal_users"],
)

# odml:google3-begin(INTERNAL_PHYSICAL_MOBILE_TESTING_INFRA)
# OpenCL Accelerator module for LiteRT. The Accelerator is not bundled with the LiteRT runtime.
# Adding this module to the LiteRT runtime will enable OpenCL support via Accelerator API.
# Static version of the OpenCL Accelerator module for LiteRT.
cc_library(
    name = "ml_drift_cl_accelerator",
    srcs = ["ml_drift_cl_accelerator.cc"],
    hdrs = [
    ],
    defines = [
        "LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR",
    ],
    visibility = [
        "//litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
        "//third_party/odml/litert:__subpackages__",
        "//third_party/odml/litert_lm/runtime:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_accelerator",
        "//litert/c:litert_accelerator_registration",
        "//litert/c:litert_common",
        "//litert/c:litert_logging",
        "//litert/c:litert_opaque_options",
        "//litert/c/options:litert_gpu_options",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_options",
        "//litert/cc/options:accelerator_options",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
        "//tflite/core/c:private_common",
        "//third_party/odml/infra/ml_drift_delegate:ml_drift_cl_litert",
    ],
    alwayslink = 1,
)

# This module requires the shared library of LiteRT runtime `libLiteRtRuntimeCApi.so`.
cc_library(
    name = "ml_drift_cl_accelerator_runtimecapi",
    srcs = ["ml_drift_cl_accelerator.cc"],
    hdrs = [
    ],
    visibility = [
        "//litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
        "//third_party/odml/litert:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/c/options:litert_gpu_options",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_options",
        "//litert/cc/options:accelerator_options",
        "//third_party/odml/infra/ml_drift_delegate:ml_drift_cl_litert_runtimecapi",
    ],
)

cc_shared_library(
    name = "ml_drift_cl_accelerator_so",
    additional_linker_inputs = export_lrt_runtime_only_script(),
    features = select({
        # Allow unresolved symbols which will be defined in the executable. There are no
        # linker flags to allow unresolved symbols only of a given pattern like LiteRt*.
        "//litert/c:resolve_symbols_in_exec": ["-no_undefined"],
        "//conditions:default": [],
    }),
    shared_lib_name = "libLiteRtGpuAccelerator.so",
    user_link_flags = export_lrt_runtime_only_linkopt() + select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }) + ["-Wl,-soname=libLiteRtGpuAccelerator.so"],
    deps = [
        ":ml_drift_cl_accelerator_runtimecapi",
    ],
)

cc_library(
    name = "ml_drift_cl_accelerator_shared_lib",
    srcs = [":ml_drift_cl_accelerator_so"],
    linkstatic = 1,
    visibility = [
        "//litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
        "//third_party/odml/litert:__subpackages__",
    ],
)

# Copies the shared library so that it is available for use in test data as libLiteRtDispatch_Qualcomm.so.
copy_file(
    name = "copy_ml_drift_cl_accelerator_so",
    src = ":ml_drift_cl_accelerator_so",
    target = "libLiteRtGpuAccelerator.so",
    visibility = [
        "//litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
        "//third_party/odml/litert:__subpackages__",
    ],
)
# odml:google3-end

cc_library(
    name = "ml_drift_tensor_conversion",
    srcs = ["ml_drift_tensor_conversion.cc"],
    hdrs = ["ml_drift_tensor_conversion.h"],
    deps = [
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_element_type",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_tensor_buffer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:tensor",
        "@ml_drift//ml_drift/common:data_type",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:status",
        "@ml_drift//ml_drift/common/task:tensor_desc",
    ],
)

cc_test(
    name = "ml_drift_tensor_conversion_test",
    srcs = ["ml_drift_tensor_conversion_test.cc"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        # TODO: b/401125945 - fix and enable asan.
        "noasan",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":ml_drift_tensor_conversion",
        # TODO: b/401125945 - fix and enable heapcheck.
        "@com_google_googletest//:gtest_main_no_heapcheck",
        "@ml_drift//ml_drift/cl:environment",
        "@ml_drift//ml_drift/cl:opencl_wrapper",
        "@ml_drift//ml_drift/common:shape",
        "@ml_drift//ml_drift/common:tensor",
        "@ml_drift//ml_drift/common/task:tensor_desc",
        "//litert/c:litert_any",
        "//litert/c:litert_environment",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_any",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_model",
        "//litert/cc/internal:litert_tensor_buffer",
        "//litert/cc/internal:litert_environment",
        "//litert/core:environment",
        "//litert/test:matchers",
        "//tflite/kernels:test_util",
    ],
)
