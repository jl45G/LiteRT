load("//litert/build_common:litert_build_defs.bzl", "copy_file", "export_lrt_runtime_only_linkopt", "export_lrt_runtime_only_script")

# Experimental libraries & runtimes -- DO NOT use or depend on these.
package(default_applicable_licenses = ["//devtools/compliance/licenses:no_external_contributions"])

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
        ":accelerator_options",
        ":accelerator_options_payload",
        "//litert/c:litert_accelerator",
        "//litert/c:litert_accelerator_registration",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_macros",
        "//tflite:framework",
        "//tflite/c:c_api_types",
        "//tflite/core:framework",
        "//tflite/core/c:private_common",
        "//third_party/odml/infra/ml_drift_delegate:ml_drift_cl_litert",
    ],
    alwayslink = 1,
)

cc_library(
    name = "accelerator_options",
    srcs = ["accelerator_options.cc"],
    hdrs = ["accelerator_options.h"],
    visibility = [
        "//litert:__subpackages__",
        "//tflite/experimental/litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
        "//third_party/odml/litert:__subpackages__",
        "//third_party/odml/litert_lm/runtime:__subpackages__",
    ],
    deps = [
        ":accelerator_options_payload",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_handle",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_opaque_options",
    ],
)

cc_test(
    name = "accelerator_options_test",
    srcs = ["accelerator_options_test.cc"],
    deps = [
        ":accelerator_options",
        ":accelerator_options_payload",
        "//litert/c:litert_common",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "accelerator_options_payload",
    srcs = ["accelerator_options_payload.cc"],
    hdrs = ["accelerator_options_payload.h"],
    visibility = [
        "//litert:__subpackages__",
        "//tflite/experimental/litert:__subpackages__",
        "//third_party/odml/infra:__subpackages__",
        "//third_party/odml/litert:__subpackages__",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "accelerator_options_payload_test",
    srcs = ["accelerator_options_payload_test.cc"],
    deps = [
        ":accelerator_options_payload",
        "//litert/c:litert_common",
        "//litert/c:litert_opaque_options",
        "//litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
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
        ":accelerator_options",
        ":accelerator_options_payload",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/cc:litert_macros",
        "//third_party/odml/infra/ml_drift_delegate:ml_drift_cl_litert_runtimecapi",
    ],
)

cc_shared_library(
    name = "ml_drift_cl_accelerator_so",
    additional_linker_inputs = export_lrt_runtime_only_script(),
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
        "//litert/cc:litert_tensor_buffer",
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
        "//litert/c:litert_environment",
        "//litert/c:litert_model",
        "//litert/c:litert_tensor_buffer",
        "//litert/c:litert_tensor_buffer_types",
        "//litert/cc:litert_any",
        "//litert/cc:litert_layout",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//litert/core:environment",
        "//tflite/kernels:test_util",
    ],
)
