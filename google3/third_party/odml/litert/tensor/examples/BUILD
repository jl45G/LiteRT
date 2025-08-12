load(
    "//litert/build_common:special_rule.bzl",
    "gl_native_deps",
    "gles_deps",
    "gles_linkopts",
)
load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
)

cc_binary(
    name = "run_mha_cc",
    srcs = ["run_mha_cc.cc"],
    data = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }) + gles_linkopts(),
    tags = [],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/options:litert_gpu_options",
        "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + gles_deps() + gl_native_deps(),
)

cc_binary(
    name = "multi_head_attention",
    srcs = ["multi_head_attention.cc"],
    deps = [
        "//third_party/odml/litert/tensor",
        "//third_party/odml/litert/tensor:arithmetic",
        "//third_party/odml/litert/tensor:buffer",
        "//third_party/odml/litert/tensor:datatypes",
        "//third_party/odml/litert/tensor:graph",
        "//third_party/odml/litert/tensor:tflite_flatbuffer_conversion",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "gemma_attention",
    srcs = ["gemma_attention.cc"],
    deps = [
        "//third_party/odml/litert/tensor",
        "//third_party/odml/litert/tensor:arithmetic",
        "//third_party/odml/litert/tensor:buffer",
        "//third_party/odml/litert/tensor:datatypes",
        "//third_party/odml/litert/tensor:graph",
        "//third_party/odml/litert/tensor:tflite_flatbuffer_conversion",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "run_gemma_gqa",
    srcs = ["run_gemma_gqa.cc"],
    data = [
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator_so",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }) + gles_linkopts(),
    tags = [],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_model",
        "//litert/cc:litert_options",
        "//litert/cc:litert_tensor_buffer",
        "//litert/cc/options:litert_gpu_options",
        "//litert/runtime/accelerators/gpu:ml_drift_cl_accelerator",
        "//tflite/c:c_api_types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + gles_deps() + gl_native_deps(),
)
