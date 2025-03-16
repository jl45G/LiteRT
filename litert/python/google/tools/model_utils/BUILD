load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

package_group(
    name = "model_utils_users",
    packages = [
        "//litert/python/...",
        "//litert/python/tools/model_utils/...",
    ],
)

py_library(
    name = "model_utils",
    srcs = glob(
        include = ["**/*.py"],
        exclude = [
            "tool/**/*",
            "test/**/*",
        ],
    ),
    visibility = [":model_utils_users"],
    deps = [
        "//pyglib:gfile",
        "//third_party/llvm/llvm-project/mlir:MLIRBindingsPythonHeaders",
        "//litert/python/google/tools/model_utils/core/tblgen",
        "//third_party/py/jax",
        "//third_party/py/mlir",
        "//third_party/py/mlir:arithmetic_dialect",
        "//third_party/py/mlir:builtin_dialect",
        "//third_party/py/mlir:chlo_dialect",
        "//third_party/py/mlir:func_dialect",
        "//third_party/py/mlir:ir",
        "//third_party/py/mlir:math_dialect",
        "//third_party/py/mlir:pass_manager",
        "//third_party/py/mlir:stablehlo_dialect",
        "//third_party/py/mlir/_mlir_libs:_mlirRegisterEverything",
        "//third_party/py/model_explorer:model_explorer_lib",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/xdsl",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/integrations:model_utils_core_pybind",
        "@org_tensorflow//tensorflow/compiler/tf2xla/python:xla",
        "@org_tensorflow//tensorflow/lite/tools:flatbuffer_utils",
    ],
)
