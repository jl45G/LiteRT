load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/llvm/llvm-project/mlir/test/python:util.bzl", "lit_py_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

lit_py_tests(
    srcs = glob(["*.filecheck.py"]),
    data = ["//third_party/llvm/llvm-project/mlir/test:lit_data"],
    driver = "//third_party/llvm/llvm-project/mlir:run_lit",
    python = ":py_test_interpreter",
)

py_binary(
    name = "py_test_interpreter",
    testonly = 1,
    srcs = ["_py_bin.py"],
    exec_properties = {"py_par.mem": "28g"},
    launcher = "//devtools/python/launcher:python_interpreter",
    main = "_py_bin.py",
    python_version = "PY3",
    deps = [
        ":passes",  # buildcleaner: keep
        "//third_party/py/absl:app",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)

pytype_strict_library(
    name = "passes",
    srcs = ["passes.py"],
    visibility = ["//litert:__subpackages__"],
    deps = ["//litert/python/google/tools/model_utils"],
)
