load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "codegen_cpp",
    srcs = ["codegen_cpp.py"],
    deps = [
        "//litert/python/tools/model_utils",
        "//third_party/py/xdsl",
        "@org_tensorflow//third_party/py/numpy:headers",
    ],
)

pytype_strict_binary(
    name = "cli",
    srcs = ["cli.py"],
    deps = [
        ":codegen_cpp",
        "//third_party/py/fire",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
