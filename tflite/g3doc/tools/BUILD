load("@org_tensorflow//tensorflow:pytype.default.bzl", "pytype_strict_binary")
load("@org_tensorflow//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "build_py_api_docs",
    srcs = ["build_py_api_docs.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

pytype_strict_binary(
    name = "build_java_api_docs",
    srcs = ["build_java_api_docs.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
