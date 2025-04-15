"""Definitions for py_wheel platforms."""

load("//third_party/bazel_skylib/lib:selects.bzl", "selects")
load("//litert:litert.bzl", "MANYLINUX_LEVEL")

def get_wheel_platform_name():
    selects.config_setting_group(
        name = "linux_x86_64",
        match_all = [
            "//third_party/bazel_platforms/os:linux",
            "//third_party/bazel_platforms/cpu:x86_64",
        ],
    )

    selects.config_setting_group(
        name = "linux_arm64",
        match_all = [
            "//third_party/bazel_platforms/os:linux",
            "//third_party/bazel_platforms/cpu:arm64",
        ],
    )

    selects.config_setting_group(
        name = "macos_arm64",
        match_all = [
            "//third_party/bazel_platforms/os:macos",
            "//third_party/bazel_platforms/cpu:arm64",
        ],
    )

    return select({
        ":linux_x86_64": MANYLINUX_LEVEL + "_x86_64",
        ":linux_arm64": MANYLINUX_LEVEL + "_arm64",
        ":macos_arm64": "macosx_12_0_arm64",
        "//conditions:default": "none",
    })
