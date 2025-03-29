load(
    "//tools/build_defs/android:rules.bzl",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//litert/kotlin/sample_apps/LiteRTClassify:__subpackages__",
    ],
)

android_library(
    name = "test",
    srcs = ["LiteRtTestActivity.kt"],
    assets = [
        "//litert/test:testdata/simple_model.tflite",
    ],
    assets_dir = "testdata",
    manifest = "AndroidManifest.xml",
    deps = [
        "//litert/kotlin:litert_kotlin_api",
        "//litert/kotlin/sample_apps/LiteRTClassify:res",
        "//third_party/java/androidx/appcompat",
    ],
)
