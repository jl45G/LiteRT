load("//tools/build_defs/android:rules.bzl", "android_application", "android_library")

package(default_applicable_licenses = ["//third_party/odml:license"])

android_library(
    name = "activity",
    srcs = ["BenchmarkActivity.kt"],
    deps = [
        "//third_party/java/androidx/activity",
        "//third_party/java/androidx/activity/compose",
        "//third_party/java/androidx/compose/material3",
        "//third_party/java/androidx/compose/runtime",
        "//litert/kotlin:litert_kotlin_api",
    ],
)

android_application(
    name = "benchmark",
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.ai.edge.litert.benchmark",
        "targetSdkVersion": "35",
    },
    deps = [":activity"],
)
