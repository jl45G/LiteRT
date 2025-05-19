load("@mediapipe//mediapipe/tasks/internal/release/android:maven_repo.bzl", "maven_artifact", "maven_repository")  # buildifier: disable=bzl-visibility
load("//testing/leakr/build_defs:leakr_test.bzl", "leakr_test")

package(default_applicable_licenses = [])

LITERT_EXPERIMENTAL_VERSION = "2.0.0-alpha"

maven_artifact(
    name = "litert_kotlin_api_maven_artifact",
    src = "//litert/kotlin:litert.aar",
    artifact_deps = [
        "androidx.lifecycle:lifecycle-runtime:2.7.0",
        "com.google.guava:guava:31.0.1-android",
        "org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.6.0",
        "com.google.android.play:ai-delivery:0.1.1-alpha01",
    ],
    artifact_id = "litert",
    group_id = "com.google.ai.edge.litert",
    inception_year = "2025",
    lib_description = "The LiteRT Android API in Kotlin.",
    lib_name = "LiteRT Android API",
    lib_url = "https://ai.google.dev/edge/litert",
    version = LITERT_EXPERIMENTAL_VERSION,
)

maven_repository(
    name = "m2repository",
    srcs = [":litert_kotlin_api_maven_artifact"],
)

genmpm(
    name = "mpm",
    # Note that V1 is using "third_party/odml/litert/android". To
    # allow Rapid to distinquish between the two, we use a different
    # package name with the "v2" suffix.
    package_name = "third_party/odml/litert/android/v2",
    srcs = [":m2repository"],
    # Suppress go/ti-gcp-platform-framework-policy-presubmit.
    tags = ["TI_GCP_GREENFIELD_PLATFORM_POLICY_CHECK_EXCEPTION: OUT_OF_SCOPE:NON_BORG"],
)

leakr_test(
    name = "m2repository_leakr_test",
    badfile_dictionary = [
        "//testing/leakr/common/dictionary:badfiles.dic",
        "//third_party/odml/leakr:badfiles.dic",
        "//third_party/odml/leakr:badfiles_release_override.dic",
    ],
    dictionaries = [
        "//testing/leakr/common/dictionary:default.dic",
        "//third_party/odml/leakr:default.dic",
    ],
    failure_msg = "Oh no! Visit go/leakr!",
    ft_recipe = [
        "//testing/leakr/common/recipes:default.ftrcp",
        "//third_party/odml/leakr:filetypes.ftrcp",
    ],
    target = ":m2repository",
)
