load(
    "//tools/build_defs/android:rules.bzl",
    "android_feature_module",
    "android_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//third_party/odml/litert:__subpackages__"],
)

android_library(
    name = "selfie_multiclass",
    assets = glob(["src/main/assets/**"]),
    assets_dir = "src/main/assets",
    custom_package = "com.google.ai.edge.examples.image_segmentation.selfie_multiclass",
)

android_feature_module(
    name = "selfie_multiclass_ai_pack",
    custom_package = "com.google.ai.edge.examples.image_segmentation.selfie_multiclass",
    feature_name = "selfie_multiclass",
    is_asset_pack = True,
    library = ":selfie_multiclass",
    manifest = "AiPackManifest.xml",
    title = "Selfie Multiclass AI Model",
)
