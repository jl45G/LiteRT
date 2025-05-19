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
    name = "selfie_multiclass_mtk",
    assets = glob(["src/main/assets/**"]),
    assets_dir = "src/main/assets",
    custom_package = "com.google.ai.edge.examples.image_segmentation.selfie_multiclass_mtk",
)

android_feature_module(
    name = "selfie_multiclass_mtk_ai_pack",
    custom_package = "com.google.ai.edge.examples.image_segmentation.selfie_multiclass_mtk",
    feature_name = "selfie_multiclass_mtk",
    is_asset_pack = True,
    library = ":selfie_multiclass_mtk",
    manifest = "AiPackManifest.xml",
    title = "Selfie Multiclass MTK AI Model",
)
