# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    assets = [
        "src/main/assets/model#group_other/selfie_multiclass.tflite",
        "src/main/assets/model#group_qti_v73/selfie_multiclass.tflite",
        "src/main/assets/model#group_qti_v75/selfie_multiclass.tflite",
        "src/main/assets/model#group_qti_v79/selfie_multiclass.tflite",
    ],
    assets_dir = "src/main/assets",
    custom_package = "com.google.aiedge.examples.image_segmentation.selfie_multiclass",
)

android_feature_module(
    name = "selfie_multiclass_ai_pack",
    custom_package = "com.google.aiedge.examples.image_segmentation.selfie_multiclass",
    feature_name = "selfie_multiclass_ai_pack",
    is_asset_pack = True,
    library = ":selfie_multiclass",
    manifest = "AiPackManifest.xml",
    title = "Selfie Multiclass AI Model",
)
