# Copyright 2025 Google LLC.
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

load("//testing/integration/guitar/build_defs:guitar_workflow.bzl", "guitar", "guitar_workflow_test")
load("//litert/google:litert_guitar.bzl", "litert_cpu_mh_guitar_test", "litert_pixel_9_mh_guitar_test", "litert_qualcomm_mh_guitar_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

pixel_9_targets = [
    "//litert/cc:litert_compiled_model_gpu_test",
    "//litert/cc:litert_compiled_model_npu_test",
    "//litert/runtime/dispatch:dispatch_delegate_google_tensor_test",
]

qualcomm_targets = [
    "//litert/runtime/dispatch:dispatch_delegate_qualcomm_test",
]

kotlin_targets = [
    "//litert/kotlin/src/tests/google:litert_kotlin_api_mh_test",
    "//litert/kotlin/src/tests/google:litert_kotlin_api_npu_mh_test",
]

# guitar run -w //litert/google/mobile_tests:litert_guitar_workflow -v citc:LOCAL
guitar_workflow_test(
    name = "litert_guitar_workflow",
    auto_deps = pixel_9_targets + qualcomm_targets + kotlin_targets,
    integration_test = guitar.IntegrationTest(tests = [
        litert_pixel_9_mh_guitar_test(pixel_9_targets),
        # Qualcomm test, sm-s928u1 on shared pool:
        litert_qualcomm_mh_guitar_test(qualcomm_targets),
        # Kotlin API test:
        litert_cpu_mh_guitar_test(kotlin_targets),
    ]),
    tags = [
        "external",
        "guitar",
        "manual",
        "notap",
    ],
)
