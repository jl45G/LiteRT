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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

pixel_9_targets = [
    "//litert/cc:litert_compiled_model_gpu_test",
    "//litert/cc:litert_compiled_model_npu_test",
    "//litert/runtime/dispatch:dispatch_delegate_google_tensor_test",
]

qualcomm_targets = [
    "//litert/runtime/dispatch:dispatch_delegate_qualcomm_test",
]

# guitar run -w //litert/google/mobile_tests:litert_guitar_workflow -v citc:LOCAL
guitar_workflow_test(
    name = "litert_guitar_workflow",
    auto_deps = [
        "//litert/kotlin:litert_kotlin_api_mh_test",
    ] + pixel_9_targets + qualcomm_targets,
    integration_test = guitar.IntegrationTest(tests = [
        guitar.Tests(
            # Pixel 9 on ODML test lab:
            args = [
                "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
                "--dimension_label=odml-test",  # To run on ODML test lab.
                "--dimension_model=\"pixel 9\"",
            ],
            blaze_flags = [
                "--config=android_arm64",
                "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
                "--android_ndk_min_sdk_version=26",
            ],
            execution_method = "DISTRIBUTED_ON_BORG",
            targets = pixel_9_targets,
        ),
        # Qualcomm test, sm-s928u1 on shared pool:
        guitar.Tests(
            args = [
                "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
                "--dimension_pool=shared",  # To run on shared pool.
                "--dimension_model=\"sm-s928u1\"",
            ],
            blaze_flags = [
                "--config=android_arm64",
                "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
                "--android_ndk_min_sdk_version=26",
            ],
            execution_method = "DISTRIBUTED_ON_BORG",
            targets = qualcomm_targets,
        ),
        # Kotlin API test:
        guitar.Tests(
            args = [
                "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
                "--dimension_label=odml-test",  # To run on ODML test lab.
                "--run_as=xeno-mh-guitar",
            ],
            blaze_flags = [
                "--config=android_arm64",
                "--android_ndk_min_sdk_version=26",
            ],
            execution_method = "DISTRIBUTED_ON_BORG",
            targets = [
                "//litert/kotlin:litert_kotlin_api_mh_test",
                "//litert/kotlin:litert_kotlin_api_npu_mh_test",
            ],
        ),
    ]),
    tags = [
        "external",
        "guitar",
        "manual",
        "notap",
    ],
)
