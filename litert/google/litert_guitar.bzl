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

"""Macros for LiteRT Guitar tests."""

load("//testing/integration/guitar/build_defs:guitar_workflow.bzl", "guitar")

# TODO(b/406280789): Integrate this into litert_device.bzl.
def litert_pixel_9_mh_guitar_test(targets):
    return guitar.Tests(
        args = [
            "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
            "--dimension_label=odml-test",  # To run on ODML test lab.
            "--dimension_model=\"pixel 9\"",
            "--run_as=xeno-mh-guitar",
        ],
        blaze_flags = [
            "--config=android_arm64",
            "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
            "--android_ndk_min_sdk_version=26",
        ],
        execution_method = "DISTRIBUTED_ON_BORG",
        targets = targets,
    )

def litert_qualcomm_mh_guitar_test(targets):
    return guitar.Tests(
        args = [
            "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
            "--dimension_pool=shared",  # To run on shared pool.
            "--dimension_model=\"sm-s928u1\"",
            "--run_as=xeno-mh-guitar",
        ],
        blaze_flags = [
            "--config=android_arm64",
            "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
            "--android_ndk_min_sdk_version=26",
        ],
        execution_method = "DISTRIBUTED_ON_BORG",
        targets = targets,
    )

def litert_cpu_mh_guitar_test(targets):
    return guitar.Tests(
        args = [
            "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
            "--dimension_label=odml-test",  # To run on ODML test lab.
            "--run_as=xeno-mh-guitar",
        ],
        blaze_flags = [
            "--config=android_arm64",
            "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
            "--android_ndk_min_sdk_version=26",
        ],
        execution_method = "DISTRIBUTED_ON_BORG",
        targets = targets,
    )
