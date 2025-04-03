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

load("//testing/integration/guitar/build_defs:guitar_workflow.bzl", "guitar", "guitar_workflow_test")  # copybara:strip
load("//litert/integration_test:litert_device.bzl", "litert_device_test", "litert_integration_test")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

# C++ TEST SCAFFOLD ################################################################################

cc_test(
    name = "gen_device_test",
    srcs = ["gen_device_test.cc"],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    data = ["//litert/integration_test/models:single_op_models"],
    linkopts = select({
        "@org_tensorflow//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = ["manual"],
    deps = [
        ":gen_device_test_lib",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "//litert/c:litert_logging",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/test:common",
        "//litert/test:matchers",
        "//litert/tools:dump",
    ],
)

cc_library(
    name = "gen_device_test_lib",
    testonly = True,
    hdrs = ["gen_device_test_lib.h"],
    deps = [
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/strings:string_view",
        "//litert/c:litert_common",
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_model",
        "//litert/cc:litert_tensor_buffer",
        "//litert/core/model",
        "//litert/test:matchers",
    ],
)

# ON DEVICE INTEGRATION TESTS ######################################################################

sh_binary(
    name = "run_on_device_driver_OSS",
    srcs = ["run_on_device_driver_OSS.sh"],
)

# NOTE: Everything here should be built with -c opt --config=android_arm64.

# CPU ----------------------------------------------------------------------------------------------

litert_integration_test(
    name = "single_op_device_tests_cpu",
    backend_id = "cpu",
    models = "//litert/integration_test/models:single_op_models",
)

litert_integration_test(
    name = "classic_ml_device_tests_cpu",
    backend_id = "cpu",
    models = "//litert/integration_test/models:classic_ml_models",
)

# QUALCOMM -----------------------------------------------------------------------------------------

litert_device_test(
    name = "qnn_smoketest",
    srcs = ["smoketests/qnn_smoketest.cc"],
    backend_id = "qualcomm",
    deps = [
        "@com_google_googletest//:gtest_main",
        "//litert/cc:litert_shared_library",
        "//litert/vendors/qualcomm:common",
        "//litert/vendors/qualcomm:qnn_manager",
    ],
)

litert_integration_test(
    name = "single_op_device_tests_qualcomm_JIT",
    backend_id = "qualcomm",
    models = "//litert/integration_test/models:single_op_models",
    skips = [
        "greater_f32",  # TODO: lukeboyer - Investigate (segfault).
        "less_f32",  # TODO: lukeboyer - Investigate (segfault).
    ],
)

litert_integration_test(
    name = "classic_ml_device_tests_qualcomm_JIT",
    backend_id = "qualcomm",
    models = "//litert/integration_test/models:classic_ml_models",
)

litert_integration_test(
    name = "pre_compiled_device_tests_qualcomm",
    backend_id = "qualcomm",
    models = "//litert/integration_test/models:pre_compiled_models",
)

# copybara:strip_begin(google-only)
# GOOGLE-ONLY ######################################################################################

# (doc) Running with MobileHarness -----------------------------------------------------------------

# FLAGS
#
# Executing any of the above with mobile harness requires the following flags:
# ```
# --config=android_arm64 \
# --copt="-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1" \
# --cache_test_results=no \
# --test_output=streamed \
# --notest_loasd \
# --test_arg=--allocation_exit_strategy=FAIL_FAST_NO_MATCH
# ```
#
# The following can be copied into your blazerc to set these with --config=litert_mh
# ```
# build:litert_mh --config=android_arm64
# build:litert_mh --notest_loasd
# build:litert_mh --copt="-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"
# build:litert_mh --cache_test_results=no
# build:litert_mh --test_output=streamed
# build:litert_mh --test_arg=--allocation_exit_strategy=FAIL_FAST_NO_MATCH
# build:litert_mh --compilation_mode=opt`
#```

# MH PERMISSIONS
#
# By default, mh tests will run as the user `odml-device-lab`
# https://mobileharness-fe.corp.google.com/devicelistview?filter=owner%3Aodml-device-lab
#
# If you are not included in this group, or it does not have access to a device you need, please
# reach out to the litert team.

# Guitar QUALCOMM ----------------------------------------------------------------------------------

# TODO: Figure out where unit tests should go. Finalize how the various tests are grouped.

guitar_workflow_test(
    name = "litert_guitar_device_tests_small_qualcomm",
    auto_deps = [
        ":qnn_smoketest",
    ],
    integration_test = guitar.IntegrationTest(
        tests = [
            guitar.Tests(
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
                targets = [":qnn_smoketest"],
            ),
        ],
    ),
    tags = [
        "external",
        "guitar",
        "manual",
        "notap",
    ],
)

guitar_workflow_test(
    name = "litert_guitar_device_integration_tests_qualcomm",
    auto_deps = [
        ":single_op_device_tests_qualcomm_JIT",
        ":class_ml_device_tests_qualcomm_JIT",
        ":pre_compiled_device_tests_qualcomm",
    ],
    integration_test = guitar.IntegrationTest(
        tests = [
            guitar.Tests(
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
                targets = [
                    ":single_op_device_tests_qualcomm_JIT",
                    ":class_ml_device_tests_qualcomm_JIT",
                    ":pre_compiled_device_tests_qualcomm",
                ],
            ),
        ],
    ),
    tags = [
        "external",
        "guitar",
        "manual",
        "notap",
    ],
)

# Guitar CPU ---------------------------------------------------------------------------------------

guitar_workflow_test(
    name = "litert_guitar_device_integration_tests_cpu",
    auto_deps = [
        ":single_op_device_tests_cpu",
        ":classic_ml_device_tests_cpu",
    ],
    integration_test = guitar.IntegrationTest(
        tests = [
            guitar.Tests(
                args = [
                    "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
                    "--dimension_pool=shared",  # To run on shared pool.
                    "--run_as=xeno-mh-guitar",
                ],
                blaze_flags = [
                    "--config=android_arm64",
                    "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
                    "--android_ndk_min_sdk_version=26",
                ],
                execution_method = "DISTRIBUTED_ON_BORG",
                targets = [
                    ":single_op_device_tests_cpu",
                    ":classic_ml_device_tests_cpu",
                ],
            ),
        ],
    ),
    tags = [
        "external",
        "guitar",
        "manual",
        "notap",
    ],
)

# copybara:strip_end
