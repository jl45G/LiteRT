// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_dispatch_delegate.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/build_stamp.h"
#include "litert/google/invoke_qualcomm_util.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/test/common.h"
#include "litert/tools/outstream.h"
#include "litert/tools/tool_display.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/profiling/time.h"

// Tool for running an arbitrary tflite w/ npu bytecode model through
// dispatch delegate.

ABSL_FLAG(std::string, model, "", "Model resulting from 'apply plugin'.");
ABSL_FLAG(std::string, err, "--", "Where to send error logs.");

namespace litert::tools {
namespace {

using ::tflite::FlatBufferModel;
using ::tflite::Interpreter;

TEST(InvokeModel, Run) {
  const std::string model_path = absl::GetFlag(FLAGS_model);
  const std::string err = absl::GetFlag(FLAGS_err);

  auto display =
      std::make_unique<ToolDisplay>(UserStream::MakeFromFlag(err), kToolName);
  auto& disp = *display;
  DumpPreamble(disp);
  auto setup_scope = disp.StartS("Setup");
  disp.Labeled() << absl::StreamFormat("MODEL_PATH: %s\n", model_path);

  // Load model and interpreter.
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(model_path);
  auto runtime =
      TflRuntime::CreateFromFlatBuffer(std::move(flatbuffer.Value()));
  ABSL_CHECK(runtime) << "Could not setup runtime";
  auto& rt = **runtime;
  auto& interp = rt.Interpreter();

  // auto env = CreateDefaultEnvironment();
  LITERT_ASSIGN_OR_ABORT(auto env, CreateDefaultEnvironment());
  litert::internal::ExternalLiteRtBufferContext buffer_context(env.Get());
  interp.SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  disp.Labeled() << absl::StreamFormat("Loaded a model of size: %lu\n",
                                       rt.Flatbuffer().Buf().Size());
  disp.Labeled() << absl::StreamFormat(
      "Created interpreter with %lu subgraphs, %lu inputs and %lu outputs\n",
      interp.subgraphs_size(), interp.inputs().size(), interp.outputs().size());

  {
    // Check model is compatible.

    auto tag_scope = disp.StartS("Checking build tag");
    auto build_tag_buf =
        GetMetadata(internal::kLiteRtBuildStampKey, *rt.Flatbuffer().Unpack());
    ABSL_CHECK(build_tag_buf) << "Could not find build tag in metadata\n";
    auto build_stamp = internal::ParseBuildStamp(*build_tag_buf);
    ABSL_CHECK(build_stamp) << "Could not parse build stamp\n";
    auto [man, model] = *build_stamp;
    disp.Labeled() << absl::StreamFormat("\n\tSOC_MAN: %s\n\tSOC_MODEL: %s\n",
                                         man, model);
  }

  // Make delegate.

  LITERT_ASSIGN_OR_ABORT(auto env_options, env.GetOptions());
  LITERT_ASSIGN_OR_ABORT(auto options,
                         CreateDispatchOptions(rt.Flatbuffer().Buf().Data()));

  auto dispatch_delegate = CreateDispatchDelegatePtr(
      std::move(env_options.Get()), std::move(options.Get()));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif
  auto invoke_scope = disp.StartS("Invoking model with npu dispatch"); // NOLINT

  ASSERT_EQ(rt.Interpreter().ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);
  ASSERT_EQ(rt.Interpreter().AllocateTensors(), kTfLiteOk);
  uint64_t start = tflite::profiling::time::NowMicros();
  ASSERT_EQ(rt.Interpreter().Invoke(), kTfLiteOk);
  uint64_t end = tflite::profiling::time::NowMicros();
  LITERT_LOG(LITERT_INFO, "Invoke took %lu microseconds", end - start);
}
}  // namespace

}  // namespace litert::tools
