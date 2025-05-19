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

#ifndef ODML_LITERT_LITERT_GOOGLE_INVOKE_QUALCOMM_UTIL_H_
#define ODML_LITERT_LITERT_GOOGLE_INVOKE_QUALCOMM_UTIL_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/runtime/dispatch/dispatch_opaque_options.h"
#include "litert/test/common.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"

namespace litert::tools {

using ::litert::Environment;
using ::litert::internal::FlatbufferWrapper;
using ::litert::internal::GetMetadata;
using ::litert::testing::TflRuntime;
using ::tflite::FlatBufferModel;
using ::tflite::Interpreter;

static constexpr absl::string_view kToolName = "INVOKE_MODEL";
static constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

inline litert::Expected<Environment> CreateDefaultEnvironment() {
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  return litert::Environment::Create(absl::MakeConstSpan(environment_options));
}

inline litert::Expected<Options> CreateDispatchOptions(const uint8_t* base) {
  LITERT_ASSIGN_OR_RETURN(auto dispatch_options,
                          internal::DispatchDelegateOptions::Create());
  LITERT_RETURN_IF_ERROR(dispatch_options.SetAllocBase(base));
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  LITERT_RETURN_IF_ERROR(options.AddOpaqueOptions(std::move(dispatch_options)));
  return options;
}

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_GOOGLE_INVOKE_QUALCOMM_UTIL_H_
