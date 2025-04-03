// Copyright 2025 Google LLC.
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
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "tensorflow/lite/profiling/time.h"  // from @org_tensorflow

ABSL_FLAG(std::string, graph, "", "Model filename to use for testing.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to the dispatch library.");
ABSL_FLAG(bool, use_gpu, false, "Use GPU Accelerator.");
ABSL_FLAG(int, signature_index, 0,
          "Index of the signature to run (default: 0).");

namespace litert {
namespace {

Expected<void> RunModel() {
  if (absl::GetFlag(FLAGS_graph).empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Model filename is empty. Use --graph to provide it.");
  }

  ABSL_LOG(INFO) << "Model: " << absl::GetFlag(FLAGS_graph);
  LITERT_ASSIGN_OR_RETURN(auto model,
                          Model::CreateFromFile(absl::GetFlag(FLAGS_graph)));

  const std::string dispatch_library_dir =
      absl::GetFlag(FLAGS_dispatch_library_dir);

  std::vector<litert::Environment::Option> environment_options = {};
  if (!dispatch_library_dir.empty()) {
    environment_options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        absl::string_view(dispatch_library_dir)});
  };

  LITERT_ASSIGN_OR_RETURN(
      auto env,
      litert::Environment::Create(absl::MakeConstSpan(environment_options)));

  ABSL_LOG(INFO) << "Create CompiledModel";
  auto accelerator = absl::GetFlag(FLAGS_use_gpu) ? kLiteRtHwAcceleratorGpu
                                                  : kLiteRtHwAcceleratorNone;
  if (accelerator == kLiteRtHwAcceleratorGpu) {
    ABSL_LOG(INFO) << "Using GPU Accelerator";
  }
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          CompiledModel::Create(env, model, accelerator));

  LITERT_ASSIGN_OR_RETURN(auto signatures, model.GetSignatures());
  ABSL_LOG(INFO) << "Model has " << signatures.size() << " signature(s)";
  for (size_t i = 0; i < signatures.size(); ++i) {
    ABSL_LOG(INFO) << "Signature " << i << ":  << signatures[i]";
  }

  // Use the signature index from command line flag
  size_t signature_index = absl::GetFlag(FLAGS_signature_index);
  if (signature_index >= signatures.size()) {
    ABSL_LOG(WARNING) << "Specified signature index " << signature_index
                      << " is out of range. Using signature 0 instead.";
    signature_index = 0;
  }
  ABSL_LOG(INFO) << "Using signature index: " << signature_index;

  ABSL_LOG(INFO) << "Prepare input buffers";
  auto input_buffers_result =
      compiled_model.CreateInputBuffers(signature_index);
  if (!input_buffers_result) {
    ABSL_LOG(ERROR) << "Failed to create input buffers: "
                    << input_buffers_result.Error().Message();
    ABSL_LOG(INFO) << "Attempting to use signature 0 as fallback";
    signature_index = 0;
    input_buffers_result = compiled_model.CreateInputBuffers(signature_index);
    if (!input_buffers_result) {
      return input_buffers_result.Error();
    }
  }
  auto input_buffers = std::move(*input_buffers_result);

  ABSL_LOG(INFO) << "Prepare output buffers";
  auto output_buffers_result =
      compiled_model.CreateOutputBuffers(signature_index);
  if (!output_buffers_result) {
    ABSL_LOG(ERROR) << "Failed to create output buffers: "
                    << output_buffers_result.Error().Message();
    return output_buffers_result.Error();
  }
  auto output_buffers = std::move(*output_buffers_result);

  ABSL_LOG(INFO) << "Run model with signature index " << signature_index;
  uint64_t start = tflite::profiling::time::NowMicros();
  auto status =
      compiled_model.Run(signature_index, input_buffers, output_buffers);
  uint64_t end = tflite::profiling::time::NowMicros();
  LITERT_LOG(LITERT_INFO, "Run took %lu microseconds", end - start);

  if (!status) {
    ABSL_LOG(ERROR) << "Model run failed: " << status.Error().Message();
    return status.Error();
  }

  ABSL_LOG(INFO) << "Model run completed successfully";

  return status;
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  auto res = litert::RunModel();
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().c_str());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
