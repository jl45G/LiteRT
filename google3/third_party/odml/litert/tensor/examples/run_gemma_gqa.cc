/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tflite/c/c_api_types.h"

namespace {

void RunWithCpu(const std::string& model_path) {
  std::cout << "--- Running on CPU ---" << std::endl;

  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto model,
                         litert::Model::CreateFromFile(model_path));
  LITERT_ASSIGN_OR_ABORT(auto options, litert::Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

  LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                         litert::CompiledModel::Create(env, model, options));

  LITERT_ASSIGN_OR_ABORT(auto input_buffers,
                         compiled_model.CreateInputBuffers());
  LITERT_ASSIGN_OR_ABORT(auto output_buffers,
                         compiled_model.CreateOutputBuffers());

  // Fill input tensors with random data.
  for (int i = 0; i < input_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_ABORT(auto ranked_tensor_type,
                           model.GetInputTensorType(0, i));
    size_t num_elements = 1;
    for (int dim : ranked_tensor_type.Layout().Dimensions()) {
      num_elements *= dim;
    }
    switch (static_cast<TfLiteType>(ranked_tensor_type.ElementType())) {
      case kTfLiteFloat32: {
        std::vector<float> input_data(num_elements);
        std::generate(input_data.begin(), input_data.end(), std::rand);
        LITERT_ABORT_IF_ERROR(
            input_buffers[i].Write(absl::MakeConstSpan(input_data)));
        break;
      }
      case kTfLiteInt32: {
        std::vector<int> input_data(num_elements);
        std::generate(input_data.begin(), input_data.end(), std::rand);
        LITERT_ABORT_IF_ERROR(
            input_buffers[i].Write(absl::MakeConstSpan(input_data)));
        break;
      }
      case kTfLiteInt8: {
        std::vector<int8_t> input_data(num_elements);
        std::generate(input_data.begin(), input_data.end(), std::rand);
        LITERT_ABORT_IF_ERROR(
            input_buffers[i].Write(absl::MakeConstSpan(input_data)));
        break;
      }
      default:
        std::cout << "Unsupported data type." << std::endl;
    }
  }

  LITERT_ABORT_IF_ERROR(compiled_model.Run(input_buffers, output_buffers));

  // Print output shapes.
  for (int i = 0; i < output_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_ABORT(auto ranked_tensor_type,
                           model.GetOutputTensorType(0, i));
    std::cout << "Output " << i << " shape: ["
              << absl::StrJoin(ranked_tensor_type.Layout().Dimensions(), ", ")
              << "]" << std::endl;
  }

  std::cout << "CPU execution successful." << std::endl;
}

void RunWithGpu(const std::string& model_path) {
  std::cout << "--- Running on GPU ---" << std::endl;

  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto model,
                         litert::Model::CreateFromFile(model_path));
  LITERT_ASSIGN_OR_ABORT(auto gpu_options, litert::GpuOptions::Create());
  LITERT_ASSIGN_OR_ABORT(auto options, litert::Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu |
                                  kLiteRtHwAcceleratorCpu);
  options.AddOpaqueOptions(std::move(gpu_options));

  LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                         litert::CompiledModel::Create(env, model, options));

  LITERT_ASSIGN_OR_ABORT(auto input_buffers,
                         compiled_model.CreateInputBuffers());
  LITERT_ASSIGN_OR_ABORT(auto output_buffers,
                         compiled_model.CreateOutputBuffers());

  // Fill input tensors with random data.
  for (int i = 0; i < input_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_ABORT(auto ranked_tensor_type,
                           model.GetInputTensorType(0, i));
    size_t num_elements = 1;
    for (int dim : ranked_tensor_type.Layout().Dimensions()) {
      num_elements *= dim;
    }
    switch (static_cast<TfLiteType>(ranked_tensor_type.ElementType())) {
      case kTfLiteFloat32: {
        std::vector<float> input_data(num_elements);
        std::generate(input_data.begin(), input_data.end(), std::rand);
        LITERT_ABORT_IF_ERROR(
            input_buffers[i].Write(absl::MakeConstSpan(input_data)));
        break;
      }
      case kTfLiteInt32: {
        std::vector<int> input_data(num_elements);
        std::generate(input_data.begin(), input_data.end(), std::rand);
        LITERT_ABORT_IF_ERROR(
            input_buffers[i].Write(absl::MakeConstSpan(input_data)));
        break;
      }
      case kTfLiteInt8: {
        std::vector<int8_t> input_data(num_elements);
        std::generate(input_data.begin(), input_data.end(), std::rand);
        LITERT_ABORT_IF_ERROR(
            input_buffers[i].Write(absl::MakeConstSpan(input_data)));
        break;
      }
      default:
        std::cout << "Unsupported data type." << std::endl;
    }
  }

  LITERT_ABORT_IF_ERROR(compiled_model.Run(input_buffers, output_buffers));

  // Print output shapes.
  for (int i = 0; i < output_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_ABORT(auto ranked_tensor_type,
                           model.GetOutputTensorType(0, i));
    std::cout << "Output " << i << " shape: ["
              << absl::StrJoin(ranked_tensor_type.Layout().Dimensions(), ", ")
              << "]" << std::endl;
  }

  std::cout << "GPU execution successful." << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [use_gpu]" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  bool use_gpu = false;
  if (argc > 2 && std::string(argv[2]) == "use_gpu") {
    use_gpu = true;
  }

  if (use_gpu) {
    RunWithGpu(model_path);
  } else {
    RunWithCpu(model_path);
  }

  return 0;
}
