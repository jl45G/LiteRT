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
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_GRAPH_TFLITE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_GRAPH_TFLITE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/graph.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace litert::tensor::graph {

struct TfLiteOpBuildInfo {
  ::tflite::BuiltinOperator builtin_code;
  std::optional<::tflite::BuiltinOptionsUnion> builtin_options = std::nullopt;

  // Present only when builtin_code is BuiltinOperator_CUSTOM.
  const std::string* custom_code = nullptr;
  const std::vector<uint8_t>* custom_options = nullptr;

  template <typename OpCodeT>
  explicit TfLiteOpBuildInfo(OpCodeT code)
      : builtin_code(static_cast<tflite::BuiltinOperator>(code)) {}

  template <typename OpCodeT, typename OpOptionsT>
  explicit TfLiteOpBuildInfo(OpCodeT code, OpOptionsT&& options)
      : builtin_code(static_cast<tflite::BuiltinOperator>(code)),
        builtin_options(::tflite::BuiltinOptionsUnion()) {
    builtin_options->Set(std::forward<OpOptionsT>(options));
  }
};

// Base class for operations that defines conversion to TfLite flatbuffer.
class TfLiteOperation : virtual public graph::Operation {
 public:
  virtual absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const = 0;
};

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_GRAPH_TFLITE_H_
