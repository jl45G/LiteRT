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
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_GRAPH_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/graph.h"

// TODO: Control this with a blaze flag.
#define LITERT_TENSOR_ENABLE_TFLITE_CONVERSION

#ifdef LITERT_TENSOR_ENABLE_TFLITE_CONVERSION

#include "third_party/odml/litert/tensor/graph_tflite.h"
#define DECL_TO_TFLITE \
  absl::StatusOr<TfLiteOpBuildInfo> ToTfLite() const override;

#else  // ifndef LITERT_TENSOR_ENABLE_TFLITE_CONVERSION

struct TfLiteOperation {};
#define DECL_TO_TFLITE

#endif  // LITERT_TENSOR_ENABLE_TFLITE_CONVERSION

namespace litert::tensor::graph {

struct AddOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Add"; }
  DECL_TO_TFLITE;
};

struct MulOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Mul"; }
  DECL_TO_TFLITE;
};

struct AbsOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Abs"; }
  DECL_TO_TFLITE;
};

struct SubOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Sub"; }
  DECL_TO_TFLITE;
};

struct DivOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Div"; }
  DECL_TO_TFLITE;
};

struct SquareOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Square"; }
  DECL_TO_TFLITE;
};

struct RsqrtOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Rsqrt"; }
  DECL_TO_TFLITE;
};

struct PowOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Pow"; }
  DECL_TO_TFLITE;
};

struct NegOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Neg"; }
  DECL_TO_TFLITE;
};

struct SqrtOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Sqrt"; }
  DECL_TO_TFLITE;
};

// Specialization of Operation graph node for softmax.
struct SoftmaxOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Softmax"; }
  float beta;
  DECL_TO_TFLITE;
};

struct BatchMatMulOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "BatchMatMul"; }
  bool adj_x;
  bool adj_y;
  DECL_TO_TFLITE;
};

struct FullyConnectedOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "FullyConnected"; }
  litert::tensor::FusedActivation activation;
  bool keep_num_dims;
  DECL_TO_TFLITE;
};

struct ConcatenationOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Concatenation"; }
  int axis;
  litert::tensor::FusedActivation activation;
  DECL_TO_TFLITE;
};

struct TransposeOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Transpose"; }
  DECL_TO_TFLITE;
};

struct TileOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Tile"; }
  DECL_TO_TFLITE;
};

struct GeluOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Gelu"; }
  bool approximate;
  DECL_TO_TFLITE;
};

struct ReshapeOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Reshape"; }
  std::vector<int> new_shape;
  DECL_TO_TFLITE;
};

struct LogisticOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Logistic"; }
  DECL_TO_TFLITE;
};

struct EmbeddingLookupOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "EmbeddingLookup"; }
  DECL_TO_TFLITE;
};

struct DynamicUpdateSliceOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "DynamicUpdateSlice"; }
  DECL_TO_TFLITE;
};

struct CustomOperation : public TfLiteOperation, virtual Operation {
  absl::string_view GetName() const override { return "Custom"; }
  std::string custom_code;
  std::vector<uint8_t> custom_options;
  DECL_TO_TFLITE;
};

#undef DECL_TO_TFLITE

}  // namespace litert::tensor::graph

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_GRAPH_H_
