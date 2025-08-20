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
#include "litert/cc/litert_macros.h"
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/arithmetic_graph.h"
#ifdef LITERT_TENSOR_ENABLE_TFLITE_CONVERSION

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/graph_tflite.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace litert::tensor::graph {

namespace {

inline absl::StatusOr<tflite::ActivationFunctionType> ToTflite(
    FusedActivation activation_function) {
  switch (activation_function) {
    case FusedActivation::kActNone:
      return tflite::ActivationFunctionType_NONE;
    case FusedActivation::kActRelu:
      return tflite::ActivationFunctionType_RELU;
    case FusedActivation::kActReluN1To1:
      return tflite::ActivationFunctionType_RELU_N1_TO_1;
    case FusedActivation::kActRelu6:
      return tflite::ActivationFunctionType_RELU6;
    case FusedActivation::kActTanh:
      return tflite::ActivationFunctionType_TANH;
    case FusedActivation::kActSignBit:
      return tflite::ActivationFunctionType_SIGN_BIT;
    default:
      return absl::InvalidArgumentError("Invalid activation function.");
  }
}

absl::StatusOr<tflite::Padding> ToTflite(const Padding padding) {
  switch (padding) {
    case kPaddingSame:
      return tflite::Padding_SAME;
    case kPaddingValid:
      return tflite::Padding_VALID;
    default:
      return absl::UnimplementedError("Unknown padding type.");
  }
}

}  // namespace

absl::StatusOr<TfLiteOpBuildInfo> AddOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ADD);
}

absl::StatusOr<TfLiteOpBuildInfo> MulOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MUL);
}

absl::StatusOr<TfLiteOpBuildInfo> AbsOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_ABS);
}

absl::StatusOr<TfLiteOpBuildInfo> SubOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SUB);
}

absl::StatusOr<TfLiteOpBuildInfo> DivOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DIV);
}

absl::StatusOr<TfLiteOpBuildInfo> SquareOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SQUARE);
}

absl::StatusOr<TfLiteOpBuildInfo> RsqrtOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RSQRT);
}

absl::StatusOr<TfLiteOpBuildInfo> PowOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_POW);
}

absl::StatusOr<TfLiteOpBuildInfo> NegOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_NEG);
}

absl::StatusOr<TfLiteOpBuildInfo> SqrtOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SQRT);
}

absl::StatusOr<TfLiteOpBuildInfo> TransposeOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TRANSPOSE);
}

absl::StatusOr<TfLiteOpBuildInfo> LogisticOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGISTIC);
}

absl::StatusOr<TfLiteOpBuildInfo> EmbeddingLookupOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_EMBEDDING_LOOKUP);
}

absl::StatusOr<TfLiteOpBuildInfo> DynamicUpdateSliceOperation::ToTfLite()
    const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE);
}

absl::StatusOr<TfLiteOpBuildInfo> TileOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_TILE);
}

absl::StatusOr<TfLiteOpBuildInfo> GeluOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GELU,
                           tflite::GeluOptionsT{.approximate = approximate});
}

absl::StatusOr<TfLiteOpBuildInfo> CastOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CAST);
}

absl::StatusOr<TfLiteOpBuildInfo> SelectV2Operation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SELECT_V2);
}

absl::StatusOr<TfLiteOpBuildInfo> SliceOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SLICE);
}

absl::StatusOr<TfLiteOpBuildInfo> LessOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LESS);
}

absl::StatusOr<TfLiteOpBuildInfo> GreaterOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_GREATER);
}

absl::StatusOr<TfLiteOpBuildInfo> MinimumOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MINIMUM);
}

absl::StatusOr<TfLiteOpBuildInfo> MaximumOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_MAXIMUM);
}

absl::StatusOr<TfLiteOpBuildInfo> LogicalAndOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_AND);
}

absl::StatusOr<TfLiteOpBuildInfo> LogicalOrOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_LOGICAL_OR);
}

absl::StatusOr<TfLiteOpBuildInfo> CosOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_COS);
}

absl::StatusOr<TfLiteOpBuildInfo> SinOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SIN);
}

absl::StatusOr<TfLiteOpBuildInfo> SoftmaxOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SOFTMAX,
                           tflite::SoftmaxOptionsT{.beta = beta});
}

absl::StatusOr<TfLiteOpBuildInfo> ReshapeOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_RESHAPE,
                           tflite::ReshapeOptionsT{.new_shape = new_shape});
}

absl::StatusOr<TfLiteOpBuildInfo> BatchMatMulOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_BATCH_MATMUL,
                           tflite::BatchMatMulOptionsT{
                               .adj_x = adj_x,
                               .adj_y = adj_y,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo> FullyConnectedOperation::ToTfLite() const {
  LITERT_ASSIGN_OR_RETURN(auto tflite_activation, ToTflite(activation));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_FULLY_CONNECTED,
                           tflite::FullyConnectedOptionsT{
                               .fused_activation_function = tflite_activation,
                               .keep_num_dims = keep_num_dims,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo> ConcatenationOperation::ToTfLite() const {
  LITERT_ASSIGN_OR_RETURN(auto tflite_activation, ToTflite(activation));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CONCATENATION,
                           tflite::ConcatenationOptionsT{
                               .axis = axis,
                               .fused_activation_function = tflite_activation,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo> CustomOperation::ToTfLite() const {
  TfLiteOpBuildInfo info(::tflite::BuiltinOperator_CUSTOM);
  info.custom_code = &custom_code;
  info.custom_options = &custom_options;
  return info;
}

absl::StatusOr<TfLiteOpBuildInfo> Conv2DOperation::ToTfLite() const {
  LITERT_ASSIGN_OR_RETURN(auto tflite_activation, ToTflite(activation));
  LITERT_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_CONV_2D,
                           tflite::Conv2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = stride_w,
                               .stride_h = stride_h,
                               .fused_activation_function = tflite_activation,
                               .dilation_w_factor = dilation_w_factor,
                               .dilation_h_factor = dilation_h_factor,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo> DepthwiseConv2DOperation::ToTfLite() const {
  LITERT_ASSIGN_OR_RETURN(auto tflite_activation, ToTflite(activation));
  LITERT_ASSIGN_OR_RETURN(auto tflite_padding, ToTflite(padding));
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                           tflite::DepthwiseConv2DOptionsT{
                               .padding = tflite_padding,
                               .stride_w = stride_w,
                               .stride_h = stride_h,
                               .depth_multiplier = depth_multiplier,
                               .fused_activation_function = tflite_activation,
                               .dilation_w_factor = dilation_w_factor,
                               .dilation_h_factor = dilation_h_factor,
                           });
}

absl::StatusOr<TfLiteOpBuildInfo> PadOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PAD);
}

absl::StatusOr<TfLiteOpBuildInfo> PadV2Operation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_PADV2);
}

absl::StatusOr<TfLiteOpBuildInfo> SumOperation::ToTfLite() const {
  return TfLiteOpBuildInfo(::tflite::BuiltinOperator_SUM,
                           tflite::ReducerOptionsT{.keep_dims = keep_dims});
}

}  // namespace litert::tensor::graph
#endif
