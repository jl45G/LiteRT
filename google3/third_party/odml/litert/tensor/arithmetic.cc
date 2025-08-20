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

#include "third_party/odml/litert/tensor/arithmetic.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>  // NOLINT(build/c++20): needed for OSS logging.
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/optional.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic_graph.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"
#include "third_party/odml/litert/tensor/shape.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor {
namespace {

// Adds the given tensors as inputs of an op.
template <class Op, class... Tensors,
          class = std::enable_if_t<
              std::conjunction_v<std::is_same<Tensors, Tensor>...>>>
void AddInputs(std::shared_ptr<Op>& operation, Tensors&... tensors) {
  static_assert(std::is_base_of_v<graph::Operation, Op>,
                "The operation is not derived from graph::Operation.");
  operation->inputs.reserve(size(operation->inputs) + sizeof...(tensors));
  auto SetOpInput = [&operation](Tensor& t) {
    GetConsumers(t.GetRaw())->push_back(operation);
    operation->inputs.push_back(t.GetRaw());
  };
  (SetOpInput(tensors), ...);
}

template <class Op>
void AddInputs(std::shared_ptr<Op>& operation, std::vector<Tensor>& tensors) {
  operation->inputs.reserve(size(operation->inputs) + size(tensors));
  for (Tensor& t : tensors) {
    GetConsumers(t.GetRaw())->push_back(operation);
    operation->inputs.push_back(t.GetRaw());
  }
}

// Creates a new output tensor for the given op.
//
// `op_loc` should be the location of the creating operation function call.
template <class Op>
Tensor AddOutput(std::shared_ptr<Op>& operation, std::source_location op_loc) {
  static_assert(std::is_base_of_v<graph::Operation, Op>,
                "The operation is not derived from graph::Operation.");
  std::shared_ptr<graph::TensorGroup> group = operation->outputs.lock();
  if (!group) {
    group = graph::NewTensorGroup(1, std::move(op_loc));
    operation->outputs = group;
    group->producer = operation;
  } else {
    group->tensors.emplace_back();
  }
  return Tensor(graph::GetTensor(group->tensors.size() - 1, group));
}

template <class Op, class... Tensors>
Tensor ElementwiseOp(std::source_location loc, Tensor a, Tensors&... tensors) {
  auto operation = std::make_shared<Op>();
  AddInputs(operation, a, tensors...);
  Tensor output = AddOutput(operation, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());

  std::vector<std::vector<int>> shapes;
  shapes.push_back(a_info.shape);
  (
      [&](Tensor& tensor) {
        const graph::TensorInformation& tensor_info = *GetInfo(tensor.GetRaw());
        shapes.push_back(tensor_info.shape);
      }(tensors),
      ...);

  absl::StatusOr<std::vector<int>> broadcasted_shape = BroadcastShapes(shapes);
  if (!broadcasted_shape.ok()) {
    return Tensor(graph::ErrorTensor(broadcasted_shape.status()));
  }
  o_info.shape = *broadcasted_shape;
  o_info.type = a_info.type;

  bool all_types_match = true;
  (
      [&](Tensor& tensor) {
        const graph::TensorInformation& tensor_info = *GetInfo(tensor.GetRaw());
        if (tensor_info.type != a_info.type) {
          all_types_match = false;
        }
      }(tensors),
      ...);

  if (!all_types_match) {
    return Tensor(graph::ErrorTensor(absl::InvalidArgumentError(
        absl::StrCat("All tensors in an elementwise operation "
                     "must have the same type. op: ",
                     operation->GetName()))));
  }

  return output;
}

Tensor Conv2DImpl(Tensor input, Tensor filter, absl::optional<Tensor> bias,
                  int stride_h, int stride_w, Padding padding,
                  int dilation_h_factor, int dilation_w_factor,
                  FusedActivation activation, std::source_location loc) {
  auto op = std::make_shared<graph::Conv2DOperation>();
  op->stride_h = stride_h;
  op->stride_w = stride_w;
  op->padding = padding;
  op->dilation_h_factor = dilation_h_factor;
  op->dilation_w_factor = dilation_w_factor;
  op->activation = activation;
  if (bias.has_value()) {
    AddInputs(op, input, filter, *bias);
  } else {
    AddInputs(op, input, filter);
  }
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& filter_info = *GetInfo(filter.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = filter_info.shape[1];
  const int filter_w = filter_info.shape[2];
  const int output_channels = filter_info.shape[0];

  int output_h = 0;
  int output_w = 0;
  if (padding == kPaddingSame) {
    output_h = (input_h + stride_h - 1) / stride_h;
    output_w = (input_w + stride_w - 1) / stride_w;
  } else if (padding == kPaddingValid) {
    output_h = (input_h - (filter_h - 1) * dilation_h_factor) / stride_h;
    output_w = (input_w - (filter_w - 1) * dilation_w_factor) / stride_w;
  }

  output_info.shape = {input_info.shape[0], output_h, output_w,
                       output_channels};

  return output;
}

Tensor DepthwiseConv2DImpl(Tensor input, Tensor filter,
                           absl::optional<Tensor> bias, int stride_h,
                           int stride_w, Padding padding, int dilation_h_factor,
                           int dilation_w_factor, int depth_multiplier,
                           FusedActivation activation,
                           std::source_location loc) {
  auto op = std::make_shared<graph::DepthwiseConv2DOperation>();
  op->stride_h = stride_h;
  op->stride_w = stride_w;
  op->padding = padding;
  op->dilation_h_factor = dilation_h_factor;
  op->dilation_w_factor = dilation_w_factor;
  op->depth_multiplier = depth_multiplier;
  op->activation = activation;
  if (bias.has_value()) {
    AddInputs(op, input, filter, *bias);
  } else {
    AddInputs(op, input, filter);
  }
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& filter_info = *GetInfo(filter.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;

  const int input_h = input_info.shape[1];
  const int input_w = input_info.shape[2];
  const int filter_h = filter_info.shape[1];
  const int filter_w = filter_info.shape[2];
  const int output_channels = filter_info.shape[3];

  int output_h = 0;
  int output_w = 0;
  if (padding == kPaddingSame) {
    output_h = (input_h + stride_h - 1) / stride_h;
    output_w = (input_w + stride_w - 1) / stride_w;
  } else if (padding == kPaddingValid) {
    output_h = (input_h - (filter_h - 1) * dilation_h_factor) / stride_h;
    output_w = (input_w - (filter_w - 1) * dilation_w_factor) / stride_w;
  }

  output_info.shape = {input_info.shape[0], output_h, output_w,
                       output_channels};

  return output;
}
}  // namespace

Tensor Add(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::AddOperation>(loc, a, b);
}

Tensor Sub(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::SubOperation>(loc, a, b);
}

Tensor Mul(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::MulOperation>(loc, a, b);
}

Tensor Div(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::DivOperation>(loc, a, b);
}

Tensor Div(Tensor a, float b, std::source_location loc) {
  Tensor b_tensor({.type = Type::kFP32,
                     .shape = {1},
                     .buffer = OwningCpuBuffer::Copy<Type::kFP32>({b})});
  return Div(a, b_tensor, loc);
}

Tensor Abs(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::AbsOperation>(loc, a);
}

Tensor Square(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::SquareOperation>(loc, a);
}

Tensor Rsqrt(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::RsqrtOperation>(loc, a);
}

Tensor Pow(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::PowOperation>(loc, a, b);
}

Tensor Neg(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::NegOperation>(loc, a);
}

Tensor Pad(Tensor a, Tensor b, std::source_location loc) {
  auto op = std::make_shared<graph::PadOperation>();
  AddInputs(op, a, b);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  o_info.shape = a_info.shape;
  if (b_info.buffer) {
    const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
    for (int i = 0; i < o_info.shape.size(); ++i) {
      o_info.shape[i] += b_data[i * 2] + b_data[i * 2 + 1];
    }
  } else {
    return Tensor(graph::ErrorTensor(absl::InvalidArgumentError(
        "The padding tensor must have a buffer.")));
  }
  return output;
}

Tensor PadV2(Tensor a, Tensor b, Tensor c, std::source_location loc) {
  auto op = std::make_shared<graph::PadV2Operation>();
  AddInputs(op, a, b, c);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  o_info.shape = a_info.shape;
  if (b_info.buffer) {
    const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
    for (int i = 0; i < o_info.shape.size(); ++i) {
      o_info.shape[i] += b_data[i * 2] + b_data[i * 2 + 1];
    }
  } else {
    return Tensor(graph::ErrorTensor(absl::InvalidArgumentError(
        "The padding tensor must have a buffer.")));
  }
  return output;
}

Tensor Sqrt(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::SqrtOperation>(loc, a);
}

Tensor Reshape(Tensor input, std::vector<int> new_shape,
               std::source_location loc) {
  auto op = std::make_shared<graph::ReshapeOperation>();
  op->new_shape = std::move(new_shape);
  AddInputs(op, input);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = op->new_shape;
  output_info.type = input_info.type;
  if (input_info.GetSize() != output_info.GetSize()) {
    return Tensor(graph::ErrorTensor(absl::InvalidArgumentError(absl::StrCat(
        "The output size must be the same as the input size. "
        "input_size: ",
        input_info.GetSize(), " output_size: ", output_info.GetSize()))));
  }
  return output;
}

Tensor Softmax(Tensor a, float beta, std::source_location loc) {
  auto operation = std::make_shared<graph::SoftmaxOperation>();
  operation->beta = beta;
  AddInputs(operation, a);
  Tensor output = AddOutput(operation, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.shape = a_info.shape;
  o_info.type = a_info.type;
  return output;
}

Tensor Sum(Tensor a, Tensor b, bool keep_dims, std::source_location loc) {
  auto op = std::make_shared<graph::SumOperation>();
  op->keep_dims = keep_dims;
  AddInputs(op, a, b);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  const graph::TensorInformation& b_info = *GetInfo(b.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());
  o_info.type = a_info.type;
  if (b_info.buffer == nullptr) {
    return Tensor(graph::ErrorTensor(absl::InvalidArgumentError(
        "The reduction tensor must have a buffer.")));
  }
  const auto b_data = b_info.buffer->Lock().As<const int32_t>().data();
  if (op->keep_dims) {
    o_info.shape = a_info.shape;
    for (int i = 0; i < b_info.shape[0]; ++i) {
      o_info.shape[b_data[i]] = 1;
    }
  } else {
    o_info.shape = {};
    for (int i = 0; i < a_info.shape.size(); ++i) {
      bool found = false;
      for (int j = 0; j < b_info.shape[0]; ++j) {
        if (i == b_data[j]) {
          found = true;
          break;
        }
      }
      if (!found) {
        o_info.shape.push_back(a_info.shape[i]);
      }
    }
  }
  return output;
}

Tensor BatchMatMul(Tensor x, Tensor y, bool adj_x, bool adj_y,
                   std::source_location loc) {
  auto op = std::make_shared<graph::BatchMatMulOperation>();
  op->adj_x = adj_x;
  op->adj_y = adj_y;
  AddInputs(op, x, y);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& x_info = *GetInfo(x.GetRaw());
  const graph::TensorInformation& y_info = *GetInfo(y.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  std::vector<int> x_shape = x_info.shape;
  std::vector<int> y_shape = y_info.shape;
  if (adj_x) {
    std::swap(x_shape[x_shape.size() - 2], x_shape[x_shape.size() - 1]);
  }
  if (adj_y) {
    std::swap(y_shape[y_shape.size() - 2], y_shape[y_shape.size() - 1]);
  }

  if (x_shape.back() != y_shape[y_shape.size() - 2]) {
    std::string x_shape_str = absl::StrJoin(x_shape, ",");
    std::string y_shape_str = absl::StrJoin(y_shape, ",");
    return Tensor(graph::ErrorTensor(absl::InvalidArgumentError(absl::StrCat(
        "The inner dimensions of the input tensors must match. x_name: ",
        x.GetName(), " y_name: ", y.GetName(), " x_shape: ", x_shape_str,
        " y_shape: ", y_shape_str, " adj_x: ", adj_x, " adj_y: ", adj_y))));
  }

  output_info.shape.reserve(x_shape.size());
  for (size_t i = 0; i < x_shape.size() - 2; ++i) {
    output_info.shape.push_back(x_shape[i]);
  }
  output_info.shape.push_back(x_shape[x_shape.size() - 2]);
  output_info.shape.push_back(y_shape[y_shape.size() - 1]);

  output_info.type = x_info.type;
  return output;
}

Tensor FullyConnected(Tensor input, Tensor weights, Tensor bias,
                      FusedActivation activation, bool keep_num_dims,
                      std::source_location loc) {
  auto op = std::make_shared<graph::FullyConnectedOperation>();
  op->activation = activation;
  op->keep_num_dims = keep_num_dims;
  AddInputs(op, input, weights, bias);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& weights_info = *GetInfo(weights.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  if (keep_num_dims) {
    output_info.shape = input_info.shape;
    output_info.shape.back() = weights_info.shape[0];
  } else {
    output_info.shape = {input_info.shape[0], weights_info.shape[0]};
  }
  output_info.type = input_info.type;
  return output;
}

Tensor FullyConnected(Tensor input, Tensor weights, FusedActivation activation,
                      bool keep_num_dims, std::source_location loc) {
  auto op = std::make_shared<graph::FullyConnectedOperation>();
  op->activation = activation;
  op->keep_num_dims = keep_num_dims;
  AddInputs(op, input, weights);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& weights_info = *GetInfo(weights.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  if (keep_num_dims) {
    output_info.shape = input_info.shape;
    output_info.shape.back() = weights_info.shape[0];
  } else {
    output_info.shape = {input_info.shape[0], weights_info.shape[0]};
  }
  output_info.type = input_info.type;
  return output;
}

Tensor Conv2D(Tensor input, Tensor filter, Tensor bias, int stride_h,
              int stride_w, Padding padding, int dilation_h_factor,
              int dilation_w_factor, FusedActivation activation,
              std::source_location loc) {
  return Conv2DImpl(input, filter, bias, stride_h, stride_w, padding,
                    dilation_h_factor, dilation_w_factor, activation, loc);
}

Tensor Conv2D(Tensor input, Tensor filter, int stride_h, int stride_w,
              Padding padding, int dilation_h_factor, int dilation_w_factor,
              FusedActivation activation, std::source_location loc) {
  return Conv2DImpl(input, filter, absl::nullopt, stride_h, stride_w, padding,
                    dilation_h_factor, dilation_w_factor, activation, loc);
}

Tensor DepthwiseConv2D(Tensor input, Tensor filter, Tensor bias, int stride_h,
                       int stride_w, Padding padding, int dilation_h_factor,
                       int dilation_w_factor, int depth_multiplier,
                       FusedActivation activation, std::source_location loc) {
  return DepthwiseConv2DImpl(input, filter, bias, stride_h, stride_w, padding,
                             dilation_h_factor, dilation_w_factor,
                             depth_multiplier, activation, loc);
}

Tensor DepthwiseConv2D(Tensor input, Tensor filter, int stride_h, int stride_w,
                       Padding padding, int dilation_h_factor,
                       int dilation_w_factor, int depth_multiplier,
                       FusedActivation activation, std::source_location loc) {
  return DepthwiseConv2DImpl(input, filter, absl::nullopt, stride_h, stride_w,
                             padding, dilation_h_factor, dilation_w_factor,
                             depth_multiplier, activation, loc);
}

Tensor Concatenation(std::vector<Tensor> inputs, int axis,
                     FusedActivation activation, std::source_location loc) {
  auto op = std::make_shared<graph::ConcatenationOperation>();
  op->axis = axis;
  op->activation = activation;
  AddInputs(op, inputs);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& first_input_info =
      *GetInfo(inputs[0].GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = first_input_info.type;
  output_info.shape = first_input_info.shape;
  for (size_t i = 1; i < inputs.size(); ++i) {
    const graph::TensorInformation& input_info = *GetInfo(inputs[i].GetRaw());
    output_info.shape[axis] += input_info.shape[axis];
  }
  return output;
}

Tensor Transpose(Tensor input, Tensor perm, std::source_location loc) {
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& perm_info = *GetInfo(perm.GetRaw());
  ABSL_CHECK_EQ(perm_info.type, Type::kI32)
      << "Transpose only supports I32 permutation types.";
  auto op = std::make_shared<graph::TransposeOperation>();
  AddInputs(op, input, perm);
  Tensor output = AddOutput(op, loc);
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  if (perm_info.buffer) {
    const auto perm_data = perm_info.buffer->Lock().As<const int32_t>();
    const auto& input_shape = input_info.shape;
    output_info.shape.resize(input_shape.size());
    for (size_t i = 0; i < perm_data.size(); ++i) {
      output_info.shape[i] = input_shape[perm_data.data()[i]];
    }
  } else {
    // If perm is not a constant, we cannot infer the shape at this time.
    // TODO(piyu): Support dynamic shape inference.
    output_info.shape = input_info.shape;
  }
  output_info.type = input_info.type;
  return output;
}

Tensor Transpose(Tensor input, const std::vector<int>& perm,
                 std::source_location loc) {
  Tensor perm_tensor({.type = Type::kI32,
                      .shape = {static_cast<int>(perm.size())},
                      .buffer = OwningCpuBuffer::Copy<Type::kI32>(perm)});
  return Transpose(input, perm_tensor, loc);
}

Tensor Tile(Tensor input, Tensor multiples, std::source_location loc) {
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& multiples_info = *GetInfo(multiples.GetRaw());
  ABSL_CHECK_EQ(multiples_info.type, Type::kI32)
      << "Tile only supports I32 multiples types.";
  auto op = std::make_shared<graph::TileOperation>();
  AddInputs(op, input, multiples);
  Tensor output = AddOutput(op, loc);
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  if (multiples_info.buffer) {
    const auto multiples_data =
        multiples_info.buffer->Lock().As<const int32_t>();
    const auto& input_shape = input_info.shape;
    output_info.shape.resize(input_shape.size());
    for (size_t i = 0; i < multiples_data.size(); ++i) {
      output_info.shape[i] = input_shape[i] * multiples_data.data()[i];
    }
  } else {
    // If multiples is not a constant, we cannot infer the shape at this time.
    // TODO(piyu): Support dynamic shape inference.
    output_info.shape = input_info.shape;
  }
  output_info.type = input_info.type;
  return output;
}

Tensor Tile(Tensor input, const std::vector<int>& multiples,
            std::source_location loc) {
  Tensor multiples_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(multiples.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(multiples)});
  return Tile(input, multiples_tensor, loc);
}

Tensor Gelu(Tensor input, bool approximate, std::source_location loc) {
  auto op = std::make_shared<graph::GeluOperation>();
  op->approximate = approximate;
  AddInputs(op, input);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = input_info.type;
  return output;
}

Tensor Cast(Tensor input, Type to, std::source_location loc) {
  auto op = std::make_shared<graph::CastOperation>();
  op->to = to;
  AddInputs(op, input);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = to;
  return output;
}

Tensor SelectV2(Tensor condition, Tensor a, Tensor b,
                std::source_location loc) {
  auto op = std::make_shared<graph::SelectV2Operation>();
  AddInputs(op, condition, a, b);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(condition.GetRaw());
  const graph::TensorInformation& value_info = *GetInfo(a.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = input_info.shape;
  output_info.type = value_info.type;
  return output;
}

Tensor Slice(Tensor input, Tensor begin, Tensor size,
             std::source_location loc) {
  auto op = std::make_shared<graph::SliceOperation>();
  AddInputs(op, input, begin, size);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& input_info = *GetInfo(input.GetRaw());
  const graph::TensorInformation& size_info = *GetInfo(size.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.type = input_info.type;
  if (size_info.buffer) {
    const auto size_data = size_info.buffer->Lock().As<const int32_t>();
    output_info.shape.assign(size_data.begin(), size_data.end());
    for (size_t i = 0; i < output_info.shape.size(); ++i) {
      if (output_info.shape[i] == -1) {
        output_info.shape[i] = input_info.shape[i];
      }
    }
  } else {
    // If size is not a constant, we cannot infer the shape at this time.
    // TODO(b/269489748): Support dynamic shape inference.
    output_info.shape = input_info.shape;
  }
  return output;
}

Tensor Slice(Tensor input, const std::vector<int>& begin,
             const std::vector<int>& size, std::source_location loc) {
  Tensor begin_tensor({.type = Type::kI32,
                       .shape = {static_cast<int>(begin.size())},
                       .buffer = OwningCpuBuffer::Copy<Type::kI32>(begin)});
  Tensor size_tensor({.type = Type::kI32,
                      .shape = {static_cast<int>(size.size())},
                      .buffer = OwningCpuBuffer::Copy<Type::kI32>(size)});
  return Slice(input, begin_tensor, size_tensor, loc);
}

Tensor Less(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::LessOperation>(loc, a, b);
}

Tensor Greater(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::GreaterOperation>(loc, a, b);
}

Tensor Minimum(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::MinimumOperation>(loc, a, b);
}

Tensor Maximum(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::MaximumOperation>(loc, a, b);
}

Tensor LogicalAnd(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::LogicalAndOperation>(loc, a, b);
}

Tensor LogicalOr(Tensor a, Tensor b, std::source_location loc) {
  return ElementwiseOp<graph::LogicalOrOperation>(loc, a, b);
}

Tensor Cos(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::CosOperation>(loc, a);
}

Tensor Sin(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::SinOperation>(loc, a);
}

Tensor Logistic(Tensor a, std::source_location loc) {
  return ElementwiseOp<graph::LogisticOperation>(loc, a);
}

Tensor EmbeddingLookup(Tensor ids, Tensor value, std::source_location loc) {
  auto op = std::make_shared<graph::EmbeddingLookupOperation>();
  AddInputs(op, ids, value);
  Tensor output = AddOutput(op, loc);

  const graph::TensorInformation& value_info = *GetInfo(value.GetRaw());
  const graph::TensorInformation& ids_info = *GetInfo(ids.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());

  output_info.type = value_info.type;
  output_info.shape = ids_info.shape;
  output_info.shape.push_back(value_info.shape.back());

  return output;
}

Tensor EmbeddingLookup(const std::vector<int>& ids, Tensor value,
                       std::source_location loc) {
  Tensor ids_tensor({.type = Type::kI32,
                     .shape = {static_cast<int>(ids.size())},
                     .buffer = OwningCpuBuffer::Copy<Type::kI32>(ids)});
  return EmbeddingLookup(ids_tensor, value, loc);
}

Tensor DynamicUpdateSlice(Tensor operand, Tensor update, Tensor start_indices,
                          std::source_location loc) {
  auto op = std::make_shared<graph::DynamicUpdateSliceOperation>();
  AddInputs(op, operand, update, start_indices);
  Tensor output = AddOutput(op, loc);
  const graph::TensorInformation& operand_info = *GetInfo(operand.GetRaw());
  graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
  output_info.shape = operand_info.shape;
  output_info.type = operand_info.type;
  return output;
}

Tensor DynamicUpdateSlice(Tensor operand, Tensor update,
                          const std::vector<int>& start_indices,
                          std::source_location loc) {
  Tensor start_indices_tensor(
      {.type = Type::kI32,
       .shape = {static_cast<int>(start_indices.size())},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(start_indices)});
  return DynamicUpdateSlice(operand, update, start_indices_tensor, loc);
}

std::vector<Tensor> Custom(std::vector<Tensor> inputs, std::string custom_code,
                           std::vector<uint8_t> custom_options,
                           const std::vector<std::vector<int>>& output_shapes,
                           const std::vector<Type>& output_types,
                           std::source_location loc) {
  auto op = std::make_shared<graph::CustomOperation>();
  op->custom_code = std::move(custom_code);
  op->custom_options = std::move(custom_options);
  for (auto& input : inputs) {
    AddInputs(op, input);
  }

  std::vector<Tensor> outputs;
  outputs.reserve(output_shapes.size());
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    Tensor output = AddOutput(op, loc);
    graph::TensorInformation& output_info = *GetInfo(output.GetRaw());
    output_info.shape = output_shapes[i];
    output_info.type = output_types[i];
    outputs.push_back(output);
  }
  return outputs;
}

}  // namespace litert::tensor
