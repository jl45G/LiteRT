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
#include "third_party/odml/litert/tensor/arithmetic_graph.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"
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
  o_info.shape = a_info.shape;
  o_info.type = a_info.type;
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
    return Tensor(graph::ErrorTensor(absl::InvalidArgumentError(
        "The inner dimensions of the input tensors must match.")));
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
