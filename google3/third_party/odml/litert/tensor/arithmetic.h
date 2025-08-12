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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_H_

#include <cstdint>
#include <source_location>  // NOLINT(build/c++20): needed for OSS logging.
#include <string>
#include <vector>

#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor {
// Possible fused activation functions.
typedef enum {
  kActNone = 0,
  kActRelu,
  kActReluN1To1,  // min(max(-1, x), 1)
  kActRelu6,      // min(max(0, x), 6)
  kActTanh,
  kActSignBit,
  kActSigmoid,
} FusedActivation;

Tensor Add(Tensor a, Tensor b,
           std::source_location loc = std::source_location::current());
Tensor Sub(Tensor a, Tensor b,
           std::source_location loc = std::source_location::current());
Tensor Mul(Tensor a, Tensor b,
           std::source_location loc = std::source_location::current());
Tensor Div(Tensor a, Tensor b,
           std::source_location loc = std::source_location::current());
Tensor Div(Tensor a, float b,
           std::source_location loc = std::source_location::current());
Tensor Abs(Tensor a,
           std::source_location loc = std::source_location::current());
Tensor Square(Tensor a,
              std::source_location loc = std::source_location::current());
Tensor Rsqrt(Tensor a,
             std::source_location loc = std::source_location::current());
Tensor Pow(Tensor a, Tensor b,
           std::source_location loc = std::source_location::current());
Tensor Neg(Tensor a,
           std::source_location loc = std::source_location::current());
Tensor Sqrt(Tensor a,
            std::source_location loc = std::source_location::current());

Tensor Reshape(Tensor input, std::vector<int> new_shape,
               std::source_location loc = std::source_location::current());

Tensor Softmax(Tensor a, float beta = 1,
               std::source_location loc = std::source_location::current());

Tensor BatchMatMul(Tensor x, Tensor y, bool adj_x = false, bool adj_y = false,
                   std::source_location loc = std::source_location::current());

Tensor FullyConnected(
    Tensor input, Tensor weights, Tensor bias,
    FusedActivation activation = kActNone, bool keep_num_dims = false,
    std::source_location loc = std::source_location::current());

Tensor FullyConnected(
    Tensor input, Tensor weights, FusedActivation activation = kActNone,
    bool keep_num_dims = false,
    std::source_location loc = std::source_location::current());

Tensor Concatenation(
    std::vector<Tensor> inputs, int axis = 0,
    FusedActivation activation = kActNone,
    std::source_location loc = std::source_location::current());

Tensor Transpose(Tensor input, Tensor perm,
                 std::source_location loc = std::source_location::current());

Tensor Transpose(Tensor input, const std::vector<int>& perm,
                 std::source_location loc = std::source_location::current());

Tensor Tile(Tensor input, Tensor multiples,
            std::source_location loc = std::source_location::current());

Tensor Tile(Tensor input, const std::vector<int>& multiples,
            std::source_location loc = std::source_location::current());

Tensor Gelu(Tensor input, bool approximate = false,
            std::source_location loc = std::source_location::current());

Tensor Logistic(Tensor a,
                std::source_location loc = std::source_location::current());

Tensor EmbeddingLookup(
    Tensor ids, Tensor value,
    std::source_location loc = std::source_location::current());

Tensor EmbeddingLookup(
    const std::vector<int>& ids, Tensor value,
    std::source_location loc = std::source_location::current());

Tensor DynamicUpdateSlice(
    Tensor operand, Tensor update, Tensor start_indices,
    std::source_location loc = std::source_location::current());

Tensor DynamicUpdateSlice(
    Tensor operand, Tensor update, const std::vector<int>& start_indices,
    std::source_location loc = std::source_location::current());

std::vector<Tensor> Custom(
    std::vector<Tensor> inputs, std::string custom_code,
    std::vector<uint8_t> custom_options,
    const std::vector<std::vector<int>>& output_shapes,
    const std::vector<Type>& output_types,
    std::source_location loc = std::source_location::current());
}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_ARITHMETIC_H_
