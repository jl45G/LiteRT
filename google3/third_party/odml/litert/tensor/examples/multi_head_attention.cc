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

#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "third_party/odml/litert/tensor/tflite_flatbuffer_conversion.h"

ABSL_FLAG(std::string, output_path, "/tmp/mha.tflite",
          "Path to write the TFLite model.");

namespace litert::tensor {

// Implements a multi-head attention layer.
// Note: This is a simplified implementation for demonstration purposes.
// For example, it uses element-wise multiplication instead of batch matrix
// multiplication for attention scores.
Tensor MultiHeadAttention(Tensor input, int d_model, int num_heads) {
  const int d_k = d_model / num_heads;

  // --- Create weights and biases ---
  // Note: Input shape is (batch_size, seq_len, d_model)
  Tensor w_q({.name = "w_q", .type = Type::kFP32, .shape = {d_model, d_model}});
  Tensor b_q({.name = "b_q", .type = Type::kFP32, .shape = {d_model}});
  Tensor w_k({.name = "w_k", .type = Type::kFP32, .shape = {d_model, d_model}});
  Tensor b_k({.name = "b_k", .type = Type::kFP32, .shape = {d_model}});
  Tensor w_v({.name = "w_v", .type = Type::kFP32, .shape = {d_model, d_model}});
  Tensor b_v({.name = "b_v", .type = Type::kFP32, .shape = {d_model}});
  Tensor w_o({.name = "w_o", .type = Type::kFP32, .shape = {d_model, d_model}});
  Tensor b_o({.name = "b_o", .type = Type::kFP32, .shape = {d_model}});

  // --- Project Q, K, V ---
  // Output shape: (batch_size, seq_len, d_model)
  Tensor q = FullyConnected(input, w_q, b_q, kActNone, true);
  Tensor k = FullyConnected(input, w_k, b_k, kActNone, true);
  Tensor v = FullyConnected(input, w_v, b_v, kActNone, true);

  // --- Reshape and Transpose for Multi-Head ---
  const auto& input_shape = graph::GetInfo(input.GetRaw())->shape;
  const int batch_size = input_shape[0];
  const int seq_len = input_shape[1];

  // Reshape to (batch_size, seq_len, num_heads, d_k)
  q = Reshape(q, {batch_size, seq_len, num_heads, d_k});
  // Transpose to (batch_size, num_heads, seq_len, d_k)
  q = Transpose(
      q, Tensor({.type = Type::kI32,
                 .shape = {4},
                 .buffer = OwningCpuBuffer::Copy<Type::kI32>(
                     {0, 2, 1, 3})}));  // (batch, num_heads, seq_len, d_k)

  // Reshape to (batch_size, seq_len, num_heads, d_k)
  k = Reshape(k, {batch_size, seq_len, num_heads, d_k});
  // Transpose to (batch_size, num_heads, seq_len, d_k)
  k = Transpose(
      k, Tensor({.type = Type::kI32,
                 .shape = {4},
                 .buffer = OwningCpuBuffer::Copy<Type::kI32>(
                     {0, 2, 1, 3})}));  // (batch, num_heads, seq_len, d_k)

  // Reshape to (batch_size, seq_len, num_heads, d_k)
  v = Reshape(v, {batch_size, seq_len, num_heads, d_k});
  // Transpose to (batch_size, num_heads, seq_len, d_k)
  v = Transpose(
      v, Tensor({.type = Type::kI32,
                 .shape = {4},
                 .buffer = OwningCpuBuffer::Copy<Type::kI32>(
                     {0, 2, 1, 3})}));  // (batch, num_heads, seq_len, d_k)

  // --- Scaled Dot-Product Attention ---
  // Shape: (batch_size, num_heads, seq_len, d_k)
  Tensor scores = Mul(q, k);  // Simplified: element-wise
  Tensor scaled_scores =
      Div(scores, Sqrt(Tensor({.type = Type::kFP32,
                               .shape = {1},
                               .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                                   {(float)d_k})})));
  // Shape: (batch_size, num_heads, seq_len, d_k)
  Tensor attention_weights = Softmax(scaled_scores);
  // Shape: (batch_size, num_heads, seq_len, d_k)
  Tensor attention_output =
      Mul(attention_weights, v);  // Simplified: element-wise

  // --- Concatenate and Final Projection ---
  // Transpose to (batch_size, seq_len, num_heads, d_k)
  attention_output = Transpose(
      attention_output,
      Tensor({.type = Type::kI32,
              .shape = {4},
              .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, 2, 1, 3})}));
  // Reshape to (batch_size, seq_len, d_model)
  attention_output = Reshape(attention_output, {batch_size, seq_len, d_model});

  // Output shape: (batch_size, seq_len, d_model)
  Tensor output = FullyConnected(attention_output, w_o, b_o, kActNone, true);
  output = Add(input, output);  // Residual connection

  return output;
}

}  // namespace litert::tensor

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const int batch_size = 1;
  const int seq_len = 128;
  const int d_model = 256;
  const int num_heads = 4;

  litert::tensor::Tensor input({.name = "input",
                                .type = litert::tensor::Type::kFP32,
                                .shape = {batch_size, seq_len, d_model}});

  litert::tensor::Tensor output =
      litert::tensor::MultiHeadAttention(input, d_model, num_heads);
  output.SetName("output");

  litert::tensor::ModelFactory model_builder;
  if (!model_builder.AddSignature({output}, "serving_default").ok()) {
    std::cerr << "Failed to add signature." << std::endl;
    return 1;
  }

  if (!model_builder.Save(absl::GetFlag(FLAGS_output_path)).ok()) {
    std::cerr << "Failed to save model." << std::endl;
    return 1;
  }

  std::cout << "Model saved to " << absl::GetFlag(FLAGS_output_path)
            << std::endl;

  return 0;
}
