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

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "third_party/odml/litert/tensor/tflite_flatbuffer_conversion.h"

ABSL_FLAG(std::string, output_path, "/tmp/gemma_attention.tflite",
          "Path to write the TFLite model.");

namespace litert::tensor {

// Implements the Gemma 2.0 attention layer.
Tensor GemmaAttention(Tensor input, int d_model, int num_heads,
                      int num_kv_heads) {
  const int head_dim = d_model / num_heads;

  // --- Create learnable parameters ---
  // Input shape: (batch_size, seq_len, d_model)
  Tensor w_q({.name = "w_q",
              .type = Type::kI8,
              .shape = {num_heads * head_dim, d_model},
              .quantization = std::make_unique<graph::QuantizationParameters>(
                  graph::QuantizationParameters{
                      .scales = {1.0f}, .zero_points = {0}})});
  Tensor b_q(
      {.name = "b_q", .type = Type::kFP32, .shape = {num_heads * head_dim}});
  Tensor w_k({.name = "w_k",
              .type = Type::kI8,
              .shape = {num_kv_heads * head_dim, d_model},
              .quantization = std::make_unique<graph::QuantizationParameters>(
                  graph::QuantizationParameters{
                      .scales = {1.0f}, .zero_points = {0}})});
  Tensor b_k(
      {.name = "b_k", .type = Type::kFP32, .shape = {num_kv_heads * head_dim}});
  Tensor w_v({.name = "w_v",
              .type = Type::kI8,
              .shape = {num_kv_heads * head_dim, d_model},
              .quantization = std::make_unique<graph::QuantizationParameters>(
                  graph::QuantizationParameters{
                      .scales = {1.0f}, .zero_points = {0}})});
  Tensor b_v(
      {.name = "b_v", .type = Type::kFP32, .shape = {num_kv_heads * head_dim}});
  Tensor w_o({.name = "w_o",
              .type = Type::kI8,
              .shape = {num_heads * head_dim, d_model},
              .quantization = std::make_unique<graph::QuantizationParameters>(
                  graph::QuantizationParameters{
                      .scales = {1.0f}, .zero_points = {0}})});
  Tensor b_o({.name = "b_o", .type = Type::kFP32, .shape = {d_model}});

  // --- Project Q, K, V ---
  // Q shape: (batch_size, seq_len, num_heads * head_dim)
  Tensor q = FullyConnected(input, w_q, kActNone, true);
  q = Add(q, b_q);
  // K shape: (batch_size, seq_len, num_kv_heads * head_dim)
  Tensor k = FullyConnected(input, w_k, kActNone, true);
  k = Add(k, b_k);
  // V shape: (batch_size, seq_len, num_kv_heads * head_dim)
  Tensor v = FullyConnected(input, w_v, kActNone, true);
  v = Add(v, b_v);

  // --- Reshape and Transpose for Multi-Head ---
  const auto& input_shape = graph::GetInfo(input.GetRaw())->shape;
  const int batch_size = input_shape[0];
  const int seq_len = input_shape[1];

  // Reshape Q to (batch_size, seq_len, num_heads, head_dim)
  q = Reshape(q, {batch_size, seq_len, num_heads, head_dim});
  // Transpose Q to (batch_size, num_heads, seq_len, head_dim)
  q = Transpose(q, {0, 2, 1, 3});

  // Reshape K to (batch_size, seq_len, num_kv_heads, head_dim)
  k = Reshape(k, {batch_size, seq_len, num_kv_heads, head_dim});
  // Transpose K to (batch_size, num_kv_heads, seq_len, head_dim)
  k = Transpose(k, {0, 2, 1, 3});

  // Reshape V to (batch_size, seq_len, num_kv_heads, head_dim)
  v = Reshape(v, {batch_size, seq_len, num_kv_heads, head_dim});
  // Transpose V to (batch_size, num_kv_heads, seq_len, head_dim)
  v = Transpose(v, {0, 2, 1, 3});

  // --- Grouped-Query Attention ---
  const int num_groups = num_heads / num_kv_heads;
  if (num_groups > 1) {
    // Repeat K and V to match the number of query heads
    // K, V shape: (batch_size, num_heads, seq_len, head_dim)
    k = Tile(k, {1, num_groups, 1, 1});
    v = Tile(v, {1, num_groups, 1, 1});
  }

  // --- Scaled Dot-Product Attention ---
  // Q * K^T
  // Q shape: (batch_size, num_heads, seq_len, head_dim)
  // K shape: (batch_size, num_heads, seq_len, head_dim)
  // scores shape: (batch_size, num_heads, seq_len, seq_len)
  Tensor scores = BatchMatMul(q, k, /*adj_x=*/false, /*adj_y=*/true);
  Tensor scaled_scores = Div(scores, std::sqrt(static_cast<float>(head_dim)));
  // attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
  Tensor attention_weights = Softmax(scaled_scores);
  // attention_output shape: (batch_size, num_heads, seq_len, head_dim)
  Tensor attention_output = BatchMatMul(attention_weights, v);

  // --- Concatenate and Final Projection ---
  // Transpose to (batch_size, seq_len, num_heads, head_dim)
  attention_output = Transpose(attention_output, {0, 2, 1, 3});
  // Reshape to (batch_size, seq_len, d_model)
  attention_output = Reshape(attention_output, {batch_size, seq_len, d_model});

  // Output shape: (batch_size, seq_len, d_model)
  Tensor output = FullyConnected(attention_output, w_o, kActNone, true);
  output = Add(output, b_o);
  output = Add(input, output);  // Residual connection

  return output;
}

}  // namespace litert::tensor

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const int batch_size = 1;
  const int seq_len = 128;
  const int d_model = 256;
  const int num_heads = 8;
  const int num_kv_heads = 2;

  litert::tensor::Tensor input({.name = "input",
                                .type = litert::tensor::Type::kFP32,
                                .shape = {batch_size, seq_len, d_model}});

  litert::tensor::Tensor output =
      litert::tensor::GemmaAttention(input, d_model, num_heads, num_kv_heads);
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
