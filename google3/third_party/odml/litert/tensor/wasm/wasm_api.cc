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
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <emscripten/val.h>

#include <cstdio>
#include <vector>

#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "third_party/odml/litert/tensor/tflite_flatbuffer_conversion.h"

using litert::tensor::Abs;
using litert::tensor::Add;
using litert::tensor::BatchMatMul;
using litert::tensor::Concatenation;
using litert::tensor::Div;
using litert::tensor::DynamicUpdateSlice;
using litert::tensor::EmbeddingLookup;
using litert::tensor::FullyConnected;
using litert::tensor::FusedActivation;
using litert::tensor::Gelu;
using litert::tensor::Logistic;
using litert::tensor::Mul;
using litert::tensor::Neg;
using litert::tensor::OwningCpuBuffer;
using litert::tensor::Pow;
using litert::tensor::Reshape;
using litert::tensor::Rsqrt;
using litert::tensor::Run;
using litert::tensor::Softmax;
using litert::tensor::Sqrt;
using litert::tensor::Square;
using litert::tensor::Sub;
using litert::tensor::Tensor;
using litert::tensor::TensorInit;
using litert::tensor::Tile;
using litert::tensor::Transpose;
using litert::tensor::Type;

Tensor createTensor(emscripten::val init) {
  TensorInit tensor_init;
  tensor_init.name = init["name"].as<std::string>();
  tensor_init.type = init["type"].as<Type>();
  tensor_init.shape = emscripten::vecFromJSArray<int>(init["shape"]);

  // Get the JS buffer object (e.g., ArrayBuffer or Uint8Array)
  std::vector<float> buffer_val =
      emscripten::vecFromJSArray<float>(init["buffer"]);

  // Log the initial values
  printf("Creating tensor: %s\n", tensor_init.name.c_str());
  printf("  Shape: ");
  for (int dim : tensor_init.shape) {
    printf("%d ", dim);
  }
  printf("\n");
  printf("  Buffer (first 10 values): ");
  for (int i = 0; i < 10 && i < buffer_val.size(); ++i) {
    printf("%f ", buffer_val[i]);
  }
  printf("\n");

  // 2. Move the vector's data into an OWNING buffer.
  //    This avoids a second copy and safely transfers ownership to the buffer.
  //    (This assumes OwningCpuBuffer has a constructor that takes
  //    std::vector&&)
  tensor_init.buffer = OwningCpuBuffer::Copy<Type::kFP32>(buffer_val);

  return Tensor(tensor_init);
}

EMSCRIPTEN_BINDINGS(my_module) {
  emscripten::function("createTensor", &createTensor);

  emscripten::class_<Tensor>("Tensor")
      .constructor<>()
      .function("setName",
                emscripten::optional_override(
                    [](Tensor& self, absl::string_view name) -> Tensor& {
                      return self.SetName(std::string(name));
                    }),
                emscripten::allow_raw_pointers())
      .function("getName", &Tensor::GetName)
      .function(
          "setBuffer",
          emscripten::optional_override(
              [](Tensor& self, std::shared_ptr<litert::tensor::Buffer> buffer) {
                self.SetBuffer(buffer);
              }),
          emscripten::allow_raw_pointers())
      .function(
          "getBuffer",
          emscripten::optional_override([](const Tensor& self)
                                            -> emscripten::val {
            auto status_or_buffer = self.GetBuffer();
            if (!status_or_buffer.ok()) {
              std::cout << "Failed to get buffer: " << status_or_buffer.status()
                        << "\n";
              return emscripten::val::null();
            }
            auto* data = status_or_buffer.value().Lock().data();
            auto info = litert::tensor::graph::GetInfo(self.GetRaw());
            size_t num_elements = 1;
            for (int dim : info->shape) {
              num_elements *= dim;
            }

            // Log the buffer values
            printf("Getting buffer for tensor: %s\n",
                   std::string(self.GetName()).c_str());
            printf("  Shape: ");
            for (int dim : info->shape) {
              printf("%d ", dim);
            }
            printf("\n");
            printf("  Buffer (first 10 values): ");
            if (info->type == Type::kFP32) {
              const float* float_data = reinterpret_cast<const float*>(data);
              for (int i = 0; i < 10 && i < num_elements; ++i) {
                printf("%f ", float_data[i]);
              }
            } else if (info->type == Type::kI32) {
              const int32_t* int_data = reinterpret_cast<const int32_t*>(data);
              for (int i = 0; i < 10 && i < num_elements; ++i) {
                printf("%d ", int_data[i]);
              }
            }
            printf("\n");

            if (info->type == Type::kFP32) {
              return emscripten::val(emscripten::typed_memory_view(
                  num_elements, reinterpret_cast<const float*>(data)));
            } else if (info->type == Type::kI32) {
              return emscripten::val(emscripten::typed_memory_view(
                  num_elements, reinterpret_cast<const int32_t*>(data)));
            }
            std::cout << "Unsupported type: " << info->type << "\n";
            return emscripten::val::null();
          }),
          emscripten::allow_raw_pointers())
      .function("getStatus", &Tensor::GetStatus)
      .function("getRaw", emscripten::optional_override([](const Tensor& self) {
                  return self.GetRaw();
                }),
                emscripten::allow_raw_pointers());

  emscripten::value_object<TensorInit>("TensorInit")
      .field("name", &TensorInit::name)
      .field("type", &TensorInit::type)
      .field("shape", &TensorInit::shape)
      .field("buffer", &TensorInit::buffer);

  emscripten::enum_<Type>("Type")
      .value("kUnknown", Type::kUnknown)
      .value("kFP32", Type::kFP32)
      .value("kI32", Type::kI32);

  emscripten::enum_<FusedActivation>("FusedActivation")
      .value("kActNone", FusedActivation::kActNone)
      .value("kActRelu", FusedActivation::kActRelu)
      .value("kActReluN1To1", FusedActivation::kActReluN1To1)
      .value("kActRelu6", FusedActivation::kActRelu6)
      .value("kActTanh", FusedActivation::kActTanh)
      .value("kActSignBit", FusedActivation::kActSignBit)
      .value("kActSigmoid", FusedActivation::kActSigmoid);

  emscripten::function("add",
                       emscripten::optional_override(
                           [](Tensor a, Tensor b) { return Add(a, b); }));
  emscripten::function("sub",
                       emscripten::optional_override(
                           [](Tensor a, Tensor b) { return Sub(a, b); }));
  emscripten::function("mul",
                       emscripten::optional_override(
                           [](Tensor a, Tensor b) { return Mul(a, b); }));
  emscripten::function("div",
                       emscripten::optional_override(
                           [](Tensor a, Tensor b) { return Div(a, b); }));
  emscripten::function("divf",
                       emscripten::optional_override(
                           [](Tensor a, float b) { return Div(a, b); }));
  emscripten::function(
      "abs", emscripten::optional_override([](Tensor a) { return Abs(a); }));
  emscripten::function("square", emscripten::optional_override(
                                     [](Tensor a) { return Square(a); }));
  emscripten::function("rsqrt", emscripten::optional_override(
                                    [](Tensor a) { return Rsqrt(a); }));
  emscripten::function("pow",
                       emscripten::optional_override(
                           [](Tensor a, Tensor b) { return Pow(a, b); }));
  emscripten::function(
      "neg", emscripten::optional_override([](Tensor a) { return Neg(a); }));
  emscripten::function(
      "sqrt", emscripten::optional_override([](Tensor a) { return Sqrt(a); }));
  emscripten::function("reshape",
                       emscripten::optional_override(
                           [](Tensor a, const std::vector<int>& new_shape) {
                             return Reshape(a, new_shape);
                           }));
  emscripten::function(
      "softmax", emscripten::optional_override(
                     [](Tensor a, float beta) { return Softmax(a, beta); }));
  emscripten::function("batchMatMul",
                       emscripten::optional_override(
                           [](Tensor a, Tensor b, bool adj_x, bool adj_y) {
                             return BatchMatMul(a, b, adj_x, adj_y);
                           }));
  emscripten::function(
      "fullyConnected",
      emscripten::optional_override([](Tensor input, Tensor weights,
                                       Tensor bias, FusedActivation activation,
                                       bool transpose_a) {
        return FullyConnected(input, weights, bias, activation, transpose_a);
      }));
  emscripten::function(
      "fullyConnectedNoBias",
      emscripten::optional_override([](Tensor input, Tensor weights,
                                       FusedActivation activation,
                                       bool transpose_a) {
        return FullyConnected(input, weights, activation, transpose_a);
      }));
  emscripten::function(
      "concatenation",
      emscripten::optional_override(
          [](std::vector<Tensor> inputs, int axis, FusedActivation activation) {
            return Concatenation(inputs, axis, activation);
          }));
  emscripten::function("transpose",
                       emscripten::optional_override(
                           [](Tensor input, const std::vector<int>& perm) {
                             return Transpose(input, perm);
                           }));
  emscripten::function("tile",
                       emscripten::optional_override(
                           [](Tensor input, const std::vector<int>& multiples) {
                             return Tile(input, multiples);
                           }));
  emscripten::function(
      "gelu", emscripten::optional_override([](Tensor a, bool approximate) {
        return Gelu(a, approximate);
      }));
  emscripten::function("logistic", emscripten::optional_override(
                                       [](Tensor a) { return Logistic(a); }));
  emscripten::function("embeddingLookup",
                       emscripten::optional_override([](Tensor a, Tensor b) {
                         return EmbeddingLookup(a, b);
                       }));
  emscripten::function(
      "dynamicUpdateSlice",
      emscripten::optional_override([](Tensor operand, Tensor update,
                                       const std::vector<int>& start_indices) {
        return DynamicUpdateSlice(operand, update, start_indices);
      }));

  emscripten::function(
      "run", emscripten::optional_override([](std::vector<Tensor> outputs) {
        absl::Status status = Run(outputs);
        if (!status.ok()) {
          return emscripten::val(status.ToString());
        }
        return emscripten::val::null();
      }));

  emscripten::register_vector<Tensor>("VectorTensor");
};
