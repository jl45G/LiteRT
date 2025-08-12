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

#include "third_party/odml/litert/tensor/tflite_flatbuffer_conversion.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/matchers.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/kernels/register.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"
#include "tflite/schema/mutable/schema_generated.h"
#include "tflite/test_util.h"

namespace litert::tensor {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::IsNull;
using ::testing::Lt;
using ::testing::Not;
using ::testing::Pointee;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::testing::UnorderedElementsAre;
using ::testing::litert::AlignmentIs;
using ::testing::status::IsOk;
using ::tflite::TfLiteArrayIs;

// Retrieves the tensor with the given `name`.
//
// `tensors` must be the list of tensor indices to search in
// `interpreter.tensors()`. Usually this is either `interpreter.inputs()` or
// `interpreter.outputs()`.
absl::StatusOr<TfLiteTensor&> GetTensor(const absl::string_view name,
                                        tflite::Interpreter& interpreter,
                                        const std::vector<int>& tensors) {
  if (name.empty()) {
    return absl::InvalidArgumentError("No name to look up.");
  }
  for (int t : tensors) {
    TfLiteTensor* tensor = interpreter.tensor(t);
    if (tensor && tensor->name == name) {
      return *tensor;
    }
  }
  return absl::NotFoundError("Named input tensor not found");
}

// Retrieves the input tensor with the given `name` in the `interpreter`.
absl::StatusOr<TfLiteTensor&> GetInputTensor(const absl::string_view name,
                                             tflite::Interpreter& interpreter) {
  return GetTensor(name, interpreter, interpreter.inputs());
}

// Retrieves the output tensor with the given `name` in the `interpreter`.
absl::StatusOr<TfLiteTensor&> GetOutputTensor(
    const absl::string_view name, tflite::Interpreter& interpreter) {
  return GetTensor(name, interpreter, interpreter.outputs());
}

TEST(SerializationTest, BuildOneSubgraphAndRunIt) {
  {
    Tensor a({.name = "a", .type = Type::kI32, .shape = {3, 3}});
    Tensor b({.name = "b", .type = Type::kI32, .shape = {3, 3}});
    Tensor c({.name = "c", .type = Type::kI32, .shape = {3, 3}});
    Tensor d = Mul(a, b);
    Tensor e = Add(c, d);
    d.SetName("d");
    e.SetName("e");
    ASSERT_THAT(Save({e, d}, "/tmp/fma.tflite"), IsOk());
  }

  auto model = tflite::FlatBufferModel::BuildFromFile("/tmp/fma.tflite");
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->inputs().size(), 3);
  ASSERT_EQ(interpreter->outputs().size(), 2);

  {  // We are scoping these tests because these references may go stale when
    // calling `Invoke`.
    ASSERT_OK_AND_ASSIGN(TfLiteTensor & a, GetInputTensor("a", *interpreter));
    ASSERT_OK_AND_ASSIGN(TfLiteTensor & b, GetInputTensor("b", *interpreter));
    ASSERT_OK_AND_ASSIGN(TfLiteTensor & c, GetInputTensor("c", *interpreter));

    EXPECT_EQ(a.type, kTfLiteInt32);
    EXPECT_THAT(a.dims, TfLiteArrayIs({3, 3}));

    EXPECT_EQ(b.type, kTfLiteInt32);
    EXPECT_THAT(b.dims, TfLiteArrayIs({3, 3}));

    EXPECT_EQ(c.type, kTfLiteInt32);
    EXPECT_THAT(c.dims, TfLiteArrayIs({3, 3}));

    {  // Setting input 0.
      const int32_t input_data_ref[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      std::memcpy(reinterpret_cast<int32_t*>(a.data.data), input_data_ref,
                  sizeof(input_data_ref));
    }
    {  // Setting input 1.
      const int32_t input_data_ref[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
      std::memcpy(reinterpret_cast<int32_t*>(b.data.data), input_data_ref,
                  sizeof(input_data_ref));
    }
    {  // Setting input 2.
      const int32_t input_data_ref[9] = {2, 2, 2, 2, 2, 2, 2, 2, 2};
      std::memcpy(reinterpret_cast<int32_t*>(c.data.data), input_data_ref,
                  sizeof(input_data_ref));
    }
  }

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  {
    ASSERT_OK_AND_ASSIGN(TfLiteTensor & d, GetOutputTensor("d", *interpreter));
    ASSERT_OK_AND_ASSIGN(TfLiteTensor & e, GetOutputTensor("e", *interpreter));

    EXPECT_EQ(d.type, kTfLiteInt32);
    EXPECT_THAT(d.dims, TfLiteArrayIs({3, 3}));

    EXPECT_EQ(e.type, kTfLiteInt32);
    EXPECT_THAT(e.dims, TfLiteArrayIs({3, 3}));

    {  // Checking output 0.
      absl::Span<const int32_t> output_data(
          reinterpret_cast<int32_t*>(d.data.data), 9);
      EXPECT_THAT(output_data, ElementsAre(9, 16, 21, 24, 25, 24, 21, 16, 9));
    }
    {  // Checking output 1.
      absl::Span<const int32_t> output_data(
          reinterpret_cast<int32_t*>(e.data.data), 9);
      EXPECT_THAT(output_data, ElementsAre(11, 18, 23, 26, 27, 26, 23, 18, 11));
    }
  }
}

TEST(SerializationTest, BuildTwoSubgraphs) {
  const std::string model_path =
      testing::TempDir() + "/" +
      testing::UnitTest::GetInstance()->current_test_info()->name() + ".tflite";
  ModelFactory model_builder;
  {
    Tensor a({.name = "a", .type = Type::kI32, .shape = {3, 3}});
    Tensor b({.name = "b", .type = Type::kI32, .shape = {3, 3}});
    Tensor c({.name = "c", .type = Type::kI32, .shape = {3, 3}});
    Tensor d = Mul(a, b).SetName("d");
    Tensor e = Add(c, d).SetName("e");
    EXPECT_THAT(model_builder.AddSubgraph({e, d}), IsOk());
  }
  {
    Tensor f({.name = "f", .type = Type::kI32, .shape = {3, 3}});
    Tensor g({.name = "g", .type = Type::kI32, .shape = {3, 3}});
    Tensor h = Add(f, g).SetName("h");
    EXPECT_THAT(model_builder.AddSubgraph({h}), IsOk());
  }
  EXPECT_THAT(model_builder.Save(model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->subgraphs_size(), 2);
  EXPECT_THAT(interpreter->subgraph(0)->inputs(), SizeIs(3));
  EXPECT_THAT(interpreter->subgraph(0)->outputs(), SizeIs(2));
  EXPECT_THAT(interpreter->subgraph(1)->inputs(), SizeIs(2));
  EXPECT_THAT(interpreter->subgraph(1)->outputs(), SizeIs(1));
}

TEST(SerializationTest, BuildTwoSignatures) {
  const std::string model_path =
      testing::TempDir() + "/" +
      testing::UnitTest::GetInstance()->current_test_info()->name() + ".tflite";
  const char kSignature1[] = "signature1";
  const char kSignature2[] = "signature2";
  ModelFactory model_builder;
  {
    Tensor a({.name = "a", .type = Type::kI32, .shape = {3, 3}});
    Tensor b({.name = "b", .type = Type::kI32, .shape = {3, 3}});
    Tensor c({.name = "c", .type = Type::kI32, .shape = {3, 3}});
    Tensor d = Mul(a, b).SetName("d");
    Tensor e = Add(c, d).SetName("e");
    EXPECT_THAT(model_builder.AddSignature({e, d}, /*name=*/kSignature1),
                IsOk());
  }
  {
    Tensor f({.name = "f", .type = Type::kI32, .shape = {3, 3}});
    Tensor g({.name = "g", .type = Type::kI32, .shape = {3, 3}});
    Tensor h = Add(f, g).SetName("h");
    EXPECT_THAT(model_builder.AddSignature({h}, /*name=*/kSignature2), IsOk());
  }
  EXPECT_THAT(model_builder.Save(model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_THAT(interpreter->signature_keys(),
              UnorderedElementsAre(Pointee(StrEq(kSignature1)),
                                   Pointee(StrEq(kSignature2))));
  EXPECT_THAT(interpreter->signature_inputs(kSignature1), SizeIs(3));
  EXPECT_THAT(interpreter->signature_outputs(kSignature1), SizeIs(2));
  EXPECT_THAT(interpreter->input_tensor_by_signature("a", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->input_tensor_by_signature("b", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->input_tensor_by_signature("c", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->output_tensor_by_signature("d", kSignature1),
              Not(nullptr));
  EXPECT_THAT(interpreter->output_tensor_by_signature("e", kSignature1),
              Not(nullptr));

  EXPECT_THAT(interpreter->signature_inputs(kSignature2), SizeIs(2));
  EXPECT_THAT(interpreter->signature_outputs(kSignature2), SizeIs(1));
  EXPECT_THAT(interpreter->input_tensor_by_signature("f", kSignature2),
              Not(nullptr));
  EXPECT_THAT(interpreter->input_tensor_by_signature("g", kSignature2),
              Not(nullptr));
  EXPECT_THAT(interpreter->output_tensor_by_signature("h", kSignature2),
              Not(nullptr));
}

TEST(SerializationTest, AddingAnEmptySubgraphFails) {
  ModelFactory model_builder;
  EXPECT_THAT(model_builder.AddSubgraph({}), Not(IsOk()));
}

TEST(SerializationTest, AddingAnEmptySignatureFails) {
  ModelFactory model_builder;
  EXPECT_THAT(model_builder.AddSignature({}, /*name=*/"sig-name"), Not(IsOk()));
}

TEST(SerializationTest, ConstantTensorWorks) {
  const std::string model_path = testing::TempDir() + "/mul.tflite";
  const int32_t a_data_ref[] = {1, 2, 3, 4};
  const int32_t b_data_ref[] = {8, 7, 6, 5};
  {
    std::shared_ptr a_data = OwningCpuBuffer::Copy<Type::kI32>(a_data_ref);
    std::shared_ptr b_data = OwningCpuBuffer::Copy<Type::kI32>(b_data_ref);
    Tensor a({.type = Type::kI32, .shape = {2, 2}, .buffer = a_data});
    Tensor b({.type = Type::kI32, .shape = {2, 2}, .buffer = b_data});
    Tensor c = Mul(a, b).SetName("c");
    ASSERT_THAT(Save({c}, model_path), IsOk());
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->inputs().size(), 0);
  ASSERT_EQ(interpreter->outputs().size(), 1);

  ASSERT_EQ(interpreter->tensors_size(), 3);
  ASSERT_THAT(interpreter->tensor(0)->data.raw_const,
              AlignmentIs(alignof(double)));
  ASSERT_THAT(interpreter->tensor(1)->data.raw_const,
              AlignmentIs(alignof(double)));

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  {
    ASSERT_OK_AND_ASSIGN(TfLiteTensor & c, GetOutputTensor("c", *interpreter));
    EXPECT_EQ(c.type, kTfLiteInt32);
    EXPECT_THAT(c.dims, TfLiteArrayIs({2, 2}));
    absl::Span<const int32_t> output_data(
        reinterpret_cast<int32_t*>(c.data.data), 4);
    EXPECT_THAT(output_data, ElementsAre(8, 14, 18, 20));
  }
}

TEST(SerializationTest, CreateFlatbufferWorks) {
  const int32_t a_data_ref[] = {1, 2, 3, 4};
  const int32_t b_data_ref[] = {8, 7, 6, 5};
  std::vector<char> model_data;
  {
    std::shared_ptr a_data = OwningCpuBuffer::Copy<Type::kI32>(a_data_ref);
    std::shared_ptr b_data = OwningCpuBuffer::Copy<Type::kI32>(b_data_ref);
    Tensor a(
        {.name = "a", .type = Type::kI32, .shape = {2, 2}, .buffer = a_data});
    Tensor b(
        {.name = "b", .type = Type::kI32, .shape = {2, 2}, .buffer = b_data});
    Tensor c = Mul(a, b).SetName("c");
    ModelFactory model_builder;
    ASSERT_OK(model_builder.AddSubgraph({c}));
    ASSERT_OK_AND_ASSIGN(model_data, model_builder.CreateFlatbuffer());
  }
  auto model = tflite::FlatBufferModel::BuildFromBuffer(model_data.data(),
                                                        model_data.size());
  ASSERT_THAT(model->GetModel()->buffers(), Pointee(SizeIs(3)));
  auto buffer_0 = model->GetModel()->buffers()->Get(0);
  EXPECT_THAT(buffer_0->offset(), Lt(model_data.size()));
  EXPECT_THAT(buffer_0->offset() + buffer_0->size(), Lt(model_data.size()));
  EXPECT_THAT(buffer_0->offset(), AlignmentIs(kCpuBufferAlignment));
  auto buffer_1 = model->GetModel()->buffers()->Get(1);
  ASSERT_THAT(buffer_1->offset(), Lt(model_data.size()));
  ASSERT_THAT(buffer_1->offset() + buffer_1->size(), Lt(model_data.size()));
  EXPECT_THAT(buffer_1->offset(), AlignmentIs(kCpuBufferAlignment));
  auto buffer_2 = model->GetModel()->buffers()->Get(2);
  ASSERT_THAT(buffer_2->offset(), Lt(model_data.size()));
  ASSERT_THAT(buffer_2->offset() + buffer_2->size(), Lt(model_data.size()));
  EXPECT_THAT(buffer_2->offset(), AlignmentIs(kCpuBufferAlignment));

  ASSERT_THAT(model->GetModel()->subgraphs(), Pointee(SizeIs(1)));
  auto subgraph = model->GetModel()->subgraphs()->Get(0);

  EXPECT_THAT(subgraph->tensors(), Pointee(SizeIs(3)));
  std::unordered_map<std::string, const tflite::Buffer*> named_buffers;
  ASSERT_THAT(subgraph->tensors()->Get(0)->name(), Not(IsNull()));
  named_buffers.emplace(
      subgraph->tensors()->Get(0)->name()->str(),
      model->GetModel()->buffers()->Get(subgraph->tensors()->Get(0)->buffer()));
  ASSERT_THAT(subgraph->tensors()->Get(1)->name(), Not(IsNull()));
  named_buffers.emplace(
      subgraph->tensors()->Get(1)->name()->str(),
      model->GetModel()->buffers()->Get(subgraph->tensors()->Get(1)->buffer()));
  ASSERT_THAT(subgraph->tensors()->Get(2)->name(), Not(IsNull()));
  named_buffers.emplace(
      subgraph->tensors()->Get(2)->name()->str(),
      model->GetModel()->buffers()->Get(subgraph->tensors()->Get(2)->buffer()));
  EXPECT_THAT(absl::Span<const int32_t>(
                  reinterpret_cast<int32_t*>(model_data.data() +
                                             named_buffers["a"]->offset()),
                  named_buffers["a"]->size() / sizeof(int32_t)),
              ElementsAreArray(a_data_ref));
  EXPECT_THAT(absl::Span<const int32_t>(
                  reinterpret_cast<int32_t*>(model_data.data() +
                                             named_buffers["b"]->offset()),
                  named_buffers["b"]->size() / sizeof(int32_t)),
              ElementsAreArray(b_data_ref));

  EXPECT_THAT(subgraph->inputs(), IsNull());
  EXPECT_THAT(subgraph->outputs(), Pointee(SizeIs(1)));
  EXPECT_THAT(subgraph->operators(), Pointee(SizeIs(1)));
  auto op = subgraph->operators()->Get(0);
  EXPECT_THAT(op->opcode_index(), Eq(0));
  EXPECT_THAT(op->inputs(), Pointee(SizeIs(2)));
  EXPECT_THAT(op->outputs(), Pointee(SizeIs(1)));

  ASSERT_THAT(model->GetModel()->operator_codes(), Pointee(SizeIs(1)));
  EXPECT_THAT(model->GetModel()->operator_codes()->Get(0)->builtin_code(),
              Eq(tflite::BuiltinOperator_MUL));
}

TEST(SerializationTest, UnnamedInputsAndOutputsAreGivenAName) {
  std::vector<char> model_data;
  {
    Tensor a({.type = Type::kI32, .shape = {2, 2}});
    Tensor b({.type = Type::kI32, .shape = {2, 2}});
    Tensor c = Abs(a);
    Tensor d = Abs(b);
    ModelFactory model_builder;
    ASSERT_OK(model_builder.AddSubgraph({c, d}));
    ASSERT_OK_AND_ASSIGN(model_data, model_builder.CreateFlatbuffer());

    EXPECT_THAT((std::vector{a.GetName(), b.GetName()}),
                UnorderedElementsAre(StrEq("unnamed_input_0"),
                                     StrEq("unnamed_input_1")));
    EXPECT_THAT((std::vector{c.GetName(), d.GetName()}),
                UnorderedElementsAre(StrEq("unnamed_output_0"),
                                     StrEq("unnamed_output_1")));
  }
  auto model = tflite::FlatBufferModel::BuildFromBuffer(model_data.data(),
                                                        model_data.size());

  auto GetInputName = [&](int idx) {
    return model->GetModel()
        ->subgraphs()
        ->Get(0)
        ->tensors()
        ->Get(model->GetModel()->subgraphs()->Get(0)->inputs()->Get(idx))
        ->name()
        ->str();
  };

  auto GetOutputName = [&](int idx) {
    return model->GetModel()
        ->subgraphs()
        ->Get(0)
        ->tensors()
        ->Get(model->GetModel()->subgraphs()->Get(0)->outputs()->Get(idx))
        ->name()
        ->str();
  };

  ASSERT_THAT(GetInputName(0), StrEq("unnamed_input_0"));
  ASSERT_THAT(GetInputName(1), StrEq("unnamed_input_1"));
  ASSERT_THAT(GetOutputName(0), StrEq("unnamed_output_0"));
  ASSERT_THAT(GetOutputName(1), StrEq("unnamed_output_1"));
}

TEST(SerializationTest, CanSerializeSoftmax) {
  const std::string model_path = testing::TempDir() + "/softmax.tflite";
  Tensor a({.type = Type::kFP32, .shape = {2, 5}});
  Tensor c = Softmax(a, 2.0f);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* softmax_options =
      reinterpret_cast<const TfLiteSoftmaxParams*>(
          node_and_reg->first.builtin_data);
  ASSERT_NE(softmax_options, nullptr);
  EXPECT_EQ(softmax_options->beta, 2.0f);
}

TEST(SerializationTest, CanSerializeFullyConnected) {
  const std::string model_path = testing::TempDir() + "/fully_connected.tflite";
  Tensor a({.type = Type::kFP32, .shape = {1, 5}});
  Tensor b({.type = Type::kFP32, .shape = {1, 5}});
  Tensor c({.type = Type::kFP32, .shape = {1}});
  Tensor d =
      FullyConnected(a, b, c, litert::tensor::FusedActivation::kActRelu, true);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* fc_options =
      reinterpret_cast<const TfLiteFullyConnectedParams*>(
          node_and_reg->first.builtin_data);
  EXPECT_EQ(fc_options->activation,
            kTfLiteActRelu);
  EXPECT_EQ(fc_options->keep_num_dims, true);
  EXPECT_EQ(fc_options->weights_format,
            kTfLiteFullyConnectedWeightsFormatDefault);
  EXPECT_EQ(fc_options->asymmetric_quantize_inputs, false);
  EXPECT_EQ(fc_options->quantized_bias_type, kTfLiteFloat32);
}

TEST(SerializationTest, CanSerializeFullyConnectedWithoutBias) {
  const std::string model_path =
      testing::TempDir() + "/fully_connected_no_bias.tflite";
  Tensor a({.type = Type::kFP32, .shape = {1, 5}});
  Tensor b({.type = Type::kFP32, .shape = {1, 5}});
  Tensor d =
      FullyConnected(a, b, litert::tensor::FusedActivation::kActRelu, true);
  ASSERT_THAT(Save({d}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* fc_options =
      reinterpret_cast<const TfLiteFullyConnectedParams*>(
          node_and_reg->first.builtin_data);
  EXPECT_EQ(fc_options->activation, kTfLiteActRelu);
  EXPECT_EQ(fc_options->keep_num_dims, true);
  EXPECT_EQ(fc_options->weights_format,
            kTfLiteFullyConnectedWeightsFormatDefault);
  EXPECT_EQ(fc_options->asymmetric_quantize_inputs, false);
  EXPECT_EQ(fc_options->quantized_bias_type, kTfLiteFloat32);
}

TEST(SerializationTest, CanSerializeBatchMatMul) {
  const std::string model_path = testing::TempDir() + "/batch_mat_mul.tflite";
  Tensor x({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor y({.type = Type::kFP32, .shape = {2, 3, 5}});
  Tensor z = BatchMatMul(x, y, true, false);
  ASSERT_THAT(Save({z}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  const auto* bmm_options =
      reinterpret_cast<const TfLiteBatchMatMulParams*>(
          node_and_reg->first.builtin_data);
  EXPECT_EQ(bmm_options->adj_x, true);
  EXPECT_EQ(bmm_options->adj_y, false);
}

TEST(SerializationTest, CanSerializeConcatenation) {
  const std::string model_path = testing::TempDir() + "/concatenation.tflite";
  Tensor a({.type = Type::kFP32, .shape = {1, 5}});
  Tensor b({.type = Type::kFP32, .shape = {1, 5}});
  Tensor c =
      Concatenation({a, b}, 0, litert::tensor::FusedActivation::kActNone);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* concatenation_options =
      reinterpret_cast<const TfLiteConcatenationParams*>(
          node_and_reg->first.builtin_data);
  ASSERT_NE(concatenation_options, nullptr);
  EXPECT_EQ(concatenation_options->axis, 0);
  EXPECT_EQ(concatenation_options->activation, kTfLiteActNone);
}

TEST(SerializationTest, CanSerializeTranspose) {
  const std::string model_path = testing::TempDir() + "/transpose.tflite";
  Tensor a({.type = Type::kFP32, .shape = {5, 5}});
  Tensor b({.type = Type::kI32, .shape = {5}});
  Tensor d =
      Transpose(a, b);
  ASSERT_THAT(Save({d}, model_path), IsOk());
}

TEST(SerializationTest, CanSerializeTransposeWithVector) {
  const std::string model_path =
      testing::TempDir() + "/transpose_with_vector.tflite";
  Tensor a({.type = Type::kFP32, .shape = {5, 5}});
  Tensor d = Transpose(a, {1, 0});
  ASSERT_THAT(Save({d}, model_path), IsOk());
}

TEST(SerializationTest, RunAddHasCorrectResults) {
  Tensor a({.type = Type::kFP32,
            .shape = {2, 2},
            .buffer = OwningCpuBuffer::Copy<Type::kFP32>({1, 2, 3, 4})});
  Tensor b({.type = Type::kFP32,
            .shape = {2, 2},
            .buffer = OwningCpuBuffer::Copy<Type::kFP32>({5, 6, 7, 8})});
  Tensor c = Add(a, b);
  ASSERT_THAT(tensor::Run({c}), IsOk());

  ASSERT_OK_AND_ASSIGN(Buffer & buffer, c.GetBuffer());
  EXPECT_THAT(buffer.Lock().As<const float>(), ElementsAre(6, 8, 10, 12));
}

TEST(SerializationTest, CanSerializeGelu) {
  const std::string model_path = testing::TempDir() + "/gelu.tflite";
  Tensor a({.type = Type::kFP32, .shape = {2, 5}});
  Tensor c = Gelu(a, true);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* gelu_options = reinterpret_cast<const TfLiteGeluParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(gelu_options, nullptr);
  EXPECT_EQ(gelu_options->approximate, true);
}

TEST(SerializationTest, CanSerializeReshape) {
  const std::string model_path = testing::TempDir() + "/reshape.tflite";
  Tensor a({.type = Type::kFP32, .shape = {1, 5, 1}});
  Tensor b = Reshape(a, {5});
  ASSERT_THAT(Save({b}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  const auto* reshape_options = reinterpret_cast<const TfLiteReshapeParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_NE(reshape_options, nullptr);
  EXPECT_THAT(
      absl::MakeSpan(reshape_options->shape, reshape_options->num_dimensions),
      ElementsAre(5));
}

TEST(SerializationTest, CanSerializeLogistic) {
  const std::string model_path = testing::TempDir() + "/logistic.tflite";
  Tensor a({.type = Type::kFP32, .shape = {2, 5}});
  Tensor c = Logistic(a);
  ASSERT_THAT(Save({c}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_LOGISTIC);
}

TEST(SerializationTest, CanSerializeDynamicUpdateSlice) {
  const std::string model_path =
      testing::TempDir() + "/dynamic_update_slice.tflite";
  Tensor operand({.type = Type::kFP32, .shape = {10, 10}});
  Tensor update({.type = Type::kFP32, .shape = {2, 2}});
  Tensor start_indices({.type = Type::kI32, .shape = {2}});
  Tensor result = DynamicUpdateSlice(operand, update, start_indices);
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE);
}

TEST(SerializationTest, CanSerializeDynamicUpdateSliceWithVector) {
  const std::string model_path =
      testing::TempDir() + "/dynamic_update_slice_with_vector.tflite";
  Tensor operand({.type = Type::kFP32, .shape = {10, 10}});
  Tensor update({.type = Type::kFP32, .shape = {2, 2}});
  Tensor result = DynamicUpdateSlice(operand, update, {0, 0});
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE);
}

TEST(SerializationTest, CanSerializeEmbeddingLookup) {
  const std::string model_path =
      testing::TempDir() + "/embedding_lookup.tflite";
  Tensor value({.type = Type::kFP32, .shape = {10, 4}});
  Tensor lookup({.type = Type::kI32, .shape = {2}});
  Tensor result = EmbeddingLookup(lookup, value);
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_EMBEDDING_LOOKUP);
}

TEST(SerializationTest, CanSerializeEmbeddingLookupWithVector) {
  const std::string model_path =
      testing::TempDir() + "/embedding_lookup_with_vector.tflite";
  Tensor value({.type = Type::kFP32, .shape = {10, 4}});
  Tensor result = EmbeddingLookup({1, 2}, value);
  ASSERT_THAT(Save({result}, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code,
            tflite::BuiltinOperator_EMBEDDING_LOOKUP);
}

TEST(SerializationTest, CanSerializeCustom) {
  const std::string model_path = testing::TempDir() + "/custom.tflite";
  Tensor a({.type = Type::kFP32, .shape = {2, 5}});
  std::vector<Tensor> outputs =
      Custom({a}, "MyCustomOp", {1, 2, 3}, {{2, 5}}, {Type::kFP32});
  ASSERT_THAT(Save(outputs, model_path), IsOk());

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  ASSERT_NE(model, nullptr);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteUnresolvedOps);

  ASSERT_EQ(interpreter->nodes_size(), 1);
  const auto* node_and_reg = interpreter->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->second.builtin_code, tflite::BuiltinOperator_CUSTOM);
  ASSERT_THAT(node_and_reg->second.custom_name, StrEq("MyCustomOp"));
  const auto* custom_options = node_and_reg->first.custom_initial_data;
  const size_t custom_size = node_and_reg->first.custom_initial_data_size;
  EXPECT_THAT(absl::MakeSpan((const uint8_t*)custom_options, custom_size),
              ElementsAre(1, 2, 3));
}

TEST(SerializationTest, CanSerializeTile) {
  const std::string model_path = testing::TempDir() + "/tile.tflite";
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  Tensor multiples({.type = Type::kI32, .shape = {2}});
  Tensor b = Tile(a, multiples);
  ASSERT_THAT(Save({b}, model_path), IsOk());
}

TEST(SerializationTest, CanSerializeTileWithVector) {
  const std::string model_path =
      testing::TempDir() + "/tile_with_vector.tflite";
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  Tensor b = Tile(a, {2, 1});
  ASSERT_THAT(Save({b}, model_path), IsOk());
}

}  // namespace
}  // namespace litert::tensor
