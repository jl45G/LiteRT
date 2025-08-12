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

#include "third_party/odml/litert/tensor/tensor.h"

#include <memory>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"

namespace litert::tensor {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::StrEq;
using ::testing::status::IsOk;

MATCHER(IsValidTensor, "") {
  return ExplainMatchResult(IsOk(), GetStatus(arg.GetRaw()), result_listener);
}

MATCHER_P(LockedPtr, matcher, "") {
  return ExplainMatchResult(matcher, arg.lock(), result_listener);
}

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetInfo(Tensor& tensor) { return GetInfo(tensor.GetRaw()); }

TEST(TensorTest, DefaultConstructedTensorIsValid) {
  Tensor a;
  EXPECT_THAT(a, IsValidTensor());
}

TEST(TensorTest, DefaultConstructedTensorIsNameless) {
  Tensor a;
  EXPECT_THAT(a.GetName(), StrEq(""));
}

TEST(TensorTest, SetNameWorks) {
  Tensor a;
  a.SetName("input1");
  EXPECT_THAT(a.GetName(), StrEq("input1"));
}

TEST(TensorTest, SetNameOnRValueWorks) {
  Tensor a = Tensor().SetName("input1");
  EXPECT_THAT(a.GetName(), StrEq("input1"));
}

TEST(TensorTest, SetBufferWorks) {
  auto expected_buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  Tensor a;
  a.SetBuffer(expected_buffer);
  ASSERT_OK_AND_ASSIGN(Buffer & buffer, a.GetBuffer());
  EXPECT_THAT(buffer.Lock(), ElementsAreArray(expected_buffer->Lock()));
}

TEST(TensorTest, SetBufferOnRValueWorks) {
  auto expected_buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  Tensor a = Tensor().SetBuffer(expected_buffer);
  ASSERT_OK_AND_ASSIGN(Buffer & buffer, a.GetBuffer());
  EXPECT_THAT(buffer.Lock(), ElementsAreArray(expected_buffer->Lock()));
}

TEST(TensorTest, DefaultConstructedTensorDontHaveAProducer) {
  Tensor a;
  // The input tensors don't have a producer.
  EXPECT_EQ(a.GetRaw().group->producer, nullptr);
}

TEST(TensorTest, InitConstructorWorks) {
  std::shared_ptr<OwningCpuBuffer> buffer =
      OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  Tensor a(
      {.name = "a", .type = Type::kI32, .shape = {2, 2}, .buffer = buffer});
  ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& a_info, GetInfo(a));
  EXPECT_EQ(a_info.type, Type::kI32);
  EXPECT_THAT(a_info.shape, ElementsAre(2, 2));
  EXPECT_THAT(a_info.name, StrEq("a"));
  EXPECT_THAT(a_info.buffer, Eq(buffer));
}

TEST(TensorTest, FullyConnectedKeepDims) {
  Tensor input({.type = Type::kFP32, .shape = {2, 2, 3, 4}});
  Tensor weights({.type = Type::kFP32, .shape = {5, 4}});
  Tensor bias({.type = Type::kFP32, .shape = {5}});
  Tensor output = FullyConnected(input, weights, bias, kActNone,
                                 /*keep_num_dims=*/true);
  ASSERT_OK_AND_ASSIGN(const auto& output_info, GetInfo(output));
  EXPECT_THAT(output_info.shape, ElementsAre(2, 2, 3, 5));
}

TEST(TensorTest, FullyConnectedFlatten) {
  Tensor input({.type = Type::kFP32, .shape = {2, 2, 3, 4}});
  Tensor weights({.type = Type::kFP32, .shape = {5, 4}});
  Tensor bias({.type = Type::kFP32, .shape = {5}});
  Tensor output = FullyConnected(input, weights, bias, kActNone,
                                 /*keep_num_dims=*/false);
  ASSERT_OK_AND_ASSIGN(const auto& output_info, GetInfo(output));
  EXPECT_THAT(output_info.shape, ElementsAre(2, 5));
}

}  // namespace
}  // namespace litert::tensor
