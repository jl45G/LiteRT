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

#include <cstdint>
#include <memory>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic_graph.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;
using ::testing::status::IsOk;
using ::testing::status::IsOkAndHolds;

MATCHER(IsValidTensor, "") {
  return ExplainMatchResult(IsOk(), GetStatus(arg.GetRaw()), result_listener);
}

MATCHER_P(LockedPtr, matcher, "") {
  return ExplainMatchResult(matcher, arg.lock(), result_listener);
}

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetConsumers(Tensor& tensor) { return GetConsumers(tensor.GetRaw()); }

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetProducer(Tensor& tensor) { return GetProducer(tensor.GetRaw()); }

// Helper function for tests. Assumes `IsValidTensor()` returns true.
auto GetInfo(Tensor& tensor) { return GetInfo(tensor.GetRaw()); }

TEST(ArithmeticTest, ChainingOpsKeepsTrackOfProducersAndConsumers) {
  Tensor a, b, c;
  Tensor d = Mul(a, b);
  Tensor e = Add(c, d);

  // The Mul op is the producer of d...
  ASSERT_OK_AND_ASSIGN(std::shared_ptr mul_op, GetProducer(d));
  EXPECT_NE(mul_op, nullptr);
  // ... and a consumer of a and b.
  EXPECT_THAT(GetConsumers(a), IsOkAndHolds(Contains(LockedPtr(mul_op))));
  EXPECT_THAT(GetConsumers(b), IsOkAndHolds(Contains(LockedPtr(mul_op))));

  // The Add op is the producer of e...
  ASSERT_OK_AND_ASSIGN(std::shared_ptr add_op, GetProducer(e));
  EXPECT_NE(add_op, nullptr);
  // ... and a consumer of c and d.
  EXPECT_THAT(GetConsumers(c), IsOkAndHolds(Contains(LockedPtr(add_op))));
  EXPECT_THAT(GetConsumers(d), IsOkAndHolds(Contains(LockedPtr(add_op))));
}

void IsAUnaryElementwiseOp(absl::string_view op_name, Tensor in, Tensor out) {
  ASSERT_THAT(in, IsValidTensor());
  ASSERT_THAT(out, IsValidTensor());

  ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& in_info, GetInfo(in));
  ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& out_info, GetInfo(out));
  EXPECT_EQ(out_info.type, in_info.type);
  EXPECT_EQ(out_info.shape, in_info.shape);

  ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(out));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->GetName(), op_name);
  EXPECT_THAT(op->inputs, UnorderedElementsAre(in.GetRaw()));
  EXPECT_THAT(op->outputs.lock(), out.GetRaw().group);
  EXPECT_THAT(GetConsumers(in), IsOkAndHolds(Contains(LockedPtr(op))));
}

void IsABinaryElementwiseOp(absl::string_view op_name, Tensor in_1, Tensor in_2,
                            Tensor out) {
  ASSERT_THAT(in_1, IsValidTensor());
  ASSERT_THAT(in_2, IsValidTensor());
  ASSERT_THAT(out, IsValidTensor());

  ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& in_1_info,
                       GetInfo(in_1));
  ASSERT_OK_AND_ASSIGN(const graph::TensorInformation& out_info, GetInfo(out));
  EXPECT_EQ(out_info.type, in_1_info.type);
  EXPECT_EQ(out_info.shape, in_1_info.shape);

  ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(out));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->GetName(), op_name);
  EXPECT_THAT(op->inputs, UnorderedElementsAre(in_1.GetRaw(), in_2.GetRaw()));
  EXPECT_THAT(op->outputs.lock(), out.GetRaw().group);
  ASSERT_THAT(GetConsumers(in_1), IsOkAndHolds(Contains(LockedPtr(op))));
  ASSERT_THAT(GetConsumers(in_2), IsOkAndHolds(Contains(LockedPtr(op))));
}

TEST(ArithmeticTest, AddWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Add", a, b, Add(a, b)));
}

TEST(ArithmeticTest, MulWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Mul", a, b, Mul(a, b)));
}

TEST(ArithmeticTest, AbsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Abs", a, Abs(a)));
}

TEST(ArithmeticTest, SubWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Sub", a, b, Sub(a, b)));
}

TEST(ArithmeticTest, DivWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Div", a, b, Div(a, b)));
}

TEST(ArithmeticTest, SquareWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Square", a, Square(a)));
}

TEST(ArithmeticTest, RsqrtWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Rsqrt", a, Rsqrt(a)));
}

TEST(ArithmeticTest, PowWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}}),
      b({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsABinaryElementwiseOp("Pow", a, b, Pow(a, b)));
}

TEST(ArithmeticTest, NegWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Neg", a, Neg(a)));
}

TEST(ArithmeticTest, SqrtWorks) {
  Tensor a({.type = Type::kFP32, .shape = {3, 3}});
  ASSERT_NO_FATAL_FAILURE(IsAUnaryElementwiseOp("Sqrt", a, Sqrt(a)));
}

TEST(ArithmeticTest, TransposeWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3, 4}});
  const std::vector<int32_t> p_data = {2, 0, 1};
  Tensor p({.type = Type::kI32, .shape = {3}, .buffer = p_data});
  Tensor b = Transpose(a, p);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 2, 3));
}

TEST(ArithmeticTest, TransposeWithVectorWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor b = Transpose(a, {2, 0, 1});
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 2, 3));
}

TEST(ArithmeticTest, SoftmaxWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Softmax(a, 1.0);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, UnorderedElementsAre(2, 4));
}

TEST(ArithmeticTest, FullyConnectedWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 8}});
  Tensor w({.type = Type::kFP32, .shape = {4, 8}});
  Tensor bias({.type = Type::kFP32, .shape = {4}});
  Tensor b = FullyConnected(a, w, bias, kActNone, false);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, FullyConnectedWithoutBiasWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 8}});
  Tensor w({.type = Type::kFP32, .shape = {4, 8}});
  Tensor b = FullyConnected(a, w, kActNone, false);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(b));
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->inputs.size(), 2);
}

TEST(ArithmeticTest, Conv2DWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  Tensor filter({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  Tensor bias({.type = Type::kFP32, .shape = {1}});
  Tensor b = Conv2D(a, filter, bias, 2, 2, kPaddingSame);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 3, 3, 1));
}

TEST(ArithmeticTest, DepthwiseConv2DWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 5, 5, 1}});
  Tensor filter({.type = Type::kFP32, .shape = {1, 3, 3, 1}});
  Tensor bias({.type = Type::kFP32, .shape = {1}});
  Tensor b = DepthwiseConv2D(a, filter, bias, 2, 2, kPaddingSame);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 3, 3, 1));
}

TEST(ArithmeticTest, PadWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {0, 0, 1, 1, 2, 2, 0, 0};
  Tensor p({.type = Type::kI32, .shape = {4, 2}, .buffer = p_data});
  Tensor b = Pad(a, p);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 4, 7, 1));
}

TEST(ArithmeticTest, PadV2Works) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const int32_t p_data[] = {0, 0, 1, 1, 2, 2, 0, 0};
  Tensor p({.type = Type::kI32,
            .shape = {4, 2},
            .buffer = OwningCpuBuffer::Copy<Type::kI32>(p_data)});
  Tensor c({.type = Type::kFP32,
            .shape = {1},
            .buffer = OwningCpuBuffer::Copy<Type::kFP32>({0.0f})});
  Tensor b = PadV2(a, p, c);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 4, 7, 1));
}

TEST(ArithmeticTest, SumKeepDimsWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = Sum(a, p, true);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1, 1, 1));
}

TEST(ArithmeticTest, SumWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3, 1}});
  const std::vector<int32_t> p_data = {1, 2};
  Tensor p({.type = Type::kI32, .shape = {2}, .buffer = p_data});
  Tensor b = Sum(a, p, false);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(1, 1));
}

TEST(ArithmeticTest, BatchMatMulWorks) {
  Tensor x({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor y({.type = Type::kFP32, .shape = {2, 4, 5}});
  Tensor z = BatchMatMul(x, y, false, false);
  ASSERT_OK_AND_ASSIGN(const auto& z_info, GetInfo(z));
  EXPECT_THAT(z_info.shape, ElementsAre(2, 3, 5));
}

TEST(ArithmeticTest, BatchMatMulFailsWithMismatchedDimensions) {
  Tensor x({.type = Type::kFP32, .shape = {2, 3, 4}});
  Tensor y({.type = Type::kFP32, .shape = {2, 5, 5}});
  Tensor z = BatchMatMul(x, y, false, false);
  EXPECT_THAT(GetStatus(z.GetRaw()), ::testing::Not(IsOk()));
}

TEST(ArithmeticTest, ConcatenationWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  Tensor b({.type = Type::kFP32, .shape = {2, 3}});
  Tensor c = Concatenation({a, b}, 1, kActNone);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 6));
}

TEST(ArithmeticTest, GeluWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Gelu(a, false);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, CastWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Cast(a, Type::kI32);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(b_info.type, Type::kI32);
}

TEST(ArithmeticTest, SelectV2Works) {
  Tensor condition({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = SelectV2(condition, a, b);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, SliceWorks) {
  Tensor a({.type = Type::kFP32, .shape = {4, 4}});
  Tensor b = Slice(a, {1, 1}, {2, 2});
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 2));
}

TEST(ArithmeticTest, LessWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Less(a, b);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, GreaterWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Greater(a, b);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, MinimumWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Minimum(a, b);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, MaximumWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b({.type = Type::kFP32, .shape = {2, 4}});
  Tensor c = Maximum(a, b);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, LogicalAndWorks) {
  Tensor a({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor b({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor c = LogicalAnd(a, b);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, LogicalOrWorks) {
  Tensor a({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor b({.type = Type::kBOOL, .shape = {2, 4}});
  Tensor c = LogicalOr(a, b);
  ASSERT_OK_AND_ASSIGN(const auto& c_info, GetInfo(c));
  EXPECT_THAT(c_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, CosWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Cos(a);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, SinWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Sin(a);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(2, 4));
}

TEST(ArithmeticTest, ReshapeWorks) {
  Tensor a({.type = Type::kFP32, .shape = {1, 2, 3}});
  Tensor b = Reshape(a, {3, 2, 1});
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(3, 2, 1));
  EXPECT_EQ(b_info.type, Type::kFP32);
}

TEST(ArithmeticTest, LogisticWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  Tensor b = Logistic(a);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, UnorderedElementsAre(2, 4));
}

TEST(ArithmeticTest, EmbeddingLookupWorks) {
  Tensor value({.type = Type::kFP32, .shape = {10, 4}});
  Tensor lookup({.type = Type::kI32, .shape = {2}});
  Tensor result = EmbeddingLookup(lookup, value);
  ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, EmbeddingLookupWithVectorWorks) {
  Tensor value({.type = Type::kFP32, .shape = {10, 4}});
  Tensor result = EmbeddingLookup({1, 2}, value);
  ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(2, 4));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, DynamicUpdateSliceWorks) {
  Tensor operand({.type = Type::kFP32, .shape = {10, 10}});
  Tensor update({.type = Type::kFP32, .shape = {2, 2}});
  Tensor start_indices({.type = Type::kI32, .shape = {2}});
  Tensor result = DynamicUpdateSlice(operand, update, start_indices);
  ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(10, 10));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, DynamicUpdateSliceWithVectorWorks) {
  Tensor operand({.type = Type::kFP32, .shape = {10, 10}});
  Tensor update({.type = Type::kFP32, .shape = {2, 2}});
  Tensor result = DynamicUpdateSlice(operand, update, {0, 0});
  ASSERT_OK_AND_ASSIGN(const auto& result_info, GetInfo(result));
  EXPECT_THAT(result_info.shape, ElementsAre(10, 10));
  EXPECT_EQ(result_info.type, Type::kFP32);
}

TEST(ArithmeticTest, CustomWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 4}});
  std::vector<Tensor> outputs =
      Custom({a}, "MyCustomOp", {1, 2, 3}, {{2, 4}}, {Type::kFP32});
  ASSERT_EQ(outputs.size(), 1);
  Tensor b = outputs[0];
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, UnorderedElementsAre(2, 4));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr op, GetProducer(b));
  ASSERT_NE(op, nullptr);
  auto* custom_op = dynamic_cast<graph::CustomOperation*>(op.get());
  ASSERT_NE(custom_op, nullptr);
  EXPECT_EQ(custom_op->custom_code, "MyCustomOp");
  EXPECT_THAT(custom_op->custom_options, ElementsAre(1, 2, 3));
}

TEST(ArithmeticTest, TileWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  const std::vector<int32_t> multiples_data = {2, 1};
  Tensor multiples(
      {.type = Type::kI32, .shape = {2}, .buffer = multiples_data});
  Tensor b = Tile(a, multiples);
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 3));
}

TEST(ArithmeticTest, TileWithVectorWorks) {
  Tensor a({.type = Type::kFP32, .shape = {2, 3}});
  Tensor b = Tile(a, {2, 1});
  ASSERT_OK_AND_ASSIGN(const auto& b_info, GetInfo(b));
  EXPECT_THAT(b_info.shape, ElementsAre(4, 3));
}

}  // namespace
}  // namespace litert::tensor
