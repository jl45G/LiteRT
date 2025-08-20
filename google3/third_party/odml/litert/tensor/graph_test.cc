#include "third_party/odml/litert/tensor/graph.h"

#include <climits>
#include <memory>
#include <source_location>  // NOLINT(build/c++20): needed for OSS logging.

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "litert/test/matchers.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"

namespace litert::tensor::graph {
namespace {

using testing::ElementsAreArray;
using testing::Not;
using testing::SizeIs;
using testing::StartsWith;
using testing::StrEq;
using testing::status::IsOk;
using testing::status::IsOkAndHolds;

MATCHER_P(IsLocation, loc, "") {
  return testing::ExplainMatchResult(testing::Eq(loc.line()), arg.line(),
                                     result_listener) &&
         testing::ExplainMatchResult(testing::Eq(loc.file_name()),
                                     arg.file_name(), result_listener);
}

TEST(TensorTest, NewTensorGroupWorks) {
  const auto loc = std::source_location::current();
  const std::shared_ptr<const TensorGroup> g = NewTensorGroup(3, loc);
  EXPECT_NE(g, nullptr);
  EXPECT_THAT(g->tensors, SizeIs(3));
  EXPECT_THAT(g->status, IsOk());
  EXPECT_EQ(g->producer, nullptr);
  EXPECT_THAT(g->loc, IsLocation(loc));
}

TEST(TensorTest, NewTensorCreatesAValidTensor) {
  const auto loc = std::source_location::current();
  Tensor a = NewTensor(loc);
  EXPECT_EQ(a.index, 0);
  ASSERT_NE(a.group, nullptr);
  EXPECT_THAT(GetStatus(a), IsOk());
  EXPECT_THAT(GetLocation(a), IsLocation(loc));
}

TEST(TensorTest, SetAndGetTensorName) {
  Tensor a = NewTensor(std::source_location::current());
  EXPECT_THAT(SetName(a, "my tensor name"), IsOk());
  EXPECT_THAT(GetName(a), IsOkAndHolds(StrEq("my tensor name")));
}

TEST(TensorTest, SetAndGetBuffer) {
  Tensor a = NewTensor(std::source_location::current());
  auto buffer = OwningCpuBuffer::Copy<Type::kI32>({1, 2, 3, 4});
  EXPECT_THAT(SetBuffer(a, buffer), IsOk());
  LITERT_ASSERT_OK_AND_ASSIGN(Buffer & retrieved_buffer, GetBuffer(a));
  EXPECT_THAT(retrieved_buffer.Lock(), ElementsAreArray(buffer->Lock()));
}

TEST(TensorTest, DefaultTensorIsInvalid) {
  Tensor a;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
  EXPECT_THAT(GetLocation(a), IsLocation(std::source_location()));
  EXPECT_THAT(GetProducer(a), Not(IsOk()));
  EXPECT_THAT(GetConsumers(a), Not(IsOk()));
}

TEST(TensorTest, TensorWithWringIndexIsInvalid) {
  Tensor a = NewTensor(std::source_location::current());
  a.index = 3;
  EXPECT_THAT(GetStatus(a), Not(IsOk()));
}

}  // namespace
}  // namespace litert::tensor::graph
