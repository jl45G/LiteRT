#include "litert/runtime/accelerators/gpu/accelerator_options_payload.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace {

using ::testing::Eq;
using ::testing::NotNull;
using ::testing::StrEq;
using ::testing::litert::IsError;

TEST(GpuAcceleratorPayload, CreationWorks) {
  EXPECT_THAT(LiteRtCreateGpuOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  const char* identifier = nullptr;
  LITERT_ASSERT_OK(
      LiteRtGetOpaqueOptionsIdentifier(compilation_options, &identifier));
  EXPECT_THAT(identifier, StrEq(LiteRtGetGpuOptionsPayloadIdentifier()));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));
  EXPECT_THAT(payload, NotNull());

  LiteRtDestroyOpaqueOptions(compilation_options);
}

TEST(GpuAcceleratorPayload, SetAndGetConstantTensorSharing) {
  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));

  bool constant_tensor_sharing = true;

  // Check the default value.
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));

  EXPECT_THAT(LiteRtGetGpuOptionsConstantTensorSharing(nullptr, payload),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_EXPECT_OK(
      LiteRtSetGpuOptionsConstantTensorSharing(compilation_options, true));
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(true));

  EXPECT_THAT(LiteRtSetGpuOptionsConstantTensorSharing(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOpaqueOptions(compilation_options);
}

TEST(GpuAcceleratorPayload, SetAndGetInfiniteFloatCapping) {
  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));

  bool infinite_float_capping = true;

  // Check the default value.
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));

  EXPECT_THAT(LiteRtGetGpuOptionsInfiniteFloatCapping(nullptr, payload),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_EXPECT_OK(
      LiteRtSetGpuOptionsInfiniteFloatCapping(compilation_options, true));
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(true));

  EXPECT_THAT(LiteRtSetGpuOptionsInfiniteFloatCapping(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOpaqueOptions(compilation_options);
}

TEST(GpuAcceleratorPayload, SetAndGetBenchmarkMode) {
  LiteRtOpaqueOptions compilation_options;
  LITERT_ASSERT_OK(LiteRtCreateGpuOptions(&compilation_options));

  LiteRtGpuOptionsPayload payload = nullptr;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsData(
      compilation_options, reinterpret_cast<void**>(&payload)));

  bool benchmark_mode = true;

  // Check the default value.
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));

  LITERT_EXPECT_OK(LiteRtSetGpuOptionsBenchmarkMode(compilation_options, true));
  LITERT_EXPECT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(true));

  EXPECT_THAT(LiteRtSetGpuOptionsBenchmarkMode(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOpaqueOptions(compilation_options);
}

}  // namespace
