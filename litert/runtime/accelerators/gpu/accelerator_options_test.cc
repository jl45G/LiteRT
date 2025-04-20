#include "litert/runtime/accelerators/gpu/accelerator_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/runtime/accelerators/gpu/accelerator_options_payload.h"
#include "litert/test/matchers.h"

using ::testing::Eq;

namespace litert::ml_drift {
namespace {

TEST(GpuOptions, EnableConstantTensorSharingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool constant_tensor_sharing = true;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));

  options.EnableConstantTensorSharing(true);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(true));

  options.EnableConstantTensorSharing(false);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsConstantTensorSharing(
      &constant_tensor_sharing, payload));
  EXPECT_THAT(constant_tensor_sharing, Eq(false));
}

TEST(GpuOptions, EnableInfiniteFloatCappingWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool infinite_float_capping = true;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));

  options.EnableInfiniteFloatCapping(true);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(true));

  options.EnableInfiniteFloatCapping(false);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsInfiniteFloatCapping(
      &infinite_float_capping, payload));
  EXPECT_THAT(infinite_float_capping, Eq(false));
}

TEST(GpuOptions, EnableBenchmarkModeWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool benchmark_mode = true;
  LITERT_ASSERT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));

  options.EnableBenchmarkMode(true);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(true));

  options.EnableBenchmarkMode(false);

  LITERT_ASSERT_OK(LiteRtGetGpuOptionsBenchmarkMode(&benchmark_mode, payload));
  EXPECT_THAT(benchmark_mode, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckTrue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  bool allow_src_quantized_fc_conv_ops = false;
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(true));

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(true));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckDefaultValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  bool allow_src_quantized_fc_conv_ops = true;

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions,
     EnableAllowSrcQuantizedFcConvOpsCheckFalseValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // The default value is false, set it to true before resetting to false.
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(true));
  LITERT_EXPECT_OK(options.EnableAllowSrcQuantizedFcConvOps(false));
  bool allow_src_quantized_fc_conv_ops = true;

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
          &allow_src_quantized_fc_conv_ops, payload));
  EXPECT_THAT(allow_src_quantized_fc_conv_ops, Eq(false));
}

TEST(GpuAcceleratorCompilationOptions, CheckDelegatePrecisionDefaultPrecision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionFp16;
  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionDefault));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp16Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
  options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp16);

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp16));
}

TEST(GpuAcceleratorCompilationOptions, SetDelegatePrecisionFp32Precision) {
  LITERT_ASSERT_OK_AND_ASSIGN(GpuOptions options, GpuOptions::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtGpuOptionsPayload payload,
                              options.GetData<LiteRtGpuOptionsPayloadT>());

  // Check the default value.
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;

  options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32);

  LITERT_ASSERT_OK(
      LiteRtGetGpuAcceleratorCompilationOptionsPrecision(&precision, payload));
  EXPECT_THAT(precision, Eq(kLiteRtDelegatePrecisionFp32));
}

}  // namespace
}  // namespace litert::ml_drift
