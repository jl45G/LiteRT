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
#include "third_party/odml/litert/tensor/buffer_cast.h"

#include <memory>

#include "testing/base/public/gunit.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"

namespace litert::tensor {
namespace {

TEST(BufferCastTest, CastOwningCpuBufferWorks) {
  const float reference_data[] = {1, 2, 3, 4, 5};
  std::shared_ptr ref_buffer =
      OwningCpuBuffer::Copy<Type::kFP32>(reference_data);
  Buffer* buffer = ref_buffer.get();

  EXPECT_EQ(As<OwningCpuBuffer>(buffer),
            static_cast<OwningCpuBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<MutableSpanCpuBuffer>(buffer),
            static_cast<MutableSpanCpuBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<MutableBuffer>(buffer),
            static_cast<MutableBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<SpanCpuBuffer>(buffer),
            static_cast<SpanCpuBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<Buffer>(buffer), static_cast<Buffer*>(ref_buffer.get()));
}

TEST(BufferCastTest, CastMutableSpanCpuBufferWorks) {
  float reference_data[] = {1, 2, 3, 4, 5};
  std::shared_ptr ref_buffer =
      std::make_shared<MutableSpanCpuBuffer>(reference_data);
  Buffer* buffer = ref_buffer.get();

  EXPECT_EQ(As<OwningCpuBuffer>(buffer), nullptr);
  EXPECT_EQ(As<MutableSpanCpuBuffer>(buffer),
            static_cast<MutableSpanCpuBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<MutableBuffer>(buffer),
            static_cast<MutableBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<SpanCpuBuffer>(buffer),
            static_cast<SpanCpuBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<Buffer>(buffer), static_cast<Buffer*>(ref_buffer.get()));
}

TEST(BufferCastTest, CastSpanCpuBufferWorks) {
  const float reference_data[] = {1, 2, 3, 4, 5};
  std::shared_ptr ref_buffer = std::make_shared<SpanCpuBuffer>(reference_data);
  Buffer* buffer = ref_buffer.get();

  EXPECT_EQ(As<OwningCpuBuffer>(ref_buffer.get()), nullptr);
  EXPECT_EQ(As<MutableSpanCpuBuffer>(ref_buffer.get()), nullptr);
  EXPECT_EQ(As<MutableBuffer>(ref_buffer.get()), nullptr);
  EXPECT_EQ(As<SpanCpuBuffer>(buffer),
            static_cast<SpanCpuBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<Buffer>(buffer), static_cast<Buffer*>(ref_buffer.get()));
}

TEST(BufferCastTest, ConsecutiveCastsWork) {
  const float reference_data[] = {1, 2, 3, 4, 5};
  std::shared_ptr ref_buffer =
      OwningCpuBuffer::Copy<Type::kFP32>(reference_data);

  // Cast downwards
  OwningCpuBuffer* const o_cast = As<OwningCpuBuffer>(ref_buffer.get());
  EXPECT_EQ(o_cast, static_cast<OwningCpuBuffer*>(ref_buffer.get()));

  // Cast upwards
  MutableBuffer* const m_cast = As<MutableBuffer>(o_cast);
  EXPECT_EQ(m_cast, static_cast<MutableBuffer*>(ref_buffer.get()));

  // Cast sideways
  SpanCpuBuffer* s_cast = As<SpanCpuBuffer>(m_cast);
  EXPECT_EQ(s_cast, static_cast<SpanCpuBuffer*>(ref_buffer.get()));
  EXPECT_EQ(As<MutableBuffer>(s_cast),
            static_cast<MutableBuffer*>(ref_buffer.get()));
}

}  // namespace
}  // namespace litert::tensor
