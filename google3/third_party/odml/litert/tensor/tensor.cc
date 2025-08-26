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

#include <cstddef>
#include <memory>
#include <source_location>  // NOLINT(build/c++20): needed for OSS logging.
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"

namespace litert::tensor {

Tensor::Tensor(std::source_location loc)
    : impl_(graph::NewTensor(std::move(loc))) {}

Tensor::Tensor(graph::Tensor impl) : impl_(impl) {}

Tensor::Tensor(TensorInit init, std::source_location loc)
    : Tensor(std::move(loc)) {
  graph::TensorInformation& info = *GetInfo(impl_);
  info.name = std::move(init.name);
  info.type = init.type;
  info.shape = std::move(init.shape);
  std::visit(
      [&info, this](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, std::shared_ptr<Buffer>>) {
          // Case for a pre-made Buffer
          info.buffer = std::forward<decltype(arg)>(arg);
        } else if constexpr (std::is_same_v<T, std::vector<std::byte>>) {
          // Case for a raw byte vector
          auto owned_buffer = std::make_shared<OwningCpuBuffer>(
              std::forward<decltype(arg)>(arg));
          info.buffer = std::move(owned_buffer);
        } else {
          // Generic case for any other std::vector<T>
          // This now handles std::vector<float>, std::vector<int32_t>, etc.
          using BufferResultType =
              decltype(CreateBufferFromTypedVector(arg, info.type));
          if constexpr (std::is_same_v<
                            BufferResultType,
                            absl::StatusOr<std::shared_ptr<Buffer>>>) {
            absl::StatusOr<std::shared_ptr<Buffer>> buffer =
                CreateBufferFromTypedVector(arg, info.type);
            if (buffer.ok()) {
              info.buffer = *std::move(buffer);
            } else {
              ABSL_LOG(ERROR) << "Failed to create buffer from typed vector: "
                              << buffer.status();
            }
          } else {
            ABSL_LOG(ERROR)
                << "Unsupported std::vector type in TensorInit buffer.";
          }
        }
      },
      std::move(init.buffer));
  info.quantization = init.quantization;
}

Tensor& Tensor::SetName(std::string name) & {
  if (absl::Status s = graph::SetName(GetRaw(), std::move(name)); !s.ok()) {
    ABSL_LOG(ERROR) << "Error when setting tensor name: " << s;
  }
  return *this;
}

Tensor&& Tensor::SetName(std::string name) && {
  return std::move(this->SetName(std::move(name)));
}

absl::string_view Tensor::GetName() const {
  absl::StatusOr<const std::string&> name = graph::GetName(GetRaw());
  if (name.ok()) {
    return *name;
  }
  return "";
}

Tensor& Tensor::SetBuffer(std::shared_ptr<Buffer> buffer) & {
  if (absl::Status s = graph::SetBuffer(GetRaw(), std::move(buffer)); !s.ok()) {
    ABSL_LOG(ERROR) << "Error when setting tensor buffer: " << s;
  }
  return *this;
}

Tensor&& Tensor::SetBuffer(std::shared_ptr<Buffer> buffer) && {
  return std::move(this->SetBuffer(std::move(buffer)));
}

absl::StatusOr<Buffer&> Tensor::GetBuffer() const {
  // graph::GetBuffer expects a non-const graph::Tensor&, but GetRaw() returns
  // a const graph::Tensor& in this const method. Using const_cast to
  // temporarily remove the const qualifier. The underlying graph::GetBuffer
  // should ideally be updated to accept const graph::Tensor&.
  return graph::GetBuffer(const_cast<graph::Tensor&>(GetRaw()));
}

absl::Status Tensor::GetStatus() const { return graph::GetStatus(GetRaw()); }

template <typename T>
absl::StatusOr<std::shared_ptr<Buffer>> Tensor::CreateBufferFromTypedVector(
    const std::vector<T>& buffer_vec, Type type) {
  // 1. Perform the type check using the helper trait
  constexpr Type expected_type = CppTypeToEnum<T>::value;
  if (type != expected_type) {
    // You can create a more descriptive error message here
    return absl::InvalidArgumentError(
        "Mismatched tensor and buffer types provided.");
  }

  // 2. Perform the data conversion (same logic for all types)
  const auto* data_ptr = reinterpret_cast<const std::byte*>(buffer_vec.data());
  const size_t byte_size = buffer_vec.size() * sizeof(T);
  std::vector<std::byte> byte_buffer(data_ptr, data_ptr + byte_size);

  // 3. Create and set the buffer
  auto owned_buffer = OwningCpuBuffer::Copy(
      reinterpret_cast<const char*>(byte_buffer.data()), byte_size);
  return std::move(owned_buffer);
}
}  // namespace litert::tensor
