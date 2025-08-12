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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_TENSOR_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_TENSOR_H_

#include <cstdint>
#include <memory>
#include <source_location>  // NOLINT(build/c++20): needed for OSS logging.
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/graph.h"

namespace litert::tensor {

struct TensorInit {
  std::string name;
  Type type = Type::kUnknown;
  std::vector<int32_t> shape;
  std::shared_ptr<Buffer> buffer;
  std::unique_ptr<graph::QuantizationParameters> quantization;
};

class Tensor {
 public:
  explicit Tensor(std::source_location loc = std::source_location::current());
  explicit Tensor(TensorInit init,
                  std::source_location loc = std::source_location::current());
  explicit Tensor(graph::Tensor impl);

  // Sets the tensor name.
  //
  // Tensors are nameless by default.
  Tensor& SetName(std::string name) &;
  Tensor&& SetName(std::string name) &&;

  // Gets the tensor name.
  //
  // A nameless tensor will return an empty string.
  absl::string_view GetName() const;

  Tensor& SetBuffer(std::shared_ptr<Buffer> buffer) &;
  Tensor&& SetBuffer(std::shared_ptr<Buffer> buffer) &&;

  absl::StatusOr<Buffer&> GetBuffer();

  // Gets the tensor status.
  absl::Status GetStatus() const;

  // Gets the underlying graph tensor.
  graph::Tensor& GetRaw() { return impl_; }

  // Gets the underlying graph tensor.
  const graph::Tensor& GetRaw() const { return impl_; }

  friend bool operator==(const Tensor& a, const Tensor& b) = default;

 private:
  graph::Tensor impl_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_TENSOR_H_
