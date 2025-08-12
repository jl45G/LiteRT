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
#include <source_location>  // NOLINT(build/c++20): needed for OSS logging.
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/buffer.h"
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
  info.buffer = init.buffer;
  info.quantization = std::move(init.quantization);
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

absl::StatusOr<Buffer&> Tensor::GetBuffer() {
  return graph::GetBuffer(GetRaw());
}

absl::Status Tensor::GetStatus() const { return graph::GetStatus(GetRaw()); }

}  // namespace litert::tensor
