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

#include "third_party/odml/litert/tensor/graph.h"

#include <cstddef>
#include <memory>
#include <source_location>  // NOLINT(build/c++20): needed for OSS logging.
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"
#include "third_party/odml/litert/tensor/buffer.h"

namespace litert::tensor::graph {

Operation::~Operation() = default;

Tensor ErrorTensor(absl::Status status, std::source_location op_loc) {
  Tensor t{.group = NewTensorGroup(0, std::move(op_loc)), .index = 0};
  t.group->status = std::move(status);
  return t;
}

Tensor NewTensor(std::source_location op_loc) {
  return Tensor{.group = NewTensorGroup(1, std::move(op_loc)), .index = 0};
}

Tensor NewTensor(std::shared_ptr<TensorGroup>& group) {
  group->tensors.emplace_back();
  return {.group = group, .index = group->tensors.size() - 1};
}

std::shared_ptr<TensorGroup> NewTensorGroup(size_t count,
                                            std::source_location op_loc) {
  auto group = std::make_shared<TensorGroup>();
  group->tensors.resize(count);
  group->loc = std::move(op_loc);
  return group;
}

Tensor GetTensor(size_t index, std::shared_ptr<TensorGroup> group) {
  return Tensor{.group = std::move(group), .index = index};
}

absl::StatusOr<TensorInformation&> GetInfo(Tensor& tensor) {
  LITERT_RETURN_IF_ERROR(GetStatus(tensor));
  return tensor.group->tensors[tensor.index];
}

absl::StatusOr<const TensorInformation&> GetInfo(const Tensor& tensor) {
  LITERT_RETURN_IF_ERROR(GetStatus(tensor));
  return tensor.group->tensors[tensor.index];
}

absl::StatusOr<std::vector<std::weak_ptr<Operation>>&> GetConsumers(
    Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  return info.consumers;
}

absl::StatusOr<std::shared_ptr<Operation>&> GetProducer(const Tensor& tensor) {
  LITERT_RETURN_IF_ERROR(GetStatus(tensor));
  return tensor.group->producer;
}

absl::StatusOr<const std::string&> GetName(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  if (!info.name.empty()) {
    return info.name;
  }
  return absl::NotFoundError("This tensor doesn't have a name");
}

absl::Status SetName(Tensor& tensor, std::string name) {
  LITERT_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  info.name = std::move(name);
  return absl::OkStatus();
}

absl::Status SetBuffer(Tensor& tensor, std::shared_ptr<Buffer> buffer) {
  LITERT_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  info.buffer = std::move(buffer);
  return absl::OkStatus();
}

absl::StatusOr<Buffer&> GetBuffer(Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  if (info.buffer) {
    return *info.buffer;
  }
  return absl::NotFoundError(
      "This tensor doesn't have an associated data buffer");
}

absl::Status GetStatus(const Tensor& tensor) {
  if (!tensor.group) {
    return absl::InvalidArgumentError(
        "Tensor doesn't point to a tensor group.");
  }
  if (!tensor.group->status.ok()) {
    return tensor.group->status;
  }
  if (tensor.index >= tensor.group->tensors.size()) {
    return absl::InvalidArgumentError(
        "Tensor index doesn't exist in its group.");
  }
  return absl::OkStatus();
}

std::source_location GetLocation(const Tensor& tensor) {
  if (!tensor.group) {
    return std::source_location();
  }
  return tensor.group->loc;
}

}  // namespace litert::tensor::graph
