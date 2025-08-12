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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_TFLITE_FLATBUFFER_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_TFLITE_FLATBUFFER_CONVERSION_H_

#include <cstddef>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/graph.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "tflite/schema/mutable/schema_generated.h"

namespace litert::tensor {

absl::Status Save(std::vector<Tensor> outputs, absl::string_view path);

absl::Status Run(std::vector<Tensor> outputs);

class ModelFactory {
 public:
  ModelFactory();

  // Saves the model to `path`.
  absl::Status Save(absl::string_view path);

  // Creates an interpreter from the model.
  absl::StatusOr<std::vector<char>> CreateFlatbuffer();

  // Adds a subgraph to the model.
  absl::Status AddSubgraph(std::vector<Tensor> outputs);

  // Adds a new signature to the model.
  //
  // The signature inputs and outputs are named using the tensor names.
  absl::Status AddSignature(std::vector<Tensor> outputs, std::string name);

 protected:
  // Explores a new subgraph that is reachable from the given output tensors.
  absl::Status Explore(std::vector<Tensor> outputs);

  // Adds the previously explored subgraph to the flatbuffer object
  // representation.
  absl::Status Build();

  // Updates the FINISHED flatbuffer builder TFLite buffer data with the
  // corresponding sizes and offsets.
  absl::Status UpdateBufferData(flatbuffers::FlatBufferBuilder& fbb);

  absl::Status WriteBufferData(std::ofstream& output_file);

 private:
  struct TensorSerializationInfo {
    int index = -1;
    bool is_output = false;
  };

  struct OpSerializationInfo {};

  struct BufferSerializationInfo {
    int index = -1;
    size_t serialization_offset = 0;
  };

  absl::flat_hash_map<graph::Tensor, TensorSerializationInfo> tensors_;
  absl::flat_hash_map<std::shared_ptr<graph::Operation>, OpSerializationInfo>
      operations_;
  std::deque<std::shared_ptr<graph::Operation>> execution_plan_;
  absl::flat_hash_map<std::shared_ptr<Buffer>, BufferSerializationInfo>
      buffers_;
  std::deque<std::shared_ptr<Buffer>> buffer_list_;
  tflite::ModelT model_;
  size_t allocation_size_;
};
}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_TFLITE_FLATBUFFER_CONVERSION_H_
