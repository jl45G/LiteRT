// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/qualcomm/context_binary_info.h"

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemContext.h"

namespace litert {
namespace qnn {

namespace {

Expected<void> InsertQnnTensors(int num_qnn_tensors, Qnn_Tensor_t* qnn_tensors,
                                std::vector<::qnn::TensorWrapper>& tensors) {
  if (num_qnn_tensors == 0) {
    return {};
  }
  
  // Get current size without clearing existing entries
  size_t current_size = tensors.size();
  LITERT_LOG(LITERT_INFO, "Current tensor count: %zu, adding %d more", 
             current_size, num_qnn_tensors);
  
  // Reserve space for new entries
  tensors.reserve(current_size + num_qnn_tensors);
  
  // Add all new tensors
  for (int i = 0; i < num_qnn_tensors; ++i) {
    tensors.emplace_back(qnn_tensors[i]);
    // TODO: chunhsue@qti handle invalid access of qnn_tensor error.
  }
  
  LITERT_LOG(LITERT_INFO, "New tensor count after insertion: %zu", tensors.size());
  return {};
}

Expected<void> InsertQnnGraphInfos(
    int num_qnn_graph_infos, QnnSystemContext_GraphInfo_t* qnn_graph_infos,
    std::vector<GraphInfo>* graphs) {
  LITERT_LOG(LITERT_INFO, "Inserting %d QNN graph info(s)",
             num_qnn_graph_infos);

  if (num_qnn_graph_infos == 0) {
    LITERT_LOG(LITERT_WARNING, "No QNN graph infos found in the binary!");
    return {};
  }

  // Get current size without clearing existing entries
  size_t current_size = graphs->size();
  LITERT_LOG(LITERT_INFO, "Current size before insertion: %zu", current_size);
  
  // Reserve space for new entries
  graphs->reserve(current_size + num_qnn_graph_infos);

  // Process and add all new graphs
  for (int i = 0; i < num_qnn_graph_infos; ++i) {
    LITERT_LOG(LITERT_INFO, "Processing QNN graph info %d with version %d", i,
               qnn_graph_infos[i].version);

    auto graph = GraphInfo::Create(qnn_graph_infos[i]);
    if (!graph) {
      LITERT_LOG(LITERT_ERROR, "Failed to create GraphInfo for graph %d: %s", i,
                 graph.Error().Message().c_str());
      return Unexpected(graph.Error());
    }

    LITERT_LOG(LITERT_INFO,
              "Successfully created GraphInfo for graph %d, name: %s", i,
              graph->Name().c_str());
    graphs->push_back(std::move(*graph));
  }

  LITERT_LOG(LITERT_INFO, "Size after insertion: %zu", graphs->size());
  return {};
}

}  // namespace

Expected<GraphInfo> GraphInfo::Create(
    const QnnSystemContext_GraphInfo_t& graph_info) {
  GraphInfo info;
  auto status = info.Init(graph_info);
  if (status) {
    return info;
  } else {
    return Unexpected(status.Error());
  }
}

Expected<void> GraphInfo::Init(const QnnSystemContext_GraphInfo_t& graph_info) {
  if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    const auto& graph_info_ = graph_info.graphInfoV1;
    name_ = graph_info_.graphName;
    LITERT_LOG(LITERT_INFO, "Found qnn graph: %s", name_.c_str());

    if (auto status = InsertQnnTensors(graph_info_.numGraphInputs,
                                       graph_info_.graphInputs, inputs_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnTensors(graph_info_.numGraphOutputs,
                                       graph_info_.graphOutputs, outputs_);
        !status) {
      return Unexpected(status.Error());
    }

  } else if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    const auto& graph_info_ = graph_info.graphInfoV2;
    name_ = graph_info_.graphName;
    LITERT_LOG(LITERT_INFO, "Found qnn graph: %s", name_.c_str());

    if (auto status = InsertQnnTensors(graph_info_.numGraphInputs,
                                       graph_info_.graphInputs, inputs_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnTensors(graph_info_.numGraphOutputs,
                                       graph_info_.graphOutputs, outputs_);
        !status) {
      return Unexpected(status.Error());
    }
  } else if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
    const auto& graph_info_ = graph_info.graphInfoV3;
    name_ = graph_info_.graphName;
    LITERT_LOG(LITERT_INFO, "Found qnn graph: %s", name_.c_str());

    if (auto status = InsertQnnTensors(graph_info_.numGraphInputs,
                                       graph_info_.graphInputs, inputs_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnTensors(graph_info_.numGraphOutputs,
                                       graph_info_.graphOutputs, outputs_);
        !status) {
      return Unexpected(status.Error());
    }

  } else {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported graph info version.");
  }
  return {};
}

Expected<void> ContextBinaryInfo::Init(
    const QnnSystemContext_BinaryInfo_t& binary_info) {
  LITERT_LOG(LITERT_INFO, 
             "Initializing context binary info with version %d (current graphs: %zu)",
             binary_info.version, graphs_.size());
             
  if (binary_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    const auto& context_binary_info = binary_info.contextBinaryInfoV1;
    LITERT_LOG(LITERT_INFO, "Processing binary info v1 with %d graphs and %d tensors",
              context_binary_info.numGraphs, context_binary_info.numContextTensors);
    if (auto status = InsertQnnTensors(context_binary_info.numContextTensors,
                                       context_binary_info.contextTensors,
                                       context_tensors_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnGraphInfos(context_binary_info.numGraphs,
                                          context_binary_info.graphs, &graphs_);
        !status) {
      return Unexpected(status.Error());
    }

  } else if (binary_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    const auto& context_binary_info = binary_info.contextBinaryInfoV2;
    LITERT_LOG(LITERT_INFO, "Processing binary info v2 with %d graphs and %d tensors",
              context_binary_info.numGraphs, context_binary_info.numContextTensors);
    if (auto status = InsertQnnTensors(context_binary_info.numContextTensors,
                                       context_binary_info.contextTensors,
                                       context_tensors_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnGraphInfos(context_binary_info.numGraphs,
                                          context_binary_info.graphs, &graphs_);
        !status) {
      return Unexpected(status.Error());
    }
  } else if (binary_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    const auto& context_binary_info = binary_info.contextBinaryInfoV3;
    LITERT_LOG(LITERT_INFO, "Processing binary info v3 with %d graphs and %d tensors",
              context_binary_info.numGraphs, context_binary_info.numContextTensors);
    if (auto status = InsertQnnTensors(context_binary_info.numContextTensors,
                                       context_binary_info.contextTensors,
                                       context_tensors_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnGraphInfos(context_binary_info.numGraphs,
                                          context_binary_info.graphs, &graphs_);
        !status) {
      return Unexpected(status.Error());
    }
  } else {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported context binary version.");
  }
  
  LITERT_LOG(LITERT_INFO, "After initialization, context has %zu graphs",
             graphs_.size());
  return {};
}

Expected<ContextBinaryInfo> ContextBinaryInfo::Create(
    QnnManager& qnn, const void* exec_bytecode_ptr, size_t exec_bytecode_size) {
  // Static cache to preserve context binary info across multiple calls
  // The key is a combination of the bytecode pointer and size
  struct BytecodeKey {
    const void* ptr;
    size_t size;
    
    bool operator==(const BytecodeKey& other) const {
      return ptr == other.ptr && size == other.size;
    }
  };
  
  struct BytecodeKeyHash {
    std::size_t operator()(const BytecodeKey& key) const {
      return std::hash<const void*>()(key.ptr) ^ std::hash<size_t>()(key.size);
    }
  };
  
  static std::unordered_map<BytecodeKey, ContextBinaryInfo, BytecodeKeyHash> binary_info_cache;
  
  // Create a key for the current bytecode
  BytecodeKey key{exec_bytecode_ptr, exec_bytecode_size};
  
  // Check if we already have this binary info in cache
  auto cache_it = binary_info_cache.find(key);
  if (cache_it != binary_info_cache.end()) {
    LITERT_LOG(LITERT_INFO, "Using cached context binary info with %zu graphs",
               cache_it->second.Graphs().size());
               
    // Log all cached graphs to help with debugging
    for (size_t i = 0; i < cache_it->second.Graphs().size(); ++i) {
      LITERT_LOG(LITERT_INFO, "Cached graph %zu: %s", 
                 i, cache_it->second.Graphs()[i].Name().c_str());
    }
    
    return cache_it->second;
  }
  
  // Not in cache, proceed with normal extraction and initialization
  auto system_context_handle = qnn.CreateSystemContextHandle();
  if (!system_context_handle) {
    return Unexpected(system_context_handle.Error());
  }

  LITERT_LOG(LITERT_INFO,
             "Extracting QNN binary info from bytecode (size: %zu)",
             exec_bytecode_size);

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size = 0;
  if (auto status = qnn.SystemApi()->systemContextGetBinaryInfo(
          system_context_handle->get(), const_cast<void*>(exec_bytecode_ptr),
          exec_bytecode_size, &binary_info, &binary_info_size);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get context binary info: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get context binary info");
  }

  if (!binary_info) {
    LITERT_LOG(LITERT_ERROR, "Null binary info", "");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Null binary info");
  }

  LITERT_LOG(LITERT_INFO, "Successfully extracted QNN binary info (size: %zu)",
             binary_info_size);

  // Log binary info version
  int version = binary_info->version;
  LITERT_LOG(LITERT_INFO, "QNN Binary info version: %d", version);

  // Create context binary info object and initialize it
  ContextBinaryInfo info;
  auto status = info.Init(*binary_info);

  if (!status) {
    LITERT_LOG(LITERT_ERROR, "Failed to initialize context binary info: %s",
               status.Error().Message().c_str());
    return Unexpected(status.Error());
  }
  
  LITERT_LOG(LITERT_INFO,
             "Successfully initialized context binary info with %zu graphs",
             info.Graphs().size());
  
  // Log all initialized graphs
  for (size_t i = 0; i < info.Graphs().size(); ++i) {
    LITERT_LOG(LITERT_INFO, "Graph %zu: %s", i, info.Graphs()[i].Name().c_str());
  }
  
  // Cache the result for future use with the same bytecode
  binary_info_cache[key] = info;
  LITERT_LOG(LITERT_INFO, "Cached binary info for future use");
  
  return info;
}
}

}  // namespace qnn
}  // namespace litert
