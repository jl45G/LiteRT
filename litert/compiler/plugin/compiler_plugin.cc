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

#include "litert/compiler/plugin/compiler_plugin.h"

#include <stdlib.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_op_options.h"
#include "litert/cc/litert_shared_library.h"
#include "litert/compiler/plugin/algo.h"
#include "litert/core/build_stamp.h"
#include "litert/core/dynamic_loading.h"
#include "litert/core/environment.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/version.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"

namespace litert::internal {

//
// CompiledResult
//

Expected<BufferRef<uint8_t>> CompiledResult::ByteCode(
    LiteRtParamIndex byte_code_idx) const {
  const void* data;
  size_t size;
  LITERT_RETURN_IF_ERROR(parent_.get_compiled_result_byte_code(
      compiled_result_handle_, byte_code_idx, &data, &size));
  return BufferRef(data, size);
}

Expected<LiteRtParamIndex> CompiledResult::NumByteCodeModules() const {
  LiteRtParamIndex byte_code_idx;
  LITERT_RETURN_IF_ERROR(parent_.get_compiled_result_num_byte_code(
      compiled_result_handle_, &byte_code_idx));
  return byte_code_idx;
}

Expected<LiteRtParamIndex> CompiledResult::NumCalls() const {
  LiteRtParamIndex num_calls;
  LITERT_RETURN_IF_ERROR(parent_.get_compiled_result_num_calls(
      compiled_result_handle_, &num_calls));
  return num_calls;
}

Expected<CallInformation> CompiledResult::CallInfo(
    LiteRtParamIndex call_idx) const {
  const void* data;
  size_t size;
  LiteRtParamIndex byte_code_idx;

  LITERT_RETURN_IF_ERROR(parent_.get_compiled_result_call_info(
      compiled_result_handle_, call_idx, &data, &size, &byte_code_idx));

  absl::string_view call_info_str(reinterpret_cast<const char*>(data), size);
  return ::litert::internal::CallInformation(call_info_str, byte_code_idx);
}

CompiledResult::~CompiledResult() {
  if (compiled_result_handle_ != nullptr) {
    parent_.destroy_compiled_result(compiled_result_handle_);
  }
}

CompiledResult::CompiledResult(CompiledResult&& other)
    : parent_(other.parent_),
      compiled_result_handle_(other.compiled_result_handle_) {
  other.parent_ = {};
  other.compiled_result_handle_ = nullptr;
}

CompiledResult& CompiledResult::operator=(CompiledResult&& other) {
  if (this != &other) {
    parent_ = other.parent_;
    other.parent_ = {};

    compiled_result_handle_ = other.compiled_result_handle_;
    other.compiled_result_handle_ = nullptr;
  }
  return *this;
}

//
// CompilerPlugin
//

namespace {

#define RESOLVE_API_FUNC(name, dest) \
  LITERT_ASSIGN_OR_RETURN(dest, lib.LookupSymbol<decltype(dest)>(name.data()));

LiteRtStatus ResolvePluginApi(SharedLibrary& lib,
                              LiteRtCompilerPluginApi& result) {
  RESOLVE_API_FUNC(kLiteRtGetCompilerPluginVersion,
                   result.get_compiler_plugin_version);
  RESOLVE_API_FUNC(kLiteRtGetCompilerPluginSupportedHardware,
                   result.get_compiler_plugin_supported_hardware);
  RESOLVE_API_FUNC(kLiteRtGetCompilerPluginSocManufacturer,
                   result.get_compiler_plugin_soc_manufacturer);
  RESOLVE_API_FUNC(kLiteRtGetNumCompilerPluginSupportedSocModels,
                   result.get_num_compiler_plugin_supported_models);
  RESOLVE_API_FUNC(kLiteRtGetCompilerPluginSupportedSocModel,
                   result.get_compiler_plugin_supported_soc_model);

  RESOLVE_API_FUNC(kLiteRtCreateCompilerPlugin, result.create_compiler_plugin);
  RESOLVE_API_FUNC(kLiteRtDestroyCompilerPlugin,
                   result.destroy_compiler_plugin);

  RESOLVE_API_FUNC(kLiteRtCompilerPluginPartition,
                   result.compiler_plugin_partition);
  RESOLVE_API_FUNC(kLiteRtCompilerPluginCompile,
                   result.compiler_plugin_compile);

  RESOLVE_API_FUNC(kLiteRtDestroyCompiledResult,
                   result.destroy_compiled_result);
  RESOLVE_API_FUNC(kLiteRtCompiledResultNumByteCodeModules,
                   result.get_compiled_result_num_byte_code);
  RESOLVE_API_FUNC(kLiteRtGetCompiledResultByteCode,
                   result.get_compiled_result_byte_code);
  RESOLVE_API_FUNC(kLiteRtGetCompiledResultCallInfo,
                   result.get_compiled_result_call_info);
  RESOLVE_API_FUNC(kLiteRtGetNumCompiledResultCalls,
                   result.get_compiled_result_num_calls);

  return kLiteRtStatusOk;
}

Expected<std::vector<std::string>> GetSocModels(
    const LiteRtCompilerPluginApi& api, LiteRtCompilerPlugin plugin_handle) {
  std::vector<std::string> soc_models;

  LiteRtParamIndex num_models;
  LITERT_RETURN_IF_ERROR(
      api.get_num_compiler_plugin_supported_models(plugin_handle, &num_models));

  for (LiteRtParamIndex i = 0; i < num_models; ++i) {
    const char* model;
    if (api.get_compiler_plugin_supported_soc_model(plugin_handle, i, &model) !=
        kLiteRtStatusOk) {
      continue;
    }
    soc_models.push_back(std::string(model));
  }

  return soc_models;
}

// Sort plugins so that we first apply those supporting NPU, then those
// supporting GPU, and finally those supporting CPU.
void SortPlugins(std::vector<CompilerPlugin>& compiler_plugins) {
  std::sort(compiler_plugins.begin(), compiler_plugins.end(),
            [](auto& x, auto& y) {
              auto x_supported_hardware = x.SupportedHardware();
              auto y_supported_hardware = y.SupportedHardware();
              if (x_supported_hardware && y_supported_hardware) {
                bool x_npu = (*x_supported_hardware & kLiteRtHwAcceleratorNpu);
                bool x_gpu = (*x_supported_hardware & kLiteRtHwAcceleratorGpu);
                bool x_cpu = (*x_supported_hardware & kLiteRtHwAcceleratorCpu);
                bool y_npu = (*y_supported_hardware & kLiteRtHwAcceleratorNpu);
                bool y_gpu = (*y_supported_hardware & kLiteRtHwAcceleratorGpu);
                bool y_cpu = (*y_supported_hardware & kLiteRtHwAcceleratorCpu);
                int x_score = 100 * x_npu + 10 * x_gpu + x_cpu;
                int y_score = 100 * y_npu + 10 * y_gpu + y_cpu;
                return x_score < y_score;
              }
              return true;
            });
}

}  // namespace

Expected<CompilerPlugin> CompilerPlugin::LoadPlugin(
    const absl::string_view lib_path, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  CompilerPlugin plugin;
  LITERT_LOG(LITERT_INFO, "Loading plugin at: %s", lib_path.data());

  LITERT_ASSIGN_OR_RETURN(
      plugin.lib_,
      SharedLibrary::Load(lib_path, RtldFlags::Now().Local().DeepBind()));
  LITERT_LOG(LITERT_INFO, "Loaded plugin at: %s", lib_path.data());

  LITERT_RETURN_IF_ERROR(ResolvePluginApi(plugin.lib_, plugin.plugin_api_));
  LITERT_LOG(LITERT_INFO, "Resolved plugin api at: %s", lib_path.data());

  LITERT_RETURN_IF_ERROR(plugin.plugin_api_.create_compiler_plugin(
      &plugin.plugin_handle_, env, options));
  LITERT_LOG(LITERT_INFO, "Initialize plugin at: %s", lib_path.data());

  auto api_version = plugin.ApiVersion();
  if (!api_version) {
    return api_version.Error();
  }

  LITERT_RETURN_IF_ERROR(litert::internal::IsSameVersionAsRuntime(*api_version),
                         Unexpected(kLiteRtStatusErrorWrongVersion,
                                    "Unsupported compiler plugin version"));

  // This should never change throughout the lifetime of the compiler
  // plugin so save to avoid recalling.
  auto soc_models = GetSocModels(plugin.plugin_api_, plugin.plugin_handle_);
  if (!soc_models) {
    return soc_models.Error();
  }
  plugin.soc_models_ = *soc_models;

  return plugin;
}

Expected<std::vector<CompilerPlugin>> CompilerPlugin::LoadPlugins(
    absl::Span<const absl::string_view> lib_search_paths,
    LiteRtEnvironmentOptions env, LiteRtOptions options) {
  std::vector<std::string> plugin_lib_paths;
  for (auto search_path : lib_search_paths) {
    // Skip paths that are not valid.
    if (Exists(search_path)) {
      LITERT_RETURN_IF_ERROR(
          FindLiteRtCompilerPluginSharedLibs(search_path, plugin_lib_paths));
    }
  }

  std::vector<CompilerPlugin> loaded_plugins;
  loaded_plugins.reserve(lib_search_paths.size());

  for (const auto& lib_path : plugin_lib_paths) {
    LITERT_LOG(LITERT_INFO, "Loading plugin at: %s", lib_path.c_str());
    auto plugin = LoadPlugin(lib_path, env, options);
    if (!plugin.HasValue()) {
      continue;
    }
    loaded_plugins.push_back(std::move(plugin.Value()));
  }

  // Sort plugins.
  SortPlugins(loaded_plugins);

  return loaded_plugins;
}

CompilerPlugin::CompilerPlugin(CompilerPlugin&& other)
    : soc_models_(std::move(other.soc_models_)),
      lib_(std::move(other.lib_)),
      plugin_api_(std::move(other.plugin_api_)),
      plugin_handle_(std::move(other.plugin_handle_)) {
  other.soc_models_ = {};
  other.plugin_api_ = {};
  other.lib_.Close();
  other.plugin_handle_ = nullptr;
}

CompilerPlugin& CompilerPlugin::operator=(CompilerPlugin&& other) {
  if (this != &other) {
    std::swap(soc_models_, other.soc_models_);
    std::swap(lib_, other.lib_);
    std::swap(plugin_api_, other.plugin_api_);
    std::swap(plugin_handle_, other.plugin_handle_);
  }
  return *this;
}

CompilerPlugin::~CompilerPlugin() {
  if (plugin_handle_ != nullptr) {
    plugin_api_.destroy_compiler_plugin(plugin_handle_);
  }
}

std::string CompilerPlugin::DebugString() const {
  std::string version_str = "?";
  if (auto version = ApiVersion(); version) {
    version_str = absl::StrFormat("%d.%d.%d", version->major, version->minor,
                                  version->patch);
  }
  return absl::StrFormat("%s compiler plugin (ver %s)", SocManufacturer(),
                         version_str);
}

Expected<LiteRtApiVersion> CompilerPlugin::ApiVersion() const {
  LiteRtApiVersion api_version;
  LITERT_RETURN_IF_ERROR(plugin_api_.get_compiler_plugin_version(&api_version));
  return api_version;
}

Expected<LiteRtHwAccelerators> CompilerPlugin::SupportedHardware() const {
  LiteRtHwAccelerators supported_hardware;
  LITERT_RETURN_IF_ERROR(plugin_api_.get_compiler_plugin_supported_hardware(
      plugin_handle_, &supported_hardware));
  return supported_hardware;
}

Expected<std::vector<LiteRtOpWithPartitionIndex>> CompilerPlugin::Partition(
    LiteRtSubgraph subgraph, absl::string_view soc_model) {
  LiteRtOpListT ops;
  const char* soc_model_str = !soc_model.empty() ? soc_model.data() : nullptr;
  LITERT_RETURN_IF_ERROR(plugin_api_.compiler_plugin_partition(
      plugin_handle_, soc_model_str, subgraph, &ops));
  return ops.Values();
}

Expected<CompiledResult> CompilerPlugin::Compile(LiteRtModel partitions,
                                                 absl::string_view soc_model) {
  CompiledResult result = MakeResult();
  // If the user has passed an soc_model, then we use it; otherwise we let the
  // backend pick the appropriate one by passing nullptr as soc_model. This is
  // important for on-device compilation, where the backend must determine the
  // SoC model based on the user device.
  const char* soc_model_str = !soc_model.empty() ? soc_model.data() : nullptr;
  LITERT_RETURN_IF_ERROR(plugin_api_.compiler_plugin_compile(
      plugin_handle_, soc_model_str, partitions,
      &result.compiled_result_handle_));
  return result;
}

namespace {

LiteRtStatus PartitionSubgraph(
    std::vector<LiteRtOpWithPartitionIndex> selected_ops,
    LiteRtSubgraphT& subgraph, std::vector<LiteRtOp>& res_ops,
    LiteRtModelT& model) {
  // Group selected ops into connected islands.
  auto islands = GroupPartitions(selected_ops);
  if (islands.empty()) {
    return kLiteRtStatusOk;
  }

  // For each connected island, slice into new subgraph and replace use with
  // single dispatch op.
  for (auto& island : islands) {
    auto& new_subgraph = model.EmplaceSubgraph();
    auto* dispatch_op = OutlinePartition(subgraph, &new_subgraph, island);
    res_ops.push_back(dispatch_op);
  }

  return kLiteRtStatusOk;
}

}  // namespace

Expected<PartitionResult> PartitionModel(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition) {
  // This algorithm decides the subgraphs to be partitioned by the plugin. This
  // is a trivial process with the exception of composite ops and their
  // decomposition subgraphs. Currently, we deploy the most naive approach to
  // handling composite ops.
  //
  // There are a couple cases to consider:
  // 1. The composite op is an "odml.npu_call", in which case it represents a
  // partition which was explicitly requested by the model author.
  //
  // In this case, the the composite itself is always selected, regardless of
  // whether the plugin selects it. Its subgraph is not passed to the partition
  // function and it is passed in its entirety to the compilation function.
  //
  // More advanced behavior could include:
  // * Ensuring the plugin can compile the entire partition, and inlining it if
  // not.
  //
  // 2. Standard non npu_call composite ops.
  //
  // 2.1 Composite op is supported by the plugin. Since a plugin is capable of
  // directly compile the composite op, the composite op will be seleceted
  // during partitioning. Therefore, all ops in the decomposition subgraph will
  // be ignored, furthermore, the decomposition subgraph will be removed from
  // the model.
  //
  // 2.2 Composite op is not supported by the plugin. In this case, the
  // composite op will not be selected during partitioning. To give best effort
  // compilation, we inline the decomposition subgraph into the main subgraph,
  // that is, replace the composite op with the ops in the decomposition
  // subgraph. After re-partitioning, there are chances that the decomposition
  // Ops are now selected by the plugin.
  //
  // More advanced behavior could include:
  // * Allowing the plugin to compile the decomposition subgraph in the case
  // it cannot lower the composite directly. Potentially inline in this case
  // contingent on the availability of a suitable CPU kernel for the composite
  // op.
  //
  // ASSUMPTIONS:
  // * npu_call ops ARE NOT nested within decompositions of other npu_call ops.
  // * Standard composite ops ARE allowed to be nested within decompositions of
  // npu_call ops.
  // * No two npu_call ops share the same subgraph.
  // * Any composite op in a decomposition subgraph will be treated as a regular
  // op and inlined back into the main subgraph.

  // Find decomposition subgraphs and npu_call ops. These will be used to filter
  // subgraphs passed to the plugin and pass on auto-selected npu_call
  // partitions.
  absl::flat_hash_set<uint32_t> decomp_subgraphs;
  auto input_num_sgs = model.NumSubgraphs();
  std::vector<size_t> selected_composite_subgraph_indexes;

  ForEachIr(&model, [&](LiteRtOp op) {
    auto info = GetOptionsAs<CompositeOptions>(op);
    if (!info) {
      return;
    }
    decomp_subgraphs.insert(info->subgraph);
  });

  // Build partition result via calling plugin on non-decomposition subgraphs.
  std::vector<LiteRtOp> dispatch_ops;
  for (auto i = 0; i < input_num_sgs; ++i) {
    if (decomp_subgraphs.contains(i)) {
      continue;
    }
    if (!subgraphs_to_partition.empty() &&
        !subgraphs_to_partition.contains(i)) {
      continue;
    }
    auto* subgraph = model.Subgraphs()[i];
    auto selected_ops = compiler_plugin.Partition(subgraph, soc_model);
    // TODO ensure selected ops don't contain npu_calls.
    if (!selected_ops) {
      return selected_ops.Error();
    }
    // Find all composite ops (except npu_calls) that are not selected.
    std::vector<LiteRtOp> ops_to_inline;
    for (auto& op : subgraph->Ops()) {
      auto info = GetOptionsAs<CompositeOptions>(op);
      if (!info) {
        continue;
      }
      if (info->name == CompositeOptions::kNpuCall) {
        continue;
      }
      auto is_composite_selected = std::any_of(
          selected_ops->begin(), selected_ops->end(),
          [&op](const auto& selected_op) { return selected_op.first == op; });
      if (!is_composite_selected) {
        ops_to_inline.push_back(op);
      }
    }

    // Inline all composite ops that are not selected.
    for (auto& op : ops_to_inline) {
      auto info = GetOptionsAs<CompositeOptions>(op);
      auto status =
          InlineSubgraph(model, *op, model.Subgraphs()[info->subgraph]);
      if (status) {
        // Now the decomposition subgraph is inlined into the main subgraph,
        // therefore its also "selected".
        selected_composite_subgraph_indexes.push_back(info->subgraph);
      }
    }

    // Re-do partitioning
    selected_ops->clear();
    selected_ops = compiler_plugin.Partition(subgraph, soc_model);
    LITERT_LOG(
        LITERT_INFO,
        "PartitionSubgraph After composite inlining: %d, selected num ops: %lu",
        i, selected_ops->size());

    // Record all decomposition subgraph indexes, where its compositie op will
    // be compiled without relying on the decomposition body.
    for (auto& op : *selected_ops) {
      auto info = GetOptionsAs<CompositeOptions>(op.first);
      if (!info) {
        continue;
      }
      if (info->name == CompositeOptions::kNpuCall) {
        continue;
      }
      selected_composite_subgraph_indexes.push_back(info->subgraph);
    }
    auto num_selected_ops = selected_ops->size();
    auto num_ops = subgraph->Ops().size();

    auto num_partitions = dispatch_ops.size();
    LITERT_RETURN_IF_ERROR(PartitionSubgraph(std::move(*selected_ops),
                                             *subgraph, dispatch_ops, model));
    num_partitions = dispatch_ops.size() - num_partitions;
    LITERT_LOG(LITERT_INFO,
               "PartitionSubgraph: %d, selected num ops: %lu, from totoal ops: "
               "%lu, num partitions: %lu",
               i, num_selected_ops, num_ops, num_partitions);
  }
  ABSL_DCHECK_EQ(dispatch_ops.size(), model.NumSubgraphs() - input_num_sgs);

  // Update input_num_sgs to account for removed decomposition subgraphs.
  input_num_sgs -= selected_composite_subgraph_indexes.size();
  // Remove all decomposition subgraphs from the model.
  model.Yank(std::move(selected_composite_subgraph_indexes));

  // Collect all npu_call ops, and their decomposition subgraphs indexes.
  // Note:  we do this after partitioning and removing decomposition subgraphs,
  // so subgraph indexes of npu_calls are also updated.
  std::vector<CompositeOptions> npu_calls;
  ForEachIr(&model, [&](LiteRtOp op) {
    auto info = GetOptionsAs<CompositeOptions>(op);
    if (!info) {
      return;
    }
    if (info->name == CompositeOptions::kNpuCall) {
      npu_calls.push_back(std::move(*info));
    }
  });

  // Add collect all the subgraphs to be compiled. These are the bodies of
  // outlined partitions or npu_calls.
  std::vector<size_t> decomps_to_compile;
  for (auto& npu_call : npu_calls) {
    auto* op = npu_call.op;
    MakeDispatchOp(*op);
    dispatch_ops.push_back(op);
    decomps_to_compile.push_back(npu_call.subgraph);
  }
  for (auto i = input_num_sgs; i < model.NumSubgraphs(); ++i) {
    // Outlined subgraphs are pushed to the back of the model.
    decomps_to_compile.push_back(i);
  }

  // Create a new model split from the input model.
  auto new_model = model.Yank(std::move(decomps_to_compile));
  LITERT_LOG(LITERT_INFO, "Compiler input prepared for subgraphs %lu subgraphs",
             new_model.Subgraphs().size());

  return PartitionResult{std::move(dispatch_ops), std::move(new_model)};
}

Expected<PartitionResult> PartitionModelDirect(
    std::vector<LiteRtOpWithPartitionIndex> selected_ops, LiteRtModelT& model) {
  if (model.Subgraphs().size() != 1) {
    // Only single subgraphs supported for direct partitioning.
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }
  // Accumulate partition results for each subgraph in model.
  auto* subgraph = model.Subgraphs().front();
  std::vector<LiteRtOp> dispatch_ops;
  LITERT_RETURN_IF_ERROR(PartitionSubgraph(std::move(selected_ops), *subgraph,
                                           dispatch_ops, model));
  ABSL_DCHECK_EQ(dispatch_ops.size(), model.NumSubgraphs() - 1);

  std::vector<size_t> decomps_to_compile;
  for (auto i = 1; i < model.NumSubgraphs(); ++i) {
    // Outlined subgraphs are pushed to the back of the model.
    decomps_to_compile.push_back(i);
  }
  auto new_model = model.Yank(std::move(decomps_to_compile));

  return PartitionResult{std::move(dispatch_ops), std::move(new_model)};
}

Expected<void> ApplyPluginWithPartition(CompilerPlugin& compiler_plugin,
                                        LiteRtModelT& model,
                                        PartitionResult partitions,
                                        absl::string_view soc_model) {
  auto& dispatch_ops = partitions.first;
  auto& sliced_model = partitions.second;

  // Pass sliced subgraphs to plugin for compilation.
  auto compiled_result = compiler_plugin.Compile(&sliced_model, soc_model);
  if (!compiled_result) {
    return compiled_result.Error();
  }

  // Register byte code buffers as external buffers. Map the byte code indices
  // to the registered buffer ids.
  auto num_byte_code = compiled_result->NumByteCodeModules();
  if (!num_byte_code) {
    return num_byte_code.Error();
  }

  std::vector<LiteRtParamIndex> byte_code_idx_to_buf_id(*num_byte_code);

  for (auto i = 0; i < *num_byte_code; ++i) {
    auto byte_code = compiled_result->ByteCode(i);
    if (!byte_code) {
      return byte_code.Error();
    }

    // TODO: This copy could probably be avoided.
    OwningBufferRef<uint8_t> owned_byte_code(byte_code->Data(),
                                             byte_code->Size());
    const auto buf_id =
        model.Buffers()->RegisterOwnedBuffer(std::move(owned_byte_code));

    byte_code_idx_to_buf_id[i] = buf_id;
  }

  // Register byte code buffers and add edges from dispatch ops to them.
  for (auto i = 0; i < dispatch_ops.size(); ++i) {
    auto* dispatch_op = dispatch_ops.at(i);

    auto call_info = compiled_result->CallInfo(i);
    if (!call_info) {
      return call_info.Error();
    }
    auto [name, byte_code_idx] = *call_info;
    const auto buf_id = byte_code_idx_to_buf_id[byte_code_idx];

    model.AttachAssetToOp(dispatch_op, buf_id, std::string(name));
  }

  // Tag the model with make/model from the plugin.
  auto build_stamp =
      MakeBuildStamp(compiler_plugin.SocManufacturer(), soc_model);
  if (!build_stamp) {
    return build_stamp.Error();
  }

  if (auto status =
          model.PushMetadata(kLiteRtBuildStampKey, std::move(*build_stamp));
      status != kLiteRtStatusOk) {
    return Error(status);
  }

  return {};
}

Expected<void> ApplyPlugin(
    CompilerPlugin& compiler_plugin, LiteRtModelT& model,
    absl::string_view soc_model,
    const absl::flat_hash_set<uint32_t>& subgraphs_to_partition) {
  // Collect partitions to pass to compilation.
  auto partitions =
      PartitionModel(compiler_plugin, model, soc_model, subgraphs_to_partition);
  if (!partitions) {
    return partitions.Error();
  }
  return ApplyPluginWithPartition(compiler_plugin, model,
                                  std::move(*partitions), soc_model);
}

Expected<ApplyPluginsResult> ApplyPlugins(
    LiteRtEnvironment environment, LiteRtOptions options, LiteRtModel model,
    LiteRtHwAcceleratorSet selected_hw_accelerators, bool* mutated) {
  auto option =
      environment->GetOption(kLiteRtEnvOptionTagCompilerPluginLibraryDir);
  if (!option.has_value() || option->type != kLiteRtAnyTypeString) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Compiler plugin is not configured");
  }
  std::string compiler_plugin_lib_path = option->str_value;

  const std::array<const absl::string_view, 1>
      compiler_plugin_lib_search_paths = {compiler_plugin_lib_path};

  auto compiler_plugins = litert::internal::CompilerPlugin::LoadPlugins(
      compiler_plugin_lib_search_paths, &environment->GetOptions(), options);
  if (!compiler_plugins) {
    return compiler_plugins.Error();
  }
  if (compiler_plugins->empty()) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "No compiler plugin found");
  }

  std::vector<std::string> success_messages;
  std::vector<std::string> error_messages;

  ApplyPluginsResult result;
  result.num_applied_plugins = 0;
  for (auto& compiler_plugin : *compiler_plugins) {
    auto plugin_name = compiler_plugin.DebugString();

    auto plugin_supported_hardware = compiler_plugin.SupportedHardware();
    if (!plugin_supported_hardware) {
      error_messages.push_back(absl::StrCat(
          plugin_name, " ", plugin_supported_hardware.Error().Message()));
      continue;
    }

    if (*plugin_supported_hardware & selected_hw_accelerators) {
      auto status = ApplyPlugin(compiler_plugin, *model);
      if (mutated != nullptr) {
        *mutated = true;
      }
      if (!status) {
        error_messages.push_back(
            absl::StrCat(plugin_name, " ", status.Error().Message()));
        continue;
      }

      success_messages.push_back(absl::StrCat(plugin_name));
      result.num_applied_plugins++;
    }
  }

  result.success_message = absl::StrJoin(success_messages, ", ");
  result.error_message = absl::StrJoin(error_messages, ", ");

  return result;
}

}  // namespace litert::internal
