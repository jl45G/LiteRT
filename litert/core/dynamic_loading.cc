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

#include "litert/core/dynamic_loading.h"

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define access _access
#define R_OK 4
#else
#include <dlfcn.h>
#include <unistd.h>
// clang-format off
#ifndef __ANDROID__
#if __has_include(<link.h>)
#include <link.h>
#endif
#endif
// clang-format on
#endif

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_shared_library.h"
#include "litert/core/filesystem.h"

namespace litert::internal {

namespace {

#ifdef _WIN32
static constexpr absl::string_view kLdLibraryPath = "PATH";
static constexpr absl::string_view kPathSeparator = ";";
#else
static constexpr absl::string_view kLdLibraryPath = "LD_LIBRARY_PATH";
static constexpr absl::string_view kPathSeparator = ":";
#endif

bool EnvPathContains(absl::string_view path, absl::string_view var_value) {
  return absl::EndsWith(var_value, path) ||
         absl::StrContains(var_value, absl::StrCat(path, kPathSeparator));
}

}  // namespace

#ifdef _WIN32
static constexpr absl::string_view kSo = ".dll";
#else
static constexpr absl::string_view kSo = ".so";
#endif

LiteRtStatus FindLiteRtSharedLibsHelper(const std::string& search_path,
                                        const std::string& lib_pattern,
                                        bool full_match,
                                        std::vector<std::string>& results) {
  if (!Exists(search_path)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // TODO implement path glob in core/filesystem.h and remove filesystem
  // include from this file.
  for (const auto& entry : std::filesystem::directory_iterator(
           search_path,
           std::filesystem::directory_options::skip_permission_denied)) {
    const auto& path = entry.path();
    if (access(path.c_str(), R_OK) != 0) {
      continue;
    }
    if (entry.is_regular_file()) {
      if (full_match) {
        if (path.string().find(lib_pattern) != -1) {
          LITERT_LOG(LITERT_VERBOSE, "Found shared library: %s", path.c_str());
          results.push_back(path);
        }
      } else {
        const auto stem = path.stem().string();
        const auto ext = path.extension().string();
        if (stem.find(lib_pattern) == 0 && kSo == ext) {
          LITERT_LOG(LITERT_VERBOSE, "Found shared library: %s", path.c_str());
          results.push_back(path);
        }
      }
    } else if (entry.is_directory()) {
      FindLiteRtSharedLibsHelper(path, lib_pattern, full_match, results);
    }
  }

  return kLiteRtStatusOk;
}

static const char kCompilerPluginLibPatternFmt[] = "CompilerPlugin";

LiteRtStatus FindLiteRtCompilerPluginSharedLibs(
    absl::string_view search_path, std::vector<std::string>& results) {
  std::string root(search_path);
  const std::string lib_pattern =
      absl::StrCat(kLiteRtSharedLibPrefix, kCompilerPluginLibPatternFmt);
  return FindLiteRtSharedLibsHelper(root, lib_pattern, /*full_match=*/false,
                                    results);
}

static const char kDispatchLibPatternFmt[] = "Dispatch";

LiteRtStatus FindLiteRtDispatchSharedLibs(absl::string_view search_path,
                                          std::vector<std::string>& results) {
  std::string root(search_path.data());
  const std::string lib_pattern =
      absl::StrCat(kLiteRtSharedLibPrefix, kDispatchLibPatternFmt);
  return FindLiteRtSharedLibsHelper(root, lib_pattern, /*full_match=*/false,
                                    results);
}

LiteRtStatus PutLibOnLdPath(absl::string_view search_path,
                            absl::string_view lib_pattern) {
  std::vector<std::string> results;
  LITERT_RETURN_IF_ERROR(FindLiteRtSharedLibsHelper(
      std::string(search_path), std::string(lib_pattern), true, results));
  if (results.empty()) {
    LITERT_LOG(LITERT_INFO, "No match found in %s", search_path.data());
    return kLiteRtStatusOk;
  }

  const auto lib_dir = std::filesystem::path(results[0]).parent_path().string();
  absl::string_view ld = getenv(kLdLibraryPath.data());

  if (EnvPathContains(lib_dir, ld)) {
    LITERT_LOG(LITERT_INFO, "dir already in LD_LIBRARY_PATH");
    return kLiteRtStatusOk;
  }

  std::string new_ld;
  if (ld.empty()) {
    new_ld = lib_dir;
  } else {
    new_ld = absl::StrCat(ld, kPathSeparator, lib_dir);
  }

  LITERT_LOG(LITERT_INFO, "Adding %s to %s", new_ld.c_str(), kLdLibraryPath.data());
#ifdef _WIN32
  _putenv_s(kLdLibraryPath.data(), new_ld.c_str());
#else
  setenv(kLdLibraryPath.data(), new_ld.c_str(), /*overwrite=*/1);
#endif

  return kLiteRtStatusOk;
}

LiteRtStatus OpenLib(absl::string_view so_path, void** lib_handle,
                     bool log_failure) {
  if (!lib_handle) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *lib_handle = nullptr;
  
  if (so_path.empty()) {
    if (log_failure) {
      LITERT_LOG(LITERT_ERROR, "Cannot open library with empty path");
    }
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Use SharedLibrary class for cross-platform support
  auto lib_result = ::litert::SharedLibrary::Load(so_path, ::litert::RtldFlags::Default());
  
  if (!lib_result.ok()) {
    if (log_failure) {
      LITERT_LOG(LITERT_ERROR, "Failed to open library %s: %s", 
                 std::string(so_path).c_str(), lib_result.Error().message().c_str());
    }
    return kLiteRtStatusErrorDynamicLoading;
  }

  // Extract the handle from the SharedLibrary object
  // Note: We need to leak the SharedLibrary object here because the caller
  // expects to manage the raw handle directly
  auto* shared_lib = new ::litert::SharedLibrary(std::move(lib_result.Value()));
  *lib_handle = shared_lib->Handle();
  
  // Important: We're intentionally leaking the SharedLibrary object here
  // because the API expects raw handles. In a proper implementation,
  // we should track these objects and provide a CloseLib function.
  
  return kLiteRtStatusOk;
}

}  // namespace litert::internal
