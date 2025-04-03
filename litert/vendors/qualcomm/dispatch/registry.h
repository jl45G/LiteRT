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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_

#include <vector>

#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"

namespace litert::qnn {

template <typename H, typename V>
class Registry {
 public:
  Expected<H> Register(const V& value) {
    // TODO: improve this linear search by keeping an index to the first unused
    // element.
    for (auto i = 0; i < entries_.size(); ++i) {
      auto& entry = entries_[i];
      if (!entry.used) {
        entry.value = value;
        entry.used = true;
        LITERT_LOG(LITERT_INFO,
                   "Registered value in existing slot %d, handle: %p", i,
                   static_cast<H>(i));
        return static_cast<H>(i);
      }
    }
    // Grow the set of entries.
    H handle = static_cast<H>(entries_.size());
    entries_.emplace_back(value);
    LITERT_LOG(LITERT_INFO, "Registered value in new slot %zu, handle: %p",
               entries_.size() - 1, handle);
    return handle;
  }

  Expected<void> Unregister(H handle) {
    LITERT_LOG(LITERT_INFO, "Unregistering handle %p (registry size: %zu)",
               handle, entries_.size());

    if (handle < 0 || handle >= entries_.size()) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to unregister handle %p: out of range [0,%zu)", handle,
                 entries_.size());
      return Unexpected(kLiteRtStatusErrorNotFound, "Unexpected handle");
    }

    if (!entries_[handle].used) {
      LITERT_LOG(LITERT_WARNING, "Handle %p is already unregistered", handle);
    }

    entries_[handle].used = false;
    LITERT_LOG(LITERT_INFO, "Successfully unregistered handle %p", handle);
    return {};
  }

  Expected<V*> Get(H handle) {
    LITERT_LOG(LITERT_INFO, "Getting handle %p (registry size: %zu)", handle,
               entries_.size());

    if (handle < 0 || handle >= entries_.size()) {
      LITERT_LOG(LITERT_ERROR, "Failed to get handle %p: out of range [0,%zu)",
                 handle, entries_.size());
      return Unexpected(kLiteRtStatusErrorNotFound, "Unexpected handle");
    }

    if (!entries_[handle].used) {
      LITERT_LOG(LITERT_WARNING, "Getting unused handle %p", handle);
    }

    LITERT_LOG(LITERT_INFO, "Successfully got handle %p", handle);
    return &entries_[handle].value;
  }

 private:
  struct Entry {
    V value;
    bool used;
    explicit Entry(const V& v) : value(v), used(true) {}
  };

  std::vector<Entry> entries_;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_
