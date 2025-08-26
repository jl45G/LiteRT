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

#include "third_party/odml/litert/tensor/buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::tensor {

litert::Expected<std::shared_ptr<OwningCpuBuffer>> OwningCpuBuffer::Own(
    OwningCpuBuffer::CustomAllocPtr data, size_t bytes) {
  if (!IsAligned(data.get())) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Data isn't aligned to kCpuBufferAlignment.");
  }
  return std::make_shared<OwningCpuBuffer>(kPass, data.release(), bytes,
                                           data.get_deleter());
}

std::shared_ptr<OwningCpuBuffer> OwningCpuBuffer::Copy(const char* data,
                                                       size_t bytes) {
  CustomAllocPtr copied_data = AlignedAlloc(bytes);
  std::memcpy(copied_data.get(), data, bytes);
  return std::make_shared<OwningCpuBuffer>(kPass, copied_data.release(), bytes,
                                           copied_data.get_deleter());
}

OwningCpuBuffer::CustomAllocPtr OwningCpuBuffer::AlignedAlloc(size_t bytes) {
  static_assert(
      kCpuBufferAlignment >= alignof(ptrdiff_t) &&
          kCpuBufferAlignment % alignof(ptrdiff_t) == 0,
      "kCpuBufferAlignment must be a multiple of the alignment of the "
      "offset prefix type.");
  std::byte* data =
      new std::byte[bytes + kCpuBufferAlignment + sizeof(ptrdiff_t)];
  // We must move `data` by at least the size of the offset prefix.
  const uintptr_t min_data_offset =
      reinterpret_cast<uintptr_t>(data) + sizeof(ptrdiff_t);
  const ptrdiff_t displacement =
      kCpuBufferAlignment - (min_data_offset % kCpuBufferAlignment);
  std::byte* const aligned_data = data + sizeof(ptrdiff_t) + displacement;
  reinterpret_cast<ptrdiff_t*>(aligned_data)[-1] = aligned_data - data;
  return CustomAllocPtr(aligned_data, &AlignedFree);
}

void OwningCpuBuffer::AlignedFree(std::byte* ptr) {
  ptrdiff_t offset = *(reinterpret_cast<ptrdiff_t*>(ptr) - 1);
  delete[] (ptr - offset);
}

}  // namespace litert::tensor
