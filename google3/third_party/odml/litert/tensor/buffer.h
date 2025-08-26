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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

#include "litert/cc/litert_expected.h"
#include "third_party/odml/litert/tensor/buffer_cast.h"
#include "third_party/odml/litert/tensor/datatypes.h"

namespace litert::tensor {

static constexpr size_t kCpuBufferAlignment = 64;

// Provides access to data stored in a buffer.
//
// Unlocks the buffer when destroyed.
template <class T = std::byte>
class LockedBufferSpan {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = T&;
  using iterator = T*;
  using const_iterator = const T*;

  LockedBufferSpan(std::byte* data, std::function<void(std::byte*)> unlock,
                   size_t bytes)
      : data_(data, std::move(unlock)), bytes_(bytes) {}

  LockedBufferSpan(
      std::unique_ptr<std::byte, std::function<void(std::byte*)>> data,
      size_t bytes)
      : data_(std::move(data)), bytes_(bytes) {}

  // Casts the span to a specific type.
  //
  // Warning: This transfers the lock management to the returned
  // `LockedBufferSpan`.
  template <class U>
  [[nodiscard]] LockedBufferSpan<U> As() && {
    static_assert(
        std::is_const_v<U> || !std::is_const_v<T>,
        "Cannot cast from a constant buffer span to a non constant one.");
    return LockedBufferSpan<U>(std::move(data_), bytes_);
  }

  T* data() const { return reinterpret_cast<T*>(data_.get()); }
  size_t size() const { return bytes_ / sizeof(T); }
  T* begin() { return data(); }
  T* end() { return data() + bytes_ / sizeof(T); }
  const T* begin() const { return data(); }
  const T* end() const { return data() + bytes_ / sizeof(T); }
  const T* cbegin() const { return data(); }
  const T* cend() const { return data() + bytes_ / sizeof(T); }

 private:
  std::unique_ptr<std::byte, std::function<void(std::byte*)>> data_;
  size_t bytes_;
};

// The main interface for buffers.
class Buffer : LITERT_TENSOR_BUFFER_TYPE_TRAIT {
 public:
  virtual ~Buffer() = default;

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  //
  // Warning: the RAII object may not manage the data lifetime, only whether
  // it's accessible from the cpu or not.
  virtual LockedBufferSpan<const std::byte> Lock() = 0;
};

class MutableBuffer : public virtual Buffer {
 public:
  ~MutableBuffer() override = default;

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  //
  // Warning: the RAII object may not manage the data lifetime, only whether
  // it's accessible from the cpu or not.
  virtual LockedBufferSpan<std::byte> LockMutable() = 0;
};

// Provides a view to constant data.
class SpanCpuBuffer : public virtual Buffer {
 public:
  SpanCpuBuffer() = default;
  ~SpanCpuBuffer() override = default;

  // Creates a viewing buffer holding `data` of size `bytes`.
  SpanCpuBuffer(const std::byte* data, size_t bytes)
      : bytes_(bytes), data_(const_cast<std::byte*>(data)) {}

  // Creates a viewing buffer from a C++ array.
  template <class T, size_t N>
  explicit SpanCpuBuffer(const std::array<T, N>& array)
      : SpanCpuBuffer(reinterpret_cast<const std::byte*>(array.data()),
                      sizeof(T) * N) {}

  // Creates a viewing buffer from a C array.
  template <class T, size_t N>
  explicit SpanCpuBuffer(const T (&arr)[N])
      : SpanCpuBuffer(reinterpret_cast<const std::byte*>(arr), sizeof(arr)) {}

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  //
  // Note: For CPU buffers, (un)locking is a no-op.
  LockedBufferSpan<const std::byte> Lock() override {
    return {data_, [](std::byte*) {}, bytes_};
  }

  LITERT_TENSOR_BUFFER_CAN_CAST_TO(Buffer);

 protected:
  size_t bytes_ = 0;
  std::byte* data_ = nullptr;
};

// Provides a view to mutable data.
class MutableSpanCpuBuffer : public SpanCpuBuffer, public MutableBuffer {
 public:
  MutableSpanCpuBuffer() = default;
  ~MutableSpanCpuBuffer() override = default;

  // Creates a viewing buffer holding `data` of size `bytes`.
  MutableSpanCpuBuffer(std::byte* data, size_t bytes)
      : SpanCpuBuffer(data, bytes) {}

  // Creates a viewing buffer from a C array.
  template <class T, size_t N>
  explicit MutableSpanCpuBuffer(T (&arr)[N]) : SpanCpuBuffer(arr) {}

  // Don't create a mutable span buffer over constant data.
  template <class T, size_t N>
  explicit MutableSpanCpuBuffer(const T (&arr)[N]) = delete;

  // Locks the buffer so that it's accessible from the CPU and returns an RAII
  // object that allows reading the data.
  //
  // Note: For CPU buffers, (un)locking is a no-op.
  LockedBufferSpan<std::byte> LockMutable() override {
    return {data_, [](std::byte*) {}, bytes_};
  }

  LITERT_TENSOR_BUFFER_CAN_CAST_TO(Buffer, SpanCpuBuffer, MutableBuffer);
};

// Manages tensor data.
class OwningCpuBuffer : public MutableSpanCpuBuffer {
 protected:
  static constexpr struct PassKey {
  } kPass;

 public:
  using CustomAllocPtr = std::unique_ptr<std::byte, void (*)(std::byte*)>;

  ~OwningCpuBuffer() override {
    if (data_) {
      destroy_data_(data_);
      data_ = nullptr;
    }
  }

  // Builds an owning cpu buffer.
  //
  // Note: This is an internal constructor that is made public to allow building
  // smart pointers in the factory functions that are below by using the passkey
  // idiom.
  OwningCpuBuffer(PassKey, std::byte* data, size_t bytes,
                  void (*DestroyData)(std::byte*))
      : MutableSpanCpuBuffer(data, bytes), destroy_data_(DestroyData) {}

  // We want to avoid unintentional copies.
  OwningCpuBuffer(const OwningCpuBuffer&) = delete;
  OwningCpuBuffer& operator=(const OwningCpuBuffer&) = delete;

  OwningCpuBuffer(OwningCpuBuffer&& other)
      : OwningCpuBuffer(kPass, other.data_, other.bytes_, other.destroy_data_) {
    other.bytes_ = 0;
    other.data_ = nullptr;
    other.destroy_data_ = [](std::byte*) {};
  }

  OwningCpuBuffer& operator=(OwningCpuBuffer&& other) {
    bytes_ = other.bytes_;
    data_ = other.data_;
    destroy_data_ = other.destroy_data_;
    other.bytes_ = 0;
    other.data_ = nullptr;
    other.destroy_data_ = [](std::byte*) {};
    return *this;
  }

  LITERT_TENSOR_BUFFER_CAN_CAST_TO(Buffer, SpanCpuBuffer, MutableBuffer,
                                   MutableSpanCpuBuffer);

  // Transfers ownership of `data` to a new `OwningCpuBuffer`.
  static litert::Expected<std::shared_ptr<OwningCpuBuffer>> Own(
      CustomAllocPtr data, size_t bytes);

  // Builds an `OwningCpuBuffer` and copy the given data to it.
  //
  // Note: This is not done with a constructor to force copies to be explicit.
  static std::shared_ptr<OwningCpuBuffer> Copy(const char* data, size_t bytes);

  // Builds an `OwningCpuBuffer` by copying the elements in the given sequence.
  //
  // - `type`: The underlying storage type.
  template <Type type, class Sequence>
  static std::shared_ptr<OwningCpuBuffer> Copy(Sequence&& seq) {
    using std::begin;
    using std::end;
    using std::size;
    const size_t bytes = NativeStorage<type>::BufferSize(size(seq));
    CustomAllocPtr copied_data = AlignedAlloc(bytes);
    std::copy(begin(seq), end(seq),
              reinterpret_cast<NativeStorage<type>::type*>(copied_data.get()));
    return std::make_shared<OwningCpuBuffer>(kPass, copied_data.release(),
                                             bytes, copied_data.get_deleter());
  }

  // Builds an `OwningCpuBuffer` by copying the elements of the given
  // initializer list.
  template <Type type, class T>
  static std::shared_ptr<OwningCpuBuffer> Copy(std::initializer_list<T>&& seq) {
    return OwningCpuBuffer::Copy<type>(seq);
  }

  // Allocates an array with an alignment of `kBufferAlignment`.
  //
  // Note: This is different from `std::aligned_alloc` because it doesn't
  // require the array size to be a multiple of the alignment.
  //
  // To do so we allocate a bigger buffer than requested, get the first
  // aligned address in it and prepend the offset to the real allocation.
  //
  // ```
  // [data    ][off][aligned_data... ]
  //                ↑
  //      This pointer is returned.
  // ```
  static CustomAllocPtr AlignedAlloc(size_t bytes);

  // Frees an array allocated with `AlignedAlloc`.
  static void AlignedFree(std::byte* ptr);

  // Returns true if the given pointer is aligned to `kBufferAlignment`.
  static bool IsAligned(const void* ptr) {
    return !(reinterpret_cast<uintptr_t>(ptr) % kCpuBufferAlignment);
  }

 protected:
  void (*destroy_data_)(std::byte*) = [](std::byte*) {};
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_H_
