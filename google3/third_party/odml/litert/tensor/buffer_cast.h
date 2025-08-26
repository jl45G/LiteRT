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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_CAST_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_CAST_H_

#if !defined(LITERT_TENSOR_USE_RTTI)
#if defined(__clang__) && __has_feature(cxx_rtti)
#define LITERT_TENSOR_USE_RTTI 1
#elif defined(__GNUG__) && defined(__GXX_RTTI)
#define LITERT_TENSOR_USE_RTTI 1
#elif defined(_MSC_VER) && defined(_CPPRTTI)
#define LITERT_TENSOR_USE_RTTI 1
#endif
#endif

#if !LITERT_TENSOR_USE_RTTI
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace litert::tensor {

template <class T>
struct TypeSentinel {
  static constexpr char kSentinel[1] = {};
};

struct BaseTable {
  // Builds the base table for the given `Bases`.
  //
  // Note: the `This` parameter should be any pointer to a concrete class
  // instance for which the table is built. It is not kept around and is only
  // used to compute offsets.
  template <class... Bases, class Derived>
  BaseTable* Build(const Derived* const This) {
    table.assign(
        {{TypeSentinel<Derived>::kSentinel, 0},
         {TypeSentinel<Bases>::kSentinel,
          (reinterpret_cast<uintptr_t>(static_cast<const Bases*>(This)) -
           reinterpret_cast<uintptr_t>(This))}...});
    return this;
  }

  std::pair<bool, ptrdiff_t> GetOffset(const char* const sentinel) const {
    for (const auto& [s, offset] : table) {
      if (s == sentinel) {
        return {true, offset};
      }
    }
    return {false, 0};
  }

  std::vector<std::pair<const char*, ptrdiff_t>> table;
};

// Provides the interface for runtime type information for buffer
// implementations when RTTI is disabled.
//
// We need that to be able to downcast a buffer to its implementation type when
// reading the graph that was built. This will allow backends to check whether
// they support the concrete type of the buffer before using it.
//
// Any buffer trait or class should implement this interface (probably
// virtually) using `BUFFER_CAN_CAST_TO(classes buffer can be casted to)`.
//
// For instance, if you have the following:
//
// ```cpp
// struct A : virtual Buffer {};
// struct B : virtual Buffer {};
// struct C : A, B {
//   // List ALL of the classes in the hierarchy that you want to
//   // be reachable.
//   BUFFER_CAN_CAST_TO(A, B, Buffer);
// };
// ```
//
// Note: If you have access to RTTI, you should use `dynamic_cast` to do this.
class BufferTypeInfo {
 public:
  virtual ~BufferTypeInfo() = default;
  // Helper method for the implementation of downcasting without RTTI.
  virtual const BaseTable& GetBaseTable() const = 0;
};

#define LITERT_TENSOR_BUFFER_CAN_CAST_TO(...)                              \
  const BaseTable& GetBaseTable() const override {                         \
    /* We cannot use non trivial static objects in out codebase so we just \
     * never destroy it. */                                                \
    static const BaseTable* table =                                        \
        (new BaseTable())->Build<__VA_ARGS__>(this);                       \
    return *table;                                                         \
  }

// Casts a buffer instance to a subclass T is that class was registered in the
// base table of the implementation type.
template <class T>
T* As(BufferTypeInfo* b) {
  if (b) {
    const auto [can_cast, offset] =
        b->GetBaseTable().GetOffset(TypeSentinel<T>::kSentinel);
    if (can_cast) {
      return reinterpret_cast<T*>(reinterpret_cast<char*>(b) + offset);
    }
  }
  return nullptr;
}

#define LITERT_TENSOR_BUFFER_TYPE_TRAIT virtual public BufferTypeInfo

}  // namespace litert::tensor

#else  // LITERT_TENSOR_USE_RTTI

namespace litert::tensor {

#define LITERT_TENSOR_BUFFER_TYPE_TRAIT EmptyBase
#define LITERT_TENSOR_BUFFER_CAN_CAST_TO(...)

// An empty class to replace the BufferTypeInfo used when RTTI is disabled.
struct EmptyBase {};

class Buffer;

// Casts a buffer instance to a subclass T is that class was registered in the
// base table of the implementation type.
//
// Returns nullptr if the cast fails.
template <class T>
T* As(Buffer* b) {
  return dynamic_cast<T*>(b);
}

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_USE_RTTI

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BUFFER_CAST_H_
