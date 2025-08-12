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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_DATATYPES_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_DATATYPES_H_

#include <cstddef>
#include <cstdint>
#include <ostream>
namespace litert::tensor {

enum class Type {
  kUnknown,
  kI4,
  kI8,
  kI16,
  kI32,
  kI64,
  kU4,
  kU8,
  kU16,
  kU32,
  kU64,
  kFP16,
  kFP32,
  kFP64,
  kBF16,
};

template <Type>
struct NativeStorage;

namespace internal {

template <class T>
struct NativeStorageImpl {
  using type = T;
  static constexpr size_t BufferSize(size_t count) {
    return sizeof(type) * count;
  }
};

}  // namespace internal

template <>
struct NativeStorage<Type::kI4> {
  using type = int8_t;
  static constexpr size_t BufferSize(size_t count) {
    return (sizeof(type) * count + 1) / 2;
  }
};

template <>
struct NativeStorage<Type::kI8> : internal::NativeStorageImpl<int8_t> {};

template <>
struct NativeStorage<Type::kI16> : internal::NativeStorageImpl<int16_t> {};

template <>
struct NativeStorage<Type::kI32> : internal::NativeStorageImpl<int32_t> {};

template <>
struct NativeStorage<Type::kI64> : internal::NativeStorageImpl<int64_t> {};

template <>
struct NativeStorage<Type::kU8> : internal::NativeStorageImpl<uint8_t> {};

template <>
struct NativeStorage<Type::kU16> : internal::NativeStorageImpl<uint16_t> {};

template <>
struct NativeStorage<Type::kU32> : internal::NativeStorageImpl<uint32_t> {};

template <>
struct NativeStorage<Type::kU64> : internal::NativeStorageImpl<uint64_t> {};

template <>
struct NativeStorage<Type::kFP32> : internal::NativeStorageImpl<float> {};

template <>
struct NativeStorage<Type::kFP64> : internal::NativeStorageImpl<double> {};

inline const char* ToString(Type t) {
#define LITERT_TENSOR_TYPE_TO_STRING_CASE(name) \
  case Type::k##name:                           \
    return #name
  switch (t) {
    LITERT_TENSOR_TYPE_TO_STRING_CASE(Unknown);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I4);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I8);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(I64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U4);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U8);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(U64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP16);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP32);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(FP64);
    LITERT_TENSOR_TYPE_TO_STRING_CASE(BF16);
  }
#undef LITERT_TENSOR_TYPE_TO_STRING_CASE
  // This return should never be reached.
  return "ERROR: litert::tensor::ToString(Type) failed.";
}

inline std::ostream& operator<<(std::ostream& os, const Type t) {
  return os << ToString(t);
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_DATATYPES_H_
