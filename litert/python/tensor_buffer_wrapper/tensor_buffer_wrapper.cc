// Copyright 2025 Google LLC.
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

#include "litert/python/tensor_buffer_wrapper/tensor_buffer_wrapper.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace tensor_buffer_wrapper {

namespace {
// Destructor for TensorBuffer PyCapsules.
static void CapsuleTensorBufferDestructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, "LiteRtTensorBuffer");
  if (ptr) {
    LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(ptr));
  }
}

// No-op deallocator for TensorBuffers.
static void NoopDeallocator(void* /*unused*/) {
  // Intentionally empty
}

// Converts a Python list to a vector of int8_t values.
bool ConvertPyListToInt8Vector(PyObject* py_list, std::vector<int8_t>* out,
                             std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for int8 data.";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);

  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      *error = "Non-integer value encountered in int8 list.";
      return false;
    }
    int64_t val = PyLong_AsLong(item);
    if ((val == -1) && PyErr_Occurred()) {
      *error = "Error converting python int to C long.";
      return false;
    }
    // Range check
    if (val < -128 || val > 127) {
      *error = "Value out of int8 range [-128..127].";
      return false;
    }
    out->push_back(static_cast<int8_t>(val));
  }
  return true;
}

// Converts a Python list to a vector of int32_t values.
bool ConvertPyListToInt32Vector(PyObject* py_list, std::vector<int32_t>* out,
                              std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for int32 data.";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);

  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      *error = "Non-integer value encountered in int32 list.";
      return false;
    }
    int64_t val = PyLong_AsLong(item);
    if ((val == -1) && PyErr_Occurred()) {
      *error = "Error converting python int to long for int32.";
      return false;
    }
    out->push_back(static_cast<int32_t>(val));
  }
  return true;
}

// Converts a Python list to a vector of float values.
bool ConvertPyListToFloatVector(PyObject* py_list, std::vector<float>* out,
                              std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for float32 data.";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);

  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* item = PyList_GetItem(py_list, i);
    double val = PyFloat_AsDouble(item);
    if ((val == -1.0) && PyErr_Occurred()) {
      // Try parsing as integer if float parsing failed
      PyErr_Clear();
      int64_t maybe_int_val = PyLong_AsLong(item);
      if ((maybe_int_val == -1) && PyErr_Occurred()) {
        *error = "Non-numeric value in float list.";
        return false;
      }
      val = static_cast<double>(maybe_int_val);
    }
    out->push_back(static_cast<float>(val));
  }
  return true;
}

// Creates a Python list from a span of int8_t values.
PyObject* BuildPyListFromInt8(absl::Span<const int8_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(data[i]));
  }
  return py_list;
}

// Creates a Python list from a span of int32_t values.
PyObject* BuildPyListFromInt32(absl::Span<const int32_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(data[i]));
  }
  return py_list;
}

// Creates a Python list from a span of float values.
PyObject* BuildPyListFromFloat(absl::Span<const float> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i,
                 PyFloat_FromDouble(static_cast<double>(data[i])));
  }
  return py_list;
}

// Converts Python dimensions to C++ vector
bool ConvertPyDimensionsToVector(PyObject* py_dimensions,
                              std::vector<int32_t>* out_dims,
                              std::string* error) {
  if (!PyList_Check(py_dimensions) && !PyTuple_Check(py_dimensions)) {
    *error = "Dimensions must be a list or tuple of integers";
    return false;
  }

  Py_ssize_t ndim = PySequence_Size(py_dimensions);
  if (ndim <= 0 || ndim > 8) {  // Max rank in LiteRT is typically 8
    *error = "Dimensions must have 1-8 elements";
    return false;
  }

  out_dims->clear();
  out_dims->reserve(ndim);

  for (Py_ssize_t i = 0; i < ndim; i++) {
    PyObject* dim_obj = PySequence_GetItem(py_dimensions, i);
    if (!PyLong_Check(dim_obj)) {
      Py_DECREF(dim_obj);
      *error = "Dimension values must be integers";
      return false;
    }

    int32_t dim = PyLong_AsLong(dim_obj);
    Py_DECREF(dim_obj);

    if (dim < 0) {
      *error = "Dimension values must be non-negative";
      return false;
    }

    out_dims->push_back(dim);
  }

  return true;
}

}  // namespace

// Constructor
TensorBufferWrapper::TensorBufferWrapper(litert::TensorBuffer tensor_buffer,
                                       bool owned)
    : tensor_buffer_(std::move(tensor_buffer)) {}

// Create a managed tensor buffer
TensorBufferWrapper* TensorBufferWrapper::CreateManagedBuffer(
    int buffer_type, int element_type, PyObject* dimensions,
    size_t buffer_size, std::string* out_error) {

  // Convert Python dimensions to C++ vector
  std::vector<int32_t> dims;
  if (!ConvertPyDimensionsToVector(dimensions, &dims, out_error)) {
    return nullptr;
  }

  // Create C API ranked tensor type
  LiteRtRankedTensorType c_tensor_type;
  c_tensor_type.element_type = static_cast<LiteRtElementType>(element_type);
  c_tensor_type.layout.rank = dims.size();

  // Copy dimensions
  for (size_t i = 0; i < dims.size() && i < 8; i++) {
    c_tensor_type.layout.dimensions[i] = dims[i];
  }
  c_tensor_type.layout.strides = nullptr;  // No strides

  // Create C++ API tensor type from C struct
  litert::RankedTensorType tensor_type(c_tensor_type);

  // Create tensor buffer
  auto buffer_or = litert::TensorBuffer::CreateManaged(
      static_cast<LiteRtTensorBufferType>(buffer_type),
      tensor_type, buffer_size);

  if (!buffer_or) {
    if (out_error) *out_error = buffer_or.Error().Message();
    return nullptr;
  }

  return new TensorBufferWrapper(std::move(*buffer_or));
}

// Create a tensor buffer from host memory
TensorBufferWrapper* TensorBufferWrapper::CreateFromHostMemory(
    int element_type, PyObject* dimensions,
    PyObject* host_memory_object, std::string* out_error) {

  // Convert Python dimensions to C++ vector
  std::vector<int32_t> dims;
  if (!ConvertPyDimensionsToVector(dimensions, &dims, out_error)) {
    return nullptr;
  }

  // Create C API ranked tensor type
  LiteRtRankedTensorType c_tensor_type;
  c_tensor_type.element_type = static_cast<LiteRtElementType>(element_type);
  c_tensor_type.layout.rank = dims.size();

  // Copy dimensions
  for (size_t i = 0; i < dims.size() && i < 8; i++) {
    c_tensor_type.layout.dimensions[i] = dims[i];
  }
  c_tensor_type.layout.strides = nullptr;  // No strides

  // Create C++ API tensor type from C struct
  litert::RankedTensorType tensor_type(c_tensor_type);

  // Acquire buffer from Python object
  Py_buffer py_buf;
  if (PyObject_GetBuffer(host_memory_object, &py_buf, PyBUF_CONTIG_RO) < 0) {
    if (out_error) *out_error = "Failed to get buffer from Python object";
    return nullptr;
  }

  // Create tensor buffer
  auto buffer_or = litert::TensorBuffer::CreateFromHostMemory(
      tensor_type, py_buf.buf, py_buf.len);

  // Release Python buffer
  PyBuffer_Release(&py_buf);

  if (!buffer_or) {
    if (out_error) *out_error = buffer_or.Error().Message();
    return nullptr;
  }

  return new TensorBufferWrapper(std::move(*buffer_or));
}

// Reports an error by setting a Python exception
PyObject* TensorBufferWrapper::ReportError(const std::string& msg) {
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  return nullptr;
}

// Converts a LiteRT error to a Python exception
PyObject* TensorBufferWrapper::ConvertErrorToPyExc(const litert::Error& error) {
  PyErr_Format(PyExc_RuntimeError, "TensorBuffer error: code=%d, message=%s",
             static_cast<int>(error.Status()), error.Message().c_str());
  return nullptr;
}

// Returns the byte width of a data type
size_t TensorBufferWrapper::ByteWidthOfDType(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "int8") return 1;
  if (dtype == "int32") return 4;
  return 0;  // Unknown type
}

// Get buffer type
PyObject* TensorBufferWrapper::GetBufferType() {
  auto type_or = tensor_buffer_.BufferType();
  if (!type_or) {
    return ConvertErrorToPyExc(type_or.Error());
  }
  return PyLong_FromLong(static_cast<long>(*type_or));
}

// Get tensor type
PyObject* TensorBufferWrapper::GetTensorType() {
  auto type_or = tensor_buffer_.TensorType();
  if (!type_or) {
    return ConvertErrorToPyExc(type_or.Error());
  }

  // Get C++ wrapper for tensor type
  auto tensor_type = *type_or;

  // Get underlying C struct from C++ wrapper
  LiteRtRankedTensorType c_type = static_cast<LiteRtRankedTensorType>(tensor_type);

  // Build a dict with tensor type info
  PyObject* dict = PyDict_New();

  // Add element type
  PyDict_SetItemString(dict, "element_type",
                    PyLong_FromLong(c_type.element_type));

  // Add rank
  PyDict_SetItemString(dict, "rank",
                    PyLong_FromLong(c_type.layout.rank));

  // Add dimensions
  PyObject* dims = PyList_New(c_type.layout.rank);
  for (int i = 0; i < c_type.layout.rank; i++) {
    PyList_SetItem(dims, i, PyLong_FromLong(c_type.layout.dimensions[i]));
  }
  PyDict_SetItemString(dict, "dimensions", dims);
  Py_DECREF(dims);

  return dict;
}

// Get the buffer size in bytes
PyObject* TensorBufferWrapper::GetSize() {
  auto size_or = tensor_buffer_.Size();
  if (!size_or) {
    return ConvertErrorToPyExc(size_or.Error());
  }
  return PyLong_FromSize_t(*size_or);
}

// Write tensor data
PyObject* TensorBufferWrapper::WriteTensor(PyObject* py_data, const std::string& dtype) {
  // Handle different data types
  std::string error;
  if (dtype == "float32") {
    std::vector<float> host_data;
    if (!ConvertPyListToFloatVector(py_data, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tensor_buffer_.Write<float>(absl::MakeConstSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    Py_RETURN_NONE;

  } else if (dtype == "int8") {
    std::vector<int8_t> host_data;
    if (!ConvertPyListToInt8Vector(py_data, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tensor_buffer_.Write<int8_t>(absl::MakeConstSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    Py_RETURN_NONE;

  } else if (dtype == "int32") {
    std::vector<int32_t> host_data;
    if (!ConvertPyListToInt32Vector(py_data, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tensor_buffer_.Write<int32_t>(absl::MakeConstSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    Py_RETURN_NONE;

  } else {
    return ReportError("WriteTensor: unsupported dtype '" + dtype + "'");
  }
}

// Read tensor data
PyObject* TensorBufferWrapper::ReadTensor(int num_elements, const std::string& dtype) {
  // Handle different data types
  if (dtype == "float32") {
    std::vector<float> host_data(num_elements, 0.0f);
    auto status = tensor_buffer_.Read<float>(absl::MakeSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    return BuildPyListFromFloat(host_data);

  } else if (dtype == "int8") {
    std::vector<int8_t> host_data(num_elements, 0);
    auto status = tensor_buffer_.Read<int8_t>(absl::MakeSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    return BuildPyListFromInt8(host_data);

  } else if (dtype == "int32") {
    std::vector<int32_t> host_data(num_elements, 0);
    auto status = tensor_buffer_.Read<int32_t>(absl::MakeSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    return BuildPyListFromInt32(host_data);

  } else {
    return ReportError("ReadTensor: unsupported dtype '" + dtype + "'");
  }
}

}  // namespace tensor_buffer_wrapper
}  // namespace litert
