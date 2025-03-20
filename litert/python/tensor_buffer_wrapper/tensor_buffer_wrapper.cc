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

#include "absl/container/flat_hash_map.h"  // from @com_google_absl  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl               // from @com_google_absl
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
// A no-op deallocator.
static void NoopDeallocator(void*) {}

// Destructor callback for the PyCapsule that owns a LiteRtTensorBuffer.
static void CapsuleTensorBufferDestructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, "LiteRtTensorBuffer");
  if (ptr) {
    LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(ptr));
  }
}

// Convert a Python list of floats -> std::vector<float>, etc.
bool ConvertPyListToFloatVector(PyObject* py_list, std::vector<float>* out,
                                std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for float32 data";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    double val = PyFloat_AsDouble(item);
    if ((val == -1.0) && PyErr_Occurred()) {
      *error = "Non-numeric value in float list.";
      return false;
    }
    out->push_back(static_cast<float>(val));
  }
  return true;
}

bool ConvertPyListToInt32Vector(PyObject* py_list, std::vector<int32_t>* out,
                                std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for int32 data";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      *error = "Non-integer value in int32 list.";
      return false;
    }
    int64_t val = PyLong_AsLong(item);
    if ((val == -1) && PyErr_Occurred()) {
      *error = "Error converting python int to int32.";
      return false;
    }
    out->push_back(static_cast<int32_t>(val));
  }
  return true;
}

bool ConvertPyListToInt8Vector(PyObject* py_list, std::vector<int8_t>* out,
                               std::string* error) {
  if (!PyList_Check(py_list)) {
    *error = "Expected a Python list for int8 data";
    return false;
  }
  Py_ssize_t length = PyList_Size(py_list);
  out->reserve(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PyList_GetItem(py_list, i);
    if (!PyLong_Check(item)) {
      *error = "Non-integer value in int8 list.";
      return false;
    }
    int64_t val = PyLong_AsLong(item);
    if ((val == -1) && PyErr_Occurred()) {
      *error = "Error converting python int to int8.";
      return false;
    }
    if (val < -128 || val > 127) {
      *error = "Value out of range for int8 [-128..127].";
      return false;
    }
    out->push_back(static_cast<int8_t>(val));
  }
  return true;
}

// Build Python list from raw data
PyObject* BuildPyListFromFloat(absl::Span<const float> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyFloat_FromDouble(data[i]));
  }
  return py_list;
}
PyObject* BuildPyListFromInt32(absl::Span<const int32_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(data[i]));
  }
  return py_list;
}
PyObject* BuildPyListFromInt8(absl::Span<const int8_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(data[i]));
  }
  return py_list;
}

}  // namespace

size_t TensorBufferWrapper::ByteWidthOfDType(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "int32") return 4;
  if (dtype == "int8") return 1;
  // Extend as needed, returning 0 for unknown
  return 0;
}

PyObject* TensorBufferWrapper::CreateFromHostMemory(PyObject* py_data,
                                                    const std::string& dtype,
                                                    Py_ssize_t num_elements) {
  // Acquire a read buffer from py_data
  Py_buffer py_buf;
  if (PyObject_GetBuffer(py_data, &py_buf, PyBUF_CONTIG_RO) < 0) {
    return nullptr;  // PyErr already set
  }

  size_t dtype_size = ByteWidthOfDType(dtype);
  if (dtype_size == 0) {
    PyBuffer_Release(&py_buf);
    return ReportError("Unsupported dtype in CreateFromHostMemory: " + dtype);
  }
  size_t required_size = static_cast<size_t>(num_elements) * dtype_size;
  if (static_cast<size_t>(py_buf.len) < required_size) {
    PyBuffer_Release(&py_buf);
    return ReportError("Python buffer is too small for required size");
  }

  // Create a LiteRtRankedTensorType for 1-D shape
  LiteRtRankedTensorType dummy_type;
  dummy_type.layout.rank = 1;
  dummy_type.layout.dimensions[0] = (int32_t)num_elements;
  dummy_type.layout.strides = nullptr;

  // Decide the element type
  if (dtype == "float32") {
    dummy_type.element_type = kLiteRtElementTypeFloat32;
  } else if (dtype == "int8") {
    dummy_type.element_type = kLiteRtElementTypeInt8;
  } else if (dtype == "int32") {
    dummy_type.element_type = kLiteRtElementTypeInt32;
  } else {
    dummy_type.element_type = kLiteRtElementTypeNone;
  }

  // Actually create the buffer
  LiteRtTensorBuffer tensor_buffer = nullptr;
  LiteRtStatus status = LiteRtCreateTensorBufferFromHostMemory(
      &dummy_type, py_buf.buf, required_size, &NoopDeallocator, &tensor_buffer);
  if (status != kLiteRtStatusOk) {
    PyBuffer_Release(&py_buf);
    return ReportError("Failed LiteRtCreateTensorBufferFromHostMemory");
  }

  // Create a PyCapsule to own the handle
  // We'll store the Py_buffer so it won't be GC'd
  struct CapsuleContext {
    Py_buffer py_buf;
    PyObject* py_obj;
    LiteRtTensorBuffer c_tensor_buffer;
  };
  auto* ctx = new CapsuleContext();
  ctx->py_buf = py_buf;
  ctx->py_obj = py_data;
  ctx->c_tensor_buffer = tensor_buffer;

  // Keep a reference to the original py_data
  Py_INCREF(py_data);

  auto capsule_destructor = [](PyObject* capsule) {
    void* raw_ptr = PyCapsule_GetPointer(capsule, "LiteRtTensorBuffer");
    auto* ctx = static_cast<CapsuleContext*>(PyCapsule_GetContext(capsule));
    if (raw_ptr) {
      LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(raw_ptr));
    }
    if (ctx) {
      PyBuffer_Release(&ctx->py_buf);
      Py_DECREF(ctx->py_obj);
      delete ctx;
    }
  };

  PyObject* capsule =
      PyCapsule_New(tensor_buffer, "LiteRtTensorBuffer", capsule_destructor);
  if (!capsule) {
    LiteRtDestroyTensorBuffer(tensor_buffer);
    PyBuffer_Release(&py_buf);
    Py_DECREF(py_data);
    delete ctx;
    return ReportError("Failed to create capsule in CreateFromHostMemory");
  }
  PyCapsule_SetContext(capsule, ctx);
  return capsule;
}

PyObject* TensorBufferWrapper::WriteTensor(PyObject* buffer_capsule,
                                           PyObject* data_list,
                                           const std::string& dtype) {
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("WriteTensor: invalid capsule");
  }
  void* ptr = PyCapsule_GetPointer(buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("WriteTensor: null pointer in capsule");
  }
  litert::TensorBuffer tb((LiteRtTensorBuffer)ptr, /*owned=*/false);

  // Convert the Python list to a C++ vector
  std::string error;
  if (dtype == "float32") {
    std::vector<float> host_data;
    if (!ConvertPyListToFloatVector(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tb.Write<float>(absl::MakeConstSpan(host_data));
    if (!status) return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;

  } else if (dtype == "int32") {
    std::vector<int32_t> host_data;
    if (!ConvertPyListToInt32Vector(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tb.Write<int32_t>(absl::MakeConstSpan(host_data));
    if (!status) return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;

  } else if (dtype == "int8") {
    std::vector<int8_t> host_data;
    if (!ConvertPyListToInt8Vector(data_list, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tb.Write<int8_t>(absl::MakeConstSpan(host_data));
    if (!status) return ConvertErrorToPyExc(status.Error());
    Py_RETURN_NONE;

  } else {
    return ReportError("WriteTensor: unsupported dtype '" + dtype + "'");
  }
}

PyObject* TensorBufferWrapper::ReadTensor(PyObject* buffer_capsule,
                                          int num_elements,
                                          const std::string& dtype) {
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("ReadTensor: invalid capsule");
  }
  void* ptr = PyCapsule_GetPointer(buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("ReadTensor: null pointer in capsule");
  }
  litert::TensorBuffer tb((LiteRtTensorBuffer)ptr, /*owned=*/false);

  if (dtype == "float32") {
    std::vector<float> data(num_elements, 0.f);
    auto status = tb.Read<float>(absl::MakeSpan(data));
    if (!status) return ConvertErrorToPyExc(status.Error());
    return BuildPyListFromFloat(data);

  } else if (dtype == "int32") {
    std::vector<int32_t> data(num_elements, 0);
    auto status = tb.Read<int32_t>(absl::MakeSpan(data));
    if (!status) return ConvertErrorToPyExc(status.Error());
    return BuildPyListFromInt32(data);

  } else if (dtype == "int8") {
    std::vector<int8_t> data(num_elements, 0);
    auto status = tb.Read<int8_t>(absl::MakeSpan(data));
    if (!status) return ConvertErrorToPyExc(status.Error());
    return BuildPyListFromInt8(data);

  } else {
    return ReportError("ReadTensor: unsupported dtype '" + dtype + "'");
  }
}

PyObject* TensorBufferWrapper::DestroyTensorBuffer(PyObject* buffer_capsule) {
  if (PyCapsule_CheckExact(buffer_capsule)) {
    if (void* ptr =
            PyCapsule_GetPointer(buffer_capsule, "LiteRtTensorBuffer")) {
      LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(ptr));
    }
  }
  // Optionally set the pointer to null to avoid double free
  // PyCapsule_SetPointer(buffer_capsule, nullptr);
  Py_RETURN_NONE;
}

PyObject* TensorBufferWrapper::ReportError(const std::string& msg) {
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  return nullptr;
}

PyObject* TensorBufferWrapper::ConvertErrorToPyExc(const litert::Error& error) {
  PyErr_Format(PyExc_RuntimeError,
               "TensorBufferWrapper error: code=%d, message=%s",
               static_cast<int>(error.Status()), error.Message().c_str());
  return nullptr;
}

}  // namespace tensor_buffer_wrapper
}  // namespace litert
