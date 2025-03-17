/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "litert/python/compiled_model_wrapper/compiled_model_wrapper.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
// #include
// "litert/python/compiled_model_wrapper/python_error_reporter.h"
// #include
// "litert/python/compiled_model_wrapper/python_utils.h"

namespace litert {
namespace compiled_model_wrapper {

namespace {
// Destructor for TensorBuffer PyCapsules.
static void CapsuleTensorBufferDestructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, "LiteRtTensorBuffer");
  if (ptr) {
    LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(ptr));
  }
}

// Creates a PyCapsule for a TensorBuffer with appropriate destructor.
PyObject* MakePyCapsuleForTensorBuffer(litert::TensorBuffer& buffer,
                                       bool owned_by_python) {
  if (owned_by_python) {
    return PyCapsule_New(buffer.Release(), "LiteRtTensorBuffer",
                         &CapsuleTensorBufferDestructor);
  } else {
    return PyCapsule_New(buffer.Get(), "LiteRtTensorBuffer",
                         /*destructor=*/nullptr);
  }
}

// Context for host memory deallocation.
struct HostMemoryDeallocatorContext {
  Py_buffer py_buf;
  PyObject* py_obj;  // Reference to ensure object isn't garbage collected
};

// Deallocator callback for host memory.
static void HostMemoryDeallocator(void* context) {
  if (!context) return;
  auto* ctx = static_cast<HostMemoryDeallocatorContext*>(context);
  PyBuffer_Release(&ctx->py_buf);
  Py_XDECREF(ctx->py_obj);
  delete ctx;
}

// Context for Python buffer capsules.
struct HostMemoryPyCapsuleContext {
  Py_buffer py_buf;
  PyObject* py_obj;                    // Reference to Python object
  LiteRtTensorBuffer c_tensor_buffer;  // Handle to C++ buffer
};

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
    long val = PyLong_AsLong(item);
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
    long val = PyLong_AsLong(item);
    if ((val == -1) && PyErr_Occurred()) {
      *error = "Error converting python int to long for int32.";
      return false;
    }
    // Range check
    if (val < std::numeric_limits<int32_t>::min() ||
        val > std::numeric_limits<int32_t>::max()) {
      *error = "Value out of int32 range.";
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
      long maybe_int_val = PyLong_AsLong(item);
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
    PyList_SetItem(py_list, i, PyLong_FromLong(static_cast<long>(data[i])));
  }
  return py_list;
}

// Creates a Python list from a span of int32_t values.
PyObject* BuildPyListFromInt32(absl::Span<const int32_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(static_cast<long>(data[i])));
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
}  // namespace

// Writes data to a tensor buffer.
PyObject* CompiledModelWrapper::WriteTensor(PyObject* buffer_capsule,
                                            PyObject* py_data,
                                            const std::string& dtype) {
  // Validate capsule
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("WriteTensor: invalid TensorBuffer capsule.");
  }
  void* ptr = PyCapsule_GetPointer(buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("WriteTensor: capsule pointer is null.");
  }
  litert::TensorBuffer tb((LiteRtTensorBuffer)ptr, false);

  // Handle different data types
  std::string error;
  if (dtype == "float32") {
    std::vector<float> host_data;
    if (!ConvertPyListToFloatVector(py_data, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tb.Write<float>(absl::MakeConstSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    Py_RETURN_NONE;

  } else if (dtype == "int8") {
    std::vector<int8_t> host_data;
    if (!ConvertPyListToInt8Vector(py_data, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tb.Write<int8_t>(absl::MakeConstSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    Py_RETURN_NONE;

  } else if (dtype == "int32") {
    std::vector<int32_t> host_data;
    if (!ConvertPyListToInt32Vector(py_data, &host_data, &error)) {
      if (!error.empty()) return ReportError(error);
    }
    auto status = tb.Write<int32_t>(absl::MakeConstSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    Py_RETURN_NONE;

  } else {
    return ReportError("WriteTensor: unsupported dtype '" + dtype + "'");
  }
}

// Reads data from a tensor buffer.
PyObject* CompiledModelWrapper::ReadTensor(PyObject* buffer_capsule,
                                           int num_elements,
                                           const std::string& dtype) {
  // Validate capsule
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("ReadTensor: invalid TensorBuffer capsule.");
  }
  void* ptr = PyCapsule_GetPointer(buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("ReadTensor: capsule pointer is null.");
  }
  litert::TensorBuffer tb((LiteRtTensorBuffer)ptr, false);

  // Handle different data types
  if (dtype == "float32") {
    std::vector<float> host_data(num_elements, 0.0f);
    auto status = tb.Read<float>(absl::MakeSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    return BuildPyListFromFloat(host_data);

  } else if (dtype == "int8") {
    std::vector<int8_t> host_data(num_elements, 0);
    auto status = tb.Read<int8_t>(absl::MakeSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    return BuildPyListFromInt8(host_data);

  } else if (dtype == "int32") {
    std::vector<int32_t> host_data(num_elements, 0);
    auto status = tb.Read<int32_t>(absl::MakeSpan(host_data));
    if (!status) {
      return ConvertErrorToPyExc(status.Error());
    }
    return BuildPyListFromInt32(host_data);

  } else {
    return ReportError("ReadTensor: unsupported dtype '" + dtype + "'");
  }
}

// Returns the byte width of a data type.
size_t CompiledModelWrapper::ByteWidthOfDType(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "int8") return 1;
  if (dtype == "int32") return 4;
  return 0;  // Unknown type
}

// Creates a tensor buffer from Python memory.
PyObject* CompiledModelWrapper::CreateTensorBufferFromMemory(
    const char* signature_key, const char* tensor_name, PyObject* py_data,
    const std::string& dtype) {
  // Get input buffer requirements
  auto sig_idx_or = model_.GetSignatureIndex(signature_key);
  if (!sig_idx_or) {
    return ReportError("Signature key not found: " +
                       std::string(signature_key));
  }
  auto buffer_req_or =
      compiled_model_.GetInputBufferRequirements(*sig_idx_or, tensor_name);
  if (!buffer_req_or) {
    return ConvertErrorToPyExc(buffer_req_or.Error());
  }
  auto buffer_req = std::move(*buffer_req_or);

  // Acquire Python buffer
  Py_buffer py_buf;
  if (PyObject_GetBuffer(py_data, &py_buf, PyBUF_CONTIG_RO) < 0) {
    return nullptr;  // Error already set by Python
  }

  // Validate buffer size
  size_t dtype_size = ByteWidthOfDType(dtype);
  if (dtype_size == 0) {
    PyBuffer_Release(&py_buf);
    return ReportError("Unsupported dtype in CreateTensorBufferFromMemory: " +
                       dtype);
  }
  auto maybe_size = buffer_req.BufferSize();
  if (!maybe_size) {
    PyBuffer_Release(&py_buf);
    return ConvertErrorToPyExc(maybe_size.Error());
  }
  size_t required_size = *maybe_size;

  if ((size_t)py_buf.len < required_size) {
    PyBuffer_Release(&py_buf);
    return ReportError("Python buffer smaller than required size: " +
                       std::to_string(py_buf.len) + " vs " +
                       std::to_string(required_size));
  }

  // Create tensor type descriptor
  LiteRtRankedTensorType dummy_type;
  size_t num_elements = required_size / dtype_size;

  if (dtype == "float32") {
    dummy_type.element_type = kLiteRtElementTypeFloat32;
  } else if (dtype == "int8") {
    dummy_type.element_type = kLiteRtElementTypeInt8;
  } else if (dtype == "int32") {
    dummy_type.element_type = kLiteRtElementTypeInt32;
  } else {
    dummy_type.element_type = kLiteRtElementTypeNone;
  }
  dummy_type.layout.rank = 1;
  dummy_type.layout.dimensions[0] = static_cast<int32_t>(num_elements);
  dummy_type.layout.strides = nullptr;

  // Create tensor buffer with no-op deallocator
  LiteRtTensorBuffer tensor_buffer = nullptr;
  auto status = LiteRtCreateTensorBufferFromHostMemory(
      &dummy_type, py_buf.buf, required_size,
      /* deallocator = */ &NoopDeallocator, &tensor_buffer);

  if (status != kLiteRtStatusOk) {
    PyBuffer_Release(&py_buf);
    return ReportError("LiteRtCreateTensorBufferFromHostMemory failed");
  }

  // Create context for capsule
  auto* ctx = new HostMemoryPyCapsuleContext();
  ctx->py_buf = py_buf;
  ctx->py_obj = py_data;
  ctx->c_tensor_buffer = tensor_buffer;

  // Keep Python object alive
  Py_INCREF(py_data);

  // Create capsule with destructor
  auto capsule_destructor = [](PyObject* capsule) {
    void* raw_ptr = PyCapsule_GetPointer(capsule, "LiteRtTensorBuffer");
    auto* ctx =
        static_cast<HostMemoryPyCapsuleContext*>(PyCapsule_GetContext(capsule));

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
    // Clean up on failure
    LiteRtDestroyTensorBuffer(tensor_buffer);
    PyBuffer_Release(&py_buf);
    Py_DECREF(py_data);
    delete ctx;
    return ReportError("Failed to create PyCapsule");
  }

  // Attach context to capsule
  PyCapsule_SetContext(capsule, ctx);

  return capsule;
}

// Constructor for CompiledModelWrapper.
CompiledModelWrapper::CompiledModelWrapper(litert::Environment env,
                                           litert::Model model,
                                           litert::CompiledModel compiled)
    : environment_(std::move(env)),
      model_(std::move(model)),
      compiled_model_(std::move(compiled)) {}

// Destructor for CompiledModelWrapper.
CompiledModelWrapper::~CompiledModelWrapper() = default;

// Reports an error by setting a Python exception.
PyObject* CompiledModelWrapper::ReportError(const std::string& msg) {
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  return nullptr;
}

// Converts a LiteRT error to a Python exception.
PyObject* CompiledModelWrapper::ConvertErrorToPyExc(
    const litert::Error& error) {
  PyErr_Format(PyExc_RuntimeError, "CompiledModel error: code=%d, message=%s",
               static_cast<int>(error.Status()), error.Message().c_str());
  return nullptr;
}

// Creates a CompiledModelWrapper from a model file.
CompiledModelWrapper* CompiledModelWrapper::CreateWrapperFromFile(
    const char* model_path, const char* compiler_plugin_path,
    const char* dispatch_library_path, int hardware_accel,
    std::string* out_error) {
  // Create environment with options
  std::vector<litert::Environment::Option> options;
  if (compiler_plugin_path && *compiler_plugin_path) {
    options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::CompilerPluginLibraryDir,
        std::string(compiler_plugin_path)});
  }
  if (dispatch_library_path && *dispatch_library_path) {
    options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        std::string(dispatch_library_path)});
  }
  auto env_or = litert::Environment::Create(options);
  if (!env_or) {
    if (out_error) *out_error = env_or.Error().Message();
    return nullptr;
  }
  litert::Environment env = std::move(*env_or);

  // Load model from file
  auto model_or = litert::Model::CreateFromFile(model_path);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  litert::Model model = std::move(*model_or);

  // Create compiled model
  auto compiled_or = litert::CompiledModel::Create(
      env, model, (LiteRtHwAccelerators)hardware_accel);
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  return new CompiledModelWrapper(std::move(env), std::move(model),
                                  std::move(*compiled_or));
}

// Converts a Python string or bytes object to a C string.
int ConvertFromPyString(PyObject* obj, char** data, Py_ssize_t* length) {
  if (PyUnicode_Check(obj)) {
    *data = const_cast<char*>(PyUnicode_AsUTF8AndSize(obj, length));
    return *data == nullptr ? -1 : 0;
  }
  return PyBytes_AsStringAndSize(obj, data, length);
}

// Creates a CompiledModelWrapper from a model buffer.
CompiledModelWrapper* CompiledModelWrapper::CreateWrapperFromBuffer(
    PyObject* model_data, const char* compiler_plugin_path,
    const char* dispatch_library_path, int hardware_accel,
    std::string* out_error) {
  // Extract buffer from Python object
  char* buf = nullptr;
  Py_ssize_t length = 0;
  if (ConvertFromPyString(model_data, &buf, &length) == -1) {
    if (out_error) *out_error = "Failed converting PyObject to buffer";
    return nullptr;
  }

  // Create environment with options
  std::vector<litert::Environment::Option> options;
  if (compiler_plugin_path && *compiler_plugin_path) {
    options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::CompilerPluginLibraryDir,
        std::string(compiler_plugin_path)});
  }
  if (dispatch_library_path && *dispatch_library_path) {
    options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        std::string(dispatch_library_path)});
  }

  auto env_or = litert::Environment::Create(options);
  if (!env_or) {
    if (out_error) *out_error = env_or.Error().Message();
    return nullptr;
  }
  litert::Environment env = std::move(*env_or);

  // Create model from buffer
  litert::BufferRef<uint8_t> ref(reinterpret_cast<uint8_t*>(buf),
                                 static_cast<size_t>(length));
  auto model_or = litert::Model::CreateFromBuffer(ref);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  litert::Model model = std::move(*model_or);

  // Create compiled model
  auto compiled_or = litert::CompiledModel::Create(
      env, model, static_cast<LiteRtHwAccelerators>(hardware_accel));
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  return new CompiledModelWrapper(std::move(env), std::move(model),
                                  std::move(*compiled_or));
}

// Returns a dictionary of all signatures in the model.
PyObject* CompiledModelWrapper::GetSignatureList() {
  auto sigs_or = model_.GetSignatures();
  if (!sigs_or) {
    return ConvertErrorToPyExc(sigs_or.Error());
  }
  auto sigs = std::move(*sigs_or);
  PyObject* py_dict = PyDict_New();

  for (size_t i = 0; i < sigs.size(); ++i) {
    const auto& sig = sigs[i];
    PyObject* sig_info = PyDict_New();

    // Add input names
    PyObject* py_in = PyList_New(0);
    for (auto& n : sig.InputNames()) {
      PyList_Append(py_in, PyUnicode_FromString(n.data()));
    }

    // Add output names
    PyObject* py_out = PyList_New(0);
    for (auto& n : sig.OutputNames()) {
      PyList_Append(py_out, PyUnicode_FromString(n.data()));
    }

    PyDict_SetItemString(sig_info, "inputs", py_in);
    PyDict_SetItemString(sig_info, "outputs", py_out);

    Py_DECREF(py_in);
    Py_DECREF(py_out);

    // Add signature to root dictionary
    PyDict_SetItemString(py_dict, sig.Key().data(), sig_info);
    Py_DECREF(sig_info);
  }
  return py_dict;
}

// Returns details about a signature by index.
PyObject* CompiledModelWrapper::GetSignatureByIndex(int signature_index) {
  auto sig_or = model_.GetSignature(signature_index);
  if (!sig_or) {
    return ConvertErrorToPyExc(sig_or.Error());
  }
  auto sig = std::move(*sig_or);

  PyObject* result = PyDict_New();
  // Add signature key
  PyDict_SetItemString(result, "key", PyUnicode_FromString(sig.Key().data()));

  // Add input names
  {
    PyObject* py_in = PyList_New(0);
    for (auto& nm : sig.InputNames()) {
      PyList_Append(py_in, PyUnicode_FromString(nm.data()));
    }
    PyDict_SetItemString(result, "inputs", py_in);
    Py_DECREF(py_in);
  }

  // Add output names
  {
    PyObject* py_out = PyList_New(0);
    for (auto& nm : sig.OutputNames()) {
      PyList_Append(py_out, PyUnicode_FromString(nm.data()));
    }
    PyDict_SetItemString(result, "outputs", py_out);
    Py_DECREF(py_out);
  }

  return result;
}

// Returns the number of signatures in the model.
PyObject* CompiledModelWrapper::GetNumSignatures() {
  auto num = model_.GetNumSignatures();
  return PyLong_FromLong((int64_t)num);
}

// Returns the index of a signature by key.
PyObject* CompiledModelWrapper::GetSignatureIndex(const char* signature_key) {
  auto idx_or = model_.GetSignatureIndex(signature_key);
  if (!idx_or) {
    // Return -1 if not found
    return PyLong_FromLong(-1);
  }
  return PyLong_FromLong((long)(*idx_or));
}

// Returns requirements for an input buffer.
PyObject* CompiledModelWrapper::GetInputBufferRequirements(int signature_index,
                                                           int input_index) {
  auto req_or = compiled_model_.GetInputBufferRequirements(
      (size_t)signature_index, (size_t)input_index);
  if (!req_or) {
    return ConvertErrorToPyExc(req_or.Error());
  }
  auto req = std::move(*req_or);
  PyObject* dict = PyDict_New();

  // Add buffer size
  auto size_or = req.BufferSize();
  if (!size_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(size_or.Error());
  }
  PyDict_SetItemString(dict, "buffer_size", PyLong_FromLong((int64_t)*size_or));

  // Add supported types
  auto types_or = req.SupportedTypes();
  if (!types_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(types_or.Error());
  }
  auto types = std::move(*types_or);
  PyObject* py_list = PyList_New((Py_ssize_t)types.size());
  for (size_t i = 0; i < types.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(types[i]));
  }
  PyDict_SetItemString(dict, "supported_types", py_list);
  Py_DECREF(py_list);

  return dict;
}

// Returns requirements for an output buffer.
PyObject* CompiledModelWrapper::GetOutputBufferRequirements(int signature_index,
                                                            int output_index) {
  auto req_or = compiled_model_.GetOutputBufferRequirements(
      (size_t)signature_index, (size_t)output_index);
  if (!req_or) {
    return ConvertErrorToPyExc(req_or.Error());
  }
  auto req = std::move(*req_or);

  PyObject* dict = PyDict_New();
  
  // Add buffer size
  auto size_or = req.BufferSize();
  if (!size_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(size_or.Error());
  }
  PyDict_SetItemString(dict, "buffer_size", PyLong_FromLong((int64_t)*size_or));

  auto types_or = req.SupportedTypes();
  if (!types_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(types_or.Error());
  }
  auto types = std::move(*types_or);
  PyObject* py_list = PyList_New((Py_ssize_t)types.size());
  for (size_t i = 0; i < types.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(types[i]));
  }
  PyDict_SetItemString(dict, "supported_types", py_list);
  Py_DECREF(py_list);

  return dict;
}

PyObject* CompiledModelWrapper::CreateInputBufferByName(
    const char* signature_key, const char* input_name) {
  auto buffer_or = compiled_model_.CreateInputBuffer(signature_key, input_name);
  if (!buffer_or) {
    return ConvertErrorToPyExc(buffer_or.Error());
  }
  auto buffer = std::move(*buffer_or);

  PyObject* capsule =
      MakePyCapsuleForTensorBuffer(buffer, /*owned_by_python=*/true);
  return capsule;
}

PyObject* CompiledModelWrapper::CreateOutputBufferByName(
    const char* signature_key, const char* output_name) {
  auto buffer_or =
      compiled_model_.CreateOutputBuffer(signature_key, output_name);
  if (!buffer_or) {
    return ConvertErrorToPyExc(buffer_or.Error());
  }
  auto buffer = std::move(*buffer_or);

  PyObject* capsule =
      MakePyCapsuleForTensorBuffer(buffer, /*owned_by_python=*/true);
  return capsule;
}

PyObject* CompiledModelWrapper::CreateInputBuffers(int signature_index) {
  auto buffers_or = compiled_model_.CreateInputBuffers((size_t)signature_index);
  if (!buffers_or) {
    return ConvertErrorToPyExc(buffers_or.Error());
  }
  auto buffers = std::move(*buffers_or);
  PyObject* py_list = PyList_New(buffers.size());
  for (size_t i = 0; i < buffers.size(); i++) {
    // Python owns them. Destroy on capsule destructor.
    PyObject* capsule = MakePyCapsuleForTensorBuffer(buffers[i], true);
    PyList_SetItem(py_list, i, capsule);  // steals ref
  }
  return py_list;
}

PyObject* CompiledModelWrapper::CreateOutputBuffers(int signature_index) {
  auto buffers_or =
      compiled_model_.CreateOutputBuffers((size_t)signature_index);
  if (!buffers_or) {
    return ConvertErrorToPyExc(buffers_or.Error());
  }
  auto buffers = std::move(*buffers_or);
  PyObject* py_list = PyList_New(buffers.size());
  for (size_t i = 0; i < buffers.size(); i++) {
    PyObject* capsule = MakePyCapsuleForTensorBuffer(buffers[i], true);
    PyList_SetItem(py_list, i, capsule);
  }
  return py_list;
}

PyObject* CompiledModelWrapper::RunByName(const char* signature_key,
                                          PyObject* input_map,
                                          PyObject* output_map) {
  if (!PyDict_Check(input_map) || !PyDict_Check(output_map)) {
    return ReportError("RunByName expects input_map & output_map as dict");
  }

  absl::flat_hash_map<absl::string_view, litert::TensorBuffer> in_map;
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer> out_map;

  PyObject* key;
  PyObject* val;
  Py_ssize_t pos = 0;
  while (PyDict_Next(input_map, &pos, &key, &val)) {
    if (!PyUnicode_Check(key)) {
      return ReportError("input_map key not a string.");
    }
    const char* nm = PyUnicode_AsUTF8(key);

    if (!PyCapsule_CheckExact(val)) {
      return ReportError("input_map value not a capsule.");
    }
    void* ptr = PyCapsule_GetPointer(val, "LiteRtTensorBuffer");
    if (!ptr) {
      return ReportError("capsule missing pointer in input_map");
    }
    in_map[nm] = litert::TensorBuffer((LiteRtTensorBuffer)ptr, false);
  }

  pos = 0;
  while (PyDict_Next(output_map, &pos, &key, &val)) {
    if (!PyUnicode_Check(key)) {
      return ReportError("output_map key not a string.");
    }
    const char* nm = PyUnicode_AsUTF8(key);

    if (!PyCapsule_CheckExact(val)) {
      return ReportError("output_map value not a capsule.");
    }
    void* ptr = PyCapsule_GetPointer(val, "LiteRtTensorBuffer");
    if (!ptr) {
      return ReportError("capsule missing pointer in output_map");
    }
    out_map[nm] = litert::TensorBuffer((LiteRtTensorBuffer)ptr, false);
  }

  auto run_or = compiled_model_.Run(signature_key, in_map, out_map);
  if (!run_or) {
    return ConvertErrorToPyExc(run_or.Error());
  }
  Py_RETURN_NONE;
}

PyObject* CompiledModelWrapper::RunByIndex(int signature_index,
                                           PyObject* input_caps_list,
                                           PyObject* output_caps_list) {
  if (!PyList_Check(input_caps_list)) {
    return ReportError("RunByIndex input_caps_list not list");
  }
  if (!PyList_Check(output_caps_list)) {
    return ReportError("RunByIndex output_caps_list not list");
  }
  std::vector<litert::TensorBuffer> inputs;
  std::vector<litert::TensorBuffer> outputs;

  Py_ssize_t n_in = PyList_Size(input_caps_list);
  inputs.reserve(n_in);
  for (Py_ssize_t i = 0; i < n_in; i++) {
    PyObject* elem = PyList_GetItem(input_caps_list, i);  // borrowed
    if (!PyCapsule_CheckExact(elem)) {
      return ReportError("input_caps_list element not a capsule");
    }
    void* ptr = PyCapsule_GetPointer(elem, "LiteRtTensorBuffer");
    if (!ptr) {
      return ReportError("Missing pointer in input capsule");
    }
    inputs.emplace_back((LiteRtTensorBuffer)ptr, false);
  }

  Py_ssize_t n_out = PyList_Size(output_caps_list);
  outputs.reserve(n_out);
  for (Py_ssize_t i = 0; i < n_out; i++) {
    PyObject* elem = PyList_GetItem(output_caps_list, i);
    if (!PyCapsule_CheckExact(elem)) {
      return ReportError("output_caps_list element not a capsule");
    }
    void* ptr = PyCapsule_GetPointer(elem, "LiteRtTensorBuffer");
    if (!ptr) {
      return ReportError("Missing pointer in output capsule");
    }
    outputs.emplace_back((LiteRtTensorBuffer)ptr, false);
  }

  auto run_or = compiled_model_.Run((size_t)signature_index, inputs, outputs);
  if (!run_or) {
    return ConvertErrorToPyExc(run_or.Error());
  }
  Py_RETURN_NONE;
}

PyObject* CompiledModelWrapper::WriteFloatTensor(
    PyObject* tensor_buffer_capsule, PyObject* float_list) {
  if (!PyCapsule_CheckExact(tensor_buffer_capsule)) {
    return ReportError(
        "WriteFloatTensor requires a valid PyCapsule for the buffer.");
  }
  void* ptr = PyCapsule_GetPointer(tensor_buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("Invalid capsule pointer in WriteFloatTensor.");
  }
  litert::TensorBuffer buffer((LiteRtTensorBuffer)ptr, false);

  std::vector<float> host_data;
  if (PyList_Check(float_list)) {
    Py_ssize_t length = PyList_Size(float_list);
    host_data.reserve(length);
    for (Py_ssize_t i = 0; i < length; i++) {
      PyObject* item = PyList_GetItem(float_list, i);
      float val = static_cast<float>(PyFloat_AsDouble(item));
      if (PyErr_Occurred()) {
        return ReportError(
            "Non-float value in float_list for WriteFloatTensor.");
      }
      host_data.push_back(val);
    }
  } else {
    return ReportError(
        "WriteFloatTensor currently only supports a Python list of float.");
  }

  auto status = buffer.Write<float>(absl::MakeConstSpan(host_data));
  if (!status) {
    return ConvertErrorToPyExc(status.Error());
  }

  Py_RETURN_NONE;
}

PyObject* CompiledModelWrapper::ReadFloatTensor(PyObject* tensor_buffer_capsule,
                                                int num_floats) {
  if (!PyCapsule_CheckExact(tensor_buffer_capsule)) {
    return ReportError(
        "ReadFloatTensor requires a valid PyCapsule for the buffer.");
  }
  void* ptr = PyCapsule_GetPointer(tensor_buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("Invalid capsule pointer in ReadFloatTensor.");
  }
  litert::TensorBuffer buffer((LiteRtTensorBuffer)ptr, false);

  std::vector<float> host_data(num_floats);
  auto status = buffer.Read<float>(absl::MakeSpan(host_data));
  if (!status) {
    return ConvertErrorToPyExc(status.Error());
  }

  PyObject* py_list = PyList_New(num_floats);
  for (int i = 0; i < num_floats; i++) {
    PyList_SetItem(py_list, i, PyFloat_FromDouble(host_data[i]));
  }
  return py_list;
}

PyObject* CompiledModelWrapper::DestroyTensorBuffer(
    PyObject* tensor_buffer_capsule) {
  if (!PyCapsule_CheckExact(tensor_buffer_capsule)) {
    return ReportError("DestroyTensorBuffer requires a valid PyCapsule.");
  }
  void* ptr = PyCapsule_GetPointer(tensor_buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("Invalid pointer in DestroyTensorBuffer.");
  }

  LiteRtDestroyTensorBuffer((LiteRtTensorBuffer)ptr);

  // PyCapsule_SetPointer(tensor_buffer_capsule, nullptr);

  Py_RETURN_NONE;
}

}  // namespace compiled_model_wrapper
}  // namespace litert
