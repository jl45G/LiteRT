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
// Utility: convert a `TensorBuffer` to a PyCapsule with optional destructor.
static void CapsuleTensorBufferDestructor(PyObject* capsule) {
  void* ptr = PyCapsule_GetPointer(capsule, "LiteRtTensorBuffer");
  if (ptr) {
    LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(ptr));
  }
}

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

// A small structure to hold the Py_buffer and a reference to the PyObject
// so we can PyBuffer_Release and Py_XDECREF in the final destructor.
struct HostMemoryDeallocatorContext {
  Py_buffer py_buf;
  PyObject* py_obj;  // We keep a ref to ensure it doesn't get GC'd
};
// This is called by LiteRt when it destroys the tensor buffer.
static void HostMemoryDeallocator(void* context) {
  if (!context) return;
  auto* ctx = static_cast<HostMemoryDeallocatorContext*>(context);
  // Release the buffer
  PyBuffer_Release(&ctx->py_buf);
  // Decrement our reference to the python object
  Py_XDECREF(ctx->py_obj);
  delete ctx;  // free the context
}
// Suppose you define a small struct to hold your pinned Py_buffer:
struct HostMemoryPyCapsuleContext {
  Py_buffer py_buf;
  PyObject* py_obj;                    // reference to the Python object
  LiteRtTensorBuffer c_tensor_buffer;  // handle to the C++ buffer
};

// No-op deallocator for the TFLite call
static void NoopDeallocator(void* /*unused*/) {
  // Does nothing
}

// Helper: read a Python list of Python ints into a std::vector of int8_t.
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
    long val = PyLong_AsLong(item);  // can be large
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

// Similar for int32
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
    long val = PyLong_AsLong(item);  // can exceed int32
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

// For float32
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
    // We treat any numeric type as float
    double val = PyFloat_AsDouble(item);
    if ((val == -1.0) && PyErr_Occurred()) {
      // Maybe it was an int, let's try PyLong_AsLong?
      PyErr_Clear();  // We'll try to parse as int
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

// The symmetrical approach for read_tensor, building a Python list from a typed
// vector
PyObject* BuildPyListFromInt8(absl::Span<const int8_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(static_cast<long>(data[i])));
  }
  return py_list;
}

PyObject* BuildPyListFromInt32(absl::Span<const int32_t> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i, PyLong_FromLong(static_cast<long>(data[i])));
  }
  return py_list;
}

PyObject* BuildPyListFromFloat(absl::Span<const float> data) {
  PyObject* py_list = PyList_New(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    PyList_SetItem(py_list, i,
                   PyFloat_FromDouble(static_cast<double>(data[i])));
  }
  return py_list;
}
}  // namespace

//----------------------------------------------
// Implementation of WriteTensor
//----------------------------------------------
PyObject* CompiledModelWrapper::WriteTensor(PyObject* buffer_capsule,
                                            PyObject* py_data,
                                            const std::string& dtype) {
  // 1) Check capsule
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("WriteTensor: invalid TensorBuffer capsule.");
  }
  void* ptr = PyCapsule_GetPointer(buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("WriteTensor: capsule pointer is null.");
  }
  litert::TensorBuffer tb((LiteRtTensorBuffer)ptr, false);

  // 2) Switch on dtype
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

//----------------------------------------------
// Implementation of ReadTensor
//----------------------------------------------
PyObject* CompiledModelWrapper::ReadTensor(PyObject* buffer_capsule,
                                           int num_elements,
                                           const std::string& dtype) {
  // 1) Check capsule
  if (!PyCapsule_CheckExact(buffer_capsule)) {
    return ReportError("ReadTensor: invalid TensorBuffer capsule.");
  }
  void* ptr = PyCapsule_GetPointer(buffer_capsule, "LiteRtTensorBuffer");
  if (!ptr) {
    return ReportError("ReadTensor: capsule pointer is null.");
  }
  litert::TensorBuffer tb((LiteRtTensorBuffer)ptr, false);

  // 2) Switch on dtype
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

size_t CompiledModelWrapper::ByteWidthOfDType(const std::string& dtype) {
  if (dtype == "float32") return 4;
  if (dtype == "int8") return 1;
  if (dtype == "int32") return 4;
  // add more if needed...
  // fallback
  return 0;
}

PyObject* CompiledModelWrapper::CreateTensorBufferFromMemory(
    const char* signature_key, const char* tensor_name, PyObject* py_data,
    const std::string& dtype) {
  // 1) Get the input buffer requirements
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

  // 2) Acquire a Python buffer
  Py_buffer py_buf;
  if (PyObject_GetBuffer(py_data, &py_buf, PyBUF_CONTIG_RO) < 0) {
    // error is already set
    return nullptr;
  }

  // 3) Check size
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

  // 4) Build a minimal LiteRtRankedTensorType
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

  // 5) Create the buffer with a no-op deallocator
  //    because we have no 'deallocator_context' param.
  LiteRtTensorBuffer tensor_buffer = nullptr;
  auto status = LiteRtCreateTensorBufferFromHostMemory(
      &dummy_type, py_buf.buf, required_size,
      /* deallocator = */ &NoopDeallocator, &tensor_buffer);

  if (status != kLiteRtStatusOk) {
    PyBuffer_Release(&py_buf);
    return ReportError("LiteRtCreateTensorBufferFromHostMemory failed");
  }

  // 6) Make a small struct to hold references for the capsule destructor
  auto* ctx = new HostMemoryPyCapsuleContext();
  ctx->py_buf = py_buf;
  ctx->py_obj = py_data;
  ctx->c_tensor_buffer = tensor_buffer;

  // We need to keep the Python object alive for the buffer's lifetime.
  Py_INCREF(py_data);

  // 7) Create a PyCapsule. We'll store 'tensor_buffer' as the pointer
  //    and stash 'ctx' in the capsule's "context."
  //    Then the destructor can do the real cleanup (PyBuffer_Release,
  //    Py_DECREF, etc.)

  auto capsule_destructor = [](PyObject* capsule) {
    // 1. Extract the c_tensor_buffer from the capsule pointer
    void* raw_ptr = PyCapsule_GetPointer(capsule, "LiteRtTensorBuffer");
    // 2. Extract our context from the capsule context
    auto* ctx =
        static_cast<HostMemoryPyCapsuleContext*>(PyCapsule_GetContext(capsule));

    // 3. Destroy the TFLite buffer (triggers the no-op c-level deallocator)
    if (raw_ptr) {
      LiteRtDestroyTensorBuffer(static_cast<LiteRtTensorBuffer>(raw_ptr));
    }

    // 4. Now do the real cleanup: unpin memory, drop Python ref
    if (ctx) {
      PyBuffer_Release(&ctx->py_buf);
      Py_DECREF(ctx->py_obj);
      delete ctx;
    }
  };

  // Create the capsule
  PyObject* capsule =
      PyCapsule_New(tensor_buffer, "LiteRtTensorBuffer", capsule_destructor);
  if (!capsule) {
    // Capsule creation failed, free everything
    LiteRtDestroyTensorBuffer(tensor_buffer);
    PyBuffer_Release(&py_buf);
    Py_DECREF(py_data);
    delete ctx;
    return ReportError("Failed to create PyCapsule");
  }

  // Attach our context to the capsule
  PyCapsule_SetContext(capsule, ctx);

  return capsule;
}

CompiledModelWrapper::CompiledModelWrapper(litert::Environment env,
                                           litert::Model model,
                                           litert::CompiledModel compiled)
    : environment_(std::move(env)),
      model_(std::move(model)),
      compiled_model_(std::move(compiled)) {}

CompiledModelWrapper::~CompiledModelWrapper() = default;

PyObject* CompiledModelWrapper::ReportError(const std::string& msg) {
  PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  return nullptr;
}

PyObject* CompiledModelWrapper::ConvertErrorToPyExc(
    const litert::Error& error) {
  PyErr_Format(PyExc_RuntimeError, "CompiledModel error: code=%d, message=%s",
               static_cast<int>(error.Status()), error.Message().c_str());
  return nullptr;
}

/*static*/
CompiledModelWrapper* CompiledModelWrapper::CreateWrapperFromFile(
    const char* model_path, const char* compiler_plugin_path,
    const char* dispatch_library_path, int hardware_accel,
    std::string* out_error) {
  // 1) Make environment
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

  // 2) Load model
  auto model_or = litert::Model::CreateFromFile(model_path);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  litert::Model model = std::move(*model_or);

  // 3) Create compiled model
  auto compiled_or = litert::CompiledModel::Create(
      env, model, (LiteRtHwAccelerators)hardware_accel);
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  // Build final wrapper
  return new CompiledModelWrapper(std::move(env), std::move(model),
                                  std::move(*compiled_or));
}

int ConvertFromPyString(PyObject* obj, char** data, Py_ssize_t* length) {
  if (PyUnicode_Check(obj)) {
    // const_cast<> is for CPython 3.7 finally adding const to the API.
    *data = const_cast<char*>(PyUnicode_AsUTF8AndSize(obj, length));
    return *data == nullptr ? -1 : 0;
  }
  return PyBytes_AsStringAndSize(obj, data, length);
}

/*static*/
CompiledModelWrapper* CompiledModelWrapper::CreateWrapperFromBuffer(
    PyObject* model_data, const char* compiler_plugin_path,
    const char* dispatch_library_path, int hardware_accel,
    std::string* out_error) {
  char* buf = nullptr;
  Py_ssize_t length = 0;
  if (ConvertFromPyString(model_data, &buf, &length) == -1) {
    if (out_error) *out_error = "Failed converting PyObject to buffer";
    return nullptr;
  }

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

  // 2) Build Model from buffer
  litert::BufferRef<uint8_t> ref(reinterpret_cast<uint8_t*>(buf),
                                 static_cast<size_t>(length));
  auto model_or = litert::Model::CreateFromBuffer(ref);
  if (!model_or) {
    if (out_error) *out_error = model_or.Error().Message();
    return nullptr;
  }
  litert::Model model = std::move(*model_or);

  // 3) Create compiled
  auto compiled_or = litert::CompiledModel::Create(
      env, model, static_cast<LiteRtHwAccelerators>(hardware_accel));
  if (!compiled_or) {
    if (out_error) *out_error = compiled_or.Error().Message();
    return nullptr;
  }

  return new CompiledModelWrapper(std::move(env), std::move(model),
                                  std::move(*compiled_or));
}

PyObject* CompiledModelWrapper::GetSignatureList() {
  auto sigs_or = model_.GetSignatures();
  if (!sigs_or) {
    return ConvertErrorToPyExc(sigs_or.Error());
  }
  auto sigs = std::move(*sigs_or);
  PyObject* py_dict = PyDict_New();

  for (size_t i = 0; i < sigs.size(); ++i) {
    const auto& sig = sigs[i];
    // signature key
    PyObject* sig_info = PyDict_New();  // dict for "inputs", "outputs"

    // gather input names
    PyObject* py_in = PyList_New(0);
    for (auto& n : sig.InputNames()) {
      PyList_Append(py_in, PyUnicode_FromString(n.data()));
    }

    // gather output names
    PyObject* py_out = PyList_New(0);
    for (auto& n : sig.OutputNames()) {
      PyList_Append(py_out, PyUnicode_FromString(n.data()));
    }

    PyDict_SetItemString(sig_info, "inputs", py_in);
    PyDict_SetItemString(sig_info, "outputs", py_out);

    Py_DECREF(py_in);
    Py_DECREF(py_out);

    // place in root dict: key is signature.Key(), value is sig_info
    PyDict_SetItemString(py_dict, sig.Key().data(), sig_info);
    Py_DECREF(sig_info);
  }
  return py_dict;
}

PyObject* CompiledModelWrapper::GetSignatureByIndex(int signature_index) {
  auto sig_or = model_.GetSignature(signature_index);
  if (!sig_or) {
    return ConvertErrorToPyExc(sig_or.Error());
  }
  auto sig = std::move(*sig_or);

  PyObject* result = PyDict_New();
  // store "key"
  PyDict_SetItemString(result, "key", PyUnicode_FromString(sig.Key().data()));

  // store "inputs" as list
  {
    PyObject* py_in = PyList_New(0);
    for (auto& nm : sig.InputNames()) {
      PyList_Append(py_in, PyUnicode_FromString(nm.data()));
    }
    PyDict_SetItemString(result, "inputs", py_in);
    Py_DECREF(py_in);
  }

  // store "outputs" as list
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

PyObject* CompiledModelWrapper::GetNumSignatures() {
  auto num = model_.GetNumSignatures();
  return PyLong_FromLong((int64_t)num);
}

PyObject* CompiledModelWrapper::GetSignatureIndex(const char* signature_key) {
  auto idx_or = model_.GetSignatureIndex(signature_key);
  if (!idx_or) {
    // return -1 if not found
    return PyLong_FromLong(-1);
  }
  return PyLong_FromLong((long)(*idx_or));
}

PyObject* CompiledModelWrapper::GetInputBufferRequirements(int signature_index,
                                                           int input_index) {
  auto req_or = compiled_model_.GetInputBufferRequirements(
      (size_t)signature_index, (size_t)input_index);
  if (!req_or) {
    return ConvertErrorToPyExc(req_or.Error());
  }
  auto req = std::move(*req_or);
  PyObject* dict = PyDict_New();

  // buffer size
  auto size_or = req.BufferSize();
  if (!size_or) {
    Py_DECREF(dict);
    return ConvertErrorToPyExc(size_or.Error());
  }
  PyDict_SetItemString(dict, "buffer_size", PyLong_FromLong((int64_t)*size_or));

  // supported types
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

PyObject* CompiledModelWrapper::GetOutputBufferRequirements(int signature_index,
                                                            int output_index) {
  auto req_or = compiled_model_.GetOutputBufferRequirements(
      (size_t)signature_index, (size_t)output_index);
  if (!req_or) {
    return ConvertErrorToPyExc(req_or.Error());
  }
  auto req = std::move(*req_or);

  PyObject* dict = PyDict_New();
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
