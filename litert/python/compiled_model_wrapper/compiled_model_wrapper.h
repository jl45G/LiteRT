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

#ifndef LITERT_PYTHON_COMPILED_MODEL_WRAPPER_COMPILED_MODEL_WRAPPER_H_
#define LITERT_PYTHON_COMPILED_MODEL_WRAPPER_COMPILED_MODEL_WRAPPER_H_

#include <Python.h>

#include <memory>
#include <string>
#include <vector>

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
// #include
// "litert/python/compiled_model_wrapper/python_error_reporter.h"

namespace litert {
namespace compiled_model_wrapper {

/**
 * CompiledModelWrapper is a bridging class that:
 *   - Owns a LiteRT Environment, Model, and CompiledModel
 *   - Exposes all relevant C++ APIs as PyObject-returning methods
 *   - Provides Python-level error reporting
 */
class CompiledModelWrapper {
 public:
  static CompiledModelWrapper* CreateWrapperFromFile(
      const char* model_path, const char* compiler_plugin_path,
      const char* dispatch_library_path, int hardware_accel,
      std::string* out_error);

  static CompiledModelWrapper* CreateWrapperFromBuffer(
      PyObject* model_data, const char* compiler_plugin_path,
      const char* dispatch_library_path, int hardware_accel,
      std::string* out_error);

  CompiledModelWrapper(litert::Environment env, litert::Model model,
                       litert::CompiledModel compiled);

  ~CompiledModelWrapper();

  // Return a PyObject describing the model’s signatures.
  PyObject* GetSignatureList();

  // Retrieve the signature by index, for advanced introspection.
  PyObject* GetSignatureByIndex(int signature_index);

  // Retrieve the number of signatures
  PyObject* GetNumSignatures();

  // Retrieve subgraph index of a signature key, or -1 if not found
  PyObject* GetSignatureIndex(const char* signature_key);

  // Retrieve buffer requirements for an input (by signature index, input
  // index).
  PyObject* GetInputBufferRequirements(int signature_index, int input_index);

  // Retrieve buffer requirements for an output (by signature index, output
  // index).
  PyObject* GetOutputBufferRequirements(int signature_index, int output_index);

  // Overload: create input buffer by signature key & input name
  PyObject* CreateInputBufferByName(const char* signature_key,
                                    const char* input_name);

  // Overload: create output buffer by signature key & output name
  PyObject* CreateOutputBufferByName(const char* signature_key,
                                     const char* output_name);

  // Create input buffers for the entire subgraph. Returns list of capsules
  PyObject* CreateInputBuffers(int signature_index);

  // Create output buffers for the entire subgraph. Returns list of capsules
  PyObject* CreateOutputBuffers(int signature_index);

  // Actually run the model with signature key, passing input_map & output_map
  // (both dicts: name -> PyCapsule).
  PyObject* RunByName(const char* signature_key, PyObject* input_map,
                      PyObject* output_map);

  // Actually run the model with signature index. Takes vector capsules
  PyObject* RunByIndex(int signature_index, PyObject* input_caps_list,
                       PyObject* output_caps_list);

  // Write a Python list[float] into the specified TensorBuffer capsule.
  PyObject* WriteFloatTensor(PyObject* tensor_buffer_capsule,
                             PyObject* float_list);

  // Read back the contents of a TensorBuffer as floats. The user must specify
  // how many floats to read.
  PyObject* ReadFloatTensor(PyObject* tensor_buffer_capsule, int num_floats);

  // If you want to explicitly free/destroy a buffer in Python, call this.
  PyObject* DestroyTensorBuffer(PyObject* tensor_buffer_capsule);

  PyObject* CreateTensorBufferFromMemory(const char* signature_key,
                                         const char* tensor_name,
                                         PyObject* py_data,
                                         const std::string& dtype);

  // Existing methods ...

  // Single method that routes to appropriate typed logic
  PyObject* WriteTensor(PyObject* buffer_capsule, PyObject* data,
                        const std::string& dtype);
  PyObject* ReadTensor(PyObject* buffer_capsule, int num_elements,
                       const std::string& dtype);

 private:
  // Possibly some helper to compute element size from dtype, etc.
  size_t ByteWidthOfDType(const std::string& dtype);
  static PyObject* ReportError(const std::string& msg);
  // Simple internal helper: convert litert::Error -> Python exception
  static PyObject* ConvertErrorToPyExc(const litert::Error& error);

  // store references
  litert::Environment environment_;
  litert::Model model_;
  litert::CompiledModel compiled_model_;
};

}  // namespace compiled_model_wrapper
}  // namespace litert

#endif  // LITERT_PYTHON_COMPILED_MODEL_WRAPPER_COMPILED_MODEL_WRAPPER_H_
