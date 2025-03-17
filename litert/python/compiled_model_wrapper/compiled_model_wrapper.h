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
 * Wrapper class for LiteRT models that provides Python bindings.
 *
 * This class manages the lifecycle of LiteRT Environment, Model, and 
 * CompiledModel objects while exposing their functionality to Python.
 * It handles Python object conversions and error reporting.
 */
class CompiledModelWrapper {
 public:
  // Creates a wrapper from a model file path.
  static CompiledModelWrapper* CreateWrapperFromFile(
      const char* model_path, const char* compiler_plugin_path,
      const char* dispatch_library_path, int hardware_accel,
      std::string* out_error);

  // Creates a wrapper from a model buffer in memory.
  static CompiledModelWrapper* CreateWrapperFromBuffer(
      PyObject* model_data, const char* compiler_plugin_path,
      const char* dispatch_library_path, int hardware_accel,
      std::string* out_error);

  CompiledModelWrapper(litert::Environment env, litert::Model model,
                       litert::CompiledModel compiled);

  ~CompiledModelWrapper();

  // Returns a Python object containing the model's signatures.
  PyObject* GetSignatureList();

  // Returns a Python object with details about the signature at the given index.
  PyObject* GetSignatureByIndex(int signature_index);

  // Returns the number of signatures in the model.
  PyObject* GetNumSignatures();

  // Returns the index of a signature by key, or -1 if not found.
  PyObject* GetSignatureIndex(const char* signature_key);

  // Returns buffer requirements for an input tensor.
  PyObject* GetInputBufferRequirements(int signature_index, int input_index);

  // Returns buffer requirements for an output tensor.
  PyObject* GetOutputBufferRequirements(int signature_index, int output_index);

  // Creates an input buffer for a tensor identified by signature key and input name.
  PyObject* CreateInputBufferByName(const char* signature_key,
                                    const char* input_name);

  // Creates an output buffer for a tensor identified by signature key and output name.
  PyObject* CreateOutputBufferByName(const char* signature_key,
                                     const char* output_name);

  // Creates all input buffers for a signature and returns them as a list of capsules.
  PyObject* CreateInputBuffers(int signature_index);

  // Creates all output buffers for a signature and returns them as a list of capsules.
  PyObject* CreateOutputBuffers(int signature_index);

  // Executes the model using a signature key and name-to-buffer mappings.
  PyObject* RunByName(const char* signature_key, PyObject* input_map,
                      PyObject* output_map);

  // Executes the model using a signature index and lists of buffer capsules.
  PyObject* RunByIndex(int signature_index, PyObject* input_caps_list,
                       PyObject* output_caps_list);

  // Writes a Python list of floats to a tensor buffer.
  PyObject* WriteFloatTensor(PyObject* tensor_buffer_capsule,
                             PyObject* float_list);

  // Reads float values from a tensor buffer.
  PyObject* ReadFloatTensor(PyObject* tensor_buffer_capsule, int num_floats);

  // Explicitly frees a tensor buffer.
  PyObject* DestroyTensorBuffer(PyObject* tensor_buffer_capsule);

  // Creates a tensor buffer from Python data in memory.
  PyObject* CreateTensorBufferFromMemory(const char* signature_key,
                                         const char* tensor_name,
                                         PyObject* py_data,
                                         const std::string& dtype);

  // Writes data to a tensor buffer with the specified data type.
  PyObject* WriteTensor(PyObject* buffer_capsule, PyObject* data,
                        const std::string& dtype);

  // Reads data from a tensor buffer with the specified data type.
  PyObject* ReadTensor(PyObject* buffer_capsule, int num_elements,
                       const std::string& dtype);

 private:
  // Returns the size in bytes of a single element of the given data type.
  size_t ByteWidthOfDType(const std::string& dtype);

  // Reports an error to Python and returns nullptr.
  static PyObject* ReportError(const std::string& msg);

  // Converts a LiteRT error to a Python exception and returns nullptr.
  static PyObject* ConvertErrorToPyExc(const litert::Error& error);

  // Member variables holding the LiteRT objects.
  litert::Environment environment_;
  litert::Model model_;
  litert::CompiledModel compiled_model_;
};

}  // namespace compiled_model_wrapper
}  // namespace litert

#endif  // LITERT_PYTHON_COMPILED_MODEL_WRAPPER_COMPILED_MODEL_WRAPPER_H_
