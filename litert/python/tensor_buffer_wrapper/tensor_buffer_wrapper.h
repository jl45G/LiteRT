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

#ifndef LITERT_PYTHON_TENSOR_BUFFER_WRAPPER_TENSOR_BUFFER_WRAPPER_H_
#define LITERT_PYTHON_TENSOR_BUFFER_WRAPPER_TENSOR_BUFFER_WRAPPER_H_

#include <Python.h>

#include <string>

#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {
namespace tensor_buffer_wrapper {

/**
 * A helper class to manage creation, reading, and writing to a
 * LiteRtTensorBuffer, exposed via pybind to Python.
 *
 * The methods here implement the logic that was originally in
 * compiled_model_wrapper for read/write. We keep all that functionality here so
 * that it's easy to create and manipulate TensorBuffers from Python without
 * going through a model.
 */
class TensorBufferWrapper {
 public:
  // Creates a new Python capsule that owns a newly created LiteRtTensorBuffer
  // from host memory, with an optional deallocator if needed.
  static PyObject* CreateFromHostMemory(PyObject* py_data,
                                        const std::string& dtype,
                                        Py_ssize_t num_elements);

  // Writes Python data into the given TensorBuffer capsule.
  static PyObject* WriteTensor(PyObject* buffer_capsule, PyObject* data_list,
                               const std::string& dtype);

  // Reads data from the given TensorBuffer capsule into a Python list.
  static PyObject* ReadTensor(PyObject* buffer_capsule, int num_elements,
                              const std::string& dtype);

  // Explicitly destroy a TensorBuffer capsule (if the Python user wants).
  static PyObject* DestroyTensorBuffer(PyObject* buffer_capsule);

  // Provide error conversion helpers.
  static PyObject* ReportError(const std::string& msg);
  static PyObject* ConvertErrorToPyExc(const litert::Error& error);

 private:
  // Helper: convert dtype string to a known byte width, or zero if unknown.
  static size_t ByteWidthOfDType(const std::string& dtype);
};

}  // namespace tensor_buffer_wrapper
}  // namespace litert

#endif  // LITERT_PYTHON_TENSOR_BUFFER_WRAPPER_TENSOR_BUFFER_WRAPPER_H_
