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
 * Wrapper class for LiteRT TensorBuffer that provides Python bindings.
 *
 * This class manages the lifecycle of LiteRT TensorBuffer objects 
 * while exposing their functionality to Python.
 * It handles Python object conversions and error reporting.
 */
class TensorBufferWrapper {
 public:
  /**
   * Creates a wrapper from an existing LiteRT TensorBuffer.
   *
   * @param tensor_buffer An existing TensorBuffer
   * @param owned Whether the wrapper should take ownership of the TensorBuffer
   */
  TensorBufferWrapper(litert::TensorBuffer tensor_buffer, bool owned = true);

  // Destructor
  ~TensorBufferWrapper() = default;

  /**
   * Creates a managed tensor buffer.
   *
   * @param buffer_type The type of tensor buffer to create
   * @param element_type The element type for the tensor
   * @param dimensions A Python tuple or list of dimensions
   * @param buffer_size The size of the buffer in bytes
   * @param out_error String to store error message if creation fails
   */
  static TensorBufferWrapper* CreateManagedBuffer(
      int buffer_type, int element_type, PyObject* dimensions, 
      size_t buffer_size, std::string* out_error);

  /**
   * Creates a TensorBuffer that wraps existing host memory.
   *
   * @param element_type The element type for the tensor
   * @param dimensions A Python tuple or list of dimensions
   * @param host_memory_object Python object providing the memory (must support buffer protocol)
   * @param out_error String to store error message if creation fails
   */
  static TensorBufferWrapper* CreateFromHostMemory(
      int element_type, PyObject* dimensions, 
      PyObject* host_memory_object, std::string* out_error);

  // Returns the underlying LiteRT TensorBuffer handle (for use in C++ only)
  litert::TensorBuffer& GetBuffer() { return tensor_buffer_; }

  // Get the buffer type
  PyObject* GetBufferType();

  // Get the tensor type
  PyObject* GetTensorType();

  // Get the buffer size in bytes
  PyObject* GetSize();

  // Writes data to the tensor buffer.
  PyObject* WriteTensor(PyObject* py_data, const std::string& dtype);

  // Reads data from the tensor buffer.
  PyObject* ReadTensor(int num_elements, const std::string& dtype);

 private:
  // Reports an error to Python and returns nullptr.
  static PyObject* ReportError(const std::string& msg);

  // Converts a LiteRT error to a Python exception and returns nullptr.
  static PyObject* ConvertErrorToPyExc(const litert::Error& error);

  // Returns the size in bytes of a single element of the given data type.
  static size_t ByteWidthOfDType(const std::string& dtype);

  // Member variables
  litert::TensorBuffer tensor_buffer_;
};

}  // namespace tensor_buffer_wrapper
}  // namespace litert

#endif  // LITERT_PYTHON_TENSOR_BUFFER_WRAPPER_TENSOR_BUFFER_WRAPPER_H_