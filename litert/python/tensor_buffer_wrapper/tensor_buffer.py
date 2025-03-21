# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper import _pywrap_litert_tensor_buffer_wrapper as _tb


class TensorBuffer:
  """Python wrapper for LiteRtTensorBuffer.

  This class provides a high-level interface to the underlying C++
  LiteRtTensorBuffer
  implementation, allowing for creation, reading, writing, and management of
  tensor
  buffers in Python.
  """

  def __init__(self, capsule):
    """Initializes a TensorBuffer with the provided PyCapsule.

    Args:
      capsule: A PyCapsule containing a pointer to a LiteRtTensorBuffer.
    """
    self._capsule = capsule

  @classmethod
  def create_from_host_memory(cls, data, dtype: str, num_elements: int):
    """Creates a new TensorBuffer referencing existing host memory.

    Args:
      data: Python object supporting the buffer protocol (e.g., numpy.ndarray,
        bytes).
      dtype: String representation of the data type (e.g., "float32", "int8").
      num_elements: Number of elements in the tensor.

    Returns:
      A new TensorBuffer instance.
    """
    cap = _tb.CreateTensorBufferFromHostMemory(data, dtype, num_elements)
    return cls(cap)

  def write(self, data_list, dtype: str):
    """Writes data to this tensor buffer.

    Args:
      data_list: Python list containing values to write to the buffer.
      dtype: String representation of the data type (e.g., "float32", "int8").
    """
    _tb.WriteTensor(self._capsule, data_list, dtype)

  def read(self, num_elements: int, dtype: str):
    """Reads data from this tensor buffer.

    Args:
      num_elements: Number of elements to read.
      dtype: String representation of the data type (e.g., "float32", "int8").

    Returns:
      A Python list containing the tensor data.
    """
    return _tb.ReadTensor(self._capsule, num_elements, dtype)

  def destroy(self):
    """Explicitly releases resources associated with this tensor buffer.

    After calling this method, the tensor buffer is no longer valid for use.
    """
    _tb.DestroyTensorBuffer(self._capsule)
    self._capsule = None

  @property
  def capsule(self):
    """Returns the underlying PyCapsule for direct C++ interoperability.

    Returns:
      The PyCapsule containing the pointer to the LiteRtTensorBuffer.
    """
    return self._capsule
