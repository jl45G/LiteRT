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

"""Python wrapper for LiteRT tensor buffers."""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper import _pywrap_litert_tensor_buffer_wrapper as _tb

# Buffer type constants
BUFFER_TYPE_HOST = 0
BUFFER_TYPE_OPENCL = 1
BUFFER_TYPE_OPENGL = 2

# Element type constants
ELEMENT_TYPE_NONE = 0
ELEMENT_TYPE_FLOAT32 = 1
ELEMENT_TYPE_INT32 = 2
ELEMENT_TYPE_UINT8 = 3
ELEMENT_TYPE_INT8 = 4
ELEMENT_TYPE_INT16 = 5
ELEMENT_TYPE_INT64 = 6
ELEMENT_TYPE_BOOL = 7
ELEMENT_TYPE_COMPLEX64 = 8
ELEMENT_TYPE_FLOAT16 = 9

# Map from string dtype to element type constants
_DTYPE_TO_ELEMENT_TYPE = {
    "float32": ELEMENT_TYPE_FLOAT32,
    "int32": ELEMENT_TYPE_INT32,
    "uint8": ELEMENT_TYPE_UINT8,
    "int8": ELEMENT_TYPE_INT8,
    "int16": ELEMENT_TYPE_INT16,
    "int64": ELEMENT_TYPE_INT64,
    "bool": ELEMENT_TYPE_BOOL,
    "complex64": ELEMENT_TYPE_COMPLEX64,
    "float16": ELEMENT_TYPE_FLOAT16,
}

# Map from element type constants to string dtype
_ELEMENT_TYPE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ELEMENT_TYPE.items()}


class TensorBuffer:
    """Python-friendly wrapper around LiteRT TensorBuffer.
    
    This class provides an easy-to-use Python interface for working with 
    LiteRT tensor buffers. It supports creating tensor buffers, reading and 
    writing tensor data, and getting information about the tensor buffer.
    """

    def __init__(self, wrapper_ptr):
        """Initializes a TensorBuffer from a C++ wrapper pointer.
        
        Args:
            wrapper_ptr: Pointer to a C++ TensorBufferWrapper object
        """
        self._buffer = wrapper_ptr

    @classmethod
    def create_managed(
        cls,
        dimensions: List[int],
        dtype: str = "float32",
        buffer_type: int = BUFFER_TYPE_HOST,
        buffer_size: Optional[int] = None,
    ) -> "TensorBuffer":
        """Creates a managed tensor buffer.
        
        Args:
            dimensions: Shape of the tensor
            dtype: Data type ("float32", "int8", or "int32")
            buffer_type: Type of buffer (host, OpenCL, OpenGL)
            buffer_size: Size in bytes. If None, calculated from dimensions and dtype
            
        Returns:
            A new TensorBuffer object
        
        Raises:
            ValueError: If dimensions or dtype are invalid
            RuntimeError: If buffer creation fails
        """
        if dtype not in _DTYPE_TO_ELEMENT_TYPE:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        element_type = _DTYPE_TO_ELEMENT_TYPE[dtype]
        
        # Calculate buffer size if not provided
        if buffer_size is None:
            element_size = 4  # Default for float32
            if dtype == "int8" or dtype == "uint8" or dtype == "bool":
                element_size = 1
            elif dtype == "int16" or dtype == "float16":
                element_size = 2
            elif dtype == "int64":
                element_size = 8
                
            # Calculate total elements
            total_elements = 1
            for dim in dimensions:
                total_elements *= dim
                
            buffer_size = total_elements * element_size
            
        wrapper = _tb.CreateManagedBuffer(
            buffer_type, element_type, dimensions, buffer_size
        )
        
        return cls(wrapper)

    @classmethod
    def create_from_host_memory(
        cls, host_memory: Any, dimensions: List[int], dtype: str = "float32"
    ) -> "TensorBuffer":
        """Creates a tensor buffer that wraps existing host memory.
        
        Args:
            host_memory: Memory object supporting buffer protocol (e.g., bytes, bytearray)
            dimensions: Shape of the tensor
            dtype: Data type of the tensor
            
        Returns:
            A new TensorBuffer object
            
        Raises:
            ValueError: If dimensions or dtype are invalid
            RuntimeError: If buffer creation fails
        """
        if dtype not in _DTYPE_TO_ELEMENT_TYPE:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        element_type = _DTYPE_TO_ELEMENT_TYPE[dtype]
        
        wrapper = _tb.CreateFromHostMemory(
            element_type, dimensions, host_memory
        )
        
        return cls(wrapper)

    def get_buffer_type(self) -> int:
        """Gets the buffer type.
        
        Returns:
            Type of the tensor buffer
        """
        return self._buffer.GetBufferType()

    def get_tensor_type(self) -> Dict[str, Any]:
        """Gets the tensor type information.
        
        Returns:
            Dictionary with element_type, rank, and dimensions
        """
        return self._buffer.GetTensorType()

    def get_size(self) -> int:
        """Gets the buffer size in bytes.
        
        Returns:
            Size of the tensor buffer in bytes
        """
        return self._buffer.GetSize()

    def get_shape(self) -> List[int]:
        """Gets the shape of the tensor.
        
        Returns:
            List of dimensions
        """
        tensor_type = self._buffer.GetTensorType()
        return tensor_type["dimensions"]

    def get_dtype(self) -> str:
        """Gets the data type of the tensor.
        
        Returns:
            String representation of the data type
        """
        tensor_type = self._buffer.GetTensorType()
        element_type = tensor_type["element_type"]
        
        if element_type in _ELEMENT_TYPE_TO_DTYPE:
            return _ELEMENT_TYPE_TO_DTYPE[element_type]
        else:
            return "unknown"

    def write(self, data: List[Union[int, float]], dtype: Optional[str] = None) -> None:
        """Writes data to the tensor buffer.
        
        Args:
            data: List of values to write
            dtype: Data type to use. If None, inferred from the tensor type
            
        Raises:
            ValueError: If dtype is invalid
            RuntimeError: If write operation fails
        """
        if dtype is None:
            dtype = self.get_dtype()
            
        if dtype not in ("float32", "int8", "int32"):
            raise ValueError(
                f"WriteTensor only supports float32, int8, and int32, got {dtype}"
            )
            
        self._buffer.WriteTensor(data, dtype)

    def read(self, num_elements: int, dtype: Optional[str] = None) -> List[Union[int, float]]:
        """Reads data from the tensor buffer.
        
        Args:
            num_elements: Number of elements to read
            dtype: Data type to use. If None, inferred from the tensor type
            
        Returns:
            List of values read from the buffer
            
        Raises:
            ValueError: If dtype is invalid
            RuntimeError: If read operation fails
        """
        if dtype is None:
            dtype = self.get_dtype()
            
        if dtype not in ("float32", "int8", "int32"):
            raise ValueError(
                f"ReadTensor only supports float32, int8, and int32, got {dtype}"
            )
            
        return self._buffer.ReadTensor(num_elements, dtype)

    def numpy(self) -> np.ndarray:
        """Converts the tensor buffer to a numpy array.
        
        Returns:
            Numpy array with the tensor data
            
        Raises:
            RuntimeError: If conversion fails
        """
        shape = self.get_shape()
        dtype = self.get_dtype()
        
        # Calculate total elements
        total_elements = 1
        for dim in shape:
            total_elements *= dim
            
        # Read data
        data = self.read(total_elements, dtype)
        
        # Convert to numpy array
        np_array = np.array(data, dtype=dtype)
        
        # Reshape to match tensor dimensions
        return np_array.reshape(shape)