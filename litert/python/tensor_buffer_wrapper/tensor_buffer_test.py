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

"""Tests for TensorBuffer Python wrapper."""

import unittest
import numpy as np

from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import (
    TensorBuffer,
    BUFFER_TYPE_HOST,
    ELEMENT_TYPE_FLOAT32,
)


class TensorBufferTest(unittest.TestCase):
    """Tests for TensorBuffer Python wrapper."""

    def test_create_managed_float(self):
        """Test creating a managed float tensor buffer."""
        dimensions = [2, 3]
        buffer = TensorBuffer.create_managed(dimensions, dtype="float32")
        
        # Check buffer properties
        self.assertEqual(buffer.get_buffer_type(), BUFFER_TYPE_HOST)
        self.assertEqual(buffer.get_dtype(), "float32")
        self.assertEqual(buffer.get_shape(), dimensions)
        self.assertEqual(buffer.get_size(), 24)  # 2*3*4 bytes
        
        # Write and read data
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        buffer.write(data)
        read_data = buffer.read(6)
        self.assertEqual(read_data, data)

    def test_create_managed_int8(self):
        """Test creating a managed int8 tensor buffer."""
        dimensions = [2, 2]
        buffer = TensorBuffer.create_managed(dimensions, dtype="int8")
        
        # Check buffer properties
        self.assertEqual(buffer.get_dtype(), "int8")
        self.assertEqual(buffer.get_shape(), dimensions)
        self.assertEqual(buffer.get_size(), 4)  # 2*2*1 bytes
        
        # Write and read data
        data = [10, 20, 30, 40]
        buffer.write(data, dtype="int8")
        read_data = buffer.read(4, dtype="int8")
        self.assertEqual(read_data, data)

    def test_create_from_host_memory(self):
        """Test creating a tensor buffer from host memory."""
        dimensions = [2, 2]
        host_memory = bytearray(16)  # 16 bytes = 4 floats
        
        buffer = TensorBuffer.create_from_host_memory(
            host_memory, dimensions, dtype="float32"
        )
        
        # Check buffer properties
        self.assertEqual(buffer.get_dtype(), "float32")
        self.assertEqual(buffer.get_shape(), dimensions)
        
        # Write and read data
        data = [1.0, 2.0, 3.0, 4.0]
        buffer.write(data)
        read_data = buffer.read(4)
        self.assertEqual(read_data, data)

    def test_numpy_conversion(self):
        """Test conversion to numpy array."""
        dimensions = [2, 3]
        buffer = TensorBuffer.create_managed(dimensions, dtype="float32")
        
        # Write data
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        buffer.write(data)
        
        # Convert to numpy
        np_array = buffer.numpy()
        
        # Check shape and values
        self.assertEqual(np_array.shape, tuple(dimensions))
        self.assertEqual(np_array.dtype, np.dtype("float32"))
        
        # Reshape data to match dimensions for comparison
        expected = np.array(data, dtype="float32").reshape(dimensions)
        np.testing.assert_array_equal(np_array, expected)


if __name__ == "__main__":
    unittest.main()