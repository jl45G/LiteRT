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

"""Unit tests for tensor_buffer.py."""

import unittest
import numpy as np

from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import TensorBuffer


def aligned_array(shape, dtype, alignment=64):
  """Allocates a NumPy array with the specified memory alignment.

  Creates a NumPy array where the underlying memory is aligned to the specified
  byte boundary, which is useful for hardware-specific optimizations.

  Args:
    shape: The shape of the array.
    dtype: The data type of the array.
    alignment: The memory alignment in bytes (default: 64).

  Returns:
    A NumPy array with the specified alignment.
  """
  size = np.prod(shape)
  itemsize = np.dtype(dtype).itemsize
  n_extra = alignment // itemsize

  # Create a 1D array with extra space to find aligned memory
  raw = np.zeros(size + n_extra, dtype=dtype)

  # Find the first aligned position
  start_index = 0
  while (raw[start_index:].ctypes.data % alignment) != 0:
    start_index += 1

  # Return the aligned slice reshaped to the requested dimensions
  aligned = raw[start_index : start_index + size].reshape(shape)
  return aligned


class TensorBufferTest(unittest.TestCase):
  """Test cases for the TensorBuffer class."""

  def test_create_from_host_memory_float(self):
    """Tests creating a TensorBuffer from host memory with float data."""
    # Create aligned numpy array
    test_data = [1.0, 2.0, 3.0, 4.0]
    arr = aligned_array((4,), np.float32)
    arr[:] = test_data

    # Create TensorBuffer referencing the array
    tb = TensorBuffer.create_from_host_memory(data=arr, dtype="float32", num_elements=4)
    
    # Verify data can be read back correctly
    result = tb.read(4, "float32")
    self.assertEqual(result, test_data)
    
    # Clean up
    tb.destroy()

  def test_create_from_host_memory_int(self):
    """Tests creating a TensorBuffer from host memory with integer data."""
    # Create aligned numpy array
    test_data = [10, 20, 30, 40]
    arr = aligned_array((4,), np.int32)
    arr[:] = test_data

    # Create TensorBuffer referencing the array
    tb = TensorBuffer.create_from_host_memory(data=arr, dtype="int32", num_elements=4)
    
    # Verify data can be read back correctly
    result = tb.read(4, "int32")
    self.assertEqual(result, test_data)
    
    # Clean up
    tb.destroy()

  def test_write_read_float(self):
    """Tests writing to and reading from a TensorBuffer with float data."""
    # Create data to write
    test_data = [1.5, 2.5, 3.5, 4.5]
    
    # Create a TensorBuffer with empty host memory
    arr = aligned_array((4,), np.float32)
    tb = TensorBuffer.create_from_host_memory(data=arr, dtype="float32", num_elements=4)
    
    # Write data to the buffer
    tb.write(test_data, "float32")
    
    # Read data back and verify
    result = tb.read(4, "float32")
    for got, expected in zip(result, test_data):
      self.assertAlmostEqual(got, expected, delta=1e-5)
    
    # Clean up
    tb.destroy()

  def test_write_read_int8(self):
    """Tests writing to and reading from a TensorBuffer with int8 data."""
    # Create data to write
    test_data = [10, 20, 30, 40]
    
    # Create a TensorBuffer with empty host memory
    arr = aligned_array((4,), np.int8)
    tb = TensorBuffer.create_from_host_memory(data=arr, dtype="int8", num_elements=4)
    
    # Write data to the buffer
    tb.write(test_data, "int8")
    
    # Read data back and verify
    result = tb.read(4, "int8")
    self.assertEqual(result, test_data)
    
    # Clean up
    tb.destroy()

  def test_destroy(self):
    """Tests that destroying a TensorBuffer works correctly."""
    # Create a TensorBuffer
    arr = aligned_array((4,), np.float32)
    tb = TensorBuffer.create_from_host_memory(data=arr, dtype="float32", num_elements=4)
    
    # Destroy the buffer
    tb.destroy()
    
    # Verify that capsule is None after destroy
    self.assertIsNone(tb._capsule)

  def test_destroy_twice(self):
    """Tests that destroying a TensorBuffer twice doesn't crash."""
    # Create a TensorBuffer
    arr = aligned_array((4,), np.float32)
    tb = TensorBuffer.create_from_host_memory(data=arr, dtype="float32", num_elements=4)
    
    # Destroy the buffer twice
    tb.destroy()
    tb.destroy()  # This should not raise any exception
    
    # Verify that capsule is still None
    self.assertIsNone(tb._capsule)

  def test_various_alignments(self):
    """Tests TensorBuffer with different memory alignments."""
    for alignment in [32, 64, 128]:
      # Create aligned numpy array
      test_data = [1.0, 2.0, 3.0, 4.0]
      arr = aligned_array((4,), np.float32, alignment=alignment)
      arr[:] = test_data
      
      # Verify the array is correctly aligned
      self.assertEqual(arr.ctypes.data % alignment, 0)
      
      # Create TensorBuffer referencing the array
      tb = TensorBuffer.create_from_host_memory(data=arr, dtype="float32", num_elements=4)
      
      # Verify data can be read back correctly
      result = tb.read(4, "float32")
      self.assertEqual(result, test_data)
      
      # Clean up
      tb.destroy()

  def test_various_dtypes(self):
    """Tests TensorBuffer with different data types."""
    dtypes = [
        ("float32", np.float32, [1.0, 2.0, 3.0, 4.0]),
        ("float64", np.float64, [1.0, 2.0, 3.0, 4.0]),
        ("int8", np.int8, [1, 2, 3, 4]),
        ("int16", np.int16, [1, 2, 3, 4]),
        ("int32", np.int32, [1, 2, 3, 4]),
        ("int64", np.int64, [1, 2, 3, 4]),
        ("uint8", np.uint8, [1, 2, 3, 4]),
        ("uint16", np.uint16, [1, 2, 3, 4]),
        ("uint32", np.uint32, [1, 2, 3, 4]),
        ("uint64", np.uint64, [1, 2, 3, 4]),
    ]
    
    for dtype_str, np_dtype, test_data in dtypes:
      # Create aligned numpy array
      arr = aligned_array((4,), np_dtype)
      arr[:] = test_data
      
      # Create TensorBuffer referencing the array
      tb = TensorBuffer.create_from_host_memory(data=arr, dtype=dtype_str, num_elements=4)
      
      # Verify data can be read back correctly
      result = tb.read(4, dtype_str)
      for got, expected in zip(result, test_data):
        if np_dtype in (np.float32, np.float64):
          self.assertAlmostEqual(got, expected, delta=1e-5)
        else:
          self.assertEqual(got, expected)
      
      # Clean up
      tb.destroy()

  def test_capsule_property(self):
    """Tests the capsule property."""
    # Create a TensorBuffer
    arr = aligned_array((4,), np.float32)
    tb = TensorBuffer.create_from_host_memory(data=arr, dtype="float32", num_elements=4)
    
    # Verify that the capsule property returns something
    self.assertIsNotNone(tb.capsule)
    
    # Clean up
    tb.destroy()


if __name__ == "__main__":
  unittest.main()