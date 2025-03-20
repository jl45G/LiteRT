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

import unittest

import numpy as np

from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import TensorBuffer


class TensorBufferTest(unittest.TestCase):

  def test_create_from_numpy_float32(self):
    """Tests creating a TensorBuffer from numpy float32 array and verifies read/write operations."""
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf = TensorBuffer.create_from_host_memory(data, "float32", len(data))
    
    # Verify initial data can be read correctly
    result = buf.read(num_elements=4, dtype="float32")
    self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])

    # Verify data can be written and read back correctly
    buf.write([5.0, 6.0, 7.0, 8.0], dtype="float32")
    new_vals = buf.read(num_elements=4, dtype="float32")
    self.assertEqual(new_vals, [5.0, 6.0, 7.0, 8.0])

    buf.destroy()

  def test_create_from_numpy_int8(self):
    """Tests creating a TensorBuffer from numpy int8 array and verifies read/write operations."""
    data = np.array([-128, 0, 10, 127], dtype=np.int8)
    buf = TensorBuffer.create_from_host_memory(data, "int8", len(data))
    
    # Verify initial data can be read correctly
    read_vals = buf.read(num_elements=4, dtype="int8")
    self.assertEqual(read_vals, [-128, 0, 10, 127])

    # Verify data can be written and read back correctly
    buf.write([-1, 12, 23, 100], dtype="int8")
    new_vals = buf.read(num_elements=4, dtype="int8")
    self.assertEqual(new_vals, [-1, 12, 23, 100])

    buf.destroy()

  def test_create_from_bytes_int32(self):
    """Tests creating a TensorBuffer from raw bytes containing int32 data."""
    # Create a bytes object with int32 data
    arr = np.array([100, 200, 300, 400], dtype=np.int32)
    raw_data = arr.tobytes()
    
    buf = TensorBuffer.create_from_host_memory(raw_data, "int32", 4)

    # Verify bytes data was correctly interpreted as int32
    read_vals = buf.read(num_elements=4, dtype="int32")
    self.assertEqual(read_vals, [100, 200, 300, 400])

    # Verify data can be modified and read back correctly
    buf.write([111, 222, 333, 444], dtype="int32")
    updated = buf.read(num_elements=4, dtype="int32")
    self.assertEqual(updated, [111, 222, 333, 444])

    buf.destroy()


if __name__ == "__main__":
  unittest.main()
