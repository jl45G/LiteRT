# tensor_buffer_test.py

import unittest

import numpy as np

# Suppose this is your new Python wrapper library for TensorBuffers:
from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import TensorBuffer


class TensorBufferTest(unittest.TestCase):

  def test_create_from_numpy_float32(self):
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf = TensorBuffer.create_from_host_memory(data, "float32", len(data))
    # Attempt reading from the buffer
    result = buf.read(num_elements=4, dtype="float32")
    self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])

    # Write new values to the buffer
    buf.write([5.0, 6.0, 7.0, 8.0], dtype="float32")
    new_vals = buf.read(num_elements=4, dtype="float32")
    self.assertEqual(new_vals, [5.0, 6.0, 7.0, 8.0])

    # Clean up
    buf.destroy()

  def test_create_from_numpy_int8(self):
    data = np.array([-128, 0, 10, 127], dtype=np.int8)
    buf = TensorBuffer.create_from_host_memory(data, "int8", len(data))
    read_vals = buf.read(num_elements=4, dtype="int8")
    self.assertEqual(read_vals, [-128, 0, 10, 127])

    # Overwrite
    buf.write([-1, 12, 23, 100], dtype="int8")
    new_vals = buf.read(num_elements=4, dtype="int8")
    self.assertEqual(new_vals, [-1, 12, 23, 100])

    buf.destroy()

  def test_create_from_bytes_int32(self):
    # Create a bytes object with some int32 data (4 elements).
    # E.g. [100, 200, 300, 400]
    arr = np.array([100, 200, 300, 400], dtype=np.int32)
    raw_data = arr.tobytes()
    # We need 4 elements
    buf = TensorBuffer.create_from_host_memory(raw_data, "int32", 4)

    # Read them back
    read_vals = buf.read(num_elements=4, dtype="int32")
    self.assertEqual(read_vals, [100, 200, 300, 400])

    # Overwrite
    buf.write([111, 222, 333, 444], dtype="int32")
    updated = buf.read(num_elements=4, dtype="int32")
    self.assertEqual(updated, [111, 222, 333, 444])

    buf.destroy()


if __name__ == "__main__":
  unittest.main()
