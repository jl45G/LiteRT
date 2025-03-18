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

import logging
import unittest
import numpy as np
from google3.third_party.odml.litert.litert.python.compiled_model_wrapper import compiled_model
from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import (
    TensorBuffer, BUFFER_TYPE_HOST, ELEMENT_TYPE_FLOAT32, ELEMENT_TYPE_INT32
)
from google3.third_party.tensorflow.python.platform import resource_loader

# Paths to test model files
MODEL_FLOAT_FILE_NAME = "testdata/simple_model_float.tflite"
MODEL_INT8_FILE_NAME = "testdata/simple_model_int.tflite"

# Test data for float model
# The float model has 2 inputs (arg0, arg1) and 1 output (tfl.add)
# It performs element-wise addition of the inputs
TEST_INPUT0_FLOAT = [1.0, 2.0, 3.0, 4.0]
TEST_INPUT1_FLOAT = [10.0, 20.0, 30.0, 40.0]
EXPECTED_OUTPUT_FLOAT = [11.0, 22.0, 33.0, 44.0]

# Test data for int8 model
# The int8 model has 1 input (arg_int8) and 1 output (out_int8)
# It increments each input value by 1
TEST_INPUT_INT8 = [0, 10, 20, 30]
EXPECTED_OUTPUT_INT8 = [1, 11, 21, 31]


def get_model_path(model_filename):
  """Returns the absolute path to a test model file."""
  return resource_loader.get_path_to_datafile(model_filename)


def aligned_array(shape, dtype, alignment=64):
  """Allocates a NumPy array with the specified memory alignment.

  Args:
    shape: The shape of the array.
    dtype: The data type of the array.
    alignment: The memory alignment in bytes.

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


class CompiledModelBasicTest(unittest.TestCase):

  def test_basic(self):
    """Tests basic functionality of the CompiledModel."""
    cm = compiled_model.CompiledModel.from_file(
        resource_loader.get_path_to_datafile(MODEL_FLOAT_FILE_NAME)
    )
    num_signatures = cm.get_num_signatures()
    self.assertEqual(num_signatures, 1)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    print(sig_idx)
    if sig_idx < 0:
      sig_idx = 0

    input_buffers = cm.create_input_buffers(sig_idx)
    output_buffers = cm.create_output_buffers(sig_idx)

    req = cm.get_input_buffer_requirements(signature_index=0, input_index=0)
    print("Required buffer size:", req["buffer_size"])

    self.assertEqual(len(input_buffers), 2)
    self.assertEqual(len(output_buffers), 1)

    # Fill inputs using the new TensorBuffer API
    input_buffers[0].write(TEST_INPUT0_FLOAT)
    input_buffers[1].write(TEST_INPUT1_FLOAT)

    # Invoke
    cm.run_by_index(sig_idx, input_buffers, output_buffers)

    # Verify output using the new TensorBuffer API
    out_values = output_buffers[0].read(len(EXPECTED_OUTPUT_FLOAT))
    logging.info("Output = %s", out_values)
    for got, expected in zip(out_values, EXPECTED_OUTPUT_FLOAT):
      self.assertAlmostEqual(got, expected, delta=1e-5)

  def test_tensor_buffer_api(self):
    """Tests the new TensorBuffer API."""
    # Create a managed tensor buffer
    buffer = TensorBuffer.create_managed(
        dimensions=[2, 2],
        dtype="float32",
        buffer_type=BUFFER_TYPE_HOST
    )
    
    # Check properties
    self.assertEqual(buffer.get_dtype(), "float32")
    self.assertEqual(buffer.get_shape(), [2, 2])
    self.assertEqual(buffer.get_size(), 16)  # 2*2*4 bytes
    
    # Write and read data
    test_data = [1.0, 2.0, 3.0, 4.0]
    buffer.write(test_data)
    read_data = buffer.read(4)
    self.assertEqual(read_data, test_data)
    
    # Convert to numpy array
    np_array = buffer.numpy()
    self.assertEqual(np_array.shape, (2, 2))
    self.assertEqual(np_array.dtype, np.dtype("float32"))
    np.testing.assert_array_equal(np_array.flatten(), test_data)

  def test_tensor_buffer_with_model(self):
    """Tests using standalone TensorBuffer objects with a model."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)
    
    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx < 0:
      sig_idx = 0
    
    # Get the required buffer sizes
    in0_req = cm.get_input_buffer_requirements(sig_idx, 0)
    in1_req = cm.get_input_buffer_requirements(sig_idx, 1)
    out_req = cm.get_output_buffer_requirements(sig_idx, 0)
    
    # Create tensor buffers directly
    in0_buffer = TensorBuffer.create_managed(
        dimensions=[4], 
        dtype="float32",
        buffer_size=in0_req["buffer_size"]
    )
    
    in1_buffer = TensorBuffer.create_managed(
        dimensions=[4], 
        dtype="float32",
        buffer_size=in1_req["buffer_size"]
    )
    
    out_buffer = TensorBuffer.create_managed(
        dimensions=[4], 
        dtype="float32",
        buffer_size=out_req["buffer_size"]
    )
    
    # Write input data
    in0_buffer.write(TEST_INPUT0_FLOAT)
    in1_buffer.write(TEST_INPUT1_FLOAT)
    
    # Run model
    input_buffers = [in0_buffer, in1_buffer]
    output_buffers = [out_buffer]
    cm.run_by_index(sig_idx, input_buffers, output_buffers)
    
    # Verify results
    output_data = out_buffer.read(len(EXPECTED_OUTPUT_FLOAT))
    self.assertEqual(output_data, EXPECTED_OUTPUT_FLOAT)
    
    # Check conversion to numpy
    output_numpy = out_buffer.numpy()
    np.testing.assert_array_equal(output_numpy, EXPECTED_OUTPUT_FLOAT)

  def test_from_file_and_signatures(self):
    """Tests loading a model from file and accessing its signatures."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)

    # Check number of signatures
    num_sigs = cm.get_num_signatures()
    self.assertGreaterEqual(num_sigs, 1)

    # Get signature list
    sig_list = cm.get_signature_list()
    self.assertIn("<placeholder signature>", sig_list)
    serving_default_info = sig_list["<placeholder signature>"]
    self.assertIn("inputs", serving_default_info)
    self.assertIn("outputs", serving_default_info)

    # Check signature index
    sig_idx = cm.get_signature_index("<placeholder signature>")
    self.assertNotEqual(sig_idx, -1)

  def test_run_by_index_float(self):
    """Tests running inference on a float model using index-based API."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx < 0:
      sig_idx = 0  # Fall back to first signature if name doesn't match

    # Create input & output buffers
    input_buffers = cm.create_input_buffers(sig_idx)
    output_buffers = cm.create_output_buffers(sig_idx)

    self.assertEqual(len(input_buffers), 2)
    self.assertEqual(len(output_buffers), 1)

    # Write data to inputs using TensorBuffer API
    input_buffers[0].write(TEST_INPUT0_FLOAT)
    input_buffers[1].write(TEST_INPUT1_FLOAT)

    # Run inference
    cm.run_by_index(sig_idx, input_buffers, output_buffers)

    # Verify results using TensorBuffer API
    out_data = output_buffers[0].read(len(EXPECTED_OUTPUT_FLOAT))
    self.assertEqual(len(out_data), len(EXPECTED_OUTPUT_FLOAT))
    for got, expect in zip(out_data, EXPECTED_OUTPUT_FLOAT):
      self.assertAlmostEqual(got, expect, delta=1e-5)

  def test_run_by_name_float(self):
    """Tests running inference on a float model using name-based API."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)

    # Create buffers by name
    in0_buffer = cm.create_input_buffer_by_name("<placeholder signature>", "arg0")
    in1_buffer = cm.create_input_buffer_by_name("<placeholder signature>", "arg1")
    out_buffer = cm.create_output_buffer_by_name(
        "<placeholder signature>", "tfl.add"
    )

    # Fill input data using the new API
    in0_buffer.write(TEST_INPUT0_FLOAT)
    in1_buffer.write(TEST_INPUT1_FLOAT)

    # Run inference using name-based API
    input_map = {"arg0": in0_buffer, "arg1": in1_buffer}
    output_map = {"tfl.add": out_buffer}
    cm.run_by_name("<placeholder signature>", input_map, output_map)

    # Verify results using the new API
    out_data = out_buffer.read(len(EXPECTED_OUTPUT_FLOAT))
    self.assertEqual(out_data, EXPECTED_OUTPUT_FLOAT)

  def test_zero_copy_with_numpy(self):
    """Tests creating a tensor buffer from NumPy array memory."""
    # Create a numpy array
    arr = np.array(TEST_INPUT0_FLOAT, dtype=np.float32)
    
    # Create a TensorBuffer that wraps this memory
    buffer = TensorBuffer.create_from_host_memory(
        arr, dimensions=[4], dtype="float32"
    )
    
    # Verify the buffer contains the expected data
    data = buffer.read(len(TEST_INPUT0_FLOAT))
    self.assertEqual(data, TEST_INPUT0_FLOAT)
    
    # Modify the numpy array and verify the buffer sees the changes
    arr[0] = 100.0
    data = buffer.read(len(TEST_INPUT0_FLOAT))
    self.assertEqual(data[0], 100.0)
    
    # Use the buffer with a model
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)
    
    in1_buffer = cm.create_input_buffer_by_name("<placeholder signature>", "arg1")
    out_buffer = cm.create_output_buffer_by_name("<placeholder signature>", "tfl.add")
    
    in1_buffer.write(TEST_INPUT1_FLOAT)
    
    # Run the model
    input_map = {"arg0": buffer, "arg1": in1_buffer}
    output_map = {"tfl.add": out_buffer}
    cm.run_by_name("<placeholder signature>", input_map, output_map)
    
    # Verify results - first element should now be 100 + 10 = 110
    expected = [110.0] + [a + b for a, b in zip(TEST_INPUT0_FLOAT[1:], TEST_INPUT1_FLOAT[1:])]
    out_data = out_buffer.read(len(expected))
    self.assertEqual(out_data[0], 110.0)
    for i in range(1, len(out_data)):
      self.assertAlmostEqual(out_data[i], expected[i], delta=1e-5)

  def test_int8_model_with_tensor_buffer(self):
    """Tests inference on an int8 quantized model with TensorBuffer."""
    int8_model_path = get_model_path(MODEL_INT8_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(int8_model_path)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    self.assertNotEqual(
        sig_idx, -1, "Model must have '<placeholder signature>' signature."
    )

    # Create buffers
    input_buffers = cm.create_input_buffers(sig_idx)
    output_buffers = cm.create_output_buffers(sig_idx)
    self.assertEqual(len(input_buffers), 1)
    self.assertEqual(len(output_buffers), 1)

    # Write input data
    input_buffers[0].write(TEST_INPUT_INT8, dtype="int32")

    # Run inference
    cm.run_by_index(sig_idx, input_buffers, output_buffers)

    # Verify results
    out_data = output_buffers[0].read(len(TEST_INPUT_INT8), dtype="int32")
    self.assertEqual(out_data, EXPECTED_OUTPUT_INT8)

    # Test numpy integration
    input_numpy = np.array(TEST_INPUT_INT8, dtype=np.int32)
    numpy_buffer = TensorBuffer.create_from_host_memory(
        input_numpy, dimensions=[4], dtype="int32"
    )
    
    # Create fresh output buffer
    out_buffer = cm.create_output_buffer_by_name(
        "<placeholder signature>", "out_int8"
    )
    
    # Run with numpy-backed buffer
    input_map = {"arg_int8": numpy_buffer}
    output_map = {"out_int8": out_buffer}
    cm.run_by_name("<placeholder signature>", input_map, output_map)
    
    # Verify with numpy array
    output_numpy = out_buffer.numpy()
    np.testing.assert_array_equal(output_numpy, EXPECTED_OUTPUT_INT8)


if __name__ == "__main__":
  unittest.main()
