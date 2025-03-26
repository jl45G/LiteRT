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

"""Unit tests for compiled_model.py."""

import os
import unittest
from unittest import mock
import numpy as np

from google3.third_party.odml.litert.litert.python.compiled_model_wrapper.compiled_model import CompiledModel
from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import TensorBuffer
from google3.third_party.tensorflow.python.platform import resource_loader

# Paths to test model files
MODEL_FLOAT_FILE_NAME = "testdata/simple_model_float.mlir"
MODEL_INT_FILE_NAME = "testdata/simple_model_int.mlir"


def get_model_path(model_filename):
  """Returns the absolute path to a test model file.

  Args:
    model_filename: Name of the model file in the testdata directory.

  Returns:
    String containing the absolute path to the model file.
  """
  return resource_loader.get_path_to_datafile(model_filename)


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


class CompiledModelUnitTest(unittest.TestCase):
  """Unit tests for the CompiledModel class."""
  
  def setUp(self):
    """Set up test resources."""
    # Paths to test models
    self.float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    self.int_model_path = get_model_path(MODEL_INT_FILE_NAME)
    
    # Test data
    self.test_input0_float = [1.0, 2.0, 3.0, 4.0]
    self.test_input1_float = [10.0, 20.0, 30.0, 40.0]
    self.expected_output_float = [11.0, 22.0, 33.0, 44.0]
    
    self.test_input_int = [0, 10, 20, 30]
    self.expected_output_int = [1, 11, 21, 31]

  def test_constructor(self):
    """Tests the constructor with a mock C++ model pointer."""
    mock_cpp_model = mock.MagicMock()
    cm = CompiledModel(mock_cpp_model)
    self.assertEqual(cm._model, mock_cpp_model)

  def test_from_file(self):
    """Tests the from_file class method."""
    # Test with default parameters
    cm = CompiledModel.from_file(self.float_model_path)
    self.assertIsNotNone(cm)
    self.assertIsNotNone(cm._model)
    
    # Test with optional parameters
    cm = CompiledModel.from_file(
        self.float_model_path,
        compiler_plugin="test_plugin",
        dispatch_library="test_library",
        hardware_accel=1
    )
    self.assertIsNotNone(cm)
    self.assertIsNotNone(cm._model)

  def test_from_buffer(self):
    """Tests the from_buffer class method."""
    # Read model file into buffer
    with open(self.float_model_path, "rb") as f:
      model_data = f.read()
    
    # Test with default parameters
    cm = CompiledModel.from_buffer(model_data)
    self.assertIsNotNone(cm)
    self.assertIsNotNone(cm._model)
    
    # Test with optional parameters
    cm = CompiledModel.from_buffer(
        model_data,
        compiler_plugin="test_plugin",
        dispatch_library="test_library",
        hardware_accel=1
    )
    self.assertIsNotNone(cm)
    self.assertIsNotNone(cm._model)

  def test_signature_methods(self):
    """Tests signature-related methods."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Test get_num_signatures
    num_sigs = cm.get_num_signatures()
    self.assertGreaterEqual(num_sigs, 1)
    
    # Test get_signature_list
    sig_list = cm.get_signature_list()
    self.assertIsInstance(sig_list, dict)
    
    # Test get_signature_index
    sig_key = "<placeholder signature>"
    sig_idx = cm.get_signature_index(sig_key)
    self.assertNotEqual(sig_idx, -1)
    
    # Test get_signature_by_index
    sig_info = cm.get_signature_by_index(sig_idx)
    self.assertIsInstance(sig_info, dict)

  def test_buffer_requirements(self):
    """Tests buffer requirement methods."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Test get_input_buffer_requirements
    sig_idx = cm.get_signature_index("<placeholder signature>") 
    if sig_idx == -1:
      sig_idx = 0  # Fallback
    
    input_req = cm.get_input_buffer_requirements(sig_idx, 0)
    self.assertIsInstance(input_req, dict)
    self.assertIn("buffer_size", input_req)
    self.assertIn("alignment", input_req)
    
    # Test get_output_buffer_requirements
    output_req = cm.get_output_buffer_requirements(sig_idx, 0)
    self.assertIsInstance(output_req, dict)
    self.assertIn("buffer_size", output_req)
    self.assertIn("alignment", output_req)

  def test_create_input_buffers(self):
    """Tests creating input buffers."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Test create_input_buffers
    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx == -1:
      sig_idx = 0  # Fallback
    
    input_bufs = cm.create_input_buffers(sig_idx)
    self.assertEqual(len(input_bufs), 2)
    self.assertIsInstance(input_bufs[0], TensorBuffer)
    self.assertIsInstance(input_bufs[1], TensorBuffer)
    
    # Clean up
    for buf in input_bufs:
      buf.destroy()

  def test_create_output_buffers(self):
    """Tests creating output buffers."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Test create_output_buffers
    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx == -1:
      sig_idx = 0  # Fallback
    
    output_bufs = cm.create_output_buffers(sig_idx)
    self.assertEqual(len(output_bufs), 1)
    self.assertIsInstance(output_bufs[0], TensorBuffer)
    
    # Clean up
    for buf in output_bufs:
      buf.destroy()

  def test_create_buffers_by_name(self):
    """Tests creating buffers by name."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Test create_input_buffer_by_name
    in0_buf = cm.create_input_buffer_by_name("<placeholder signature>", "arg0")
    self.assertIsInstance(in0_buf, TensorBuffer)
    in0_buf.destroy()
    
    # Test create_output_buffer_by_name
    out_buf = cm.create_output_buffer_by_name("<placeholder signature>", "tfl.add")
    self.assertIsInstance(out_buf, TensorBuffer)
    out_buf.destroy()

  def test_run_by_index(self):
    """Tests running inference using run_by_index method."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Get signature index
    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx == -1:
      sig_idx = 0  # Fallback
    
    # Create input and output buffers
    input_bufs = cm.create_input_buffers(sig_idx)
    output_bufs = cm.create_output_buffers(sig_idx)
    
    # Write test data to input buffers
    input_bufs[0].write(self.test_input0_float, "float32")
    input_bufs[1].write(self.test_input1_float, "float32")
    
    # Run inference
    cm.run_by_index(sig_idx, input_bufs, output_bufs)
    
    # Verify results
    out_data = output_bufs[0].read(len(self.expected_output_float), "float32")
    for got, expected in zip(out_data, self.expected_output_float):
      self.assertAlmostEqual(got, expected, delta=1e-5)
    
    # Clean up
    for buf in input_bufs:
      buf.destroy()
    for buf in output_bufs:
      buf.destroy()

  def test_run_by_name(self):
    """Tests running inference using run_by_name method."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Create buffers by name
    in0_buf = cm.create_input_buffer_by_name("<placeholder signature>", "arg0")
    in1_buf = cm.create_input_buffer_by_name("<placeholder signature>", "arg1")
    out_buf = cm.create_output_buffer_by_name("<placeholder signature>", "tfl.add")
    
    # Write test data to input buffers
    in0_buf.write(self.test_input0_float, "float32")
    in1_buf.write(self.test_input1_float, "float32")
    
    # Prepare input and output maps
    input_map = {"arg0": in0_buf, "arg1": in1_buf}
    output_map = {"tfl.add": out_buf}
    
    # Run inference
    cm.run_by_name("<placeholder signature>", input_map, output_map)
    
    # Verify results
    out_data = out_buf.read(len(self.expected_output_float), "float32")
    for got, expected in zip(out_data, self.expected_output_float):
      self.assertAlmostEqual(got, expected, delta=1e-5)
    
    # Clean up
    in0_buf.destroy()
    in1_buf.destroy()
    out_buf.destroy()

  def test_alignment_requirements(self):
    """Tests that buffer alignment requirements are respected."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Get signature index
    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx == -1:
      sig_idx = 0  # Fallback
    
    # Get input buffer requirements
    input_req = cm.get_input_buffer_requirements(sig_idx, 0)
    alignment = input_req.get("alignment", 64)  # Default to 64 if not specified
    
    # Create aligned arrays with the required alignment
    arr0 = aligned_array((4,), np.float32, alignment=alignment)
    arr0[:] = self.test_input0_float
    arr1 = aligned_array((4,), np.float32, alignment=alignment)
    arr1[:] = self.test_input1_float
    
    # Create tensor buffers from aligned arrays
    tb0 = TensorBuffer.create_from_host_memory(arr0, "float32", 4)
    tb1 = TensorBuffer.create_from_host_memory(arr1, "float32", 4)
    
    # Create output buffer
    output_bufs = cm.create_output_buffers(sig_idx)
    
    # Run inference
    cm.run_by_index(sig_idx, [tb0, tb1], output_bufs)
    
    # Verify results
    out_data = output_bufs[0].read(len(self.expected_output_float), "float32")
    for got, expected in zip(out_data, self.expected_output_float):
      self.assertAlmostEqual(got, expected, delta=1e-5)
    
    # Clean up
    tb0.destroy()
    tb1.destroy()
    for buf in output_bufs:
      buf.destroy()

  def test_int_model(self):
    """Tests running inference on an integer model."""
    # Create model
    cm = CompiledModel.from_file(self.int_model_path)
    
    # Get signature index
    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx == -1:
      sig_idx = 0  # Fallback
    
    # Create input and output buffers
    input_bufs = cm.create_input_buffers(sig_idx)
    output_bufs = cm.create_output_buffers(sig_idx)
    
    # Write test data to input buffer
    input_bufs[0].write(self.test_input_int, "int32")
    
    # Run inference
    cm.run_by_index(sig_idx, input_bufs, output_bufs)
    
    # Verify results
    out_data = output_bufs[0].read(len(self.expected_output_int), "int32")
    self.assertEqual(out_data, self.expected_output_int)
    
    # Clean up
    for buf in input_bufs:
      buf.destroy()
    for buf in output_bufs:
      buf.destroy()

  def test_error_handling(self):
    """Tests error handling for invalid inputs."""
    # Create model
    cm = CompiledModel.from_file(self.float_model_path)
    
    # Test invalid signature index
    with self.assertRaises(Exception):
      cm.get_signature_by_index(-1)
    
    # Test invalid signature name
    self.assertEqual(cm.get_signature_index("nonexistent_signature"), -1)


if __name__ == "__main__":
  unittest.main()