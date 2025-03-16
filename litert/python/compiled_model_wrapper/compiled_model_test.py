# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from google3.third_party.tensorflow.python.platform import resource_loader

# --------------------------------------------------------------------------------
# Constants for demonstration
# --------------------------------------------------------------------------------

# We'll assume we have two test models:
# 1) A float model, e.g. "simple_float_model.tflite" that:
#    - signature name "<placeholder signature>"
#    - has 2 float32 inputs "arg0" and "arg1"
#    - produces 1 float32 output "tfl.add"
#    - shape of each input is [4] (for example)
#
# 2) An int8 model, e.g. "simple_int8_model.tflite" that:
#    - signature name "<placeholder signature>"
#    - has 1 int8 input "arg_int8"
#    - produces 1 int8 output "out_int8"
#    - shape of input is [4]
#
# Adjust these names to match your real model(s).

MODEL_FLOAT_FILE_NAME = "testdata/simple_model_float.tflite"
MODEL_INT8_FILE_NAME = "testdata/simple_model_int.tflite"

# Test vectors for the float model example
TEST_INPUT0_FLOAT = [1.0, 2.0, 3.0, 4.0]
TEST_INPUT1_FLOAT = [10.0, 20.0, 30.0, 40.0]
EXPECTED_OUTPUT_FLOAT = [11.0, 22.0, 33.0, 44.0]

# Test vectors for the int8 model example
TEST_INPUT_INT8 = [0, 10, 20, 30]
EXPECTED_OUTPUT_INT8 = [
    1,
    11,
    21,
    31,
]  # e.g. a +1 operation. Just a hypothetical


def get_model_path(model_filename):
  return resource_loader.get_path_to_datafile(model_filename)


def aligned_array(shape, dtype, alignment=64):
  """Allocate a NumPy array with the specified alignment in bytes."""
  # Number of elements requested
  size = np.prod(shape)
  itemsize = np.dtype(dtype).itemsize

  # We'll over-allocate by 'alignment - 1' items (worst case) so that
  # we can find at least one place where the array is aligned.
  n_extra = alignment // itemsize

  # Create a 1D array of the needed size + the extra:
  raw = np.zeros(size + n_extra, dtype=dtype)

  # Find the first index within raw where the data pointer is aligned:
  start_index = 0
  while (raw[start_index:].ctypes.data % alignment) != 0:
    start_index += 1

  # Slice out the aligned portion
  aligned = raw[start_index : start_index + size].reshape(shape)
  return aligned


class CompiledModelBasicTest(unittest.TestCase):

  def test_basic(self):
    cm = compiled_model.CompiledModel.from_file(
        resource_loader.get_path_to_datafile(MODEL_FLOAT_FILE_NAME)
    )
    num_signatures = cm.get_num_signatures()
    self.assertEqual(num_signatures, 1)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    print(sig_idx)
    if sig_idx < 0:
      sig_idx = 0

    input_caps_list = cm.create_input_buffers(sig_idx)
    output_caps_list = cm.create_output_buffers(sig_idx)

    req = cm.get_input_buffer_requirements(signature_index=0, input_index=0)
    print("Required buffer size:", req["buffer_size"])

    self.assertEqual(len(input_caps_list), 2)
    self.assertEqual(len(output_caps_list), 1)

    # Fill inputs
    cm.write_float_tensor(input_caps_list[0], TEST_INPUT0_FLOAT)
    cm.write_float_tensor(input_caps_list[1], TEST_INPUT1_FLOAT)

    # Invoke
    cm.run_by_index(sig_idx, input_caps_list, output_caps_list)

    # Verify output
    out_values = cm.read_float_tensor(
        output_caps_list[0], len(EXPECTED_OUTPUT_FLOAT)
    )
    logging.info("Output = %s", out_values)
    for got, expected in zip(out_values, EXPECTED_OUTPUT_FLOAT):
      self.assertAlmostEqual(got, expected, delta=1e-5)

  def test_from_file_and_signatures(self):
    """Load a float model from file and check signatures, inputs, outputs."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)

    # Check number of signatures
    num_sigs = cm.get_num_signatures()
    self.assertGreaterEqual(num_sigs, 1)

    # Get signature list
    sig_list = cm.get_signature_list()
    self.assertIn("<placeholder signature>", sig_list)  # expecting that key
    serving_default_info = sig_list["<placeholder signature>"]
    self.assertIn("inputs", serving_default_info)
    self.assertIn("outputs", serving_default_info)

    # Check signature index
    sig_idx = cm.get_signature_index("<placeholder signature>")
    self.assertNotEqual(sig_idx, -1)

  def test_run_by_index_float(self):
    """Create buffers for a float model, write data, run by index, read result."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    if sig_idx < 0:
      sig_idx = 0  # fallback if the signature name is different

    # Create input & output buffers (copy-based approach)
    input_caps_list = cm.create_input_buffers(sig_idx)
    output_caps_list = cm.create_output_buffers(sig_idx)

    # We expect 2 inputs, 1 output (from the hypothetical model)
    self.assertEqual(len(input_caps_list), 2)
    self.assertEqual(len(output_caps_list), 1)

    # Write data to each input
    cm.write_tensor(input_caps_list[0], TEST_INPUT0_FLOAT, "float32")
    cm.write_tensor(input_caps_list[1], TEST_INPUT1_FLOAT, "float32")

    # Run by index
    cm.run_by_index(sig_idx, input_caps_list, output_caps_list)

    # Read result
    out_data = cm.read_tensor(
        output_caps_list[0], len(EXPECTED_OUTPUT_FLOAT), "float32"
    )
    self.assertEqual(len(out_data), len(EXPECTED_OUTPUT_FLOAT))
    for got, expect in zip(out_data, EXPECTED_OUTPUT_FLOAT):
      self.assertAlmostEqual(got, expect, delta=1e-5)

    # Destroy buffers
    for cap in input_caps_list:
      cm.destroy_tensor_buffer(cap)
    for cap in output_caps_list:
      cm.destroy_tensor_buffer(cap)

  def test_run_by_name_float(self):
    """Run inference by name (dictionary of input->capsule, output->capsule)."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)

    # We'll create buffers individually by name
    in0_caps = cm.create_input_buffer_by_name("<placeholder signature>", "arg0")
    in1_caps = cm.create_input_buffer_by_name("<placeholder signature>", "arg1")
    out_caps = cm.create_output_buffer_by_name(
        "<placeholder signature>", "tfl.add"
    )

    # Fill data
    cm.write_tensor(in0_caps, TEST_INPUT0_FLOAT, "float32")
    cm.write_tensor(in1_caps, TEST_INPUT1_FLOAT, "float32")

    # Run by name
    input_map = {"arg0": in0_caps, "arg1": in1_caps}
    output_map = {"tfl.add": out_caps}
    cm.run_by_name("<placeholder signature>", input_map, output_map)

    # Check results
    out_data = cm.read_tensor(out_caps, len(EXPECTED_OUTPUT_FLOAT), "float32")
    self.assertEqual(out_data, EXPECTED_OUTPUT_FLOAT)

    # Destroy
    cm.destroy_tensor_buffer(in0_caps)
    cm.destroy_tensor_buffer(in1_caps)
    cm.destroy_tensor_buffer(out_caps)

  def test_from_buffer(self):
    """Load the same float model from an in-memory buffer."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    with open(float_model_path, "rb") as f:
      model_data = f.read()

    cm = compiled_model.CompiledModel.from_buffer(model_data)

    self.assertGreaterEqual(cm.get_num_signatures(), 1)

  def test_int8_model_inference(self):
    """If you have a second model that uses int8, test it here."""
    int8_model_path = get_model_path(MODEL_INT8_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(int8_model_path)

    sig_idx = cm.get_signature_index("<placeholder signature>")
    self.assertNotEqual(
        sig_idx, -1, "Model must have 'serving_int8' signature."
    )

    # create buffers
    input_caps_list = cm.create_input_buffers(sig_idx)
    output_caps_list = cm.create_output_buffers(sig_idx)
    self.assertEqual(len(input_caps_list), 1)
    self.assertEqual(len(output_caps_list), 1)

    # write int8
    cm.write_tensor(input_caps_list[0], TEST_INPUT_INT8, "int32")

    # run
    cm.run_by_index(sig_idx, input_caps_list, output_caps_list)

    # read
    out_data = cm.read_tensor(
        output_caps_list[0], len(TEST_INPUT_INT8), "int32"
    )
    self.assertEqual(out_data, EXPECTED_OUTPUT_INT8)

    # clean
    cm.destroy_tensor_buffer(input_caps_list[0])
    cm.destroy_tensor_buffer(output_caps_list[0])

  def test_zero_copy_input(self):
    """Demonstrate creating an input buffer from existing memory."""
    print(">>>create output buffer by name")
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)
    print(">>>create output buffer by name")

    # Suppose the model input shape is [4], so we need 4 floats (16 bytes).
    # We'll create a numpy array with data in it.
    arr = aligned_array((4,), np.float32)
    arr[:] = TEST_INPUT0_FLOAT

    # create zero-copy capsule:
    print(">create output buffer by name")
    zero_copy_caps = cm.create_input_buffer_from_memory(
        "<placeholder signature>", "arg0", arr, "float32"
    )
    #
    # For the second input, let's do normal creation
    input1_caps = cm.create_input_buffer_by_name(
        "<placeholder signature>", "arg1"
    )
    cm.write_tensor(input1_caps, TEST_INPUT1_FLOAT, "float32")

    # Create output
    print("create output buffer by name")
    out_caps = cm.create_output_buffer_by_name(
        "<placeholder signature>", "tfl.add"
    )

    # run
    input_map = {"arg0": zero_copy_caps, "arg1": input1_caps}
    output_map = {"tfl.add": out_caps}
    print("run by name")
    cm.run_by_name("<placeholder signature>", input_map, output_map)

    # read
    out_data = cm.read_tensor(out_caps, len(EXPECTED_OUTPUT_FLOAT), "float32")
    self.assertEqual(out_data, EXPECTED_OUTPUT_FLOAT)

    # cleanup
    cm.destroy_tensor_buffer(zero_copy_caps)
    cm.destroy_tensor_buffer(input1_caps)
    cm.destroy_tensor_buffer(out_caps)

  def test_destroy_buffer_twice(self):
    """Check that calling destroy_tensor_buffer() multiple times is safe."""
    float_model_path = get_model_path(MODEL_FLOAT_FILE_NAME)
    cm = compiled_model.CompiledModel.from_file(float_model_path)

    in0_caps = cm.create_input_buffer_by_name("<placeholder signature>", "arg0")
    cm.write_tensor(in0_caps, TEST_INPUT0_FLOAT, "float32")

    # destroy once
    cm.destroy_tensor_buffer(in0_caps)

    # destroying again should not crash or cause an error
    # depending on your implementation, it might no-op or raise a mild warning
    try:
      cm.destroy_tensor_buffer(in0_caps)
    except Exception as e:
      self.fail(f"Second destroy call raised an exception: {e}")


if __name__ == "__main__":
  unittest.main()
