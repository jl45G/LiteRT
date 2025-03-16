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
# ==============================================================================

"""Python lite rt compiled model."""
from typing import Any, Dict, cast

from google3.third_party.odml.litert.litert.python.compiled_model_wrapper import _pywrap_litert_compiled_model_wrapper as _cm


class CompiledModel:
  """Python-friendly wrapper around the C++ CompiledModelWrapper."""

  def __init__(self, c_model_ptr):
    # c_model_ptr is a pointer to a CompiledModelWrapper in C++.
    self._model = c_model_ptr

  @classmethod
  def from_file(
      cls, model_path, compiler_plugin="", dispatch_library="", hardware_accel=0
  ):
    ptr = _cm.CreateCompiledModelFromFile(
        model_path, compiler_plugin, dispatch_library, hardware_accel
    )
    return cls(ptr)

  @classmethod
  def from_buffer(
      cls, model_data, compiler_plugin="", dispatch_library="", hardware_accel=0
  ):
    ptr = _cm.CreateCompiledModelFromBuffer(
        model_data, compiler_plugin, dispatch_library, hardware_accel
    )
    return cls(ptr)

  def get_signature_list(self):
    return self._model.GetSignatureList()

  def get_signature_by_index(self, index):
    return self._model.GetSignatureByIndex(index)

  def get_num_signatures(self):
    return self._model.GetNumSignatures()

  def get_signature_index(self, key):
    return self._model.GetSignatureIndex(key)

  def get_input_buffer_requirements(self, signature_index, input_index):
    return self._model.GetInputBufferRequirements(signature_index, input_index)

  def get_output_buffer_requirements(self, signature_index, output_index):
    return self._model.GetOutputBufferRequirements(
        signature_index, output_index
    )

  def create_input_buffer_by_name(self, signature_key, input_name):
    return self._model.CreateInputBufferByName(signature_key, input_name)

  def create_output_buffer_by_name(self, signature_key, output_name):
    return self._model.CreateOutputBufferByName(signature_key, output_name)

  def create_input_buffers(self, signature_index):
    """Returns a list of PyCapsules for that signature's inputs."""
    return self._model.CreateInputBuffers(signature_index)

  def create_input_buffer_from_memory(
      self, signature_key: str, input_name: str, data: Any, dtype: str
  ) -> object:
    """Create a TensorBuffer that wraps the *existing* memory of `data`.

    - data must support the Python buffer protocol (e.g. bytes, bytearray, numpy
    array). - dtype is a string like "float32", "int8", etc. We need it to
    interpret the memory size per element.

    Returns a PyCapsule that can be passed to 'run_by_name' or 'run_by_index'.
    """
    return self._model.CreateTensorBufferFromMemory(
        signature_key, input_name, data, dtype
    )

  def create_output_buffers(self, signature_index):
    """Returns a list of PyCapsules for that signature's outputs."""
    return self._model.CreateOutputBuffers(signature_index)

  def run_by_name(self, signature_key, input_map, output_map):
    """Run the model using signature key, passing input_map & output_map dicts of {str: capsule}."""
    return self._model.RunByName(signature_key, input_map, output_map)

  def run_by_index(self, signature_index, input_caps_list, output_caps_list):
    """Run the model using signature index, passing lists of input, output capsules."""
    return self._model.RunByIndex(
        signature_index, input_caps_list, output_caps_list
    )

  # New methods for data I/O:

  def write_float_tensor(self, tensor_buffer_capsule, float_list):
    """Write `float_list` into the given TensorBuffer capsule."""
    self._model.WriteFloatTensor(tensor_buffer_capsule, float_list)

  def read_float_tensor(self, tensor_buffer_capsule, num_floats):
    """Return a Python list of float read from the buffer."""
    return self._model.ReadFloatTensor(tensor_buffer_capsule, num_floats)

  def destroy_tensor_buffer(self, tensor_buffer_capsule):
    """Explicitly destroy the buffer, if you need to free memory early."""
    self._model.DestroyTensorBuffer(tensor_buffer_capsule)

  def write_tensor(self, tensor_buffer_capsule, data, dtype: str):
    """Write `data` (a Python list of numbers) into the given TensorBuffer capsule,

    interpreted as `dtype` (e.g. "float32", "int8", "int32").
    """
    self._model.WriteTensor(tensor_buffer_capsule, data, dtype)

  def read_tensor(self, tensor_buffer_capsule, num_elements: int, dtype: str):
    """Read `num_elements` items from the given TensorBuffer capsule,

    interpreting them as `dtype` (e.g. "float32", "int8", "int32").
    Returns a Python list of the read values.
    """
    return self._model.ReadTensor(tensor_buffer_capsule, num_elements, dtype)
