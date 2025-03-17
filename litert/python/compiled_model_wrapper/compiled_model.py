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

"""Python wrapper for LiteRT compiled models."""
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
    """Creates a list of PyCapsules for the signature's inputs."""
    return self._model.CreateInputBuffers(signature_index)

  def create_input_buffer_from_memory(
      self, signature_key: str, input_name: str, data: Any, dtype: str
  ) -> object:
    """Creates a TensorBuffer that wraps the existing memory of `data`.

    Args:
      signature_key: The signature key to use.
      input_name: The name of the input tensor.
      data: Data object that supports the Python buffer protocol (e.g., bytes, 
        bytearray, numpy array).
      dtype: String representation of the data type (e.g., "float32", "int8").

    Returns:
      A PyCapsule that can be passed to 'run_by_name' or 'run_by_index'.
    """
    return self._model.CreateTensorBufferFromMemory(
        signature_key, input_name, data, dtype
    )

  def create_output_buffers(self, signature_index):
    """Creates a list of PyCapsules for the signature's outputs."""
    return self._model.CreateOutputBuffers(signature_index)

  def run_by_name(self, signature_key, input_map, output_map):
    """Runs the model using signature key with input and output maps.
    
    Args:
      signature_key: The signature key to use.
      input_map: Dictionary mapping input names to tensor buffer capsules.
      output_map: Dictionary mapping output names to tensor buffer capsules.
    """
    return self._model.RunByName(signature_key, input_map, output_map)

  def run_by_index(self, signature_index, input_caps_list, output_caps_list):
    """Runs the model using signature index with input and output capsule lists.
    
    Args:
      signature_index: The index of the signature to use.
      input_caps_list: List of input tensor buffer capsules.
      output_caps_list: List of output tensor buffer capsules.
    """
    return self._model.RunByIndex(
        signature_index, input_caps_list, output_caps_list
    )

  def write_float_tensor(self, tensor_buffer_capsule, float_list):
    """Writes float data into the given TensorBuffer capsule.
    
    Args:
      tensor_buffer_capsule: The tensor buffer to write to.
      float_list: List of float values to write.
    """
    self._model.WriteFloatTensor(tensor_buffer_capsule, float_list)

  def read_float_tensor(self, tensor_buffer_capsule, num_floats):
    """Reads float data from the given TensorBuffer capsule.
    
    Args:
      tensor_buffer_capsule: The tensor buffer to read from.
      num_floats: Number of float values to read.
      
    Returns:
      A list of float values.
    """
    return self._model.ReadFloatTensor(tensor_buffer_capsule, num_floats)

  def destroy_tensor_buffer(self, tensor_buffer_capsule):
    """Explicitly destroys the buffer to free memory.
    
    Args:
      tensor_buffer_capsule: The tensor buffer to destroy.
    """
    self._model.DestroyTensorBuffer(tensor_buffer_capsule)

  def write_tensor(self, tensor_buffer_capsule, data, dtype: str):
    """Writes data into the given TensorBuffer capsule.
    
    Args:
      tensor_buffer_capsule: The tensor buffer to write to.
      data: List of numeric values to write.
      dtype: String representation of the data type (e.g., "float32", "int8").
    """
    self._model.WriteTensor(tensor_buffer_capsule, data, dtype)

  def read_tensor(self, tensor_buffer_capsule, num_elements: int, dtype: str):
    """Reads data from the given TensorBuffer capsule.
    
    Args:
      tensor_buffer_capsule: The tensor buffer to read from.
      num_elements: Number of elements to read.
      dtype: String representation of the data type (e.g., "float32", "int8").
      
    Returns:
      A list of the read values.
    """
    return self._model.ReadTensor(tensor_buffer_capsule, num_elements, dtype)
