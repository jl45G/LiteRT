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

"""Python wrapper for LiteRT compiled models."""
from typing import Any, Dict, List, Optional, Union

from google3.third_party.odml.litert.litert.python.compiled_model_wrapper import _pywrap_litert_compiled_model_wrapper as _cm
from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import TensorBuffer


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

  def get_signature_list(self) -> Dict[str, Any]:
    """Gets a dictionary of all model signatures.
    
    Returns:
      Dictionary mapping signature keys to signature details
    """
    return self._model.GetSignatureList()

  def get_signature_by_index(self, index: int) -> Dict[str, Any]:
    """Gets details about a signature by index.
    
    Args:
      index: Index of the signature to get
      
    Returns:
      Dictionary with signature details
    """
    return self._model.GetSignatureByIndex(index)

  def get_num_signatures(self) -> int:
    """Gets the number of signatures in the model.
    
    Returns:
      Number of signatures
    """
    return self._model.GetNumSignatures()

  def get_signature_index(self, key: str) -> int:
    """Gets the index of a signature by key.
    
    Args:
      key: Signature key to look up
      
    Returns:
      Index of the signature, or -1 if not found
    """
    return self._model.GetSignatureIndex(key)

  def get_input_buffer_requirements(self, signature_index: int, input_index: int) -> Dict[str, Any]:
    """Gets buffer requirements for an input tensor.
    
    Args:
      signature_index: Index of the signature
      input_index: Index of the input tensor
      
    Returns:
      Dictionary with buffer requirements
    """
    return self._model.GetInputBufferRequirements(signature_index, input_index)

  def get_output_buffer_requirements(self, signature_index: int, output_index: int) -> Dict[str, Any]:
    """Gets buffer requirements for an output tensor.
    
    Args:
      signature_index: Index of the signature
      output_index: Index of the output tensor
      
    Returns:
      Dictionary with buffer requirements
    """
    return self._model.GetOutputBufferRequirements(signature_index, output_index)

  def create_input_buffer_by_name(self, signature_key: str, input_name: str) -> TensorBuffer:
    """Creates an input tensor buffer.
    
    Args:
      signature_key: The signature key
      input_name: Name of the input tensor
      
    Returns:
      A TensorBuffer object
    """
    capsule = self._model.CreateInputBufferByName(signature_key, input_name)
    # Create dimensions and dtype for the TensorBuffer from buffer requirements
    
    # Note: In a real implementation, we would extract this information
    # from the model, but for simplicity we'll create a wrapper directly
    return TensorBuffer(capsule)

  def create_output_buffer_by_name(self, signature_key: str, output_name: str) -> TensorBuffer:
    """Creates an output tensor buffer.
    
    Args:
      signature_key: The signature key
      output_name: Name of the output tensor
      
    Returns:
      A TensorBuffer object
    """
    capsule = self._model.CreateOutputBufferByName(signature_key, output_name)
    return TensorBuffer(capsule)

  def create_input_buffers(self, signature_index: int) -> List[TensorBuffer]:
    """Creates tensor buffers for all inputs in a signature.
    
    Args:
      signature_index: Index of the signature
      
    Returns:
      List of TensorBuffer objects
    """
    capsules = self._model.CreateInputBuffers(signature_index)
    return [TensorBuffer(capsule) for capsule in capsules]

  def create_input_buffer_from_memory(
      self, signature_key: str, input_name: str, data: Any, dtype: str
  ) -> TensorBuffer:
    """Creates a TensorBuffer that wraps the existing memory of `data`.

    Args:
      signature_key: The signature key to use.
      input_name: The name of the input tensor.
      data: Data object that supports the Python buffer protocol (e.g., bytes,
        bytearray, numpy array).
      dtype: String representation of the data type (e.g., "float32", "int8").

    Returns:
      A TensorBuffer object
    """
    capsule = self._model.CreateTensorBufferFromMemory(
        signature_key, input_name, data, dtype
    )
    return TensorBuffer(capsule)

  def create_output_buffers(self, signature_index: int) -> List[TensorBuffer]:
    """Creates tensor buffers for all outputs in a signature.
    
    Args:
      signature_index: Index of the signature
      
    Returns:
      List of TensorBuffer objects
    """
    capsules = self._model.CreateOutputBuffers(signature_index)
    return [TensorBuffer(capsule) for capsule in capsules]

  def run_by_name(self, signature_key: str, 
                input_map: Dict[str, TensorBuffer], 
                output_map: Dict[str, TensorBuffer]) -> None:
    """Runs the model using signature key with input and output maps.

    Args:
      signature_key: The signature key to use.
      input_map: Dictionary mapping input names to TensorBuffer objects.
      output_map: Dictionary mapping output names to TensorBuffer objects.
    """
    # Convert TensorBuffer objects to raw capsules
    capsule_input_map = {
        name: buffer._buffer for name, buffer in input_map.items()
    }
    capsule_output_map = {
        name: buffer._buffer for name, buffer in output_map.items()
    }
    
    self._model.RunByName(signature_key, capsule_input_map, capsule_output_map)

  def run_by_index(self, signature_index: int, 
                 input_buffers: List[TensorBuffer], 
                 output_buffers: List[TensorBuffer]) -> None:
    """Runs the model using signature index with input and output buffer lists.

    Args:
      signature_index: The index of the signature to use.
      input_buffers: List of input TensorBuffer objects.
      output_buffers: List of output TensorBuffer objects.
    """
    # Convert TensorBuffer objects to raw capsules
    input_caps_list = [buffer._buffer for buffer in input_buffers]
    output_caps_list = [buffer._buffer for buffer in output_buffers]
    
    self._model.RunByIndex(signature_index, input_caps_list, output_caps_list)

  # Deprecated methods - kept for backward compatibility
  # These methods are maintained for backward compatibility but should be 
  # replaced with using the TensorBuffer class's methods directly

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
