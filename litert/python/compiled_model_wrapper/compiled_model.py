# Located at: google3/litert/python/compiled_model_wrapper/compiled_model.py

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

from typing import Any, Dict, List

# C++ binding for compiled models
from google3.third_party.odml.litert.litert.python.compiled_model_wrapper import (
    _pywrap_litert_compiled_model_wrapper as _cm,
)
# The Python-level TensorBuffer class that wraps PyCapsules
from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper.tensor_buffer import TensorBuffer


class CompiledModel:
  """Python-friendly wrapper around the C++ CompiledModelWrapper."""

  def __init__(self, c_model_ptr):
    self._model = c_model_ptr  # pointer to C++ CompiledModelWrapper

  @classmethod
  def from_file(
      cls,
      model_path: str,
      compiler_plugin: str = "",
      dispatch_library: str = "",
      hardware_accel: int = 0,
  ) -> "CompiledModel":
    ptr = _cm.CreateCompiledModelFromFile(
        model_path, compiler_plugin, dispatch_library, hardware_accel
    )
    return cls(ptr)

  @classmethod
  def from_buffer(
      cls,
      model_data: bytes,
      compiler_plugin: str = "",
      dispatch_library: str = "",
      hardware_accel: int = 0,
  ) -> "CompiledModel":
    ptr = _cm.CreateCompiledModelFromBuffer(
        model_data, compiler_plugin, dispatch_library, hardware_accel
    )
    return cls(ptr)

  def get_signature_list(self) -> Dict[str, Dict[str, List[str]]]:
    return self._model.GetSignatureList()

  def get_signature_by_index(self, index: int) -> Dict[str, Any]:
    return self._model.GetSignatureByIndex(index)

  def get_num_signatures(self) -> int:
    return self._model.GetNumSignatures()

  def get_signature_index(self, key: str) -> int:
    return self._model.GetSignatureIndex(key)

  def get_input_buffer_requirements(
      self, signature_index: int, input_index: int
  ) -> Dict[str, Any]:
    return self._model.GetInputBufferRequirements(signature_index, input_index)

  def get_output_buffer_requirements(
      self, signature_index: int, output_index: int
  ) -> Dict[str, Any]:
    return self._model.GetOutputBufferRequirements(
        signature_index, output_index
    )

  def create_input_buffer_by_name(
      self, signature_key: str, input_name: str
  ) -> TensorBuffer:
    """Creates an input TensorBuffer for the specified signature & input name."""
    capsule = self._model.CreateInputBufferByName(signature_key, input_name)
    return TensorBuffer(capsule)

  def create_output_buffer_by_name(
      self, signature_key: str, output_name: str
  ) -> TensorBuffer:
    """Creates an output TensorBuffer for the specified signature & output name."""
    capsule = self._model.CreateOutputBufferByName(signature_key, output_name)
    return TensorBuffer(capsule)

  def create_input_buffers(self, signature_index: int) -> List[TensorBuffer]:
    """Creates a list of TensorBuffers for the signature's inputs."""
    capsule_list = self._model.CreateInputBuffers(signature_index)
    return [TensorBuffer(c) for c in capsule_list]

  def create_output_buffers(self, signature_index: int) -> List[TensorBuffer]:
    """Creates a list of TensorBuffers for the signature's outputs."""
    capsule_list = self._model.CreateOutputBuffers(signature_index)
    return [TensorBuffer(c) for c in capsule_list]

  def run_by_name(
      self,
      signature_key: str,
      input_map: Dict[str, TensorBuffer],
      output_map: Dict[str, TensorBuffer],
  ) -> None:
    """Runs the model by signature name with input/output TensorBuffers."""
    # Convert TensorBuffer objects to raw capsules
    capsule_input_map = {k: v.capsule for k, v in input_map.items()}
    capsule_output_map = {k: v.capsule for k, v in output_map.items()}
    self._model.RunByName(signature_key, capsule_input_map, capsule_output_map)

  def run_by_index(
      self,
      signature_index: int,
      input_buffers: List[TensorBuffer],
      output_buffers: List[TensorBuffer],
  ) -> None:
    """Runs the model by signature index with input/output TensorBuffers."""
    input_capsules = [tb.capsule for tb in input_buffers]
    output_capsules = [tb.capsule for tb in output_buffers]
    self._model.RunByIndex(signature_index, input_capsules, output_capsules)
