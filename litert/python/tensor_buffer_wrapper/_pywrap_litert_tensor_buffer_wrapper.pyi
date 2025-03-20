# _pywrap_litert_tensor_buffer_wrapper.pyi

"""
Python type stubs for the LiteRT TensorBuffer wrapper.
"""

from typing import Any, List

def CreateTensorBufferFromHostMemory(
        py_data: Any,
        dtype: str,
        num_elements: int
) -> object:
    """
    Creates a TensorBuffer from existing host memory.
    Returns a PyCapsule object that represents the buffer.
    """
    ...

def WriteTensor(
        capsule: object,
        data_list: list,
        dtype: str
) -> None:
    """
    Writes data into the tensor buffer.
    """
    ...

def ReadTensor(
        capsule: object,
        num_elements: int,
        dtype: str
) -> list:
    """
    Reads data from the tensor buffer and returns a Python list.
    """
    ...

def DestroyTensorBuffer(
        capsule: object
) -> None:
    """
    Explicitly destroys the tensor buffer capsule and frees its resources.
    """
    ...
