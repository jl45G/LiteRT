"""Type stub for LiteRT TensorBuffer wrapper module."""

from typing import Any, Dict, List, Tuple, Union

class TensorBufferWrapper:
    """Wrapper for a LiteRT TensorBuffer."""

    def GetBufferType(self) -> int:
        """Gets the buffer type."""
        ...

    def GetTensorType(self) -> Dict[str, Any]:
        """Gets the tensor type information."""
        ...

    def GetSize(self) -> int:
        """Gets the buffer size in bytes."""
        ...

    def WriteTensor(self, data: List[Union[int, float]], dtype: str) -> None:
        """Writes data to the tensor buffer.
        
        Args:
            data: List of values to write
            dtype: Data type ("float32", "int8", or "int32")
        """
        ...

    def ReadTensor(self, num_elements: int, dtype: str) -> List[Union[int, float]]:
        """Reads data from the tensor buffer.
        
        Args:
            num_elements: Number of elements to read
            dtype: Data type ("float32", "int8", or "int32")
            
        Returns:
            List of values read from the buffer
        """
        ...

def CreateManagedBuffer(
    buffer_type: int,
    element_type: int,
    dimensions: List[int],
    buffer_size: int
) -> TensorBufferWrapper:
    """Creates a managed tensor buffer.
    
    Args:
        buffer_type: Type of buffer to create
        element_type: Element type ID
        dimensions: Shape of the tensor
        buffer_size: Size of the buffer in bytes
        
    Returns:
        A TensorBufferWrapper
    """
    ...

def CreateFromHostMemory(
    element_type: int,
    dimensions: List[int],
    host_memory: Any
) -> TensorBufferWrapper:
    """Creates a tensor buffer from host memory.
    
    Args:
        element_type: Element type ID
        dimensions: Shape of the tensor
        host_memory: Memory object supporting buffer protocol
        
    Returns:
        A TensorBufferWrapper
    """
    ...