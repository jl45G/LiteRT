from google3.third_party.odml.litert.litert.python.tensor_buffer_wrapper import _pywrap_litert_tensor_buffer_wrapper as _tb


class TensorBuffer:
  """A higher-level class wrapping a LiteRtTensorBuffer PyCapsule."""

  def __init__(self, capsule):
    # The capsule is the actual handle to the C++ LiteRtTensorBuffer
    self._capsule = capsule

  @classmethod
  def create_from_host_memory(cls, data, dtype: str, num_elements: int):
    """Creates a new TensorBuffer from `data` (a Python object

    with buffer protocol, e.g. bytes or numpy array).
    """
    cap = _tb.CreateTensorBufferFromHostMemory(data, dtype, num_elements)
    return cls(cap)

  def write(self, data_list, dtype: str):
    """Writes python list of values to this buffer."""
    _tb.WriteTensor(self._capsule, data_list, dtype)

  def read(self, num_elements: int, dtype: str):
    """Reads from this buffer."""
    return _tb.ReadTensor(self._capsule, num_elements, dtype)

  def destroy(self):
    """Explicitly free resources. After this, self._capsule is invalid."""
    _tb.DestroyTensorBuffer(self._capsule)
    self._capsule = None

  @property
  def capsule(self):
    """Return the underlying PyCapsule (if needed)."""
    return self._capsule
