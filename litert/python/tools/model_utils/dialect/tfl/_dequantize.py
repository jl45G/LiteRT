"""tfl.dequantize operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.dequantize")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class DequantizeOp(core.MlirOpBase):
  """Dequantize operator.

  Converts quantized array of integers to floating-points according to the
  quantization parameters.
  """

  name = "tfl.dequantize"

  input = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase = None,
      *,
      location=None,
  ):
    input_val = SSAValue.get(input)
    result_types = [result_type]
    super().__init__(
        operands=[input_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(DequantizeOp)
def dequantize(*args, **kwargs):
  return DequantizeOp(*args, **kwargs).output
