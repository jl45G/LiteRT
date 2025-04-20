"""tfl.cast operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.cast")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class CastOp(core.MlirOpBase):
  """Cast operator.

  Casts input from input type to output type.
  """

  name = "tfl.cast"

  input = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase,
      *,
      location=None,
  ):
    input_val = SSAValue.get(input)
    super().__init__(
        operands=[input_val],
        result_types=[result_type],
        location=location,
        attributes={},
    )

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(CastOp)
def cast(*args, **kwargs):
  return CastOp(*args, **kwargs).output
