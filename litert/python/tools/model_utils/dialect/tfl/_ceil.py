"""tfl.ceil operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.ceil")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class CeilOp(core.MlirOpBase):
  """Ceil operator.

  Returns element-wise ceil value of the input.
  """

  name = "tfl.ceil"

  x = irdl.operand_def()
  y = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      x: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    x = SSAValue.get(x)
    result_types = [result_type or self._infer_result_type(x)]
    super().__init__(
        operands=[x],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      x: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(x)
    return input_type

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(CeilOp)
def ceil(*args, **kwargs):
  return CeilOp(*args, **kwargs).y
