"""tfl.cos operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.cos")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class CosOp(core.MlirOpBase):
  """Cosine operator.

  Computes element-wise Cosine of input
  """

  name = "tfl.cos"

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


@_utils.op_builder_wraps(CosOp)
def cos(*args, **kwargs):
  return CosOp(*args, **kwargs).y
