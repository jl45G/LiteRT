"""tfl.equal operation definition."""

import jax.numpy as jnp
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.equal")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class EqualOp(core.MlirOpBase):
  """Equal operator.

  Returns the truth element of x == y element-wise.
  """

  name = "tfl.equal"

  x = irdl.operand_def()
  y = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      x: SSAValue | core.MlirOpBase,
      y: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    x_val = SSAValue.get(x)
    y_val = SSAValue.get(y)
    result_types = [result_type or self._infer_result_type(x_val, y_val)]
    super().__init__(
        operands=[x_val, y_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      x_val: SSAValue | core.MlirOpBase,
      y_val: SSAValue | core.MlirOpBase,
  ):
    x_type = _utils.get_tensor_type(x_val)
    y_type = _utils.get_tensor_type(y_val)

    # ResultsBroadcastableShape trait implies output shape is broadcasted shape.
    try:
      output_shape = jnp.broadcast_shapes(x_type.shape, y_type.shape)
    except ValueError as e:
      raise ValueError(
          f"Shapes are not broadcastable: {x_type.shape} vs {y_type.shape}"
      ) from e

    # Result element type is always boolean (i1) according to the spec.
    return mlir.RankedTensorType(output_shape, "i1")

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(EqualOp)
def equal(*args, **kwargs):
  return EqualOp(*args, **kwargs).output
