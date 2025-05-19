"""FileCheck test for func dialect op builders."""

# RUN: %PYTHON %s | FileCheck %s --dump-input=always

from collections.abc import Callable

from absl import app
from xdsl import irdl

import google3.third_party.odml.litert.litert.python.tools.model_utils as mu
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir

SSAValue = irdl.SSAValue


def TY(shape, elty):  # pylint: disable=invalid-name
  """Short-cut for creating a RankedTensorType."""
  return mlir.RankedTensorType(shape, elty)


def build_module_and_print(*input_types):
  def inner(builder: Callable[..., SSAValue | tuple[SSAValue, ...]]):
    module = model_builder.build_module_from_py_func(*input_types)(builder)
    testing.print_ir(getattr(builder, "__name__"), module)

    # Check no errors during round-trip conversion
    ir_module = mu.transform.convert_to_mlir(module)
    mu.transform.read_mlir(operation=ir_module)
    print("Round-trip transformation succeeded")

  return inner


# pylint: disable=redefined-builtin
# pylint: disable=line-too-long
# pylint: disable=unused-variable
@testing.run_in_ir_context
def main(_) -> None:

  @build_module_and_print(TY([2, 2], "f32"))
  def return_no_values(_):
    # CHECK-LABEL: return_no_values
    func.ReturnOp()
    # CHECK: return

  @build_module_and_print(TY([2, 2], "f32"))
  def return_one_value(x):
    # CHECK-LABEL: return_one_value
    func.ReturnOp(x)
    # CHECK: return %arg0 : tensor<2x2xf32>

  @build_module_and_print(TY([2, 2], "f32"), TY([2, 2], "f32"))
  def return_two_values(x, y):
    # CHECK-LABEL: return_two_values
    func.ReturnOp(x, y)
    # CHECK: return %arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>

  @build_module_and_print(TY([2, 2], "f32"))
  def return_build_no_values(_):
    # CHECK-LABEL: return_build_no_values
    func.ReturnOp.build(operands=[], result_types=[])
    # CHECK: return

  @build_module_and_print(TY([2, 2], "f32"))
  def return_build_one_value(x):
    # CHECK-LABEL: return_build_one_value
    func.ReturnOp.build(operands=[x], result_types=[TY([2, 2], "f32")])
    # CHECK: return %arg0 : tensor<2x2xf32>

  @build_module_and_print(TY([2, 2], "f32"), TY([2, 2], "f32"))
  def return_build_two_values(x, y):
    # CHECK-LABEL: return_build_two_values
    func.ReturnOp.build(
        operands=[x, y],
        result_types=[TY([2, 2], "f32"), TY([2, 2], "f32")],
    )
    # CHECK: return %arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>


if __name__ == "__main__":
  app.run(main)
