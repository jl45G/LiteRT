"""FileCheck test for MLIR builtin dialect ops and builders."""

# RUN: %PYTHON %s | FileCheck %s --dump-input=always

from absl import app
from xdsl import irdl

import google3.third_party.odml.litert.litert.python.tools.model_utils as mu
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir

SSAValue = irdl.SSAValue


def TY(shape, elty):  # pylint: disable=invalid-name
  """Short-cut for creating a RankedTensorType."""
  return mlir.RankedTensorType(shape, elty)


# pylint: disable=redefined-builtin
# pylint: disable=line-too-long
# pylint: disable=unused-variable
@testing.run_in_ir_context
def main(_) -> None:

  @model_builder.build_module_from_py_func(TY([2, 2], "f32"))
  def simple_op_with_array_attr(x):
    return mlir.MlirOp(
        "test.simple_op",
        [x],
        result_types=[TY([2, 2], "f32")],
        attributes={
            "arr": mlir.ArrayAttr(
                [mlir.StringAttr("foo"), mlir.IntegerAttr(123)],
            )
        },
    ).results[0]

  testing.print_ir("simple_op_with_array_attr", simple_op_with_array_attr)
  # CHECK-LABEL: simple_op_with_array_attr
  # CHECK: test.simple_op
  # CHECK-SAME: {arr = ["foo", 123 : i32]}

  simple_op_with_array_attr_round_tripped, _ = mu.transform.read_mlir(
      content=mu.transform.convert_to_mlir(
          simple_op_with_array_attr
      ).operation.get_asm()
  )
  testing.print_ir(
      "simple_op_with_array_attr_round_tripped",
      simple_op_with_array_attr_round_tripped,
  )
  # CHECK-LABEL: simple_op_with_array_attr_round_tripped
  # CHECK: test.simple_op
  # CHECK-SAME: {arr = ["foo", 123 : i32]}

  @model_builder.build_module_from_py_func(TY([2, 2], "f32"))
  def simple_op_with_dict_attr(x):
    return mlir.MlirOp(
        "test.simple_op",
        [x],
        result_types=[TY([2, 2], "f32")],
        attributes={
            "dict": mlir.DictAttr({
                "foo": mlir.StringAttr("foo"),
                "bar": mlir.IntegerAttr(123),
            })
        },
    ).results[0]

  testing.print_ir("simple_op_with_dict_attr", simple_op_with_dict_attr)
  # CHECK-LABEL: simple_op_with_dict_attr
  # CHECK: test.simple_op
  # CHECK-SAME: bar = 123 : i32
  # CHECK-SAME: foo = "foo"

  simple_op_with_dict_attr_round_tripped, _ = mu.transform.read_mlir(
      content=mu.transform.convert_to_mlir(
          simple_op_with_dict_attr
      ).operation.get_asm()
  )
  testing.print_ir(
      "simple_op_with_dict_attr_round_tripped",
      simple_op_with_dict_attr_round_tripped,
  )
  # CHECK-LABEL: simple_op_with_dict_attr_round_tripped
  # CHECK: test.simple_op
  # CHECK-SAME: bar = 123 : i32
  # CHECK-SAME: foo = "foo"


if __name__ == "__main__":
  app.run(main)
