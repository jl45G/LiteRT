"""LiteRT Tensor C++ code generator from TFLite flatbuffer."""

import contextlib
import dataclasses
import re
from typing import Any, Callable, Sequence, cast

import numpy as np
from xdsl import irdl

from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl

__all__ = ["TensorCppCodeGenerator"]

SSAValue = irdl.SSAValue

OpGeneratorT = Callable[["TensorCppCodeGenerator", mu.core.MlirOpBase], None]

op_generators: dict[str, OpGeneratorT] = {}


def register_op_generator(op_name: str):
  def decorator(fn: OpGeneratorT):
    op_generators[op_name] = fn
    return fn

  return decorator


def attr_expr(attr: Any) -> str:
  """Returns the C++ expression for the given attribute."""

  if isinstance(attr, mlir.StringAttr):
    return f'"{attr.data}"'
  if isinstance(attr, mlir.IntegerAttr):
    return str(attr.data)
  if isinstance(attr, mlir.FloatAttr):
    return str(attr.data)
  if isinstance(attr, mlir.BoolAttr):
    return "true" if attr.data else "false"
  raise ValueError(f"Unsupported attr: {attr}")


def fa_expr(attr: mlir.StringAttr | str) -> str:
  """Returns the C++ expression for the given fused activation."""

  if isinstance(attr, mlir.StringAttr):
    attr = attr.data

  if attr == "NONE":
    return "FusedActivation::kActNone"
  if attr == "RELU":
    return "FusedActivation::kActRelu"
  if attr == "RELU6":
    return "FusedActivation::kActRelu6"
  raise ValueError(f"Unsupported fused activation: {attr}")


def tensor_type_expr(elty: str) -> str:
  elty = elty.lower()
  if elty == "f32":
    return "Type::kFP32"
  if elty == "i32":
    return "Type::kI32"
  raise ValueError(f"Unsupported tensor type: {elty}")


def value_expr(arr: Any) -> str:
  """Returns the C++ expression for the given value (array or scalar)."""

  if isinstance(arr, (int, float)):
    return str(arr)
  if isinstance(arr, np.ndarray):
    arr = arr.tolist()

  if isinstance(arr, (list, tuple)):
    return "{" + ", ".join(map(value_expr, arr)) + "}"

  raise ValueError(f"Unsupported value type: {type(arr)}")


@dataclasses.dataclass(frozen=True)
class TensorCppCodeIO:
  """TensorCppCodeGenerator input/output information."""

  name: str
  ssa_value: SSAValue

  @property
  def shape(self) -> tuple[int, ...]:
    return tuple(self.ssa_value.type.shape)

  @property
  def tensor_type(self) -> str:
    return tensor_type_expr(self.ssa_value.type.elty)


@dataclasses.dataclass(frozen=True)
class TensorCppCodeWeight:
  """TensorCppCodeGenerator weight information."""

  name: str
  ssa_value: SSAValue

  def numpy(self) -> np.ndarray:
    return self.origin_op.numpy()

  @property
  def origin_op(self) -> tfl.ConstantOp:
    owner = self.ssa_value.owner
    if not isinstance(owner, tfl.ConstantOp):
      raise ValueError(f"Weight {self.name} is not a constant op")
    return owner

  @property
  def shape(self) -> tuple[int, ...]:
    return tuple(self.ssa_value.type.shape)

  @property
  def tensor_type(self) -> str:
    return tensor_type_expr(self.ssa_value.type.elty)


def to_upper_camel_case(name: str) -> str:
  """Converts a name to UpperCamelCase."""
  return name.title().replace("_", "").replace(" ", "")


@dataclasses.dataclass
class TensorCppCodeGenerator:
  """LiteRT Tensor C++ code generator from TFLite flatbuffer.

  This class is used to generate C++ codes for a given signature in a  model. It
  traverses the model in topological order and generate C++ codes for each op.
  The generated codes are available via the following methods:
  - model: The C++ codes for the model function.
  """

  def __init__(
      self,
      module: str | bytes | mlir.ModuleOp,
      signature_name: str = "serving_default",
      *,
      generated_model_name: str | None = None,
  ):
    if isinstance(module, str):
      self.module, self._ctx = mu.read_flatbuffer(module)
    elif isinstance(module, bytes):
      self.module, self._ctx = mu.read_flatbuffer(content=module)
    elif isinstance(module, mlir.ModuleOp):
      self.module = module
      self._ctx = contextlib.nullcontext()
    else:
      raise ValueError(f"Unsupported module type: {type(module)}")

    self.signature_name = str(signature_name)
    self.generated_model_name = generated_model_name or self.signature_name

    # Invoke the entry point to generate C++ codes.
    self._generate()

  def model(self) -> str:
    """Returns the C++ codes for the model function."""
    lines = []
    indent_level = 0

    def empty_line() -> None:
      nonlocal lines
      lines.append("")

    @contextlib.contextmanager
    def indent():
      nonlocal indent_level
      indent_level += 1
      yield
      indent_level -= 1

    def write(*code: str | Sequence[str]):
      nonlocal lines, indent_level
      for line in code:
        if isinstance(line, (list, tuple)):
          write(*line)
        else:
          lines.append(" " * indent_level + line)

    model_name = self.generated_model_name
    inputs_struct_name = f"{to_upper_camel_case(model_name)}Inputs"
    output_struct_name = f"{to_upper_camel_case(model_name)}Outputs"

    # Add includes
    write("#include <string>")
    write("#include <unordered_map>")
    write("#include <utility>")
    write('#include "third_party/odml/litert/tensor/tensor.h"')
    write('#include "third_party/odml/litert/tensor/arithmetic.h"')
    write("using namespace litert::tensor;")
    empty_line()
    empty_line()

    # Generated code input struct
    write(f"struct {inputs_struct_name} {{")
    with indent():
      write("// Input tensors")
      for info in self._inputs:
        write(
            f'Tensor {info.name} = Tensor({{.name = "{info.name}", .type ='
            f" {info.tensor_type}, .shape = {value_expr(info.shape)}}});"
        )
      write("// Weight tensors")
      for info in self._weights:
        write(
            f'Tensor {info.name} = Tensor({{.name = "{info.name}", .type ='
            f" {info.tensor_type}, .shape = {value_expr(info.shape)}}});"
        )

      write("std::unordered_map<std::string, Tensor*> tensors() {")
      with indent():
        write("std::unordered_map<std::string, Tensor*> tensors;")
        for info in self._inputs + self._weights:
          write(f'tensors.emplace("{info.name}", &{info.name});')
        write("return std::move(tensors);")
      write("}")
    write("};")

    # Generated code output struct
    empty_line()
    write(f"struct {output_struct_name} {{")
    with indent():
      write("// Output tensors")
      for info in self._outputs:
        write(f"Tensor {info.name};")

      write("std::unordered_map<std::string, Tensor*> tensors() {")
      with indent():
        write("std::unordered_map<std::string, Tensor*> tensors;")
        for info in self._outputs:
          write(f'tensors.emplace("{info.name}", &{info.name});')
        write("return std::move(tensors);")
      write("}")
    write("};")

    # Generate model function body
    empty_line()
    write(
        f"inline {output_struct_name} {model_name}"
        + f"({inputs_struct_name} inputs__) {{",
    )
    with indent():
      # Assign input tensors to local variables.
      for info in self._inputs:
        write(f"Tensor& {self.tensor(info.ssa_value)} = inputs__.{info.name};")
      for info in self._weights:
        write(f"Tensor& {self.tensor(info.ssa_value)} = inputs__.{info.name};")

      empty_line()
      for stmt in self.op_stmts:
        write(stmt)

      empty_line()
      write(f"return {output_struct_name}{{")
      with indent():
        for info in self._outputs:
          write(f".{info.name} = std::move({self.tensor(info.ssa_value)}),")
      write("};")

    write("}")
    empty_line()
    return "\n".join(lines)

  def _generate(self) -> None:
    """Entry point for building C++ codes components from the module."""
    self._tensor_name_mapping = {}
    self._inputs = []
    self._outputs = []
    self._weights = []

    self.op_stmts = []

    target_fn = None
    for fn in self.module.ops:
      sig = mu.SignatureBuilder(fn)
      if sig.name == self.signature_name:
        target_fn = fn
        break

    if not target_fn or not isinstance(target_fn, func.FuncOp):
      raise ValueError(
          f"Signature {self.signature_name} not found in the module."
      )

    sig = mu.SignatureBuilder(target_fn)

    for name, ssa_value in sig.get_inputs_map().items():
      self._generate_input(ssa_value, name)

    for name, ssa_value in sig.get_outputs_map().items():
      self._generate_output(ssa_value, name)

    for op in target_fn.ops:
      self._generate_op(op)

  def tensor(self, value: SSAValue) -> str:
    """Get tensor name for a value."""
    if value not in self._tensor_name_mapping:
      owner = value.owner
      # Special case for weights. If the value comes from a constant op and the
      # tensor name is not generated yet, treate it as a weight and generate the
      # tensor name with the weight name.
      if isinstance(owner, tfl.ConstantOp):
        self._generate_weight(owner)

    tensor_name = self._tensor_name_mapping.get(value, None)
    if tensor_name is None:
      raise ValueError(f"Tensor name not found for value: {value}")
    return tensor_name

  def _generate_weight(self, op: tfl.ConstantOp) -> None:
    value = op.results[0]
    if value in self._tensor_name_mapping:
      return
    name = f"w_{len(self._weights)}"
    self._tensor_name_mapping[value] = name
    self._weights.append(TensorCppCodeWeight(name=name, ssa_value=value))

  def _generate_input(self, value: SSAValue, name: str | None = None) -> None:
    if name is None:
      name = f"input_{len(self._inputs)}"
    self._tensor_name_mapping[value] = name
    self._inputs.append(TensorCppCodeIO(name=name, ssa_value=value))

  def _generate_output(self, value: SSAValue, name: str | None = None) -> None:
    if name is None:
      name = f"output_{len(self._outputs)}"

    var_name = name + "_out__"
    self._tensor_name_mapping[value] = var_name
    self._outputs.append(TensorCppCodeIO(name=name, ssa_value=value))

  def _generate_op(self, op: mu.core.MlirOpBase) -> None:
    """Generates C++ codes for the given op."""

    if op.name in (tfl.ConstOp.name, tfl.NoValueOp.name, func.ReturnOp.name):
      # Special case for ops that are handled by other generators.
      return

    generator = op_generators.get(str(op.name), None)
    if not generator:
      raise ValueError(f"Op generator not found: {op}")

    for result in op.results:
      # Result name may be defined when the result is a model output.
      name = self._tensor_name_mapping.get(result, f"t_{len(self.op_stmts)}")
      self._tensor_name_mapping[result] = name

    # Extract layer string from op location.
    loc = str(op.location)
    if loc := re.findall('".*"', loc):
      loc = loc[0].split(";")[0].strip().strip('";')
      self.op_stmts.append(f"// {loc}")

    generator(self, op)


@register_op_generator("tfl.add")
def _gen_add(self: TensorCppCodeGenerator, op: tfl.AddOp) -> None:
  assert op.fused_activation_function == "NONE"
  x, y = op.operands
  z = op.results[0]
  # TODO(cnchan): Handle fused_activation_function
  self.op_stmts.append(
      f"Tensor {self.tensor(z)} = Add({self.tensor(x)}, {self.tensor(y)});"
  )


@register_op_generator("tfl.fully_connected")
def _gen_fully_connected(self, op: tfl.FullyConnectedOp):
  x, filter_, bias = op.operands
  out = op.results[0]
  if isinstance(bias.owner, tfl.NoValueOp):
    # bias is optional
    bias = ""
  else:
    bias = f", {self.tensor(bias)}"

  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = FullyConnected({self.tensor(x)},"
      f" {self.tensor(filter_)}{bias},"
      f" /*fused_activation_function=*/{fa_expr(op.attributes['fused_activation_function'])},"
      f" /*keep_num_dims=*/{attr_expr(op.attributes['keep_num_dims'])});"
  )


@register_op_generator("tfl.transpose")
def _gen_transpose(self, op: tfl.TransposeOp):
  x, perm = op.operands
  out = op.results[0]

  # Assume static
  perm = cast(tfl.ConstOp, perm.owner).numpy()
  perm_s = value_expr(perm)
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Transpose({self.tensor(x)}, {perm_s});"
  )


@register_op_generator("tfl.pad")
def _gen_pad(self, op):
  x, pad = op.operands
  out = op.results[0]

  pad = pad.owner.numpy().flatten()
  pad_s = value_expr(pad)
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Pad({self.tensor(x)}, {pad_s});"
  )


@register_op_generator("tfl.sum")
def _gen_sum(self, op: tfl.SumOp):
  x, axis = op.operands
  out = op.results[0]

  # Assume static
  axis_s = value_expr(cast(tfl.ConstOp, axis.owner).numpy())
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Sum({self.tensor(x)}, {axis_s},"
      f" /*keep_dims=*/{attr_expr(op.attributes['keep_dims'])});"
  )


@register_op_generator("tfl.rsqrt")
def _gen_rsqrt(self, op):
  x = op.operands[0]
  out = op.results[0]
  self.op_stmts.append(f"Tensor {self.tensor(out)} = Rsqrt({self.tensor(x)});")


@register_op_generator("tfl.sub")
def _gen_sub(self, op: tfl.SubOp):
  assert op.fused_activation_function == "NONE"
  x, y = op.operands
  z = op.results[0]
  # TODO(cnchan): Handle fused_activation_function
  self.op_stmts.append(
      f"Tensor {self.tensor(z)} = Sub({self.tensor(x)}, {self.tensor(y)});"
  )


@register_op_generator("tfl.mul")
def _gen_mul(self, op: tfl.MulOp):
  assert op.fused_activation_function == "NONE"
  x, y = op.operands
  z = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(z)} = Mul({self.tensor(x)}, {self.tensor(y)});"
  )


@register_op_generator("tfl.reshape")
def _gen_reshape(self, op: tfl.ReshapeOp):
  x, shape = op.operands
  out = op.results[0]

  # Assume static
  shape = cast(tfl.ConstOp, shape.owner).numpy()
  shape_s = value_expr(shape)
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Reshape({self.tensor(x)}, {shape_s});"
  )


@register_op_generator("tfl.embedding_lookup")
def _gen_embedding_lookup(self, op: tfl.EmbeddingLookupOp):
  indices, weights = op.operands
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = EmbeddingLookup({self.tensor(indices)},"
      f" {self.tensor(weights)});"
  )


@register_op_generator("tfl.depthwise_conv_2d")
def _gen_depthwise_conv_2d(self, op: tfl.DepthwiseConv2DOp):
  x, filter_, bias = op.operands
  out = op.results[0]
  if "none" in str(bias.type).lower():
    # bias is optional
    bias = ""
  else:
    bias = f", {self.tensor(bias)}"
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = DepthwiseConv2D({self.tensor(x)},"
      f" {self.tensor(filter_)}{bias},"
      f" /*padding=*/{attr_expr(op.attributes['padding'])},"
      f" /*stride_w=*/{attr_expr(op.attributes['stride_w'])},"
      f" /*stride_h=*/{attr_expr(op.attributes['stride_h'])},"
      f" /*dilation_w_factor=*/{attr_expr(op.attributes['dilation_w_factor'])},"
      f" /*dilation_h_factor=*/{attr_expr(op.attributes['dilation_h_factor'])},"
      f" /*fused_activation_function=*/{fa_expr(op.attributes['fused_activation_function'])}"
      ");"
  )


@register_op_generator("tfl.concatenation")
def _gen_concatenation(self, op: tfl.ConcatenationOp):
  xs = op.operands
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} ="
      f" Concatenation({{{','.join([self.tensor(x) for x in xs])}}},"
      f" /*axis=*/{attr_expr(op.attributes['axis'])},"
      f" /*fused_activation_function=*/{fa_expr(op.attributes['fused_activation_function'])}"
      ");"
  )


@register_op_generator("tfl.conv_2d")
def _gen_conv_2d(self, op: tfl.Conv2DOp):
  x, filter_, bias = op.operands
  out = op.results[0]
  if "none" in str(bias.type).lower():
    # bias is optional
    bias = ""
  else:
    bias = f", {self.tensor(bias)}"
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Conv2D({self.tensor(x)},"
      f" {self.tensor(filter_)}{bias},"
      f" /*padding=*/{attr_expr(op.attributes['padding'])},"
      f" /*stride_w=*/{attr_expr(op.attributes['stride_w'])},"
      f" /*stride_h=*/{attr_expr(op.attributes['stride_h'])},"
      f" /*dilation_w_factor=*/{attr_expr(op.attributes['dilation_w_factor'])},"
      f" /*dilation_h_factor=*/{attr_expr(op.attributes['dilation_h_factor'])},"
      f" /*fused_activation_function=*/{fa_expr(op.attributes['fused_activation_function'])}"
      ");"
  )


@register_op_generator("tfl.cast")
def _gen_cast(self, op: tfl.CastOp):
  x = op.operands[0]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Cast({self.tensor(x)},"
      f" /*type=*/{tensor_type_expr(out.type.elty)});"
  )


@register_op_generator("tfl.gelu")
def _gen_gelu(self, op: tfl.GeluOp):
  x = op.operands[0]
  out = op.results[0]
  self.op_stmts.append(f"Tensor {self.tensor(out)} = Gelu({self.tensor(x)});")


@register_op_generator("tfl.cos")
def _gen_cos(self, op: tfl.CosOp):
  x = op.operands[0]
  out = op.results[0]
  self.op_stmts.append(f"Tensor {self.tensor(out)} = Cos({self.tensor(x)});")


@register_op_generator("tfl.sin")
def _gen_sin(self, op):
  x = op.operands[0]
  out = op.results[0]
  self.op_stmts.append(f"Tensor {self.tensor(out)} = Sin({self.tensor(x)});")


@register_op_generator("tfl.softmax")
def _gen_softmax(self, op: tfl.SoftmaxOp):
  x = op.operands[0]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Softmax({self.tensor(x)});"
  )


@register_op_generator("tfl.greater")
def _gen_greater(self, op):
  x = op.operands[0]
  y = op.operands[1]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Greater({self.tensor(x)},"
      f" {self.tensor(y)});"
  )


@register_op_generator("tfl.less")
def _gen_less(self, op):
  x = op.operands[0]
  y = op.operands[1]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Less({self.tensor(x)}, {self.tensor(y)});"
  )


@register_op_generator("tfl.logical_and")
def _gen_logical_and(self, op):
  x = op.operands[0]
  y = op.operands[1]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = LogicalAnd({self.tensor(x)},"
      f" {self.tensor(y)});"
  )


@register_op_generator("tfl.logical_or")
def _gen_logical_or(self, op):
  x = op.operands[0]
  y = op.operands[1]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = LogicalOr({self.tensor(x)},"
      f" {self.tensor(y)});"
  )


@register_op_generator("tfl.select_v2")
def _gen_select_v2(self, op):
  x = op.operands[0]
  y = op.operands[1]
  z = op.operands[2]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = SelectV2({self.tensor(x)},"
      f" {self.tensor(y)}, {self.tensor(z)});"
  )


@register_op_generator("tfl.minimum")
def _gen_minimum(self, op):
  x = op.operands[0]
  y = op.operands[1]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Minimum({self.tensor(x)},"
      f" {self.tensor(y)});"
  )


@register_op_generator("tfl.maximum")
def _gen_maximum(self, op):
  x = op.operands[0]
  y = op.operands[1]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Maximum({self.tensor(x)},"
      f" {self.tensor(y)});"
  )


@register_op_generator("tfl.slice")
def _gen_slice(self, op):
  x = op.operands[0]
  begin = op.operands[1]
  size = op.operands[2]
  # Assume begin and size are static
  begin = begin.owner.numpy()
  begin_s = value_expr(begin)
  size = size.owner.numpy()
  size_s = value_expr(size)
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = Slice({self.tensor(x)}, {begin_s},"
      f" {size_s});"
  )


@register_op_generator("tfl.dynamic_update_slice")
def _gen_dynamic_update_slice(self, op):
  x = op.operands[0]
  y = op.operands[1]
  z = op.operands[2]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = DynamicUpdateSlice({self.tensor(x)},"
      f" {self.tensor(y)}, {self.tensor(z)});"
  )


@register_op_generator("tfl.batch_matmul")
def _gen_batch_matmul(self, op: tfl.BatchMatMulOp):
  x = op.operands[0]
  y = op.operands[1]
  out = op.results[0]
  self.op_stmts.append(
      f"Tensor {self.tensor(out)} = BatchMatMul({self.tensor(x)},"
      f" {self.tensor(y)}, /*adj_x=*/{attr_expr(op.attributes['adj_x'])},"
      f" /*adj_y=*/{attr_expr(op.attributes['adj_y'])});"
  )
