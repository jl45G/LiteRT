r"""LiteRT Tensor CodeGen CLI.

Usage:
  blaze run -c opt //third_party/odml/litert/tensor/codegen:cli -- {INPUT_MODEL}
"""

import fire
from google3.third_party.odml.litert.tensor.codegen import codegen_cpp


def cpp(
    input_model: str,
    signature: str = "serving_default",
    generated_model_name: str | None = None,
):
  generator = codegen_cpp.TensorCppCodeGenerator(
      input_model,
      signature_name=signature,
      generated_model_name=generated_model_name,
  )
  print(generator.model())


def main(_):
  fire.Fire(cpp)


if __name__ == "__main__":
  fire.run()
