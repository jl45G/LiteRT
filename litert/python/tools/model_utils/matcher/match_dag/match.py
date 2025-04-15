from xdsl.irdl import SSAValue

from google3.third_party.odml.litert.litert.python.tools import model_utils as mu
from google3.third_party.odml.litert.litert.python.tools.model_utils.matcher import context
from google3.third_party.odml.litert.litert.python.tools.model_utils.matcher.match_dag import executor
from google3.third_party.odml.litert.litert.python.tools.model_utils.matcher.match_dag import generator


def MatchDag(dag: str, op_or_value: mu.core.MlirOpBase | SSAValue):

  code = generator.parse_match_dag(dag)

  if not isinstance(op_or_value, SSAValue):
    op = op_or_value
    if not op.results:
      raise context.NoMatchException()
    value = op.results[0]
  else:
    value = op_or_value

  results = executor.execute_match_dag(code, value)
  return results
