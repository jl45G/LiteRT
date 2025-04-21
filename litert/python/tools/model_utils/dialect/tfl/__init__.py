"""TFL dialect definitions."""

# pylint: disable=redefined-builtin

from ._abs import *
from ._add import *
from ._arg_max import *
from ._arg_min import *
from ._atan2 import *
from ._average_pool_2d import *
from ._batch_matmul import *
from ._bitcast import *
from ._bitwse_xor import *
from ._broadcast_args import *
from ._broadcast_to import *
from ._cast import *
from ._ceil import *
from ._concatenation import *
from ._const import *
from ._conv_2d import *
from ._cos import *
from ._cumsum import *
from ._custom import *
from ._depthwise_conv_2d import *
from ._dequantize import *
from ._div import *
from ._dynamic_update_slice import *
# TODO(cnchan): Update import style with dialect refactor.
# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
# pylint: disable=g-bad-import-order
from .const_bytes_attr import ConstBytesAttr
from .exp import ExpOp, exp
from .fill import FillOp, fill
from .fully_connected import FullyConnectedOp, fully_connected
from .log import LogOp, log
from .log_softmax import LogSoftmaxOp, log_softmax
from .logistic import LogisticOp, logistic
from .max_pool_2d import MaxPool2DOp, max_pool_2d
from .maximum import MaximumOp, maximum
from .mean import MeanOp, mean
from .mul import MulOp, mul
from .reshape import ReshapeOp, reshape
from .rsqrt import RsqrtOp, rsqrt
from .select import SelectOp, select
from .select_v2 import SelectV2Op, select_v2
from .shape import ShapeOp, shape
from .slice import SliceOp, slice
from .softmax import SoftmaxOp, softmax
from .split import SplitOp, split
from .sub import SubOp, sub
from .sum import SumOp, sum
from .tanh import TanhOp, tanh
from .tile import TileOp, tile
from .transpose import TransposeOp, transpose
# pylint: enable=g-importing-member
# pylint: enable=g-multiple-import
# pylint: enable=g-bad-import-order
