from torch.nn import Parameter
from torch.utils._pytree import tree_map
from torch.amp import autocast
from torch import svd
from torch import det
from torch.nn.functional import pairwise_distance
from torch import isin
from torch.distributions import Geometric
# from torch import attention