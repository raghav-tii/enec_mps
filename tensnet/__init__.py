__version__ = "0.1.3"
import torch
import sys

from .networks import *
from .linalg import *
from .utils import *
from .fluids import *
from .learning import *

torch.set_default_device(device())
torch.set_default_dtype(torch.float)

# Adding support for unpickling deprecated module names

sys.modules["tensnet._base_mps"] = sys.modules["tensnet.networks.base_mps"]
