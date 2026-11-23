"""tensnet: matrix product state (MPS) and operator (MPO) data structures, finite-difference
operators and AMEn / DMRG linear solvers, as used by the 21-rod bundle solver in scripts/."""

__version__ = "0.1.4"
import sys

from .linalg import *
from .networks import *
from .utils import *

# Adding support for unpickling deprecated module names
sys.modules["tensnet._base_mps"] = sys.modules["tensnet.networks.base_mps"]
