import warnings
from typing import Iterator, List, Tuple, Union

from einops import einsum, rearrange

from ..networks import MPS, MPS_Node
from ..utils import validate_tracks


class Tensor_Track:
    def __init__(self, _tracks: List[MPS]) -> None:
        self.tracks = validate_tracks(_tracks)

    def __getitem__(self, i):
        return self.tracks[i]

    def ndim(self):
        return len(self.tracks)

    def __len__(self):
        return len(self.tracks[0])

    def __setitem__(self, i, val, j=None):
        if j is None:
            self.tracks[i] = val
        else:
            self.tracks[i][j] = val

    def __iter__(self) -> Iterator[MPS]:
        return iter(self.tracks)

    def copy(self):
        _copy = Tensor_Track([x.copy() for x in self])
        return _copy

    def __add__(self, T: "Tensor_Track"):
        assert len(self) == len(
            T
        ), "Addition possible only with Tracks of equal number of sites/nodes."
        sum = Tensor_Track([MPS([MPS_Node()] * len(self))] * self.ndim())
        for ix, (m1, m2) in enumerate(zip(self, T)):
            sum[ix] = m1 + m2
        return sum

    def __mul__(self, T: Union[float, "Tensor_Track"]):
        prod = self.copy()
        if isinstance(T, float):
            prod[0] = self[0] * T
        elif isinstance(T, Tensor_Track):
            assert all(len(self) == len(T)), ...
            "Element-wise multiplication possible only with Tracks of same shaped tensors."
            for ix, (m1, m2) in enumerate(zip(self, T)):
                prod[ix] = m1 * m2
        return prod

    def contract(self):
        warnings.warn(
            "Careful while calling the method of contraction. This will lead to exponential memory usage",
            ResourceWarning,
        )
        new_mps = []
        for ix, x in enumerate(zip(*[mps for mps in self])):
            new_mps.append(MPS_Node(contract_crossbond(x)))
        new_mps = MPS(new_mps)
        for ix, x in enumerate(new_mps):
            if ix == 0:
                temp = x.copy()
            else:
                temp = MPS_Node(temp.contract(x))
        vec = temp.tensor.reshape(-1)
        return vec


def contract_crossbond(x: Tuple[MPS_Node]):
    tensors = [node.tensor for node in x]
    return rearrange(
        einsum(tensors[0], tensors[1], "i j k l,p q r l->i p j q k r"),
        "i p j q k r->(i p)(j q)(k r)",
    )
