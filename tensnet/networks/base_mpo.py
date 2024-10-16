import math
import warnings
from typing import List, Union

import opt_einsum as oe
import torch
from einops import einsum, rearrange

from ..utils import (
    device,
    einpat_part_opip,
    get_local_inds,
    unshuffle_axis,
    validate_nodes,
    qr,
    rq,
)
from .base_mps import MPS, MPS_Node, random_mps


class MPO_Node:
    def __init__(
        self,
        tensor=torch.tensor([]),
        canonical: str = "None",
        boundary: str = None,
    ):
        self.tensor = tensor
        self.canonical = canonical
        self.boundary = boundary

    def __matmul__(
        self,
        other: Union[MPS_Node, "MPO_Node"],
        einstr=["l i r o, a i b -> l a o r b", "l a o r b ->(l a) o (r b)"],
    ):
        """Applying MPO node on MPS"""
        if type(other) == MPS_Node:
            new_tensor = rearrange(
                einsum(self.tensor, other.tensor, einstr[0]), einstr[1]
            )
            return MPS_Node(new_tensor, boundary=other.boundary)
        if type(other) == MPO_Node:
            einstr = ["l i r d, a d b o -> l a i r b o", "l a i r b o->(l a) i (r b) o"]
            new_tensor = rearrange(
                einsum(self.tensor, other.tensor, einstr[0]), einstr[1]
            )
            return MPO_Node(new_tensor, boundary=other.boundary)

    def copy(self):
        """Returns a deep copy of the node with the same attributes."""
        return MPO_Node(self.tensor.detach().clone(), self.canonical, self.boundary)

    def contract(
        self, other: "MPO_Node", einstr=["l b e, e q r->l b q r", "l b q r-> l (b q) r"]
    ):
        """Given two nodes, and einstein summation notation, contract the corresponding tensors and return the tensor.

        Args:
            other (MPS_Node): The second node to be contracted with the current node.
            einstr (list, optional): The Einstein Summation string,
                along with the rearrangement pattern for einops package.
                Defaults to ['l b e, e q r->l b q r','l b q r-> l (b q) r'].

        Returns:
            Tensor: Returns the contracted tensor (of data type that is used in the initialization of the MPS_Node.)
        """
        return rearrange(einsum(self.tensor, other.tensor, einstr[0]), einstr[1])

    def transpose(self):
        einstr = "l i r o -> l o r i"
        tensor = rearrange(self.tensor.detach().clone(), einstr)
        return MPO_Node(tensor)

    def dagger(self):
        tensor = torch.conj(self.tensor)
        return MPO_Node(tensor)

    def conj(self):
        tensor = torch.conj(self.tensor)
        return MPO_Node(tensor)

    def outer(self, other):
        tensor = self.contract(
            other,
            einstr=[
                "l1 i1 r1 o2, l2 i2 r2 o2 -> l1 l2 i1 i2 r1 r2 o1 o2",
                "l1 l2 i1 i2 r1 r2 o1 o2 -> (l1 l2) (i1 i2) (r1 r2) (o1 o2)",
            ],
        )
        return MPO_Node(tensor)

    @property
    def order(self):
        return len(self.tensor.shape)

    @property
    def ndim(self):
        return len(self.tensor.shape)

    @property
    def dim(self):
        return list(self.tensor.shape)

    @property
    def shape(self):
        return self.tensor.shape

    def reshape(self, shape):
        self.tensor = self.tensor.reshape(shape)


class MPO:
    def __init__(
        self, nodes: list[MPO_Node] = None, validate: bool = True, canonical=None
    ):
        self.canonical = canonical
        if nodes is not None:
            if validate is True:
                self.nodes = validate_nodes(nodes)
            else:
                self.nodes = nodes
        else:
            self.nodes = []

    def __len__(self):
        return self.nodes.__len__()

    def __repr__(self):
        main_str = f"{type(self)} | Length of MPO: {len(self)} | \n\n"
        str1 = ""
        str2 = ""
        str3 = ""
        str4 = ""
        for i in self.shapes():
            str1 = str1 + str(i[0]) + "--T--"
            str2 = str2 + " " * len(str(i[0])) + "  |  "
            str3 = str3 + " " * len(str(i[0])) + "  " + str(i[1]) + "  "
            str4 = str4 + " " * len(str(i[0])) + "  " + str(i[3]) + "  "
        str1 = str1 + str(self.shapes()[-1][-2])
        return main_str + str3 + "\n" + str2 + "\n" + str1 + "\n" + str2 + "\n" + str4

    def __setitem__(self, i, val):
        self.nodes[i] = val if isinstance(val, MPO_Node) else MPO_Node(val)

    def __getitem__(self, i):
        return self.nodes[i]

    def __iter__(self):
        return iter(self.nodes)

    def shapes(self):
        return [i.shape for i in self]

    def copy(self):
        """Returns a deep copy of the MPO"""
        mpo_copy = MPO([node.copy() for node in self])
        return mpo_copy

    def get_elem(self, int_index: list[int]):
        mps = self.mpo_to_mps()
        in_qix = get_local_inds(int_index[0], len(self))
        out_qix = get_local_inds(int_index[1], len(self))
        mps_qix = [2 * i + 1 * o for i, o in zip(in_qix, out_qix)]
        return mps.get_elem(mps_qix)

    def mpo_to_mps(self):
        # Convert MPO into an mps of physical dimension d^2.
        new_nodes = []
        for mpo_node in self:
            tensor = rearrange(mpo_node.tensor, "l i r o-> l (i o) r")
            new_nodes.append(MPS_Node(tensor))
        return MPS(new_nodes)

    def compress(self, accuracy: float = None, max_bd: int = None, indices=None):
        mps = self.mpo_to_mps()
        mps.compress(accuracy, max_bd, indices)
        self.nodes = mps_to_mpo(mps).nodes

    def apply(self, other: Union[MPS, "MPO"]):
        assert len(self) == len(other), "MPO and MPS should have equal no of sites"
        new_nodes = [mpo_node @ node for mpo_node, node in zip(self, other)]
        if type(other) == MPS:
            mps = MPS(new_nodes)
            mps.compress()
            return mps
        if type(other) == MPO:
            mpo = MPO(new_nodes)
            mpo.compress()
            return mpo

    def __add__(self, other: "MPO"):
        self = self.mpo_to_mps()
        mps = other.mpo_to_mps()
        return mps_to_mpo(self + mps)

    def __mul__(self, other: Union[float, int, "MPO"]):
        self = self.mpo_to_mps()
        mps = other.mpo_to_mps() if isinstance(other, MPO) else other
        return mps_to_mpo(self * mps)

    def __matmul__(self, other: Union[MPS, "MPO"]):
        warnings.warn(
            "Use 'apply()' method if you want a compressed MPS as a result. Using @ will return an uncompressed MPS."
        )
        assert len(self) == len(other), "MPO and MPS should have equal no of sites"
        new_nodes = [mpo_node @ node for mpo_node, node in zip(self, other)]
        if type(other) == MPS:
            return MPS(new_nodes)
        if type(other) == MPO:
            return MPO(new_nodes)

    def inner_prod(self, mps_i: MPS, mps_j: MPS = None):
        """Returns the inner product of mps(i and j), defined the bilinear form of the operator= (x_i,O,x_j)
        If mps_j is None then, mps_j= mps_i.copy()
        """
        if mps_j is None:
            mps_j = self.apply(mps_i)
        else:
            mps_j = self.apply(mps_j)
        return mps_i @ mps_j

    def partial_inner_prod(self, mps_i: MPS, mps_j: MPS = None, skip_indices=None):
        if mps_j is None:
            mps_j = mps_i.copy()
        if skip_indices is None:
            return self.inner_prod(mps_i, mps_j)
        else:
            pattern = einpat_part_opip(len(self), tuple(skip_indices))
            arrays = []
            for i, (n1, n2, n3) in enumerate(zip(mps_i, self, mps_j)):
                if i in skip_indices:
                    arrays += [n2.tensor]
                else:
                    arrays += [n1.tensor, n2.tensor, n3.tensor]
            matrix = oe.contract(pattern, *arrays)
            n1 = int(math.sqrt(matrix.numel()))
            matrix = matrix.reshape(n1, n1)
            return matrix

    def left_canonical(self):
        new = [node.tensor.permute([0, 3, 1, 2]) for node in self]
        d = new[0].shape[1]
        L = len(new)
        sigma = new[0].reshape(-1, new[0].shape[-1])
        A_new = []
        for i in range(1, L):
            A_1 = new[i].reshape(new[i].shape[0], -1, new[i].shape[-1])
            Q, R = qr(sigma)
            tensor = Q.reshape(-1, d, d, Q.shape[-1])
            A_new.append(MPO_Node(tensor.permute([0, 2, 3, 1])))
            sigma = torch.einsum("np,pqr -> nqr", R, A_1)
            sigma = sigma.reshape(-1, sigma.shape[-1])
        tensor = sigma.reshape(-1, d, d, 1)
        A_new.append(MPO_Node(tensor.permute([0, 2, 3, 1])))

        return MPO(A_new, "left")

    def right_canonical(self):
        new = [node.tensor.permute([0, 3, 1, 2]) for node in self]
        d = new[0].shape[1]
        L = len(new)
        sigma = new[-1].reshape(new[-1].shape[0], -1)
        A_new = []
        for i in range(1, L):
            A_1 = new[-(i + 1)].reshape(
                new[-(i + 1)].shape[0], -1, new[-(i + 1)].shape[-1]
            )
            R, Q = rq(sigma)
            tensor = Q.reshape(Q.shape[0], d, d, -1)
            A_new.append(MPO_Node(tensor.permute([0, 2, 3, 1])))
            sigma = torch.einsum("npq,qr -> npr", A_1, R)
            sigma = sigma.reshape(sigma.shape[0], -1)
        tensor = sigma.reshape(1, d, d, -1)
        A_new.append(MPO_Node(tensor.permute([0, 2, 3, 1])))

        return MPO(list(reversed(A_new)), "right")

    def contract(self):
        mps = self.mpo_to_mps()
        d = self[0].shape[1]
        return unshuffle_axis(mps.contract(), L=len(self), d=d)

    def transpose(self):
        nodes = [node.transpose() for node in self]
        return MPO(nodes)

    def conj(self):
        result = MPO([node.conj() for node in self])
        return result

    def dagger(self):
        result = self.transpose()
        return result.conj()

    def sample(self, i, j):
        d = self[0].shape[1]
        L = len(self)
        ind1 = get_local_inds(i, L, d)
        ind2 = get_local_inds(j, L, d)
        ans = torch.eye(1)
        for k in range(0, L):
            ans = ans @ self[k].tensor[:, ind1[k], :, ind2[k]]
        return ans.item()

    @property
    def norm(self):
        if self.canonical != "left":
            A1 = self.left_canonical()
        else:
            A1 = self.copy()
        k = -1
        return torch.sqrt(einsum(A1[k].tensor, A1[k].tensor, "l i r o, l i r o->"))


def list_to_mpo(A: List):
    nodes = []
    for i, tensor in enumerate(A):
        if i == 0:
            nodes.append(MPO_Node(tensor, boundary="left"))
        elif i == len(A) - 1:
            nodes.append(MPO_Node(tensor, boundary="right"))
        else:
            nodes.append(MPO_Node(tensor))
    return MPO(nodes)


def mps_mpo_mul(mps: MPS, mpo: MPO):
    assert len(mps) == len(mpo)
    einstr = ["l d r, s d t u->l s d r t u", "l s d r t u-> (l s) d (r t) u"]
    result = []
    for node1, node2 in zip(mps, mpo):
        result.append(MPO_Node(node1.contract(node2, einstr)))
    return MPO(result)


def mps_to_mpo(mps: MPS):
    """
    Convert MPS of physical dimension 4 into MPO.

    Inputs- mps: MPS
    returns: MPO
    """
    new_nodes = []
    d = mps[0].shape[1]
    assert math.sqrt(d).is_integer()
    for mpo_node in mps:
        tensor = rearrange(mpo_node.tensor, "l (i o) r-> l i r o", i=int(math.sqrt(d)))
        new_nodes.append(MPO_Node(tensor))
    return MPO(new_nodes)


def identity_mpo(L: int, bd: int = 1, pd: int = 2):
    M = torch.zeros((bd, pd, bd, pd))
    for i in range(pd):
        M[:, i, :, i] = torch.eye(bd)
    return list_to_mpo([M[:1, :, :, :]] + [M] * (L - 2) + [M[:, :, :1, :]])


def identity_mpo_node(bd: int = 1, pd: int = 2):
    M = torch.zeros((bd, pd, bd, pd))
    for i in range(pd):
        M[:, i, :, i] = torch.eye(bd)
    return MPO_Node(M)

def random_mpo(L: int, bd: int = 1, pd: int = 2, var="shape"):
    return mps_to_mpo(random_mps(L, bd, pd**2, var))


def mpo_to_torch(mpo):
    if type(mpo[0].tensor) is not torch.Tensor:
        mpo_torch = MPO(
            [MPO_Node(torch.Tensor(node.tensor.detach()).to(device())) for node in mpo]
        )
        return mpo_torch
    else:
        return mpo


def mpo_to_numpy(mpo):
    mpo_numpy = MPO([MPO_Node(node.tensor.detach().numpy()) for node in mpo])
    return mpo_numpy
