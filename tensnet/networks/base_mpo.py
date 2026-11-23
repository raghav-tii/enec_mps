import math
import warnings
from typing import Union

import torch
from einops import einsum, rearrange

from ..utils import get_local_inds, unshuffle_axis, validate_nodes
from ..utils.config import device as default_device
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

    def __repr__(self) -> str:
        return (
            "Shape="
            + str(self.shape)
            + " | Boundary Cond="
            + str(self.boundary)
            + " | Canonical="
            + str(self.canonical)
        )

    def __matmul__(
        self,
        other: Union[MPS_Node, "MPO_Node"],
        einstr=["l i r o, a i b -> l a o r b", "l a o r b ->(l a) o (r b)"],
    ):
        """Applying MPO node on MPS"""
        if isinstance(other, MPS_Node):
            new_tensor = rearrange(einsum(self.tensor, other.tensor, einstr[0]), einstr[1])
            return MPS_Node(new_tensor, boundary=other.boundary)
        if isinstance(other, MPO_Node):
            einstr = ["l i r d, a d b o -> l a i r b o", "l a i r b o->(l a) i (r b) o"]
            new_tensor = rearrange(einsum(self.tensor, other.tensor, einstr[0]), einstr[1])
            return MPO_Node(new_tensor, boundary=other.boundary)

    def copy(self):
        """Returns a deep copy of the node with the same attributes."""
        return MPO_Node(self.tensor.detach().clone(), self.canonical, self.boundary)

    def contract(self, other: "MPO_Node", einstr=["l b e, e q r->l b q r", "l b q r-> l (b q) r"]):
        """Given two nodes, and einstein summation notation, contract the corresponding tensors and return the tensor.

        Args:
            other (MPO_Node): The second node to be contracted with the current node.
            einstr (list, optional): The Einstein Summation string,
                along with the rearrangement pattern for einops package.
                Defaults to ['l b e, e q r->l b q r','l b q r-> l (b q) r'].

        Returns:
            Tensor: Returns the contracted tensor (of data type that is used in the initialization of the MPS_Node.)
        """
        return rearrange(einsum(self.tensor, other.tensor, einstr[0]), einstr[1])

    def transpose(self):
        """Returns the transposed version of the node, with the two physical legs swapped."""
        einstr = "l i r o -> l o r i"
        tensor = rearrange(self.tensor.detach().clone(), einstr)
        return MPO_Node(tensor)

    def conj(self):
        """Given a complex-input node, returns the conjugate of the node; otherwise returns a copy of the input."""
        tensor = torch.conj(self.tensor)
        return MPO_Node(tensor)

    def outer(self, other: "MPO_Node"):
        """Given two nodes, returns the outer product of the two.

        Args:
            other (MPO_Node): The second MPO Node.

        Returns:
            MPO_Node: The outer product of the two nodes.
        """
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
        """Returns shape of MPO Node"""
        return self.tensor.shape

    def reshape(self, shape):
        """Reshape MPO Node given a valid shape. Is an inplace operation.

        Args:
            shape: valid shape, either as torch.Tensor, list or tuple.
        """
        self.tensor = self.tensor.reshape(shape)


class MPO:
    """
    `MPO` class to store and process a list of MPO_Nodes.

    The `MPO` class represents Matrix Product Operators (MPO), which is a powerful representation
    for representing linear operators or 2-dimensional data in a compact form.
    The class provides various methods for manipulating and analyzing MPO,
    including tensor contractions, canonical form transformations, and compression techniques.
    It also supports arithmetic operations between MPO instances,
    such as addition, subtraction, and multiplication; as well as operations with MPS instances
    such as matrix multiplication.

    Attributes:
        shapes(): Returns a list of shapes for each node in the MPO.
        copy(): Returns a deep copy of the MPO.
        get_elem(int_index): Returns a single element of the MPO based on the provided two indices as a list.
        mpo_to_mps():
    """

    def __init__(self, nodes: list[MPO_Node] = None, validate: bool = True, canonical=None):
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

    def to(self, device=None, dtype=None):
        """Returns a new MPO with every node moved to `device`/`dtype`. Leaves self untouched."""
        nodes = [
            MPO_Node(node.tensor.to(device=device, dtype=dtype), node.canonical, node.boundary)
            for node in self
        ]
        return MPO(nodes, validate=False)

    def mpo_to_mps(self):
        # Convert MPO into an mps of physical dimension d^2.
        new_nodes = []
        for mpo_node in self:
            tensor = rearrange(mpo_node.tensor, "l i r o-> l (i o) r")
            new_nodes.append(MPS_Node(tensor))
        return MPS(new_nodes)

    def compress(
        self,
        accuracy: float = None,
        max_bd: int = None,
        indices=None,
        compress_method=None,
    ):
        mps = self.mpo_to_mps()
        mps_c = mps.compress(accuracy, max_bd, indices, compress_method)

        return mps_to_mpo(mps_c)

    def apply(self, other: Union[MPS, "MPO"]):
        assert len(self) == len(other), "MPO and MPS should have equal no of sites"
        new_nodes = [mpo_node @ node for mpo_node, node in zip(self, other)]
        if isinstance(other, MPS):
            mps = MPS(new_nodes)
            mps_compressed = mps.compress()
            return mps_compressed
        if isinstance(other, MPO):
            mpo = MPO(new_nodes)
            mpo_compressed = mpo.compress()
            return mpo_compressed

    def __add__(self, other: "MPO"):
        self = self.mpo_to_mps()
        mps = other.mpo_to_mps()
        return mps_to_mpo(self + mps)

    def __sub__(self, other: "MPO"):
        self = self.mpo_to_mps()
        mps = other.mpo_to_mps()
        return mps_to_mpo(self - mps)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        self = self.mpo_to_mps()

        if isinstance(other, MPO):
            other = other.mpo_to_mps()

        return mps_to_mpo(self * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other: Union[MPS, "MPO"]):
        warnings.warn(
            "Use 'apply()' method if you want a compressed result."
            + "Using @ will return an uncompressed MPS/MPO and potentially lead to exponential memory usage.",
            ResourceWarning,
        )
        assert len(self) == len(other), "MPO and MPS should have equal no of sites"
        new_nodes = [mpo_node @ node for mpo_node, node in zip(self, other)]
        if isinstance(other, MPS):
            return MPS(new_nodes)
        if isinstance(other, MPO):
            return MPO(new_nodes)

    # def __rmatmul__(self, other):
    #     assert len(self) == len(other)
    #     if isinstance(other, MPS):
    #         einstr = ["l d r, s d t u->l s d r t u", "l s d r t u-> (l s) d (r t) u"]
    #         result = []
    #         for node1, node2 in zip(other, self):
    #             result.append(MPO_Node(node1.contract(node2, einstr)))
    #         return MPO(result)
    #     if isinstance(self, MPO):
    #         return MPO([mpo_node @ node for mpo_node, node in zip(other,self)])

    def left_canonical(self):
        mpo = mps_to_mpo(self.mpo_to_mps().left_canonical())
        mpo.canonical = "left"
        return mpo

    def right_canonical(self):
        mpo = mps_to_mpo(self.mpo_to_mps().right_canonical())
        mpo.canonical = "right"
        return mpo

    def contract(self):
        """Contract the whole MPO into a dense ``(d**L, d**L)`` matrix.

        .. warning::
            **Transpose convention.** The returned matrix is the *transpose* of
            the linear operator implemented by :meth:`apply`. That is, if ``A``
            is this MPO then for any vector ``x``::

                A.apply(x) == A.contract().T @ x        # NOT A.contract() @ x

            The two differ because :meth:`apply` contracts the MPO's
            ``output_physical`` legs against the MPS, whereas :meth:`contract`
            flattens the ``(input_physical, output_physical)`` legs into matrix
            (row, column) indices in the opposite order. The discrepancy is
            invisible for *symmetric* operators (identity, the 2nd-derivative
            Laplacian, etc.) since ``M == M.T`` there, but it flips the sign of
            *antisymmetric* operators such as the first-derivative stencil
            ``get_diff_mpo(dim, 1, ...)``.

            When building a reference dense system or checking residuals, prefer
            staying inside the MPO/MPS convention (use :meth:`apply`), or insert
            an explicit ``.T`` when going through :meth:`contract`.

        Warning:
            Materializing the dense matrix costs ``O(d**(2L))`` memory; only use
            this on small networks (testing / debugging).

        Returns:
            torch.Tensor: The dense matrix representation (transposed relative to
            :meth:`apply`; see the convention note above).
        """
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

    def normalize(self):
        mps = self.mpo_to_mps()
        norm_mps, lognorm = mps.normalize()
        return mps_to_mpo(norm_mps), lognorm

    @property
    def log_norm(self):
        mps = self.mpo_to_mps()
        return mps.log_norm

    @property
    def norm(self):
        return torch.exp(self.log_norm)

    def mean(self):
        """Returns the mean of the MPO"""
        mps = self.mpo_to_mps()
        return mps.mean()

    def sum(self):
        """Returns the sum of the MPO"""
        mps = self.mpo_to_mps()
        return mps.sum()

    def query(self, inds_x: int | list[int] | slice, inds_y: int | list[int] | slice):
        d = self[0].shape[1]
        L = len(self)
        if isinstance(inds_x, int):
            inds_x = [inds_x]
        elif isinstance(inds_x, slice):
            inds_x = list(range(d**L))[inds_x]

        if isinstance(inds_y, int):
            inds_y = [inds_y]
        elif isinstance(inds_y, slice):
            inds_y = list(range(d**L))[inds_y]

        inds = [(x, y) for x in inds_x for y in inds_y]
        bitstrings = []
        for i in range(len(inds)):
            x_ind = get_local_inds(inds[i][0], L, d)
            y_ind = get_local_inds(inds[i][1], L, d)
            bitstrings.append([d * x + y for x, y in zip(x_ind, y_ind)])

        dev = self[0].tensor.device
        queries = torch.ones(1, len(bitstrings), device=dev)
        bitstrings = torch.tensor(bitstrings, device=dev).permute(1, 0)
        for k, node in enumerate(self.mpo_to_mps().nodes):
            queries = einsum(queries, node[bitstrings[k]], "a s, a s r -> r s")

        return queries.reshape(len(inds_x), len(inds_y)).squeeze()


def list_to_mpo(A: list):
    nodes = []
    for i, tensor in enumerate(A):
        if i == 0:
            nodes.append(MPO_Node(tensor, boundary="left"))
        elif i == len(A) - 1:
            nodes.append(MPO_Node(tensor, boundary="right"))
        else:
            nodes.append(MPO_Node(tensor))
    return MPO(nodes)


def mps_mpo_col_mul(mps: MPS, mpo: MPO):
    assert len(mps) == len(mpo)
    einstr = ["l u r, s d t u->l s d r t u", "l s d r t u-> (l s) d (r t) u"]
    result = []
    for node1, node2 in zip(mps, mpo):
        result.append(MPO_Node(node1.contract(node2, einstr)))
    return MPO(result)


def mps_to_mpo(mps: MPS):
    """
    Convert MPS of physical dimension $d^2$ into MPO.
    """
    new_nodes = []
    d = mps[0].shape[1]
    assert math.sqrt(d).is_integer()
    for mpo_node in mps:
        tensor = rearrange(mpo_node.tensor, "l (i o) r-> l i r o", i=int(math.sqrt(d)))
        new_nodes.append(MPO_Node(tensor))
    return MPO(new_nodes)


def identity_mpo(L: int, bd: int = 1, pd: int = 2, dtype=torch.float64, device=None):
    device = device if device is not None else default_device()
    M = torch.zeros((bd, pd, bd, pd), dtype=dtype, device=device)
    for i in range(pd):
        M[:, i, :, i] = torch.eye(bd, dtype=dtype, device=device)
    return list_to_mpo([M[:1, :, :, :]] + [M] * (L - 2) + [M[:, :, :1, :]])


def identity_mpo_node(bd: int = 1, pd: int = 2, dtype=torch.float64, device=None):
    device = device if device is not None else default_device()
    M = torch.zeros((bd, pd, bd, pd), dtype=dtype, device=device)
    for i in range(pd):
        M[:, i, :, i] = torch.eye(bd, dtype=dtype, device=device)
    return MPO_Node(M)


def random_mpo(L: int, bd: int = 1, pd: int = 2, var="shape"):
    return mps_to_mpo(random_mps(L, bd, pd**2, var))
