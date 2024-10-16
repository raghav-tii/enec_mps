import warnings
from typing import List, Union

import numpy as np
import torch
from einops import einsum, rearrange

from ..utils import (
    BD_LIMIT,
    COMP_ACC,
    device,
    COMPRESS_METHOD,
    get_local_inds,
    qr,
    rq,
    schmidt_coeffs,
    svd,
    val_addability,
    validate_nodes,
)


class MPS_Node:
    """
    `MPS_Node()` class has the attributes that determine an arbitrary node of an MPS.

    It holds a tensor, if the tensor is in left or right canonical form and
    the boundary condition of the node.
    - Periodic boundary conditions are yet to be implemented.

    The axes of the tensor are stored in the following order:
    A tensor T(l,d,r) is represented by:
    -l--T--r- with `d` being the physical leg (dimension of the hilbert space)

    """

    # TODO Implement Periodic Boundary Conditions

    def __init__(
        self,
        tensor=torch.tensor([]),
        canonical: str = "None",
        boundary: str = None,
    ):
        self.tensor = tensor
        self.canonical = canonical
        self.boundary = boundary

    def __getitem__(self, i):
        return MPS_Node(self.tensor[:, [i], :])

    def __repr__(self) -> str:
        return "Shape=" + str(self.shape) + " | Boundary Cond=" + str(self.boundary)

    def __add__(self, other: "MPS_Node"):
        """Direct sum of two MPS Nodes. The resulting State Vectors will be added per component.

        Args:
            other (MPS_Node)

        Raises:
            ValueError: If the other node if not compatible with the current node.
            Compatibility requires the tensors to have the same order and the same physical dimension.

        Returns:
            MPS_Node: The sum is returned in the MPS_Node format.
        """

        if not val_addability(self, other):
            raise ValueError("Cannot add the given tensors")
        sumt = torch.zeros(
            self.shape[0] + other.shape[0],
            self.shape[1],
            self.shape[-1] + other.shape[-1],
        )
        for j in range(self.shape[1]):
            sumt[:, j, :] = torch.block_diag(
                self.tensor[:, j, :], other.tensor[:, j, :]
            )
        if self.boundary == "left":
            sumt = torch.sum(sumt, axis=0).reshape(1, self.shape[1], -1)
        if self.boundary == "right":
            sumt = torch.sum(sumt, axis=2).reshape(-1, self.shape[1], 1)
        return MPS_Node(sumt, boundary=self.boundary)

    def __mul__(self, other: Union[float, int, "MPS_Node"]):
        """Kronecker Product of two MPS_Nodes or Scalar multiplication of a single node.
        The resulting state vector will be either scaled by the scalar value
        or multiplied entrywise with the other vector.


        Args:
            other (Union[float, MPS_Node]): If other is of type float, then it is used a scalar.

        Raises:
            TypeError: If the type of other is not float or MPS_Node.

        Returns:
            MPS_Node: The sum is returned in the MPS_Node format.
        """
        if isinstance(other, float):
            tensor = self.tensor * other
        elif isinstance(other, int):
            tensor = self.tensor * float(other)
        elif isinstance(other, MPS_Node):
            tensor = rearrange(
                einsum(self.tensor, other.tensor, "l q r,u q v->l u q r v"),
                "l u q r v -> (l u) q (r v)",
            )
        else:
            raise TypeError(
                "Expected type Float or MPS_Node but got {} type variable for multiplication".format(
                    type(other)
                )
            )
        return MPS_Node(tensor, boundary=self.boundary)

    def contract(
        self, other: "MPS_Node", einstr=["l b e, e q r->l b q r", "l b q r-> l (b q) r"]
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

    def copy(self):
        """Returns a deep copy of the node with the same attributes."""
        return MPS_Node(self.tensor.detach().clone(), self.canonical, self.boundary)

    def average(self, axis: int = 1):
        einstr = {
            0: "l q r->q r",
            1: "l q r->l r",
            2: "l q r->l q",
        }
        tensor = einsum(self.tensor, einstr[axis]) * (1.0 / self.shape[axis])
        return MPS_Node(tensor)

    def dagger(self):
        tensor = torch.conj(self.tensor)
        return MPS_Node(tensor)

    def conj(self):
        tensor = torch.conj(self.tensor)
        return MPS_Node(tensor)

    def outer(self, other):
        tensor = self.contract(
            other,
            einstr=[
                "l1 d1 r1, l2 d2 r2 -> l1 l2 d1 d2 r1 r2",
                "l1 l2 d1 d2 r1 r2 -> (l1 l2) (d1 d2) (r1 r2)",
            ],
        )
        return MPS_Node(tensor)

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


class MPS:
    """
    `MPS()` class to store and process a list of MPS_Nodes.
    """

    def __init__(
        self, nodes: list[MPS_Node] = None, validate: bool = True, eps_local=None
    ):
        if nodes is None:
            self.nodes = []
        else:
            if validate:
                self.nodes = validate_nodes(nodes)
            else:
                self.nodes = nodes

        self.eps = eps_local

    def __len__(self):
        return self.nodes.__len__()

    def __repr__(self):
        main_str = f"{type(self)} | Length of MPS: {len(self)} | \n\n"
        str1 = ""
        str2 = ""
        str3 = ""
        for i in self.shapes():
            str1 = str1 + str(i[0]) + "--T--"
            str2 = str2 + " " * len(str(i[0])) + "  |  "
            str3 = str3 + " " * len(str(i[0])) + "  " + str(i[1]) + "  "
        str1 = str1 + str(self.shapes()[-1][-1])
        return main_str + str1 + "\n" + str2 + "\n" + str3

    def __getitem__(self, i):
        return self.nodes[i]

    def __setitem__(self, i, val):
        self.nodes[i] = val if isinstance(val, MPS_Node) else MPS_Node(val)

    def __iter__(self):
        return iter(self.nodes)

    def get_elem(self, int_index: Union[int, list[int]]):
        if isinstance(int_index, int):
            indices = get_local_inds(int_index, len(self))
        else:
            indices = int_index
        for ix, x in enumerate(self):
            qix = indices[ix]
            if ix == 0:
                temp = x.copy()[qix]
            else:
                temp = MPS_Node(temp.contract(x[qix]))
        return temp.tensor.reshape(-1).item()

    def shapes(self):
        return [i.shape for i in self]

    @property
    def tensors(self):
        return [i.tensor for i in self]

    def copy(self):
        """Returns a deep copy of the MPS"""
        mps_copy = MPS([node.copy() for node in self])
        return mps_copy

    def compress(self, accuracy: float = None, max_bd: int = None, indices=None):
        if accuracy is None:
            accuracy = self.eps
        if accuracy is None:
            accuracy = COMP_ACC()
        if max_bd is None:
            max_bd = BD_LIMIT()

        if COMPRESS_METHOD() == "off":
            return self
        elif max_bd is None or max_bd > self.max_rank or COMPRESS_METHOD() == "svd":
            self.right_canonical()
            new_nodes = []
            carry_over = None

            for ix, node in enumerate(self):
                d1 = node.shape[1]

                if ix == 0:
                    U, S, V = svd(
                        V_folding(node.tensor),
                        eps=accuracy,
                        bdmax=max_bd,
                    )
                    new_nodes.append(MPS_Node(V_unfolding(U, d1), "left"))
                elif node.boundary is None:
                    M = V_folding(einsum(carry_over, node.tensor, "l b, b d r-> l d r"))
                    U, S, V = svd(M, eps=accuracy, bdmax=max_bd)
                    new_nodes.append(MPS_Node(V_unfolding(U, d1), "left"))
                elif ix == len(self) - 1:
                    new_nodes.append(
                        MPS_Node(
                            einsum(carry_over, node.tensor, "l b, b d r-> l d r"),
                            boundary="right",
                        )
                    )
                carry_over = torch.diag(S) @ V
            self.nodes = validate_nodes(new_nodes)

            return MPS(new_nodes)

        else:
            N = len(self)
            R_tensors = []

            for node in self:
                shape = node.shape
                std = 1.0 / (shape[0] * shape[1] * shape[2])
                R_tensors.append(
                    torch.normal(0, std, size=shape, dtype=self[0].tensor.dtype)
                )

            R = list_to_mps(R_tensors)
            W_list = partial_contractions(self, R)
            X = [self[0].tensor]
            for i in range(0, N - 1):
                d1 = self[i].shape[1]

                Z_n = V_folding(X[i])
                Y_n = Z_n @ W_list[i]
                V_xn, _ = qr(Y_n)
                X[-1] = V_unfolding(V_xn, d1)
                M_n = V_xn.T @ Z_n
                H_xn = M_n @ H_folding(self[i + 1].tensor)
                X.append(H_unfolding(H_xn, d1))

            mps = list_to_mps(X, canonical="left")
            new_nodes = []
            for ix in range(N, 0, -1):
                ix = ix - 1
                node = mps[ix]
                d1 = node.shape[1]

                if ix == N - 1:
                    U, S, V = svd(
                        H_folding(node.tensor),
                        eps=accuracy,
                        bdmax=max_bd,
                    )
                    new_nodes.append(MPS_Node(H_unfolding(V, d1), "right"))
                elif node.boundary is None:
                    M = H_folding(einsum(node.tensor, carry_over, "l d b, b r->l d r"))
                    U, S, V = svd(M, eps=accuracy, bdmax=max_bd)
                    new_nodes.append(MPS_Node(H_unfolding(V, d1), "right"))
                elif ix == 0:
                    new_nodes.append(
                        MPS_Node(
                            einsum(node.tensor, carry_over, "l d b, b r->l d r"),
                            boundary="left",
                        )
                    )
                carry_over = U @ torch.diag(S)
            self.nodes = validate_nodes(new_nodes[::-1])

            return MPS(new_nodes)

    def right_canonical(self):
        carry_over = None
        if self.canonical_form != 0 or len(self) == 1:
            new_nodes = []
            for _, node in enumerate(reversed(self)):
                d1 = node.shape[1]

                if node.boundary == "right" or carry_over is None:
                    R, Q = rq(H_folding(node.tensor))
                    new_nodes.append(
                        MPS_Node(
                            H_unfolding(Q, d1),
                            canonical="right",
                            boundary="right",
                        )
                    )
                    carry_over = R
                elif node.boundary is None:
                    R, Q = rq(
                        H_folding(einsum(node.tensor, carry_over, "l d b, b r-> l d r"))
                    )
                    new_nodes.append(MPS_Node(H_unfolding(Q, d1), canonical="right"))
                    carry_over = R
                elif node.boundary == "left":
                    new_nodes.append(
                        MPS_Node(
                            einsum(node.tensor, carry_over, "l d b, b r-> l d r"),
                            boundary="left",
                        )
                    )
                    carry_over = None
            self.nodes = new_nodes[::-1]
            return carry_over

    def left_canonical(self):
        carry_over = None
        if self.canonical_form != len(self) - 1 or len(self) == 1:
            new_nodes = []
            for _, node in enumerate(self):
                d1 = node.shape[1]

                if node.boundary == "left" or carry_over is None:
                    Q, R = qr(V_folding(node.tensor))
                    new_nodes.append(
                        MPS_Node(
                            V_unfolding(Q, d1),
                            canonical="left",
                            boundary="left",
                        )
                    )
                    carry_over = R
                elif node.boundary is None:
                    Q, R = qr(
                        V_folding(einsum(carry_over, node.tensor, "l b, b d r-> l d r"))
                    )
                    new_nodes.append(MPS_Node(V_unfolding(Q, d1), canonical="left"))
                    carry_over = R
                elif node.boundary == "right":
                    new_nodes.append(
                        MPS_Node(
                            einsum(carry_over, node.tensor, "l b, b d r-> l d r"),
                            boundary="right",
                        )
                    )
                    carry_over = None
            self.nodes = new_nodes
            return carry_over

    def mixed_canonical(self, index: int = None):
        if index is None:
            raise ValueError("Index value expected of type int, but received none.")
        if self.canonical_form is None:
            self.left_canonical()

        curind = self.canonical_form

        if curind < index:
            left_nodes = self[:curind]
            transform_nodes = self[curind:index]
            tensor = self[index].tensor
            right_nodes = self[index + 1 :]
            transform_nodes = MPS(transform_nodes, False)
            carry = transform_nodes.left_canonical()
            node = MPS_Node(
                einsum(carry, tensor, "l b, b d r -> l d r"), canonical="None"
            )

            self.nodes = validate_nodes(
                left_nodes + transform_nodes.nodes + [node] + right_nodes
            )
        elif curind > index:
            left_nodes = self[:index]
            tensor = self[index].tensor
            transform_nodes = self[index + 1 : curind + 1]
            right_nodes = self[curind + 1 :]
            transform_nodes = MPS(transform_nodes, False)
            carry = transform_nodes.right_canonical()
            node = MPS_Node(
                einsum(tensor, carry, "l d b, b r -> l d r"), canonical="None"
            )
            self.nodes = validate_nodes(
                left_nodes + [node] + transform_nodes.nodes + right_nodes
            )

    def average(self, site_indices: Union[int, list] = None):
        raise NotImplementedError()
        # mps = self.copy()
        # if site_indices == -1:
        #     average = MPS([x.average() for x in mps])
        # else:
        #     average = MPS(
        #         [
        #             x.average() if ix in site_indices else x.copy()
        #             for ix, x in enumerate(mps)
        #         ]
        #     )
        # for j, node in enumerate(self):
        #     if node.order == 2:
        #         if node.boundary == "left":
        #             mps.append(
        #                 MPS_Node(
        #                     node.contract(
        #                         self[j + 1], einstr=["l e, e d r-> l d r", "... -> ..."]
        #                     )
        #                 )
        #             )
        #         else:
        #             mps[-1] = MPS_Node(
        #                 node.contract(
        #                     self[j - 1], einstr=["e r, l d e-> l d r", "... -> ..."]
        #                 )
        #             )
        #     else:
        #         mps.append(self[j])
        # for j, node in enumerate(mps):
        #     if node.shape == 2:
        #         mps.nodes.pop(j)
        # return average

    def conj(self):
        nodes = [node.conj() for node in self]
        return MPS(nodes)

    def dagger(self):
        nodes = [node.conj() for node in self]
        return MPS(nodes)

    def outer_prod(self, other):
        """
        Outer product of two MPSs.

        Use `mps_to_mpo` function for converting the resulting MPS to an MPO.

        Args:
            other (MPS)

        Returns:
            MPS: resulting MPS with physical dimesion d1, d2
        """
        nodes = [node1.outer(node2) for node1, node2 in zip(self, other)]
        mps = MPS(nodes)
        return mps

    def sample(self, i):
        d = self[0].shape[1]
        L = len(self)
        ind1 = get_local_inds(i, L, d)
        ans = torch.eye(1)
        for k in range(0, L):
            ans = ans @ self[k].tensor[:, ind1[k], :]
        return ans.item()

    def mean(self):
        mps_equal_sup = MPS(
            [MPS_Node(torch.tensor([0.5, 0.5]).reshape(1, 2, 1))] * len(self)
        )
        return self @ mps_equal_sup

    def contract(self):
        warnings.warn(
            "Careful while calling the method of contraction. This will lead to exponential memory usage",
            ResourceWarning,
        )
        for ix, x in enumerate(self):
            if ix == 0:
                temp = x.copy()
            else:
                temp = MPS_Node(temp.contract(x))
        vec = temp.tensor.reshape(-1)
        return vec

    def __add__(self, other: "MPS"):
        assert len(self) == len(
            other
        ), "Addition possible only with MPS of equal number of sites/nodes."
        sum_nodes = []
        for t1, t2 in zip(self, other):
            sum_nodes.append(t1 + t2)
        result = MPS(sum_nodes)
        result.compress()
        return result

    def __sub__(self, other: "MPS"):
        assert len(self) == len(
            other
        ), "Addition possible only with MPS of equal number of sites/nodes."
        sum_nodes = []
        for t1, t2 in zip(self, other):
            sum_nodes.append(t1 + t2 * (-1.0))
        result = MPS(sum_nodes)
        result.compress()
        return result

    def __mul__(self, other: Union[float, int, "MPS"]):
        if type(other) == MPS:
            assert len(self) == len(other), ...
            "Element-wise multiplication possible only with MPS of same shaped tensors."
            nodes = []
            for t1, t2 in zip(self, other):
                nodes.append(t1 * t2)
            result = MPS(nodes)
            result.compress()
            return result
        elif isinstance(other, (float, int)):
            prod = self.copy()
            prod[0] = prod[0] * other
            return prod

    def __matmul__(self, other: "MPS"):
        einstr = ["l d r, s d t->l s r t", "l s r t-> (l s) (r t)"]
        result = None
        for _, (node1, node2) in enumerate(zip(self, other)):
            if result is None:
                result = node1.contract(node2, einstr)
            else:
                result = einsum(result, node1.contract(node2, einstr), "a b,b c->a c")
        return result

    def inner_prod(self, other: "MPS"):
        return self @ other

    def partial_inner_prod(self):
        # TODO Use opt_einsum here.
        raise DeprecationWarning()

    def schmidt_coeffs(self):
        coeffs = []
        for ix, node in enumerate(self):
            self.mixed_canonical(ix)
            S = schmidt_coeffs(V_folding(node.tensor))
            coeffs.append(S)
        return coeffs

    def ent_entropy(self):
        entropy = []
        for ix, node in enumerate(self):
            self.mixed_canonical(ix)
            S = schmidt_coeffs(V_folding(node.tensor))
            entropy.append(-2.0 * torch.dot(S**2, torch.log2(S)).item())
        return entropy

    @property
    def ranks_list(self):
        return [node.dim[0] for node in self] + [self[-1].dim[2]]

    @property
    def max_rank(self):
        return torch.max(torch.tensor([node.dim[0] for node in self]))

    @property
    def norm(self):
        if self.canonical_form is None:
            self.left_canonical()
        node = self[self.canonical_form]
        trace = einsum(node.tensor, node.tensor, "l d r, l d r->")
        return torch.sqrt(trace)

    @property
    def canonical_form(self):
        a = np.array([node.canonical for node in self])
        ix = np.where(a == "None")[0]
        if len(ix) != 1:
            return None
        else:
            return ix.item()


# TENSOR/MPS UTILITIES


def V_folding(tensor):
    return rearrange(tensor, "l d r -> (l d) r")


def V_unfolding(tensor, pd):
    return rearrange(tensor, "(l d) r -> l d r", d=pd)


def H_folding(tensor):
    return rearrange(tensor, "l d r -> l (d r)")


def H_unfolding(tensor, pd):
    return rearrange(tensor, "l (d r) -> l d r", d=pd)


def partial_contractions(X: MPS, Y: MPS):
    assert len(X) == len(Y)
    N = len(X)
    d1 = X[N - 1].shape[1]
    W_list = [0] * (N - 1)
    W_list[N - 2] = H_folding(X[N - 1].tensor) @ H_folding(Y[N - 1].tensor).T
    for i in range(N - 2, 0, -1):
        d1 = X[i].shape[1]
        V_z = V_folding(X[i].tensor) @ W_list[i]
        T_z = V_unfolding(V_z, d1)
        W_list[i - 1] = H_folding(T_z) @ H_folding(Y[i].tensor).T
    return W_list


# MPS GENERATION TOOLS


def list_to_mps(A: List[torch.tensor], canonical="None"):
    nodes = []
    for i, tensor in enumerate(A):
        if i == 0:
            nodes.append(MPS_Node(tensor, canonical=canonical, boundary="left"))
        elif i == len(A) - 1:
            nodes.append(MPS_Node(tensor, boundary="right"))
        else:
            nodes.append(MPS_Node(tensor, canonical=canonical))
    if not torch.is_tensor(A[0]):
        return mps_to_torch(MPS(nodes))

    return MPS(nodes)


def identity_mps(L: int, bd: int = 1, pd: int = 2, requires_grad=False):
    M = torch.zeros((bd, pd, bd), requires_grad=requires_grad)
    for i in range(pd):
        M[:, i, :] = torch.eye(bd)

    nodes = [M[:1, :, :]]
    for i in range(L - 2):
        nodes.append(M)
    nodes.append(M[:, :, :1])
    return list_to_mps(nodes)


def delta(L: int, index: int, bd: int = 1, pd: int = 2):
    v1 = torch.zeros((1, pd, bd))
    v2 = torch.zeros((bd, pd, 1))

    nodes = []
    ix = get_local_inds(index, L)
    print(ix)
    v1[0, ix[0], :] = torch.ones(bd)
    v2[:, ix[-1], 0] = torch.ones(bd).T
    for _, i in enumerate(ix[1:-1]):
        M = torch.zeros((bd, pd, bd))
        M[:, i, :] = torch.eye(bd)
        nodes.append(M)
    return list_to_mps([v1] + nodes + [v2])


def random_mps(L, bd, pd=2, var="shape", requires_grad=False):
    mps = []
    diml = 1
    for i in range(L):
        i = min(i + 1, L - i - 1)
        dimr = min(pd**i, bd)
        shape = (diml, pd, dimr)
        diml = dimr

        if var == "shape":
            std = 1.0 / (shape[0] * shape[1] * shape[2])
        else:
            std = var
        mps.append(
            MPS_Node(torch.normal(0, std, size=shape, requires_grad=requires_grad))
        )
    return MPS(mps)


# MPS NUMPY/TORCH UTILS


def mps_to_torch(mps):
    if type(mps[0].tensor) is not torch.Tensor:
        mps_torch = MPS(
            [MPS_Node(torch.Tensor(node.tensor).to(device())) for node in mps]
        )
        return mps_torch
    else:
        return mps


def mps_to_numpy(mps):
    mps_numpy = MPS([MPS_Node(node.tensor.detach().numpy()) for node in mps])
    return mps_numpy
