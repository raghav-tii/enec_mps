import warnings
from typing import List, Union

import numpy as np
import torch
from einops import einsum, rearrange

from ..utils import (
    BD_LIMIT,
    COMP_ACC,
    COMPRESS_METHOD,
    NORM_PRESERVATION,
    device,
    get_local_inds,
    qr,
    rq,
    svd,
    val_addability,
    validate_nodes,
)
from ..utils.config import device as default_device


class MPS_Node:
    """
    Represents a node in a Matrix Product State (MPS).

    An MPS_Node holds a tensor and information about its canonical form and boundary conditions.
    The tensor represents the node's contribution to the overall MPS.

    Attributes:
        tensor (torch.Tensor): The tensor associated with this node. Its axes are ordered as (l, d, r),
            where l is the left virtual bond dimension, d is the physical dimension, and r is the right
            virtual bond dimension.
        canonical (str): Indicates whether the tensor is in left or right canonical form.
            Possible values: "None", "left", "right".
        boundary (str, optional): Specifies the boundary condition of the node. Defaults to None.
            Possible values: "left", "right", None. Periodic boundary conditions are not yet implemented.
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
        return self.tensor[:, i, :]

    def __repr__(self) -> str:
        return (
            "Shape="
            + str(self.shape)
            + " | Boundary Cond="
            + str(self.boundary)
            + " | Canonical="
            + str(self.canonical)
        )

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
            dtype=self.tensor.dtype,
            device=self.tensor.device,
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
        if isinstance(other, (float, int, complex, torch.Tensor)):
            tensor = self.tensor * other
            return MPS_Node(tensor, boundary=self.boundary, canonical=self.canonical)
        elif isinstance(other, MPS_Node):
            tensor = rearrange(
                einsum(self.tensor, other.tensor, "l q r,u q v->l u q r v"),
                "l u q r v -> (l u) q (r v)",
            )
            return MPS_Node(tensor, boundary=self.boundary)
        else:
            raise TypeError(
                "Expected type Scalar or MPS_Node but got {} type variable for multiplication".format(
                    type(other)
                )
            )

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
    """`MPS` class to store and process a list of MPS_Nodes.

    The `MPS` class represents a Matrix Product State (MPS), which is a powerful
    tool for representing quantum states and other high-dimensional tensors in a compact form.
    The class provides various methods for manipulating and analyzing MPS,
    including tensor contractions, canonical form transformations, and compression techniques.
    It also supports arithmetic operations between MPS instances,
    such as addition, subtraction, and multiplication.

    Attributes:
        get_elem(int_index): Returns a single element of the MPS based on the provided index.
        shapes(): Returns a list of shapes for each node in the MPS.
        requires_grad_(): Sets requires_grad to True for all tensors in the MPS.
        parameters(): Returns the tensors of the MPS nodes.
        copy(): Returns a deep copy of the MPS.
        softcopy(): Returns a soft copy of the MPS.
        compress(accuracy, max_bd, indices, compress_method): Compresses the MPS.
        right_canonical(): Converts the MPS to right canonical form.
        left_canonical(): Converts the MPS to left canonical form.
        mixed_canonical(index): Converts the MPS to mixed canonical form at index.
        average(site_indices): Averages over the specified site indices.
        conj(): Returns the complex conjugate of the MPS.
        dagger(): Returns the dagger (conjugate transpose) of the MPS.
        outer_prod(other): Computes the outer product of two MPSs.
        get_slice(qindices, pindices): Slices the MPS by selecting physical indices.
        sample(n_samples, seed): Samples the MPS (Deprecated).
        query(i): Queries the MPS at a given index.
        query_parallel(bitstrings): Queries the MPS in parallel for multiple bitstrings.
        mean(): Calculates the mean of the MPS.
        contract(): Contracts the MPS into a vector.
        rand_prod(other, max_bd): Performs a random Hadamard product of two MPSs.
        inner_prod(other): Computes the inner product of two MPSs.
        partial_inner_prod(): Computes the partial inner product (Deprecated).
        schmidt_coeffs(): Calculates the Schmidt coefficients for each bond.
        ent_entropy(): Calculates the entanglement entropy for each bond.

    """

    def __init__(
        self,
        nodes: list[MPS_Node] = None,
        validate: bool = True,
        eps_local=None,
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


    def shapes(self):
        return [i.shape for i in self]

    @property
    def tensors(self):
        return [i.tensor for i in self]

    @property
    def num_params(self):
        """Return the number of parameters in the MPS."""
        return sum([node.shape.numel() for node in self])



    def parameters(self):
        return self.tensors

    def copy(self):
        """Returns a deep copy of the MPS"""
        mps_copy = MPS([node.copy() for node in self])
        return mps_copy

    def to(self, device=None, dtype=None):
        """Returns a new MPS with every node moved to `device`/`dtype`. Leaves self untouched."""
        nodes = [
            MPS_Node(node.tensor.to(device=device, dtype=dtype), node.canonical, node.boundary)
            for node in self
        ]
        return MPS(nodes, validate=False)


    def compress(
        self,
        accuracy: float = None,
        max_bd: int = None,
        indices=None,
        compress_method=None,
    ):
        """Compresses the MPS tensor network, reducing its bond dimension.

        This method offers different compression algorithms based on the `compress_method` parameter.
        It can perform a full SVD-based compression or utilize randomized alternatives to speed up computation.

            accuracy (float, optional): Desired accuracy of the compression. Lower values result in higher compression.
            Defaults to the MPS's local epsilon value (`self.eps`) or a global compression accuracy setting.
            max_bd (int, optional): Maximum allowed bond dimension after compression. Defaults to a global bond dimension limit.
            indices (Any, optional): Indices to consider during compression (currently not used). Defaults to None.
            compress_method (str, optional): Compression method to use. Options are:
            - "off": No compression is performed.
            - "svd": SVD-based compression.
            - "randomized": Based on sketched entries of the tensors and QR decomposition.
            - "randomized_svd": A combined approach. Defaults to the global compression method setting.

            MPS: A new MPS object representing the compressed version.
        """
        if accuracy is None:
            accuracy = self.eps
        if accuracy is None:
            accuracy = COMP_ACC()
        if max_bd is None:
            max_bd = BD_LIMIT()

        if NORM_PRESERVATION():
            mps_lognorm = self.log_norm

        if compress_method is None:
            compress_method = COMPRESS_METHOD()

        if compress_method == "off":
            return self
        elif max_bd is None or max_bd > self.max_rank or compress_method == "svd":
            mps = self.right_canonical()
            new_nodes = []

            T = mps[0].tensor
            d1 = T.shape[1]

            with torch.no_grad():
                T_norm = torch.norm(T)
                lognorm = torch.log(T_norm)

            if T_norm != 0.0:
                T = T / T_norm

            U, S, V = svd(
                V_folding(T),
                eps=accuracy,
                bdmax=max_bd,
            )
            carry_over = S.type(T.dtype).unsqueeze(1) * V

            new_nodes.append(MPS_Node(V_unfolding(U, d1), "left", boundary="left"))

            for i in range(1, len(mps) - 1):
                T = einsum(carry_over, mps[i].tensor, "l b, b d r-> l d r")
                d1 = T.shape[1]

                with torch.no_grad():
                    T_norm = torch.norm(T)
                    lognorm = lognorm + torch.log(T_norm)

                if T_norm != 0.0:
                    T = T / T_norm

                U, S, V = svd(V_folding(T), eps=accuracy, bdmax=max_bd)
                carry_over = S.type(T.dtype).unsqueeze(1) * V

                new_nodes.append(MPS_Node(V_unfolding(U, d1), "left"))

            T = einsum(carry_over, mps[-1].tensor, "l b, b d r-> l d r")
            d1 = T.shape[1]

            with torch.no_grad():
                T_norm = torch.norm(T)
                lognorm = lognorm + torch.log(T_norm)

            if T_norm != 0.0:
                T = T / T_norm

            new_nodes.append(
                MPS_Node(
                    T,
                    boundary="right",
                )
            )
            dist_norm = torch.exp(lognorm / len(self)).item()

            if NORM_PRESERVATION():
                dist_norm = torch.exp(mps_lognorm / len(self)).item()

            new_nodes = [node * dist_norm for node in new_nodes]

            return MPS(new_nodes)

        elif compress_method == "randomized_kr":
            return self._compress_randomized_kr(accuracy, max_bd, mps_lognorm if NORM_PRESERVATION() else None)

        else:
            N = len(self)
            d1 = self[0].shape[1]
            R = random_mps(
                N, bd=max_bd + 10, pd=d1, var="shape", device=self[0].tensor.device
            )

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

            mps_result = list_to_mps(X, canonical="left")

            if NORM_PRESERVATION():
                mps_result, _ = mps_result.normalize()
                mps_result = mps_result.distribute_norm(mps_lognorm)

            if "svd" in compress_method:
                mps_result = mps_result.compress(accuracy, max_bd, indices, "svd")

            return mps_result

    def _compress_randomized_kr(self, accuracy, max_bd, preserve_lognorm=None):
        """Khatri-Rao-aware randomized TT rounding with adaptive rank selection.

        Exploits the TT core structure: right-environment sketches are
        propagated core-by-core (via ``partial_contractions``), then a
        left-to-right sweep projects each core onto the sketched range and
        truncates the small projected matrix via SVD with the caller's
        ``accuracy`` / ``max_bd`` thresholds.  This gives accuracy-controlled
        compression (like the deterministic SVD path) at randomized cost.

        Inspired by Al Daas, Ballard, Grigori et al. 2025 (arXiv:2511.03598).
        """
        N = len(self)
        pd = self[0].shape[1]
        dev = self[0].tensor.device
        dtype = self[0].tensor.dtype

        mps = self.right_canonical()

        cur_max_rank = int(mps.max_rank)
        sketch_bd = max(cur_max_rank, (max_bd or 0)) + 10
        R = random_mps(N, bd=sketch_bd, pd=pd, var="shape", device=dev)
        if R[0].tensor.dtype != dtype:
            for node in R:
                node.tensor = node.tensor.to(dtype)

        W_list = partial_contractions(mps, R)

        new_nodes = []
        carry = None
        lognorm = 0.0

        for i in range(N - 1):
            d_i = mps[i].shape[1]
            T = mps[i].tensor if carry is None else einsum(carry, mps[i].tensor, "l b, b d r -> l d r")

            with torch.no_grad():
                T_norm = torch.norm(T)
                lognorm = lognorm + torch.log(T_norm)
            if T_norm != 0.0:
                T = T / T_norm

            Z = V_folding(T)
            Y = Z @ W_list[i]
            Q, _ = qr(Y)

            M = Q.T.conj() @ Z

            U_m, S_m, V_m = svd(M, eps=accuracy, bdmax=max_bd)

            new_core = V_unfolding(Q @ U_m, d_i)
            new_nodes.append(MPS_Node(new_core, canonical="left",
                                      boundary="left" if i == 0 else None))
            carry = S_m.type(dtype).unsqueeze(1) * V_m

        T_last = mps[-1].tensor if carry is None else einsum(carry, mps[-1].tensor, "l b, b d r -> l d r")
        d_last = mps[-1].shape[1]
        with torch.no_grad():
            T_norm = torch.norm(T_last)
            lognorm = lognorm + torch.log(T_norm)
        if T_norm != 0.0:
            T_last = T_last / T_norm

        new_nodes.append(MPS_Node(T_last, boundary="right"))

        dist_norm = torch.exp(lognorm / N).item()
        if preserve_lognorm is not None:
            dist_norm = torch.exp(preserve_lognorm / N).item()
        new_nodes = [node * dist_norm for node in new_nodes]

        return MPS(new_nodes)

    def right_canonical(self):
        """Bring the MPS into right canonical form.

        This method transforms the MPS into its right canonical form using QR decomposition.
        The canonical form is useful for various MPS operations and algorithms.

        Returns:
            MPS: The MPS in right canonical form.
            torch.Tensor: The carry-over tensor from the canonicalization process (only if the MPS has more than one node).
            float: The log norm of the tensors (only if the MPS has more than one node).
        Note:
            - If the MPS is already in right canonical form or has only one node, it returns itself.
            - The boundary tensors are handled appropriately during the canonicalization.
        """

        if self.canonical_form != 0 or len(self) == 1:
            new_nodes = []
            lognorm = 0.0

            T = self[-1].tensor
            d1 = T.shape[1]

            with torch.no_grad():
                T_norm = torch.norm(T)
                lognorm = lognorm + torch.log(T_norm)

            if T_norm != 0.0:
                T = T / T_norm

            R, Q = rq(H_folding(T))
            new_nodes.insert(
                0,
                MPS_Node(
                    H_unfolding(Q, d1),
                    canonical="right",
                    boundary="right",
                ),
            )
            carry_over = R

            if len(self) == 1:
                return MPS(new_nodes), carry_over, lognorm

            for i in range(len(self) - 1, 0, -1):
                node = self[i - 1]
                T = einsum(node.tensor, carry_over, "l d b, b r-> l d r")
                d1 = T.shape[1]

                with torch.no_grad():
                    T_norm = torch.norm(T)
                    lognorm = lognorm + torch.log(T_norm)
                if T_norm != 0.0:
                    T = T / T_norm

                if node.boundary == "left":
                    new_nodes.insert(
                        0,
                        MPS_Node(
                            T,
                            boundary="left",
                        ),
                    )
                else:
                    R, Q = rq(H_folding(T))
                    carry_over = R

                    new_nodes.insert(
                        0,
                        MPS_Node(
                            H_unfolding(Q, d1),
                            canonical="right",
                        ),
                    )

            if node.boundary == "left":
                dist_norm = torch.exp(lognorm / len(self)).item()
                new_nodes = [node * dist_norm for node in new_nodes]

                return MPS(new_nodes)
            else:
                return MPS(new_nodes), carry_over, lognorm
        else:
            return self

    def left_canonical(self):
        """Bring the MPS into left canonical form.

        This method transforms the MPS into its left canonical form using QR decomposition.
        The canonical form is useful for various MPS operations and algorithms.

        Returns:
            MPS: The MPS in left canonical form.
            torch.Tensor: The carry-over tensor from the canonicalization process (only if the MPS has more than one node).
            float: The log norm of the tensors (only if the MPS has more than one node).
        Note:
            - If the MPS is already in left canonical form or has only one node, it returns itself.
            - The boundary tensors are handled appropriately during the canonicalization.
        """
        if self.canonical_form != len(self) - 1 or len(self) == 1:
            new_nodes = []
            lognorm = 0.0
            T = self[0].tensor
            d1 = T.shape[1]

            with torch.no_grad():
                T_norm = torch.norm(T)
                lognorm = lognorm + torch.log(T_norm)

            if T_norm != 0.0:
                T = T / T_norm

            Q, R = qr(V_folding(T))
            new_nodes.append(
                MPS_Node(
                    V_unfolding(Q, d1),
                    canonical="left",
                    boundary="left",
                )
            )

            carry_over = R

            if len(self) == 1:
                return MPS(new_nodes), carry_over, lognorm

            for i in range(1, len(self)):
                node = self[i]
                T = einsum(carry_over, node.tensor, "l b, b d r-> l d r")
                d1 = T.shape[1]

                with torch.no_grad():
                    T_norm = torch.norm(T)
                    lognorm = lognorm + torch.log(T_norm)

                if T_norm != 0.0:
                    T = T / T_norm

                if node.boundary == "right":
                    new_nodes.append(
                        MPS_Node(
                            T,
                            boundary="right",
                        )
                    )
                else:
                    Q, R = qr(V_folding(T))
                    carry_over = R

                    new_nodes.append(MPS_Node(V_unfolding(Q, d1), canonical="left"))

            if node.boundary == "right":
                dist_norm = torch.exp(lognorm / len(self)).item()
                new_nodes = [node * dist_norm for node in new_nodes]

                return MPS(new_nodes)
            else:
                return MPS(new_nodes), carry_over, lognorm
        else:
            return self



    def conj(self):
        nodes = [node.conj() for node in self]
        return MPS(nodes)

    def dagger(self):
        nodes = [node.conj() for node in self]
        return MPS(nodes)



    def sample(self, n_samples: int, seed: int = None):
        """
        Sample the MPS, assuming it to be a positive function.

        Args:
            n_samples (int): Number of samples
            seed (int, optional): Seed for the sampler for repeatability. Defaults to None.

        Returns:
            torch.tensor: Sampled indices of the MPS.
        """

        raise DeprecationWarning(
            "Please use `MPS.query()` for getting the value of tensor in the given index."
        )

    def query(self, inds: Union[int, List[int], slice]):
        """Evaluates the MPS network for an index, a list of indices or a slice.

        This method computes the output of the MPS network for multiple indices simultaneously.
        It iterates through the nodes of the MPS, performing an Einstein summation (einsum) operation
        at each node to update the query vector.

        Args:
            inds (int or list of int or slice): An index (int), a list of indices, or a slice instance (such
            as slice(0,4) == [0:4] or slice(None,10) == [:10]) to query the MPS.

        Warning:
            While using slice, be wary of size of MPS (d**L) as this can lead to large memory usage.

        Returns:
            torch.Tensor: A tensor containing the output of the MPS for each of the indices.
            The shape of the output tensor is (len(indices)).

        Note:
            - The input `inds` should be a list of ints.
            - The values in the list should be valid indices of the vector.

        """
        d = self[0].shape[1]
        L = len(self)
        if isinstance(inds, int):
            inds = [inds]
        if isinstance(inds, slice):
            inds = list(range(d**L))[inds]
        bitstrings = []
        for i in range(len(inds)):
            bitstrings.append(get_local_inds(inds[i], L, d))
        dev = self[0].tensor.device
        queries = torch.ones(1, len(bitstrings), device=dev)
        bitstrings = torch.tensor(bitstrings, device=dev).permute(1, 0)
        for k, node in enumerate(self.nodes):
            queries = einsum(queries, node[bitstrings[k]], "a s, a s r -> r s")

        return queries.flatten()

    def mean(self):
        d = self[0].shape[1]
        dev = self[0].tensor.device
        mps_equal_sup = MPS(
            [MPS_Node(torch.tensor([1 / d] * d, device=dev).reshape(1, d, 1))] * len(self)
        )
        return self @ mps_equal_sup

    def sum(self):
        d = self[0].shape[1]
        L = len(self)
        return self.mean() * (d**L)

    def contract(self):
        """Contract the entire MPS into a single vector.

        Warning:
            This method should be used with caution as it can lead to exponential memory usage,
            especially for large MPS networks. It is recommended to avoid this method if possible
            and instead use more memory-efficient contraction algorithms.

        Returns:
            torch.Tensor: A vector representing the contraction of the entire MPS.
        """
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

    def __add__(self, other: Union["MPS", float, int]):
        if isinstance(other, (float, int)):
            mps1 = ones_mps(len(self), pd=self[0].shape[1], device=self[0].tensor.device)
            mps1[0] = mps1[0] * other
            other = mps1
        assert len(self) == len(
            other
        ), "Addition possible only with MPS of equal number of sites/nodes."
        sum_nodes = []
        for t1, t2 in zip(self, other):
            sum_nodes.append(t1 + t2)
        result = MPS(sum_nodes)
        result_comp = result.compress()
        return result_comp

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: "MPS"):
        assert len(self) == len(
            other
        ), "Addition possible only with MPS of equal number of sites/nodes."
        sum_nodes = []
        other = other * -1.0
        for t1, t2 in zip(self, other):
            sum_nodes.append(t1 + t2)
        result = MPS(sum_nodes)
        result_comp = result.compress()
        return result_comp

    def __mul__(self, other):
        if isinstance(other, MPS):
            pd = self[0].shape[1]
            bd = self.max_rank
            if (
                BD_LIMIT() is None
                or COMPRESS_METHOD() == "svd"
                or (pd != 2 and bd < BD_LIMIT())
            ):
                nodes = []
                for t1, t2 in zip(self, other):
                    nodes.append(t1 * t2)
                result = MPS(nodes)
                result_comp = result.compress()
                return result_comp

            else:
                result = self.rand_prod(other, BD_LIMIT())
                if "svd" in COMPRESS_METHOD():
                    result_comp = result.compress(compress_method="svd")
                    return result_comp

                return result
        else:
            prod = MPS(self[:-1] + [self[-1] * other])
            return prod

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other: "MPS"):
        einstr = ["l d r, s d t->l s r t", "l s r t-> (l r) (s t)"]
        node1, node2 = self[0], other[0]
        result = node1.contract(node2, einstr)
        for node1, node2 in zip(self[1:], other[1:]):
            result = einsum(
                result,
                node1.tensor,
                node2.tensor,
                "a b, a d p, b d q-> p q",
            )
        return result.squeeze()

    def rand_prod(self, other, max_bd):
        mps1 = self.tensors
        mps2 = other.tensors

        new_nodes = rand_Hadamard_prod(mps1, mps2, max_bd)
        mps_result = list_to_mps(new_nodes)

        return mps_result





    @property
    def ranks_list(self):
        return [node.dim[0] for node in self] + [self[-1].dim[2]]

    @property
    def max_rank(self):
        return torch.max(torch.tensor([node.dim[0] for node in self]))

    def normalize(self):
        """Normalizes the MPS and computes the log normalization factor.

        Returns:
            tuple: A tuple containing the normalized MPS and half of the log normalization factor.
               The normalized MPS is an MPS object constructed from the normalized tensors.
               The log normalization factor is a scalar representing the logarithm of the product of the norms.
        """

        einstr = ["l d r, s d t->l s r t", "l s r t-> (l r) (s t)"]
        norm_nodes = []
        log_norm = 0.0
        node = self[0]
        result = node.contract(node.conj(), einstr)
        norm = torch.norm(result)

        if norm != 0.0:
            norm_nodes.append(node.tensor / torch.sqrt(norm))
            result = result / norm
        else:
            norm_nodes.append(node.tensor)
        log_norm = log_norm + torch.log(norm)

        for node in self[1:]:
            result = einsum(
                result, node.tensor, node.tensor.conj(), "a b, a d p, b d q-> p q"
            )

            norm = torch.norm(result)
            if norm != 0.0:
                norm_nodes.append(node.tensor / torch.sqrt(norm))
                result = result / norm
            else:
                norm_nodes.append(node.tensor)
            log_norm = log_norm + torch.log(norm)

        return list_to_mps(norm_nodes), 0.5 * log_norm

    def distribute_norm(self, lognorm):
        """
        Distributes the given lognorm with the nodes of the MPS.

        Args:
            lognorm (torch.tensor): Log norm that is to be distributed over the current MPS nodes.

        Returns:
            MPS: MPS with the distributed norm.
        """
        dist_norm = torch.exp(lognorm / len(self))
        new_nodes = [node * dist_norm for node in self]

        return MPS(new_nodes)

    @property
    def log_norm(self):
        """
        Computes the log norm of the MPS using inner product.

        Returns:
            lognorm (torch.tensor): Log norm of the MPS.
        """
        einstr = ["l d r, s d t->l s r t", "l s r t-> (l r) (s t)"]
        log_norm = 0.0
        node = self[0]
        result = node.contract(node.conj(), einstr)
        norm = torch.norm(result)

        if norm != 0.0:
            result = result / norm
        log_norm = log_norm + torch.log(norm)

        for node in self[1:]:
            result = einsum(
                result, node.tensor, node.tensor.conj(), "a b,a d p, b d q->p q"
            )

            norm = torch.norm(result)
            if norm != 0.0:
                result = result / norm
            log_norm = log_norm + torch.log(norm)

        return 0.5 * log_norm

    @property
    def norm(self):
        """
        Computes the norm of the MPS.

        Returns:
            norm (torch.tensor): Norm of the MPS.
        """
        return torch.exp(self.log_norm)

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




def identity_mps(L: int, bd: int = 1, pd: int = 2, dtype=torch.float64, device=None):
    device = device if device is not None else default_device()
    M = torch.zeros((bd, pd, bd), dtype=dtype, device=device)
    for i in range(pd):
        M[:, i, :] = torch.eye(bd, dtype=dtype, device=device)

    nodes = [M[:1, :, :]]
    for i in range(L - 2):
        nodes.append(M)
    nodes.append(M[:, :, :1])

    return list_to_mps(nodes)


def ones_mps(L: int, pd: int = 2, dtype=torch.float64, device=None):
    device = device if device is not None else default_device()
    A = random_mps(L, 1, pd, var=1.0, dtype=dtype, device=device)
    A = list_to_mps([torch.ones_like(i.tensor, dtype=dtype, device=device) for i in A])
    return A




def random_mps(
    L,
    bd,
    pd=2,
    var="shape",
    requires_grad=False,
    dtype=torch.float64,
    device=None,
):
    device = device if device is not None else default_device()
    mps = []
    diml = 1

    for i in range(L):
        i = min(i + 1, L - i - 1)
        dimr = min(pd**i, bd)
        shape = (diml, pd, dimr)
        diml = dimr

        if var == "shape":
            std = 1.0 / (shape[0])**0.5
        else:
            std = var
        mps.append(
            MPS_Node(
                torch.normal(
                    0,
                    std,
                    size=shape,
                    requires_grad=requires_grad,
                    dtype=dtype,
                    device=device,
                )
            )
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




# MPS RANDOMIZED COMPRESSION UTILS
def envs_Hprod(mps1, mps2, mps_rand):
    W = {}
    dim = len(mps1)
    W[dim - 1] = torch.einsum(
        "in,jn,kn->ijk",
        mps1[dim - 1][:, :, 0],
        mps2[dim - 1][:, :, 0],
        mps_rand[dim - 1][:, :, 0],
    )
    sh = W[dim - 1].shape
    W[dim - 1] = W[dim - 1].reshape(sh[0] * sh[1], sh[2])
    for i in range(dim - 2, 0, -1):
        U, S, Vh = torch.linalg.svd(W[i + 1], full_matrices=False)
        U = U.reshape(mps1[i].shape[2], mps2[i].shape[2], U.shape[1])
        WL = torch.einsum("inj,jks,onk->ions", mps1[i], U, mps2[i])
        WR = torch.einsum("inj,sj->ins", mps_rand[i], Vh)
        WL = WL.reshape(WL.shape[0] * WL.shape[1], WL.shape[2], WL.shape[3])
        W[i] = torch.einsum("s,ins,jns->ij", S, WL, WR)
    return W


def compression_prod_randomized(mps1, mps2, envs):
    mps0 = torch.einsum("inj,kno->iknjo", mps1[0], mps2[0])
    sh = mps0.shape
    mps0 = mps0.reshape(sh[0] * sh[1], sh[2], sh[3] * sh[4])
    mps_new = [mps0]
    for i in range(len(mps1) - 1):
        Z = V_folding(mps_new[i])
        Y = Z @ envs[i + 1]
        Q, _ = torch.linalg.qr(Y)
        M = Q.T @ Z
        mps_new[i] = Q.reshape(Q.shape[0] // 2, 2, Q.shape[1])
        M = M.reshape(M.shape[0], mps1[i + 1].shape[0], mps2[i + 1].shape[0])
        Tnew = torch.einsum("inj,gik,knl->gnjl", mps1[i + 1], M, mps2[i + 1])
        sh = Tnew.shape
        mps_new.append(Tnew.reshape(sh[0], sh[1], sh[2] * sh[3]))
    return mps_new


def rand_Hadamard_prod(mps1, mps2, bond_final):
    dim = len(mps1)
    mps_rand = random_mps(
        dim, bond_final, pd=mps1[0].shape[1], var=1.0, device=mps1[0].device
    )
    envs = envs_Hprod(mps1, mps2, mps_rand.tensors)
    return compression_prod_randomized(mps1, mps2, envs)
