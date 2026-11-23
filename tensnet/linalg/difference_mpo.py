"""Central-difference MPOs (the solver's default difference operators).

`get_diff_mpo` builds the MPO of a first, second or fourth derivative on a uniform grid of 2**nofq
points per axis, from fixed central stencils: 8th-order (9 points) by default, 2nd-order (3 points) when
`set_param("low_acc", True)`.  The operator is unscaled (multiply by h**-order), the grid spacing and the
qubit ordering of a 2D grid (`set_arch`) are the caller's, and the closure at a wall is a ghost-value
closure.  The rows within 4 cells of a non-periodic end are therefore not accurate: the solver
(scripts/chorin_projection.py) holds or overwrites them.  For accurate walls use `diff_mpo` from
explicit_grid_operators.py; the README compares the two.

The helpers `row_mpo`, `column_mpo`, `zeros_padding` and `get_2D_operator` build selection MPOs and
embed a 1D operator into a 2D grid; `_toeplitz` builds the banded 1D operator.
"""
from typing import Union

import torch
from scipy.linalg import toeplitz

from ..networks import (
    MPO,
    MPS,
    MPO_Node,
    identity_mpo,
    identity_mpo_node,
    list_to_mpo,
    matrix_to_mpo,
    mps_mpo_col_mul,
    mps_to_mpo,
    vector_to_mps,
)
from ..utils import ARCH_2D, get_local_inds, nested_list_to_tensor, parameters
from ..utils.config import device as default_device


def _toeplitz(nofq: int, coef: dict, bc: list[str], dtype=torch.float64, device=None):
    if isinstance(bc, str):
        bc = [
            bc,
        ] * 2
    device = device if device is not None else default_device()
    A = []
    Id = torch.eye(2, 2, dtype=dtype, device=device)
    J = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    Z = torch.zeros([2, 2], dtype=dtype, device=device)

    if bc[0] == "periodic":
        Q = X
    else:
        Q = J.T
    if bc[1] == "periodic":
        P = X
    else:
        P = J

    A.append(nested_list_to_tensor([[Id, Q, P]]).permute([0, 3, 1, 2]))

    M = nested_list_to_tensor(
        [
            [Id, J.T, J],
            [Z, J, Z],
            [Z, Z, J.T],
        ]
    ).permute([0, 3, 1, 2])

    A += [M] * (nofq - 3)

    M1 = nested_list_to_tensor(
        [
            [Id, J.T, J, J, Z, Z, J.T, Z, Z],
            [Z, J, Z, Z, Id, Z, J, Id, Z],
            [Z, Z, J.T, J.T, Z, Id, Z, Z, Id],
        ]
    ).permute([0, 3, 1, 2])
    A.append(M1)

    L = nested_list_to_tensor(
        [
            [coef[0] * Id + coef[1] * J + coef[-1] * J.T],
            [coef[-1] * J + coef[-2] * Id],
            [coef[1] * J.T + coef[2] * Id],
            [coef[3] * J],
            [coef[-3] * J],
            [coef[3] * J.T],
            [coef[-3] * J.T],
            [coef[-4] * Id],
            [coef[4] * Id],
        ]
    ).permute([0, 3, 1, 2])
    A.append(L)
    return list_to_mpo(A)


def get_lcendif_coef(order):
    if order == 1:
        a1 = {}
        a1[0] = 0.0
        a1[1] = 1.0 / 2.0
        a1[2] = 0.0
        a1[3] = 0.0
        a1[4] = 0.0
        for i in range(1, 5):
            a1[-i] = -a1[i]
        return a1
    elif order == 2:
        a2 = {}
        a2[0] = -2.0
        a2[1] = 1.0
        a2[2] = 0.0
        a2[3] = 0.0
        a2[4] = 0.0
        for i in range(1, 5):
            a2[-i] = a2[i]
        return a2
    elif order == 4:
        a4 = {}
        a4[0] = 6.0
        a4[1] = -4.0
        a4[2] = 1.0
        a4[3] = 0.0
        a4[4] = 0.0
        for i in range(1, 5):
            a4[-i] = a4[i]
        return a4


def get_cendif_coef(order):
    if order == 1:
        a1 = {}
        a1[0] = 0.0
        a1[1] = 4.0 / 5.0
        a1[2] = -1.0 / 5.0
        a1[3] = 4.0 / 105.0
        a1[4] = -1.0 / 280.0
        for i in range(1, 5):
            a1[-i] = -a1[i]
        return a1
    elif order == 2:
        a2 = {}
        a2[0] = -205.0 / 72.0
        a2[1] = 8.0 / 5.0
        a2[2] = -1.0 / 5.0
        a2[3] = 8.0 / 315.0
        a2[4] = -1.0 / 560.0
        for i in range(1, 5):
            a2[-i] = a2[i]
        return a2
    elif order == 4:
        a4 = {}
        a4[0] = 91.0 / 8
        a4[1] = -122.0 / 15
        a4[2] = 169.0 / 60
        a4[3] = -2.0 / 5
        a4[4] = 7.0 / 240
        for i in range(1, 5):
            a4[-i] = a4[i]
        return a4


def column_mpo(nofq: int, index=None, device=None):
    device = device if device is not None else default_device()
    A = []

    if index is None or index == 0:
        R = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device)
        A += [nested_list_to_tensor([[R]]).permute([0, 3, 1, 2])] * nofq
        return list_to_mpo(A)
    elif index == -1:
        R = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device=device)
        A += [nested_list_to_tensor([[R]]).permute([0, 3, 1, 2])] * nofq
        return list_to_mpo(A)
    elif index > 0:
        qix = get_local_inds(index, nofq=nofq)
        L = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device)
        K = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device=device)
        for i in range(nofq):
            if qix[i] == 1:
                A.append(nested_list_to_tensor([[K]]).permute([0, 3, 1, 2]))
            elif qix[i] == 0:
                A.append(nested_list_to_tensor([[L]]).permute([0, 3, 1, 2]))

        return list_to_mpo(A)


def row_mpo(nofq: int, index=None, device=None):
    return column_mpo(nofq, index, device=device).transpose()


def zeros_padding(mpo: MPO, totlen: int, corner: str = "ul"):
    dev, dt = mpo[0].tensor.device, mpo[0].tensor.dtype
    if corner == "ur":
        tensor = [[torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=dev, dtype=dt)]]
    elif corner == "ul":
        tensor = [[torch.tensor([[1.0, 0.0], [0.0, 0.0]], device=dev, dtype=dt)]]
    elif corner == "lr":
        tensor = [[torch.tensor([[0.0, 0.0], [0.0, 1.0]], device=dev, dtype=dt)]]
    elif corner == "ll":
        tensor = [[torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=dev, dtype=dt)]]
    else:
        raise ValueError()
    nodes = [MPO_Node(nested_list_to_tensor(tensor).permute([0, 3, 1, 2]))] * (
        totlen - len(mpo)
    ) + mpo.nodes

    return MPO(nodes)


def get_2D_operator(
    P: MPO,
    axis: int,
    nofq,
    arch=None,
    Q=None,
):
    if arch is None:
        arch = ARCH_2D()

    if arch == "stacked":
        if Q is None:
            Q = identity_mpo(nofq[int(not axis)], dtype=P[0].tensor.dtype, device=P[0].tensor.device)
        if axis == 0:
            return MPO(P.nodes + Q.nodes)
        if axis == 1:
            return MPO(Q.nodes + P.nodes)
    elif arch == "sequential":
        assert abs(nofq[0] - nofq[1]) == 0, (
            "Sequential architecture currently support when nofqx ==nofqy"
        )
        if axis == 0:
            nodes = []
            for j, node in enumerate(P):
                nodes.append(node)
                if Q is None:
                    nodes.append(
                        identity_mpo_node(
                            node.shape[2], node.shape[1],
                            dtype=node.tensor.dtype, device=node.tensor.device,
                        )
                    )
                else:
                    Qtensor = Q[j].tensor
                    nodes.append(
                        MPO_Node(
                            Qtensor.expand(
                                node.shape[2], Qtensor.shape[1], node.shape[2]
                            )
                        )
                    )

            return MPO(nodes)
        if axis == 1:
            nodes = []

            for j, node in enumerate(P):
                if Q is None:
                    nodes.append(
                        identity_mpo_node(
                            node.shape[0], node.shape[1],
                            dtype=node.tensor.dtype, device=node.tensor.device,
                        )
                    )
                else:
                    Qtensor = Q[j].tensor
                    nodes.append(
                        MPO_Node(
                            Qtensor.expand(
                                node.shape[0], Qtensor.shape[1], node.shape[0]
                            )
                        )
                    )
                nodes.append(node)

            return MPO(nodes)

    elif arch == "xyrev":
        if axis == 0:
            return MPO(P.nodes + Q.nodes)
        if axis == 1:
            return MPO(Q.nodes + P.nodes[::-1])
    elif arch == "xrevy":
        if axis == 0:
            return MPO(P.nodes[::-1] + Q.nodes)
        if axis == 1:
            return MPO(Q.nodes + P.nodes)


def _add_exact(A: MPO, B: MPO) -> MPO:
    """A + B as an uncompressed direct sum. MPO.__add__ truncates to the global
    set_accuracy / set_bd_limit, and even a rank-only SVD recompression costs
    ~eps * sqrt(2**nofq) per entry, which h**-order amplifies at large nofq."""
    return mps_to_mpo(MPS([p + q for p, q in zip(A.mpo_to_mps(), B.mpo_to_mps())]))


def get_boundary_correction(order: int, nofq, bc, device=None) -> MPO:
    device = device if device is not None else default_device()
    corr_mpo = None
    a = get_cendif_coef(order)
    if parameters("low_acc"):
        a = get_lcendif_coef(order)

    rc = torch.tensor([0.0, 1.0] + [0.0] * 6, device=device)
    newcl = rc
    newcr = torch.flip(rc, dims=(0,))

    if bc[0] == "neumann":
        cfs2 = torch.tensor([a[i] for i in range(-4, 0)], device=device)
        m1 = torch.triu(torch.tensor(toeplitz(cfs2.to("cpu")), device=device))
        m2 = torch.flip(
            torch.stack([torch.roll(newcl, j) for j in range(4)]), dims=(0,)
        )
        ml = m1 @ m2
        left_coef = -torch.squeeze(ml[0, :])
        left_corr = mps_mpo_col_mul(vector_to_mps(left_coef, d=2, N=3), row_mpo(3, 0, device=device))
        if corr_mpo is None:
            corr_mpo = zeros_padding(left_corr, nofq, "ul")
        else:
            corr_mpo = corr_mpo + zeros_padding(left_corr, nofq, "ul")
        for j in range(1, 4):
            left_coef = -torch.squeeze(ml[j, :])
            left_corr = mps_mpo_col_mul(
                vector_to_mps(left_coef, d=2, N=3), row_mpo(3, j, device=device)
            )
            corr_mpo = corr_mpo + zeros_padding(left_corr, nofq, "ul")
    if bc[1] == "neumann":
        cfs2 = torch.tensor([a[i] for i in range(4, 0, -1)], device=device)
        m1 = torch.tril(torch.tensor(toeplitz(cfs2.to("cpu")), device=device))
        m2 = torch.stack([torch.roll(newcr, -j) for j in range(4)])
        mr = m1 @ m2
        mr = torch.flip(mr, dims=(0,))
        right_coef = -torch.squeeze(mr[0, :])
        right_corr = mps_mpo_col_mul(vector_to_mps(right_coef, 2, 3), row_mpo(3, 7, device=device))
        if corr_mpo is None:
            corr_mpo = zeros_padding(right_corr, nofq, "lr")
        else:
            corr_mpo = corr_mpo + zeros_padding(right_corr, nofq, "lr")
        for j in range(1, 4):
            right_coef = -torch.squeeze(mr[j, :])
            right_corr = mps_mpo_col_mul(
                vector_to_mps(right_coef, 2, 3), row_mpo(3, 7 - j, device=device)
            )
            corr_mpo = corr_mpo + zeros_padding(right_corr, nofq, "lr")
    return corr_mpo


def get_diff_mpo(
    dim: int,
    order: int,
    axis: int = None,
    nofq: Union[int, list[int]] = None,
    bc="periodic",
    arch=None,
    device=None,
):
    """Central-difference MPO of the `order`-th derivative, unscaled.

    Parameters
    ----------
    dim : {1, 2}
        Dimension of the grid.
    order : {1, 2, 4}
        Derivative order.  For `axis=-1` it is the total order of the mixed derivative (2: d2/dxdy).
    axis : int
        Axis to differentiate (2D only); -1 for the mixed derivative.
    nofq : int or list of int
        Qubits per axis (the grid has 2**nofq points); defaults to the globals Lx, Ly of `set_param`.
    bc : str or list of str
        "periodic", "dirichlet" or "neumann", one per end ordered [x-low, x-high, y-low, y-high];
        a single string applies to every end.
    arch : str, optional
        Site order of a 2D grid; defaults to the global `set_arch`.  The solver uses "sequential".
    device : optional
        Device of the MPO tensors.

    Row j of the 1D operator is sum_k a[k] f[j + k], |k| <= 4, with a the central coefficients of the
    chosen order.  Multiply the result by h**-order to get the derivative.  The stencil is 8th-order
    accurate in the interior (2nd-order with `set_param("low_acc", True)`).

    Closures, which set the ghost values f[-k] and f[n-1+k] beyond an end:

    "periodic"   wrap around; every row has the interior accuracy.  Spacing h = (b - a) / n.
    "dirichlet"  zero ghost values (the band is truncated).  The 4 rows next to the end do not converge
                 under refinement for a generic f; callers hold or overwrite them.
    "neumann"    mirrored ghost values at the end, which corrects the 4 rows next to it.  Those rows are
                 still not a consistent derivative: the error in those rows grows as h shrinks.

    scripts/chorin_projection.Operators therefore treats the 4 rows at each end as boundary rows
    (`inflow_rows=4`): it holds them at their initial values instead of trusting this stencil there.
    """
    device = device if device is not None else default_device()
    if arch is None:
        arch = ARCH_2D()

    if isinstance(bc, str):
        bc = [bc] * 2 * dim

    if isinstance(nofq, list):
        assert len(nofq) == dim

    if isinstance(nofq, int):
        nofq = [nofq] * dim

    if nofq is None:
        if dim == 1:
            nofq = [parameters("Lx")]
        if dim == 2:
            nofq = [parameters("Lx"), parameters("Ly")]

    if dim not in [1, 2]:
        raise ValueError(
            "Currently supported derivatives are only for 1d and 2d functions"
        )
    a = get_cendif_coef(order)

    if parameters("low_acc"):
        a = get_lcendif_coef(order)

    if dim == 1:
        P = _toeplitz(nofq[0], a, bc, device=device)
        if "open" in bc or "neumann" in bc:
            P = P + get_boundary_correction(order, nofq[0], bc, device=device) * -1.0
        return P
    else:
        if axis == -1:
            rec_order = order // 2
            # Returns cross derivative of same order
            P = get_diff_mpo(
                dim=2, axis=0, order=rec_order, nofq=nofq, bc=bc, arch=arch, device=device
            )
            Q = get_diff_mpo(
                dim=2, axis=1, order=rec_order, nofq=nofq, bc=bc, arch=arch, device=device
            )
            ein_str = [
                "l1 x r1 o, l2 i r2 x-> l1 l2 i r1 r2 o",
                "l1 l2 i r1 r2 o-> (l1 l2) i (r1 r2) o",
            ]
            nodes = []
            for nodex, nodey in zip(P.nodes, Q.nodes):
                nodes.append(MPO_Node(nodex.contract(nodey, einstr=ein_str)))
            return MPO(nodes)
        else:
            bcaxis = bc[axis * 2 : axis * 2 + 2]
            P = _toeplitz(nofq[axis], a, bcaxis, device=device)
            if "open" in bcaxis or "neumann" in bcaxis:
                P = P + get_boundary_correction(order, nofq[axis], bcaxis, device=device) * -1.0
            return get_2D_operator(P, axis, nofq, arch)
