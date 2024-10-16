from typing import Union

import torch
from scipy.linalg import toeplitz

from ..networks import (
    MPO,
    MPO_Node,
    identity_mpo,
    identity_mpo_node,
    list_to_mpo,
    mps_mpo_mul,
    vector_to_mps,
)
from ..utils import ARCH_2D, get_local_inds, nested_list_to_tensor, parameters


def triangular_mpo(L: int, flag: str = "upper"):
    A = []
    Id = torch.eye(2, 2)
    if flag == "upper":
        J = torch.tensor([[0, 1], [0, 0]])
    elif flag == "lower":
        J = torch.tensor([[0, 1], [0, 0]]).T

    Ones = torch.ones([2, 2])
    Z = torch.zeros([2, 2])

    A.append(torch.tensor([[Id, J]]).permute([0, 3, 1, 2]))
    M = torch.tensor([[Id, J], [Z, Ones]]).permute([0, 3, 1, 2])
    A += [M] * (L - 2)
    A.append(torch.tensor([[Id + J], [Ones]]).permute([0, 3, 1, 2]))

    return list_to_mpo(A)


def shift_mpo(L: int, flag: str = "up"):
    A = []
    Id = torch.eye(2, 2)
    if flag == "up":
        J = torch.tensor([[0, 1], [0, 0]])
    elif flag == "down":
        J = torch.tensor([[0, 1], [0, 0]]).T

    Z = torch.zeros([2, 2])

    A.append(torch.tensor([[Id, J]]).permute([0, 3, 1, 2]))
    M = torch.tensor([[Id, J], [Z, J.T]]).permute([0, 3, 1, 2])
    A += [M] * (L - 2)
    A.append(torch.tensor([[J], [J.T]]).permute([0, 3, 1, 2]))

    return list_to_mpo(A)


def _construct_toeplitz(nofq: int, coef: dict, bc: list[str]):
    a = coef.copy()
    A = []
    Id = torch.eye(2, 2)
    J = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    X = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    Z = torch.zeros([2, 2])
    if bc[0] == "periodic":
        Q = X
    else:
        Q = J.T
    if bc[1] == "periodic":
        P = X
    else:
        P = J

    A.append(nested_list_to_tensor([[Id, Q, P]]).permute([0, 3, 1, 2]))

    M = nested_list_to_tensor([[Id, J.T, J], [Z, J, Z], [Z, Z, J.T]]).permute(
        [0, 3, 1, 2]
    )

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
            [a[0] * Id + a[1] * J + a[-1] * J.T],
            [a[-1] * J + a[-2] * Id],
            [a[1] * J.T + a[2] * Id],
            [a[3] * J],
            [a[-3] * J],
            [a[3] * J.T],
            [a[-3] * J.T],
            [a[-4] * Id],
            [a[4] * Id],
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


def _get_fwddif_coef(order):
    if order == 1:
        a1 = [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0, 0.0, 0.0, 0.0]
        return a1
    elif order == 2:
        a2 = [
            35.0 / 12.0,
            -26.0 / 3.0,
            19.0 / 2.0,
            -14.0 / 3.0,
            11.0 / 12.0,
            0.0,
            0.0,
            0.0,
        ]
        return a2
    elif order == 4:
        a4 = [1.0, -4.0, 6.0, -4.0, 1.0, 0.0, 0.0, 0.0]
        return a4


def column_mpo(nofq: int, index=None):
    A = []

    if index is None or index == 0:
        R = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        A += [nested_list_to_tensor([[R]]).permute([0, 3, 1, 2])] * nofq
        return list_to_mpo(A)
    elif index == -1:
        R = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        A += [nested_list_to_tensor([[R]]).permute([0, 3, 1, 2])] * nofq
        return list_to_mpo(A)
    elif index > 0:
        qix = get_local_inds(index, nofq=nofq)
        L = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        K = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        for i in range(nofq):
            if qix[i] == 1:
                A.append(nested_list_to_tensor([[K]]).permute([0, 3, 1, 2]))
            elif qix[i] == 0:
                A.append(nested_list_to_tensor([[L]]).permute([0, 3, 1, 2]))

        return list_to_mpo(A)


def corner_mpo(cfs, flag: str = "upper"):
    A = []
    Id = torch.eye(2, 2)
    if flag == "upper":
        J = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    elif flag == "lower":
        J = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    A.append(nested_list_to_tensor([[Id, J]]).permute([0, 3, 1, 2]))
    A.append(
        nested_list_to_tensor(
            [[cfs[0] * Id + cfs[1] * J], [cfs[2] * Id + cfs[1] * J.T + cfs[3] * J]]
        ).permute([0, 3, 1, 2])
    )

    return list_to_mpo(A)


def row_mpo(nofq: int, index=None):
    return column_mpo(nofq, index).transpose()


def zeros_padding(mpo: MPO, totlen: int, corner: str = "ul"):
    if corner == "ur":
        tensor = [[torch.tensor([[0.0, 1.0], [0.0, 0.0]])]]
    elif corner == "ul":
        tensor = [[torch.tensor([[1.0, 0.0], [0.0, 0.0]])]]
    elif corner == "lr":
        tensor = [[torch.tensor([[0.0, 0.0], [0.0, 1.0]])]]
    elif corner == "ll":
        tensor = [[torch.tensor([[0.0, 0.0], [1.0, 0.0]])]]
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
):
    if arch is None:
        arch = ARCH_2D()
    Q = identity_mpo(nofq[int(not axis)])
    if arch == "stacked":
        if axis == 0:
            return MPO(P.nodes + Q.nodes)
        if axis == 1:
            return MPO(Q.nodes + P.nodes)
    elif arch == "sequential":
        if axis == 0:
            nodes = []
            for node in P:
                nodes.append(node)
                nodes.append(identity_mpo_node(node.shape[2],node.shape[1]))
            return MPO(nodes)
        if axis == 1:
            nodes = []
            for node in P:
                nodes.append(identity_mpo_node(node.shape[0],node.shape[1]))
                nodes.append(node)
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

    return MPO(P.nodes + Q.nodes)


def get_dirichlet_correction(order: int, nofq, axis=None, arch=None):
    """Provide axis data if the operator is acting on 2D field."""
    if isinstance(nofq, list):
        nofq1 = nofq[axis]
    else:
        nofq1 = nofq
    if arch is None:
        arch = ARCH_2D()

    a = get_cendif_coef(order)
    if parameters("low_acc"):
        a = get_lcendif_coef(order)

    # Dirichlet Correction Operators
    m1 = torch.tensor([a[i] for i in range(-4, 0)])
    m2 = torch.tensor([a[i] for i in range(4, 0, -1)])
    lcor = corner_mpo(m1, "upper")
    rcor = corner_mpo(m2, "lower")

    lcor = zeros_padding(lcor, nofq1, "ul")
    rcor = zeros_padding(rcor, nofq1, "lr")

    if isinstance(nofq, int):
        return lcor, rcor

    if len(nofq) == 2:
        lcor = get_2D_operator(lcor, axis, nofq, arch)
        rcor = get_2D_operator(rcor, axis, nofq, arch)
        return lcor, rcor


def get_neumann_correction(order: int, nofq, axis=None, arch=None):
    if arch is None:
        arch = ARCH_2D()
    a = get_cendif_coef(order)
    if parameters("low_acc"):
        a = get_lcendif_coef(order)

    if isinstance(nofq, list):
        nofq1 = nofq[axis]
    else:
        nofq1 = nofq

    lcor = identity_mpo(nofq1) * 0.0
    rcor = identity_mpo(nofq1) * 0.0

    m1 = torch.tensor([a[i] for i in range(-4, 0)])
    m2 = torch.tensor([a[i] for i in range(4, 0, -1)])
    lcor = corner_mpo(m1, "upper")
    rcor = corner_mpo(m2, "lower")

    lcor = zeros_padding(lcor, nofq1, "ul")
    rcor = zeros_padding(rcor, nofq1, "lr")

    if isinstance(nofq, int):
        return lcor, rcor
    if len(nofq) == 2:
        lcor = get_2D_operator(lcor, axis, nofq, arch)
        rcor = get_2D_operator(rcor, axis, nofq, arch)
        return lcor, rcor


def get_boundary_correction(order: int, nofq, bc) -> MPO:
    corr_mpo = None
    a = get_cendif_coef(order)
    if parameters("low_acc"):
        a = get_lcendif_coef(order)

    rc = torch.tensor([0.0, 1.0] + [0.0] * 6)
    newcl = rc
    newcr = torch.flip(rc, dims=(0,))

    if bc[0] == "neumann":
        cfs2 = torch.tensor([a[i] for i in range(-4, 0)])
        m1 = torch.triu(torch.tensor(toeplitz(cfs2.to("cpu"))))
        m2 = torch.flip(
            torch.stack([torch.roll(newcl, j) for j in range(4)]), dims=(0,)
        )
        ml = m1 @ m2
        left_coef = -torch.squeeze(ml[0, :])
        left_corr = mps_mpo_mul(vector_to_mps(left_coef, d=2, N=3), row_mpo(3, 0))
        if corr_mpo is None:
            corr_mpo = zeros_padding(left_corr, nofq, "ul")
        else:
            corr_mpo = corr_mpo + zeros_padding(left_corr, nofq, "ul")
        for j in range(1, 4):
            left_coef = -torch.squeeze(ml[j, :])
            left_corr = mps_mpo_mul(vector_to_mps(left_coef, d=2, N=3), row_mpo(3, j))
            corr_mpo = corr_mpo + zeros_padding(left_corr, nofq, "ul")
    if bc[1] == "neumann":
        cfs2 = torch.tensor([a[i] for i in range(4, 0, -1)])
        m1 = torch.tril(torch.tensor(toeplitz(cfs2.to("cpu"))))
        m2 = torch.stack([torch.roll(newcr, -j) for j in range(4)])
        mr = m1 @ m2
        mr = torch.flip(mr, dims=(0,))
        right_coef = -torch.squeeze(mr[0, :])
        right_corr = mps_mpo_mul(vector_to_mps(right_coef, 2, 3), row_mpo(3, 7))
        if corr_mpo is None:
            corr_mpo = zeros_padding(right_corr, nofq, "lr")
        else:
            corr_mpo = corr_mpo + zeros_padding(right_corr, nofq, "lr")
        for j in range(1, 4):
            right_coef = -torch.squeeze(mr[j, :])
            right_corr = mps_mpo_mul(vector_to_mps(right_coef, 2, 3), row_mpo(3, 7 - j))
            corr_mpo = corr_mpo + zeros_padding(right_corr, nofq, "lr")
    return corr_mpo


def get_diff_mpo(
    dim: int,
    order: int,
    axis: int = None,
    nofq: Union[int, list[int]] = None,
    bc="periodic",
    arch=None,
):
    """If axis is -1, then cross-derivatives of the same order as `order` are returned."""
    if arch is None:
        arch = ARCH_2D()

    if isinstance(bc, str):
        bc = [bc] * 2 * dim

    if hasattr(nofq, "__len__"):
        assert len(nofq) == dim
    elif nofq is not None:
        nofq = [nofq] * dim
    elif nofq is None:
        nofq = [parameters("Lx"), parameters("Ly")]

    if dim not in [1, 2]:
        raise ValueError(
            "Currently supported derivatives are only for 1d and 2d functions"
        )
    a = get_cendif_coef(order)

    if parameters("low_acc"):
        a = get_lcendif_coef(order)

    if dim == 1:
        P = _construct_toeplitz(nofq[0], a, bc)
        if "open" in bc or "neumann" in bc:
            P = P + get_boundary_correction(order, nofq[0], bc) * -1.0
        return P
    else:
        if axis == -1:
            rec_order = order // 2
            # Returns cross derivative of same order
            P = get_diff_mpo(
                dim=2, axis=0, order=rec_order, nofq=nofq, bc=bc, arch=arch
            )
            Q = get_diff_mpo(
                dim=2, axis=1, order=rec_order, nofq=nofq, bc=bc, arch=arch
            )
            ein_str = [
                "l1 x r1 o, l2 i r2 x-> l1 l2 i r1 r2 o",
                "l1 l2 i r1 r2 o-> (l1 l2) i (r1 r2) o",
            ]
            nodes = []
            for nodex, nodey in zip(P.nodes, Q.nodes):
                nodes.append(MPO_Node(nodex.contract(nodey, einstr=ein_str)))
            return MPO(nodes)

        assert axis in [
            0,
            1,
        ], "Axis chosen is not compatible with the dimension of the function"
        bcaxis = bc[axis * 2 : axis * 2 + 2]
        P = _construct_toeplitz(nofq[axis], a, bcaxis)
        if "open" in bcaxis or "neumann" in bcaxis:
            P = P + get_boundary_correction(order, nofq[axis], bcaxis) * -1.0
        return get_2D_operator(P, axis, nofq, arch)
