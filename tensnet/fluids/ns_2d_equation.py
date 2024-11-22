import torch
from tqdm import tqdm
import time

from ..linalg import (
    get_2D_operator,
    get_diff_mpo,
    get_dirichlet_correction,
    linsystem_solver,
    row_mpo,
)
from ..networks import MPO, MPS, MPO_Node, MPS_Node, identity_mpo, identity_mps
from ..utils import parameters, save_object


def construct_A(nofq: list[int], dt: float, dx: float, dy: float, mu: float = 1e3):
    if isinstance(nofq, int):
        nofq = [nofq, nofq]
    k1 = -1 * mu * (dt**2)

    Ix = identity_mpo(L=nofq[0])
    Iy = identity_mpo(L=nofq[1])
    Dxx = get_diff_mpo(dim=2, order=2, axis=0) * (1.0 / dx**2)
    Dyy = get_diff_mpo(dim=2, order=2, axis=1) * (1.0 / dy**2)
    Dxy = get_diff_mpo(dim=2, order=2, axis=-1) * (1.0 / (dx * dy))

    t1 = torch.zeros((1, 2, 1, 2))
    t1[:, 0, :, 0] = 1
    t2 = torch.zeros((1, 2, 1, 2))
    t2[:, 0, :, 1] = 1
    t3 = torch.zeros((1, 2, 1, 2))
    t3[:, 1, :, 0] = 1
    t4 = torch.zeros((1, 2, 1, 2))
    t4[:, 1, :, 1] = 1

    q1 = MPO_Node(t1)
    q2 = MPO_Node(t2)
    q3 = MPO_Node(t3)
    q4 = MPO_Node(t4)

    I_00 = MPO([q1] + Ix.nodes + Iy.nodes)
    I_11 = MPO([q4] + Ix.nodes + Iy.nodes)
    D1 = MPO([q1] + Dxx.nodes)
    D4 = MPO([q4] + Dyy.nodes)
    D2 = MPO([q2] + Dxy.nodes)
    D3 = MPO([q3] + Dxy.nodes)

    mpo = I_00 + I_11 + (D1 + D2 + D3 + D4) * k1

    return mpo


def vel_predictor(
    ux: MPS,
    uy: MPS,
    nofq: list[int],
    nu: float,
    dt: float,
    Dx: MPO,
    Dy: MPO,
    Dxx: MPO,
    Dyy: MPO,
    uxp=None,
    uyp=None,
    bc="periodic",
    ret_comps=False,
):
    if isinstance(nofq, int):
        nofq = [nofq, nofq]

    if uxp is None:
        uxp = ux.copy()
    if uyp is None:
        uyp = uy.copy()

    t1 = torch.zeros((1, 2, 1))
    t1[:, 0, :] = 1
    t2 = torch.zeros((1, 2, 1))
    t2[:, 1, :] = 1

    q1 = MPS_Node(t1)
    q2 = MPS_Node(t2)

    k1 = -dt
    k2 = nu * dt

    bx = (
        ux * Dx.apply(ux) * k1
        + uy * Dy.apply(ux) * k1
        + Dxx.apply(ux) * k2
        + Dyy.apply(ux) * k2
    )

    if bc[0] == "dirichlet":
        Lx = nofq[0]
        delrow = row_mpo(Lx, 0) + row_mpo(Lx, 1) + row_mpo(Lx, 2) + row_mpo(Lx, 3)
        delrow = identity_mpo(Lx) + delrow * identity_mpo(Lx) * -1.0
        delrow = get_2D_operator(delrow, 0, nofq)
        bx = delrow.apply(bx)

    bx = bx + uxp

    by = (
        ux * Dx.apply(uy) * k1
        + uy * Dy.apply(uy) * k1
        + Dxx.apply(uy) * k2
        + Dyy.apply(uy) * k2
        + uyp
    )
    if ret_comps:
        return bx, by

    b = MPS([q1] + bx.nodes) + MPS([q2] + by.nodes)
    b.compress()

    return b


def get_2d_divergence(mpsx, mpsy, Dx, Dy, nofq=None, bc=None):
    delrow = row_mpo(nofq[0], 0)
    if not parameters("low_acc"):
        delrow = (
            delrow + row_mpo(nofq[0], 1) + row_mpo(nofq[0], 2) + row_mpo(nofq[0], 3)
        )
    delrow = identity_mpo(nofq[0]) + delrow * identity_mpo(nofq[0]) * -1.0
    delrow = get_2D_operator(delrow, 0, nofq)
    b = Dx.apply(mpsx) + Dy.apply(mpsy)
    if bc[0] == "dirichlet":
        b = delrow.apply(b)

    return b


def get_2d_vec_comp(mps: MPS, index: int = None):
    if index is None:
        return (get_2d_vec_comp(mps, 0), get_2d_vec_comp(mps, 1))
    temp = mps[0][index]
    new_node = MPS_Node(temp.contract(mps[1]))
    return MPS([new_node] + mps.nodes[2:])


def euler_time_stepping(
    ux: MPS,
    uy: MPS,
    nofq: list[int],
    dt: float,
    N: int,
    nu,
    dx: float = None,
    dy: float = None,
    mask: MPS = None,
    opt_method="dmrg2",
    opt_accuracy=1e-6,
    opt_iter=6,
    opt_paths_iter=100,
    bdtrunc=None,
    saveit=None,
    UID=None,
    verbose=False,
    ret_times=False,
    inv_mpo=None,
):
    if bdtrunc:
        bdtruncp = bdtrunc
    else:
        bdtruncp = int(30)

    if isinstance(nofq, int):
        nofq = [nofq, nofq]

    if dx is None:
        dx = parameters("bx") - parameters("ax")
        dx /= (2 ** nofq[0]) - 1
    if dy is None:
        dy = parameters("by") - parameters("ay")
        dy /= (2 ** nofq[1]) - 1

    phiout = identity_mps(sum(nofq)) * 1.0 * dt
    phi_temp = None

    bcv = ["dirichlet", "neumann", "periodic", "periodic"]
    bcp = ["neumann", "dirichlet", "periodic", "periodic"]

    Dxp = get_diff_mpo(dim=2, order=1, axis=0, nofq=nofq, bc=bcp) * (1.0 / dx)
    Dyp = get_diff_mpo(dim=2, order=1, axis=1, nofq=nofq, bc=bcp) * (1.0 / dy)
    Dxx = get_diff_mpo(dim=2, order=2, axis=0, nofq=nofq, bc=bcp) * (1.0 / dx**2)
    Dyy = get_diff_mpo(dim=2, order=2, axis=1, nofq=nofq, bc=bcp) * (1.0 / dy**2)

    Dx1 = get_diff_mpo(dim=2, order=1, axis=0, bc=bcv) * (1.0 / dx)
    Dy1 = get_diff_mpo(dim=2, order=1, axis=1, bc=bcv) * (1.0 / dy)
    Dxx1 = get_diff_mpo(dim=2, order=2, axis=0, bc=bcv) * (1.0 / dx**2)
    Dyy1 = get_diff_mpo(dim=2, order=2, axis=1, bc=bcv) * (1.0 / dy**2)

    _, rbop2 = get_dirichlet_correction(2, nofq=nofq, axis=0)
    rbop2 = rbop2 * (1.0 / dx**2)
    _, rbop1 = get_dirichlet_correction(1, nofq=nofq, axis=0)
    rbop1 = rbop1 * (1.0 / dx**1)

    A = Dxx + Dyy

    if mask is None:
        mask = identity_mps(L=len(ux))
    solx = [ux]
    soly = [uy]
    solp = []
    b = get_2d_divergence(solx[-1], soly[-1], Dx1, Dy1, nofq, bcv)
    if bcp[1] == "dirichlet":
        b = b + rbop2.apply(phiout) * -1.0

    error = []

    if "dmrg" in opt_method:
        dmrg_sites = int(opt_method[-1])

    times = []
    for i in tqdm(range(N)):
        _opt_flag = False
        if i % opt_paths_iter == 0:
            _opt_flag = True

        start = time.time()

        if "dmrg" in opt_method:
            phi = linsystem_solver(
                A=A,
                b=b,
                x=phi_temp,
                max_iter=opt_iter,
                max_rank=bdtruncp,
                eps=opt_accuracy,
                dmrg_sites=dmrg_sites,
                opt_paths=_opt_flag,
                verbose=verbose,
            )
            if verbose:
                error.append(phi[1])
                phi = phi[0]

        vxp = solx[-1] + Dxp.apply(phi) * (-1.0)

        if bcp[1] == "dirichlet":
            vxp = vxp + rbop1.apply(phiout) * -1.0
        vxp = vxp * mask

        vyp = (Dyp.apply(phi) * (-1.0) + soly[-1]) * mask

        vxp.compress(max_bd=bdtrunc)
        vyp.compress(max_bd=bdtrunc)

        solx[-1] = vxp.copy()
        soly[-1] = vyp.copy()

        vx, vy = vel_predictor(
            solx[-1],
            soly[-1],
            nofq,
            nu,
            dt,
            Dx1,
            Dy1,
            Dxx1,
            Dyy1,
            bc=bcv,
            ret_comps=True,
        )

        solp.append(phi * (1.0 / dt))
        solx.append(vx)
        soly.append(vy)

        stop = time.time()

        times.append(stop - start)

        if bcp[1] == "dirichlet":
            b = b + rbop2.apply(phiout) * -1.0

        if saveit is not None and i % saveit == 0:
            save_object(
                (solx[-saveit:], soly[-saveit:], solp[-saveit:]),
                f"../results/temp/{UID}_int_sol_{i}.tmp",
            )
    results = (solx, soly, solp)

    if ret_times:
        results += (times,)

    if verbose:
        return results + (error,)

    return results


def time_stepping_multiiter(
    ux: MPS,
    uy: MPS,
    nofq: list[int],
    dt: float,
    N: int,
    Niter: int,
    nu,
    dx: float = None,
    dy: float = None,
    mask: MPS = None,
    opt_method="dmrg2",
    opt_accuracy=1e-6,
    opt_iter=6,
    opt_flag=False,
    scheme="second",
    saveit=None,
    UID=None,
    verbose=False,
):
    assert scheme in (
        "first",
        "second",
    ), " Only first and second order RK schemes are available now."
    if isinstance(nofq, int):
        nofq = [nofq, nofq]
    if dx is None:
        dx = parameters("bx") - parameters("ax")
        dx /= 2 ** nofq[0]
    if dy is None:
        dy = parameters("by") - parameters("ay")
        dy /= 2 ** nofq[1]
    Dx = get_diff_mpo(dim=2, order=1, axis=0) * (1.0 / dx)
    Dy = get_diff_mpo(dim=2, order=1, axis=1) * (1.0 / dy)
    Dxx = get_diff_mpo(dim=2, order=2, axis=0) * (1.0 / dx**2)
    Dyy = get_diff_mpo(dim=2, order=2, axis=1) * (1.0 / dy**2)

    Dx1 = get_diff_mpo(dim=2, order=1, axis=0) * (1.0 / dx)
    Dy1 = get_diff_mpo(dim=2, order=1, axis=1) * (1.0 / dy)
    Dxx1 = get_diff_mpo(dim=2, order=2, axis=0) * (1.0 / dx**2)
    Dyy1 = get_diff_mpo(dim=2, order=2, axis=1) * (1.0 / dy**2)
    A = Dxx + Dyy
    if mask is None:
        mask = identity_mps(L=len(ux))
    solx = [ux]
    soly = [uy]
    b = Dx.apply(solx[-1]) + Dy.apply(soly[-1])

    error = []
    divergence = []

    if "dmrg" in opt_method:
        dmrg_sites = int(opt_method[-1])

    for i in tqdm(range(N), miniters=min(25, N // 5)):
        for _ in range(Niter):
            _opt_flag = False
            if i % 15 == 0:
                _opt_flag = True

            elif "dmrg" in opt_method:
                phi = linsystem_solver(
                    A=A,
                    b=b,
                    max_iter=opt_iter,
                    max_rank=min(b.max_rank, 30),
                    eps=opt_accuracy,
                    dmrg_sites=dmrg_sites,
                    opt_paths=_opt_flag,
                    verbose=verbose,
                )

            if verbose:
                error.append(phi[1])
                phi = phi[0]

            solx[-1] = Dx.apply(phi) * (-1.0) + solx[-1]  # * mask
            soly[-1] = Dy.apply(phi) * (-1.0) + soly[-1]  # * mask

        b = Dx.apply(solx[-1]) + Dy.apply(soly[-1])
        divergence.append(b.norm)

        solx[-1] = solx[-1] * mask
        soly[-1] = soly[-1] * mask

        if scheme == "second" and i > 0:
            vx, vy = vel_predictor(
                solx[-1],
                soly[-1],
                nofq,
                nu,
                dt,
                Dx1,
                Dy1,
                Dxx1,
                Dyy1,
                solx[-2],
                soly[-2],
                ret_comps=True,
            )
        elif scheme == "first" or i == 0:
            vx, vy = vel_predictor(
                solx[-1], soly[-1], nofq, nu, dt, Dx1, Dy1, Dxx1, Dyy1, ret_comps=True
            )
        solx.append(vx)
        soly.append(vy)

        b = Dx.apply(solx[-1]) + Dy.apply(soly[-1])

        if saveit is not None and i % saveit == 0:
            save_object(
                (solx[-saveit:], soly[-saveit:]),
                f"../results/temp/{UID}_int_sol_{i}.tmp",
            )
    if verbose:
        return solx, soly, error, divergence

    return solx, soly

def Poisson(
    ux: MPS,
    uy: MPS,
    nofq: list[int],
    dt: float,
    N: int,
    nu,
    dx: float = None,
    dy: float = None,
    mask: MPS = None,
    opt_method="dmrg2",
    opt_accuracy=1e-6,
    opt_iter=6,
    opt_paths_iter=100,
    bdtrunc=None,
    saveit=None,
    UID=None,
    verbose=False,
    ret_times=False,
    inv_mpo=None,
):
    if bdtrunc:
        bdtruncp = bdtrunc
    else:
        bdtruncp = int(30)

    if isinstance(nofq, int):
        nofq = [nofq, nofq]

    if dx is None:
        dx = parameters("bx") - parameters("ax")
        dx /= (2 ** nofq[0]) - 1
    if dy is None:
        dy = parameters("by") - parameters("ay")
        dy /= (2 ** nofq[1]) - 1

    phiout = identity_mps(sum(nofq)) * 1.0 * dt
    phi_temp = None

    bcv = ["dirichlet", "neumann", "periodic", "periodic"]
    bcp = ["dirichlet", "dirichlet", "dirichlet", "dirichlet"]

    Dxp = get_diff_mpo(dim=2, order=1, axis=0, nofq=nofq, bc=bcp) * (1.0 / dx)
    Dyp = get_diff_mpo(dim=2, order=1, axis=1, nofq=nofq, bc=bcp) * (1.0 / dy)
    Dxx = get_diff_mpo(dim=2, order=2, axis=0, nofq=nofq, bc=bcp) * (1.0 / dx**2)
    Dyy = get_diff_mpo(dim=2, order=2, axis=1, nofq=nofq, bc=bcp) * (1.0 / dy**2)

    Dx1 = get_diff_mpo(dim=2, order=1, axis=0, bc=bcv) * (1.0 / dx)
    Dy1 = get_diff_mpo(dim=2, order=1, axis=1, bc=bcv) * (1.0 / dy)
    Dxx1 = get_diff_mpo(dim=2, order=2, axis=0, bc=bcv) * (1.0 / dx**2)
    Dyy1 = get_diff_mpo(dim=2, order=2, axis=1, bc=bcv) * (1.0 / dy**2)

    _, rbop2 = get_dirichlet_correction(2, nofq=nofq, axis=0)
    rbop2 = rbop2 * (1.0 / dx**2)
    _, rbop1 = get_dirichlet_correction(1, nofq=nofq, axis=0)
    rbop1 = rbop1 * (1.0 / dx**1)

    A = Dxx + Dyy

    if mask is None:
        mask = identity_mps(L=len(ux))
    solx = [ux]
    soly = [uy]
    solp = []
    # b = get_2d_divergence(solx[-1], soly[-1], Dx1, Dy1, nofq, bcv)
    # if bcp[1] == "dirichlet":
    #     b = b + rbop2.apply(phiout) * -1.0
    b = ux

    error = []

    if "dmrg" in opt_method:
        dmrg_sites = int(opt_method[-1])

    times = []
    for i in tqdm(range(N)):
        _opt_flag = False
        if i % opt_paths_iter == 0:
            _opt_flag = True

        start = time.time()

        if "dmrg" in opt_method:
            phi = linsystem_solver(
                A=A,
                b=b,
                x=phi_temp,
                max_iter=opt_iter,
                max_rank=bdtruncp,
                eps=opt_accuracy,
                dmrg_sites=dmrg_sites,
                opt_paths=_opt_flag,
                verbose=verbose,
            )
            if verbose:
                error.append(phi[1])
                phi = phi[0]

        vxp = solx[-1] + Dxp.apply(phi) * (-1.0)

        if bcp[1] == "dirichlet":
            vxp = vxp + rbop1.apply(phiout) * -1.0
        vxp = vxp * mask

        vyp = (Dyp.apply(phi) * (-1.0) + soly[-1]) * mask

        vxp.compress(max_bd=bdtrunc)
        vyp.compress(max_bd=bdtrunc)

        solx[-1] = vxp.copy()
        soly[-1] = vyp.copy()

        vx, vy = vel_predictor(
            solx[-1],
            soly[-1],
            nofq,
            nu,
            dt,
            Dx1,
            Dy1,
            Dxx1,
            Dyy1,
            bc=bcv,
            ret_comps=True,
        )

        solp.append(phi * (1.0 / dt))
        solx.append(vx)
        soly.append(vy)

        stop = time.time()

        times.append(stop - start)

        if bcp[1] == "dirichlet":
            b = b + rbop2.apply(phiout) * -1.0

        if saveit is not None and i % saveit == 0:
            save_object(
                (solx[-saveit:], soly[-saveit:], solp[-saveit:]),
                f"../results/temp/{UID}_int_sol_{i}.tmp",
            )
    # results = (solx, soly, solp)
    results = (solx, uy, solp)
    print('lengths of solx, soly, solp, ux: ')
    print(len(solx))
    print(len(soly))
    print(len(solp))
    print(len(ux))


    if ret_times:
        results += (times,)

    if verbose:
        return results + (error,)

    return results

