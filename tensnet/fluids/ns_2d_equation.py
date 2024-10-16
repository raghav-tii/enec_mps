import torch
from tqdm import tqdm
import time

from ..linalg import (
    get_2D_operator,
    get_diff_mpo,
    get_dirichlet_correction,
    iterative_inversion,
    jacobi_iter,
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


def construct_b(
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
    ret_times=True,
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
    rtol = 2.0

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

    if opt_method == "iter_inversion":
        # inv_A = iterative_inversion(
        #     mpo=A,
        #     eps=opt_accuracy,
        #     opt_flag=True,
        #     max_iter=opt_iter,
        #     verbose=verbose,
        # )
        # if verbose:
        #     error.append(inv_A[1])
        #     inv_A = inv_A[0]
        raise DeprecationWarning()
    if "dmrg" in opt_method:
        dmrg_sites = int(opt_method[-1])

    if "ml_inversion" in opt_method:
        assert inv_mpo is not None, ValueError("Provide valid inverse mpo.")
        inv_A = inv_mpo.copy()

    dmrg_iter = None

    if "ml_inversion" in opt_method and "dmrg" in opt_method:
        dmrg_iter = opt_method[13:-6]
        if dmrg_iter.isdigit():
            dmrg_iter = int(dmrg_iter)
        else:
            raise Exception("Enter opt_method in format ml_inversion_**_dmrg2")

    times = []
    for i in tqdm(range(N)):
        _opt_flag = False
        if i % opt_paths_iter == 0:
            _opt_flag = True
        if opt_method == "jacobi":
            raise DeprecationWarning()

        start = time.time()

        if dmrg_iter is not None and i % dmrg_iter == 0:
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
        elif "ml_inversion" in opt_method:
            phi = inv_A.apply(b)
            if verbose:
                errmps = A.apply(phi) + b * -1.0
                error.append(errmps.norm)
        elif "dmrg" in opt_method:
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

        b1 = get_2d_divergence(vxp, vyp, Dx1, Dy1, nofq, bcv)

        solx[-1] = vxp.copy()
        soly[-1] = vyp.copy()
        if bdtrunc:
            vxp.compress(max_bd=bdtrunc)
            vyp.compress(max_bd=bdtrunc)
            b2 = get_2d_divergence(vxp, vyp, Dx1, Dy1, nofq, bcv)
            if b2.norm <= rtol * b1.norm:
                b = b2
                solx[-1] = vxp
                soly[-1] = vyp

        vx, vy = construct_b(
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

        # phi_temp = phi.copy()
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


def semi_implicit_time_stepping(
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
    rtol = 2.0

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

    if opt_method == "iter_inversion":
        inv_A = iterative_inversion(
            mpo=A,
            eps=opt_accuracy,
            opt_flag=True,
            max_iter=opt_iter,
            verbose=verbose,
        )
        if verbose:
            error.append(inv_A[1])
            inv_A = inv_A[0]
    elif "dmrg" in opt_method:
        dmrg_sites = int(opt_method[-1])

    elif opt_method == "ml_inversion":
        assert inv_mpo is not None, ValueError("Provide valid inverse mpo.")
        inv_A = inv_mpo.copy()

    for i in tqdm(range(N)):
        _opt_flag = False
        if i % opt_paths_iter == 0:
            _opt_flag = True
        if opt_method == "jacobi":
            raise DeprecationWarning()
        elif "dmrg" in opt_method:
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
        elif opt_method == "iter_inversion" or opt_method == "ml_inversion":
            phi = inv_A.apply(b)
            if verbose:
                errmps = A.apply(phi) + b * -1.0
                error.append(errmps.norm)

        vxp = solx[-1] + Dxp.apply(phi) * (-1.0)

        if bcp[1] == "dirichlet":
            vxp = vxp + rbop1.apply(phiout) * -1.0
        vxp = vxp * mask

        vyp = (Dyp.apply(phi) * (-1.0) + soly[-1]) * mask

        b1 = get_2d_divergence(vxp, vyp, Dx1, Dy1, nofq, bcv)

        solx[-1] = vxp.copy()
        soly[-1] = vyp.copy()
        if bdtrunc:
            vxp.compress(max_bd=bdtrunc)
            vyp.compress(max_bd=bdtrunc)
            b2 = get_2d_divergence(vxp, vyp, Dx1, Dy1, nofq, bcv)
            if b2.norm <= rtol * b1.norm:
                b = b2
                solx[-1] = vxp
                soly[-1] = vyp

        vx, vy = construct_b(
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

        # phi_temp = phi.copy()
        solp.append(phi * (1.0 / dt))
        solx.append(vx)
        soly.append(vy)

        if bcp[1] == "dirichlet":
            b = b + rbop2.apply(phiout) * -1.0

        if saveit is not None and i % saveit == 0:
            save_object(
                (solx[-saveit:], soly[-saveit:], solp[-saveit:]),
                f"../results/temp/{UID}_int_sol_{i}.tmp",
            )

    if verbose:
        return solx, soly, solp, error

    return solx, soly, solp


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
    if opt_method == "iter_inversion":
        error1 = []
        inv_A, err = iterative_inversion(
            mpo=A,
            eps=opt_accuracy,
            opt_flag=opt_flag,
            max_iter=opt_iter,
            verbose=verbose,
        )
        error.append(err)
    if "dmrg" in opt_method:
        dmrg_sites = int(opt_method[-1])

    for i in tqdm(range(N), miniters=min(25, N // 5)):
        for _ in range(Niter):
            _opt_flag = False
            if i % 15 == 0:
                _opt_flag = True
            if opt_method == "jacobi":
                raise DeprecationWarning()
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
            elif opt_method == "iter_inversion":
                phi = inv_A.apply(b)
                error_mps = A.apply(phi) + b * -1.0
                error1.append(error_mps.norm)

            if verbose and opt_method in ["dmrg2", "dmrg1"]:
                error.append(phi[1])
                phi = phi[0]

            solx[-1] = Dx.apply(phi) * (-1.0) + solx[-1]  # * mask
            soly[-1] = Dy.apply(phi) * (-1.0) + soly[-1]  # * mask

        b = Dx.apply(solx[-1]) + Dy.apply(soly[-1])
        divergence.append(b.norm)

        solx[-1] = solx[-1] * mask
        soly[-1] = soly[-1] * mask

        if scheme == "second" and i > 0:
            vx, vy = construct_b(
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
            vx, vy = construct_b(
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
        error.append(error1)
        return solx, soly, error, divergence

    return solx, soly


def rk_time_stepping(
    ux: MPS,
    uy: MPS,
    nofq: list[int],
    dt: float,
    N: int,
    nu,
    dx: float = None,
    dy: float = None,
    mu: float = 1e4,
    mask: MPS = None,
    opt_method="dmrg2",
    opt_accuracy=1e-3,
    opt_flag=False,
    scheme="first",
    saveit=None,
    UID=None,
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
    A = construct_A(nofq, dt, dx, dy, mu)
    if mask is None:
        mask = identity_mps(L=len(ux))

    solx = [ux]
    soly = [uy]
    b = construct_b(solx[-1], soly[-1], nofq, nu, dt, dx, dy)
    if "dmrg" in opt_method:
        dmrg_sites = int(opt_method[-1])

    for i in tqdm(range(N), miniters=min(25, N // 5)):
        _opt_flag = False
        if i % 15 == 0:
            _opt_flag = True
        if opt_method == "jacobi":
            raise DeprecationWarning()
        elif "dmrg" in opt_method:
            u1 = linsystem_solver(
                A=A,
                b=b,
                max_rank=min(b.max_rank, 40),
                eps=opt_accuracy,
                dmrg_sites=dmrg_sites,
                opt_paths=_opt_flag,
            )

        solx.append(get_2d_vec_comp(u1, 0) * mask)
        soly.append(get_2d_vec_comp(u1, 1) * mask)
        if scheme == "second":
            b = construct_b(
                solx[-1], soly[-1], nofq, nu, dt, dx, dy, solx[-2], soly[-2]
            )
        elif scheme == "first":
            b = construct_b(solx[-1], soly[-1], nofq, nu, dt, dx, dy)
        if saveit is not None and i % saveit == 0:
            save_object(
                (solx[-saveit:], soly[-saveit:]),
                f"../results/temp/{UID}_int_sol_{i}.tmp",
            )
        if u1 is None:
            break
    return solx, soly


def construct_steady_A(nofq: int, dx: float, dy: float, mu: float = 1e3):
    """This for the stacked architecture. Only works with x and y, coarse to fine in both"""
    k1 = -1 * mu / (dx**2)
    # ein_str = [
    #     "l1 x r1 o, l2 i r2 x-> l1 l2 i r1 r2 o",
    #     "l1 l2 i r1 r2 o-> (l1 l2) i (r1 r2) o",
    # ]

    t1 = torch.zeros((1, 2, 1, 2))
    t1[:, 0, :, 0] = 1
    t2 = torch.zeros((1, 2, 1, 2))
    t2[:, 0, :, 1] = 1
    t3 = torch.zeros((1, 2, 1, 2))
    t3[:, 1, :, 0] = 1
    t4 = torch.zeros((1, 2, 1, 2))
    t4[:, 1, :, 1] = 1

    # q1 = MPO_Node(t1)
    # q2 = MPO_Node(t2)
    # q3 = MPO_Node(t3)
    # q4 = MPO_Node(t4)

    # Dx = get_diff_mpo(dim=2, order=1, axis=0) * (1.0 / dx)
    # Dy = get_diff_mpo(dim=2, order=1, axis=1) * (1.0 / dy)
    Dxx = get_diff_mpo(dim=2, order=2, axis=0) * (1.0 / dx**2)
    Dyy = get_diff_mpo(dim=2, order=2, axis=1) * (1.0 / dy**2)
    Dxy = get_diff_mpo(dim=2, order=2, axis=-1) * (1.0 / (dx * dy))

    # Dx4 = MPO(T4.nodes + I.nodes).compress()
    # Dy4 = MPO(I.nodes + T4.nodes).compress()
    # Dxxyy = MPO(
    #     [
    #         MPO_Node(nodex.contract(nodey, einstr=ein_str))
    #         for nodex, nodey in zip(T2.nodes + I.nodes, I.nodes + T2.nodes)
    #     ]
    # ).compress()
    # Dyyxx = MPO(
    #     [
    #         MPO_Node(nodey.contract(nodex, einstr=ein_str))
    #         for nodex, nodey in zip(I.nodes + T2.nodes, T2.nodes + I.nodes)
    #     ]
    # ).compress()
    # comp4 = Dx4 + Dy4 + Dxxyy + Dyyxx
    # D_00 = MPO([q1] + comp4.nodes)
    # D_11 = MPO([q4] + comp4.nodes)

    mpo = (Dxx * k1) + (Dxy * k1) + (Dxy * k1) + (Dyy * k1)
    mpo.compress()

    return mpo


def construct_steady_b(ux: MPS, uy: MPS, nofq: int, dx: float, dy: float, nu: float):
    """This for the stacked architecture. Only works with x and y, coarse to fine in both"""
    k1 = -1.0 / dx
    # k2 = nu / (dx**2)

    # I = identity_mpo(L=nofq)

    t1 = torch.zeros((1, 2, 1))
    t1[:, 0, :] = 1
    t2 = torch.zeros((1, 2, 1))
    t2[:, 1, :] = 1

    q1 = MPS_Node(t1)
    q2 = MPS_Node(t2)
    Dx = get_diff_mpo(dim=2, order=1, axis=0) * (1.0 / dx)
    Dy = get_diff_mpo(dim=2, order=1, axis=1) * (1.0 / dy)
    Dxx = get_diff_mpo(dim=2, order=2, axis=0) * (1.0 / dx**2)
    Dyy = get_diff_mpo(dim=2, order=2, axis=1) * (1.0 / dy**2)

    comp2 = Dxx + Dyy

    bx = comp2.apply(ux * Dx.apply(ux) * k1 + uy * Dy.apply(ux) * k1).compress()
    by = comp2.apply(ux * Dx.apply(uy) * k1 + uy * Dy.apply(uy) * k1).compress()

    b = MPS([q1] + bx.nodes) + MPS([q2] + by.nodes)
    return b.compress()


def steady_state_solver(
    ux: MPS,
    uy: MPS,
    nofq: int,
    dx: float,
    nu,
    iterations=20,
    mu: float = 1e4,
    mask: MPS = None,
    method="jacobi",
):
    A = construct_steady_A(nofq, dx, mu)
    if mask is None:
        mask = identity_mps(L=len(ux))
    b = construct_steady_b(ux, uy, nofq, dx, nu)
    if "dmrg" in method:
        dmrg_sites = int(method[-1])
    for i in tqdm(range(iterations)):
        if method == "jacobi":
            u1 = jacobi_iter(mpo=A, mps=b)
        elif "dmrg" in method:
            u1 = linsystem_solver(
                A=A, b=b, max_rank=min(b.max_rank, 40), dmrg_sites=dmrg_sites
            )
        if u1 is None:
            break
        if i % 5 == 0:
            save_object(u1, f"int_sol_{i}.tmp")

        ux1 = get_2d_vec_comp(u1, 0) * mask
        uy1 = get_2d_vec_comp(u1, 1) * mask
        ux1.compress()
        uy1.compress()
        solx = ux1.copy()
        soly = uy1.copy()
        b = construct_b(ux1, uy1, nofq, dx, nu)

    return solx, soly
