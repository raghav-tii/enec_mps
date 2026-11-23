#!/usr/bin/env python3
"""Dense reference for chorin_projection.py, to separate algorithm bugs from MPS error.

Runs the same Chorin rotational incremental pressure-correction on ordinary
numpy arrays with an exact Poisson solve, so the only thing it does not share
with the MPS version is the tensor-network machinery.  Any disagreement is
MPS truncation or DMRG error; any agreement on a wrong answer is a bug in the
scheme itself, which is the distinction the bundle's pressure problem needs.

The operators are not re-derived.  They are the very MPOs `chorin_projection.Operators`
builds, contracted to matrices, so the two solvers share a discretisation by
construction rather than by my getting a stencil right twice.  Note
`MPO.contract()` returns the TRANSPOSE of the matrix that acts on a vector
(verified: `A.T @ f` reproduces `mpo.apply(f)` to 6e-15), the same reversed
convention as tensnet's `@`.

Cost is n^2 x n^2 dense, so Q=5 is 1024^2 per operator and Q=6 is 4096^2.
Q=7 would be 16384^2 -- 2 GB per operator -- so this is a small-grid check,
which is all it needs to be.

    python3 dense_projection_check.py            # Q=5, compare dense against MPS
    python3 dense_projection_check.py --Q 6 --steps 8 [--rk4]
"""

import argparse

import numpy as np
import torch

import tensnet as tt

import chorin_projection as C

torch.set_default_dtype(torch.float64)


def densify(mpo):
    """Matrix that acts on a shuffled-order vector, from an MPO."""
    return np.asarray(mpo.contract()).T


class DenseOps:
    """The same operators chorin_projection uses, as matrices."""

    def __init__(self, ops):
        self.Q, self.dx, self.dy = ops.Q, ops.dx, ops.dy
        self.Dx, self.Dy = densify(ops.Dx), densify(ops.Dy)
        self.Gx, self.Gy = densify(ops.Gx), densify(ops.Gy)
        self.Dxx, self.Dyy = densify(ops.Dxx), densify(ops.Dyy)
        self.lap = densify(ops.lap)
        self.hold = None if ops.hold is None else densify(ops.hold)
        self.isheld = getattr(ops, "held_projection", False)
        self.outflow = densify(ops.outflow) if self.isheld else None

    def held(self, v):
        return v if self.hold is None else self.hold @ v

    def divergence(self, u, v):
        return self.Dx @ u + self.Dy @ v

    def laplacian(self, f):
        return self.Dxx @ f + self.Dyy @ f

    def solve(self, b):
        """Exact Poisson solve -- no DMRG, no rank cap, no sweeps."""
        try:
            return np.linalg.solve(self.lap, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.lap, b, rcond=None)[0]


def dense_step(ux, uy, ops, dt, nu, mask=None, pressure=None, rotational=True, rk4=False):
    """One step, algebraically identical to chorin_projection.projection_step (and its `rk4`)."""

    def rhs(u, v):
        ix = -(u * (ops.Dx @ u) + v * (ops.Dy @ u)) + nu * ops.laplacian(u)
        iy = -(u * (ops.Dx @ v) + v * (ops.Dy @ v)) + nu * ops.laplacian(v)
        if pressure is not None:
            ix = ix - ops.Gx @ pressure
            iy = iy - ops.Gy @ pressure
        return ops.held(ix), ops.held(iy)

    def stage(c, k):
        u, v = ux + c * dt * k[0], uy + c * dt * k[1]
        if mask is not None:
            u, v = u * mask, v * mask
        if getattr(ops, "isheld", False):
            u, v = ops.outflow @ u, ops.outflow @ v
        return u, v

    k = rhs(ux, uy)
    if rk4:
        k2 = rhs(*stage(0.5, k))
        k3 = rhs(*stage(0.5, k2))
        k4 = rhs(*stage(1.0, k3))
        k = tuple((a + 2 * b + 2 * c + d) / 6 for a, b, c, d in zip(k, k2, k3, k4))
    us, vs = stage(1.0, k)
    return dense_project(us, vs, ops, dt, nu, pressure, rotational)


def dense_project(us, vs, ops, dt, nu, pressure=None, rotational=True):
    """The projection half of dense_step: algebraically identical to
    chorin_projection.project_divergence_free."""
    b = ops.held(ops.divergence(us, vs))
    phi = ops.solve(b)
    # Correction NOT held, matching chorin_projection: holding the right-hand side
    # restricts the projection to the rows we mean to fix, holding the
    # correction as well breaks the projection identity.
    gx, gy = ops.Gx @ phi, ops.Gy @ phi
    if getattr(ops, "isheld", False):  # chorin_projection's held projection: correction held too
        gx, gy = ops.held(gx), ops.held(gy)
    nx, ny = us - gx, vs - gy

    dp = phi / dt
    p_new = dp if pressure is None else pressure + dp
    if rotational:
        p_new = p_new + (-nu) * ops.held(ops.divergence(us, vs))
    return nx, ny, p_new


def flat(m, Q):
    """MPS -> shuffled-order dense vector."""
    return np.asarray(m.contract()).reshape(-1)


def to_mps(v, Q, eps=1e-13):
    return tt.vector_to_mps(torch.from_numpy(np.ascontiguousarray(v)), 2, 2 * Q, eps)


def compare(Q=5, steps=6, max_bd=None, opt_iter=20, nu=5e-3, seed=0, rk4=False, held=False):
    n = 2**Q
    dx = 1.0 / (n - 1)
    g = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(g, g, indexing="ij")

    ops = C.Operators(Q, dx, bcv=C.BCV_CHANNEL, bcp=C.BCP_CHANNEL, held=held)
    dops = DenseOps(ops)

    # flow past a soft disc, the configuration the bundle is made of
    r = np.sqrt((X - 0.35) ** 2 + (Y - 0.5) ** 2)
    mask_g = 1.0 / (1.0 + np.exp(-(r - 0.12) / (2 * dx)))
    mask = C.grid_to_mps(mask_g, Q)
    mask_v = flat(mask, Q)

    U = 1.0
    ux_m, uy_m = mask * U, mask * 0.0
    p_m = C.grid_to_mps(np.zeros_like(X), Q)
    ux_d, uy_d, p_d = flat(ux_m, Q), flat(uy_m, Q), flat(p_m, Q)

    dt = 0.2 * dx / U
    bd = max_bd or 4 * n  # effectively uncapped unless asked
    print(
        f"Q={Q}  grid {n}^2  dt={dt:.3e}  nu={nu}  "
        f"max_bd={'uncapped' if max_bd is None else max_bd}  "
        f"opt_iter={opt_iter}"
    )
    print(
        f"{'step':>5}{'|u| dense':>12}{'|u| mps':>12}{'u rel diff':>12}"
        f"{'p rel diff':>12}{'|div| dense':>13}{'|div| mps':>12}"
    )
    guess = None
    for k in range(steps):
        ux_d, uy_d, p_d = dense_step(ux_d, uy_d, dops, dt, nu, mask_v, p_d, rk4=rk4)
        ux_m, uy_m, p_m, guess = C.projection_step(
            ux_m,
            uy_m,
            ops,
            dt,
            nu,
            mask,
            eps=1e-12,
            max_bd=bd,
            opt_iter=opt_iter,
            guess=guess,
            pressure=p_m,
            rk4=rk4,
        )
        um, pm = flat(ux_m, Q), flat(p_m, Q)
        du = np.abs(um - ux_d).max() / max(np.abs(ux_d).max(), 1e-30)
        dp = np.abs(pm - p_d).max() / max(np.abs(p_d).max(), 1e-30)
        dd = np.abs(dops.held(dops.divergence(ux_d, uy_d))).max() * dx
        dm = float(ops.held(ops.divergence(ux_m, uy_m)).norm) * dx / float(ux_m.norm)
        print(
            f"{k:>5}{np.linalg.norm(ux_d):>12.4e}{float(ux_m.norm):>12.4e}"
            f"{du:>12.2e}{dp:>12.2e}{dd:>13.2e}{dm:>12.2e}",
            flush=True,
        )
    return ux_d, ux_m


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--Q", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument(
        "--max-bd",
        type=int,
        default=None,
        help="cap the MPS solver; default uncapped, so a "
        "disagreement is the DMRG solve rather than truncation",
    )
    ap.add_argument("--opt-iter", type=int, default=20)
    ap.add_argument("--nu", type=float, default=5e-3)
    ap.add_argument("--rk4", action="store_true", help="RK4 predictor in both solvers")
    ap.add_argument(
        "--held", action="store_true", help="chorin_projection's held projection in both solvers"
    )
    a = ap.parse_args()
    compare(a.Q, a.steps, a.max_bd, a.opt_iter, a.nu, rk4=a.rk4, held=a.held)


if __name__ == "__main__":
    main()
