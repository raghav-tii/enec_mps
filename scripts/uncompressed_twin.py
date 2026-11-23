#!/usr/bin/env python3
"""Exact-Poisson twin of an MPS bundle run: same operators, same start state,
no truncation, no DMRG.  Separates MPS error from discretisation error.

dense_projection_check.py does this with n^2 x n^2 matrices, so it stops at Q=6.  Every
chorin_projection operator is one 1D stencil on one axis, so here each is an n x n
matrix applied along its axis, and the Poisson system lap(phi) = b is the
Sylvester equation Lx phi + phi Ly^T = b, solved exactly by Bartels-Stewart
with both Schur forms computed once.  That makes Q=10 about a second a step.
The step itself is dense_projection_check.dense_step, i.e. chorin_projection.projection_step's
algebra.

Before marching, every operator is checked against its MPO on the real start
field, so a stencil, BC or bit-order mismatch aborts instead of running.

    python3 uncompressed_twin.py --tag p1A_inc_it6 --nt 1000 [--rk4]

Starts from the tag's first snapshot (with its pressure) and writes
data/results/rod_forces_exact[_rk4]_TAG.csv in rod_forces's format, rows aligned with
rod_forces_TAG.csv.
"""

import argparse
import csv
import os
import time

import numpy as np
from scipy.linalg import get_lapack_funcs, schur, solve_banded

import tensnet as tt

import bundle_simulation as es
import chorin_projection as C
import rod_forces
from dense_projection_check import dense_step, densify

tt.set_arch("sequential")


class Axis:
    """An n x n matrix acting along one axis of an [x, y] field."""

    def __init__(self, M, axis):
        self.M, self.axis = M, axis

    def __matmul__(self, f):
        return self.M @ f if self.axis == 0 else f @ self.M.T


class SepOps:
    """chorin_projection.Operators as per-axis matrices (same interface as
    dense_projection_check.DenseOps)."""

    def __init__(self, ops):
        Q, n = ops.Q, 2**ops.Q

        def m1(order, axis, bc, h):
            mpo = tt.get_diff_mpo(dim=1, order=order, nofq=[Q], bc=bc[2 * axis : 2 * axis + 2])
            return densify(mpo) / h**order

        dx, dy = ops.dx, ops.dy
        self.Dx, self.Dy = Axis(m1(1, 0, ops.bcv, dx), 0), Axis(m1(1, 1, ops.bcv, dy), 1)
        self.Gx, self.Gy = Axis(m1(1, 0, ops.bcp, dx), 0), Axis(m1(1, 1, ops.bcp, dy), 1)
        self.Dxx, self.Dyy = Axis(m1(2, 0, ops.bcv, dx), 0), Axis(m1(2, 1, ops.bcv, dy), 1)
        keep = np.ones(n)
        if ops.hold is not None:
            keep[:4] = keep[-4:] = 0.0  # chorin_projection's inflow_rows=4, both ends
        self.hold = Axis(np.diag(keep), 0)
        # lap = Gx @ Dx in tensnet's reversed composition: grad first, then div
        self.Lx, self.Ly = self.Dx.M @ self.Gx.M, self.Dy.M @ self.Gy.M
        self.isheld = getattr(ops, "held_projection", False)
        if self.isheld:
            # chorin_projection's held projection: H div(H grad phi) + (I - H) phi = b, i.e.
            # Ax phi + H phi Ly^T = b.  Ly (periodic) is symmetric, so in its
            # eigenbasis each y-mode is one banded x-system (Ax + lam_k H).
            r = 4
            copy = np.eye(n)
            copy[n - r :] = 0.0
            copy[n - r :, n - r - 1] = 1.0
            self.outflow = Axis(copy, 0)
            assert np.allclose(self.Ly, self.Ly.T), "periodic y-Laplacian not symmetric"
            self.lam, self.v = np.linalg.eigh(self.Ly)
            H = np.diag(keep)
            self.edge = -(
                1.0 / dx**2 + 1.0 / dy**2
            )  # chorin_projection's held rows: the Laplacian's sign and size
            Ax = H @ self.Dx.M @ H @ self.Gx.M + self.edge * (np.eye(n) - H)
            i, j = np.nonzero(np.abs(Ax) > 0)
            self.lu = (int((i - j).max()), int((j - i).max()))
            self.ab = np.zeros((sum(self.lu) + 1, n))
            for d in range(-self.lu[0], self.lu[1] + 1):  # LAPACK banded storage
                self.ab[self.lu[1] - d, max(d, 0) : n + min(d, 0)] = np.diagonal(Ax, d)
            self.keep = keep
        else:
            self.r, self.u = schur(self.Lx, output="real")
            self.s, self.v = schur(self.Ly, output="real")
            (self.trsyl,) = get_lapack_funcs(("trsyl",), (self.r, self.s, self.Lx))

    def held(self, v):
        return self.hold @ v

    def divergence(self, u, v):
        return self.Dx @ u + self.Dy @ v

    def laplacian(self, f):
        return self.Dxx @ f + self.Dyy @ f

    def lap(self, f):
        if self.isheld:
            k = self.keep[:, None]
            return k * (self.Dx.M @ (k * (self.Gx.M @ f)) + f @ self.Ly.T) + self.edge * (1 - k) * f
        return self.Lx @ f + f @ self.Ly.T

    def solve(self, b):
        """Exact: Lx phi + phi Ly^T = b (scipy.linalg.solve_sylvester, Schur cached)."""
        if self.isheld:
            bt, out = b @ self.v, np.empty_like(b)
            for k, lam in enumerate(self.lam):
                ab = self.ab.copy()
                ab[self.lu[1]] += lam * self.keep
                out[:, k] = solve_banded(self.lu, ab, bt[:, k])
            return out @ self.v.T
        y, scale, info = self.trsyl(self.r, self.s, self.u.T @ b @ self.v, tranb="C")
        assert info >= 0, info
        return self.u @ (scale * y) @ self.v.T


def check(ops, sep, m):
    """Every separable operator against its MPO on a real field."""
    Q = ops.Q
    g = C.mps_to_grid(m, Q)
    pairs = [
        ("Dx", ops.Dx, sep.Dx),
        ("Dy", ops.Dy, sep.Dy),
        ("Gx", ops.Gx, sep.Gx),
        ("Gy", ops.Gy, sep.Gy),
        ("Dxx", ops.Dxx, sep.Dxx),
        ("Dyy", ops.Dyy, sep.Dyy),
    ]
    if sep.isheld:
        pairs.append(("outflow", ops.outflow, sep.outflow))
    worst = 0.0
    for name, mpo, s in pairs:
        want, got = C.mps_to_grid(mpo.apply(m), Q), s @ g
        err = np.abs(got - want).max() / np.abs(want).max()
        worst = max(worst, err)
        print(f"  check {name:4s} {err:.1e}")
    for name, want, got in [
        ("hold", C.mps_to_grid(ops.held(m), Q), sep.held(g)),
        ("lap", C.mps_to_grid(ops.lap.apply(m), Q), sep.lap(g)),
    ]:
        err = np.abs(got - want).max() / np.abs(want).max()
        worst = max(worst, err)
        print(f"  check {name:4s} {err:.1e}")
    b = sep.lap(g)
    err = np.abs(sep.solve(b) - g).max() / np.abs(g).max()
    print(f"  check solve {err:.1e}")
    assert worst < 1e-8 and err < 1e-6, (worst, err)


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--tag", required=True)
    a.add_argument("--nt", type=int, default=1000)
    a.add_argument("--results", default=es.RESULTS)
    a.add_argument("--mask-dir", default=es.MASK_DIR)
    a.add_argument("--reference", default=os.environ.get("QCFD_REFERENCE", es.REFERENCE))
    a.add_argument(
        "--rk4",
        action="store_true",
        help="RK4 predictor (chorin_projection's --rk4) instead of Euler",
    )
    args = a.parse_args()

    cfg = tt.load_object(os.path.join(args.results, f"{args.tag}_params.pfile"))
    Q, dt, nu = cfg["p"] + cfg["Lf"], cfg["dt"], cfg["nu"]
    ops = C.Operators(
        Q, cfg["dx"], cfg["dy"], bcv=C.BCV_CHANNEL, bcp=C.BCP_CHANNEL, held=cfg.get("held", False)
    )
    sep = SepOps(ops)
    mask = es.load_or_build_mask(
        cfg["p"],
        cfg["Lf"],
        cfg.get("d_over_pitch", es.D_OVER_PITCH),
        args.mask_dir,
        accuracy=cfg["eps"],
    )
    ux, uy, p = tt.load_object(es.find_snapshots(args.tag, args.results)[0][2])
    check(ops, sep, ux)
    mask, ux, uy, p = (C.mps_to_grid(f, Q) for f in (mask, ux, uy, p))

    bxs = rod_forces.boxes(cfg)
    names = es.rod_names(cfg["p"], cfg["Lf"])
    rcd, _ = es.reference_coefficients(args.reference)
    ref_cd = np.array([rcd[k] for k in names])
    name = f"exact{'_rk4' if args.rk4 else ''}_{args.tag}"
    out = os.path.join(args.results, f"rod_forces_{name}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["step", "time", "closure"] + [f"Cd_{k}" for k in names] + [f"Cl_{k}" for k in names]
        )
        for step in range(args.nt + 1):
            t0 = time.time()
            if step:
                ux, uy, p = dense_step(ux, uy, sep, dt, nu, mask, p, rk4=args.rk4)
            cd, cl, clo, _ = rod_forces.rod_coefficients(ux, uy, cfg, bxs, dense=True)
            w.writerow([step, (step + 1) * dt, np.abs(clo).mean()] + list(cd) + list(cl))
            fh.flush()
            div = np.linalg.norm(sep.held(sep.divergence(ux, uy))) * cfg["dx"] / np.linalg.norm(ux)
            print(
                f"  step {step:4d}  |u|={np.linalg.norm(ux):.4e}  div={div:.1e}"
                f"  <Cd>={cd.mean():+.4f}  <Cl>={cl.mean():+.4f}"
                f"  corr={np.corrcoef(cd, ref_cd)[0, 1]:+.4f}  clo={np.abs(clo).mean():.3f}"
                f"  {time.time() - t0:.1f}s",
                flush=True,
            )
            if not np.isfinite(cd).all():
                raise SystemExit("non-finite forces")
    np.savez(os.path.join(args.results, f"{name}_last.npz"), ux=ux, uy=uy, p=p, step=step)


if __name__ == "__main__":
    main()
