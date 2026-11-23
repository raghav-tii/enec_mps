"""Poisson solvers scored against the EXACT phi, in dense (untruncated) norms.

    python3 poisson_rank_benchmark.py SNAPSHOT_OR_NPZ [--tag TAG] [--bd 40] [--mps-pred]
                                      [--oracle-only]

For a state (ux, uy, p) it builds the RK4 predictor densely, b = held(div u*), and the exact phi
by the Sylvester solve of uncompressed_twin.SepOps.  It then scores

  oracle   phi_ex truncated to rank k (TT-SVD): the floor for any rank-k phi
  solvers  AMEn and two-site DMRG at max_rank --bd, from a cold start and from the oracle,
           for 6, 12 and 25 sweeps

with these metrics: the true residual ||lap phi - b|| / ||b||, the residual as the solver
measures it (MPS residual truncated at the bond limit), the error of phi, the error of grad phi
(= the error of the velocity correction) and |div u| dx / |u| after the correction.

--mps-pred forms b the way a run does (u* from chorin_projection.predict_velocity truncated to
--bd) instead of densely.  --oracle-only skips the solvers.  Run from scripts/.
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

import tensnet as tt

import bundle_simulation as es
import chorin_projection as C
import uncompressed_twin as E

EPS = 1e-12
RANKS = (25, 40, 80, 120, 200)


@dataclass
class Problem:
    """One projection step: operators, predictor u*, right-hand side b and the exact solution."""

    Q: int
    dx: float
    dt: float
    bd: int
    ops: C.Operators
    dense: E.SepOps
    us: np.ndarray
    vs: np.ndarray
    b: np.ndarray
    phi_ex: np.ndarray

    @property
    def u_norm(self):
        return np.linalg.norm(self.us)

    def div_after(self, ux, uy):
        """|div u| dx / |u| of a velocity field, with the boundary rows held."""
        return (
            np.linalg.norm(self.dense.held(self.dense.divergence(ux, uy))) * self.dx / self.u_norm
        )


# --- command line -------------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = tt.load_object(os.path.join(es.RESULTS, f"{args.tag}_params.pfile"))
    problem = build_problem(args, cfg)
    report_exact(problem, args.state)

    b_m = C.grid_to_mps(problem.b, problem.Q, EPS)
    phi_ex_m = C.grid_to_mps(problem.phi_ex, problem.Q, EPS)
    report_ranks(problem, b_m, phi_ex_m)
    for k in RANKS:
        score(problem, b_m, f"oracle trunc(phi_ex,{k})", phi_ex_m.copy().compress(max_bd=k), 0)
    report_velocity_truncation(problem)
    if args.oracle_only:
        raise SystemExit("DONE (oracle only)")
    score_solvers(problem, b_m, phi_ex_m)
    print("DONE")


def parse_args():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("state", help="snapshot file (ux, uy, p as MPS) or .npz with dense ux, uy, p")
    a.add_argument("--tag", default="p1A_inc_it6", help="run whose params.pfile gives the settings")
    a.add_argument("--bd", type=int, default=40)
    a.add_argument("--oracle-only", action="store_true", help="skip the MPS solvers")
    a.add_argument(
        "--mps-pred",
        action="store_true",
        help="u* from the MPS predictor at max_bd, b in MPS uncapped: what a run solves",
    )
    return a.parse_args()


# --- the problem --------------------------------------------------------------------------------


def build_problem(args, cfg):
    """Dense RK4 predictor of the saved state, b = held(div u*), and the exact solve."""
    Q, dt, nu, dx = cfg["p"] + cfg["Lf"], cfg["dt"], cfg["nu"], cfg["dx"]
    ops = C.Operators(Q, dx, dx, bcv=C.BCV_CHANNEL, bcp=C.BCP_CHANNEL)
    dense = E.SepOps(ops)
    mask_m = es.load_or_build_mask(
        cfg["p"], cfg["Lf"], cfg["d_over_pitch"], es.MASK_DIR, accuracy=EPS
    )
    ux, uy, p = load_state(args.state, Q)

    us, vs = dense_rk4_predictor(dense, ux, uy, p, dt, nu, C.mps_to_grid(mask_m, Q))
    b = dense.held(dense.divergence(us, vs))
    if args.mps_pred:
        us, vs, b = mps_predictor(ops, (ux, uy, p), mask_m, cfg, args.bd, (us, vs))
    return Problem(Q, dx, dt, args.bd, ops, dense, us, vs, b, dense.solve(b))


def load_state(path, Q):
    if path.endswith(".npz"):
        z = np.load(path)
        return z["ux"], z["uy"], z["p"]
    return tuple(C.mps_to_grid(f, Q) for f in tt.load_object(path))


def dense_rk4_predictor(S, ux, uy, p, dt, nu, mask):
    """RK4 predictor exactly as dense_projection_check.dense_step (rk4), before the projection."""

    def rhs(u, v):
        fx = -(u * (S.Dx @ u) + v * (S.Dy @ u)) + nu * S.laplacian(u) - S.Gx @ p
        fy = -(u * (S.Dx @ v) + v * (S.Dy @ v)) + nu * S.laplacian(v) - S.Gy @ p
        return S.held(fx), S.held(fy)

    def stage(c, k):
        return (ux + c * dt * k[0]) * mask, (uy + c * dt * k[1]) * mask

    k1 = rhs(ux, uy)
    k2 = rhs(*stage(0.5, k1))
    k3 = rhs(*stage(0.5, k2))
    k4 = rhs(*stage(1.0, k3))
    us = (ux + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])) * mask
    vs = (uy + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])) * mask
    return us, vs


def mps_predictor(ops, fields, mask_m, cfg, bd, dense_us_vs):
    """u*, v* and b from the MPS predictor at bond cap `bd`, as dense arrays."""
    Q = cfg["p"] + cfg["Lf"]
    tt.set_bd_limit(bd)
    ux, uy, p = (C.grid_to_mps(f, Q, EPS).compress(max_bd=bd) for f in fields)
    us_m, vs_m = C.predict_velocity(ux, uy, ops, cfg["dt"], cfg["nu"], mask_m, EPS, bd, p, rk4=True)
    with C.compression_limits(EPS, None):
        b_mps = ops.held(ops.divergence(us_m, vs_m))
    us_dense = dense_us_vs[0]
    error = np.linalg.norm(C.mps_to_grid(us_m, Q) - us_dense) / np.linalg.norm(us_dense)
    print(
        f"  MPS predictor: rank(u*)={int(us_m.max_rank)} rank(b)={int(b_mps.max_rank)}  "
        f"|u*_mps - u*_dense|/|u*| = {error:.2e}",
        flush=True,
    )
    return C.mps_to_grid(us_m, Q), C.mps_to_grid(vs_m, Q), C.mps_to_grid(b_mps, Q)


# --- reports ------------------------------------------------------------------------------------


def report_exact(pr, state):
    S, b = pr.dense, pr.b
    residual = np.linalg.norm(S.lap(pr.phi_ex) - b) / np.linalg.norm(b)
    after = pr.div_after(pr.us - S.Gx @ pr.phi_ex, pr.vs - S.Gy @ pr.phi_ex)
    before = np.linalg.norm(b) * pr.dx / pr.u_norm
    print(
        f"state {state}\n  |b|={np.linalg.norm(b):.4e}  exact: res={residual:.1e}"
        f"  div after={after:.1e}  div before={before:.3e}",
        flush=True,
    )


def report_ranks(pr, b_m, phi_ex_m):
    print(f"  rank(b) uncapped = {int(b_m.max_rank)}  rank(phi_ex) = {int(phi_ex_m.max_rank)}")
    for k in RANKS:
        bk = C.mps_to_grid(b_m.copy().compress(max_bd=k), pr.Q)
        print(f"    ||b - b_{k}||/||b|| = {np.linalg.norm(bk - pr.b) / np.linalg.norm(pr.b):.3e}")


def report_velocity_truncation(pr):
    """Floor set by the velocity cap: project exactly, then truncate the corrected velocity."""
    S = pr.dense
    un, vn = pr.us - S.Gx @ pr.phi_ex, pr.vs - S.Gy @ pr.phi_ex
    for k in (40, 80, 120, 200):
        uk = C.mps_to_grid(C.grid_to_mps(un, pr.Q, EPS).compress(max_bd=k), pr.Q)
        vk = C.mps_to_grid(C.grid_to_mps(vn, pr.Q, EPS).compress(max_bd=k), pr.Q)
        print(
            f"  exact phi, velocity trunc {k:3d}: div_after={pr.div_after(uk, vk):.2e}"
            f"  |u err|={np.linalg.norm(uk - un) / np.linalg.norm(un):.2e}",
            flush=True,
        )


# --- scoring ------------------------------------------------------------------------------------


def score_solvers(pr, b_m, phi_ex_m):
    """AMEn (last and best iterate) and two-site DMRG, from a cold and from the oracle start."""
    tt.set_bd_limit(pr.bd)
    starts = {
        "cold": tt.identity_mps(2 * pr.Q) * pr.dt,
        "oracle40": phi_ex_m.copy().compress(max_bd=pr.bd),
    }
    for guess, x0 in starts.items():
        for sweeps in (6, 12, 25):
            t0 = time.time()
            with C.pinned_accuracy(EPS):
                last, best = tt.linsys_amen(
                    A=pr.ops.lap,
                    b=b_m,
                    x=x0.copy(),
                    max_rank=pr.bd,
                    max_iter=sweeps,
                    eps=EPS,
                    save_best=True,
                )
            seconds = time.time() - t0
            score(pr, b_m, f"amen it={sweeps} {guess} last", last, seconds)
            score(pr, b_m, f"amen it={sweeps} {guess} best", best, seconds)
        t0 = time.time()
        with C.pinned_accuracy(EPS):
            sol = C._solve_poisson(pr.ops.lap, b_m, x0.copy(), EPS, pr.bd, 6, "dmrg2", None)
        score(pr, b_m, f"dmrg2 it=6 {guess}", sol, time.time() - t0)


def score(pr, b_m, name, phi_m, seconds):
    """Residuals and errors of one candidate phi against the exact solution."""
    S, phi_ex = pr.dense, pr.phi_ex
    phi = C.mps_to_grid(phi_m, pr.Q)

    def gradient_norm(f):
        return np.hypot(np.linalg.norm(S.Gx @ f), np.linalg.norm(S.Gy @ f))

    residual = np.linalg.norm(S.lap(phi) - pr.b) / np.linalg.norm(pr.b)
    phi_error = np.linalg.norm(phi - phi_ex) / np.linalg.norm(phi_ex)
    grad_error = gradient_norm(phi - phi_ex) / gradient_norm(phi_ex)
    div_after = pr.div_after(pr.us - S.Gx @ phi, pr.vs - S.Gy @ phi)
    with C.pinned_accuracy(EPS):  # the residual as project_divergence_free measures it
        tt.set_bd_limit(pr.bd)
        solver_residual = float((pr.ops.lap.apply(phi_m) + b_m * (-1.0)).norm) / float(b_m.norm)
    print(
        f"  {name:28s} rank={int(phi_m.max_rank):4d}  true_res={residual:.3f}"
        f"  prod_res={solver_residual:.3f}  phi_err={phi_error:.3e}  grad_err={grad_error:.3e}"
        f"  div_after={div_after:.2e}  {seconds:6.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    tt.set_arch("sequential")
    main()
