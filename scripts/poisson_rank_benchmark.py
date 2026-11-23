"""Poisson solvers scored against the EXACT phi, in dense (untruncated) norms.

For a state (ux, uy, p) builds the RK4 predictor densely, b = held(div u*),
phi_ex by the exact Sylvester solve (uncompressed_twin.SepOps), then scores:
  oracle  : phi_ex truncated to rank k (TT-SVD) -- the floor for a rank-k phi
  solvers : chorin_projection's Poisson call (and variants) at max_rank 40, BD_LIMIT 40
Metrics: true residual ||lap phi - b||/||b||, the residual as production
measures it (MPS residual truncated at BD_LIMIT), phi error, grad-phi error
(= error in the velocity correction), and |div|dx/|u| after the correction.

Run from scripts/.  --mps-pred forms b as production does (u* from
chorin_projection.predict_velocity truncated to --bd); --oracle-only skips the solvers.

    python3 poisson_rank_benchmark.py SNAPSHOT_OR_NPZ [--tag p1A_inc_it6] [--mps-pred]
"""
import argparse
import os
import time

import numpy as np
import tensnet as tt
import chorin_projection as C
import bundle_simulation as es
import uncompressed_twin as E

tt.set_arch("sequential")
a = argparse.ArgumentParser()
a.add_argument("state"); a.add_argument("--tag", default="p1A_inc_it6")
a.add_argument("--bd", type=int, default=40)
a.add_argument("--oracle-only", action="store_true", help="skip the MPS solvers")
a.add_argument("--mps-pred", action="store_true", help="u* from chorin_projection.predict_velocity(rk4) at max_bd, b in MPS uncapped: what production solves")
args = a.parse_args()
cfg = tt.load_object(os.path.join(es.RESULTS, f"{args.tag}_params.pfile"))
Q, dt, nu, dx = cfg["p"] + cfg["Lf"], cfg["dt"], cfg["nu"], cfg["dx"]
EPS, BD = 1e-12, args.bd
ops = C.Operators(Q, dx, dx, bcv=C.BCV_CHANNEL, bcp=C.BCP_CHANNEL)
S = E.SepOps(ops)
mask = C.mps_to_grid(es.load_or_build_mask(cfg["p"], cfg["Lf"], cfg["d_over_pitch"], es.MASK_DIR, accuracy=EPS), Q)
if args.state.endswith(".npz"):
    z = np.load(args.state); ux, uy, p = z["ux"], z["uy"], z["p"]
else:
    ux, uy, p = (C.mps_to_grid(f, Q) for f in tt.load_object(args.state))

# RK4 predictor exactly as dense_projection_check.dense_step (rk4), stopping before the projection
def f(u, v):
    fx = -(u * (S.Dx @ u) + v * (S.Dy @ u)) + nu * S.laplacian(u) - S.Gx @ p
    fy = -(u * (S.Dx @ v) + v * (S.Dy @ v)) + nu * S.laplacian(v) - S.Gy @ p
    return S.held(fx), S.held(fy)
at = lambda c, k: ((ux + c * dt * k[0]) * mask, (uy + c * dt * k[1]) * mask)
k1 = f(ux, uy); k2 = f(*at(.5, k1)); k3 = f(*at(.5, k2)); k4 = f(*at(1, k3))
us = (ux + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])) * mask
vs = (uy + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])) * mask
b = S.held(S.divergence(us, vs))
if args.mps_pred:
    tt.set_bd_limit(BD)
    m = [C.grid_to_mps(f, Q, EPS) for f in (ux, uy, p)]
    m = [x.compress(max_bd=BD) for x in m]
    mask_m = es.load_or_build_mask(cfg["p"], cfg["Lf"], cfg["d_over_pitch"], es.MASK_DIR, accuracy=EPS)
    us_m, vs_m = C.predict_velocity(m[0], m[1], ops, dt, nu, mask_m, EPS, BD, m[2], rk4=True)
    with C.compression_limits(EPS, None):
        b_mps = ops.held(ops.divergence(us_m, vs_m))
    print(f"  MPS predictor: rank(u*)={int(us_m.max_rank)} rank(b)={int(b_mps.max_rank)}  "
          f"|u*_mps - u*_dense|/|u*| = {np.linalg.norm(C.mps_to_grid(us_m, Q) - us) / np.linalg.norm(us):.2e}", flush=True)
    us, vs, b = C.mps_to_grid(us_m, Q), C.mps_to_grid(vs_m, Q), C.mps_to_grid(b_mps, Q)
phi_ex = S.solve(b)
gex = np.hypot(np.linalg.norm(S.Gx @ phi_ex), np.linalg.norm(S.Gy @ phi_ex))
nu_s = np.linalg.norm(us)
print(f"state {args.state}\n  |b|={np.linalg.norm(b):.4e}  exact: res={np.linalg.norm(S.lap(phi_ex) - b) / np.linalg.norm(b):.1e}"
      f"  div after={np.linalg.norm(S.held(S.divergence(us - S.Gx @ phi_ex, vs - S.Gy @ phi_ex))) * dx / nu_s:.1e}"
      f"  div before={np.linalg.norm(b) * dx / nu_s:.3e}", flush=True)

b_m = C.grid_to_mps(b, Q, EPS)
print(f"  rank(b) uncapped = {int(b_m.max_rank)}  rank(phi_ex) = {int(C.grid_to_mps(phi_ex, Q, EPS).max_rank)}")
for k in (25, 40, 80, 120, 200):
    bk = C.mps_to_grid(b_m.copy().compress(max_bd=k), Q)
    print(f"    ||b - b_{k}||/||b|| = {np.linalg.norm(bk - b) / np.linalg.norm(b):.3e}")


def score(name, phi_m, secs):
    ph = C.mps_to_grid(phi_m, Q)
    res = np.linalg.norm(S.lap(ph) - b) / np.linalg.norm(b)
    eph = np.linalg.norm(ph - phi_ex) / np.linalg.norm(phi_ex)
    eg = np.hypot(np.linalg.norm(S.Gx @ (ph - phi_ex)), np.linalg.norm(S.Gy @ (ph - phi_ex))) / gex
    dv = np.linalg.norm(S.held(S.divergence(us - S.Gx @ ph, vs - S.Gy @ ph))) * dx / nu_s
    with C.pinned_accuracy(EPS):  # the residual as chorin_projection.project_divergence_free measures it, under BD_LIMIT
        tt.set_bd_limit(BD)
        r40 = float((ops.lap.apply(phi_m) + b_m * (-1.0)).norm) / float(b_m.norm)
    print(f"  {name:28s} rank={int(phi_m.max_rank):4d}  true_res={res:.3f}  prod_res={r40:.3f}"
          f"  phi_err={eph:.3e}  grad_err={eg:.3e}  div_after={dv:.2e}  {secs:6.1f}s", flush=True)


phi_ex_m = C.grid_to_mps(phi_ex, Q, EPS)
for k in (25, 40, 80, 120, 200):
    score(f"oracle trunc(phi_ex,{k})", phi_ex_m.copy().compress(max_bd=k), 0)

# velocity-cap floor: exact projection, then the corrected velocity truncated to k
un, vn = us - S.Gx @ phi_ex, vs - S.Gy @ phi_ex
for k in (40, 80, 120, 200):
    uk = C.mps_to_grid(C.grid_to_mps(un, Q, EPS).compress(max_bd=k), Q)
    vk = C.mps_to_grid(C.grid_to_mps(vn, Q, EPS).compress(max_bd=k), Q)
    print(f"  exact phi, velocity trunc {k:3d}: div_after={np.linalg.norm(S.held(S.divergence(uk, vk))) * dx / nu_s:.2e}"
          f"  |u err|={np.linalg.norm(uk - un) / np.linalg.norm(un):.2e}", flush=True)
if args.oracle_only:
    raise SystemExit("DONE (oracle only)")
tt.set_bd_limit(BD)
x_cold = tt.identity_mps(2 * Q) * dt
x_orac = phi_ex_m.copy().compress(max_bd=BD)
for guess_name, x0 in (("cold", x_cold), ("oracle40", x_orac)):
    for it in (6, 12, 25):
        t0 = time.time()
        with C.pinned_accuracy(EPS):
            sol, best = tt.linsys_amen(A=ops.lap, b=b_m, x=x0.copy(), max_rank=BD, max_iter=it, eps=EPS, save_best=True)
        dtm = time.time() - t0
        score(f"amen it={it} {guess_name} last", sol, dtm)
        score(f"amen it={it} {guess_name} best", best, dtm)
    for it in (6,):
        t0 = time.time()
        with C.pinned_accuracy(EPS):
            sol = C._solve_poisson(ops.lap, b_m, x0.copy(), EPS, BD, it, "dmrg2", None)
        score(f"dmrg2 it={it} {guess_name}", sol, time.time() - t0)
print("DONE")
