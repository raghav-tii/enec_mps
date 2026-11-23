#!/usr/bin/env python3
"""ENEC rod-bundle simulations in MPS/QTT form.

The geometry is a 5x5 lattice of circular rods with the four corners removed
(21 rods), immersed in a square domain with inflow at x = ax, outflow at
x = bx and periodic side walls.

The mask is stored as a product of two MPS blocks: a coarse ``2**p x 2**p``
lattice carrying the rod *positions*, and a fine ``2**Lf x 2**Lf`` block
carrying a *single* rod.  Concatenating their nodes tiles the rod into every
flagged lattice cell, so the full fluid mask on a ``2**(p+Lf)`` grid is

    mask = 1 - (positions (x) rod)

Refining means raising ``Lf`` while holding ``p`` fixed: the rod pitch and
diameter stay put in physical units and only the grid gets finer, which is
what makes a resolution sweep a convergence study rather than a different
problem at every size.

Subcommands
-----------
    mask       build (or load) the mask MPS and cache it to disk
    warmstart  interpolate a Star-CCM+ VTK snapshot onto the grid
    run        time-step the flow and save one snapshot per step
    check      runnable self-check; needs no prior results

Lift and drag of the saved snapshots come from rod_forces.py.

Examples
--------
    python bundle_simulation.py run --lf 5 --re 100 --nt 200 --tag q10
    python rod_forces.py --tag q10 --stride 5
    python bundle_simulation.py check
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

import tensnet as tt

import chorin_projection

torch.set_default_dtype(torch.float64)

# tensnet keeps the qubit ordering and the finite-difference stencil in global
# state, and get_diff_mpo builds its MPO for whatever is set at the time.  This
# module always encodes fields with shuffle_axis, which only matches the
# operators under arch "sequential" -- measured: an analytic derivative comes
# back to 2e-13 with it and is off by a factor of ~40 without.  Setting it at
# import covers every entry point, including `post` run on its own, which used
# to build its stress operators under whatever the default was.
tt.set_arch("sequential")

tt.set_param("low_acc", False)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.environ.get("QCFD_DATA", os.path.join(_REPO, "data"))

RESULTS = os.environ.get("QCFD_RESULTS", os.path.join(DATA, "results"))  # snapshots in RESULTS/temp

MASK_DIR = os.path.join(DATA, "mps_masks")  # enec_mask_p*_f*_d*.mps cache

IC_DIR = os.path.join(DATA, "initial_conditions")  # Star-CCM+ warm starts

REFERENCE = os.path.join(DATA, "star_ccm")  # C_D.csv / C_L.csv monitors

ROD_GRID = (5, 5)  # coarse rod lattice, corners removed below

HALF_BLOCK = 0.15  # fine-block half width, in fine-block coordinates

D_OVER_PITCH = 2 / 3  # rod diameter / pitch; this repo's original geometry

REF_D_OVER_PITCH = 0.5  # the Star-CCM+ cyl_array reference (D=1, pitch=2)

R_REF, ALPHA = 0.1, 1e3  # indicator sharpness, calibrated at this radius

X0_FRAC, Y0_FRAC = 13 / 32, 6 / 32  # bundle corner, as a fraction of the domain (p <= 5)

UPSTREAM_PITCHES = 12.5  # p >= 6: inlet to bundle centre, 25D at pitch 2D as in Star-CCM+

# Boundary conditions used by tensnet's projection solver; post-processing has
# to rebuild the same derivative MPOs, so keep this next to the solver call.
BCV = ["dirichlet", "neumann", "periodic", "periodic"]

FACES = ("L", "R", "B", "T")

REF_U = 1.0  # the reference is non-dimensional: D = 1, U = 1, rho = 1

SNAP = re.compile(r"_c(\d+)_int_sol_(\d+)\.tmp$")


# ---- Command line ------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    if args.cmd == "warmstart":
        build_warm_starts(args)
    elif args.cmd == "mask":
        load_or_build_mask(args.p, args.lf, args.d_over_pitch, args.mask_dir, args.mask, args.eps)
    elif args.cmd == "run":
        return run_simulation(args)
    else:
        self_check(args)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def geometry(q):
        q.add_argument(
            "--p", type=int, default=5, help="coarse lattice qubits per axis; sets the rod pitch"
        )
        q.add_argument(
            "--lf", type=int, default=5, help="fine-block qubits per axis; refine this, not --p"
        )
        q.add_argument(
            "--d-over-pitch",
            type=float,
            default=D_OVER_PITCH,
            help=f"rod diameter / pitch (default {D_OVER_PITCH:.4f}; "
            f"the Star-CCM+ reference is {REF_D_OVER_PITCH})",
        )

    m = sub.add_parser("mask", help="build or load the mask MPS")
    geometry(m)
    m.add_argument("--mask", default=None, help="explicit mask file to load/write")
    m.add_argument("--mask-dir", default=MASK_DIR, help="directory for the mask cache")
    m.add_argument("--eps", type=float, default=1e-12)

    r = sub.add_parser("run", help="time-step the flow, then post-process")
    geometry(r)
    r.add_argument("--re", type=float, default=100.0, help="Reynolds number per rod")
    r.add_argument("--u", type=float, default=1.0, help="inflow velocity")
    r.add_argument(
        "--device",
        default="cpu",
        help="torch device for the time stepping, e.g. cuda; snapshots "
        "are always written from the CPU, so post-processing runs anywhere",
    )
    r.add_argument("--dt", type=float, default=None, help="override the CFL timestep")
    r.add_argument("--cfl", type=float, default=0.4)
    r.add_argument("--nt", type=int, default=200, help="number of steps")
    r.add_argument(
        "--t-end", type=float, default=None, help="physical end time; overrides --nt when given"
    )
    r.add_argument("--max-bd", type=int, default=40)
    r.add_argument("--eps", type=float, default=1e-12)
    r.add_argument("--opt-iter", type=int, default=6, help="DMRG sweeps per solve")
    r.add_argument(
        "--rhs-bd",
        type=int,
        default=0,
        help="bond-dimension cap for the Poisson right-hand side; "
        "0 means uncapped (accuracy-limited only), which is "
        "what keeps the cap from discarding a quarter of it",
    )
    r.add_argument(
        "--no-rotational",
        action="store_true",
        help="plain incremental pressure-correction instead of the rotational form",
    )
    r.add_argument(
        "--plain-pressure",
        action="store_true",
        help="plain Chorin: never feed the pressure (warm start "
        "or previous step) back into the predictor",
    )
    r.add_argument(
        "--rk4",
        action="store_true",
        help="RK4 predictor; forward Euler (the default, kept only "
        "to continue old runs) is unstable at CFL 0.4 here",
    )
    r.add_argument(
        "--phi-bd",
        type=int,
        default=0,
        help="rank cap for the pressure increment phi (0: same as "
        "--max-bd); the projection is rank-limited, not "
        "solver-limited, at chi=40",
    )
    r.add_argument(
        "--poisson-guard",
        type=float,
        default=1.0,
        help="abort when the Poisson relative residual reaches "
        "this; 1.0 means phi is no better than zero",
    )
    r.add_argument(
        "--solver",
        default=chorin_projection.POISSON_SOLVER,
        choices=["dmrg2", "amen"],
        help="Poisson solver; amen is cheaper per sweep but needs "
        "more sweeps for the same residual",
    )
    r.add_argument(
        "--local-solver",
        default=None,
        choices=["direct", "cg", "gmres", "matfree"],
        help="supercore solver inside each DMRG sweep (tensnet's default is the dense direct one)",
    )
    r.add_argument("--report", type=int, default=10, help="print |div u| every N steps; 0 disables")
    r.add_argument(
        "--perturb",
        type=float,
        default=1e-3,
        help="uniform cross-flow at t=0, as a fraction of U; "
        "0 keeps the exactly symmetric IC, which cannot shed",
    )
    r.add_argument(
        "--cv-half", type=float, default=0.45, help="control-volume half width, in rod pitches"
    )
    r.add_argument(
        "--warm-start",
        default=None,
        metavar="MPS",
        help="initial velocity from a file built by `warmstart`; "
        '"auto" resolves the standard name for this geometry',
    )
    r.add_argument(
        "--save-full",
        action="store_true",
        help="also write the whole run as one .mpssol; the per-step "
        "snapshots already hold the same data",
    )
    r.add_argument(
        "--restart",
        action="store_true",
        help="continue this tag from its last snapshot, in a new chunk",
    )
    r.add_argument("--tag", default="enec", help="run identifier used in filenames")
    r.add_argument("--mask", default=None)
    r.add_argument("--mask-dir", default=MASK_DIR)

    w = sub.add_parser(
        "warmstart",
        help="build initial-condition MPS files from a VTK snapshot",
        description="Run this where the VTK snapshots live. It "
        "writes one small .mps per resolution; pass one to "
        "`run --warm-start`.",
    )
    w.add_argument("--vtk", required=True, help="Star-CCM+ VTK snapshot")
    w.add_argument("--p", type=int, default=5)
    w.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[5, 6],
        help="fine-block qubit counts to build, one file each",
    )
    w.add_argument(
        "--d-over-pitch",
        type=float,
        default=REF_D_OVER_PITCH,
        help=f"default {REF_D_OVER_PITCH}, matching the reference",
    )
    w.add_argument("--max-bd", type=int, default=40)
    w.add_argument("--eps", type=float, default=1e-12)
    w.add_argument("--out-dir", default=IC_DIR)
    w.add_argument("--mask-dir", default=MASK_DIR)

    sub.add_parser("check", help="run the self-check")

    args = ap.parse_args(argv)
    if args.cmd == "run" and args.t_end is not None:
        args.nt = None
    return args


# ---- Simulation and warm start -----------------------------------------------
def run_simulation(args):
    Q = args.p + args.lf
    ax, bx, ay, by = 0.0, 1.0, 0.0, 1.0
    dx = (bx - ax) / (2**Q - 1)
    dy = (by - ay) / (2**Q - 1)

    pitch = rod_pitch(args.p, args.lf, ax, bx)
    D = args.d_over_pitch * pitch  # rod diameter, physical units
    nu = args.u * D / args.re

    # Explicit advection + diffusion in the predictor, so both limits bind.
    dt = args.dt or min(args.cfl * dx / args.u, 0.25 * dx**2 / nu)
    Nt = args.nt if args.nt is not None else max(1, int(round(args.t_end / dt)))
    if Nt < 1:
        raise SystemExit("--nt must be >= 1")

    tt.set_bd_limit(args.max_bd)
    tt.set_accuracy(args.eps)
    tt.set_arch("sequential")
    tt.set_comp_method("randomized_svd")
    for k, v in (
        ("ax", ax),
        ("bx", bx),
        ("Lx", Q),
        ("ay", ay),
        ("by", by),
        ("Ly", Q),
        ("ux_val", args.u),
        ("low_acc", False),
    ):
        tt.set_param(k, v)

    cfg = {
        "p": args.p,
        "Lf": args.lf,
        "Q": Q,
        "ax": ax,
        "bx": bx,
        "ay": ay,
        "by": by,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "Nt": Nt,
        "u": args.u,
        "nu": nu,
        "Re": args.re,
        "D": D,
        "pitch": pitch,
        "eps": args.eps,
        "max_bd": args.max_bd,
        "opt_iter": args.opt_iter,
        "cv_half": args.cv_half,
        "tag": args.tag,
        "perturb": args.perturb,
        "d_over_pitch": args.d_over_pitch,
        "local_solver": args.local_solver,
        "solver": args.solver,
        "warm_start": args.warm_start,
        "poisson_guard": args.poisson_guard,
        "plain_pressure": args.plain_pressure,
        "rk4": args.rk4,
        "held": True,
        "phi_bd": args.phi_bd,
    }
    shed = D / (0.16 * args.u)  # St ~ 0.16 for an isolated cylinder at Re~100
    print("Configuration:")
    for k, v in cfg.items():
        print(f"  {k:10s} {v}")
    print(f"  grid       {2**Q}^2   rod diameter {D / dx:.1f} cells")
    print(
        f"  t_end      {Nt * dt:.4e}   ~{Nt * dt / shed:.2f} shedding periods (T_shed ~ {shed:.3e})"
    )
    print(f"  CFL adv    {args.u * dt / dx:.3f}   CFL diff {nu * dt / dx**2:.3f}")

    # Snapshots go to results/temp/, matching where the earlier
    # tensnet-driven runs put them so post-processing is unchanged.
    os.makedirs(os.path.join(RESULTS, "temp"), exist_ok=True)
    params = os.path.join(RESULTS, f"{args.tag}_params.pfile")
    if args.restart and os.path.exists(params):
        # post-processing reads the time axis off dt and the snapshot count,
        # so a restart that quietly changed the physics would corrupt it.
        prev = tt.load_object(params)
        drift = {
            k: (prev.get(k, False), cfg.get(k, False))
            for k in ("p", "Lf", "dt", "nu", "u", "Re", "rk4", "held")
            if prev.get(k, False) != cfg.get(k, False)
        }
        if drift:
            raise SystemExit(f"--restart changes the run: {drift}. Use a new --tag.")
    tt.save_object(cfg, params)

    mask = load_or_build_mask(
        args.p, args.lf, args.d_over_pitch, args.mask_dir, args.mask, accuracy=args.eps
    )
    # the mask is built (or loaded) on the CPU -- tensnet's builders are not
    # device-clean -- and everything from here on lands on args.device
    tt.set_device(args.device)
    mask = mask.to(device=args.device)

    def on_device(*fs):
        return [f if f is None else f.to(device=args.device) for f in fs]

    done = find_snapshots(args.tag)
    if done and not args.restart:
        raise SystemExit(
            f"{len(done)} snapshots already exist for tag '{args.tag}'. "
            f"Pass --restart to continue from the last one, or use a new --tag."
        )

    chunk = 0
    p0 = None
    if done:
        chunk = done[-1][0] + 1
        ux, uy, p0 = on_device(*tt.load_object(done[-1][2]))
        print(
            f"Restarting from {done[-1][2]} "
            f"({len(done)} steps done, t={len(done) * dt:.5e}) as chunk {chunk}"
        )
    elif args.warm_start:
        ic = args.warm_start
        if ic == "auto":
            ic = warm_start_path(args.p, args.lf, args.d_over_pitch)
        ux, uy, p0 = on_device(
            *load_warm_start(ic, args.p, args.lf, args.d_over_pitch, args.u, args.max_bd)
        )
        # One cleanup projection, phi discarded.  The reference solves a
        # different discretisation on a different mesh in a different
        # domain, so its field is not divergence-free here -- measured
        # |div u| dx/|u| = 2.1e-3 straight off the interpolator.  Left in,
        # the first real step's phi has to absorb all of it and p = phi/dt
        # amplifies that by 1/dt, which is what put the pressure 50x out.
        # The discarded phi is a projection artefact, not a pressure, so p0
        # stays the reference field.
        ops0 = chorin_projection.Operators(
            Q,
            dx,
            dy,
            bcv=chorin_projection.BCV_CHANNEL,
            bcp=chorin_projection.BCP_CHANNEL,
            held=cfg["held"],
            inflow=args.u,
        )
        d_before = float(ops0.held(ops0.divergence(ux, uy)).norm) * dx / float(ux.norm)
        ux, uy, _, _ = chorin_projection.project_divergence_free(
            ux, uy, ops0, dt, eps=args.eps, max_bd=args.max_bd, opt_iter=max(args.opt_iter, 12)
        )
        d_after = float(ops0.held(ops0.divergence(ux, uy)).norm) * dx / float(ux.norm)
        print(
            f"Cleaned the warm start: |div u|dx/|u| {d_before:.3e} -> "
            f"{d_after:.3e} (phi discarded, pressure kept)"
        )
    else:
        ux = mask * args.u
        uy = initial_uy(mask, Q, args.u, args.perturb)
    print(f"Initial ranks: ux={int(ux.max_rank)}, uy={int(uy.max_rank)}")

    # The time stepping is chorin_projection.march_in_time.  Its module docstring
    # explains the three pitfalls of a naive projection that it avoids.
    ops = chorin_projection.Operators(
        Q,
        dx,
        dy,
        bcv=chorin_projection.BCV_CHANNEL,
        bcp=chorin_projection.BCP_CHANNEL,
        held=cfg["held"],
        inflow=args.u,
    )
    local_solver = tt.make_local_solver(args.local_solver) if args.local_solver else None
    try:
        solution = chorin_projection.march_in_time(
            ux,
            uy,
            ops,
            dt,
            nu,
            Nt,
            mask=mask,
            eps=args.eps,
            max_bd=args.max_bd,
            opt_iter=args.opt_iter,
            uid=f"{args.tag}_c{chunk:03d}",
            outdir=os.path.join(RESULTS, "temp"),
            local_solver=local_solver,
            report=args.report,
            solver=args.solver,
            pressure=None if args.plain_pressure else p0,
            rotational=not args.no_rotational,
            rhs_bd=args.rhs_bd or None,
            guard=args.poisson_guard,
            plain=args.plain_pressure,
            rk4=args.rk4,
            phi_bd=args.phi_bd or None,
        )
    except Exception as exc:
        # Either the projection stopped working (march says so explicitly) or
        # the explicit predictor went non-finite inside a compression SVD.
        kept = find_snapshots(args.tag)
        print(
            f"\nTime stepping FAILED after {len(kept)} saved steps "
            f"(t={len(kept) * dt:.4e} of {Nt * dt:.4e}): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            f"Lower --cfl (currently {args.u * dt / dx:.3f}), raise "
            "--opt-iter so the Poisson solve converges, or raise --max-bd. "
            "Snapshots up to the failure are kept.",
            file=sys.stderr,
        )
        return 1
    # savetmp already wrote every step to results/temp/, so the monolithic
    # .mpssol is a second copy of the same data -- at 1200 steps that is
    # several GB of pure duplication.  Opt in when a single file is wanted.
    if args.save_full:
        out = os.path.join(RESULTS, f"Re_{int(args.re)}_{args.tag}_c{chunk:03d}.mpssol")
        tt.save_object(solution, out)
        print(f"Saved solution -> {out}")

    print(f"Per-rod forces: python3 rod_forces.py --tag {args.tag}")


def build_warm_starts(args):
    """One initial-condition file per resolution."""
    written = []
    for Lf in args.sizes:
        mask = load_or_build_mask(args.p, Lf, args.d_over_pitch, args.mask_dir, None, args.eps)
        print(f"Building warm start for p={args.p}, Lf={Lf} (grid {2 ** (args.p + Lf)}^2):")
        ux, uy, pr = build_warm_start(
            args.vtk, args.p, Lf, args.d_over_pitch, mask, args.eps, args.max_bd
        )
        path = warm_start_path(args.p, Lf, args.d_over_pitch, args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)
        tt.save_object(
            {
                "ux": ux,
                "uy": uy,
                "pressure": pr,
                "p": args.p,
                "Lf": Lf,
                "d_over_pitch": args.d_over_pitch,
                "max_bd": args.max_bd,
                "source": os.path.abspath(args.vtk),
            },
            path,
        )
        size = os.path.getsize(path) / 1e6
        print(f"  wrote {path} ({size:.1f} MB)")
        written.append(path)
    print("\nRun with --warm-start <file>:")
    for path in written:
        print(f"  {path}")


def self_check(_args=None):
    """One runnable check over the pieces that can actually be wrong."""
    # 1. Shuffled-index mapping agrees with tensnet's own shuffle.
    L = 3
    a = torch.arange(4**L).reshape(2**L, 2**L).double()
    s = tt.shuffle_axis(a.flatten(), L).flatten()
    ix, iy = np.meshgrid(np.arange(2**L), np.arange(2**L), indexing="ij")
    got = s[shuffled_index(ix, iy, L).ravel().tolist()].reshape(2**L, 2**L)
    assert torch.allclose(got, a), "shuffled_index disagrees with tt.shuffle_axis"

    # 2. The mask MPS agrees with the mask evaluated directly on the grid.
    acc = 1e-10
    built = tt.unshuffle_axis(build_mask(5, 5, accuracy=acc).contract(), 10, 2)
    xb = torch.linspace(-HALF_BLOCK, HALF_BLOCK, 32)
    xb, yb = torch.meshgrid(xb, xb, indexing="ij")
    rod = cylinder(xb, yb, 0.0, 0.0, rod_radius(D_OVER_PITCH))
    ix0, iy0 = lattice_origin(5)
    pad = torch.from_numpy(np.pad(rod_positions(), ((ix0, 32 - ix0 - 5), (iy0, 32 - iy0 - 5))))
    # Tile the rod into every flagged lattice cell, directly on the grid.
    gx, gy = torch.meshgrid(torch.arange(1024), torch.arange(1024), indexing="ij")
    want = 1.0 - pad[gx // 32, gy // 32] * rod[gx % 32, gy % 32]
    err = float((built - want).abs().max())
    assert err < 100 * acc, f"mask MPS differs from the dense mask by {err:.2e}"
    assert int(pad.sum()) == 21, "expected 21 rods"
    # The CV boxes must land on the rods the mask actually contains.
    for cx, cy in rod_centres(5, 5):
        assert want[int(cx), int(cy)] < 1e-9, "rod centre is not inside a rod"
    # From p = 6 the bundle sits as in Star-CCM+: centre 12.5 pitches (25D)
    # from the inlet, ~100D of wake behind it, every control volume inside.
    c6 = np.array(rod_centres(6, 5)) / 2**5
    assert abs(c6[:, 0].mean() - 12.5) < 0.05, f"p=6 bundle centre at {c6[:, 0].mean():.2f} pitches"
    assert 2**6 - c6[:, 0].mean() > 50, "p=6 leaves less than 100D of wake"
    assert len(control_volumes(6, 5, d_over_pitch=REF_D_OVER_PITCH)[0]) == 21
    print(f"  mask MPS matches the dense mask (max err {err:.2e}, 21 rods)")

    # 3. Control-volume force on a hydrostatic field p = -G x, u = v = 0.
    #    sigma_xx = sigma_yy = G x, sigma_xy = 0, so the only surviving term is
    #    the x-face pressure difference: Fx = G * (2h dx) * ((2h+1) dy), Fy = 0.
    p, Lf, half_frac = 5, 5, 0.45
    rods, h = control_volumes(p, Lf, half_frac)
    Q, dx = p + Lf, 1.0 / (2 ** (p + Lf) - 1)
    dy, G = dx, 3.7
    r = rods[0]
    xs = np.array([_unshuffle_x(i, Q) for i in r["B"]])
    xl, xr = xs.min(), xs.max()
    ny = len(r["L"])
    zero = {f: np.zeros(len(r[f])) for f in FACES}
    sxx = {
        "L": np.full(ny, G * xl * dx),
        "R": np.full(ny, G * xr * dx),
        "B": G * xs * dx,
        "T": G * xs * dx,
    }
    fx, fy = control_volume_forces(zero, zero, sxx, zero, dict(sxx), dx, dy)
    expect = G * (xr - xl) * dx * ny * dy
    assert abs(fx - expect) < 1e-12 * abs(expect), f"CV Fx {fx} != {expect}"
    assert abs(fy) < 1e-12 * abs(expect), f"CV Fy {fy} != 0"
    assert xr - xl == 2 * h - 1, f"CV width {xr - xl} != {2 * h - 1}"
    print(f"  CV force on p=-Gx field exact ({len(rods)} rods, h={h} cells)")

    # 4. The IC is symmetric without a kick, and asymmetric with one.
    #    This is the whole reason initial_uy exists -- if it ever silently
    #    stops breaking the symmetry, no run will ever shed.
    n = want.shape[1]
    cys = [cy for _, cy in rod_centres(5, 5)]
    j = (int(round(cys[0] + cys[-1])) - np.arange(n)) % n  # reflect y, periodic
    assert float((want - want[:, j]).abs().max()) < 1e-11, (
        "mask is not y-symmetric; initial_uy's premise no longer holds"
    )

    mask_mps = build_mask(5, 5, accuracy=acc)
    assert float(initial_uy(mask_mps, 10, 1.0, 0.0).norm) == 0.0, (
        "perturb=0 must give an exactly zero uy"
    )
    kick = tt.unshuffle_axis(initial_uy(mask_mps, 10, 1.0, 1e-3).contract(), 10, 2)
    # The mirror-symmetric state needs v to be ODD: v(x, mirror(y)) = -v(x,y).
    # A constant v is even instead, so it violates that by 2x its amplitude
    # everywhere outside the rods -- that violation IS the symmetry break.
    peak = float(kick.abs().max())
    break_amp = float((kick[:, j] + kick).abs().max())
    assert abs(peak - 1e-3) < 1e-9, f"kick peak {peak:.3e} != the requested 1e-3"
    assert abs(break_amp - 2e-3) < 1e-6, (
        f"kick violates the odd-mode condition by {break_amp:.3e}, expected 2e-3"
    )
    inside = want < 0.5  # hard rod region, not the smooth interface
    assert float(kick[inside].abs().max()) < 1e-8, "the kick leaks inside the rods"
    print(
        f"  IC kick breaks the mirror symmetry (peak {peak:.2e}, "
        f"odd-mode violation {break_amp:.2e}, rank 1)"
    )

    # 5. Rod naming and the warm-start coordinate map.  The map is what
    #    silently ruins a warm start, so pin it: every rod centre of ours must
    #    land exactly on a reference rod centre, which sit on even integers.
    names = rod_names(5, 5)
    want_names = {
        f"{r}{c}"
        for r in range(1, 6)
        for c in range(1, 6)
        if (r, c) not in {(1, 1), (1, 5), (5, 1), (5, 5)}
    }
    assert set(names) == want_names, f"rod names {sorted(set(names))}"
    assert len(names) == len(set(names)) == 21

    n = 2 ** (5 + 5)
    ccx, ccy = bundle_centre(5, 5)
    scale = 2.0 / rod_pitch(5, 5)  # our length -> reference D units
    got = sorted(
        (round((gx / (n - 1) - ccx) * scale, 6), round((gy / (n - 1) - ccy) * scale, 6))
        for gx, gy in rod_centres(5, 5)
    )
    want = sorted(
        (x, y)
        for x in (-4.0, -2.0, 0.0, 2.0, 4.0)
        for y in (-4.0, -2.0, 0.0, 2.0, 4.0)
        if abs(x) + abs(y) != 8
    )
    assert got == want, f"warm-start map misplaces rods:\n{got}\nvs\n{want}"
    print("  rod names match the 21 monitors; warm-start map lands rods on the reference lattice")

    # 6. The derivative MPO has to match the encoding we hand it.  This is the
    #    check that catches a wrong tt.set_arch: without "sequential" the
    #    relative error below is O(10), not O(1e-13).
    Qd = 8
    nd = 2**Qd
    dxd = 1.0 / (nd - 1)
    gd = np.linspace(0, 1, nd)
    Xd, Yd = np.meshgrid(gd, gd, indexing="ij")
    fd = np.sin(2 * np.pi * Xd) * np.cos(4 * np.pi * Yd)
    ref_x = 2 * np.pi * np.cos(2 * np.pi * Xd) * np.cos(4 * np.pi * Yd)
    ref_y = -4 * np.pi * np.sin(2 * np.pi * Xd) * np.sin(4 * np.pi * Yd)
    md = tt.vector_to_mps(tt.shuffle_axis(torch.from_numpy(fd), Qd).flatten(), 2, 2 * Qd, 1e-14)
    core = slice(8, -8)  # the edges use the BC stencils, not the interior one
    for axis, want in ((0, ref_x), (1, ref_y)):
        D = tt.get_diff_mpo(dim=2, order=1, axis=axis, nofq=[Qd, Qd], bc=BCV)
        got = tt.unshuffle_axis((D @ md).contract(), Qd, 2).numpy() / dxd
        rel = np.abs(got[core, core] - want[core, core]).max() / np.abs(want).max()
        assert rel < 1e-10, (
            f"d/d{'xy'[axis]} MPO disagrees with the analytic derivative by "
            f"{rel:.2e}; the MPS encoding and tt.set_arch do not match"
        )
    print("  derivative MPOs match the shuffle_axis encoding")

    # 7. Warm-start files round-trip, rescale by u, and refuse a geometry
    #    they were not built for -- nothing else about them is trusted.
    import tempfile

    base = tt.identity_mps(L=10) * 0.25
    with tempfile.TemporaryDirectory() as tmp:
        wpath = warm_start_path(5, 5, REF_D_OVER_PITCH, tmp)
        tt.save_object(
            {
                "ux": base,
                "uy": base * 0.0,
                "p": 5,
                "Lf": 5,
                "d_over_pitch": REF_D_OVER_PITCH,
                "source": "selfcheck",
            },
            wpath,
        )
        wux, _, _ = load_warm_start(wpath, 5, 5, REF_D_OVER_PITCH, 2.0, 40)
        assert float((wux - base * 2.0).norm) < 1e-9, "warm start lost its scaling"
        for bad in ((5, 6, REF_D_OVER_PITCH), (5, 5, D_OVER_PITCH)):
            try:
                load_warm_start(wpath, *bad, 1.0, 40)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"warm start accepted wrong geometry {bad}")
        try:
            load_warm_start(os.path.join(tmp, "nope.mps"), 5, 5, REF_D_OVER_PITCH, 1.0, 40)
        except SystemExit:
            pass
        else:
            raise AssertionError("missing warm start was not reported")
    print("  warm-start files round-trip and reject the wrong geometry")

    print("check: OK")


def _unshuffle_x(flat, Q):
    """Recover ix from a shuffled flat index (self-check helper)."""
    return int("".join(str((int(flat) >> k) & 1) for k in range(2 * Q - 1, 0, -2)), 2)


def initial_uy(mask, Q, u, perturb):
    """Cross-stream velocity at t=0, with the symmetry break the wake needs.

    The bundle is exactly mirror-symmetric in y about its own centreline
    (verified to 5e-14 on the 1024^2 grid), y is periodic, and ux = U*mask,
    uy = 0 is symmetric too.  The equations preserve that symmetry, so a
    deterministic solver started this way never sheds -- Cl stays pinned at
    zero and no amount of extra timesteps produces an oscillation.  Real
    codes break the symmetry with roundoff; here the only accidental source
    is randomized_svd truncation noise, which is uncontrolled and
    unrepeatable.  So break it on purpose.

    The kick is a uniform cross-flow, i.e. the free stream at a tiny angle of
    attack (--perturb 1e-3 is 0.06 degrees).  A constant v is odd under the
    mirror, so it breaks the symmetry exactly as a random field would, but it
    is rank 1, needs no seed, and -- the reason it is not random noise -- its
    peak equals its RMS.  A random MPS peaks at ~50x its RMS, and feeding
    isolated 5%-of-U spikes to an explicit predictor is a good way to trip the
    very instability the CFL limit exists to avoid.
    """
    return mask * (perturb * u)


def load_warm_start(path, p, Lf, d_over_pitch, u, max_bd):
    """Load an initial condition built by the `warmstart` subcommand.

    A run only ever loads the resulting small .mps file; it never needs the VTK.
    The stored geometry is checked rather than trusted -- an IC built for a
    different grid or a different bundle would otherwise start a run that
    looks fine and means nothing.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"No warm-start file at {path}. Build it with:\n"
            f"  python3 bundle_simulation.py warmstart --vtk <snapshot.vtk> "
            f"--p {p} --sizes {Lf} --d-over-pitch {d_over_pitch}\n"
            f"(it is written to data/initial_conditions; `make warmstart` does this for "
            f"the report grids)."
        )
    obj = tt.load_object(path)
    if not isinstance(obj, dict) or "ux" not in obj:
        raise SystemExit(
            f"{path} is not a warm-start file. Build one with:\n"
            f"  python3 bundle_simulation.py warmstart --p {p} --lf {Lf} "
            f"--d-over-pitch {d_over_pitch} --vtk <snapshot.vtk>"
        )
    want = {"p": p, "Lf": Lf, "d_over_pitch": d_over_pitch}
    got = {k: obj[k] for k in want}
    if got != want:
        raise SystemExit(f"{path} was built for {got}, this run is {want}")
    print(
        f"Warm start loaded from {path} (built from "
        f"{os.path.basename(obj.get('source', '?'))}, U=1 scaled to u={u})"
    )
    ux = (obj["ux"] * u).compress(max_bd=max_bd)
    uy = (obj["uy"] * u).compress(max_bd=max_bd)
    # Pressure scales as u^2 (rho = 1).  Files written before pressure was
    # stored simply have none; the run then builds it over the first step
    # instead of starting from the reference.
    pr = obj.get("pressure")
    if pr is not None:
        pr = (pr * u**2).compress(max_bd=max_bd)
    print(
        f"  ranks: ux={int(ux.max_rank)}, uy={int(uy.max_rank)}, "
        f"p={'none (old file)' if pr is None else int(pr.max_rank)}"
    )
    return ux, uy, pr


def build_warm_start(vtk, p, Lf, d_over_pitch, mask, eps, max_bd, ax=0.0, bx=1.0, ay=0.0, by=1.0):
    """Initial velocity sampled from a Star-CCM+ VTK snapshot, at U = 1.

    The reference case is the same 21-rod bundle but non-dimensionalised on
    its own scales (D = 1, pitch = 2, U = 1) in a 100D x 80D far-field box,
    while this solver runs a 64D y-periodic box.  Only the pitch ratio is
    matched (pass --d-over-pitch 0.5); the outer field is NOT, so the warm
    start reproduces the reference near the rods and then relaxes towards
    this domain's own boundary conditions.

    Nearest neighbour, not linear.  Measured against the reference's own 21
    monitors it is as accurate (Cd correlation 0.9984 against 0.9982), four
    times faster, and gives a *lower* rank field (251 against 365) because
    the reference mesh is finer than this grid wherever it matters.  It also
    needs no separate handling for the ~20% of our box that reaches past
    their mesh: nearest extends the field continuously instead of leaving
    NaN to patch.
    """
    from scipy.interpolate import NearestNDInterpolator

    from star_vtk import read_star_vtk

    Q = p + Lf
    if Q > 12:
        raise SystemExit(
            f"warm start needs a dense {2**Q}^2 grid ({4**Q:,} points) before "
            f"it can be compressed; Q <= 12 only. Build the coarse sizes and "
            f"prolongate, or use cross approximation."
        )
    centres, cells = read_star_vtk(vtk)
    # Pressure too: without it the solver rebuilds p from nothing each step
    # and, as plain Chorin, amplifies the Poisson error by 1/dt.  Measured
    # against this very field, that put the computed pressure 5x to 94x too
    # high with no trend over fourteen steps.
    vel = np.column_stack([cells["Velocity"][:, :2], cells["Pressure"]])

    # our physical coordinates -> reference coordinates (D = 1, pitch = 2)
    scale = 2.0 / rod_pitch(p, Lf, ax, bx)
    cx, cy = bundle_centre(p, Lf, ax, bx, ay, by)
    n = 2**Q
    gx = (np.linspace(ax, bx, n) - cx) * scale
    gy = (np.linspace(ay, by, n) - cy) * scale
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    print(
        f"  our box is x=[{gx[0]:.1f},{gx[-1]:.1f}] y=[{gy[0]:.1f},{gy[-1]:.1f}] "
        f"in D; reference mesh is "
        f"x=[{centres[:, 0].min():.1f},{centres[:, 0].max():.1f}] "
        f"y=[{centres[:, 1].min():.1f},{centres[:, 1].max():.1f}]"
    )

    V = NearestNDInterpolator(centres, vel)(np.column_stack([X.ravel(), Y.ravel()]))

    saved = tt.COMP_ACC()
    tt.set_accuracy(eps)
    try:
        out = []
        for comp in (0, 1, 2):
            field = torch.from_numpy(V[:, comp].reshape(n, n).copy())
            m = tt.vector_to_mps(tt.shuffle_axis(field, Q).flatten(), 2, 2 * Q, eps)
            # Mask BEFORE truncating: compressing the unmasked field first and
            # multiplying after leaves ~1e-2 of U moving inside the rods,
            # because the product's rank is cut back down to max_bd anyway.
            # Pressure is NOT masked -- it is defined inside the rods too and
            # zeroing it there would put a spurious jump on every surface.
            m = m * mask if comp < 2 else m
            out.append(m.compress(max_bd=max_bd, accuracy=eps))
    finally:
        tt.set_accuracy(saved)
    print(
        f"  ranks: ux={int(out[0].max_rank)}, uy={int(out[1].max_rank)}, p={int(out[2].max_rank)}"
    )
    return out[0], out[1], out[2]


def warm_start_path(p, Lf, d_over_pitch, out_dir=IC_DIR):
    return os.path.join(out_dir, f"enec_ic_p{p}_f{Lf}_d{round(d_over_pitch * 1000)}.mps")


# ---- Saved runs and the Star-CCM+ reference ----------------------------------
def find_snapshots(tag, results=RESULTS):
    """Every snapshot for a tag, in time order, across restart chunks.

    A long run does not fit one walltime, so `run --restart` continues in a
    fresh chunk whose UID is <tag>_c<NNN>; tensnet restarts its own step
    counter at 0 each time, so ordering is (chunk, step) and the global step
    is the position in this list.  Both globs are anchored on "_c<digits>_" /
    "_int_sol_" so a tag never picks up a longer sibling tag's files.
    """
    found = []
    for path in glob.glob(os.path.join(results, "temp", f"{tag}_c*_int_sol_*.tmp")):
        m = SNAP.search(path)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), path))
    for path in glob.glob(os.path.join(results, "temp", f"{tag}_int_sol_*.tmp")):
        m = re.search(r"_int_sol_(\d+)\.tmp$", path)  # pre-chunk layout
        if m:
            found.append((-1, int(m.group(1)), path))
    return sorted(found)


def rod_names(p, Lf):
    """Star-CCM+ monitor suffix for each rod, in this module's rod order.

    Their names are <row><col> with the row counted across y (row 3 is the
    centreline, which carries Cl ~ 0, and rows 1/5 and 2/4 are mirror pairs)
    and the column across x (column 1 is the upstream face, Cd ~ 1.03-1.20).
    Our own index is (a, b) = (x, y), so the digits come out swapped.
    """
    return [f"{b + 1}{a + 1}" for a, b in np.argwhere(rod_positions() == 1)]


def reference_coefficients(path, frac=0.2):
    """Steady per-rod (Cd, Cl) from the reference C_D.csv / C_L.csv."""
    import csv

    def load(name):
        rows = list(csv.reader(Path(os.path.join(path, name)).read_text().splitlines()))
        head = [c.split(" Monitor")[0].split("_")[-1] for c in rows[0][1:]]
        arr = np.array([[float(x) for x in r] for r in rows[1:]])
        tail = arr[int((1 - frac) * len(arr)) :, 1:]
        return dict(zip(head, tail.mean(axis=0)))

    return load("C_D.csv"), load("C_L.csv")


# ---- Rod-bundle geometry -----------------------------------------------------
def build_mask(p, Lf, d_over_pitch=D_OVER_PITCH, accuracy=1e-12):
    """Build the fluid mask MPS (1 in the fluid, 0 inside the rods).

    tensnet's MPS arithmetic compresses against the *global* accuracy, not the
    one handed to ``compress``, so set it for the duration of the build and put
    it back -- otherwise a caller sitting at the 1e-6 default silently gets a
    mask that is off by ~1e-5 with no warning.
    """
    saved = tt.COMP_ACC()
    tt.set_accuracy(accuracy)
    try:
        return _build_mask(p, Lf, d_over_pitch, accuracy)
    finally:
        tt.set_accuracy(saved)


def load_or_build_mask(
    p, Lf, d_over_pitch=D_OVER_PITCH, cache_dir=MASK_DIR, path=None, accuracy=1e-12
):
    """Load a cached mask if one matches this geometry, otherwise build it.

    The cache key carries ``p`` and ``Lf`` separately on purpose: a name keyed
    only on the total qubit count collides between geometries (p=5,Lf=8 and
    p=8,Lf=5 are both 26 qubits but different bundles), which would silently
    run the wrong problem.
    """
    # The diameter/pitch ratio is in the name too: two bundles can share p
    # and Lf and still be different geometries.
    path = path or os.path.join(
        cache_dir, f"enec_mask_p{p}_f{Lf}_d{round(d_over_pitch * 1000)}.mps"
    )
    if os.path.exists(path):
        mask = tt.load_object(path)
        if len(mask) != 2 * (p + Lf):
            raise ValueError(
                f"{path} has {len(mask)} sites, expected {2 * (p + Lf)} for p={p}, Lf={Lf}"
            )
        print(f"Loaded mask from {path} (rank {int(mask.max_rank)})")
        return mask.compress(accuracy=accuracy)

    print(f"No mask at {path}; building it (p={p}, Lf={Lf}, D/pitch={d_over_pitch:.4f})")
    mask = build_mask(p, Lf, d_over_pitch, accuracy=accuracy)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tt.save_object(mask, path)
    print(f"Built and cached mask (rank {int(mask.max_rank)}) -> {path}")
    return mask


def _build_mask(p, Lf, d_over_pitch, accuracy):
    xb = torch.linspace(-HALF_BLOCK, HALF_BLOCK, 2**Lf)
    xb, yb = torch.meshgrid(xb, xb, indexing="ij")
    rod = tt.vector_to_mps(
        tt.shuffle_axis(cylinder(xb, yb, 0.0, 0.0, rod_radius(d_over_pitch)), Lf).flatten(),
        2,
        2 * Lf,
        accuracy,
    ).compress(accuracy=accuracy)

    pos = rod_positions()
    ix0, iy0 = lattice_origin(p)
    pad = np.pad(
        pos,
        ((ix0, 2**p - ix0 - pos.shape[0]), (iy0, 2**p - iy0 - pos.shape[1])),
    )
    lattice = tt.vector_to_mps(
        tt.shuffle_axis(torch.from_numpy(pad), p).flatten(), 2, 2 * p, accuracy
    )

    mask = tt.identity_mps(2 * (p + Lf)) - tt.MPS(lattice.nodes + rod.nodes)
    return mask.compress(accuracy=accuracy)


def cylinder(x, y, xc, yc, r, alpha=None):
    """Smooth indicator of a disc: ~1 inside, decaying to 0 outside.

    ``v + |v|`` clamps the interior to exactly zero, so the exponential is
    flat at 1 inside the rod and only the outside is smoothed.  The interface
    width is ~1/(4*alpha*r) in block coordinates -- a fixed physical length,
    independent of Lf, so refining resolves the same interface instead of
    sharpening it.
    """
    # Scale alpha with 1/r so the interface stays the same physical width
    # whatever the diameter/pitch ratio is.
    alpha = ALPHA * R_REF / r if alpha is None else alpha
    v = (x - xc) ** 2 + (y - yc) ** 2 - r**2
    v = v + torch.abs(v)
    return torch.exp(-alpha * v)


def rod_positions():
    """Coarse occupancy lattice: 1 where a rod sits, 0 elsewhere."""
    pos = np.ones(ROD_GRID)
    pos[0, 0] = pos[0, -1] = pos[-1, 0] = pos[-1, -1] = 0.0
    return pos


def rod_radius(d_over_pitch):
    """Rod radius in fine-block coordinates, from the diameter/pitch ratio."""
    return d_over_pitch * HALF_BLOCK


def lattice_origin(p):
    """Index of the bundle's lower-left coarse cell on a 2**p lattice.

    p <= 5 is the half-domain placement of the q10 run (bundle centre
    31D from the inlet of a 64D box, 33D of wake).  From p = 6 (a 128D box at
    pitch 2D) the bundle sits where Star-CCM+ has it, its centre 25D from the
    inlet, leaving ~100D of wake, and is centred across the periodic width.
    """
    if p <= 5:
        return round(X0_FRAC * 2**p), round(Y0_FRAC * 2**p)
    return round(UPSTREAM_PITCHES - ROD_GRID[0] / 2), 2 ** (p - 1) - ROD_GRID[1] // 2


def rod_centres(p, Lf):
    """Grid coordinates (in cells) of every rod centre."""
    n = 2**Lf
    ix0, iy0 = lattice_origin(p)
    return [
        ((ix0 + a) * n + (n - 1) / 2, (iy0 + b) * n + (n - 1) / 2)
        for a, b in np.argwhere(rod_positions() == 1)
    ]


def rod_pitch(p, Lf, ax=0.0, bx=1.0):
    """Physical centre-to-centre rod spacing.

    NOT (bx-ax)/2**p.  The grid is linspace(ax, bx, 2**Q), so it has 2**Q
    points but only 2**Q - 1 intervals, while the mask tiles 2**p blocks of
    exactly 2**Lf cells.  The rods therefore sit 2**Lf cells apart, which is
    2**Lf/(2**Q - 1) of the domain -- larger than 1/2**p by a factor
    2**Q/(2**Q - 1).  It is only 0.1% at Q=10, but it is the difference
    between a warm start landing on the reference rods and missing them.
    """
    return 2**Lf / (2 ** (p + Lf) - 1) * (bx - ax)


def bundle_centre(p, Lf, ax=0.0, bx=1.0, ay=0.0, by=1.0):
    """Physical coordinates of the bundle's centre rod."""
    n = 2 ** (p + Lf) - 1
    cx = np.mean([c[0] for c in rod_centres(p, Lf)])
    cy = np.mean([c[1] for c in rod_centres(p, Lf)])
    return ax + cx / n * (bx - ax), ay + cy / n * (by - ay)


# ---- Control volumes and probe indices ---------------------------------------
def control_volumes(p, Lf, half_frac=0.45, d_over_pitch=D_OVER_PITCH):
    """Flat indices of each rod's control-volume boundary, one dict per rod.

    ``half_frac`` is the CV half width as a fraction of the rod pitch.  It has
    to exceed the rod radius (1/3 of the pitch) and stay below half a pitch,
    or the box would either cut the rod or swallow its neighbour -- in a
    bundle the per-rod force is only defined on its own cell.
    """
    Q, n = p + Lf, 2**Lf
    if not (d_over_pitch / 2 < half_frac < 0.5):
        raise ValueError(
            f"--cv-half must lie in ({d_over_pitch / 2:.3f}, 0.5) for "
            f"D/pitch={d_over_pitch:.4f}, got {half_frac}"
        )
    h = int(round(half_frac * n))

    rods = []
    for cx, cy in rod_centres(p, Lf):
        # Rod centres sit on half-integers (a 2**Lf block has no centre cell),
        # so round() lands the box half a cell off centre and biases every
        # per-rod Cl -- the centreline rods came out at Cl ~ -0.09 against the
        # reference's ~0.  ceil/floor keeps the box centred on the rod.
        xl, xr = int(np.ceil(cx - h)), int(np.floor(cx + h))
        yl, yt = int(np.ceil(cy - h)), int(np.floor(cy + h))
        assert xl + xr == 2 * cx and yl + yt == 2 * cy, "CV box is off-centre"
        if min(xl, yl) < 0 or max(xr, yt) >= 2**Q:
            raise ValueError(f"control volume of rod at {(cx, cy)} leaves the domain")
        xs, ys = np.arange(xl, xr + 1), np.arange(yl, yt + 1)
        rods.append(
            {
                "L": shuffled_index(np.full_like(ys, xl), ys, Q),
                "R": shuffled_index(np.full_like(ys, xr), ys, Q),
                "B": shuffled_index(xs, np.full_like(xs, yl), Q),
                "T": shuffled_index(xs, np.full_like(xs, yt), Q),
            }
        )
    return rods, h


def shuffled_index(ix, iy, Q):
    """Flat index of grid point (ix, iy) in tensnet's shuffled (QTT) ordering.

    ``tt.shuffle_axis`` interleaves the bits of the two axes, x first, most
    significant bit first.  Going the other way lets us pull a handful of
    points out of an MPS with ``query`` instead of contracting the whole
    2**Q x 2**Q field, which at Q=13 is 67M doubles per field.
    """
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    flat = np.zeros(np.broadcast(ix, iy).shape, dtype=np.int64)
    for k in range(Q):
        flat = (flat << 1) | ((ix >> (Q - 1 - k)) & 1)
        flat = (flat << 1) | ((iy >> (Q - 1 - k)) & 1)
    return flat


def control_volume_forces(u, v, sxx, sxy, syy, dx, dy):
    """Force per unit depth on the body inside one control volume.

    Momentum balance over the CV, with rho = 1 and outward normals
    (-1,0), (+1,0), (0,-1), (0,+1) on faces L, R, B, T:

        F_i = closed_int [ sigma_ij n_j - u_i u_j n_j ] dS

    where sigma = -p I + nu (grad u + grad u^T).  The transient term
    d/dt int_V u_i dV is dropped (the cycle means are unaffected; see rod_forces.py).
    """
    fx = (
        (sxx["R"] - sxx["L"]).sum() * dy
        + (sxy["T"] - sxy["B"]).sum() * dx
        - ((u["R"] * u["R"]).sum() - (u["L"] * u["L"]).sum()) * dy
        - ((u["T"] * v["T"]).sum() - (u["B"] * v["B"]).sum()) * dx
    )
    fy = (
        (sxy["R"] - sxy["L"]).sum() * dy
        + (syy["T"] - syy["B"]).sum() * dx
        - ((v["R"] * u["R"]).sum() - (v["L"] * u["L"]).sum()) * dy
        - ((v["T"] * v["T"]).sum() - (v["B"] * v["B"]).sum()) * dx
    )
    return fx, fy


if __name__ == "__main__":
    sys.exit(main())
