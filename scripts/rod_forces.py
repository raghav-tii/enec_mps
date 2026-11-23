#!/usr/bin/env python3
"""Per-rod drag and lift on the ENEC bundle, from the velocity alone.

The solver's pressure is not trustworthy (the Poisson solve floors), but the
force on a rod only needs p on its control-volume ring, and the ring lies in
the fluid, where the steady momentum equation gives

    grad p = -(u.grad)u + nu lap u

Integrating that tangentially around the closed ring recovers p up to a
constant, and a constant contributes nothing to the force (the ring has as
many points on L as on R, on B as on T).  The ring pressure then goes through
bundle_simulation.control_volume_forces exactly like a stored one would.

The unsteady term -du/dt is deliberately dropped: differencing snapshots
amplifies their noise by 1/(2 dt) and made the estimate worse, not better.
The loop-closure mismatch (what is left on returning to the start) is kept
and reported: it measures how far the velocity is from admitting ANY
single-valued pressure, i.e. the estimator's own error bar.

Calibrated on the Star-CCM+ warm start: <Cd> 0.5773 vs reference 0.5675,
per-rod Cd corr 0.9977.

Where the velocity comes from
-----------------------------
probes (default)  u and v are read with MPS.query on a small window around each
                  rod (its control volume plus the 4 cells the 8th-order
                  stencils reach), 37 x 37 points at the standard cv_half.  No
                  operator is applied to the MPS and no full 2**Q x 2**Q array
                  is built, so the cost does not grow with the grid.
--dense           both fields are contracted to full arrays and differenced
                  once for all rods.  The report forces were made this way;
                  it needs 2 * 4**Q doubles, which is fine up to Q = 12.
Both feed the same stencils, so they agree to round-off (see selfcheck).

    python3 rod_forces.py --tag TAG [--stride N] [--dense] [--out data/results/rod_forces_TAG.csv]

Processes whatever snapshots exist, so it runs fine on a tag still being
written.  Rerun with --append to process only the snapshots not yet in the output
file (a full rerun recomputes everything, and moves old rows in the last digit);
--workers N spreads the snapshots over N processes.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

import tensnet as tt

import bundle_simulation as es

tt.set_arch("sequential")

tt.set_accuracy(1e-12)

# 8th-order central differences: the same coefficients tensnet's get_diff_mpo
# uses (get_cendif_coef), so these forces use the solver's discretisation.
A1 = {1: 4.0 / 5.0, 2: -1.0 / 5.0, 3: 4.0 / 105.0, 4: -1.0 / 280.0}

A2 = {0: -205.0 / 72.0, 1: 8.0 / 5.0, 2: -1.0 / 5.0, 3: 8.0 / 315.0, 4: -1.0 / 560.0}

HALO = 4  # reach of the 8th-order stencils: the ring needs u, v this far outside itself

_idx = np.argwhere(es.rod_positions() == 1)

MIR = np.array(
    [{tuple(r): i for i, r in enumerate(_idx)}[(a, 4 - b)] for a, b in _idx]
)  # rod index of each rod's y mirror

JOB = {}  # set by main, inherited by forked workers


# ---- Command line and per-snapshot driver ------------------------------------
def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--tag", required=True)
    a.add_argument("--stride", type=int, default=1)
    a.add_argument("--out", default=None, help="default data/results/rod_forces_TAG.csv")
    a.add_argument("--results", default=es.RESULTS)
    a.add_argument(
        "--reference",
        default=os.environ.get("QCFD_REFERENCE", es.REFERENCE),
        metavar="DIR",
        help="Star-CCM+ C_D.csv/C_L.csv directory",
    )
    a.add_argument(
        "--dense",
        action="store_true",
        help="contract the full fields instead of probing around each rod (the report forces)",
    )
    a.add_argument(
        "--append",
        action="store_true",
        help="add rows only for snapshots whose step is not in --out yet",
    )
    a.add_argument("--workers", type=int, default=1, help="processes over snapshots")
    args = a.parse_args()
    selfcheck()

    cfg = tt.load_object(os.path.join(args.results, f"{args.tag}_params.pfile"))
    bxs = boxes(cfg)
    names = es.rod_names(cfg["p"], cfg["Lf"])
    rcd, rcl = es.reference_coefficients(args.reference)
    ref_cd = np.array([rcd[n] for n in names])
    print(
        f"reference <Cd> {ref_cd.mean():.4f} (D/pitch 0.5; this run "
        f"{cfg.get('d_over_pitch', es.D_OVER_PITCH)})"
    )

    snaps = list(enumerate(es.find_snapshots(args.tag, args.results)))[:: args.stride]
    if not snaps:
        raise SystemExit(f"no snapshots for tag '{args.tag}'")
    out = args.out or os.path.join(args.results, f"rod_forces_{args.tag}.csv")
    done = set()
    if args.append and os.path.exists(out):
        done = {int(r["step"]) for r in csv.DictReader(Path(out).read_text().splitlines())}
        snaps = [x for x in snaps if x[0] not in done]
    print(
        f"{len(snaps)} snapshots -> {out}" + (f" ({len(done)} already there)" if done else ""),
        flush=True,
    )

    JOB.update(cfg=cfg, bxs=bxs, ref_cd=ref_cd, dense=args.dense)

    if args.workers > 1:
        import multiprocessing as mp

        import torch

        pool = mp.get_context("fork").Pool(args.workers, lambda: torch.set_num_threads(1))
        results = pool.imap(process_snapshot, snaps)  # ordered, so rows stay in step order
    else:
        results = map(process_snapshot, snaps)
    hist = []  # (<Cd>, <Cl>, corr) per processed snapshot, for the summary
    new = not (args.append and done)
    with open(out, "w" if new else "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(
                ["step", "time", "closure"]
                + [f"Cd_{n}" for n in names]
                + [f"Cl_{n}" for n in names]
            )
        for _step, res, line in results:
            print(line, flush=True)
            if res is None:
                continue
            w.writerow(res[0])
            fh.flush()
            hist.append(res[1:])
    if not hist:
        print("nothing new")
        return
    h = np.array(hist)
    print(
        f"summary: <Cd> {h[0, 0]:+.4f} -> {h[-1, 0]:+.4f} (min {h[:, 0].min():+.4f}"
        f", max {h[:, 0].max():+.4f})  <Cl> {h[0, 1]:+.4f} -> {h[-1, 1]:+.4f}  "
        f"corr(Cd,ref) {h[0, 2]:+.3f} -> {h[-1, 2]:+.3f}  (ref <Cd> {ref_cd.mean():.4f})"
    )


def process_snapshot(item):
    """One CSV row for one snapshot: (step, (row, <Cd>, <Cl>, corr) or None, log line)."""
    cfg, bxs, ref_cd, dense = JOB["cfg"], JOB["bxs"], JOB["ref_cd"], JOB["dense"]
    step, (_c, _i, path) = item
    try:
        ux, uy, _p = tt.load_object(path)
    except Exception as e:  # the newest file of a live run may be half-written
        return step, None, f"  step {step:4d}  skipped ({type(e).__name__}: {e})"
    cd, cl, clo, u = rod_coefficients(ux, uy, cfg, bxs, dense)
    t = (step + 1) * cfg["dt"]
    row = [step, t, np.abs(clo).mean()] + list(cd) + list(cl)
    return (
        step,
        (row, cd.mean(), cl.mean(), np.corrcoef(cd, ref_cd)[0, 1]),
        (
            f"  step {step:4d}  t={t:.4e}  <Cd>={cd.mean():+.4f}  <Cl>={cl.mean():+.4f}"
            f"  corr={np.corrcoef(cd, ref_cd)[0, 1]:+.4f}  sym(Cd)={sym_cd(cd):.3f}"
            + (f"  asym(ux)={field_asym(u, cfg, bxs):.3e}" if dense else "")
            + f"  clo={np.abs(clo).mean():.3f}"
        ),
    )


# ---- Forces of one snapshot --------------------------------------------------
def rod_coefficients(ux, uy, cfg, bxs, dense=False):
    """Per-rod (Cd, Cl, closure in Cd units), and the dense u when `dense`
    (None for probes).  ux, uy are MPS; with `dense` they may also be dense [x, y] arrays."""
    Q, dx, dy, nu = cfg["p"] + cfg["Lf"], cfg["dx"], cfg["dy"], cfg["nu"]
    if dense:
        u, v = (
            m if isinstance(m, np.ndarray) else tt.unshuffle_axis(m.contract(), Q, 2).numpy()
            for m in (ux, uy)
        )
        f = velocity_derivatives(u, v, dx, dy, nu)
        out = [rod_force(f, box, cfg) for box in bxs]
    else:
        u, out = None, []
        for box in bxs:
            wu, wv, local = probe_window(ux, uy, box, Q)
            out.append(rod_force(velocity_derivatives(wu, wv, dx, dy, nu), local, cfg))
    cd, cl, clo = (np.array(c) for c in zip(*out))
    return cd, cl, clo, u


def rod_force(f, box, cfg):
    """(Cd, Cl, closure in Cd units) of one rod.  `f` is velocity_derivatives of
    any array in which `box` indexes the control-volume ring."""
    dx, dy, nu = cfg["dx"], cfg["dy"], cfg["nu"]
    norm = 0.5 * cfg["u"] ** 2 * cfg["D"]
    pts, tan, ax = loop_walk(box)
    ix, iy = pts[:, 0], pts[:, 1]
    ploop, closure = loop_pressure(f["gx"][ix, iy], f["gy"][ix, iy], pts, tan, ax, dx, dy)
    pmap = np.zeros_like(f["u"])
    pmap[ix, iy] = ploop
    fs = face_slices(box)
    on = {
        k: {face: F[i] for face, i in fs.items()}
        for k, F in dict(
            u=f["u"], v=f["v"], dudx=f["dudx"], dvdy=f["dvdy"], sh=f["sh"], p=pmap
        ).items()
    }
    sxx = {face: 2 * nu * on["dudx"][face] - on["p"][face] for face in es.FACES}
    syy = {face: 2 * nu * on["dvdy"][face] - on["p"][face] for face in es.FACES}
    sxy = {face: nu * on["sh"][face] for face in es.FACES}
    fx, fy = es.control_volume_forces(on["u"], on["v"], sxx, sxy, syy, dx, dy)
    return fx / norm, fy / norm, closure / norm * dx * (box[1] - box[0] + 1)


def probe_window(ux, uy, box, Q):
    """u, v on the control-volume ring of `box` plus HALO cells, read with MPS.query.
    Returns the arrays and the ring's box in their index coordinates."""
    xl, xr, yl, yt = box
    gx = np.arange(xl - HALO, xr + HALO + 1)
    gy = np.arange(yl - HALO, yt + HALO + 1)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    flat = es.shuffled_index(X, Y, Q).ravel().tolist()
    u, v = (np.asarray(m.query(flat), dtype=np.float64).reshape(X.shape) for m in (ux, uy))
    return u, v, (HALO, HALO + xr - xl, HALO, HALO + yt - yl)


def velocity_derivatives(u, v, dx, dy, nu):
    """Derivative fields of u, v, and grad p from the steady momentum equation."""
    dudx, dudy, dvdx, dvdy = d1(u, 0, dx), d1(u, 1, dy), d1(v, 0, dx), d1(v, 1, dy)
    return dict(
        u=u,
        v=v,
        dudx=dudx,
        dvdy=dvdy,
        sh=dudy + dvdx,
        gx=-(u * dudx + v * dudy) + nu * (d2(u, 0, dx) + d2(u, 1, dy)),
        gy=-(u * dvdx + v * dvdy) + nu * (d2(v, 0, dx) + d2(v, 1, dy)),
    )


# ---- Control-volume ring and pressure from its gradient ----------------------
def boxes(cfg):
    """CV boxes (xl, xr, yl, yt), checked against control_volumes' indices so
    the rod order and box edges are exactly the ones `coefficients` uses."""
    p, Lf, Q = cfg["p"], cfg["Lf"], cfg["p"] + cfg["Lf"]
    ref, h = es.control_volumes(p, Lf, cfg["cv_half"], cfg.get("d_over_pitch", es.D_OVER_PITCH))
    out = []
    for i, (cx, cy) in enumerate(es.rod_centres(p, Lf)):
        xl, xr = int(np.ceil(cx - h)), int(np.floor(cx + h))
        yl, yt = int(np.ceil(cy - h)), int(np.floor(cy + h))
        xs, ys = np.arange(xl, xr + 1), np.arange(yl, yt + 1)
        assert np.array_equal(es.shuffled_index(np.full_like(ys, xl), ys, Q), ref[i]["L"])
        assert np.array_equal(es.shuffled_index(xs, np.full_like(xs, yt), Q), ref[i]["T"])
        out.append((xl, xr, yl, yt))
    return out


def loop_walk(box):
    """Closed counter-clockwise walk around the ring: points, the unit tangent
    of the edge leading INTO each point, and that edge's axis (0=x, 1=y)."""
    xl, xr, yl, yt = box
    pts, tan, ax = [], [], []

    def seg(coords, t, a):
        for c in coords:
            pts.append(c)
            tan.append(t)
            ax.append(a)

    seg([(x, yl) for x in range(xl, xr + 1)], (1, 0), 0)  # B, +x
    seg([(xr, y) for y in range(yl + 1, yt + 1)], (0, 1), 1)  # R, +y
    seg([(x, yt) for x in range(xr - 1, xl - 1, -1)], (-1, 0), 0)  # T, -x
    seg([(xl, y) for y in range(yt - 1, yl, -1)], (0, -1), 1)  # L, -y
    tan[0], ax[0] = (0, -1), 1  # point 0 is reached by the closing L edge
    return np.array(pts), np.array(tan, float), np.array(ax)


def face_slices(box):
    xl, xr, yl, yt = box
    return {
        "L": (np.full(yt - yl + 1, xl), np.arange(yl, yt + 1)),
        "R": (np.full(yt - yl + 1, xr), np.arange(yl, yt + 1)),
        "B": (np.arange(xl, xr + 1), np.full(xr - xl + 1, yl)),
        "T": (np.arange(xl, xr + 1), np.full(xr - xl + 1, yt)),
    }


def loop_pressure(gx, gy, pts, tan, ax, dx, dy):
    """p on the ring from grad p by trapezoid around the closed walk.

    Returns (p, closure).  The closure is redistributed linearly in arc
    length (the standard PIV fix), which makes p independent of walk
    direction.
    """
    ds = np.where(ax == 0, dx, dy)
    # Edge k runs from point k-1 to point k along tan[k].  BOTH endpoints are
    # projected onto tan[k]: at a corner the tail's own incoming edge has a
    # different tangent, and using it rotates grad p by 90 degrees.
    head = gx * tan[:, 0] + gy * tan[:, 1]
    tail = np.roll(gx, 1) * tan[:, 0] + np.roll(gy, 1) * tan[:, 1]
    step = 0.5 * (head + tail) * ds
    p = np.concatenate([[0.0], np.cumsum(step[1:])])
    closure = p[-1] + step[0]
    s = np.concatenate([[0.0], np.cumsum(ds[1:])])
    return p - closure * s / (s[-1] + ds[0]), closure


# ---- Stencils ----------------------------------------------------------------
def d1(f, axis, h):
    out = np.zeros_like(f)
    for k, c in A1.items():
        out += c * (np.roll(f, -k, axis) - np.roll(f, k, axis))
    return out / h


def d2(f, axis, h):
    out = A2[0] * f
    for k in (1, 2, 3, 4):
        out += A2[k] * (np.roll(f, -k, axis) + np.roll(f, k, axis))
    return out / h**2


# ---- Diagnostics and self-check ----------------------------------------------
def sym_cd(cd):
    """Mean per-rod Cd mirror violation, relative to the bundle's mean |Cd|."""
    return float((np.abs(cd - cd[MIR]) / np.abs(cd).mean()).mean())


def field_asym(u, cfg, bxs):
    """||antisymmetric part of ux|| / ||ux|| over the bundle window, y
    reflected about the bundle axis.  Window only: the bundle sits off-centre
    in a y-periodic box, so a whole-domain norm reports that, not the flow."""
    cys = [c[1] for c in es.rod_centres(cfg["p"], cfg["Lf"])]
    j = (int(round(min(cys) + max(cys))) - np.arange(u.shape[1])) % u.shape[1]
    w = (
        slice(min(b[0] for b in bxs), max(b[1] for b in bxs) + 1),
        slice(min(b[2] for b in bxs), max(b[3] for b in bxs) + 1),
    )
    return np.linalg.norm(u[w] - u[:, j][w]) / (2 * np.linalg.norm(u[w]))


def selfcheck():
    """The line integral must rebuild an analytic pressure from its gradient, and
    probes must give the same forces as the dense contraction."""
    h = 1e-3
    x, y = np.meshgrid(np.arange(100) * h, np.arange(100) * h, indexing="ij")
    p = np.sin(7 * x) * np.cos(5 * y) + x * y
    gx, gy = 7 * np.cos(7 * x) * np.cos(5 * y) + y, -5 * np.sin(7 * x) * np.sin(5 * y) + x
    pts, tan, ax = loop_walk((10, 80, 20, 90))
    ix, iy = pts[:, 0], pts[:, 1]
    got, c = loop_pressure(gx[ix, iy], gy[ix, iy], pts, tan, ax, h, h)
    want = p[ix, iy]
    assert abs(c) < 1e-6 and np.abs(got - got.mean() - want + want.mean()).max() < 1e-6, c

    Q = 10  # smooth 1024^2 field with a few modes: low rank, so exact to the MPS tolerance
    cfg = dict(
        p=5, Lf=5, dx=1 / (2**Q - 1), dy=1 / (2**Q - 1), nu=1e-3, u=1.0, D=0.03, cv_half=0.45
    )
    g = np.linspace(0, 1, 2**Q)
    X, Y = np.meshgrid(g, g, indexing="ij")
    ux, uy = (
        tt.vector_to_mps(tt.shuffle_axis(torch.as_tensor(f), Q).flatten(), 2, 2 * Q, 1e-13)
        for f in (1 + 0.3 * np.sin(9 * X) * np.cos(7 * Y), 0.2 * np.cos(8 * X + 3 * Y))
    )
    bxs = boxes(cfg)[:3]
    probes, dense = (rod_coefficients(ux, uy, cfg, bxs, dense=d)[:3] for d in (False, True))
    for a, b in zip(probes, dense):
        assert np.abs(a - b).max() < 1e-8 * np.abs(b).max(), (a, b)


if __name__ == "__main__":
    main()
