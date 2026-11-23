"""Inflow and outflow planes of every STRIDE-th snapshot of a run: is the inflow held?

    python3 inlet.py TAG STRIDE

Writes   data/archive/inlet_<TAG>.csv with, per snapshot, u on the inlet plane (min, max, mean, rms deviation
         from U), max |v| there, and the mean u on row 4 (the first row the hold does not cover) and on the
         outlet plane.  The domain is y-periodic with uniform spacing, so a plane's mean u is its volume
         flux over the domain height.
Used by  plots.py.
"""
import os
import sys

import numpy as np
import tensnet as tt

from _paths import IC_DIR, MASK_DIR, OUT, RESULTS, SOLVER
sys.path.insert(0, SOLVER)
import bundle_simulation as es

tt.set_arch("sequential")
tag, stride = sys.argv[1], int(sys.argv[2])
R = RESULTS
cfg = tt.load_object(os.path.join(R, f"{tag}_params.pfile"))
Q, U = cfg["p"] + cfg["Lf"], cfg["u"]
dense = lambda m: tt.unshuffle_axis(m.contract(), Q, 2).numpy()
ws = tt.load_object(os.path.expanduser(os.environ.get(
    "IC", f"{IC_DIR}/enec_ic_p{cfg['p']}_f{cfg['Lf']}_d500.mps")))
snaps = es.find_snapshots(tag, R)
out = os.path.expanduser(f"{OUT}/inlet_{tag}.csv")
with open(out, "w") as fh:
    fh.write("step,t,in_min,in_max,in_mean,in_rms_dev,in_vmax,row4_mean,out_mean\n")
    todo = [(0, ws["ux"], ws["uy"])]
    todo += [(k + 1, None, snaps[k][2]) for k in range(stride - 1, len(snaps), stride)]
    for step, ux, uy in todo:
        if ux is None:
            try:
                ux, uy, _ = tt.load_object(uy)
            except Exception:
                continue
        u, v = dense(ux), dense(uy)
        fh.write(f"{step},{step * cfg['dt'] / cfg['D']:.3f},{u[0].min():.4f},{u[0].max():.4f},"
                 f"{u[0].mean():.4f},{np.sqrt(((u[0] - U) ** 2).mean()):.4f},"
                 f"{np.abs(v[0]).max():.4f},{u[4].mean():.4f},{u[-1].mean():.4f}\n")
        fh.flush()
        print(step, flush=True)
print("ok", out)
