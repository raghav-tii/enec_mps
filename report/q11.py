"""Whole-domain speed and vorticity of a run at the start, middle and end, every 2nd cell.

    python3 q11.py TAG                    # the report used q11_t2500_bd60_rk4

Writes   data/archive/q11_<TAG>.npz
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
tag = sys.argv[1]
R = RESULTS
cfg = tt.load_object(os.path.join(R, f"{tag}_params.pfile"))
p, Lf = cfg["p"], cfg["Lf"]
Q = p + Lf
h = 2.0 / 2 ** Lf  # cell size in D
dense = lambda m: tt.unshuffle_axis(m.contract(), Q, 2).numpy()
snaps = es.find_snapshots(tag, R)
ws = tt.load_object(os.path.expanduser(f"{IC_DIR}/enec_ic_p{p}_f{Lf}_d500.mps"))
mask = dense(tt.load_object(os.path.expanduser(f"{MASK_DIR}/enec_mask_p{p}_f{Lf}_d500.mps")))
todo = [0, len(snaps) // 2, len(snaps) - 1]
out = dict(mask=mask[::2, ::2].astype(np.float16), h=h, centres=np.array(es.rod_centres(p, Lf), float),
           step=np.array([0] + [k + 1 for k in todo[1:]]), t=np.array(
               [0.0] + [(k + 1) * cfg["dt"] / cfg["D"] for k in todo[1:]]))
for i, k in enumerate(todo):
    u, v = (dense(ws["ux"]), dense(ws["uy"])) if i == 0 else tuple(
        dense(f) for f in tt.load_object(snaps[k][2])[:2])
    w = np.gradient(v, h, axis=0) - np.gradient(u, h, axis=1)
    out[f"speed{i}"] = np.hypot(u, v)[::2, ::2].astype(np.float16)
    out[f"vort{i}"] = w[::2, ::2].astype(np.float16)
    print(i, k, flush=True)
np.savez_compressed(os.path.expanduser(f"{OUT}/q11_{tag}.npz"), **out)
print("ok")
