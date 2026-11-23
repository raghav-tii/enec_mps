"""Velocity near the bundle from both runs at the same global step, for the comparison figure.

    python3 fields11.py Q10_SNAPSHOT Q11_SNAPSHOT

Each argument is a snapshot file (data/results/temp/<tag>_c###_int_sol_<k>.tmp).  The report used
the 986th step of each run (q10 chunk 1 step 485, q11 chunk 2 step 426).
Writes   data/archive/fields_common.npz
Used by  plots.py.
"""
import os
import sys

import numpy as np
import tensnet as tt

from _paths import SOLVER
sys.path.insert(0, SOLVER)
import bundle_simulation as es

tt.set_arch("sequential")
out = {}
for key, path, p, Lf in (("q10", sys.argv[1], 5, 5), ("q11", sys.argv[2], 6, 5)):
    Q = p + Lf
    n, h = 2 ** Q, 2.0 / 2 ** Lf                      # cells, cell size in D
    cen = np.array(es.rod_centres(p, Lf), float)
    cx, cy = cen[:, 0].mean(), cen[:, 1].mean()
    X, Y = (np.arange(n) - cx) * h, (np.arange(n) - cy) * h
    ix, iy = np.where(np.abs(X) < 9)[0], np.where(np.abs(Y) < 7)[0]
    ux, uy, _ = tt.load_object(path)
    for k, f in (("ux", ux), ("uy", uy)):
        d = tt.unshuffle_axis(f.contract(), Q, 2).numpy()
        out[f"{key}_{k}"] = d[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1].astype(np.float32)
    # rods: discs of diameter D at the lattice centres (cells), as in the solver's mask
    GX, GY = np.meshgrid(X[ix], Y[iy], indexing="ij")
    rel = lambda c: np.hypot(GX - (c[0] - cx) * h, GY - (c[1] - cy) * h)
    out[f"{key}_solid"] = np.any([rel(c) < 0.5 for c in cen], axis=0)
    out[f"{key}_X"], out[f"{key}_Y"] = X[ix], Y[iy]
    print(key, out[f"{key}_ux"].shape, flush=True)
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fields_common.npz"), **out)
