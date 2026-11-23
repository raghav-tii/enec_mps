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

import bundle_simulation as es
import chorin_projection as C
from _paths import OUT

tt.set_arch("sequential")
HALF_WIDTH_X, HALF_WIDTH_Y = 9, 7  # window around the bundle centre, in D


def main(q10_snapshot, q11_snapshot):
    out = {}
    for key, path, p, Lf in (("q10", q10_snapshot, 5, 5), ("q11", q11_snapshot, 6, 5)):
        out.update(near_bundle(key, path, p, Lf))
        print(key, out[f"{key}_ux"].shape, flush=True)
    np.savez_compressed(os.path.join(OUT, "fields_common.npz"), **out)


def near_bundle(key, path, p, Lf):
    """ux, uy cropped to the bundle window, the window's coordinates (in D) and a rod mask."""
    Q = p + Lf
    n, h = 2**Q, 2.0 / 2**Lf  # cells, cell size in D
    centres = np.array(es.rod_centres(p, Lf), float)
    cx, cy = centres[:, 0].mean(), centres[:, 1].mean()
    X, Y = (np.arange(n) - cx) * h, (np.arange(n) - cy) * h
    ix = np.where(np.abs(X) < HALF_WIDTH_X)[0]
    iy = np.where(np.abs(Y) < HALF_WIDTH_Y)[0]
    window = (slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1))

    ux, uy, _p = tt.load_object(path)
    out = {
        f"{key}_{k}": C.mps_to_grid(f, Q)[window].astype(np.float32)
        for k, f in (("ux", ux), ("uy", uy))
    }

    # rods: discs of diameter D at the lattice centres, as in the solver's mask
    GX, GY = np.meshgrid(X[ix], Y[iy], indexing="ij")
    out[f"{key}_solid"] = np.any(
        [np.hypot(GX - (c[0] - cx) * h, GY - (c[1] - cy) * h) < 0.5 for c in centres], axis=0
    )
    out[f"{key}_X"], out[f"{key}_Y"] = X[ix], Y[iy]
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
