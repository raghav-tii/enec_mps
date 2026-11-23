"""Whole-domain speed and vorticity of a run at the start, middle and end, every 2nd cell.

    python3 q11.py TAG                    # the report used q11_t2500_bd60_rk4

Writes   data/archive/q11_<TAG>.npz
Used by  plots.py.
"""

import os
import sys

import numpy as np

import tensnet as tt

import bundle_simulation as es
import chorin_projection as C
from _paths import OUT, RESULTS, mask_file, warm_start_file

tt.set_arch("sequential")


def main(tag):
    cfg = tt.load_object(os.path.join(RESULTS, f"{tag}_params.pfile"))
    p, Lf = cfg["p"], cfg["Lf"]
    Q, h = p + Lf, 2.0 / 2**Lf  # h: cell size in D
    snaps = es.find_snapshots(tag, RESULTS)
    picks = [0, len(snaps) // 2, len(snaps) - 1]  # start (the warm start), middle, end

    warm = tt.load_object(warm_start_file(p, Lf))
    mask = C.mps_to_grid(tt.load_object(os.path.expanduser(mask_file(p, Lf))), Q)
    out = dict(
        mask=mask[::2, ::2].astype(np.float16),
        h=h,
        centres=np.array(es.rod_centres(p, Lf), float),
        step=np.array([0] + [k + 1 for k in picks[1:]]),
        t=np.array([0.0] + [(k + 1) * cfg["dt"] / cfg["D"] for k in picks[1:]]),
    )
    for i, k in enumerate(picks):
        ux, uy = (warm["ux"], warm["uy"]) if i == 0 else tt.load_object(snaps[k][2])[:2]
        u, v = C.mps_to_grid(ux, Q), C.mps_to_grid(uy, Q)
        vorticity = np.gradient(v, h, axis=0) - np.gradient(u, h, axis=1)
        out[f"speed{i}"] = np.hypot(u, v)[::2, ::2].astype(np.float16)
        out[f"vort{i}"] = vorticity[::2, ::2].astype(np.float16)
        print(i, k, flush=True)
    np.savez_compressed(os.path.expanduser(f"{OUT}/q11_{tag}.npz"), **out)
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
