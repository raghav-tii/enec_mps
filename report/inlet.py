"""Inflow and outflow planes of every STRIDE-th snapshot of a run: is the inflow held?

    python3 inlet.py TAG STRIDE

Writes   data/archive/inlet_<TAG>.csv with, per snapshot, u on the inlet plane (min, max,
         mean, rms deviation from U), max |v| there, and the mean u on row 4 (the first row
         the hold does not cover) and on the outlet plane.  The domain is y-periodic with
         uniform spacing, so a plane's mean u is its volume flux over the domain height.
Used by  plots.py.
"""

import os
import sys

import numpy as np

import tensnet as tt

import bundle_simulation as es
import chorin_projection as C
from _paths import OUT, RESULTS, warm_start_file

tt.set_arch("sequential")
HEADER = "step,t,in_min,in_max,in_mean,in_rms_dev,in_vmax,row4_mean,out_mean"


def main(tag, stride):
    cfg = tt.load_object(os.path.join(RESULTS, f"{tag}_params.pfile"))
    Q = cfg["p"] + cfg["Lf"]
    warm = tt.load_object(warm_start_file(cfg["p"], cfg["Lf"]))
    snaps = es.find_snapshots(tag, RESULTS)

    out = os.path.expanduser(f"{OUT}/inlet_{tag}.csv")
    with open(out, "w") as fh:
        fh.write(HEADER + "\n")
        steps = [(0, warm["ux"], warm["uy"])]  # step 0 is the warm start
        for k in range(stride - 1, len(snaps), stride):
            try:
                ux, uy, _p = tt.load_object(snaps[k][2])
            except Exception:
                continue
            steps.append((k + 1, ux, uy))
        for step, ux, uy in steps:
            fh.write(plane_statistics(step, C.mps_to_grid(ux, Q), C.mps_to_grid(uy, Q), cfg) + "\n")
            fh.flush()
            print(step, flush=True)
    print("ok", out)


def plane_statistics(step, u, v, cfg):
    """One CSV row: u on the inlet plane (x = 0), |v| there, u on row 4 and on the outlet."""
    inlet = u[0]
    return (
        f"{step},{step * cfg['dt'] / cfg['D']:.3f},{inlet.min():.4f},{inlet.max():.4f},"
        f"{inlet.mean():.4f},{np.sqrt(((inlet - cfg['u']) ** 2).mean()):.4f},"
        f"{np.abs(v[0]).max():.4f},{u[4].mean():.4f},{u[-1].mean():.4f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
