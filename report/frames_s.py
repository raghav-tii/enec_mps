"""Cropped ux, uy of every STRIDE-th snapshot of a run, for the flow-evolution figure.

    python3 frames_s.py TAG STRIDE        # the report used STRIDE = 50

Reads    <results>/<TAG>_params.pfile and snapshots, the warm start (frame 0) and the mask
Writes   data/archive/frames_<TAG>_s<STRIDE>.npz: ux, uy (frames x cells), t (in D/U), mask,
         window origin and cell size
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
HALF_WIDTH_X, HALF_WIDTH_Y = 9.5, 7.5  # crop around the bundle centre, in D


def main(tag, stride):
    cfg = tt.load_object(os.path.join(RESULTS, f"{tag}_params.pfile"))
    p, Lf = cfg["p"], cfg["Lf"]
    crop = Crop(p, Lf)

    warm = tt.load_object(warm_start_file(p, Lf))
    ux, uy = [crop(warm["ux"])], [crop(warm["uy"])]
    t = [0.0]
    snaps = es.find_snapshots(tag, RESULTS)
    for k in range(stride - 1, len(snaps), stride):
        try:
            vx, vy, _p = tt.load_object(snaps[k][2])
        except Exception:
            continue
        ux.append(crop(vx))
        uy.append(crop(vy))
        t.append((k + 1) * cfg["dt"] / cfg["D"])
        print(k, flush=True)

    mask = tt.load_object(os.path.expanduser(mask_file(p, Lf)))
    np.savez_compressed(
        os.path.expanduser(f"{OUT}/frames_{tag}_s{stride}.npz"),
        ux=np.array(ux), uy=np.array(uy), t=np.array(t),
        ref_ux=ux[0], ref_uy=uy[0], mask=crop(mask),
        x0=crop.x0, y0=crop.y0, h=crop.h,
    )  # fmt: skip
    print("ok", len(t))


class Crop:
    """Dense, cropped, single-precision view of an MPS field around the bundle."""

    def __init__(self, p, Lf):
        self.Q = p + Lf
        centres = np.array(es.rod_centres(p, Lf), dtype=float)
        cx, cy, cells_per_D = centres[:, 0].mean(), centres[:, 1].mean(), 2**Lf / 2
        self.window = (
            slice(int(cx - HALF_WIDTH_X * cells_per_D), int(cx + HALF_WIDTH_X * cells_per_D) + 1),
            slice(int(cy - HALF_WIDTH_Y * cells_per_D), int(cy + HALF_WIDTH_Y * cells_per_D) + 1),
        )
        self.x0 = (self.window[0].start - cx) / cells_per_D
        self.y0 = (self.window[1].start - cy) / cells_per_D
        self.h = 1 / cells_per_D

    def __call__(self, mps):
        return C.mps_to_grid(mps, self.Q)[self.window].astype(np.float32)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
