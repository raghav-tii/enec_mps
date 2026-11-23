"""Cropped ux, uy of every STRIDE-th snapshot of a run, for the flow-evolution figure.

    python3 frames_s.py TAG STRIDE        # the report used STRIDE = 50

Reads    <results>/<TAG>_params.pfile and snapshots, the warm start (frame 0) and the mask
Writes   data/archive/frames_<TAG>_s<STRIDE>.npz: ux, uy (frames x cells), t (in D/U), mask, window origin and cell size
Used by  plots.py.
"""
import os, sys
import numpy as np
import tensnet as tt
from _paths import IC_DIR, MASK_DIR, OUT, RESULTS, SOLVER
sys.path.insert(0, SOLVER)
import bundle_simulation as es

tt.set_arch("sequential")
tag, stride = sys.argv[1], int(sys.argv[2])
R = RESULTS
cfg = tt.load_object(os.path.join(R, f"{tag}_params.pfile"))
p, Lf = cfg["p"], cfg["Lf"]
Q = p + Lf
cen = np.array(es.rod_centres(p, Lf), dtype=float)
cx, cy, c = cen[:, 0].mean(), cen[:, 1].mean(), 2**Lf / 2   # c = cells per D
ix = slice(int(cx - 9.5 * c), int(cx + 9.5 * c) + 1)
iy = slice(int(cy - 7.5 * c), int(cy + 7.5 * c) + 1)
dense = lambda m: tt.unshuffle_axis(m.contract(), Q, 2).numpy()[ix, iy].astype(np.float32)

ws = tt.load_object(os.path.expanduser(os.environ.get("IC", f"{IC_DIR}/enec_ic_p{p}_f{Lf}_d500.mps")))
mask = tt.load_object(os.path.expanduser(f"{MASK_DIR}/enec_mask_p{p}_f{Lf}_d500.mps"))
U, V, T = [dense(ws["ux"])], [dense(ws["uy"])], [0.0]
snaps = es.find_snapshots(tag, R)
for k in range(stride - 1, len(snaps), stride):
    try:
        ux, uy, _ = tt.load_object(snaps[k][2])
    except Exception:
        continue
    U.append(dense(ux)); V.append(dense(uy)); T.append((k + 1) * cfg["dt"] / cfg["D"])
    print(k, flush=True)
np.savez_compressed(os.path.expanduser(f"{OUT}/frames_{tag}_s{stride}.npz"),
                    ux=np.array(U), uy=np.array(V), t=np.array(T),
                    ref_ux=U[0], ref_uy=V[0], mask=dense(mask),
                    x0=(ix.start - cx) / c, y0=(iy.start - cy) / c, h=1 / c)
print("ok", len(T))
