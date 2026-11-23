"""Dense ux, uy of the newest snapshot of each run, plus the warm start, mask and rod centres.

    python3 fields.py TAG [TAG ...]       # e.g. q10_t2500_bd40_rk4 q11_t2500_bd60_rk4

Reads    <results>/<TAG>_params.pfile and <results>/temp/ snapshots, data/initial_conditions, data/mps_masks
Writes   data/archive/fields.npz: <TAG>_ux, <TAG>_uy, <TAG>_nsteps, ref_ux, ref_uy (warm start), mask,
         centres, names (rod names) and cfg_* (the run's scalar settings)
Used by  plots.py.
Options  FIELDS_OUT=path overrides the output file; IC=path the warm-start file.
"""
import os, sys
import numpy as np
import tensnet as tt
from _paths import IC_DIR, MASK_DIR, OUT, RESULTS, SOLVER
sys.path.insert(0, SOLVER)
import bundle_simulation as es

tt.set_arch("sequential")
R = RESULTS
out = {}
cfg = None
for tag in sys.argv[1:]:
    cfg = tt.load_object(os.path.join(R, f"{tag}_params.pfile"))
    snaps = es.find_snapshots(tag, R)
    for path in (s[2] for s in reversed(snaps)):
        try:
            ux, uy, _ = tt.load_object(path); break
        except Exception:
            continue
    Q = cfg["p"] + cfg["Lf"]
    for k, f in (("ux", ux), ("uy", uy)):
        out[f"{tag}_{k}"] = tt.unshuffle_axis(f.contract(), Q, 2).numpy().astype(np.float32)
    out[f"{tag}_nsteps"] = len(snaps)
    print(tag, len(snaps), flush=True)
p, Lf = cfg["p"], cfg["Lf"]
ws = tt.load_object(os.path.expanduser(os.environ.get("IC", f"{IC_DIR}/enec_ic_p{p}_f{Lf}_d500.mps")))
for k in ("ux", "uy"):
    out[f"ref_{k}"] = tt.unshuffle_axis(ws[k].contract(), p + Lf, 2).numpy().astype(np.float32)
out["centres"] = np.array(es.rod_centres(p, Lf), dtype=float)
m = tt.load_object(os.path.expanduser(f"{MASK_DIR}/enec_mask_p{p}_f{Lf}_d500.mps"))
out["mask"] = tt.unshuffle_axis(m.contract(), p + Lf, 2).numpy().astype(np.float32)
out["names"] = np.array(es.rod_names(p, Lf))
for k, v in cfg.items():
    if np.isscalar(v):
        out[f"cfg_{k}"] = v
np.savez_compressed(os.path.expanduser(os.environ.get("FIELDS_OUT", f"{OUT}/fields.npz")), **out)
print("ok")
