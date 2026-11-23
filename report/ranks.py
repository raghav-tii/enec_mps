"""Bond dimensions and parameter counts of every snapshot, plus warm-start compressibility.

    python3 ranks.py TAG [TAG ...]

Writes   data/archive/ranks.json: per tag the ranks and parameter counts of ux, uy, p over time, and "ic": the
         relative error of the warm start truncated to chi = 10, 20, 40, 80, 160.
Used by  plots.py and cost_scaling.py.
"""
import json, os, sys
import numpy as np
import tensnet as tt
from _paths import IC_DIR, MASK_DIR, OUT, RESULTS, SOLVER
sys.path.insert(0, SOLVER)
import bundle_simulation as es

tt.set_arch("sequential")
R = RESULTS
out = {}
for tag in sys.argv[1:]:
    cfg = tt.load_object(os.path.join(R, f"{tag}_params.pfile"))
    rows = []
    for k, (_c, _i, path) in enumerate(es.find_snapshots(tag, R)):
        try:
            fs = tt.load_object(path)
        except Exception:
            continue
        rows.append({"t": (k + 1) * cfg["dt"] / cfg["D"],
                     "ranks": [[int(r) for r in f.ranks_list] for f in fs],
                     "params": [int(f.num_params) for f in fs]})
    out[tag] = rows
    print(tag, len(rows), rows[-1]["params"], flush=True)

# compressibility of the Star-CCM+ initial condition at the bond dimensions of interest
tt.set_accuracy(1e-12)
ws = tt.load_object(os.path.expanduser(os.environ.get("IC", f"{IC_DIR}/enec_ic_p5_f5_d500.mps")))
Q = 10
comp = {}
for k in ("ux", "uy", "pressure"):
    full = tt.unshuffle_axis(ws[k].contract(), Q, 2).numpy()
    comp[k] = {"rank": int(ws[k].max_rank), "params": int(ws[k].num_params), "err": {}}
    for bd in (10, 20, 40, 80, 160):
        m = ws[k].copy().compress(max_bd=bd) if hasattr(ws[k], "copy") else (ws[k] * 1.0).compress(max_bd=bd)
        d = tt.unshuffle_axis(m.contract(), Q, 2).numpy()
        comp[k]["err"][bd] = float(np.linalg.norm(d - full) / np.linalg.norm(full))
    print(k, comp[k], flush=True)
out["ic"] = comp
json.dump(out, open(os.path.expanduser(f"{OUT}/ranks.json"), "w"))
print("ok")
