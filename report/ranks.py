"""Bond dimensions and parameter counts of every snapshot, plus warm-start compressibility.

    python3 ranks.py TAG [TAG ...]

Writes   data/archive/ranks.json: per tag the ranks and parameter counts of ux, uy, p over
         time, and "ic": the relative error of the warm start truncated to chi = 10, 20,
         40, 80, 160.
Used by  plots.py and cost_scaling.py.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

import tensnet as tt

import bundle_simulation as es
import chorin_projection as C
from _paths import OUT, RESULTS, warm_start_file

tt.set_arch("sequential")
TRUNCATIONS = (10, 20, 40, 80, 160)  # bond dimensions of interest for the warm start


def main(tags):
    out = {tag: snapshot_ranks(tag) for tag in tags}
    out["ic"] = warm_start_compressibility()
    Path(os.path.expanduser(f"{OUT}/ranks.json")).write_text(json.dumps(out))
    print("ok")


def snapshot_ranks(tag):
    """Bond dimensions and parameter counts of ux, uy, p of every snapshot of a run."""
    cfg = tt.load_object(os.path.join(RESULTS, f"{tag}_params.pfile"))
    rows = []
    for k, (_chunk, _step, path) in enumerate(es.find_snapshots(tag, RESULTS)):
        try:
            fields = tt.load_object(path)
        except Exception:
            continue
        rows.append(
            {
                "t": (k + 1) * cfg["dt"] / cfg["D"],
                "ranks": [[int(r) for r in f.ranks_list] for f in fields],
                "params": [int(f.num_params) for f in fields],
            }
        )
    print(tag, len(rows), rows[-1]["params"], flush=True)
    return rows


def warm_start_compressibility(Q=10):
    """Rank, parameters and truncation error of the warm start at each bond dimension."""
    tt.set_accuracy(1e-12)
    warm = tt.load_object(warm_start_file(5, 5))
    out = {}
    for name in ("ux", "uy", "pressure"):
        field = warm[name]
        full = C.mps_to_grid(field, Q)
        errors = {}
        for bd in TRUNCATIONS:
            truncated = C.mps_to_grid(field.copy().compress(max_bd=bd), Q)
            errors[bd] = float(np.linalg.norm(truncated - full) / np.linalg.norm(full))
        out[name] = {"rank": int(field.max_rank), "params": int(field.num_params), "err": errors}
        print(name, out[name], flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
