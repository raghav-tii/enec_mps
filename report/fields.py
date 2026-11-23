"""Dense ux, uy of the newest snapshot of each run, plus the warm start, mask and rod centres.

    python3 fields.py TAG [TAG ...]       # e.g. q10_t2500_bd40_rk4 q11_t2500_bd60_rk4

Reads    <results>/<TAG>_params.pfile and <results>/temp/ snapshots,
         data/initial_conditions, data/mps_masks
Writes   data/archive/fields.npz: <TAG>_ux, <TAG>_uy, <TAG>_nsteps, ref_ux, ref_uy (warm
         start), mask, centres, names (rod names) and cfg_* (the run's scalar settings)
Used by  plots.py.
Options  FIELDS_OUT=path overrides the output file; IC=path the warm-start file.
"""

import os
import sys

import numpy as np

import tensnet as tt

import bundle_simulation as es
import chorin_projection as C
from _paths import OUT, RESULTS, mask_file, warm_start_file

tt.set_arch("sequential")


def main(tags):
    fields = {}
    for tag in tags:
        cfg = tt.load_object(os.path.join(RESULTS, f"{tag}_params.pfile"))
        fields.update(newest_velocity(tag, cfg))
    fields.update(reference_fields(cfg))  # all runs share one geometry
    fields.update({f"cfg_{k}": v for k, v in cfg.items() if np.isscalar(v)})
    out = os.environ.get("FIELDS_OUT", f"{OUT}/fields.npz")
    np.savez_compressed(os.path.expanduser(out), **fields)
    print("ok")


def newest_velocity(tag, cfg):
    """Dense ux, uy of the newest readable snapshot (a live run's last file may be half-written)."""
    snaps = es.find_snapshots(tag, RESULTS)
    for _chunk, _step, path in reversed(snaps):
        try:
            ux, uy, _p = tt.load_object(path)
            break
        except Exception:
            continue
    Q = cfg["p"] + cfg["Lf"]
    print(tag, len(snaps), flush=True)
    return {
        f"{tag}_ux": C.mps_to_grid(ux, Q).astype(np.float32),
        f"{tag}_uy": C.mps_to_grid(uy, Q).astype(np.float32),
        f"{tag}_nsteps": len(snaps),
    }


def reference_fields(cfg):
    """Warm start, mask, rod centres and names of the geometry."""
    p, Lf = cfg["p"], cfg["Lf"]
    warm = tt.load_object(warm_start_file(p, Lf))
    mask = tt.load_object(os.path.expanduser(mask_file(p, Lf)))
    return {
        "ref_ux": C.mps_to_grid(warm["ux"], p + Lf).astype(np.float32),
        "ref_uy": C.mps_to_grid(warm["uy"], p + Lf).astype(np.float32),
        "mask": C.mps_to_grid(mask, p + Lf).astype(np.float32),
        "centres": np.array(es.rod_centres(p, Lf), dtype=float),
        "names": np.array(es.rod_names(p, Lf)),
    }


if __name__ == "__main__":
    main(sys.argv[1:])
