"""Compare per-rod forces of a rerun with the archived production run, step by step.

    python3 scripts/compare_rod_forces.py data/results/rod_forces_TAG.csv
    data/archive/rod_forces_q10_t2500_bd40_rk4.csv

Both files are rod_forces.py output (one row per step: 21 Cd and 21 Cl values).  For
every step present in both it prints the bundle-mean Cd, the per-rod RMS
deviation of Cd and Cl relative to the archived values, and the per-rod
correlation.  The archived run itself scatters by ~0.1-0.2 % in u and
1.5-2.5 % in v between two executions of the same step (randomised SVD in the
compression; final report Sec. 5.7.1), so agreement is judged against that
spread, not bit for bit.
"""

import csv
import sys
from pathlib import Path

import numpy as np


def load(path):
    rows = list(csv.DictReader(Path(path).read_text().splitlines()))
    cd = [k for k in rows[0] if k.startswith("Cd_")]
    cl = [k for k in rows[0] if k.startswith("Cl_")]
    out = {}
    for r in rows:
        out[int(r["step"])] = (
            np.array([float(r[k]) for k in cd]),
            np.array([float(r[k]) for k in cl]),
        )
    return out


new, ref = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(new) & set(ref))
if not common:
    raise SystemExit("no common steps")
print(
    f"{'step':>5} {'<Cd> new':>9} {'<Cd> ref':>9} {'dCd rms/|Cd|':>13} "
    f"{'dCl rms/|Cl|':>13} {'corr Cd':>8} {'corr Cl':>8}"
)
worst = 0.0
for s in common:
    (cdn, cln), (cdr, clr) = new[s], ref[s]
    ecd = np.sqrt(np.mean((cdn - cdr) ** 2)) / np.sqrt(np.mean(cdr**2))
    ecl = np.sqrt(np.mean((cln - clr) ** 2)) / np.sqrt(np.mean(clr**2))
    worst = max(worst, ecd)
    print(
        f"{s:5d} {cdn.mean():9.4f} {cdr.mean():9.4f} {ecd:13.2e} {ecl:13.2e} "
        f"{np.corrcoef(cdn, cdr)[0, 1]:8.4f} {np.corrcoef(cln, clr)[0, 1]:8.4f}"
    )
print(f"worst relative per-rod Cd deviation over {len(common)} steps: {worst:.2e}")
