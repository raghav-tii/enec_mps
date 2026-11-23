"""Appendix A of the final report: analytical cost scaling and measured rank/accuracy data.

    python3 report/cost_scaling.py          # writes report/fig/cost_scaling.pdf and prints the tables

Reproduces
  * Figure 19  -- relative local-solve cost against bond dimension, normalised at chi = 30:
                  chi^4 (one matrix-vector product of an assembled dense local matrix,
                  i.e. one inner iteration) and chi^6 (one dense SVD / pseudoinverse).
  * Table 8    -- storage (chi^2), iterative step (chi^4) and direct factorisation (chi^6)
                  increases for a two- and fourfold bond-dimension increase.
  * A.4        -- pressure-increment gradient error against chi, from the measured
                  data/archive/poisson_rank.csv (written by scripts/poisson_rank_benchmark.py).
  * Sec. 2.4   -- parameters per field at chi = 40 on the 1024^2 grid and the 31:1 ratio,
                  from data/archive/ranks.json (written by report/ranks.py).

Nothing here runs the solver; the measured inputs are archived in data/archive.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "archive")
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
try:
    import shutil
    usetex = shutil.which("latex") is not None
except Exception:
    usetex = False
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.4, "legend.frameon": False})

# --- Table 8 and Figure 19: analytical ---------------------------------------
print("Table 8: relative increase at fixed core count and local mode size")
print(f"  {'chi factor':>10}  {'storage chi^2':>13}  {'iterative chi^4':>15}  {'direct chi^6':>12}")
for f in (2, 4):
    print(f"  {f:>9}x  {f**2:>12}x  {f**4:>14}x  {f**6:>11}x")

chi = np.linspace(30, 120, 200)
ref = 30.0
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.semilogy(chi, (chi / ref) ** 4, color=BLUE, label=r"Iterative step: fixed inner iteration count, $\chi^4$")
ax.semilogy(chi, (chi / ref) ** 6, color=ORANGE, label=r"Dense SVD / pseudoinverse, $\chi^6$")
for c in (60, 120):
    for p, col in ((4, BLUE), (6, ORANGE)):
        v = (c / ref) ** p
        ax.plot(c, v, "o", ms=4, color=col)
        ax.annotate(f"{v:.0f}" + (r"$\times$" if usetex else "x"), (c, v), textcoords="offset points",
                    xytext=(-6, 6), ha="right", fontsize=8, color=INK)
ax.set_xlabel(r"Bond dimension $\chi$ (reference: 30)")
ax.set_ylabel("Relative local-solve cost")
ax.set_xticks([30, 60, 90, 120])
ax.legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "cost_scaling.pdf"))
print(f"wrote {os.path.join(FIG, 'cost_scaling.pdf')}")

# --- A.4: gradient error of the pressure increment against chi ----------------
path = os.path.join(DATA, "poisson_rank.csv")
rows = list(csv.DictReader(open(path)))
by_chi = {}
for r in rows:
    if r["method"] != "best":   # best rank-chi approximation of the exact increment
        continue
    by_chi.setdefault(int(r["chi"]), []).append(float(r["grad_err"]))
print("\nA.4: relative error of grad(phi) against chi (min-max over the archived states)")
for c in sorted(by_chi):
    e = 100 * np.array(by_chi[c])
    print(f"  chi={c:4d}: {e.min():5.1f} - {e.max():5.1f} %   ({len(e)} rows)")

# --- Sec. 2.4: storage of one field at the production chi --------------------
rk = json.load(open(os.path.join(DATA, "ranks.json")))
for tag, val in rk.items():
    if isinstance(val, list) and val:
        params = val[-1]["params"]
        n = 2 ** 20
        print(f"\nSec. 2.4 ({tag}): parameters per field at the last snapshot {params}; "
              f"ratio {n / max(params):.0f}:1 against {n} grid values")
        break
