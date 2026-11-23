# Report figures and tables

`make report` (from the repository root) regenerates every figure and table of
the final report from the archived results in `data/archive` and the Star-CCM+
data in `data/star_ccm`. Figures go to `report/fig/*.pdf`; the numbers and table
rows quoted in the text go to `report/fig/{numbers,rods,rods11,table}.tex`.
It needs LaTeX (matplotlib `usetex`); the Docker image has it.

| Script | What it does |
|---|---|
| `plots.py` | all figures and tables except Appendix A |
| `cost_scaling.py` | Appendix A: cost scaling figure, Table 8, gradient error against χ, parameter count |

## Refreshing the archived data

The files in `data/archive` are the outputs of the extractors below, run on a
finished run's snapshots (`QCFD_REPORT_DATA=../data/results` makes them read
and write a rerun's `data/results` instead; the README's *Usage* section has the
full sequence).

| Script | Command | Writes (in `data/archive`) |
|---|---|---|
| `fields.py` | `python3 fields.py TAG [TAG ...]` | `fields.npz`: dense velocity of the newest snapshots, warm start, mask |
| `fields11.py` | `python3 fields11.py Q10_SNAPSHOT Q11_SNAPSHOT` | `fields_common.npz`: both runs at the same step |
| `frames_s.py` | `python3 frames_s.py TAG 50` | `frames_<TAG>_s50.npz`: every 50th snapshot, cropped |
| `inlet.py` | `python3 inlet.py TAG 50` | `inlet_<TAG>.csv`: inflow and outflow plane statistics |
| `ranks.py` | `python3 ranks.py TAG [TAG ...]` | `ranks.json`: bond dimensions and parameter counts |
| `q11.py` | `python3 q11.py TAG` | `q11_<TAG>.npz`: whole-domain speed and vorticity |
| `compress.py` | `python3 compress.py` | `compress.json`: truncation error of the Star-CCM+ field against χ |
| `wake.py` | `python3 wake.py` | `star_wake.csv`: wake unsteadiness; needs the full Star-CCM+ snapshot archive (`QCFD_STAR_TAR`) |
| `../scripts/rod_forces.py` | `./post TAG --dense` | `rod_forces_<TAG>.csv`: per-rod forces |
| `../scripts/poisson_rank_benchmark.py` | `python3 poisson_rank_benchmark.py --help` | `poisson_rank.csv`: Poisson accuracy against χ |
| `gpu_scaling.py`, `gpu_bench.py` | need a CUDA device; see their docstrings | `gpu_cluster.csv` (Table 7) |

`star.py` (Star-CCM+ fields on a regular grid) and `_paths.py` (where inputs and
outputs live, overridable with `QCFD_DATA`, `QCFD_RESULTS`, `QCFD_REFERENCE`,
`QCFD_REPORT_DATA`, `QCFD_STAR_TAR`) are helpers used by the scripts above.
