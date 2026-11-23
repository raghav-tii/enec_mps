# qcfd-mps

Reproducibility package for the KU/ENTC–TII project *Quantum Inspired Algorithm–CFD*. It contains:

* the MPS (tensor-train) Navier–Stokes solver used for the 21-rod bundle;
* the `tensnet` library it runs on;
* the archived results and Star-CCM+ reference data behind the final report;
* the scripts that turn the archived results into the report's figures and tables;
* the workshop tutorial notebook.

## Table of Contents

- [Introduction](#introduction)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [The tutorial notebook](#the-tutorial-notebook)
- [Finite-difference operators: the two builders](#finite-difference-operators-the-two-builders)
- [Per-rod forces: probes and dense](#per-rod-forces-probes-and-dense)
- [What reproduces what](#what-reproduces-what)
- [File formats](#file-formats)
- [Notes on reproducibility](#notes-on-reproducibility)
- [Troubleshooting](#troubleshooting)

---

## Introduction

The solver integrates the 2D incompressible Navier–Stokes equations on a
uniform grid. It stores velocity and pressure as matrix product states (MPS),
applies derivatives as matrix product operators (MPO), and does the pressure
solve with AMEn. All of this works on the compressed form.

Time stepping uses a Chorin rotational incremental pressure-correction scheme
with an RK4 predictor. The 21 rods (a 5×5 lattice without its corners) are
imposed by volume penalisation through an MPS mask. Per-rod drag and lift are
computed on control volumes around each rod, from the velocity alone
(`rod_forces.py`).

The report's main results come from two runs. Both start from the Star-CCM+
solution at tU/D = 2500:

| Run | Grid / domain | χ (bond dimension) | Steps | Report |
|---|---|---|---|---|
| `q10_t2500_bd40_rk4`, "half domain" | 1024², 64D × 64D | 40 | 5530 | §5.2–5.5, §5.7.1 |
| `q11_t2500_bd60_rk4`, "full domain" | 2048², 128D × 128D | 60 | 1486 | §5.2–5.5 |

## Repository layout

```
tensnet/                 MPS/MPO data structures, finite-difference operators, AMEn/DMRG solvers
tests/                   golden test of the difference operators
scripts/                 the solver
  chorin_projection.py   projection step: Operators, predictor (Euler/RK4), project, step, march
  bundle_simulation.py   geometry/mask, Star-CCM+ warm start, time stepping (subcommands mask, warmstart, run, check)
  rod_forces.py          per-rod Cd/Cl from saved snapshots, by probes (default) or dense contraction
  compare_rod_forces.py  a rerun's forces against the archived run
  star_vtk.py            reader for the Star-CCM+ VTK files
  dense_projection_check.py, uncompressed_twin.py, poisson_rank_benchmark.py
                         dense twin of one projection step; uncompressed twin of a whole run; Poisson accuracy against χ
  run, post, check, clean   wrappers: ./run q10|q11|smoke, ./post TAG, ./check, ./clean
report/                  scripts that regenerate the report figures and tables:
                         plots.py, cost_scaling.py (Appendix A) and the extractors that wrote data/archive
tutorial/                mps_cfd_building_blocks.ipynb
data/
  star_ccm/              Star-CCM+ per-rod monitors C_D.csv, C_L.csv; VTK snapshots tU/D = 2500…2640
  initial_conditions/    warm starts at tU/D = 2500 for the 1024² and 2048² runs (MPS)
  mps_masks/             rod-bundle masks for both grids (MPS)
  archive/               outputs of the production runs (forces, field extracts, ranks, benchmarks)
  results/               your runs: <tag>_params.pfile, temp/<tag>_c###_int_sol_<k>.tmp, rod_forces_<tag>.csv (untracked)
```

## Installation

### Docker (recommended)

The image contains Python, PyTorch, JupyterLab, `tensnet` and LaTeX (the report figures use `usetex`).

```bash
docker build -t qcfd-mps .
docker run --rm -p 8888:8888 qcfd-mps          # tutorial notebook: open the URL it prints
```

For a CUDA build, add `--build-arg TORCH_INDEX=` and run with `--gpus all`.

### Without Docker

You need Python 3.10 or newer, and LaTeX if you want the report figures.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 -r requirements.txt
pip install -e .                               # tensnet
```

A CUDA GPU is optional. The self-checks, the small demo run, the notebook and
the report figures all run on a laptop CPU. A full production run takes days,
even on a GPU.

## Usage

Run these from the repository root, or prefix them with
`docker run --rm -it qcfd-mps` (add `-v "$PWD/data/results:/work/data/results"`
to keep run outputs, `-v "$PWD/fig:/work/report/fig"` to keep the figures).

```bash
make notebook         # JupyterLab on tutorial/ (port 8888)
make report           # report/fig/*.pdf and the numbers/tables (*.tex) from data/archive
make check            # difference-operator test, solver suite, probe-vs-dense forces, dense twin
make smoke            # 256² bundle: warm start, 3 RK4 steps, per-rod forces (a few minutes)
make warmstart        # rebuild data/initial_conditions from data/star_ccm/star_t2500.vtk
```

### Running the production simulations

```bash
cd scripts
DEVICE=cuda ./run q10             # 1024², chi=40: advances 500 steps per call
DEVICE=cuda ./run q10 --restart   # continue; repeat until 5530 steps (tU/D 2638.2)
./post q10_t2500_bd40_rk4 --workers 8    # per-rod forces + comparison with data/archive
```

The 2048², χ = 60 run is the same with `q11`; it reaches tU/D 2537.2 after 1486 steps.

* The `q10` and `q11` presets in `scripts/run` hold the settings of the report
  runs: Re = 100 per rod, D/pitch = 0.5, CFL 0.4, RK4, held projection, AMEn
  with 6 sweeps, and tolerance 1e-12.
* A run can be stopped at any time and resumed with `--restart`.
* Each call to `./run` writes one snapshot per step (velocity and pressure as
  MPS) to `data/results/temp/`.
* `./post` writes `data/results/rod_forces_<tag>.csv`.

| Run | Cost per step |
|---|---|
| q10 | RTX A6000 ≈ 64 s; 16 CPU threads 151–177 s; 2 laptop threads ≈ 40–60 s at the start of the run, while the ranks are still being built up |
| q11 | RTX A6000 ≈ 186 s |

To refresh the report data from a rerun:

```bash
cd report
QCFD_REPORT_DATA=../data/results python3 fields.py q10_t2500_bd40_rk4
QCFD_REPORT_DATA=../data/results python3 frames_s.py q10_t2500_bd40_rk4 50
QCFD_REPORT_DATA=../data/results python3 inlet.py q10_t2500_bd40_rk4 50
QCFD_REPORT_DATA=../data/results python3 ranks.py q10_t2500_bd40_rk4
QCFD_REPORT_DATA=../data/results python3 q11.py q11_t2500_bd60_rk4
# copy the refreshed files (and data/results/rod_forces_*.csv) into data/archive/, then: make report
```

The report scripts locate their inputs through `report/_paths.py`. The
defaults follow the layout above, and you can override them with these
environment variables:

* `QCFD_DATA`
* `QCFD_RESULTS`
* `QCFD_REFERENCE`
* `QCFD_REPORT_DATA`
* `QCFD_STAR_TAR`: the full Star-CCM+ VTK archive, needed only for snapshot times not in `data/star_ccm`.

## The tutorial notebook

`tutorial/mps_cfd_building_blocks.ipynb` builds the solver up from its parts, in
nine sections that each run in seconds to about 35 s on a laptop CPU: storing a
field as an MPS (§1–2), immersed-boundary masks (§3), difference operators (§4),
field algebra and truncation (§5), the pressure-Poisson solve (§6), projection
time steps (§7), error against χ on the archived full-domain run (§8) and the
accuracy–cost trade-off (§9).

`make notebook`, or the `docker run -p 8888:8888 qcfd-mps` command above, starts
JupyterLab in `tutorial/`; open the URL it prints. The notebook imports the
solver modules from `../scripts` and reads
`../data/archive/q11_q11_t2500_bd60_rk4.npz`, so it must be started from
`tutorial/` (both commands do). It ships with its outputs, so you can read it
before running it.

## Finite-difference operators: the two builders

`tensnet` offers two ways to build the derivative MPOs. The solver uses the
first; the notebook (§4) demonstrates the second.

| | `get_diff_mpo` (default, used by the solver) | `diff_mpo` (explicit grid, alternative) |
|---|---|---|
| Module | `tensnet/linalg/difference_mpo.py` | `tensnet/linalg/explicit_grid_operators.py` |
| Stencil | 9-point, 8th-order central; 3-point, 2nd-order with `set_param("low_acc", True)` | interior accuracy 2, 4, 6 or 8, chosen with `accuracy=` |
| Derivatives | orders 1, 2, 4, and the mixed derivative (`axis=-1`) | orders 1, 2, per axis |
| Scaling | none: multiply by `h**-order` yourself | included: built from `bounds` and the grid type |
| Grid | not stated; the caller picks the spacing | stated: `"periodic"` (h = L/n), `"vertex"` (h = L/(n-1), nodes on both walls) or `"cell"` (h = L/n, walls half a cell outside) |
| Walls | ghost values: `"dirichlet"` zero ghosts, `"neumann"` mirrored ghosts. The 4 rows next to each wall are not a consistent derivative: their error does not shrink (Dirichlet) or grows (Neumann) under refinement | summation-by-parts closures of order `accuracy/2`; boundary conditions are imposed weakly through the data vector `bc_mps`, so `A u + b` approximates the derivative. Wall rows converge |
| Symmetry | none | `weighted=True` returns H·A, symmetric (semi)definite for order 2, so symmetric solvers such as DMRG apply |
| Global state | site order `set_arch`, stencil `low_acc` | only the default site order `set_arch` |
| Cost | cheapest to build | each builder recompresses to the exact rank |

**Why the solver uses `get_diff_mpo`.** It is the discretisation the report runs
used, so `data/archive` is reproduced with it. Its weak wall rows cost nothing
there: `chorin_projection.Operators` treats the 4 rows at each end as boundary
rows (`inflow_rows=4`) and holds them, and the rods are interior. The golden
test `tests/difference_mpo_golden_test.py` fixes its matrices for every
dimension, order and boundary pair the solver builds.

**When to use `diff_mpo`.** For new work with walls inside the physical domain,
for convergence studies, or when a symmetric operator is needed. Periodic
grids are the same stencil, so the two agree there up to the scaling.
The solver does not use it: switching `chorin_projection.Operators` to it
changes the discretisation of the boundary rows and the report numbers with it.

## Per-rod forces: probes and dense

`rod_forces.py` needs the velocity on each rod's control-volume ring and 4
cells around it (the reach of the 8th-order stencils). It gets it in one of two ways:

| | Default: probes | `--dense` |
|---|---|---|
| How | `MPS.query` on a 37 × 37 window around each rod | contract both fields to full 2^Q × 2^Q arrays |
| Cost | independent of the grid size | 2 · 4^Q doubles; fine up to Q = 12 |
| Used for | runs, `./post`, any grid | the report forces (`data/archive/rod_forces_*.csv`) |

Both feed the same stencils and agree to round-off; `rod_forces.selfcheck()`
tests this on a 1024² field. For the report-size grids you can compare them on your own run:

```bash
./post TAG                    # probes
./post TAG --dense --out ../data/results/rod_forces_TAG_dense.csv
```

## What reproduces what

| Report item | Command | Inputs |
|---|---|---|
| WP1/WP2 building blocks: encoding, operators, projection, Poisson solve (library + solver correctness) | `make check`, `make notebook` | none |
| WP3 §5.2–5.5: per-rod Cd/Cl vs Star-CCM+ (Table 5, Fig. 14), Star-CCM+ histories (Fig. 12), control volumes (Fig. 11), velocity/vorticity (Fig. 13), drift, full-domain run | `make report` | `data/archive`, `data/star_ccm` |
| §2.4 and compression figure: truncation error of the Star-CCM+ field vs χ | `cd report && python3 compress.py` | `data/star_ccm/star_t2500.vtk` |
| §5.7.2 Table 7: GPU vs CPU DMRG timings | `python3 report/gpu_bench.py` (CUDA), `report/gpu_scaling.py` | none |
| Appendix A: Fig. 19, Table 8, A.4 gradient error (53–56 % at χ=40, 6–7 % at χ=120), 33 448 parameters per field (31:1) | `make report` | `data/archive/poisson_rank.csv`, `ranks.json` |
| The flow runs themselves (the source of `data/archive`) | `scripts/run q10`, `scripts/run q11`, then `scripts/post --dense` | `data/initial_conditions`, `data/mps_masks` |

The archived files in `data/archive/`, and the script that wrote each:

| File | Written by |
|---|---|
| `rod_forces_q10_t2500_bd40_rk4.csv` (5530 steps), `rod_forces_q11_t2500_bd60_rk4.csv` (1486 steps) | `scripts/rod_forces.py --dense` |
| `fields.npz`, `fields_common.npz`, `frames_q10_t2500_bd40_rk4_s50.npz`, `q11_q11_t2500_bd60_rk4.npz` | `report/fields.py`, `fields11.py`, `frames_s.py`, `q11.py` |
| `inlet_q10_t2500_bd40_rk4.csv`, `ranks.json` | `report/inlet.py`, `report/ranks.py` |
| `compress.json`, `star_wake.csv` | `report/compress.py`, `report/wake.py` |
| `poisson_rank.csv` | `scripts/poisson_rank_benchmark.py` |
| `gpu_cluster.csv` | `report/gpu_scaling.py` |

## File formats

| File | Content |
|---|---|
| `data/results/<tag>_params.pfile` | Python dict (`tensnet.load_object`) with the run settings: `p`, `Lf`, `Q`, `dx`, `dy`, `dt`, `nu`, `D`, `u`, `Re`, `eps`, `d_over_pitch`, `cv_half`, `held`, … |
| `data/results/temp/<tag>_c<chunk>_int_sol_<k>.tmp` | one snapshot: the tuple `(ux, uy, p)` of MPS. `chunk` counts `--restart` calls, `k` the steps within it |
| `data/results/rod_forces_<tag>.csv` | one row per snapshot: `step`, `time` (solver units; divide by `D` for tU/D), `closure` (loop-closure mismatch of the ring pressure, in Cd units), then `Cd_<rod>` and `Cl_<rod>` for the 21 rods. Rods are named `<row><column>`, 11…55 without the corners |
| `data/initial_conditions/enec_ic_p<p>_f<Lf>_d500.mps` | dict with the MPS `ux`, `uy`, `pressure` and the geometry it was built for |
| `data/mps_masks/enec_mask_p<p>_f<Lf>_d500.mps` | the fluid mask MPS (1 in the fluid, 0 inside the rods) |

Grid size is 2^(p+Lf) per side: `p` qubits for the 5×5 rod lattice, `Lf` for one
rod's block. `q10` is p = 5, Lf = 5 (1024²); `q11` is p = 6, Lf = 5 (2048²).
An MPS stores a field with its two index bits interleaved, x first and most
significant bit first; `bundle_simulation.shuffled_index(ix, iy, Q)` gives the
flat index of a grid point.

## Notes on reproducibility

**Bitwise reproduction.** Runs are not reproducible bit for bit, and were not
in production either: the compression uses randomised SVD, and the pressure
solve is deliberately not fully converged (relative residual ≈ 0.3–0.4).

* In production, two executions of the same step differed by 0.1–0.2 % in u and by 1.5–2.5 % in v (report §5.7.1).
* The AMEn contraction was refactored for memory after the runs. That refactor is exact up to round-off, which then grows through the unconverged solve in the same way.

Compare reruns statistically, with `scripts/compare_rod_forces.py` (`./post` runs it).

**Star-CCM+ reference.** `data/star_ccm` holds the per-rod monitors `C_D.csv`,
`C_L.csv` (Re = 100 per rod, D/pitch = 0.5) and snapshots at tU/D = 2500, 2520,
2540, 2570, 2600 and 2640. The full time series of snapshots is not shipped; the
scripts that need other times (`QCFD_STAR_TAR`) are only needed to regenerate
archive files that are already in `data/archive`.

**Warm starts and masks.** `make warmstart` rebuilds both from `star_t2500.vtk`.
The 1024² mask matches the production mask to 4e-14.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: tensnet` | Run `pip install -e .` from the repository root (the Docker image already has it). |
| Report figures fail with a LaTeX error or `latex not found` | `report/plots.py` needs LaTeX (matplotlib `usetex`): install `texlive-latex-extra`, `texlive-fonts-recommended`, `cm-super` and `dvipng`, or use the Docker image. |
| `no snapshots for tag '…'` | The tag has no files in `data/results/temp/`; run `./run` first, or set `QCFD_RESULTS` to the directory that holds `<tag>_params.pfile` and `temp/`. |
| `No warm-start file at …` or `… was built for …` | `--warm-start auto` looks for `data/initial_conditions/enec_ic_p<p>_f<Lf>_d<1000·D/pitch>.mps`; build it with `make warmstart`. |
| `Time stepping FAILED after N saved steps` | The projection stopped converging. Lower `--cfl`, raise `--opt-iter` or `--max-bd`. Snapshots up to the failure are kept; continue with `--restart`. |
| `./run q10` seems to stop after 500 steps | By design: each call advances 500 steps. Call `./run q10 --restart` again. |
| Forces differ from the archive in the third digit | Expected: runs are not bitwise reproducible (see above). Use `./post`, which compares statistically. |
| `./post` is slow on a large grid | Use the default probes (not `--dense`), and `--workers N` to spread snapshots over processes. |
| Out of memory with `--dense` | It needs 2·4^Q doubles per snapshot per worker; drop `--dense` or `--workers`. |
| Notebook cannot import `chorin_projection` | The kernel was not started in `tutorial/`; use `make notebook`. |
| Docker: no JupyterLab URL reachable | Publish the port: `docker run --rm -p 8888:8888 qcfd-mps`, then use the URL with token that it prints. |
