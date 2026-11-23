# Scripts

The solver and its wrappers. Run the wrappers from anywhere; they `cd` here.
Outputs go to `../data/results/`.

| Wrapper | What it does |
|---|---|
| `./check` | difference-operator test, solver suites, probe-vs-dense forces, dense twin |
| `./run smoke` | 256² bundle: build a warm start and mask in `data/results/smoke/`, then 3 RK4 steps |
| `./run q10 [--restart]` | 1024², χ = 40 report run, 500 steps per call (5530 in the report); `DEVICE=cuda` for a GPU |
| `./run q11 [--restart]` | 2048², χ = 60 report run, 300 steps per call (1486 in the report) |
| `./post TAG [--workers N] [--dense]` | per-rod Cd/Cl (`rod_forces.py`) → `data/results/rod_forces_TAG.csv`, compared with `data/archive` |
| `./clean` | delete run outputs in `data/results/` |

## The Python entry points

* `bundle_simulation.py`
  * `mask`: build or load the bundle mask in `data/mps_masks/`.
  * `warmstart --vtk F --p P --sizes LF...`: interpolate a Star-CCM+ VTK onto the grid and write the result to `data/initial_conditions/`.
  * `run`: time-step and save one snapshot per step to `data/results/temp/<tag>_c<chunk>_int_sol_<k>.tmp`.
  * `check`: self-check.
* `chorin_projection.py`: the projection method: `Operators`, `predict_velocity` (Euler/RK4), `project_divergence_free`, `projection_step`, `march_in_time`. Run it directly for its test suite.
* `rod_forces.py`: per-rod forces from the velocity alone, by probes (default) or dense contraction (`--dense`). The ring pressure is integrated from the steady momentum equation. These are the forces quoted in the report.
* `compare_rod_forces.py`: a rerun's forces against an archived run, step by step.
* `dense_projection_check.py`: a dense n²×n² twin of `chorin_projection.projection_step`, for small grids.
* `uncompressed_twin.py`: the same run without truncation, from its first snapshot: same operators, exact Sylvester Poisson solve. Separates compression error from discretisation error.
* `poisson_rank_benchmark.py`: Poisson accuracy against χ. It wrote `data/archive/poisson_rank.csv`.
* `star_vtk.py`: reads Star-CCM+ VTK files with numpy only.
