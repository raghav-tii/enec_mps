# Data

| Directory | Content |
|---|---|
| `star_ccm/` | Star-CCM+ reference: per-rod drag and lift monitors `C_D.csv`, `C_L.csv`, and snapshots `star_t<tU/D>.vtk` at tU/D = 2500, 2520, 2540, 2570, 2600, 2640 (cell data `Velocity`, `Pressure`, `Vorticityk`; read with `scripts/star_vtk.py`) |
| `initial_conditions/` | warm starts at tU/D = 2500 as MPS: `enec_ic_p5_f5_d500.mps` (1024²) and `enec_ic_p6_f5_d500.mps` (2048²). Rebuild with `make warmstart` |
| `mps_masks/` | rod-bundle masks as MPS, same two grids. Built on demand by `bundle_simulation.py mask` if missing |
| `archive/` | outputs of the report runs: per-rod forces, field extracts, bond dimensions, benchmarks. Which script wrote which file: see `report/README.md` |
| `results/` | your own runs (untracked): `<tag>_params.pfile`, `temp/<tag>_c###_int_sol_<k>.tmp` snapshots, `rod_forces_<tag>.csv`. Empty on a fresh checkout; `scripts/clean` empties it |

File names: `p<p>_f<Lf>` is the grid, 2^(p+Lf) per side (`p`: qubits of the 5×5
rod lattice, `Lf`: qubits of one rod's block); `d500` is D/pitch = 0.5.
