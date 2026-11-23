# tensnet

Matrix product states (MPS) and operators (MPO), finite-difference operators on MPS grids,
and AMEn / two-site DMRG linear solvers, in PyTorch.

```bash
pip install -e .              # from the repository root; numpy, scipy, torch, einops
python3 -m unittest discover -s tests -p "*_test.py"
```

## What is in it

| Module | Contents |
|---|---|
| `networks` | `MPS`, `MPO`, `MPS_Node`, `MPO_Node`, compression and canonicalisation, `vector_to_mps`, `matrix_to_mpo`, `identity_mpo`/`identity_mps`, `random_mps`/`random_mpo` |
| `linalg` | AMEn and two-site DMRG linear solvers (`linsys_amen`, `linsys_dmrg2`, `make_local_solver`), central-difference MPOs (`difference_mpo.py`: `get_diff_mpo`, `get_2D_operator`, `row_mpo`, …), and the explicit-grid operators with SBP wall closures (`explicit_grid_operators.py`: `diff_mpo`, `grid_points`, `grid_spacing`, `mpo_sum`, `bc_mps`, …) |
| `utils` | settings (`set_accuracy`, `set_bd_limit`, `set_arch`, `set_param`, `set_device`, `set_comp_method`), `shuffle_axis`/`unshuffle_axis`, `save_object`/`load_object`, SVD/QR helpers |

## Difference operators

`get_diff_mpo(dim, order, axis, nofq, bc)` builds the 8th-order central-difference
MPO of the first, second or fourth derivative. `bc` is `"periodic"`, `"dirichlet"` or
`"neumann"` for each end. These are the operators the 21-rod bundle solver uses.
`../tests/difference_mpo_golden_test.py` compares them with golden matrices.
