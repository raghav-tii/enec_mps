"""get_diff_mpo must reproduce the difference operators the report runs used.

The golden matrices cover every dimension, derivative order and boundary pair
the solver builds, on a 2**3 (per axis) grid.
"""
import os
import unittest

import numpy as np
import torch

import tensnet as tt

torch.set_default_dtype(torch.float64)
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "get_diff_mpo_golden_q3.npz")
BCS = (["dirichlet", "neumann"], ["neumann", "dirichlet"], ["dirichlet", "dirichlet"], ["periodic", "periodic"])


class DifferenceMPOGolden(unittest.TestCase):
    def setUp(self):
        self.saved = (tt.ARCH_2D(), tt.COMP_ACC())
        tt.set_arch("sequential")
        tt.set_accuracy(1e-12)

    def tearDown(self):
        tt.set_arch(self.saved[0])
        tt.set_accuracy(self.saved[1])

    def test_matches_golden_matrices(self):
        golden, Q = np.load(GOLDEN), 3
        for dim in (1, 2):
            for order in (1, 2):
                for bc in BCS:
                    for ax in ((0,) if dim == 1 else (0, 1)):
                        full = bc if dim == 1 else (bc + ["periodic"] * 2 if ax == 0 else ["periodic"] * 2 + bc)
                        key = f"{dim}{order}{''.join(b[0] for b in bc)}{ax}"
                        with self.subTest(key=key):
                            m = tt.get_diff_mpo(dim=dim, order=order, axis=ax, nofq=[Q] * dim, bc=full)
                            np.testing.assert_allclose(np.asarray(m.contract()), golden[key], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
