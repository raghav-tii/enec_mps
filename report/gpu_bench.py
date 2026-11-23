"""CPU vs CUDA timing of linsys_dmrg2 for the report's GPU table (Table 7).

    OMP_NUM_THREADS=4 python3 gpu_bench.py          # needs a CUDA device

One warm-up, then three runs per device on a shifted periodic 2D Laplacian, at chi = 10, 20 and 30.
Prints the sorted CPU and GPU times and the peak GPU memory; nothing is written.
"""

import time

import torch

import tensnet as tt

torch.set_default_dtype(torch.float64)
Lx = Lt = 8


def build(device):
    torch.manual_seed(0)
    A = tt.get_diff_mpo(dim=2, axis=0, order=2, nofq=[Lx, Lt], bc="periodic") + tt.get_diff_mpo(
        dim=2, axis=1, order=2, nofq=[Lx, Lt], bc="periodic"
    )
    A = A.to(device=device, dtype=torch.float64)
    A = A * -1.0 + tt.identity_mpo(Lx + Lt, dtype=torch.float64, device=device) * 4.0
    b = tt.random_mps(Lx + Lt, bd=4, pd=2, dtype=torch.float64, device=device)
    x0 = tt.random_mps(Lx + Lt, bd=4, pd=2, dtype=torch.float64, device=device)
    return A, b, x0


def run(device, chi):
    A, b, x0 = build(device)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tt.linsys_dmrg2(A, b, x0, trunc_eps=1e-10, max_rank=chi, max_iter=2)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, (
        torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    )


run("cuda", 10)  # warm-up
for chi in (10, 20, 30):
    c = sorted(run("cpu", chi)[0] for _ in range(3))
    g = [run("cuda", chi) for _ in range(3)]
    print(
        chi,
        "cpu",
        [round(x, 3) for x in c],
        "cuda",
        sorted(round(x[0], 3) for x in g),
        "MB %.1f" % g[-1][1],
        flush=True,
    )
