"""GPU against CPU: tensnet solver kernels and one ENEC time step.

Run once per device, in separate processes so no global state leaks:
    python3 gpu_scaling.py cpu  > cpu.csv
    python3 gpu_scaling.py cuda > cuda.csv
    python3 gpu_scaling.py cpu enec   # ENEC step only: a second CPU run, the noise floor
Threads come from OMP_NUM_THREADS.  Kernels: two-site DMRG and AMEn
on a shifted periodic 2D Laplacian (the report's laptop problem, 16 tensors,
and the ENEC grid size, 20 tensors) over a range of bond dimensions, in double
and single precision.  Each problem is built on the CPU from a fixed seed and
then moved, so both devices solve the same system.  ENEC: chorin_projection.projection_step with the
production settings of the incremental run, from its snapshot at step 500.
Timings are medians of up to three runs after a warm-up; a run over 60 s is
not repeated.  Output rows: part,device,dtype,sites,chi,seconds,runs,peak_mb,note
"""

import os
import statistics
import sys
import time

import numpy as np
import torch

import tensnet as tt

dev = sys.argv[1]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
tt.set_arch("sequential")
tt.set_param("low_acc", False)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def timed(fn, reps=3, limit=60.0):
    times = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
        if times[-1] > limit:
            break
    return statistics.median(times), len(times)


def peak():
    return torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")


def row(*a):
    print(",".join(str(x) for x in a), flush=True)


def problem(L, dtype):
    """-lap + 4 I on a periodic 2**L x 2**L grid, a rank-4 b and x0; built on CPU."""
    torch.manual_seed(0)
    nofq = [L, L]
    A = tt.get_diff_mpo(
        dim=2, axis=0, order=2, nofq=nofq, bc="periodic", device="cpu"
    ) + tt.get_diff_mpo(dim=2, axis=1, order=2, nofq=nofq, bc="periodic", device="cpu")
    A = A * -1.0 + tt.identity_mpo(2 * L, device="cpu") * 4.0
    b = tt.random_mps(2 * L, bd=4, pd=2, device="cpu")
    x0 = tt.random_mps(2 * L, bd=4, pd=2, device="cpu")
    return (
        A.to(device=dev, dtype=dtype),
        b.to(device=dev, dtype=dtype),
        x0.to(device=dev, dtype=dtype),
    )


SOLVE = {
    "dmrg2": lambda A, b, x, chi: tt.linsys_dmrg2(
        A, b, x, trunc_eps=1e-10, max_rank=chi, max_iter=2
    ),
    "amen": lambda A, b, x, chi: tt.linsys_amen(A, b, x, max_rank=chi, max_iter=2, eps=1e-10),
}
CASES = [  # (solver, sites per axis, bond dimensions, dtypes)
    ("dmrg2", 8, (10, 20, 30, 40, 60), (torch.float64, torch.float32)),
    ("amen", 8, (10, 20, 40, 80), (torch.float64, torch.float32)),
    ("amen", 10, (10, 20, 40, 80), (torch.float64, torch.float32)),  # local solves are dense
    ("dmrg2", 10, (10, 20, 40), (torch.float64,)),
]

row("part,device,dtype,sites,chi,seconds,runs,peak_mb,note")
A, b, x0 = problem(6, torch.float64)
SOLVE["dmrg2"](A, b, x0, 10)  # warm-up: CUDA context, cuBLAS/cuSOLVER handles
for name, L, chis, dtypes in CASES if len(sys.argv) == 2 else ():
    for dtype in dtypes:
        torch.set_default_dtype(dtype)
        A, b, x0 = problem(L, dtype)
        for chi in chis:
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                s, n = timed(lambda: SOLVE[name](A, b, x0, chi))
                row(name, dev, str(dtype)[6:], 2 * L, chi, f"{s:.3f}", n, f"{peak():.0f}", "")
            except Exception as exc:  # out of memory or an unported path: record, move on
                row(
                    name,
                    dev,
                    str(dtype)[6:],
                    2 * L,
                    chi,
                    "",
                    0,
                    "",
                    f"{type(exc).__name__}: {exc}"[:200].replace(",", ";"),
                )
                if dev == "cuda":
                    torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

# --- one ENEC time step with the production settings ---------------------------
from _paths import MASK_DIR, RESULTS, SOLVER

sys.path.insert(0, SOLVER)
import bundle_simulation as es
import chorin_projection

TAG, STEP = "p1A_inc_it6", 500
R = RESULTS
cfg = tt.load_object(os.path.join(R, f"{TAG}_params.pfile"))
Q = cfg["Q"]
tt.set_bd_limit(cfg["max_bd"])
tt.set_accuracy(cfg["eps"])
tt.set_comp_method("randomized_svd")
for k, v in (
    ("ax", cfg["ax"]),
    ("bx", cfg["bx"]),
    ("Lx", Q),
    ("ay", cfg["ay"]),
    ("by", cfg["by"]),
    ("Ly", Q),
    ("ux_val", cfg["u"]),
    ("low_acc", False),
):
    tt.set_param(k, v)
tt.set_device(dev)
try:
    ux, uy, p = (f.to(device=dev) for f in tt.load_object(es.find_snapshots(TAG, R)[STEP - 1][2]))
    mask = es.load_or_build_mask(cfg["p"], cfg["Lf"], cfg["d_over_pitch"], MASK_DIR).to(device=dev)
    ops = chorin_projection.Operators(
        Q,
        cfg["dx"],
        cfg["dy"],
        bcv=chorin_projection.BCV_CHANNEL,
        bcp=chorin_projection.BCP_CHANNEL,
    )
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    guess, times = None, []
    for k in range(4):
        sync()
        t0 = time.perf_counter()
        ux, uy, p, guess = chorin_projection.projection_step(
            ux,
            uy,
            ops,
            cfg["dt"],
            cfg["nu"],
            mask,
            cfg["eps"],
            cfg["max_bd"],
            cfg["opt_iter"],
            guess=guess,
            solver=cfg["solver"],
            pressure=p,
            rotational=True,
            guard=cfg["poisson_guard"],
        )
        sync()
        times.append(time.perf_counter() - t0)
        if k == 0:  # first step, for the CPU-GPU comparison
            np.savez(
                f"enec_step1_{'_'.join(sys.argv[1:])}.npz",
                ux=chorin_projection.mps_to_grid(ux.to(device="cpu"), Q),
                uy=chorin_projection.mps_to_grid(uy.to(device="cpu"), Q),
            )
    row(
        "enec_step",
        dev,
        "float64",
        2 * Q,
        cfg["max_bd"],
        f"{statistics.median(times[1:]):.3f}",
        len(times) - 1,
        f"{peak():.0f}",
        "first step %.1f s" % times[0],
    )
except Exception as exc:
    import traceback

    traceback.print_exc(file=sys.stderr)
    row(
        "enec_step",
        dev,
        "float64",
        2 * Q,
        cfg["max_bd"],
        "",
        0,
        "",
        f"{type(exc).__name__}: {exc}"[:200].replace(",", ";"),
    )
