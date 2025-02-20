# %%
import sys
from datetime import datetime

import numpy as np
import torch
import tensnet as tt

torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")


# %% Importing arguments from CLI
if len(sys.argv) == 7:
    ux_val = float(sys.argv[1])
    N = int(sys.argv[2])
    nu = float(sys.argv[3])
    dt = float(sys.argv[4])
    Lx = int(sys.argv[5])
    Ly = int(sys.argv[6])
else:
    raise SyntaxError(
        "Please provide all arguments in the order of InitialVelinX,Timesteps,Viscosity,SizeOfTimestep"
    )
# %%
ux_val = 1.0
N = 10
nu = 1.0
dt = 1e-3
Lx = 12
Ly = 12
# %% Setting various parameters
d = 2  # Physical dimensions
ax = 0.0
ay = 0.0
bx = 1.0
by = 1.0

rad = 0.15
Re = np.round(ux_val * 2 * rad / nu)
eps = 1e-16

dx = (bx - ax) / (d**Lx - 1)
dy = (by - ay) / (d**Ly - 1)

opt_iter = 4  # DMRG Iterations
# %%
tt.set_bd_limit(150)
tt.set_accuracy(eps)
tt.set_arch("sequential")
tt.set_comp_method("svd")
tt.set_param("ax", ax)
tt.set_param("bx", bx)
tt.set_param("Lx", Lx)
tt.set_param("ay", ay)
tt.set_param("by", by)
tt.set_param("Ly", Ly)
tt.set_param("ux_val", ux_val)
tt.set_param(
    "low_acc", False
)  # Finite difference scheme (low-acc if True = 3rd order central diff)

cfl_cond = ux_val * dt / dx

# %%
mask_mps = tt.load_object("mps/enec_mask.mps")

mask_mps = mask_mps.compress(accuracy=eps)
var = int(datetime.now().timestamp())
# %%
params = {
    "lenx": bx - ax,
    "leny": by - ay,
    "Re": Re,
    "CFL Number": np.round(cfl_cond, 2),
    "NoQ": [Lx, Ly],
    "dx": dx,
    "dt": dt,
    "low_acc": tt.parameters("low_acc"),
    "mask": mask_mps,
    "ux_val": ux_val,
    "visc": nu,
    "Iteration limit for DMRG": opt_iter,
    "TimeSteps": N,
    "UID": var,
    "config_params": tt.parameters(),
}

print("Parameters:" + f"{params}")

tt.save_object(params, f"../results/param_{var}.pfile")

# %% Initial Conditions

print("Default initialization of velocity fields")
ux = mask_mps * ux_val
uy = tt.identity_mps(L=Lx + Ly) * 0.0


# %% Time Stepping
solution = tt.euler_stepping_dmrg(
    ux=ux,
    uy=uy,
    nofq=[Lx, Ly],
    dt=dt,
    dx=dx,
    dy=dy,
    N=N,
    nu=nu,
    mask=mask_mps,
    bdtrunc=tt.BD_LIMIT(),
    opt_iter=opt_iter,
    saveit=100,
    UID=var,
)

# %%
tt.save_object(solution, f"../results/Re_{int(Re)}_{var}.mpssol")
