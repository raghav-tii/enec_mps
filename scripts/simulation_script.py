from datetime import datetime
import numpy as np
import torch
import os
import sys

sys.path.append("..")

from tensnet import (
    set_accuracy,
    set_bd_limit,
    load_object,
    save_object,
    set_param,
    modify_2inputs,
    cross_approximation,
    euler_time_stepping,
    COMP_ACC,
    BD_LIMIT,
    parameters,
    identity_mps,
)

if len(sys.argv) == 8:
    ux_val = float(sys.argv[1])
    N = int(sys.argv[2])
    nu = float(sys.argv[3])
    dt = float(sys.argv[4])
    object = sys.argv[5]
    Lx = int(sys.argv[6])
    Ly = int(sys.argv[7])
else:
    raise AssertionError("Please provide all arguments as specified in README.md")

# %%
flag_init_conds = False  # Set to true if initial conditions are provided to overrite the default ones.
d = 2  # Currenltly only 2d simulation are supported
ax = 0.0
ay = 0.0
bx = 8.0
by = 4.0
rad = 0.15
Re = np.round(ux_val * 2 * rad / nu)

if (
    object == "square"
    or object == "squares"
    or object == "wing"
    or object == "wing_incl"
):
    ax = -2.0
    bx = 6.0
    ay = -2.0
    by = 2.0

if object == "square" or object == "squares":
    rad = 0.3
    Re = np.round(ux_val * rad / nu)
elif object == "wing" or object == "wing_incl":
    rad = 0.24
    Re = np.round(ux_val * 1.0 / nu)


x = np.linspace(ax, bx, d**Lx)
y = np.linspace(ay, by, d**Ly)
dx = (x[1] - x[0]).item()
dy = (y[1] - y[0]).item()

var = int(datetime.now().timestamp())

method = "dmrg2"
timestep_scheme = "first"
opt_acc = 1e-6
opt_iter = 2
# %%
eps = 1e-16
set_bd_limit(30)
set_accuracy(eps)
set_param("ax", ax)
set_param("bx", bx)
set_param("Lx", Lx)
set_param("ay", ay)
set_param("by", by)
set_param("Ly", Ly)
set_param("ux_val", ux_val)
set_param("low_acc", False)

cfl_cond = ux_val * dt / dx
int_Re = int(Re)


# %%
def circle(x, y):
    # Be careful with the domain. Function support should be btw 0 and 4.
    if object == "new":
        values = (x - bx / 16) ** 2 + (y - by / 2) ** 2 - rad**2
    else:
        values = (x - bx / 8) ** 2 + (y - by / 2) ** 2 - rad**2
    values += torch.abs(values)
    return values


def squares(x, y):
    square1 = torch.abs(x + y + 0.5) + torch.abs(y - x - 0.5) - 0.2
    square2 = torch.abs(x + y - 0.5) + torch.abs(y - x - 0.5) - 0.2
    square3 = torch.abs(x + y + 0.5) + torch.abs(y - x + 0.5) - 0.2
    square4 = torch.abs(x + y - 0.5) + torch.abs(y - x + 0.5) - 0.2
    values = square1 * square2 * square3 * square4
    values += torch.abs(values)
    return values


def square(x, y):
    values = torch.abs(x + y + 0.5) + torch.abs(y - x - 0.5) - rad
    values += torch.abs(values)
    return values


def symmetric_wing(x, y):
    t = 5 * rad
    x0 = -1.0
    if object == "wing_incl":
        k = 0.4
    else:
        k = 0.0
    csq = 0.2969 * t
    c1 = -0.1260 * t
    c2 = -0.3516 * t
    c3 = 0.2843 * t
    c4 = -0.1015 * t
    arg = torch.where(
        (torch.abs(x - 0.5 - x0) <= 0.5),
        x - x0,
        torch.sign(x0 - x) + torch.sign(x - x0 - 1),
    )
    v = c1 * arg + c2 * arg**2 + c3 * arg**3 + c4 * arg**4 + csq * arg**0.5
    values = v**2 - (y - k * (arg - 0.5)) ** 2
    values += torch.abs(values)
    return values


@modify_2inputs
def mask(x, y):
    if object == "square":
        values = 1 - torch.exp(-1e3 * square(x, y))
    elif object == "squares":
        values = 1 - torch.exp(-1e3 * squares(x, y))
    elif object == "wing" or object == "wing_incl":
        values = torch.exp(-1e4 * symmetric_wing(x, y))
    elif object == "circle":
        values = 1 - torch.exp(-1e3 * circle(x, y))
    else:
        raise RuntimeError(
            "Expected the immersed object to be among {square,squares, wing, wing_incl or circle}"
        )
    return values


# %%

mask_mps = cross_approximation(
    function=mask, nofq=Lx + Ly, pd=2, eps=1e-6, max_iter=100
)
mask_mps.compress(accuracy=eps, max_bd=None)
# %%

params = {
    "lenx": bx - ax,
    "leny": by - ay,
    "Re": Re,
    "CFL": np.round(cfl_cond, 3),
    "NoQ": [Lx, Ly],
    "method": method,
    "scheme": timestep_scheme,
    "dx": dx,
    "dt": dt,
    "low_acc": parameters("low_acc"),
    "ResumeInit": flag_init_conds
    and os.path.exists(f"../data/{object}initcondx_{Lx+Ly}_{int(Re)}.mps"),
    "new_mask": object,
    "mask": mask_mps,
    "ux_val": ux_val,
    "visc": nu,
    "DMRG_Threshold": opt_acc,
    "DMRG_maxSweeps": opt_iter,
    "Timesteps": N,
    "UID": var,
    "config": parameters(),
    "bd_limit": BD_LIMIT(),
    "comp_acc": COMP_ACC(),
}

save_object(params, f"../results/param_{var}.pfile")
# %% Initial Conditions
if (
    os.path.exists(f"../data/{object}initcondx_{Lx+Ly}_{int_Re}.mps")
    and flag_init_conds
):
    ux = load_object(f"../data/{object}initcondx_{Lx+Ly}_{int_Re}.mps")
    uy = load_object(f"../data/{object}initcondy_{Lx+Ly}_{int_Re}.mps")
else:
    ux = mask_mps * ux_val
    uy = identity_mps(L=Lx + Ly) * 0.0
# %% Time Stepping
solution = euler_time_stepping(
    ux=ux,
    uy=uy,
    nofq=[Lx, Ly],
    dt=dt,
    dx=dx,
    dy=dy,
    N=N,
    nu=nu,
    mask=mask_mps,
    bdtrunc=BD_LIMIT(),
    opt_method=method,
    opt_accuracy=opt_acc,
    opt_iter=opt_iter,
    opt_paths_iter=100,
    scheme=timestep_scheme,
    saveit=None,
    UID=var,
)


save_object(solution, f"../results/Re_{int(Re)}_{method}_{var}.mpssol")
