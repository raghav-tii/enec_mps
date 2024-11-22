# %%
import sys

sys.path.append("..")

# %%
import os
import tensnet as tt
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
# %%
UID = 1732186808 #None
params = tt.load_object(f"../data/results/param_{UID}.pfile")

if os.path.exists(
    f"../data/results/Re_{int(params['Re'])}_{params['method']}_{UID}.mpssol"
):
    results = tt.load_object(
        f"../data/results/Re_{int(params['Re'])}_{params['method']}_{UID}.mpssol"
    )
else:
    (velx, vely, pressure) = tt.load_object(
        f"../data/results/temp/{UID}_int_sol_{0}.tmp"
    )
    for i in range(1, 2):
        (solx1, soly1, solp1) = tt.load_object(
            f"../data/results/temp/{UID}_int_sol_{100*i}.tmp"
        )
        velx += solx1
        vely += soly1
        pressure += solp1

if len(results) == 4:
    velx, vely, pressure, _ = results
elif len(results) == 3:
    velx, vely, pressure = results

# %%
ax = 0.0
bx = params["lenx"]
ay = 0.0
by = params["leny"]
Lx = params["NoQ"][0]
Ly = params["NoQ"][1]
nu = params["visc"]
# %%
startx = 0
stopx = None
starty = 0
stopy = None
# %%
d = 2
L = [Lx, Ly]
eps = 1e-12
tt.set_accuracy(eps)
x = torch.linspace(ax, bx, d**Lx)
y = torch.linspace(ay, by, d**Ly)
dx = (bx - ax) / (2**Lx - 1)
dy = (by - ay) / (2**Ly - 1)
dt = params["dt"]
# %%
mask_mps = tt.mps_to_torch(params["mask"])
mask_mps.compress(accuracy=eps, max_bd=None)
print(mask_mps)
cleanmask = torch.round(1 - mask_mps.contract().reshape(d**Lx, d**Ly).T) == 1.0
# %%
timestep = -1
ux = velx[timestep]
uy = vely[timestep]
###################################
print(len(velx))
print(len(vely))
print(len(pressure))
p  = pressure[timestep]
p_ex = uy
error = p + uy*-1.0
tt.plot_pressure_rect(p, error, L=L)
###################################
# tt.plot_ux_uy_rect(ux, uy, L=L)
# %%
# # %% ENERGY CONSERVATION CHECK.
# Emean = []
# TEmean = []
# vymean = torch.zeros_like(velx[0].contract())
# vxmean = torch.ones_like(vely[0].contract()) * params["ux_val"]

# for mpsx, mpsy in zip(tqdm(velx), tqdm(vely)):
#     ux = mpsx.contract()
#     uy = mpsy.contract()
#     uxd = ux**2 - vxmean**2
#     uyd = uy**2 - vymean**2
#     Emean.append(0.5 * torch.mean(ux**2 + uy**2))
#     TEmean.append(0.5 * torch.mean(uxd + uyd))
# # %%
# Emean = torch.tensor(Emean)
# plt.plot(Emean.numpy())
# print(Emean.mean(), "+-", Emean.std())
# # %%
# plt.plot(TEmean)
