#%%
import tensnet as tt
import numpy as np
import itertools
import torch

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# %%
tt.set_accuracy(1e-16)
tt.set_bd_limit(None)
tt.set_param("low_acc", False)


# %%
def cylinder(x, y, xc, yc, r, alpha=1e3):

    values = (x - xc) ** 2 + (y - yc) ** 2 - r**2
    values += torch.abs(values)

    values = torch.exp(-alpha * values)

    return values


def get_cylinders_array(
    x,
    y,
    grid_coordinates,
    r,
    position_array,
    spacing,
    shift=None,
):
    grid = position_array.shape
    if shift is None:
        shift = np.zeros(grid)
    points = np.array(
        list(itertools.product(*[np.arange(dim) + s for dim, s in zip(grid, shift)]))
    )
    mask = position_array.flatten() == 1

    array = None
    gx = grid_coordinates[0]
    gy = grid_coordinates[1]

    for xc, yc in points[mask]:
        if array is None:
            array = cylinder(x, y, xc * spacing + gx, yc * spacing + gy, r)
        else:
            array += cylinder(x, y, xc * spacing + gx, yc * spacing + gy, r)

    return array


# %%
ax = -0.3 / 2
bx = 0.3 / 2
ay = -0.3 / 2
by = 0.3 / 2

#%%
Lx = 8
Ly = 8
dx = (bx - ax) / (2**Lx - 1)
dy = (by - ay) / (2**Ly - 1)

tt.set_param("ax", ax)
tt.set_param("bx", bx)
tt.set_param("Lx", Lx)

tt.set_param("ay", ay)
tt.set_param("by", by)
tt.set_param("Ly", Ly)

tt.set_param("dx", dx)
tt.set_param("dy", dy)

tt.set_arch("sequential")
# %%
grid = [5, 5]
positions = np.ones(grid)
positions[0, 0] = 0
positions[0, -1] = 0
positions[-1, 0] = 0
positions[-1, -1] = 0
gc = [0.0, 0.0]

x = torch.linspace(ax, bx, 2**Lx)
y = torch.linspace(ay, by, 2**Ly)
x, y = torch.meshgrid(x, y)

r = 0.1
p = 0.3

# %%
print(f"Generating a {grid}-grid of cylinders with radius = {r}")

ax = -1
bx = 1
ay = -1
by = 1

x1 = torch.linspace(-1, 1, 2**Lx)
y1 = torch.linspace(-1, 1, 2**Ly)
x1, y1 = torch.meshgrid(x1, y1)
mask = get_cylinders_array(x1, y1, gc, r, positions, p, shift=[-2, -2])
plt.imshow(mask.T, aspect=dy / dx)
plt.colorbar()


# %%
@tt.modify_2inputs
def mask_oracle(x, y):
    return get_cylinders_array(x, y, gc, r, positions, p, shift=[-2, -2])


@tt.modify_2inputs
def cylinder_oracle(x, y):
    return cylinder(x, y, 0, 0, r)


# %%
mask = cylinder(x, y, 0, 0, r)
plt.imshow(mask.T, aspect=dy / dx)
plt.colorbar()
#%%
mask_mps_svd=tt.vector_to_mps(tt.shuffle_axis(mask,Lx).flatten(),2,Lx+Ly)
# %%
mask_mps_cross = tt.cross_approximation(
    cylinder_oracle,
    Lx + Ly,
    2,
    max_rank=100,
    max_iter=35,
    verbose=True,
)
#%%
mask_mps = mask_mps_svd.compress()
# %%
plt.imshow(tt.unshuffle_axis(mask_mps.contract().reshape(2**Lx, 2**Ly),Lx,2).T, aspect=dy / dx)
plt.colorbar(shrink=0.525)
plt.show()
#%%
mask_mps
#%%
p = 4
pad_pos = torch.from_numpy(np.pad(positions,((6,5),(3,8))))
pos_mps = tt.vector_to_mps(tt.shuffle_axis(pad_pos,p).flatten(),2,2*p,tt.COMP_ACC())
print(pos_mps)

#%%
plt.imshow(tt.unshuffle_axis(pos_mps.contract().reshape(2 ** p, -1),p,2).T, aspect=dy / dx)
plt.colorbar(shrink=0.525)
plt.show()

# %%
t1 = torch.zeros((1, 2, 1))
t1[:, 0, :] = 1
t1[:, 1, :] = 1

k=1
# k = mask_mps.ranks_list[Lx]
t2 = torch.zeros((k, 2, k))
t2[:, 0, :] = torch.diag(torch.ones(k))
t2[:, 1, :] = torch.diag(torch.ones(k))

q1 = tt.MPS_Node(t1)
q2 = tt.MPS_Node(t2)

unit_mps = tt.list_to_mps([t1]*4)
delta_mps = tt.delta(4,10)

mps1 = unit_mps + delta_mps*-1
# %%
mps2 = tt.identity_mps(24)-tt.MPS(pos_mps.nodes + mask_mps.nodes)
print(mps2)
# %%
plt.imshow(tt.unshuffle_axis(mps2.contract().reshape(2 ** (Lx + p), -1),Lx+p,2).T, aspect=dy / dx)
plt.colorbar(shrink=0.525)
plt.show()


# %%
tt.save_object(mps2,"enec_mask.mps")
# %% Approach 2
def get_cylinders_array_mps(
    qubits,
    max_rank,
    max_iter,
    grid_coordinates,
    r,
    position_array,
    spacing,
    shift=None,
):
    grid = position_array.shape
    if shift is None:
        shift = np.zeros(grid)
    points = np.array(
        list(itertools.product(*[np.arange(dim) + s for dim, s in zip(grid, shift)]))
    )
    mask = position_array.flatten() == 1
    mask_mps = None

    gx = grid_coordinates[0]
    gy = grid_coordinates[1]

    for xc, yc in points[mask]:
        if mask_mps is None:

            @tt.modify_2inputs
            def func(x, y):
                return cylinder(x, y, xc * spacing + gx, yc * spacing + gy, r)

            mask_mps = tt.cross_approximation(
                func,
                qubits,
                2,
                max_rank=max_rank,
                max_iter=max_iter,
                verbose=False,
            )
            print(mask_mps.max_rank)

        else:

            @tt.modify_2inputs
            def func(x, y):
                return cylinder(x, y, xc * spacing + gx, yc * spacing + gy, r) - 1

            temps = tt.cross_approximation(
                func,
                qubits,
                2,
                max_rank=max_rank,
                max_iter=max_iter,
                verbose=False,
            )
            mask_mps = mask_mps + temps
            print(mask_mps.max_rank)

        plt.imshow(mask_mps.contract().reshape(2**Lx, 2**Ly).T, aspect=dy / dx)
        plt.colorbar(shrink=0.525)
        plt.show()

    return mask_mps


# %%
mask_mps = get_cylinders_array_mps(Lx + Ly, 35, 15, gc, r, positions, p, shift=[-2, -2])
# %%
mask_mps
# %%
plt.imshow(mask_mps.contract().reshape(2**Lx, 2**Ly).T, aspect=dy / dx)
plt.colorbar(shrink=0.525)
# %%
