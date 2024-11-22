import matplotlib.pyplot as plt
import numpy
from matplotlib import patches

from .config import parameters


def mask_clean(x, y, flag):
    bx = parameters("bx")
    by = parameters("by")
    rad = 0.15
    if flag == 0:
        values = (x - bx / 8) ** 2 + (y - by / 2) ** 2 <= rad**2
    elif flag == 1:
        values = (x - bx / 4) ** 2 + (y - by / 2) ** 2 <= rad**2
    return values



def plot_ux_uy_rect(
    ux, uy, startx=0, stopx=None, starty=0, stopy=None, d=2, L=[9, 8], title=None
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ux_ans = ux.contract().reshape(d ** L[0], d ** L[1]).T
    uy_ans = uy.contract().reshape(d ** L[0], d ** L[1]).T
    p0 = ax[0].imshow(ux_ans[starty:stopy, startx:stopx].numpy(), aspect="equal")
    ax[0].set_xlabel(r"$V_x$")
    plt.colorbar(p0, ax=ax[0])
    p1 = ax[1].imshow(uy_ans[starty:stopy, startx:stopx].numpy(), aspect="equal")
    ax[1].set_xlabel(r"$V_y$")
    plt.colorbar(p1, ax=ax[1])
    if title:
        plt.suptitle(title)
    plt.show()


def plot_pressure_rect(
    ux, uy, startx=0, stopx=None, starty=0, stopy=None, d=2, L=[9, 8], title=None
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ux_ans = ux.contract().reshape(d ** L[0], d ** L[1]).T
    uy_ans = uy.contract().reshape(d ** L[0], d ** L[1]).T
    p0 = ax[0].imshow(ux_ans[starty:stopy, startx:stopx].numpy(), aspect="equal")
    ax[0].set_xlabel(r"$pressure$")
    plt.colorbar(p0, ax=ax[0])
    p1 = ax[1].imshow(uy_ans[starty:stopy, startx:stopx].numpy(), aspect="equal")
    ax[1].set_xlabel(r"$error$")
    plt.colorbar(p1, ax=ax[1])
    if title:
        plt.suptitle(title)
    plt.show()

def plot_streamlines_rect(
    ux1,
    uy1,
    startx=0,
    stopx=None,
    starty=0,
    stopy=None,
    density=3,
    lw=0.5,
    d=2,
    cleanmask=None,
    L=None,
    broken_streamlines=False,
    flag=0,
):
    if L is None:
        Lx = parameters("Lx")
        Ly = parameters("Ly")
    else:
        Lx = L[0]
        Ly = L[1]
    X, Y = numpy.meshgrid(
        numpy.linspace(parameters("ax"), parameters("bx"), d**Lx),
        numpy.linspace(parameters("ay"), parameters("by"), d**Ly),
    )
    if cleanmask is None:
        matrix_clean = mask_clean(X, Y, flag)
    else:
        matrix_clean = cleanmask
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    x_v = X[starty:stopy, startx:stopx]
    y_v = Y[starty:stopy, startx:stopx]
    ux_ans = ux1.contract().numpy().reshape(d**Lx, d**Ly).T
    uy_ans = uy1.contract().numpy().reshape(d**Lx, d**Ly).T
    u_mag = numpy.sqrt(ux_ans**2 + uy_ans**2)
    ux_ans[matrix_clean] = float("nan")
    uy_ans[matrix_clean] = float("nan")

    h = ax.streamplot(
        x_v,
        y_v,
        ux_ans[starty:stopy, startx:stopx],
        uy_ans[starty:stopy, startx:stopx],
        color=u_mag[starty:stopy, startx:stopx],
        cmap="plasma",
        density=density,
        linewidth=lw,
        broken_streamlines=broken_streamlines,
        # arrowstyle="-",
    )
    ax.set_aspect("equal")
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    fig.colorbar(h.lines, cax=cax)
    mt = ~cleanmask[starty:stopy, startx:stopx]
    ax.invert_yaxis()
    ax.imshow(
        mt.numpy(),
        extent=(x_v[0, 0], x_v[0, -1], y_v[-1, 0], y_v[0, 0]),
        alpha=0.7,
        cmap="gray",
        aspect="equal",
    )
    for art in ax.get_children():
        if not isinstance(art, patches.FancyArrowPatch):
            continue
        art.set_visible(False)
    plt.show()


def plot_quiver(
    ux1,
    uy1,
    cleanmask,
    startx=0,
    stopx=None,
    starty=0,
    stopy=None,
    stepx=15,
    stepy=15,
    scale=None,
    norm=True,
    d=2,
    L=[8, 7],
):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    x_val, y_val = numpy.meshgrid(
        numpy.linspace(parameters("ax"), parameters("bx"), d ** L[0]),
        numpy.linspace(parameters("ay"), parameters("by"), d ** L[1]),
    )
    x_val = x_val[starty:stopy:stepy, startx:stopx:stepx].numpy()
    y_val = y_val[starty:stopy:stepy, startx:stopx:stepx].numpy()
    ux_ans = ux1.contract().reshape(d ** L[0], d ** L[1]).T
    uy_ans = -1 * uy1.contract().reshape(d ** L[0], d ** L[1]).T
    ux_ans[cleanmask] = float("nan")
    uy_ans[cleanmask] = float("nan")
    ux_ans = ux_ans[starty:stopy:stepy, startx:stopx:stepx].numpy()
    uy_ans = uy_ans[starty:stopy:stepy, startx:stopx:stepx].numpy()
    u_mag = numpy.sqrt(ux_ans**2 + uy_ans**2) if norm else numpy.ones_like(ux_ans)
    ax.quiver(x_val, y_val, ux_ans / u_mag, uy_ans / u_mag, [u_mag], scale=scale)
    ax.set_aspect("equal")
    mt = ~cleanmask[starty:stopy, startx:stopx]
    ax.invert_yaxis()
    ax.imshow(
        mt.numpy(),
        extent=(x_val[0, 0], x_val[0, -1], y_val[-1, 0], y_val[0, 0]),
        alpha=0.7,
        cmap="gray",
        aspect="equal",
    )

    plt.show()
