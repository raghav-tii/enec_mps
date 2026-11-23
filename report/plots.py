"""Figures and tables for the ENEC validation report.

    cd report && python3 plots.py          # or: make report

Reads    data/archive (rod forces, field extracts, ranks, ...) and data/star_ccm (monitors, snapshots);
         QCFD_REPORT_DATA / QCFD_REFERENCE override those directories (see _paths.py)
Writes   report/fig/*.pdf (the figures) and report/fig/{numbers,rods,rods11,table}.tex: the numbers
         and tables quoted in the report, one macro or row each
Needs    LaTeX (matplotlib usetex).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from star import star_on_grid

from _paths import OUT, REF
MAIN = "q10_t2500_bd40_rk4"
# Times are reported from the start of the Star-CCM+ run: the MPS runs start from its snapshot at tU/D = 2500 (cyl_array@250000000.vtk)
T0 = 2500.0
SOLVER = "MPS solver"
COLOR = {MAIN: "#2a78d6", "q11": "#1baf7a", "ref": "#0b0b0b", "cv": "#eb6834"}
INK, MUTED, SOLID = "#52514e", "#b5b3ad", "#8a8983"
W = 6.5    # text width in inches
os.makedirs("fig", exist_ok=True)
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 9,
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "lines.linewidth": 1.3, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def tag(a, s):
    a.set_title(s, loc="left", fontsize=9)


def load_ref(name):
    rows = list(csv.reader(open(os.path.join(REF, name))))
    head = [c.split(" Monitor")[0].split("_")[-1] for c in rows[0][1:]]
    a = np.array([[float(x) for x in r] for r in rows[1:]])
    return head, a[:, 0], a[:, 1:]


heads, tref, CD = load_ref("C_D.csv")
_, _, CL = load_ref("C_L.csv")
tail = slice(int(0.8 * len(tref)), None)
ref_cd = dict(zip(heads, CD[tail].mean(0)))
ref_cl = dict(zip(heads, CL[tail].mean(0)))

F = np.load(f"{OUT}/fields.npz")
Q11 = "q11_t2500_bd60_rk4"   # 2048 x 2048, 128D box, chi = 60
D = float(F["cfg_D"]) / float(F["cfg_u"])  # time unit D/U in solver time


# rod diameter in solver time units (results/<tag>_params.pfile "D"): the 2048^2 run
# has a different box, hence a different D, and its times must use its own
D_TAG = {Q11: 0.007816316560820713}


def load_run(tag):
    D = D_TAG.get(tag, globals()["D"])
    rows = list(csv.reader(open(f"{OUT}/rod_forces_{tag}.csv")))
    names = [c[3:] for c in rows[0] if c.startswith("Cd_")]
    a = np.array([[float(x) for x in r] for r in rows[1:]])
    n = len(names)
    return dict(names=names, t=T0 + a[:, 1] / D, cd=a[:, 3:3 + n], cl=a[:, 3 + n:3 + 2 * n])


runs = {MAIN: load_run(MAIN), Q11: load_run(Q11)}
names = runs[MAIN]["names"]
# Star-CCM+ monitors on the MPS time axis: the reference is its actual time series
# at the same tU/D as each MPS step, not its late-time average.  rcd/rcl are the
# means of that series over the MPS run's own window (per-rod bars, tables).
star_at = lambda A, t: np.column_stack([np.interp(t, tref, A[:, heads.index(n)]) for n in names])
st_cd, st_cl = star_at(CD, runs[MAIN]["t"]), star_at(CL, runs[MAIN]["t"])
rcd, rcl = st_cd.mean(0), st_cl.mean(0)
order = sorted(range(len(names)), key=lambda i: (names[i][1], names[i][0]))
# per-rod values: mean and spread over the whole run (it never settles)
m_cd, s_cd = runs[MAIN]["cd"].mean(0), runs[MAIN]["cd"].std(0)
m_cl, s_cl = runs[MAIN]["cl"].mean(0), runs[MAIN]["cl"].std(0)
corr = lambda a, b: np.corrcoef(a, b)[0, 1]

BAND = 50  # steps in the moving window of the time-series bands


def band(a, t, y, c, lw=1.3, **kw):
    """Raw trace (faint), moving average and a +-1 std band (50 % opacity)."""
    k = np.ones(BAND) / BAND
    m = np.convolve(y, k, "valid")
    sd = np.sqrt(np.maximum(np.convolve(y ** 2, k, "valid") - m ** 2, 0))
    tc = np.convolve(t, k, "valid")
    a.plot(t, y, color=c, lw=0.5, alpha=0.35)
    a.fill_between(tc, m - sd, m + sd, color=c, alpha=0.5, lw=0)
    return a.plot(tc, m, color=c, lw=lw, **kw)


# --- bundle-mean history: both MPS runs vs the Star-CCM+ monitors ----------------
# top row: the whole 1024^2 run (the 2048^2 run covers its first few time units);
# bottom row: that early window, both runs
r10, r11 = runs[MAIN], runs[Q11]
t11 = r11["t"][-1]
fig, ax = plt.subplots(2, 2, figsize=(W, 4.4))
for row, (tmax, lab) in enumerate(((r10["t"][-1], ""), (t11, " (early window)"))):
    m10 = r10["t"] <= tmax + 1e-9
    for k, key in enumerate(("cd", "cl")):
        a = ax[row, k]
        band(a, r10["t"][m10], r10[key][m10].mean(1), COLOR[MAIN], label=f"{SOLVER}, half domain")
        band(a, r11["t"], r11[key].mean(1), COLOR["q11"], label=f"{SOLVER}, full domain")
        ws = (tref >= T0) & (tref <= tmax + 1)
        S = (CD if key == "cd" else CL)[ws].mean(1)
        a.plot(tref[ws], S, color=COLOR["ref"], ls="--", lw=1, label="Star-CCM+")
        a.set_ylabel(r"$\langle C_D\rangle$" if k == 0 else r"$\langle C_L\rangle$")
        a.set_xlabel("$tU/D$"); tag(a, f"({'abcd'[2 * row + k]})")
        if row == 1: a.set_xlim(T0, tmax)
fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 0.98))
fig.tight_layout(); fig.savefig("fig/history.pdf"); plt.close(fig)

# --- per-rod bars: reference vs solver -----------------------------------------
k11 = [runs[Q11]["names"].index(n) for n in names]  # full-domain run: its own window, its own Star-CCM+ mean
q11 = {key: (runs[Q11][key][:, k11].mean(0), runs[Q11][key][:, k11].std(0),
             star_at(A, runs[Q11]["t"]).mean(0)) for key, A in (("cd", CD), ("cl", CL))}
x = np.arange(len(names))
fig, ax = plt.subplots(2, 1, figsize=(W, 3.6), sharex=True)
for k, (R, M, S, lab) in enumerate(((rcd, m_cd, s_cd, "$C_D$"), (rcl, m_cl, s_cl, "$C_L$"))):
    qM, qS, _ = q11["cd" if k == 0 else "cl"]
    ax[k].bar(x - 0.27, R[order], 0.26, color=COLOR["ref"], label="Star-CCM+")
    ax[k].bar(x, M[order], 0.26, color=COLOR[MAIN], label=f"{SOLVER}, half domain",
              yerr=S[order], error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK))
    ax[k].bar(x + 0.27, qM[order], 0.26, color=COLOR["q11"], label=f"{SOLVER}, full domain",
              yerr=qS[order], error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK))
    ax[k].set_ylabel(lab); ax[k].axhline(0, color=INK, lw=0.6)
    ax[k].grid(axis="x", visible=False)
    for b in (2.5, 7.5, 12.5, 17.5):  # column boundaries (3, 5, 5, 5, 3 rods)
        ax[k].axvline(b, color=MUTED, lw=0.6)
    tag(ax[k], f"({'ab'[k]})")
for c, xc in enumerate((1, 5, 10, 15, 19)):
    ax[0].text(xc, 1.02, f"column {c + 1}", transform=ax[0].get_xaxis_transform(),
               ha="center", fontsize=8, color=INK)
ax[1].set_xticks(x, [names[i] for i in order])
ax[1].set_xlabel("Rod")
ax[0].set_ylim(0, 1.75)
ax[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))
fig.tight_layout(); fig.savefig("fig/per_rod.pdf"); plt.close(fig)

# --- parity -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(W * 0.8, 2.5))
for k, (R, M, S, lab) in enumerate(((rcd, m_cd, s_cd, "C_D"), (rcl, m_cl, s_cl, "C_L"))):
    qM, qS, qR = q11["cd" if k == 0 else "cl"]
    lo, hi = min(R.min(), M.min(), qM.min(), 0) - 0.1, max(R.max(), M.max(), qM.max()) + 0.1
    ax[k].plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    ax[k].errorbar(R, M, yerr=S, fmt="o", ms=4, color=COLOR[MAIN], mec="white",
                   mew=0.5, elinewidth=0.7, ecolor=INK, capsize=1.5,
                   label="Half domain" if k == 0 else None)
    ax[k].errorbar(qR, qM, yerr=qS, fmt="s", ms=3.5, color=COLOR["q11"], mec="white",
                   mew=0.5, elinewidth=0.7, ecolor=INK, capsize=1.5,
                   label="Full domain" if k == 0 else None)
    ax[k].set_xlabel(f"${lab}$, Star-CCM+"); ax[k].set_ylabel(f"${lab}$, MPS solver")
    ax[k].set_aspect("equal"); ax[k].set_xlim(lo, hi); ax[k].set_ylim(lo, hi)
    tag(ax[k], f"({'ab'[k]})")
ax[0].legend(loc="upper left")
fig.tight_layout(); fig.savefig("fig/parity.pdf"); plt.close(fig)

# --- tables and numbers quoted in the text --------------------------------------
dev = 100 * (m_cd.mean() / rcd.mean() - 1)
worst = [names[i] for i in np.argsort(-abs(m_cd / rcd - 1))[:4]]  # largest per-rod deviations
with open("fig/rods.tex", "w") as fh:
    for i in order:
        fh.write(f"{names[i]} & {rcd[i]:.3f} & {m_cd[i]:.3f} & "
                 f"{100 * (m_cd[i] / rcd[i] - 1):+.0f} & {rcl[i]:.3f} & {m_cl[i]:.3f} \\\\\n")
with open("fig/numbers.tex", "w") as fh:
    for k, v in dict(
            tstart="%.0f" % T0, tend="%.1f" % runs[MAIN]["t"][-1],
            tdurmain="%.0f" % (runs[MAIN]["t"][-1] - T0),
            worstrods=", ".join(worst[:3]) + " and " + worst[3],
            clworst=names[int(np.argmax(abs(m_cl - rcl)))],
            clworstmps="%.2f" % m_cl[np.argmax(abs(m_cl - rcl))],
            clworstref="%.2f" % rcl[np.argmax(abs(m_cl - rcl))],
            cddev="%.1f" % abs(dev),
            trmsbundle="%.1f" % (100 * np.sqrt(np.mean((runs[MAIN]["cd"].mean(1) / st_cd.mean(1) - 1) ** 2))),
            trmsrod="%.0f" % (100 * np.sqrt(np.mean((runs[MAIN]["cd"] / st_cd - 1) ** 2))),
            tcorrmin="%.2f" % min(corr(a, b) for a, b in zip(runs[MAIN]["cd"], st_cd)),
            tcorrmean="%.3f" % np.mean([corr(a, b) for a, b in zip(runs[MAIN]["cd"], st_cd)]), cdsense="higher" if dev > 0 else "lower",
            cdcorr="%.2f" % corr(m_cd, rcd), clcorr="%.2f" % corr(m_cl, rcl),
            ).items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# --- the two rods furthest from Star-CCM+, through time: its monitors up to T0,
# where the MPS run starts from its state, then the MPS solver ------------------
rods3 = list(worst[:2]) + ["31"]
fig, axs = plt.subplots(2, 3, figsize=(W, 3.6), sharex=True)
TW = 10.0  # length of the Star-CCM+ history shown before T0
win = tref > T0 - TW
for c, rod in enumerate(rods3):
    i, j = names.index(rod), heads.index(rod)
    for r, (R, M, ref, lab) in enumerate(((CD, runs[MAIN]["cd"], rcd, "$C_D$"),
                                          (CL, runs[MAIN]["cl"], rcl, "$C_L$"))):
        a = axs[r, c]
        t1 = runs[MAIN]["t"][-1]
        w = win & (tref < t1 + 1)
        a.plot(tref[w], R[w, j], color=COLOR["ref"], label="Star-CCM+")
        band(a, runs[MAIN]["t"], M[:, i], COLOR[MAIN], lw=1.0, label=f"{SOLVER}, half domain")
        M11 = runs[Q11]["cd" if r == 0 else "cl"]
        band(a, runs[Q11]["t"], M11[:, i], COLOR["q11"], lw=1.0, label=f"{SOLVER}, full domain")
        a.axvline(T0, color=MUTED, lw=0.8)
        a.set_xlim(T0 - TW, t1)
        if c == 0: a.set_ylabel(lab)
        tag(a, f"({'abcdef'[3 * r + c]}) rod {rod}")
for a in axs[-1]: a.set_xlabel("$tU/D$")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 0.98))
fig.tight_layout(); fig.savefig("fig/rod_history.pdf"); plt.close(fig)
with open("fig/numbers.tex", "a") as fh:
    fh.write(f"\\newcommand{{\\twin}}{{{TW:.0f}}}\n")
    for k, rod in zip("abc", rods3):
        i, j = names.index(rod), heads.index(rod)
        cdm, clm = runs[MAIN]["cd"][:, i], runs[MAIN]["cl"][:, i]
        for key, v in dict(rod=rod, cdref="%.2f" % rcd[i], cdlo="%.2f" % cdm.min(), cdhi="%.2f" % cdm.max(),
                           clref="%.2f" % rcl[i], cllo="%.2f" % clm.min(), clhi="%.2f" % clm.max(),
                           cdstd="%.3f" % CD[win, j].std(), clstd="%.3f" % CL[win, j].std()).items():
            fh.write(f"\\newcommand{{\\r{k}{key}}}{{{v}}}\n")
# inflow of the MPS run (inlet.py): warm start, step 1, end
inl = np.genfromtxt(f"{OUT}/inlet_{MAIN}.csv", delimiter=",", names=True)
with open("fig/numbers.tex", "a") as fh:
    for key, v in dict(outdeficit="%.0f" % (100 * (1 - inl["out_mean"][0])),
                       inflowlo="%.3f" % inl["in_mean"].min(),
                       inflowhi="%.3f" % inl["in_mean"].max()).items():
        fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")

# --- velocity fields ---------------------------------------------------------
mask = F["mask"]
n = mask.shape[0]
cen = F["centres"]
h = 2.0 / 2 ** int(F["cfg_Lf"])        # cell size in D (pitch = 2D = 2**Lf cells)
cx, cy = cen[:, 0].mean(), cen[:, 1].mean()
X = (np.arange(n) - cx) * h
Y = (np.arange(n) - cy) * h
ix = np.where(np.abs(X) < 9)[0]; iy = np.where(np.abs(Y) < 7)[0]
sl = (slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1))
ext = [X[ix[0]], X[ix[-1]], Y[iy[0]], Y[iy[-1]]]
solid = mask[sl] < 0.5
star = star_on_grid(X[ix], Y[iy])
fld = lambda s, k: np.ma.masked_where(
    solid, star[..., "xy".index(k[1])] if s == "ref" else F[f"{s}_{k}"][sl])
mag = lambda s: np.ma.sqrt(fld(s, "ux") ** 2 + fld(s, "uy") ** 2)
tsnap = {MAIN: T0 + int(F[MAIN + "_nsteps"]) * float(F["cfg_dt"]) / D}
TSTAR = int(round(tsnap[MAIN] / 10) * 10)  # Star-CCM+ snapshot nearest the end of the run


def show(a, f, cmap, vlim, extent=ext):
    im = a.imshow(f.T, origin="lower", extent=extent, cmap=cmap, vmin=vlim[0],
                  vmax=vlim[1], interpolation="bilinear", rasterized=True)
    a.set_facecolor(SOLID); a.grid(False)
    return im


# near-bundle velocity of both MPS runs at the same step (fields11.py): the 2048^2 run
# covers only its first qtdur time units, so the comparison is at its end
C = np.load(f"{OUT}/fields_common.npz")
TQ = T0 + r11["t"][-1] - T0                                  # tU/D of that step
TS2 = int(round(TQ / 10) * 10)                               # nearest Star-CCM+ snapshot
sq = star_on_grid(C["q10_X"], C["q10_Y"], TS2)
cmask = lambda k, f: np.ma.masked_where(C[f"{k}_solid"], f)
cfld = lambda k, c: cmask(k, C[f"{k}_{c}"])
cmag = lambda k: np.ma.sqrt(cfld(k, "ux") ** 2 + cfld(k, "uy") ** 2)
cext = [C["q10_X"][0], C["q10_X"][-1], C["q10_Y"][0], C["q10_Y"][-1]]
fig, axs = plt.subplots(3, 3, figsize=(W, 6.3), sharex=True, sharey=True,
                        layout="constrained")
rows = ((lambda k: cmag(k) if k != "ref" else cmask("q10", np.hypot(sq[..., 0], sq[..., 1])),
         "Blues", (0, 2.2), "$|\\mathbf u|/U$"),
        (lambda k: cfld(k, "ux") if k != "ref" else cmask("q10", sq[..., 0]), "RdBu_r", (-2.2, 2.2), "$u/U$"),
        (lambda k: cfld(k, "uy") if k != "ref" else cmask("q10", sq[..., 1]), "RdBu_r", (-0.8, 0.8), "$v/U$"))
heads_ = (("ref", "Star-CCM+"), ("q10", "MPS, half domain"), ("q11", "MPS, full domain"))
for r, (key, cmap, vlim, lab) in enumerate(rows):
    for c, (k, head) in enumerate(heads_):
        im = show(axs[r, c], key(k), cmap, vlim, cext)
        tag(axs[r, c], f"({'abcdefghi'[3 * r + c]}) {head}" if r == 0 else f"({'abcdefghi'[3 * r + c]})")
    fig.colorbar(im, ax=axs[r], shrink=0.9, label=lab)
    axs[r, 0].set_ylabel("$y/D$")
for a in axs[-1]: a.set_xlabel("$x/D$")
fig.savefig("fig/fields.pdf", dpi=300)
with open("fig/numbers.tex", "a") as fh:
    fh.write("\\newcommand{\\tfieldstar}{%d}\n\\newcommand{\\tfieldmps}{%.1f}\n" % (TS2, TQ)); plt.close(fig)

# whole computational domains on common axes, so the wake beyond the MPS outlet
# shows; the MPS domain is y-periodic, so it is rolled to put the bundle at y=0
roll = int(round(n / 2 - cy))
Yf = (np.arange(n) - cy - roll) * h
mu, mv = (np.roll(F[f"{MAIN}_{k}"], roll, axis=1) for k in ("ux", "uy"))
mw = np.gradient(mv, h, axis=0) - np.gradient(mu, h, axis=1)
msolid = np.roll(mask, roll, axis=1) < 0.5
sx, sy = np.arange(-24.7, 74.8, 0.1), np.arange(-39.4, 39.45, 0.1)  # inside the cell centres
sf = star_on_grid(sx, sy, TSTAR)
fig, axs = plt.subplots(2, 2, figsize=(W, 4.8), sharex=True, sharey=True,
                        layout="constrained")
for c, (cmap, vlim, lab, sq, mq) in enumerate((
        ("Blues", (0, 2.2), "$|\\mathbf u|/U$", np.hypot(sf[..., 0], sf[..., 1]), np.hypot(mu, mv)),
        ("RdBu_r", (-0.5, 0.5), "$\\omega_z D/U$", sf[..., 2], mw))):
    show(axs[0, c], sq, cmap, vlim, [sx[0], sx[-1], sy[0], sy[-1]])
    im = show(axs[1, c], np.ma.masked_where(msolid, mq), cmap, vlim, [X[0], X[-1], Yf[0], Yf[-1]])
    axs[0, c].axvline(X[-1], color=INK, ls="--", lw=0.8)
    fig.colorbar(im, ax=axs[:, c], shrink=0.6, label=lab)
    tag(axs[0, c], f"({'ab'[c]}) Star-CCM+, $tU/D={TSTAR}$")
    tag(axs[1, c], f"({'cd'[c]}) {SOLVER}, $tU/D={tsnap[MAIN]:.1f}$")
for a in axs.flat:
    a.set_facecolor("#eeeeea"); a.set_xlim(X[0], sx[-1]); a.set_ylim(sy[0], sy[-1])
for a in axs[:, 0]: a.set_ylabel("$y/D$")
for a in axs[-1]: a.set_xlabel("$x/D$")
fig.savefig("fig/full.pdf", dpi=300); plt.close(fig)
# free stream (|y| > 15D) where both domains exist, and where the Star-CCM+ wake varies in time
fs = lambda xs, ys, q: q[(xs[:, None] > -25) & (xs[:, None] < X[-1]) & (np.abs(ys)[None, :] > 15)]
fsm, fss = fs(X, Yf, mu), fs(sx, sy, sf[..., 0])
wk = np.genfromtxt(f"{OUT}/star_wake.csv", delimiter=",", names=True)
wstd = np.maximum(wk["std_u"], wk["std_v"])
with open("fig/numbers.tex", "a") as fh:
    for key, v in dict(fsmlo="%.2f" % fsm.min(), fsmhi="%.2f" % fsm.max(),
                       fsslo="%.2f" % np.nanmin(fss), fsshi="%.2f" % np.nanmax(fss),
                       wakeonset="%.0f" % wk["x"][np.argmax(wstd >= 0.1)], wakestd="%.1f" % wstd.max(),
                       refstd="%.3f" % max(CD[tail].std(0).max(), CL[tail].std(0).max()),
                       xout="%.0f" % (X[-1] + h / 2), xin="%.0f" % -(X[0] - h / 2)).items():
        fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")

# degradation: the same deviation at a few times over the run (every 20th step
# saved by frames_s.py by frames_s.py; frame 0 is the initial state)
Fr = np.load(f"{OUT}/frames_{MAIN}_s50.npz")
fx = Fr["x0"] + Fr["h"] * np.arange(Fr["ux"].shape[1])
fy = Fr["y0"] + Fr["h"] * np.arange(Fr["ux"].shape[2])
fsolid = Fr["mask"] < 0.5
fstar = star_on_grid(fx, fy)[..., 0]
fdev = Fr["ux"] - fstar
box = ~fsolid & (np.abs(fx)[:, None] < 9) & (np.abs(fy)[None, :] < 7)
frms = np.sqrt((fdev[:, box] ** 2).mean(1))
fig, axs = plt.subplots(2, 3, figsize=(W, 3.4), sharex=True, sharey=True,
                        layout="constrained")
TSNAP = tuple(np.linspace(0, Fr["t"][-1], 6).round())
frame = lambda tk: int(np.argmin(np.abs(Fr["t"] - tk)))
for a, p, tk in zip(axs.flat, "abcdef", TSNAP):
    k = frame(tk)
    im = show(a, np.ma.masked_where(fsolid, fdev[k]), "RdBu_r", (-0.5, 0.5),
              [fx[0], fx[-1], fy[0], fy[-1]])
    a.set_xlim(-9, 9); a.set_ylim(-7, 7)
    tag(a, f"({p}) $tU/D={T0 + Fr['t'][k]:.0f}$, rms ${frms[k]:.2f}U$")
for a in axs[:, 0]: a.set_ylabel("$y/D$")
for a in axs[-1]: a.set_xlabel("$x/D$")
fig.colorbar(im, ax=axs, shrink=0.9, label=r"$(u-u_{\mathrm{ref}})/U$")
fig.savefig("fig/evolution.pdf", dpi=300); plt.close(fig)
kpk = int(np.argmax(frms))
with open("fig/numbers.tex", "a") as fh:
    for key, v in dict(rmsinit="%.2f" % frms[0], rmspeak="%.2f" % frms[kpk],
                       tpeak="%.0f" % (T0 + Fr["t"][kpk]), rmsend="%.2f" % frms[-1],
                       tframes="%.0f" % (T0 + Fr["t"][-1]),
                       rmshalf="%.2f" % frms[frame(25)], rmsqtr="%.2f" % frms[frame(10)]).items():
        fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")

# wake profiles at the same times: one blue ramp, light (early) to dark (late)
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")  # validated --ordinal
fin = np.abs(fy) < 7
fig, ax = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
for k, xd in enumerate((6.0, 8.0)):
    j = int(np.argmin(np.abs(fx - xd)))
    for tk, col in zip(TSNAP[1:], RAMP):
        ax[k].plot(fy[fin], Fr["ux"][frame(tk)][j][fin], color=col, lw=1.1,
                   label=f"$tU/D={T0 + Fr['t'][frame(tk)]:.0f}$")
    ax[k].plot(fy[fin], fstar[j][fin], "--", color=COLOR["ref"], lw=1.1, label="Star-CCM+")
    ax[k].set_xlabel("$y/D$"); tag(ax[k], f"({'ab'[k]}) $x/D={xd:.0f}$")
ax[0].set_ylabel("$u/U$")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, 0.98), handlelength=1.5, columnspacing=1)
fig.tight_layout(); fig.savefig("fig/profiles.pdf"); plt.close(fig)
# velocity differences quoted in the text: last frame, whole region and 2D behind the bundle
j6 = int(np.argmin(np.abs(fx - 6.0)))
u6, s6 = Fr["ux"][-1][j6][fin], fstar[j6][fin]
inner = np.abs(fy[fin]) < 4  # between the outer shear layers
with open("fig/numbers.tex", "a") as fh:
    for key, v in dict(dumax="%.1f" % np.abs(fdev[-1][box]).max(),
                       jetref="%.2f" % s6[inner].max(), jetmps="%.2f" % u6[inner].max(),
                       edgelo="%.2f" % (s6[0] - u6[0]), edgehi="%.2f" % (s6[-1] - u6[-1])).items():
        fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")

# control volumes and rod numbering on the reference velocity-magnitude field
fig, a = plt.subplots(figsize=(3.4, 3.0))
show(a, mag("ref"), "Blues", (0, 2.2))
half = int(round(float(F["cfg_cv_half"]) * 2 ** int(F["cfg_Lf"])))
for (px, py), name in zip(cen, F["names"]):
    xl, xr = np.ceil(px - half), np.floor(px + half)
    yl, yt = np.ceil(py - half), np.floor(py + half)
    a.add_patch(Rectangle(((xl - cx) * h, (yl - cy) * h), (xr - xl) * h, (yt - yl) * h,
                          fill=False, ec=COLOR["cv"], lw=0.9))
    a.text((px - cx) * h, (py - cy) * h, str(name), ha="center", va="center",
           fontsize=6, color="white")
a.set_xlim(-6, 6); a.set_ylim(-5.5, 5.5)
a.set_xlabel("$x/D$"); a.set_ylabel("$y/D$")
a.annotate("", xy=(-4.6, -5.0), xytext=(-5.8, -5.0), arrowprops=dict(arrowstyle="->", color=INK))
a.text(-5.8, -4.75, "$U$", fontsize=8, color=INK)
fig.savefig("fig/cv.pdf", dpi=300); plt.close(fig)

# --- compression: bond dimensions used, and what the cap costs the reference field
import json
rk = json.load(open(f"{OUT}/ranks.json"))
cp = json.load(open(f"{OUT}/compress.json"))
chi = int(F["cfg_max_bd"])
used = np.array(rk[MAIN][-1]["ranks"][0][1:-1])
assert all(r["ranks"][0][1:-1] == list(used) for r in rk[MAIN]), "ranks vary"
k = np.arange(1, len(used) + 1)
fig, ax = plt.subplots(1, 2, figsize=(W, 2.4))
ax[0].semilogy(k, np.minimum(2.0 ** k, 2.0 ** (len(used) + 1 - k)), "o--", ms=3,
               color=MUTED, label="Full rank")
ax[0].semilogy(k, used, "o-", ms=3.5, color=COLOR[MAIN], label=f"MPS solver ($\\chi={chi}$)")
ax[0].set_xlabel("Bond index"); ax[0].set_ylabel("Bond dimension")
ax[0].set_xticks([1, 5, 10, 15, 19]); ax[0].legend(loc="lower center")
for c, (name, lab) in enumerate((("u", "$u$"), ("v", "$v$"), ("p", "$p$"))):
    xs = np.array(sorted(int(x) for x in cp[name] if int(x) <= 160))  # 320 hits float32 input
    ax[1].loglog(xs, [cp[name][str(x)] for x in xs], "o-", ms=3.5,
                 color=("#2a78d6", "#eb6834", "#1baf7a")[c], label=lab)
ax[1].axvline(chi, color=INK, ls=":", lw=1)
ax[1].set_xlabel("Maximum bond dimension $\\chi$")
ax[1].set_ylabel("Relative $L_2$ truncation error"); ax[1].legend(loc="lower left")
for a, s in zip(ax, "ab"): tag(a, f"({s})")
fig.tight_layout(); fig.savefig("fig/compression.pdf"); plt.close(fig)
npar = rk[MAIN][-1]["params"][0]
with open("fig/numbers.tex", "a") as fh:
    for key, v in dict(chimax=chi, nparams=f"\\num{{{npar}}}", ratio=f"{4 ** 10 / npar:.0f}",
                       erru=f"{cp['u'][str(chi)] * 100:.1f}", errv=f"{cp['v'][str(chi)] * 100:.1f}",
                       errp=f"{cp['p'][str(chi)] * 100:.1f}").items():
        fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")

# --- error sources: numbers for table 2 and section 5 --------------------------
WIN = 50          # steps per averaging window
dt = float(F["cfg_dt"])
LONG = runs[MAIN]                        # MPS, RK4
smooth = lambda y: np.convolve(y, np.ones(WIN) / WIN, mode="valid")
bmean = lambda d: d["cd"].mean(1)
rms = lambda a, b: 100 * np.sqrt(np.mean((a / b - 1) ** 2))

# numbers for the error budget (section 5)
pr = list(csv.DictReader(open(f"{OUT}/poisson_rank.csv")))
# table 1: Star-CCM+ (its monitors over the run window) and the MPS solver
tr_bundle = 100 * np.sqrt(np.mean((LONG["cd"].mean(1) / st_cd.mean(1) - 1) ** 2))
tr_rod = 100 * np.sqrt(np.mean((LONG["cd"] / st_cd - 1) ** 2))
with open("fig/table.tex", "w") as fh:
    fh.write("Bundle-mean $C_D$ & %.3f & %.3f \\\\\n" % (rcd.mean(), m_cd.mean()))
    fh.write("Deviation from Star-CCM+ (\\%%) & {---} & %+.1f \\\\\n" % dev)
    fh.write("Per-rod $C_D$ deviation, rms (\\%%) & {---} & %.0f \\\\\n" % rms(m_cd, rcd))
    fh.write("Per-rod correlation, $C_D$ & {---} & %.3f \\\\\n" % corr(m_cd, rcd))
    fh.write("Bundle-mean $C_L$ & %.3f & %.3f \\\\\n" % (rcl.mean(), m_cl.mean()))
    fh.write("Per-rod correlation, $C_L$ & {---} & %.3f \\\\\n" % corr(m_cl, rcl))
    fh.write("\\midrule\nStep by step, against the monitors at the same $tU/D$ & & \\\\\n")
    fh.write("Bundle-mean $C_D$ deviation, rms (\\%%) & {---} & %.1f \\\\\n" % tr_bundle)
    fh.write("Per-rod $C_D$ deviation, rms (\\%%) & {---} & %.0f \\\\\n" % tr_rod)
half = len(LONG["t"]) // 2
hdev = lambda a: "%+.0f" % (100 * (a.mean() / rcd.mean() - 1))
bench = lambda m, k: [100 * float(r["grad_err"]) for r in pr
                      if r["method"] == m and int(r["chi"]) == k]
rng_ = lambda v, f="%.0f": (f % min(v)) + "--" + (f % max(v))
with open("fig/numbers.tex", "a") as fh:
    for key, v in dict(
            nlong=f"\\num{{{len(LONG['t'])}}}",
            mpsstep="64", runhours="%.0f" % (len(LONG["t"]) * 64 / 3600),  # 62-66 s/step, job logs
            cdlo="%.2f" % smooth(bmean(LONG)).min(), cdhi="%.2f" % smooth(bmean(LONG)).max(),
            cflU="%.1f" % (float(F["cfg_u"]) * dt / float(F["cfg_dx"])),
            cdwlo=hdev(smooth(bmean(LONG)).min()), cdwhi=hdev(smooth(bmean(LONG)).max()),
            thalf="%.1f" % LONG["t"][half], cdfirst=hdev(LONG["cd"][:half]), cdsecond=hdev(LONG["cd"][half:]),
            cdrms="%.0f" % rms(m_cd, rcd), nbad="%d" % (np.abs(m_cd / rcd - 1) > 0.2).sum(),
            gforty=rng_(bench("best", 40)), geighty=rng_(bench("best", 80)),
            gonetwenty=rng_(bench("best", 120)), gtwohundred=rng_(bench("best", 200), "%.1f")).items():
        fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")

# --- Star-CCM+ history (appendix) ----------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.2))
for k, (A, lab) in enumerate(((CD, "$C_D$"), (CL, "$C_L$"))):
    ax[k].plot(tref, A, lw=0.5, color=MUTED)
    ax[k].plot(tref, A.mean(1), color=COLOR["ref"], lw=1.3, label="Mean of 21 rods")
    ax[k].axvspan(tref[tail][0], tref[-1], color="#eeeeea", zorder=0)
    ax[k].set_xlabel("$tU/D$"); ax[k].set_ylabel(lab); tag(ax[k], f"({'ab'[k]})")
ax[0].set_ylim(0, 2); ax[1].set_ylim(-0.6, 0.6)
ax[0].legend(loc="upper right")
fig.tight_layout(); fig.savefig("fig/ref_history.pdf"); plt.close(fig)
print("figures written")

# --- appendix: Q=11 run (2048 x 2048, 128D box), early part of the run ----------
if os.path.exists(f"{OUT}/rod_forces_{Q11}.csv") and os.path.exists(f"{OUT}/q11_{Q11}.npz"):
    k11 = [r11["names"].index(nm) for nm in names]       # same rod order as the tables above
    c11d, c11l = r11["cd"][:, k11], r11["cl"][:, k11]
    w10 = r10["t"] <= t11                               # the 1024^2 run over the same physical time
    c10d, c10l = r10["cd"][w10], r10["cl"][w10]
    # Star-CCM+ monitors at the steps of each run, and their means over this window
    s10d, s10l = star_at(CD, r10["t"][w10]), star_at(CL, r10["t"][w10])
    s11d, s11l = star_at(CD, r11["t"]), star_at(CL, r11["t"])
    wd, wl = s11d.mean(0), s11l.mean(0)
    with open("fig/rods11.tex", "w") as fh:
        for i in order:
            fh.write(f"{names[i]} & {wd[i]:.3f} & {c10d.mean(0)[i]:.3f} & {c11d.mean(0)[i]:.3f} & "
                     f"{wl[i]:.3f} & {c10l.mean(0)[i]:.3f} & {c11l.mean(0)[i]:.3f} \\\\\n")
        fh.write("\\midrule\nMean & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n" % (
            wd.mean(), c10d.mean(), c11d.mean(), wl.mean(), c10l.mean(), c11l.mean()))
    trb = lambda c, S: 100 * np.sqrt(np.mean((c.mean(1) / S.mean(1) - 1) ** 2))
    trr = lambda c, S: 100 * np.sqrt(np.mean((c / S - 1) ** 2))
    tcm = lambda c, S: np.mean([corr(a, b) for a, b in zip(c, S)])
    with open("fig/numbers.tex", "a") as fh:
        for key, v in dict(
                qtend="%.1f" % t11, qtdur="%.1f" % (t11 - T0), qnsteps=f"\\num{{{len(r11['t'])}}}",
                qrmsA="%.0f" % rms(c10d.mean(0), wd), qrmsB="%.0f" % rms(c11d.mean(0), wd),
                qcorrA="%.2f" % corr(c10d.mean(0), wd), qcorrB="%.3f" % corr(c11d.mean(0), wd),
                qdevA="%+.1f" % (100 * (c10d.mean() / wd.mean() - 1)),
                qdevB="%+.1f" % (100 * (c11d.mean() / wd.mean() - 1)),
                qclA="%.2f" % corr(c10l.mean(0), wl), qclB="%.2f" % corr(c11l.mean(0), wl),
                qtrbA="%.1f" % trb(c10d, s10d), qtrbB="%.1f" % trb(c11d, s11d),
                qtrrA="%.0f" % trr(c10d, s10d), qtrrB="%.0f" % trr(c11d, s11d),
                qtcA="%.3f" % tcm(c10d, s10d), qtcB="%.3f" % tcm(c11d, s11d),
                ).items():
            fh.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")

    G = np.load(f"{OUT}/q11_{Q11}.npz")
    h11 = float(G["h"]); cen11 = G["centres"]
    n11 = 2 * G["mask"].shape[0]
    X11 = (np.arange(0, n11, 2) - cen11[:, 0].mean()) * h11
    Y11 = (np.arange(0, n11, 2) - cen11[:, 1].mean()) * h11
    s11 = G["mask"] < 0.5
    tstar = int(round(T0 / 10) * 10)
    gx, gy = np.arange(-24.7, 74.8, 0.2), np.arange(-39.4, 39.45, 0.2)
    sfs = star_on_grid(gx, gy, tstar)
    ext11 = [X11[0], X11[-1], Y11[0], Y11[-1]]
    rows11 = [("Star-CCM+", tstar, None)] + [("MPS solver", T0 + float(G["t"][i]), i) for i in (1, 2)]
    fig, axs = plt.subplots(len(rows11), 2, figsize=(W, 7.4), sharex=True, sharey=True,
                            layout="constrained")
    for r, (lab, tt_, i) in enumerate(rows11):
        for c, (cmap, vlim, key, cl) in enumerate((("Blues", (0, 2.2), "speed", "$|\\mathbf u|/U$"),
                                                   ("RdBu_r", (-0.4, 0.4), "vort", "$\\omega_z D/U$"))):
            a = axs[r, c]
            if i is None:
                f = np.hypot(sfs[..., 0], sfs[..., 1]) if c == 0 else sfs[..., 2]
                im = show(a, f, cmap, vlim, [gx[0], gx[-1], gy[0], gy[-1]])
            else:
                f = np.ma.masked_where(s11, G[f"{key}{i}"].astype(float))
                im = show(a, f, cmap, vlim, ext11)
            a.set_facecolor("#eeeeea"); a.set_xlim(-24.7, 74.8); a.set_ylim(-39.4, 39.4)  # the Star-CCM+ domain
            tag(a, f"({'abcdefgh'[2 * r + c]}) {lab}, $tU/D={tt_:.1f}$")
            if r == len(rows11) - 1: a.set_xlabel("$x/D$")
            if c == 0: a.set_ylabel("$y/D$")
            if r == 0: fig.colorbar(im, ax=axs[:, c], shrink=0.7, aspect=40, location="top", label=cl)
    fig.savefig("fig/q11_full.pdf", dpi=250); plt.close(fig)
