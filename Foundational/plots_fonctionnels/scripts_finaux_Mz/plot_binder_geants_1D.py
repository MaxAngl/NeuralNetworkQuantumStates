import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

TRAINS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../logs/Trains_1D_geants",
)

H0_MIN, H0_MAX = 0.7, 1.3

FONTSIZE_LABEL  = 16
FONTSIZE_TICK   = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TITLE  = 17
LW = 1.8

# Collect all runs
run_dirs = {}
for name in sorted(os.listdir(TRAINS_DIR)):
    if not name.startswith("run_L"):
        continue
    path = os.path.join(TRAINS_DIR, name)
    if not os.path.isdir(path):
        continue
    L = int(name.split("_")[1][1:])
    run_dirs[L] = path

sizes = sorted(run_dirs.keys())

# Color palette: one color per L
cmap = cm.get_cmap("viridis", len(sizes))
colors = {L: cmap(i) for i, L in enumerate(sizes)}

fig, ax = plt.subplots(figsize=(9, 6))

for L in sizes:
    fpath = os.path.join(run_dirs[L], "binder_data.npz")
    if not os.path.exists(fpath):
        print(f"L={L}: binder_data.npz manquant, passage.")
        continue

    d = np.load(fpath)
    h0 = d["h0"]
    m2 = d["m2"]
    m4 = d["m4"]

    mask = (h0 >= H0_MIN) & (h0 <= H0_MAX)
    if mask.sum() < 2:
        print(f"L={L}: moins de 2 points dans [{H0_MIN}, {H0_MAX}], passage.")
        continue

    binder = 1.0 - m4[mask] / (3.0 * m2[mask] ** 2)

    ax.plot(
        h0[mask], binder,
        color=colors[L], lw=LW,
        label=rf"$L={L}$",
    )

ax.axhline(2 / 3, color="gray", lw=1.2, ls=":", alpha=0.7, label=r"$U=2/3$ (FM)")
ax.axhline(0.0,   color="gray", lw=1.2, ls="--", alpha=0.7, label=r"$U=0$ (PM)")

ax.set_xlabel(r"Champ transverse $h_0$", fontsize=FONTSIZE_LABEL)
ax.set_ylabel(r"Cumulant de Binder $U_4$", fontsize=FONTSIZE_LABEL)
ax.set_title(
    r"Cumulant de Binder — Ising 1D, $\sigma=0$ (CLP)",
    fontsize=FONTSIZE_TITLE,
)
ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
ax.set_xlim(H0_MIN, H0_MAX)
ax.legend(
    fontsize=FONTSIZE_LEGEND,
    loc="upper right",
    framealpha=0.9,
    title="Taille $L$",
    title_fontsize=FONTSIZE_LEGEND,
    ncol=2,
)
ax.grid(True, alpha=0.3, linestyle=":")
fig.tight_layout()

out_dir = TRAINS_DIR
for ext in ("pdf", "png"):
    out = os.path.join(out_dir, f"binder_sigma0_allL.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Sauvegardé dans {out_dir}/binder_sigma0_allL.pdf")
