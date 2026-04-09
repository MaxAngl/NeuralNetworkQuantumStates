"""
Binder cumulant collapsé pour le 2D TFIM sans désordre (sigma=0).

X-axis : (h0 - hc) * L^(1/nu)   avec hc=3.044, nu=0.63
Y-axis : B = 1 - <Mz^4> / (3 * <Mz^2>^2)

Toutes les tailles L sur le même graphe.

Usage:
    python plot_binder_collapsed_2D.py [--output-dir DIR]
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default=None)
args = parser.parse_args()

BASE_DIR   = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_autour_transi_2D"
L_LIST     = [3, 4, 5, 6, 8, 10]
output_dir = args.output_dir or BASE_DIR

HC_INF = 3.044
NU     = 0.63   # exposant critique 3D Ising


def binder_cumulant(samples):
    """samples: (N_samples, nb_spins), valeurs ±1"""
    mz  = samples.mean(axis=1)
    mz2 = np.mean(mz ** 2)
    mz4 = np.mean(mz ** 4)
    return 1.0 - mz4 / (3.0 * mz2 ** 2)


def load_chunks(L):
    """Charge et fusionne tous les chunks pour L.
    Retourne liste triée de (h0_value, samples_array)."""
    d = os.path.join(BASE_DIR, f"L={L}")
    chunk_files = sorted(
        glob.glob(os.path.join(d, f"is_samples_2D_L{L}_chunk*.npz")),
        key=lambda fn: int(fn.split("chunk")[-1].replace(".npz", ""))
    )
    points = {}
    for fn in chunk_files:
        f = np.load(fn)
        h_ref      = f["h_ref"]       # (n_h0_chunk, nb_spins)
        samples    = f["samples"]     # (n_h0_chunk, N_samples, nb_spins)
        h0_indices = f["h0_indices"]  # (n_h0_chunk,)
        for j in range(len(h0_indices)):
            idx = int(h0_indices[j])
            if idx not in points:
                h0_val = float(h_ref[j, 0])
                points[idx] = (h0_val, samples[j])
    return [(points[idx][0], points[idx][1]) for idx in sorted(points.keys())]


# ==========================================
# CHARGEMENT ET CALCUL
# ==========================================
results = {}

for L in L_LIST:
    try:
        pts = load_chunks(L)
    except Exception as e:
        print(f"L={L}: erreur — {e}")
        continue

    h0_arr = np.array([p[0] for p in pts])
    B_arr  = np.array([binder_cumulant(p[1]) for p in pts])

    order  = np.argsort(h0_arr)
    h0_arr = h0_arr[order]
    B_arr  = B_arr[order]

    # Variable réduite pour le collapse
    x_arr = (h0_arr - HC_INF) * L ** (1.0 / NU)

    results[L] = {"h0": h0_arr, "x": x_arr, "B": B_arr}
    print(f"L={L}: {len(h0_arr)} points, "
          f"x ∈ [{x_arr.min():.2f}, {x_arr.max():.2f}], "
          f"B ∈ [{B_arr.min():.3f}, {B_arr.max():.3f}]")

if not results:
    print("Aucune donnée trouvée.")
    sys.exit(1)

# ==========================================
# PLOT COLLAPSÉ
# ==========================================
colors    = plt.cm.plasma(np.linspace(0.1, 0.9, len(L_LIST)))
color_map = {L: colors[i] for i, L in enumerate(L_LIST)}

fig, ax = plt.subplots(figsize=(8, 5.5))

for L, d in results.items():
    ax.plot(d["x"], d["B"], marker='o', markersize=4, linewidth=1.2,
            color=color_map[L], label=f"$L={L}$")

ax.axvline(x=0.0, color='gray', linestyle=':', alpha=0.6,
           label=rf"$h_c^{{\infty}}={HC_INF}$")
ax.axhline(y=2/3, color='k', linestyle='--', linewidth=0.8, alpha=0.5,
           label=r"$B=2/3$ (ordonné)")
ax.axhline(y=0.0, color='k', linestyle='-.', linewidth=0.8, alpha=0.5,
           label=r"$B=0$ (désordonné)")

ax.set_xlabel(
    r"$(h_0 - h_c^\infty)\, L^{1/\nu}$" + rf"    [$h_c={HC_INF},\ \nu={NU}$]",
    fontsize=13
)
ax.set_ylabel(
    r"$B = 1 - \langle M_z^4\rangle\, /\, (3\langle M_z^2\rangle^2)$",
    fontsize=12
)
ax.set_title(
    r"Binder cumulant collapsé — 2D TFIM sans désordre ($\sigma=0$)",
    fontsize=13
)
ax.legend(fontsize=10)
ax.grid(True, ls="--", alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(output_dir, f"binder_cumulant_collapsed_2D.{ext}")
    plt.savefig(out, dpi=150)
    print(f"Sauvegardé : {out}")

plt.close()
