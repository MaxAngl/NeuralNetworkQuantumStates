"""
Binder cumulant B = 1 - <Mz^4> / (3 * <Mz^2>^2) pour sigma=0 (sans désordre).

Chargement depuis les fichiers is_samples_1D_L{L}_merged.npz (L=16,24,36,48)
et is_samples_1D_L{L}_chunk*.npz (L=64,80, merge à la volée).

Pour sigma=0, h_ref = h0 → pas de repondération IS nécessaire.
On utilise la valeur h0 réelle depuis h_ref (champ uniforme) pour éviter
les problèmes d'indices entre différentes grilles.

Usage:
    python plot_binder.py [--output-dir DIR]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default=None)
args = parser.parse_args()

BASE_DIR = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_autour_transi_1D"
L_LIST = [16, 24, 36, 48, 64, 80]
output_dir = args.output_dir or BASE_DIR


def binder_cumulant(samples):
    """
    samples: (N_samples, L), valeurs ±1
    Retourne B = 1 - <Mz^4> / (3 * <Mz^2>^2)
    """
    mz = samples.mean(axis=1)          # (N_samples,) — magnétisation par spin
    mz2 = np.mean(mz ** 2)
    mz4 = np.mean(mz ** 4)
    return 1.0 - mz4 / (3.0 * mz2 ** 2)


def load_merged(L):
    """Charge depuis le fichier mergé (L <= 48).
    Retourne liste de (h0_value, samples_array)."""
    path = os.path.join(BASE_DIR, f"L={L}", f"is_samples_1D_L{L}_merged.npz")
    f = np.load(path)
    # h_ref shape: (20, L) — champ uniforme h0 par site
    h0_values = f["h_ref"][:, 0]       # valeur h0 réelle (uniforme → prendre site 0)
    samples = f["samples"]             # (20, N_samples, L)
    return [(float(h0_values[j]), samples[j]) for j in range(len(h0_values))]


def load_chunks(L):
    """Charge depuis les chunks (L >= 64).
    Retourne liste de (h0_value, samples_array)."""
    d = os.path.join(BASE_DIR, f"L={L}")
    chunk_files = sorted(
        [fn for fn in os.listdir(d) if fn.startswith(f"is_samples_1D_L{L}_chunk")],
        key=lambda fn: int(fn.replace(f"is_samples_1D_L{L}_chunk", "").replace(".npz", ""))
    )
    points = []
    for fn in chunk_files:
        f = np.load(os.path.join(d, fn))
        # h_ref shape: (n_h0_chunk, L)
        h0_values = f["h_ref"][:, 0]
        for j in range(len(h0_values)):
            points.append((float(h0_values[j]), f["samples"][j]))
    # Dédupliquer par h0 (garder le premier si doublons)
    seen = {}
    for h0, samp in points:
        h0_r = round(h0, 8)
        if h0_r not in seen:
            seen[h0_r] = samp
    # Trier par h0
    return [(h0, seen[h0]) for h0 in sorted(seen.keys())]


# ==========================================
# CHARGEMENT ET CALCUL
# ==========================================
results = {}

for L in L_LIST:
    try:
        if L <= 48:
            points = load_merged(L)
        else:
            points = load_chunks(L)
    except FileNotFoundError as e:
        print(f"L={L}: fichier manquant — {e}")
        continue

    h0_arr = np.array([p[0] for p in points])
    B_arr = np.array([binder_cumulant(p[1]) for p in points])

    # Trier par h0
    order = np.argsort(h0_arr)
    h0_arr, B_arr = h0_arr[order], B_arr[order]

    results[L] = {"h0": h0_arr, "B": B_arr}
    print(f"L={L}: {len(h0_arr)} points, h0 ∈ [{h0_arr.min():.3f}, {h0_arr.max():.3f}], "
          f"B ∈ [{B_arr.min():.3f}, {B_arr.max():.3f}]")

if not results:
    print("Aucune donnée trouvée.")
    sys.exit(1)

# ==========================================
# PLOT
# ==========================================
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(L_LIST)))
color_map = {L: colors[i] for i, L in enumerate(L_LIST)}

fig, ax = plt.subplots(figsize=(8, 5.5))

for L, d in results.items():
    ax.plot(d["h0"], d["B"], marker='o', markersize=4, linewidth=1.2,
            color=color_map[L], label=f"$L={L}$")

ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.6, label=r"$h_c^{\infty}=1.0$")
ax.axhline(y=2/3, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label=r"$B=2/3$ (ordonné)")
ax.axhline(y=0.0, color='k', linestyle='-.', linewidth=0.8, alpha=0.5, label=r"$B=0$ (désordonné)")

ax.set_xlabel(r"Champ transverse $h_0$", fontsize=13)
ax.set_ylabel(r"$B = 1 - \langle M_z^4\rangle / (3\langle M_z^2\rangle^2)$", fontsize=12)
ax.set_title(r"Binder cumulant — 1D TFIM sans désordre ($\sigma=0$)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, ls="--", alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()

out_pdf = os.path.join(output_dir, "binder_multiL_sigma0.pdf")
out_png = os.path.join(output_dir, "binder_multiL_sigma0.png")
plt.savefig(out_pdf)
plt.savefig(out_png, dpi=150)
print(f"Sauvegardé: {out_pdf}")
plt.show()
