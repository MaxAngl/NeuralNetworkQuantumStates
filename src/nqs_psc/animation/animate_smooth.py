import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from pathlib import Path

# --- Configuration et Constantes Globales ---
# 1) Renseigner votre dossier
PATH = Path(r"C:\Users\mouts\OneDrive\Bureau\X\2A\PSC\NQS\logs\run_2025-12-04_10-41-44")

# Optionnel: Activer la sauvegarde
SAVE_ANIMATION = False
SAVE_PATH = PATH / "spin_animation_final.mp4"
FPS = 15

# CONSTANTES POUR L'ESTHÉTIQUE
L_AMP_GLOBAL = 2.5  # Longueur maximale du spin (facteur d'agrandissement)
LINEWIDTH_GLOBAL = 5.0  # Épaisseur des flèches

# ---------------------------------------------------------
# 2) Chargement log + meta (Code inchangé)
# ---------------------------------------------------------
meta_path = PATH / "meta.json"
log_path = PATH / "log.json"

try:
    if meta_path.exists() and log_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        with open(log_path) as f:
            log = json.load(f)
    else:
        runtime_path = PATH / "runtime_log.json"
        with open(runtime_path) as f:
            data = json.load(f)
        meta = data["meta"]
        log = data["log"]
except FileNotFoundError:
    print(f"Erreur: Fichiers log ou meta introuvables dans {PATH}")
    exit()

# ---------------------------------------------------------
# 3) Détection des spins (Code inchangé)
# ---------------------------------------------------------
spin_keys = sorted(
    [k for k in log if k.startswith("op_")], key=lambda s: int(s.split("_")[1])
)
N = len(spin_keys)


# ---------------------------------------------------------
# 4) Grille (1D, 2D, 3D) (Code inchangé)
# ---------------------------------------------------------
n_dim = meta.get("n_dim", 3)
L_list = meta.get("L", [])
L = (
    L_list[0]
    if isinstance(L_list, list) and L_list
    else meta.get("L", int(N ** (1 / n_dim) if N > 0 else 1))
)

if n_dim == 1:
    dims = (L,)
elif n_dim == 2:
    dims = (L, L)
elif n_dim == 3:
    dims = (L, L, L)
else:
    raise ValueError("Dimension non supportée")

assert np.prod(dims) == N, "Mismatch entre log et grille"


# ---------------------------------------------------------
# 5) Extraction des valeurs σᶻ (Code inchangé)
# ---------------------------------------------------------
n_iter = len(log[spin_keys[0]])
spin_data = np.zeros((n_iter, N))

for idx, key in enumerate(spin_keys):
    spin_data[:, idx] = [v[1] for v in log[key]]

spin_data = spin_data.reshape((n_iter,) + dims)


# ---------------------------------------------------------
# 6) Construction grille 3D (coordonnées) (Code inchangé)
# ---------------------------------------------------------
coords = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")

if n_dim == 1:
    x = coords[0]
    y = np.zeros_like(x)
    z = np.zeros_like(x)
elif n_dim == 2:
    x, y = coords
    z = np.zeros_like(x)
else:
    x, y, z = coords

x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()


# ---------------------------------------------------------
# 7) Fonctions utilitaires pour le Quiver
# ---------------------------------------------------------
def get_spin_colors(spin):
    """Détermine la couleur des flèches et des sphères."""
    color = "darkred" if spin < 0 else "darkblue"
    scatter_color = "red" if spin < 0 else "blue"
    return color, scatter_color


def get_quiver_properties(spin):
    """Calcule la longueur effective du vecteur spin en utilisant L_AMP_GLOBAL."""
    L_base = 0.1
    L_eff = L_base + L_AMP_GLOBAL * abs(spin)
    w = L_eff * np.sign(spin)
    return w


# ---------------------------------------------------------
# 8) ANIMATION (Flèches très grandes)
# ---------------------------------------------------------
fig = plt.figure(figsize=(12, 10), facecolor="white")
ax = fig.add_subplot(111, projection="3d", facecolor="lightgray")

# --- Paramètres Graphiques ---
dim_x = dims[0] if n_dim >= 1 else 1
dim_y = dims[1] if n_dim >= 2 else 1
dim_z = dims[2] if n_dim == 3 else 1

# CORRECTION CLÉ: Calcul de la marge en utilisant la constante globale L_AMP_GLOBAL
margin = 1.5 * max(1, L_AMP_GLOBAL)

# Ajustement des limites pour tenir compte des flèches plus longues
ax.set_xlim(-margin / 2, dim_x - 1 + margin / 2)
ax.set_ylim(-margin / 2, dim_y - 1 + margin / 2 if n_dim >= 2 else 0.5)
ax.set_zlim(-margin / 2, dim_z - 1 + margin / 2 if n_dim == 3 else 0.5)

ax.set_box_aspect((dim_x, dim_y, dim_z))

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

ax.set_title(
    f"Évolution des spins $\\sigma^z$ ({dim_x}x{dim_y}x{dim_z} grille)",
    fontsize=16,
    color="black",
)

quivers = []
scatter_plot = None


def create_initial_quivers_and_scatter():
    """Crée les flèches et les points initiaux (frame 0)."""
    global scatter_plot
    it = product(*(np.arange(d) for d in dims))
    initial_spin_values = spin_data[0].flatten()

    # 1. Création des Boules (Points Scatter)
    scatter_colors = [get_spin_colors(s)[1] for s in initial_spin_values]

    scatter_plot = ax.scatter(
        x_flat,
        y_flat,
        z_flat,
        c=scatter_colors,
        marker="o",
        s=150,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    # 2. Création des Flèches (Quivers)
    for idx in it:
        spin = spin_data[0][idx]
        w = get_quiver_properties(spin)
        color, _ = get_spin_colors(spin)

        qv = ax.quiver(
            x[idx],
            y[idx],
            z[idx],
            0,
            0,
            w,
            color=color,
            linewidth=LINEWIDTH_GLOBAL,  # Utilisation de la constante globale
            length=1.0,
            pivot="middle",
            arrow_length_ratio=0.3,
            antialiased=True,
            normalize=False,
        )
        quivers.append(qv)


def update(frame):
    """Met à jour les propriétés (longueur/couleur) des flèches et des sphères."""
    global scatter_plot
    it = product(*(np.arange(d) for d in dims))
    current_spin_values = spin_data[frame].flatten()

    # --- Mise à jour des Boules ---
    current_scatter_colors = [get_spin_colors(s)[1] for s in current_spin_values]
    scatter_plot.set_color(current_scatter_colors)

    # --- Mise à jour des Flèches ---
    new_quivers = []
    for qv, idx in zip(quivers, it):
        spin = spin_data[frame][idx]
        w = get_quiver_properties(spin)
        color, _ = get_spin_colors(spin)

        # Supprimer/Recréer l'objet Quiver
        qv.remove()

        new_qv = ax.quiver(
            x[idx],
            y[idx],
            z[idx],
            0,
            0,
            w,
            color=color,
            linewidth=LINEWIDTH_GLOBAL,  # Utilisation de la constante globale
            length=1.0,
            pivot="middle",
            arrow_length_ratio=0.3,
            antialiased=True,
            normalize=False,
        )
        new_quivers.append(new_qv)

    quivers[:] = new_quivers

    ax.set_title(
        f"Évolution des spins $\\sigma^z$ – Itération {frame}",
        fontsize=16,
        color="black",
    )
    return [scatter_plot] + quivers


# --- Lancement ---
create_initial_quivers_and_scatter()

# Intervalle de temps entre les frames (en ms)
interval_ms = 1000 / FPS
anim = FuncAnimation(fig, update, frames=n_iter, interval=interval_ms, blit=False)

if SAVE_ANIMATION:
    print(f"Sauvegarde de l'animation vers : {SAVE_PATH}...")

    if SAVE_PATH.suffix.lower() == ".gif":
        writer = PillowWriter(fps=FPS)
    elif SAVE_PATH.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=FPS)
    else:
        print("Format non supporté. Utilisation de .mp4 par défaut.")
        writer = FFMpegWriter(fps=FPS)

    anim.save(SAVE_PATH, writer=writer, dpi=200)
    print("Sauvegarde terminée.")
else:
    plt.show()
