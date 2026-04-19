"""
Plot de la loss function (moyenne des énergies sur toutes les configurations de train)
en fonction des itérations, avec un zoom sur les 100 dernières itérations.

Usage:
    python plot_loss_disordered_1D_L100.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
LOG_PATH = "../../logs/Trains_finaux_disordered_1D/run_L=100/log_data.json.log"
META_PATH = "../../logs/Trains_finaux_disordered_1D/run_L=100/meta.json"
OUT_DIR   = "../../logs/Trains_finaux_disordered_1D/run_L=100"

with open(LOG_PATH) as f:
    data = json.load(f)

with open(META_PATH) as f:
    meta = json.load(f)

iters  = np.array(data["Energy"]["iters"])
mean_E = np.array(data["Energy"]["Mean"]["real"])

# ==========================================
# FIGURE PRINCIPALE
# ==========================================
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(iters, mean_E, color="steelblue", linewidth=1.0)

ax.set_xlabel("Itération", fontsize=13)
ax.set_ylabel(r"$\langle E \rangle$ (énergie moyenne)", fontsize=13)
ax.set_title(f"Loss function — 1D désordonné, $L=100$\n"
             f"({meta['total_configs_train']} configurations d'entraînement)", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.4)

# ==========================================
# ZOOM SUR LES 100 DERNIÈRES ITÉRATIONS
# ==========================================
n_zoom = 100
mask = iters >= iters[-1] - (n_zoom - 1)

# Inset positionné en haut à droite dans l'espace libre
axins = ax.inset_axes([0.52, 0.55, 0.44, 0.38])  # [x0, y0, largeur, hauteur] en coordonnées axes

axins.plot(iters[mask], mean_E[mask], color="steelblue", linewidth=1.0)

x0, x1 = iters[mask][0], iters[mask][-1]
y_min, y_max = mean_E[mask].min(), mean_E[mask].max()
margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.5
axins.set_xlim(x0 - 0.5, x1 + 0.5)
axins.set_ylim(y_min - margin, y_max + margin)

axins.set_xlabel("Itération", fontsize=8)
axins.set_ylabel(r"$\langle E \rangle$", fontsize=8)
axins.tick_params(labelsize=7)
axins.grid(True, linestyle="--", alpha=0.4)
axins.set_title(f"{n_zoom} dernières itérations", fontsize=8)

# Connecteurs entre le plot principal et l'inset
ax.indicate_inset_zoom(axins, edgecolor="gray", linewidth=0.8)

# ==========================================
# SAUVEGARDE
# ==========================================
fig.tight_layout()

for ext, dpi in [("pdf", 100), ("png", 100)]:
    out = f"{OUT_DIR}/loss_vs_iters_L100.{ext}"
    plt.savefig(out, dpi=dpi)
    print(f"Sauvegardé : {out}")

plt.close(fig)
