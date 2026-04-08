import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_PATH = "/users/eleves-a/2024/nikola.audit/NeuralNetworkQuantumStates/Foundational/rami_perso/2D_FNQS"
TAILLES_L = range(4, 11)
TARGET_SIGMA = 0.0
MIN_SAMPLES = 10  # Nombre minimum d'échantillons valides pour calculer le cumulant

plt.figure(figsize=(10, 7))
colors = plt.cm.jet(np.linspace(0, 1, len(TAILLES_L)))

# ==========================================
# 2. BOUCLE SUR LES TAILLES DE SYSTÈME
# ==========================================
for i, L in enumerate(TAILLES_L):
    folder_name = f"Run_2D_L{L}_FNQS"
    data_name = f"is_data_2D_L{L}_full.npz"
    data_path = os.path.join(BASE_PATH, folder_name, data_name)

    if not os.path.exists(data_path):
        print(f"⚠️  Fichier manquant pour L={L} : {data_path}")
        continue

    print(f"🔄 Traitement de L={L}...")
    data = np.load(data_path)

    sigma_grid = data["sigma_grid"]
    h0_grid   = data["h0_grid"]
    mz2_raw   = data["mz2_raw"]  # Shape attendue : (n_sigma, n_h0, n_samples)

    # ------------------------------------------
    # Diagnostic rapide
    # ------------------------------------------
    print(f"   Shape mz2_raw : {mz2_raw.shape}")

    # Trouver l'indice sigma = TARGET_SIGMA
    idx_s = np.where(np.isclose(sigma_grid, TARGET_SIGMA))[0]
    if len(idx_s) == 0:
        print(f"❌ Sigma={TARGET_SIGMA} non trouvé dans L={L}")
        continue
    idx_s = idx_s[0]

    # m2_samples : shape (n_h0, n_samples)
    m2_samples = mz2_raw[idx_s]

    # ------------------------------------------
    # Diagnostic NaN
    # ------------------------------------------
    n_valid = np.sum(~np.isnan(m2_samples), axis=1)  # Échantillons valides par h0
    nan_ratio = np.isnan(m2_samples).mean(axis=1)
    print(f"   Échantillons valides par h0 — min: {n_valid.min()}, max: {n_valid.max()}")
    print(f"   Ratio NaN par h0             — min: {nan_ratio.min():.2f}, max: {nan_ratio.max():.2f}")

    if n_valid.max() < MIN_SAMPLES:
        print(f"⚠️  Pas assez de données valides pour L={L}, taille ignorée.")
        continue

    # ------------------------------------------
    # Calcul des moments avec nanmean
    # NOTE : si mz2_raw contient m (et non m²), remplacer par :
    #        m2_mean = np.nanmean(m2_samples**2, axis=1)
    #        m4_mean = np.nanmean(m2_samples**4, axis=1)
    # ------------------------------------------
    m2_mean = np.nanmean(m2_samples,    axis=1)   # <m²>
    m4_mean = np.nanmean(m2_samples**2, axis=1)   # <m⁴>

    # ------------------------------------------
    # Cumulant de Binder : U4 = 1 - <m⁴> / (3 <m²>²)
    # ------------------------------------------
    denom  = 3.0 * m2_mean**2
    mask   = (n_valid >= MIN_SAMPLES) & (denom > 0)
    binder = np.where(mask, 1.0 - m4_mean / denom, np.nan)

    # ------------------------------------------
    # Plot (on ne trace que les points valides)
    # ------------------------------------------
    plt.plot(h0_grid, binder, label=f"L = {L}", color=colors[i], linewidth=1.5)

    # Signal si des points ont été masqués
    n_masked = np.sum(~mask)
    if n_masked > 0:
        print(f"   ⚠️  {n_masked} points masqués (NaN ou denom ≤ 0) pour L={L}")

# ==========================================
# 3. MISE EN FORME DU GRAPHIQUE
# ==========================================
plt.axhline(y=2/3, color='black', linestyle='--', alpha=0.5, label="Limite ordonnée (2/3)")
plt.axhline(y=0,   color='black', linestyle=':',  alpha=0.5, label="Limite gaussienne (0)")

plt.title(f"Binder Cumulant $U_4$ — $\\sigma = {TARGET_SIGMA}$ (2D Ising)")
plt.xlabel(r"Transverse Field $h_0$")
plt.ylabel(r"$U_4 = 1 - \frac{\langle m^4 \rangle}{3 \langle m^2 \rangle^2}$")
plt.grid(True, ls="--", alpha=0.4)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_plot = os.path.join(BASE_PATH, "binder_cumulant_multi_L.pdf")
plt.savefig(output_plot, bbox_inches='tight')
print(f"\n✅ Graphique sauvegardé sous : {output_plot}")
plt.show()