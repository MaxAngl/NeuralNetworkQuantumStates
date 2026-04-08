import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_PATH = "/users/eleves-a/2024/nikola.audit/NeuralNetworkQuantumStates/Foundational/rami_perso/2D_FNQS"
TAILLES_L = range(4,11)  # De 1 à 10
TARGET_SIGMA = 0.0

plt.figure(figsize=(10, 7))
colors = plt.cm.jet(np.linspace(0, 1, len(TAILLES_L)))

# ==========================================
# 2. BOUCLE SUR LES TAILLES DE SYSTÈME
# ==========================================
for i, L in enumerate(TAILLES_L):
    # Construction du chemin dynamique
    folder_name = f"Run_2D_L{L}_FNQS"
    data_name=f"is_data_2D_L{L}_full.npz"
    data_path = os.path.join(BASE_PATH, folder_name, data_name) # Ajuste 'data.npz' si besoin
    
    if not os.path.exists(data_path):
        print(f"⚠️ Fichier manquant pour L={L} : {data_path}")
        continue

    print(f"🔄 Traitement de L={L}...")
    data = np.load(data_path)
    
    
    
    
    sigma_grid = data["sigma_grid"]
    h0_grid = data["h0_grid"]
    mz2_raw = data["mz2_raw"] # Shape: (n_sigma, n_h0, n_samples)
    data = np.load(data_path)
    mz2_raw = data["mz2_raw"]

    print(f"Shape de mz2_raw : {mz2_raw.shape}")          # Doit être (n_sigma, n_h0, n_samples)
    print(f"n_samples = {mz2_raw.shape[-1]}")              # Si = 1 → problème garanti

    idx_s = 0  # exemple
    m2_samples = mz2_raw[idx_s]
    print(f"Variance inter-échantillons (h0=0) : {np.var(m2_samples[0]):.6f}")
    # Si ≈ 0 → les échantillons sont identiques ou inexistants
    # Trouver l'indice correspondant à sigma = 0
    idx_s = np.where(np.isclose(sigma_grid, TARGET_SIGMA))[0]
    if len(idx_s) == 0:
        print(f"❌ Sigma={TARGET_SIGMA} non trouvé dans L={L}")
        continue
    idx_s = idx_s[0]

    # Extraction des données pour sigma = 0
    # mz2_raw[idx_s] a une forme (n_h0, n_samples)
    m2_samples = mz2_raw[idx_s]
    
    # Calcul des moments
    m2_mean = np.nanmean(m2_samples**5, axis=1)
    m4_mean = np.nanmean(m2_samples**7, axis=1)
    
    # Calcul du Cumulant de Binder
    # U4 = 1 - <m^4> / (3 * <m^2>^2)
    binder = 1 - (m4_mean / (3 * (m2_mean**2)))

    # --- PLOT ---
    plt.plot(h0_grid, binder, label=f"L = {L}", color=colors[i], linewidth=1.5)

# ==========================================
# 3. MISE EN FORME DU GRAPHIQUE
# ==========================================
plt.axhline(y=2/3, color='black', linestyle='--', alpha=0.5, label="Limite ordonnée (2/3)")
plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label="Limite gaussienne (0)")

plt.title(f"Binder Cumulant $U_4$ pour $\sigma = {TARGET_SIGMA}$ (2D System)")
plt.xlabel(r"Transverse Field $h_0$")
plt.ylabel(r"$U_4 = 1 - \frac{\langle m^4 \rangle}{3 \langle m^2 \rangle^2}$")
plt.grid(True, ls="--", alpha=0.4)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Sauvegarde
output_plot = os.path.join(BASE_PATH, "binder_cumulant_multi_L.pdf")
plt.savefig(output_plot)
print(f"\n✅ Graphique sauvegardé sous : {output_plot}")
plt.show()