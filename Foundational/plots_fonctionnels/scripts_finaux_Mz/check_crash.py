import numpy as np

# 👇 Mets le chemin exact vers ton fichier npz L=49 ici
DATA_PATH = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_finaux_disordered_1D/run_L=49/mz2_full_data_L=49.npz"

print(f"📂 Lecture du fichier : {DATA_PATH}\n")

try:
    data = np.load(DATA_PATH)
    
    sigma_grid = data['sigma_grid']
    h0_grid = data['h0_grid']
    mz2_raw = data['mz2_raw']  # Format: (n_sigmas, n_h0, n_samples)
    
    print(f"Grille de sigmas ({len(sigma_grid)} valeurs) : {sigma_grid}")
    print(f"Grille de h0 ({len(h0_grid)} valeurs)\n")
    print("-" * 40)
    print("📊 ÉTAT DE L'AVANCEMENT :")
    print("-" * 40)
    
    for idx_s, sigma in enumerate(sigma_grid):
        # On regarde le premier sample (index 0) pour vérifier si le h0 a été calculé
        # Si c'est un NaN, c'est que le calcul n'est pas allé jusque là
        valid_h0_mask = ~np.isnan(mz2_raw[idx_s, :, 0])
        nb_h0_calcules = np.sum(valid_h0_mask)
        
        if nb_h0_calcules == len(h0_grid):
            print(f"✅ Sigma = {sigma:5.2f} : 100% terminé ({nb_h0_calcules}/{len(h0_grid)} h0)")
        elif nb_h0_calcules > 0:
            h0_arret = h0_grid[nb_h0_calcules]
            print(f"⚠️ Sigma = {sigma:5.2f} : Interrompu en cours de route !")
            print(f"   -> {nb_h0_calcules}/{len(h0_grid)} h0 calculés.")
            print(f"   -> Le script a planté au niveau de h0 = {h0_arret}")
        else:
            print(f"⏳ Sigma = {sigma:5.2f} : Non commencé (0/{len(h0_grid)} h0)")

except FileNotFoundError:
    print("❌ Fichier introuvable. Vérifie bien le chemin DATA_PATH.")
except Exception as e:
    print(f"❌ Erreur lors de la lecture : {e}")