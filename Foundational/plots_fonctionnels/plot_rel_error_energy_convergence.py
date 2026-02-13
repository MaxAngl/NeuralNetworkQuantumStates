import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import netket as nk
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION (MODIFIEZ ICI)
# ==========================================

# 👇 COLLEZ LE CHEMIN DE VOTRE RUN ENTRE LES GUILLEMETS CI-DESSOUS 👇
CHEMIN_DU_RUN = "/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates-1/logs/test_L=16/run_2026-02-11_15-47-34"

# ==========================================
# 2. LOGIQUE DE DÉTECTION DU DOSSIER
# ==========================================

# On regarde d'abord si vous avez rempli la variable ci-dessus
run_dir = None

if CHEMIN_DU_RUN and os.path.exists(CHEMIN_DU_RUN):
    run_dir = CHEMIN_DU_RUN
    print(f"✅ Utilisation du chemin manuel : {run_dir}")
else:
    # Sinon, on regarde les arguments ligne de commande
    parser = argparse.ArgumentParser(description="Plot relative energy error")
    parser.add_argument("--run", "-r", dest="run_dir", help="Path to run directory", default=None)
    args = parser.parse_args()
    
    if args.run_dir:
        run_dir = args.run_dir

# Vérification finale
if run_dir is None or not os.path.exists(run_dir):
    print("❌ Erreur : Aucun dossier de run valide trouvé.")
    print(f"   Chemin testé : {CHEMIN_DU_RUN}")
    sys.exit(1)

# ==========================================
# 3. CHARGEMENT DES METADONNÉES
# ==========================================
meta_path = os.path.join(run_dir, "meta.json")
if not os.path.exists(meta_path):
    print(f"❌ meta.json introuvable dans {run_dir}")
    sys.exit(1)

with open(meta_path, 'r') as f:
    meta = json.load(f)

# Extraction des hyperparamètres
L = meta["L"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
J_val = meta["hamiltonian"]["J"]
n_replicas = meta["n_replicas_per_h0"]

print(f"🔹 Configuration : L={L}, J={J_val:.4f}, Replicas/h0={n_replicas}")
print(f"🔹 Liste h0 train : {h0_train_list}")

# ==========================================
# 4. CHARGEMENT DES LOGS VMC
# ==========================================
candidates = ["log_data.json.log", "log_data.log", "log_data.json"]
log_path = None
for name in candidates:
    p = os.path.join(run_dir, name)
    if os.path.exists(p):
        log_path = p
        break

if log_path is None:
    print("❌ Aucun fichier de log trouvé (log_data.json.log, .log ou .json)")
    sys.exit(1)

print(f"🔹 Chargement des logs depuis : {os.path.basename(log_path)}")
with open(log_path, 'r') as f:
    log_json = json.load(f)

if "ham" in log_json:
    ham_data = log_json["ham"]
elif "Energy" in log_json:
    ham_data = log_json["Energy"]
else:
    print("❌ Pas de données 'ham' ou 'Energy' dans le log.")
    sys.exit(1)

# ==========================================
# 5. CALCUL DES ÉNERGIES EXACTES (Via .npy)
# ==========================================
exact_energies = {}

# A. Déjà dans le log ?
if "exact_energies" in log_json and len(log_json["exact_energies"]) > 0:
    print("✅ Énergies exactes trouvées dans le log.")
    exact_energies = log_json["exact_energies"]
else:
    print("⚠️ Énergies exactes absentes du log. Recalcul via configurations sauvegardées...")
    
    # B. Chargement du fichier désordre
    disorder_file = os.path.join(run_dir, "disorder.configs.npy") 
    if not os.path.exists(disorder_file):
        disorder_file = os.path.join(run_dir, "disorder_configs.npy")

    if os.path.exists(disorder_file):
        print(f"📂 Chargement du désordre depuis : {os.path.basename(disorder_file)}")
        params_list = np.load(disorder_file)
    else:
        print(f"❌ CRITIQUE : Fichier de désordre introuvable dans {run_dir}")
        print("   Impossible de calculer l'erreur relative exacte.")
        sys.exit(1)

    # --- Initialisation NetKet ---
    hi = nk.hilbert.Spin(0.5, L)

    def get_hamiltonian(pars):
        # Construction Hamiltonien : - sum h_i X_i - J sum Z_i Z_{i+1}
        hx = sum(pars[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
        hzz = sum(nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L))
        return -hx - J_val * hzz

    print(f"   Calcul exact (Lanczos) pour {len(params_list)} réplicas...")
    
    for i, pars in tqdm(enumerate(params_list), total=len(params_list)):
        H = get_hamiltonian(pars)
        E0 = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False).item()
        exact_energies[str(i)] = E0

# ==========================================
# 6. TRAÇAGE (PLOTTING)
# ==========================================
print("📈 Génération du graphique...")

h0_to_replicas = {}
for idx, h0_val in enumerate(h0_train_list):
    h0_to_replicas[h0_val] = list(range(idx * n_replicas, (idx + 1) * n_replicas))

n_h0 = len(h0_train_list)
cols = 3
rows = (n_h0 + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
axes = axes.flatten()

# Palette de couleurs
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, n_h0)]

for h0_idx, h0_val in enumerate(h0_train_list):
    ax = axes[h0_idx]
    replica_indices = h0_to_replicas[h0_val]
    current_color = colors[h0_idx]
    
    valid_plot = False
    for replica_num, replica_idx in enumerate(replica_indices):
        if replica_idx < len(ham_data):
            
            # Données VMC
            energy_data = ham_data[replica_idx]
            iters = np.array(energy_data.get("iters", []))
            
            # Lecture robuste de Mean
            means_raw = energy_data.get("Mean", {})
            if isinstance(means_raw, dict):
                if "real" in means_raw: means = np.array(means_raw["real"])
                elif "value" in means_raw: means = np.real(np.array(means_raw["value"]))
                else: means = np.array([]) 
            else:
                means = np.real(np.array(means_raw))
            
            # Données Exactes
            e_exact = exact_energies.get(str(replica_idx))
            
            if e_exact is not None and len(iters) > 0 and len(means) > 0:
                e_exact = float(e_exact)
                # Erreur relative
                rel_error = np.abs((means - e_exact) / e_exact)
                
                ax.plot(iters, rel_error, 
                       label=f"Rep {replica_num}", 
                       color=current_color, 
                       linewidth=1.5,
                       alpha=0.6)
                valid_plot = True

    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Rel Error $|E_{VMC} - E_{ex}|/|E_{ex}|$", fontsize=10)
    ax.set_title(f"$h_0 = {h0_val}$", fontsize=12, fontweight='bold', color=current_color)
    
    ax.set_yscale('log')
    ax.set_xscale('linear') 
    ax.grid(True, which="both", ls="--", alpha=0.3)

# Masquer les graphiques vides
for idx in range(n_h0, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
output_filename = f"relative_error_energy_convergence_L={L}.pdf"
output_path = os.path.join(run_dir, output_filename)
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"✅ Graphique sauvegardé : {output_path}")
plt.close()