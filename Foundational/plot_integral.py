import os
import json
import netket as nk
import netket_foundational as nkf
import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from pathlib import Path
from netket_foundational._src.model.vit import ViTFNQS

# --- 1. CONFIGURATION ET CHARGEMENT ---
# Utilisation du chemin que tu as fourni
run_dir = Path("/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/run_2026-02-04_14-10-58")
checkpoint_path = run_dir / "state_90.nk" # Ajuste si besoin au state_200.nk
meta_path = run_dir / "meta.json"
log_path = run_dir / "log_data.json"

with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
h0_train_list = np.array(meta["hamiltonian"]["h0_train_list"])
n_reps = meta["n_replicas_per_h0"]
n_h0 = len(h0_train_list)
J_val = meta["hamiltonian"]["J"]
sigma_disorder = meta["hamiltonian"]["sigma"]

# Chargement robuste du log
log_nk = nk.logging.JsonLog(str(log_path).replace(".json", ""))
print(f"✅ Log et Meta chargés pour L={L}.")

# Détection de la clé d'énergie (évite le KeyError)
energy_key = None
for k in ["ham", "Energy", "energy"]:
    if k in log_nk.data:
        energy_key = k
        break

if energy_key is None:
    raise KeyError(f"Impossible de trouver la clé d'énergie. Clés disponibles : {list(log_nk.data.keys())}")
print(f"🔑 Clé d'énergie détectée : '{energy_key}'")

# --- 2. RECONSTRUCTION ET CALCUL EXACT ---
hi = nk.hilbert.Spin(0.5, L)

def create_operator(params):
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

# Régénération des paramètres avec la seed originale
rng = np.random.default_rng(meta["seed"])
params_train = []
for h in h0_train_list:
    params_train.append(rng.normal(h, sigma_disorder, (n_reps, L)))
params_train = np.vstack(params_train)

print("Calcul des énergies exactes (Lanczos)...")
exact_energies = []
for p in tqdm(params_train):
    res = nk.exact.lanczos_ed(create_operator(p), k=1)
    exact_energies.append(res[0].item())
exact_energies = np.array(exact_energies)

# --- 3. EXTRACTION DES DONNÉES ---
convergence_errors = [] 
final_metrics = {
    "h0": [], "rel_err": [], "variance": [], 
    "r_hat": [], "v_score": [], "tau": []
}

for i in range(len(params_train)):
    h0_val = h0_train_list[i // n_reps]
    
    # Accès sécurisé aux données du log
    replica_data = log_nk.data[energy_key][i]
    iters = replica_data.iters
    means = np.array(replica_data.Mean)
    errs = np.abs(means - exact_energies[i]) / np.abs(exact_energies[i])
    
    convergence_errors.append(errs)
    
    # Métriques de la dernière itération
    final_metrics["h0"].append(h0_val)
    final_metrics["rel_err"].append(errs[-1])
    final_metrics["variance"].append(replica_data.Variance[-1])
    final_metrics["v_score"].append(replica_data.Variance[-1] / (means[-1].real**2 + 1e-12))
    final_metrics["r_hat"].append(getattr(replica_data, 'R_hat', [1.0])[-1])
    final_metrics["tau"].append(getattr(replica_data, 'TauCorr', [0.0])[-1])

# --- 4. PLOTS CONVERGENCE PAR FENÊTRE (H0) ---
n_cols = 3
n_rows = (n_h0 + n_cols - 1) // n_cols
fig1, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True)
axes = axes.flatten()
cmap_tab = cm.get_cmap("tab10")

for i in range(n_h0):
    ax = axes[i]
    # Même couleur pour toutes les courbes d'une même fenêtre
    color = cmap_tab(i % 10)
    for r in range(n_reps):
        idx = i * n_reps + r
        ax.plot(iters, convergence_errors[idx], color=color, alpha=0.4)
    ax.set_yscale('log')
    ax.set_title(f"$h_0 = {h0_train_list[i]}$")
    ax.grid(True, which="both", alpha=0.2)
    if i % n_cols == 0: ax.set_ylabel("Rel. Error Energy")

for j in range(i + 1, len(axes)): fig1.delaxes(axes[j])
fig1.tight_layout()
fig1.savefig(run_dir / f"Found_L={L}_convergence_grid.pdf")

# --- 5. SUPERPOSITION MAGMA ---

plt.figure(figsize=(10, 6))
norm = plt.Normalize(h0_train_list.min(), h0_train_list.max())
sm = cm.ScalarMappable(cmap="magma", norm=norm)

for i in range(len(convergence_errors)):
    h_val = h0_train_list[i // n_reps]
    plt.plot(iters, convergence_errors[i], color=cm.magma(norm(h_val)), alpha=0.3)

plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Relative Error")
plt.title(f"Superposition Gradient Magma (L={L})")
plt.colorbar(sm, label="$h_0$")
plt.savefig(run_dir / f"Found_L={L}_convergence_magma.pdf")

# --- 6. SCATTER PLOTS DES MÉTRIQUES ---

metrics_info = [
    ("rel_err", "Relative Error Energy", True),
    ("variance", "Variance $\sigma^2$", True),
    ("r_hat", "$\hat{R}$ (Gelman-Rubin)", False),
    ("v_score", "V-score", True),
    ("tau", "$\tau$ (Autocorr Time)", False)
]

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
axes3 = axes3.flatten()

for i, (key, label, is_log) in enumerate(metrics_info):
    ax = axes3[i]
    ax.scatter(final_metrics["h0"], final_metrics[key], c=final_metrics["h0"], cmap="magma", edgecolors='k', alpha=0.7)
    if is_log: ax.set_yscale('log')
    ax.set_xlabel("$h_0$")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(True, alpha=0.3)

fig3.delaxes(axes3[-1])
plt.tight_layout()
plt.savefig(run_dir / f"Found_L={L}_metrics_scatter.pdf")

print(f"✅ Analyse terminée. Fichiers PDF générés dans {run_dir}")