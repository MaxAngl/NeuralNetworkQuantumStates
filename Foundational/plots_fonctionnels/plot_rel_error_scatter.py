import os
import sys
import glob
import json
import numpy as np
import jax
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import msgpack
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import zipfile 

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Mettez ici le chemin de votre run
RUN_DIR = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/run_2026-02-11_13-31-21"

H0_TEST_LIST = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.5, 3.5, 4.5] 
N_TEST_PER_H0 = 20 

# ==========================================
# 2. SETUP
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if not os.path.exists(RUN_DIR):
    print(f"❌ Erreur : Le dossier {RUN_DIR} n'existe pas.")
    sys.exit(1)

meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
J_val = meta["hamiltonian"]["J"]
sigma_disorder = meta["hamiltonian"]["sigma"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
vit_params = meta["vit_config"]

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(H0_TEST_LIST + h0_train_list))

ma = ViTFNQS(
    num_layers=vit_params["num_layers"],
    d_model=vit_params["d_model"],
    heads=vit_params["heads"],
    b=vit_params["b"],
    L_eff=vit_params["L_eff"], 
    n_coups=ps.size, 
    complex=True, 
    disorder=True, 
    transl_invariant=False, 
    two_dimensional=False, 
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=1) 
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
if not checkpoints:
    print("❌ Aucun checkpoint (.nk) trouvé.")
    sys.exit(1)
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

# ==========================================
# 3. CHARGEMENT ZIP (ROBUSTE)
# ==========================================
print(f"🔹 Chargement depuis l'archive : {os.path.basename(last_checkpoint)}")

state_dict = None

if not zipfile.is_zipfile(last_checkpoint):
    with open(last_checkpoint, 'rb') as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(last_checkpoint, 'r') as zf:
        file_list = zf.namelist()
        candidates = [f for f in file_list if f.endswith('.msgpack')]
        if not candidates:
             print("❌ Erreur structure ZIP.")
             sys.exit(1)
        target_file = sorted(candidates, key=len)[-1]
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

vars_dict = None
if 'variables' in state_dict: vars_dict = state_dict['variables']
elif 'model' in state_dict and 'variables' in state_dict['model']: vars_dict = state_dict['model']['variables']
elif 'params' in state_dict: vars_dict = state_dict
elif 'vqs' in state_dict and 'variables' in state_dict['vqs']: vars_dict = state_dict['vqs']['variables']
if vars_dict is None: vars_dict = state_dict

try:
    vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
    print("✅ Poids injectés avec succès !")
except Exception as e:
    print(f"❌ Erreur lors de l'injection Flax : {e}")
    sys.exit(1)

# ==========================================
# 4. FONCTIONS DE CALCUL (VMC vs EXACT)
# ==========================================

def get_hamiltonian_op(params):
    """Construit l'opérateur Hamiltonien NetKet pour une config donnée."""
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L))
    return -ha_X - J_val * ha_ZZ

def compute_rel_error(params_batch):
    """
    Pour chaque config :
    1. Calcule E_exact (Lanczos)
    2. Calcule E_vmc (FullSumState pour précision max sur l'ansatz)
    3. Retourne l'erreur relative
    """
    rel_errors = []
    
    # Pour L=4, on peut utiliser FullSumState qui est exact pour l'ansatz (pas de bruit MC)
    # Si L > 10, il faudrait passer à MCState.
    
    for pars in params_batch:
        # --- A. Energie Exacte (Ground Truth) ---
        H = get_hamiltonian_op(pars)
        # Lanczos pour trouver la valeur propre la plus basse
        E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
        
        # --- B. Energie Variationnelle (Ansatz) ---
        _vs = vs.get_state(pars) 
        # Utilisation de FullSumState pour avoir l'espérance exacte de <psi|H|psi>
        # Cela mesure la qualité de l'approximation du réseau, sans le bruit de sampling.
        vs_fs = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, variables=_vs.variables)
        E_vmc = vs_fs.expect(H).Mean.real
        
        # --- C. Erreur Relative ---
        # error = |E_vmc - E_exact| / |E_exact|
        err = abs(E_vmc - E_exact) / abs(E_exact)
        rel_errors.append(err)
        
    return np.array(rel_errors)

# ==========================================
# 5. EXECUTION DES CALCULS
# ==========================================
print("\n🏗️  Traitement des données TRAIN (Calcul Erreur Relative)...")
train_h0 = []
train_errors = []

disorder_file = os.path.join(RUN_DIR, "disorder.configs.npy")
if not os.path.exists(disorder_file):
    disorder_file = os.path.join(RUN_DIR, "disorder_configs.npy")

if os.path.exists(disorder_file):
    train_params = np.load(disorder_file)
    train_errors = compute_rel_error(tqdm(train_params, desc="RelErr Train"))
    
    n_h0_train = len(h0_train_list)
    replicas_per_h0_train = len(train_params) // n_h0_train
    for h0 in h0_train_list:
        train_h0.extend([h0] * replicas_per_h0_train)
else:
    print("⚠️ Fichier disorder.configs.npy introuvable.")

print("\n🧪 Traitement des données TEST (Calcul Erreur Relative)...")
test_h0 = []
test_errors = []
rng = np.random.default_rng(seed=42)

for h0 in tqdm(H0_TEST_LIST, desc="RelErr Test"):
    test_configs = rng.normal(loc=h0, scale=sigma_disorder, size=(N_TEST_PER_H0, L))
    scores = compute_rel_error(test_configs)
    test_h0.extend([h0] * N_TEST_PER_H0)
    test_errors.extend(scores)

# ==========================================
# 6. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))

# Nuage de points Test
plt.scatter(
    test_h0, test_errors, 
    color='crimson', alpha=0.3, s=40, label='Test Replicas', 
    edgecolor='none', marker='o', zorder=1
)

# Nuage de points Train
if len(train_h0) > 0:
    plt.scatter(
        train_h0, train_errors, 
        color='royalblue', alpha=0.3, s=40, label='Train Replicas', 
        edgecolor='none', marker='^', zorder=1
    )

# Moyenne Test
df_test = pd.DataFrame({"h0": test_h0, "err": test_errors})
mean_test = df_test.groupby("h0")["err"].mean()
plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

# Moyenne Train
if len(train_h0) > 0:
    df_train = pd.DataFrame({"h0": train_h0, "err": train_errors})
    mean_train = df_train.groupby("h0")["err"].mean()
    plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

# Mise en forme
plt.yscale('log') # Echelle LOG impérative pour l'erreur relative
plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"Relative Energy Error $|E_{VMC} - E_{exact}| / |E_{exact}|$", fontsize=12)
plt.title(f"Energy Accuracy: Train vs Test (L={L})", fontsize=14)

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)

output_file = os.path.join(RUN_DIR, "rel_error_scatter_train_test.pdf")
plt.tight_layout()
plt.savefig(output_file)
print(f"\n✅ Graphique Erreur Relative sauvegardé : {output_file}")