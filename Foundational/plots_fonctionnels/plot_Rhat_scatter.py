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

# RUN_DIR = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/run_2026-02-04_17-13-25"
RUN_DIR = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/run_2026-02-17_13-42-07"

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

# Sampler de base (utilisé juste pour l'init)
sa = nk.sampler.MetropolisLocal(hi, n_chains=1) 
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
if not checkpoints:
    print("❌ Aucun checkpoint (.nk) trouvé.")
    sys.exit(1)
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

# ==========================================
# 3. CHARGEMENT
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
# 4. FONCTION DE CALCUL R-HAT
# ==========================================
def create_operator(params):
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L))
    return -ha_X - J_val * ha_ZZ

def compute_rhat(params_batch):
    r_hats = []
    
    # Pour R-hat, il FAUT plusieurs chaînes. On crée un sampler dédié ici.
    # n_chains=16 est suffisant pour une bonne estimation.
    sa_rhat = nk.sampler.MetropolisLocal(hi, n_chains=16) 

    # Désactivation des warnings de compilation JAX dans la boucle
    for pars in params_batch:
        _vs = vs.get_state(pars) 
        
        # On utilise MCState (Monte Carlo) pour avoir des statistiques de sampling
        mc_vs = nk.vqs.MCState(
            sampler=sa_rhat, 
            model=_vs.model, 
            variables=_vs.variables,
            n_samples=4096,        # Nombre de samples total
            n_discard_per_chain=16 # Burn-in
        )
        
        H = create_operator(pars)
        
        # Le calcul de l'espérance via MC calcule automatiquement R_hat
        # sur les chaînes de Markov
        stats = mc_vs.expect(H)
        r_hats.append(stats.R_hat)
        
    return np.array(r_hats)

# ==========================================
# 5. EXECUTION DES CALCULS
# ==========================================
print("\n🏗️  Traitement des données TRAIN...")
train_h0 = []
train_rhats = [] # Changé de v_score à rhats

disorder_file = os.path.join(RUN_DIR, "disorder.configs.npy")
if not os.path.exists(disorder_file):
    disorder_file = os.path.join(RUN_DIR, "disorder_configs.npy")

if os.path.exists(disorder_file):
    train_params = np.load(disorder_file)
    # Calcul R-hat sur Train
    train_rhats = compute_rhat(tqdm(train_params, desc="R-hat Train"))
    
    n_h0_train = len(h0_train_list)
    replicas_per_h0_train = len(train_params) // n_h0_train
    for h0 in h0_train_list:
        train_h0.extend([h0] * replicas_per_h0_train)
else:
    print("⚠️ Fichier disorder.configs.npy introuvable.")

print("\n🧪 Traitement des données TEST...")
test_h0 = []
test_rhats = [] # Changé de v_score à rhats
rng = np.random.default_rng(seed=42)

for h0 in tqdm(H0_TEST_LIST, desc="R-hat Test"):
    test_configs = rng.normal(loc=h0, scale=sigma_disorder, size=(N_TEST_PER_H0, L))
    scores = compute_rhat(test_configs)
    test_h0.extend([h0] * N_TEST_PER_H0)
    test_rhats.extend(scores)

# ==========================================
# 6. PLOTTING (R-HAT)
# ==========================================
plt.figure(figsize=(10, 6))

# --- SCATTER PLOTS ---
plt.scatter(
    test_h0, 
    test_rhats, 
    color='crimson', 
    alpha=0.3, 
    s=40, 
    label='Test Replicas', 
    edgecolor='none', 
    marker='o', 
    zorder=1
)

if len(train_h0) > 0:
    plt.scatter(
        train_h0, 
        train_rhats, 
        color='royalblue', 
        alpha=0.3, 
        s=40, 
        label='Train Replicas', 
        edgecolor='none', 
        marker='^', 
        zorder=1
    )

# --- MEAN CURVES ---
# Moyenne TEST
df_test = pd.DataFrame({"h0": test_h0, "r": test_rhats})
mean_test = df_test.groupby("h0")["r"].mean()

plt.plot(
    mean_test.index, 
    mean_test.values, 
    color='red', 
    linestyle='--', 
    linewidth=1.5, 
    label='Test Mean', 
    zorder=2
)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

# Moyenne TRAIN
if len(train_h0) > 0:
    df_train = pd.DataFrame({"h0": train_h0, "r": train_rhats})
    mean_train = df_train.groupby("h0")["r"].mean()

    plt.plot(
        mean_train.index, 
        mean_train.values, 
        color='blue', 
        linestyle='--', 
        linewidth=1.5, 
        label='Train Mean', 
        zorder=2
    )
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

# --- MISE EN FORME ---
# R-hat est proche de 1. L'échelle linéaire est souvent préférable pour voir les écarts à 1.
# Si vous voulez voir les très petites variations, vous pouvez décommenter la ligne log
plt.yscale('linear') 
# plt.yscale('log') # Décommenter si vous voulez une échelle log

plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"Gelman-Rubin $\hat{R}$", fontsize=12)
plt.title(f"Convergence Diagnostics ($\hat{{R}}$): Train vs Test (L={L})", fontsize=14)

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)

output_file = os.path.join(RUN_DIR, f"rhat_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(output_file)
print(f"\n✅ Scatter plot R-hat sauvegardé : {output_file}")