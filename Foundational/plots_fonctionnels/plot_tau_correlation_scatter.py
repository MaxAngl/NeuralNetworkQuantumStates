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
# 4. FONCTION DE CALCUL TAU CORRELATION
# ==========================================

def create_operator(params):
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L))
    return -ha_X - J_val * ha_ZZ

def compute_tau_mc(params_batch):
    """
    Calcule le temps de corrélation (tau_corr) via Monte Carlo.
    """
    taus = []
    
    # On utilise 16 chaînes pour avoir une statistique robuste sur le sampling
    sa_multi = nk.sampler.MetropolisLocal(hi, n_chains=16)

    for pars in params_batch:
        _vs_single = vs.get_state(pars) 
        
        # MCState nécessaire pour mesurer les corrélations de chaîne
        mc_vs = nk.vqs.MCState(
            sampler=sa_multi, 
            model=_vs_single.model, 
            variables=_vs_single.variables,
            n_samples=4096,         
            n_discard_per_chain=50  
        )
        
        H = create_operator(pars)
        stats = mc_vs.expect(H)
        
        # Extraction du temps de corrélation
        # C'est une mesure de l'efficacité du sampler Metropolis pour cet état spécifique
        taus.append(stats.tau_corr)
        
    return np.array(taus)

# ==========================================
# 5. EXÉCUTION
# ==========================================
print("\n🏗️  Traitement des données TRAIN (Calcul Tau)...")
train_h0 = []
train_taus = []

disorder_file = os.path.join(RUN_DIR, "disorder.configs.npy")
if not os.path.exists(disorder_file):
    disorder_file = os.path.join(RUN_DIR, "disorder_configs.npy")

if os.path.exists(disorder_file):
    train_params = np.load(disorder_file)
    train_taus = compute_tau_mc(tqdm(train_params, desc="Tau Train"))
    
    n_h0_train = len(h0_train_list)
    replicas_per_h0_train = len(train_params) // n_h0_train
    for h0 in h0_train_list:
        train_h0.extend([h0] * replicas_per_h0_train)
else:
    print("⚠️ Fichier disorder.configs.npy introuvable.")

print("\n🧪 Traitement des données TEST (Calcul Tau)...")
test_h0 = []
test_taus = []
rng = np.random.default_rng(seed=42)

for h0 in tqdm(H0_TEST_LIST, desc="Tau Test"):
    test_configs = rng.normal(loc=h0, scale=sigma_disorder, size=(N_TEST_PER_H0, L))
    scores = compute_tau_mc(test_configs)
    test_h0.extend([h0] * N_TEST_PER_H0)
    test_taus.extend(scores)

# ==========================================
# 6. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))

# Nuage de points Test
plt.scatter(
    test_h0, test_taus, 
    color='crimson', alpha=0.3, s=40, label='Test Replicas', 
    edgecolor='none', marker='o', zorder=1
)

# Nuage de points Train
if len(train_h0) > 0:
    plt.scatter(
        train_h0, train_taus, 
        color='royalblue', alpha=0.3, s=40, label='Train Replicas', 
        edgecolor='none', marker='^', zorder=1
    )

# Moyenne Test
df_test = pd.DataFrame({"h0": test_h0, "t": test_taus})
mean_test = df_test.groupby("h0")["t"].mean()
plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

# Moyenne Train
if len(train_h0) > 0:
    df_train = pd.DataFrame({"h0": train_h0, "t": train_taus})
    mean_train = df_train.groupby("h0")["t"].mean()
    plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

# Mise en forme
plt.yscale('linear') # Tau est souvent petit (< 5). Log utile si Tau explose.
plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"Correlation Time $\tau_{corr}$", fontsize=12)
plt.title(rf"Sampling Efficiency ($\tau_{{corr}}$): Train vs Test (L={L})", fontsize=14)

# Lignes de référence
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5) # Limite théorique basse

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='upper left', frameon=True, fontsize=10)

output_file = os.path.join(RUN_DIR, "tau_corr_scatter_train_test.pdf")
plt.tight_layout()
plt.savefig(output_file)
print(f"\n✅ Graphique Tau-Corr sauvegardé : {output_file}")