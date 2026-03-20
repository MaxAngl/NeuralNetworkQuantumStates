import os
import sys
import glob
import json
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import msgpack
from tqdm import tqdm
import zipfile

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 👇 MODIFIEZ LE CHEMIN ICI 👇
RUN_DIR = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/tests_echantillonnage_L=25/run_2026-03-04_19-43-01"

# Paramètres de test
H0_TEST_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] 
SIGMA_TEST_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7]  # Liste des désordres à tester
N_TEST_PER_H0 = 20 

prob_global_flip = 0.01
N_SAMPLES_MC = 2048 # Nombre de samples pour l'estimation de Mz^2

# ==========================================
# 2. SETUP ET CHARGEMENT
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from flip_rules import GlobalFlipRule

# Enveloppe sécurisée pour le type JAX
class SafeGlobalFlipRule(GlobalFlipRule):
    def transition(self, sampler, machine, parameters, state, key, sigma):
        sigma_new, log_prob = super().transition(sampler, machine, parameters, state, key, sigma)
        return jnp.asarray(sigma_new, dtype=sigma.dtype), log_prob

if not os.path.exists(RUN_DIR):
    print(f"❌ Erreur : Le dossier {RUN_DIR} n'existe pas.")
    sys.exit(1)

meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
vit_params = meta["vit_config"]

print(f"🔹 Configuration : L={L}")
print(f"🔹 Tests sur les sigmas : {SIGMA_TEST_LIST}")

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(H0_TEST_LIST)) # Approximation pour init

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

sa = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=1)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
if not checkpoints:
    print("❌ Aucun checkpoint (.nk) trouvé.")
    sys.exit(1)
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
print(f"🔹 Chargement des poids depuis : {os.path.basename(last_checkpoint)}")

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
    print(f"❌ Erreur injection Flax : {e}")
    sys.exit(1)


# ==========================================
# 3. CRÉATION DE L'OPÉRATEUR Mz^2
# ==========================================
# Mz = (1/L) * sum(Z_i)
Mz_operator = sum(nk.operator.spin.sigmaz(hi, i) for i in range(L)) * (1.0 / L)
# Mz^2 = Mz @ Mz
Mz2_operator = Mz_operator @ Mz_operator


# ==========================================
# 4. FONCTION DE CALCUL
# ==========================================
print("\n🚀 Lancement des calculs de l'aimantation...")

rng = np.random.default_rng(seed=42)

# Initialisation du sampler et du MCState (une seule fois pour la rapidité)
sa_multi = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=16)

dummy_params = np.zeros(L)
_vs_init = vs.get_state(dummy_params)
mc_vs = nk.vqs.MCState(
    sampler=sa_multi,
    model=_vs_init.model,
    variables=_vs_init.variables,
    n_samples=N_SAMPLES_MC, 
    n_discard_per_chain=200 
)

# Tableaux pour stocker toutes les données brutes
mz2_raw_array = np.zeros((len(SIGMA_TEST_LIST), len(H0_TEST_LIST), N_TEST_PER_H0))

for idx_s, sigma in enumerate(SIGMA_TEST_LIST):
    print(f"\n▶ Traitement pour sigma = {sigma}")
    
    for idx_h, h0 in enumerate(tqdm(H0_TEST_LIST, desc=f"Balayage h0")):
        # Génération des N réplicas avec la valeur absolue (comme pour l'entraînement)
        configs = np.abs(rng.normal(loc=h0, scale=sigma, size=(N_TEST_PER_H0, L)))
        
        mz2_batch = []
        for pars in configs:
            # Injection de la configuration de désordre dans le réseau
            mc_vs.variables = vs.get_state(pars).variables
            mc_vs.reset()
            
            # Calcul de <Mz^2>
            stats = mc_vs.expect(Mz2_operator)
            mz2_val = float(stats.Mean.real)
            mz2_batch.append(mz2_val)
            
        # Enregistrement direct des données brutes
        mz2_raw_array[idx_s, idx_h, :] = mz2_batch

# ==========================================
# 5. SAUVEGARDE DES DONNÉES
# ==========================================
print("\n💾 Sauvegarde des données brutes...")

data_path = os.path.join(RUN_DIR, f"mz2_full_data_L={L}.npz")

np.savez(
    data_path,
    sigma_grid = np.array(SIGMA_TEST_LIST),
    h0_grid    = np.array(H0_TEST_LIST),
    mz2_raw    = mz2_raw_array,
    L          = L
)

print(f"✅ Fichier de données généré avec succès :")
print(f"   --> {data_path}")
print("   (Prêt à être analysé par le script de plotting !)")