import os
import sys

# ==========================================
# 0. FORCE CPU (Si nécessaire)
# ==========================================
#os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

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
RUN_DIR = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_finaux_disordered_1D/run_L=49"

H0_TEST_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975,0.980,0.985,0.990,0.993,0.995,0.997,1.0,1.003,1.005,1.010,1.012,1.015,1.025,1.035, 1.05, 1.075, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0] 
SIGMA_TEST_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]  

prob_global_flip = 0.03
N_SAMPLES_MC = 2048 
n_chains=64
n_discard_per_chain=100

H0_TRANS_MIN = 0.5
H0_TRANS_MAX = 1.7

N_KEEP_BEFORE = 20
N_KEEP_TRANS  = 80
N_KEEP_AFTER  = 10

MAX_KEEP = max(N_KEEP_BEFORE, N_KEEP_TRANS, N_KEEP_AFTER)

# ==========================================
# 2. SETUP ET CHARGEMENT
# ==========================================
# Chemin absolu "en dur" vers la racine de ton dépôt où se trouve flip_rules.py
project_root = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational" 
sys.path.insert(0, project_root)
from flip_rules import GlobalFlipRule

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

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(H0_TEST_LIST)) 

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

vars_dict = state_dict.get('variables', state_dict.get('model', {}).get('variables', state_dict.get('vqs', {}).get('variables', state_dict)))

try:
    vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
    print("✅ Poids injectés avec succès !")
except Exception as e:
    print(f"❌ Erreur injection Flax : {e}")
    sys.exit(1)

Mz_operator = sum(nk.operator.spin.sigmaz(hi, i) for i in range(L)) * (1.0 / L)
Mz2_operator = Mz_operator @ Mz_operator


# ==========================================
# 3. GESTION DU DICTIONNAIRE ET REPRISE
# ==========================================
data_path = os.path.join(RUN_DIR, f"mz2_full_data_L={L}.npz")
results = {}

if os.path.exists(data_path):
    print(f"\n🔄 Reprise depuis le checkpoint : {data_path}")
    loaded_data = np.load(data_path)
    
    saved_sigmas = loaded_data['sigma_grid']
    mz2_mean = loaded_data['mz2_mean']
    mz2_min  = loaded_data['mz2_min']
    mz2_max  = loaded_data['mz2_max']
    # Sécurité au cas où l'ancien fichier npz n'aurait pas encore la clé mz2_raw
    mz2_raw  = loaded_data.get('mz2_raw', None) 
    
    for i, s in enumerate(saved_sigmas):
        if not np.isnan(mz2_mean[i]).all():
            results[float(s)] = {
                "mean": mz2_mean[i].tolist(),
                "min":  mz2_min[i].tolist(),
                "max":  mz2_max[i].tolist(),
                "raw":  []
            }
            # Reconstitution de la liste brute sans les NaN de remplissage
            if mz2_raw is not None:
                for j in range(len(H0_TEST_LIST)):
                    batch_complet = mz2_raw[i, j]
                    # On garde uniquement les valeurs qui ne sont pas des NaN
                    valeurs_valides = batch_complet[~np.isnan(batch_complet)].tolist()
                    results[float(s)]["raw"].append(valeurs_valides)

    print(f"✅ Données rechargées pour les sigmas : {list(results.keys())}")


def save_checkpoint():
    """Sauvegarde les données, y compris les valeurs brutes complètes."""
    curves_array = []
    min_array = []
    max_array = []
    raw_array = np.full((len(SIGMA_TEST_LIST), len(H0_TEST_LIST), MAX_KEEP), np.nan)
    
    for idx_s, s in enumerate(SIGMA_TEST_LIST):
        if s in results and len(results[s]["mean"]) == len(H0_TEST_LIST):
            curves_array.append(results[s]["mean"])
            min_array.append(results[s]["min"])
            max_array.append(results[s]["max"])
            
            # Remplissage du tableau 3D pour les valeurs brutes
            for idx_h, batch in enumerate(results[s]["raw"]):
                raw_array[idx_s, idx_h, :len(batch)] = batch
        else:
            nan_list = np.full(len(H0_TEST_LIST), np.nan).tolist()
            curves_array.append(nan_list)
            min_array.append(nan_list)
            max_array.append(nan_list)
            
    np.savez(
        data_path,
        sigma_grid      = np.array(SIGMA_TEST_LIST),
        h0_grid         = np.array(H0_TEST_LIST),
        mz2_mean        = np.array(curves_array),
        mz2_min         = np.array(min_array),
        mz2_max         = np.array(max_array),
        mz2_raw         = raw_array,  # NOUVEAU : Sauvegarde des données brutes
        L               = L,
        N_KEEP_BEFORE   = N_KEEP_BEFORE,
        N_KEEP_TRANS    = N_KEEP_TRANS,
        N_KEEP_AFTER    = N_KEEP_AFTER,
        H0_TRANS_MIN    = H0_TRANS_MIN,
        H0_TRANS_MAX    = H0_TRANS_MAX,
        N_SAMPLES_MC    = N_SAMPLES_MC,
    )

save_checkpoint()


# ==========================================
# 4. FONCTION DE CALCUL (ÉCHANTILLONNAGE)
# ==========================================
print("\n🚀 Lancement des calculs de l'aimantation...")

rng = np.random.default_rng(seed=42)
sa_multi = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=n_chains)

dummy_params = np.zeros(L)
_vs_init = vs.get_state(dummy_params)
mc_vs = nk.vqs.MCState(
    sampler=sa_multi,
    model=_vs_init.model,
    variables=_vs_init.variables,
    n_samples=N_SAMPLES_MC, 
    n_discard_per_chain=n_discard_per_chain
)

for sigma in SIGMA_TEST_LIST:
    # On vérifie si ce sigma a déjà été entièrement calculé
    if sigma in results and len(results[sigma]["mean"]) == len(H0_TEST_LIST):
        print(f"\n⏭️ Sigma = {sigma} déjà calculé. Passage au suivant.")
        continue

    print(f"\n▶️ Traitement pour sigma = {sigma}")
    results[sigma] = {"mean": [], "min": [], "max": [], "raw": []}
    
    base_noise = rng.normal(loc=0.0, scale=sigma, size=(MAX_KEEP, L))
    
    for h0 in tqdm(H0_TEST_LIST, desc=f"Balayage h0"):
        if h0 <= H0_TRANS_MIN:
            n_keep = N_KEEP_BEFORE
        elif h0 >= H0_TRANS_MAX:
            n_keep = N_KEEP_AFTER
        else:
            n_keep = N_KEEP_TRANS
            
        current_base_noise = base_noise[:n_keep]
        configs = np.abs(h0 + current_base_noise)
        
        mz2_batch = []
        for pars in configs:
            mc_vs.variables = vs.get_state(pars).variables
            mc_vs.reset()
            
            stats = mc_vs.expect(Mz2_operator)
            mz2_val = float(stats.Mean.real)
            mz2_batch.append(mz2_val)
            
        # SAUVEGARDE DES DONNÉES BRUTES + STATISTIQUES
        results[sigma]["raw"].append(mz2_batch)
        results[sigma]["mean"].append(np.mean(mz2_batch))
        results[sigma]["min"].append(np.min(mz2_batch))
        results[sigma]["max"].append(np.max(mz2_batch))

    # 💾 SAUVEGARDE CHECKPOINT APRÈS CHAQUE SIGMA
    save_checkpoint()
    print(f"💾 Checkpoint enregistré pour sigma = {sigma}.")

# ==========================================
# 5. FIN DU SCRIPT
# ==========================================
print(f"\n✅ Génération des données terminée.")
print(f"📁 Fichier disponible pour l'analyse : {data_path}")