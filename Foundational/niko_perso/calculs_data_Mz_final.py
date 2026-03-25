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

H0_TEST_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975,0.980,0.985,0.990,0.993,0.995,0.997, 1.0,1.003,1.005,1.010,1.012,1.015,1.025,1.035, 1.05, 1.075, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0] 
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

hi = nk.hilbert.Spin(0.5, L**2)
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
    two_dimensional=True, 
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


N_sites = hi.size # Total de sites (ex: 49)
Mz_operator = sum(nk.operator.spin.sigmaz(hi, i) for i in range(N_sites)) * (1.0 / N_sites)
Mz2_operator = Mz_operator @ Mz_operator


# ==========================================
# 3. GESTION DE LA REPRISE (CHECKPOINT INTELLIGENT)
# ==========================================
data_path = os.path.join(RUN_DIR, f"mz2_full_data_L={L}.npz")

if os.path.exists(data_path):
    print(f"\n🔄 Reprise depuis le checkpoint : {data_path}")
    loaded_data = np.load(data_path)
    
    mz2_mean_array = loaded_data['mz2_mean']
    mz2_min_array  = loaded_data['mz2_min']
    mz2_max_array  = loaded_data['mz2_max']
    mz2_raw_array  = loaded_data['mz2_raw']
else:
    print("\n🆕 Création d'un nouveau fichier de données.")
    mz2_mean_array = np.full((len(SIGMA_TEST_LIST), len(H0_TEST_LIST)), np.nan)
    mz2_min_array  = np.full((len(SIGMA_TEST_LIST), len(H0_TEST_LIST)), np.nan)
    mz2_max_array  = np.full((len(SIGMA_TEST_LIST), len(H0_TEST_LIST)), np.nan)
    mz2_raw_array  = np.full((len(SIGMA_TEST_LIST), len(H0_TEST_LIST), MAX_KEEP), np.nan)

def save_checkpoint():
    """Écrase le fichier avec les tableaux actuels."""
    np.savez(
        data_path,
        sigma_grid      = np.array(SIGMA_TEST_LIST),
        h0_grid         = np.array(H0_TEST_LIST),
        mz2_mean        = mz2_mean_array,
        mz2_min         = mz2_min_array,
        mz2_max         = mz2_max_array,
        mz2_raw         = mz2_raw_array,
        L               = L,
        N_KEEP_BEFORE   = N_KEEP_BEFORE,
        N_KEEP_TRANS    = N_KEEP_TRANS,
        N_KEEP_AFTER    = N_KEEP_AFTER,
        H0_TRANS_MIN    = H0_TRANS_MIN,
        H0_TRANS_MAX    = H0_TRANS_MAX,
        N_SAMPLES_MC    = N_SAMPLES_MC,
    )

# ==========================================
# 4. FONCTION DE CALCUL (ÉCHANTILLONNAGE)
# ==========================================
print("\n🚀 Lancement des calculs de l'aimantation...")

sa_multi = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=n_chains)

dummy_params = np.zeros(hi.size)
_vs_init = vs.get_state(dummy_params)
mc_vs = nk.vqs.MCState(
    sampler=sa_multi,
    model=_vs_init.model,
    variables=_vs_init.variables,
    n_samples=N_SAMPLES_MC, 
    n_discard_per_chain=n_discard_per_chain
)

for idx_s, sigma in enumerate(SIGMA_TEST_LIST):
    # Si la moyenne n'a aucun NaN pour ce sigma, c'est qu'il est 100% fini.
    if not np.isnan(mz2_mean_array[idx_s]).any():
        print(f"\n⏭️ Sigma = {sigma} déjà calculé. Passage au suivant.")
        continue

    print(f"\n▶️ Traitement pour sigma = {sigma}")
    
    # Génération du bruit liée à l'index du sigma pour une reproductibilité parfaite
    rng = np.random.default_rng(seed=42 + idx_s)
    base_noise = rng.normal(loc=0.0, scale=sigma, size=(MAX_KEEP, hi.size))
    
    for idx_h, h0 in enumerate(tqdm(H0_TEST_LIST, desc=f"Balayage h0")):
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
            
        # Remplissage direct dans les tableaux NumPy
        mz2_raw_array[idx_s, idx_h, :n_keep] = mz2_batch
        mz2_mean_array[idx_s, idx_h] = np.mean(mz2_batch)
        mz2_min_array[idx_s, idx_h]  = np.min(mz2_batch)
        mz2_max_array[idx_s, idx_h]  = np.max(mz2_batch)

    # 💾 SAUVEGARDE CHECKPOINT APRÈS CHAQUE SIGMA
    save_checkpoint()
    print(f"💾 Checkpoint enregistré pour sigma = {sigma}.")

# ==========================================
# 5. FIN DU SCRIPT
# ==========================================
print(f"\n✅ Génération des données terminée.")
print(f"📁 Fichier disponible pour l'analyse : {data_path}")