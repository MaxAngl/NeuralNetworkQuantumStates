import os
import sys

# ==========================================
# 0. FORCE CPU (Si nécessaire)
# ==========================================
# Décommente cette ligne si tu as l'erreur CUDA et que tu veux tourner sur CPU
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 👇 MODIFIEZ LE CHEMIN ICI 👇
RUN_DIR = r"/users/eleves-a/2024/adel.mana/Documents/NeuralNetworkQuantumStates/Foundational/logs/Trains_disordered_1D/run_2026-03-07_20-22-49"

# Paramètres de test
H0_TEST_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975,0.980,0.985,0.990,0.993,0.995,0.997, 1.0,1.003,1.005,1.010,1.012,1.015,1.025,1.035, 1.05, 1.075, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 3.0, 4.0] 
SIGMA_TEST_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]  

prob_global_flip = 0.03
N_SAMPLES_MC = 2048 # Nombre de samples pour l'estimation de Mz^2
n_chains=64
n_discard_per_chain=100

# --- PARAMÈTRES D'ÉCHANTILLONNAGE PAR ZONES ---
H0_TRANS_MIN = 0.5   # Fin de la zone 1 (Loin de la transition, petit h0)
H0_TRANS_MAX = 1.7   # Début de la zone 3 (Loin de la transition, grand h0)


N_KEEP_BEFORE = 20   # Nombre de tirages pour h0 <= H0_TRANS_MIN
N_KEEP_TRANS  = 60 # Nombre de tirages pour la zone critique
N_KEEP_AFTER  = 10   # Nombre de tirages pour h0 >= H0_TRANS_MAX

# ==========================================
# 2. SETUP ET CHARGEMENT
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
print(f"🔹 Tests sur les sigmas : {SIGMA_TEST_LIST}")
print(f"🔹 Zones d'échantillonnage : {N_KEEP_BEFORE} (h0<={H0_TRANS_MIN}) | {N_KEEP_TRANS} (Critique) | {N_KEEP_AFTER} (h0>={H0_TRANS_MAX})")

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
Mz_operator = sum(nk.operator.spin.sigmaz(hi, i) for i in range(L)) * (1.0 / L)
Mz2_operator = Mz_operator @ Mz_operator


# ==========================================
# 4. FONCTION DE CALCUL (ÉCHANTILLONNAGE CORRÉLÉ OPTIMISÉ)
# ==========================================
print("\n🚀 Lancement des calculs de l'aimantation...")

rng = np.random.default_rng(seed=42)
sa_multi = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=n_chains)

dummy_params = np.zeros(L)
_vs_init = vs.get_state(dummy_params)
print(type(_vs_init))

mc_vs = nk.vqs.MCState(
    sampler=sa_multi,
    model=_vs_init.model,
    variables=_vs_init.variables,
    n_samples=N_SAMPLES_MC, 
    n_discard_per_chain=n_discard_per_chain
)

results = {}

# On détermine la taille max du tableau de bruit une seule fois
MAX_KEEP = max(N_KEEP_BEFORE, N_KEEP_TRANS, N_KEEP_AFTER)

for sigma in SIGMA_TEST_LIST:
    results[sigma] = {"mean": [], "min": [], "max": []}
    print(f"\n▶ Traitement pour sigma = {sigma}")
    
    # GÉNÉRATION DU BRUIT DE BASE UNIQUE (Taille maximale nécessaire)
    base_noise = rng.normal(loc=0.0, scale=sigma, size=(MAX_KEEP, L))
    
    for h0 in tqdm(H0_TEST_LIST, desc=f"Balayage h0"):
        
        # SÉLECTION DU NOMBRE DE TIRAGES SELON LA ZONE
        if h0 <= H0_TRANS_MIN:
            n_keep = N_KEEP_BEFORE
        elif h0 >= H0_TRANS_MAX:
            n_keep = N_KEEP_AFTER
        else:
            n_keep = N_KEEP_TRANS
            
        # Extraction stricte pour garantir la corrélation statistique
        current_base_noise = base_noise[:n_keep]
        
        # Application du décalage h0 et du repliement (valeur absolue)
        configs = np.abs(h0 + current_base_noise)
        
        mz2_batch = []
        for pars in configs:
            mc_vs.variables = vs.get_state(pars).variables
            mc_vs.reset()
            
            stats = mc_vs.expect(Mz2_operator)
            mz2_val = float(stats.Mean.real)
            mz2_batch.append(mz2_val)
            
        results[sigma]["mean"].append(np.mean(mz2_batch))
        results[sigma]["min"].append(np.min(mz2_batch))
        results[sigma]["max"].append(np.max(mz2_batch))


# ==========================================
# 4.5 SAUVEGARDE DES DONNÉES
# ==========================================
data_path = os.path.join(RUN_DIR, f"mz2_data_L={L}.npz")

# Conversion en arrays numpy pour la sauvegarde
curves_array = np.array([results[s]["mean"] for s in SIGMA_TEST_LIST])
min_array    = np.array([results[s]["min"]  for s in SIGMA_TEST_LIST])
max_array    = np.array([results[s]["max"]  for s in SIGMA_TEST_LIST])

np.savez(
    data_path,
    sigma_grid      = np.array(SIGMA_TEST_LIST),
    h0_grid         = np.array(H0_TEST_LIST),
    mz2_mean        = curves_array,
    mz2_min         = min_array,
    mz2_max         = max_array,
    L               = L,
    N_KEEP_BEFORE   = N_KEEP_BEFORE,
    N_KEEP_TRANS    = N_KEEP_TRANS,
    N_KEEP_AFTER    = N_KEEP_AFTER,
    H0_TRANS_MIN    = H0_TRANS_MIN,
    H0_TRANS_MAX    = H0_TRANS_MAX,
    N_SAMPLES_MC    = N_SAMPLES_MC,
)
print(f"✅ Données sauvegardées : {data_path}")
# ==========================================
# 5. PLOTTING
# ==========================================
print("\n📈 Génération du graphique avec dérivée...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(SIGMA_TEST_LIST)))
max_derivatives = []

for idx, sigma in enumerate(SIGMA_TEST_LIST):
    c = colors[idx]
    
    mean_vals = np.array(results[sigma]["mean"])
    min_vals = np.array(results[sigma]["min"])
    max_vals = np.array(results[sigma]["max"])
    
    # Plot Aimantation
    ax1.plot(H0_TEST_LIST, mean_vals, marker='o', markersize=3, color=c, linewidth=1, label=rf"$\sigma = {sigma}$", zorder=3)
    ax1.plot(H0_TEST_LIST, min_vals, linestyle='--', color=c, alpha=0.5, linewidth=0.5, zorder=2)
    ax1.plot(H0_TEST_LIST, max_vals, linestyle='--', color=c, alpha=0.5, linewidth=0.5, zorder=2)
    
    # Calcul Dérivée
    derivative = np.gradient(mean_vals, H0_TEST_LIST)
    max_abs_deriv = np.max(np.abs(derivative))
    max_derivatives.append(max_abs_deriv)

ax1.set_xlabel(r"Transverse Field $h_0$", fontsize=12)
ax1.set_ylabel(r"Squared Magnetization $\langle M_z^2 \rangle$", fontsize=12)
ax1.set_title(f"Magnetization order parameter vs Transverse Field (L={L})", fontsize=14)
ax1.grid(True, which="both", ls="--", alpha=0.3)
ax1.legend(loc='upper right', frameon=True, fontsize=10, title="Disorder strength")

ax2.plot(SIGMA_TEST_LIST, max_derivatives, marker='s', markersize=6, color='crimson', linewidth=1.5, linestyle='-')
ax2.set_xlabel(r"Disorder strength $\sigma$", fontsize=12)
ax2.set_ylabel(r"$\max \left| \frac{\partial \langle M_z^2 \rangle}{\partial h_0} \right|$", fontsize=12)
ax2.set_title(r"Maximum Susceptibility vs Disorder", fontsize=14)
ax2.grid(True, which="both", ls="--", alpha=0.3)

output_file = os.path.join(RUN_DIR, f"mz2_and_derivative_L={L}_64_chains.pdf")
plt.tight_layout()
plt.savefig(output_file)
print(f"✅ Graphique sauvegardé : {output_file}")