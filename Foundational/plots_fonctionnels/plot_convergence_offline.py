import os
import sys

# ==========================================
# 0. SÉCURITÉS VRAM ET SHARDING (DOIT ÊTRE TOUT EN HAUT)
# ==========================================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"  # <-- CRÉATION DU MESH 'S'

import glob
import json
import zipfile
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk

# Double sécurité pour forcer le mesh JAX dans NetKet
nk.config.update("NETKET_EXPERIMENTAL_SHARDING", True)

import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
RUN_DIR = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/Trains_disordered_1D/run_2026-03-08_19-07-40"

N_SAMPLES = 512      
N_DISCARD = 10       
CHUNK_SIZE = 32      
PROB_GLOBAL_FLIP = 0.05

# ==========================================
# 2. SETUP DES IMPORTS ET REGLES
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if foundational_dir not in sys.path:
    sys.path.insert(0, foundational_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flip_rules import GlobalFlipRule

class SafeGlobalFlipRule(GlobalFlipRule):
    def transition(self, sampler, machine, parameters, state, key, sigma):
        sigma_new, log_prob = super().transition(sampler, machine, parameters, state, key, sigma)
        return jnp.asarray(sigma_new, dtype=sigma.dtype), log_prob

# ==========================================
# 3. CHARGEMENT META ET MODELE
# ==========================================
print(f"📂 Chargement du run : {RUN_DIR}")
with open(os.path.join(RUN_DIR, "meta.json"), 'r') as f:
    meta = json.load(f)

L = meta["L"]
J_val = meta["hamiltonian"]["J"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
vit_params = meta["vit_config"]

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=100)

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

def get_hamiltonian_op(params):
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L))
    return -ha_X - J_val * ha_ZZ

# ==========================================
# 4. SÉLECTION DES CONFIGURATIONS
# ==========================================
disorder_file = os.path.join(RUN_DIR, "disorder_configs.npy")
if not os.path.exists(disorder_file):
    disorder_file = os.path.join(RUN_DIR, "disorder.configs.npy")
params_list = np.load(disorder_file)

n_h0 = len(h0_train_list)
replicas_per_h0 = len(params_list) // n_h0

tracked_indices = [i * replicas_per_h0 for i in range(n_h0)]
tracked_h0 = [h0_train_list[i] for i in range(n_h0)]

print(f"🎯 Suivi de {n_h0} réplicas représentatives (une par h0).")

# ==========================================
# 5. INITIALISATION DES ÉVALUATEURS
# ==========================================
sa_eval = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(PROB_GLOBAL_FLIP), n_chains=16)
vs_base = nkf.FoundationalQuantumState(sa_eval, ma, ps, n_replicas=1, n_samples=1, seed=1)

print("⚙️ Pré-compilation des 10 évaluateurs (ça prend ~1 min max une seule fois)...")

evaluators = []
for idx, r_idx in enumerate(tracked_indices):
    pars = params_list[r_idx]
    H = get_hamiltonian_op(pars)
    _vs_bound = vs_base.get_state(pars)
    
    mc_vs = nk.vqs.MCState(
        sampler=sa_eval, 
        model=_vs_bound.model, 
        variables=_vs_bound.variables, 
        n_samples=N_SAMPLES, 
        n_discard_per_chain=N_DISCARD,
        chunk_size=CHUNK_SIZE
    )
    evaluators.append((H, mc_vs, pars))

# ==========================================
# 6. BOUCLE SUR LES CHECKPOINTS
# ==========================================
checkpoints = glob.glob(os.path.join(RUN_DIR, "state_*.nk"))
checkpoints = sorted(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

history_vscore = {h0: [] for h0 in tracked_h0}
steps = []

print("🚀 Lancement de l'évaluation rapide des checkpoints...")

for ckpt in tqdm(checkpoints, desc="Lecture des Checkpoints"):
    step = int(os.path.basename(ckpt).split('_')[1].split('.')[0])
    steps.append(step)
    
    state_dict = None
    if zipfile.is_zipfile(ckpt):
        with zipfile.ZipFile(ckpt, 'r') as zf:
            candidates = [f for f in zf.namelist() if f.endswith('.msgpack')]
            target_file = sorted(candidates, key=len)[-1]
            with zf.open(target_file) as f:
                state_dict = flax.serialization.msgpack_restore(f.read())
    else:
        with open(ckpt, 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
            
    vars_dict = state_dict.get('variables', state_dict.get('model', {}).get('variables', state_dict))
    vs_base.variables = flax.serialization.from_state_dict(vs_base.variables, vars_dict)
    
    for idx, (H, mc_vs, pars) in enumerate(evaluators):
        h0_val = tracked_h0[idx]
        
        _vs_bound = vs_base.get_state(pars)
        mc_vs.variables = _vs_bound.variables
        mc_vs.reset()
        
        # Astuce 50/50 100% JAX
        sigma_orig = mc_vs.sampler_state.σ
        half = sigma_orig.shape[0] // 2
        sigma_new = sigma_orig.at[:half, :L].set(1).at[half:, :L].set(-1)
        mc_vs.sampler_state = mc_vs.sampler_state.replace(σ=sigma_new)
        
        stats = mc_vs.expect(H)
        vscore = float(np.real(stats.variance) / (np.real(stats.Mean)**2 + 1e-12))
        
        history_vscore[h0_val].append(vscore)

# ==========================================
# 7. TRACÉ DU GRAPHIQUE
# ==========================================
print("\n📈 Génération du Plot...")
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, n_h0)]

for idx, h0_val in enumerate(tracked_h0):
    ax.plot(steps, history_vscore[h0_val], color=colors[idx], marker='o', markersize=4, label=f"$h_0 = {h0_val}$", alpha=0.8)

ax.set_yscale('log')
ax.set_title(f"Convergence par paramètre $h_0$ (Évaluation Offline L={L})", fontsize=14)
ax.set_xlabel("Iterations (Training Steps)", fontsize=12)
ax.set_ylabel("V-score $\\text{Var}(E)/E^2$", fontsize=12)
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

plt.tight_layout()
out_path = os.path.join(RUN_DIR, f"vscore_convergence_offline_L={L}.pdf")
plt.savefig(out_path)
print(f"✅ Plot sauvegardé : {out_path}")