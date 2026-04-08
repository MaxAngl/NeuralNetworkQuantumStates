import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# SLURM multi-GPU: initialiser jax.distributed AVANT tout import JAX/NetKet
if "SLURM_NTASKS" in os.environ and int(os.environ["SLURM_NTASKS"]) > 1:
    import jax
    jax.distributed.initialize()

import netket as nk
import netket_foundational as nkf
from src.nqs_psc.utils import save_run

import time
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax
from netket.utils import struct
import matplotlib.pyplot as plt
from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks.base import AbstractCallback
import netket_pro.distributed as nkpd
from flax import struct
from flip_rules import GlobalFlipRule

# ==========================================
# 1. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
seed = 1
rng = np.random.default_rng(seed)
k = jax.random.key(seed)

# --- PARAMÈTRES PHYSIQUES ---
L = 16
if len(sys.argv) > 1:
    L = int(sys.argv[1])

# 500 valeurs de h₀ équispacées dans [0.8, 1.2]
n_h0 = 500
h0_train_list = np.linspace(0.8, 1.2, n_h0).tolist()

sigma_disorder = 0.1
J_val = 1.0
n_replicas = 5  # réalisations désordonnées par h₀

# --- PARAMÈTRES MONTE CARLO ---
# R = 500 * (5 + 1) = 3000 systèmes
total_configs_train = n_h0 * (n_replicas + 1)

# On vise M ≈ 12000 → M/R = 4 configs par système
chains_per_replica = 1
samples_per_chain = 2
n_chains = total_configs_train * chains_per_replica   # 3000 * 1 = 3000
n_samples = n_chains * samples_per_chain               # 3000 * 2 = 6000
prob_global_flip = 0.05

# --- PARAMÈTRES D'OPTIMISATION ---
n_iter = 600
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = os.path.join(foundational_dir, "logs")

# --- CHUNK SIZE ---
# En multi-GPU, chunk_size doit diviser n_samples_per_rank = n_samples / n_gpus
# On détecte n_gpus depuis SLURM, sinon 1
n_gpus = int(os.environ.get("SLURM_NTASKS", 1))
n_samples_per_rank = n_samples // n_gpus

TARGET_CHUNK = 16
if n_samples_per_rank <= TARGET_CHUNK:
    chunk_size = n_samples_per_rank
else:
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples_per_rank % i == 0:
            chunk_size = i
            break

chunk_size_bwd = 4

print(f"🔹 {n_h0} valeurs de h₀, {n_replicas} réalisations + 1 homogène par h₀")
print(f"🔹 R = {total_configs_train} systèmes, M = {n_samples} samples total, M/R = {n_samples/total_configs_train:.1f}")
print(f"🔹 Chunk size auto-calculé : {chunk_size}")

# ==========================================
# ARCHITECTURE ViT-FNQS
# ==========================================
#
# Vision Transformer adapté pour NQS multimodal (désordre).
#
# Input : concaténation [σ₁...σ_L, γ₁...γ_L] de taille 2L
#   - σ ∈ {-1, +1}^L : configuration de spins
#   - γ ∈ ℝ^L         : champs transverses par site (couplings)
#
# Étape 1 — Embedding (mode O(N) couplings, disorder=True) :
#   - Les spins sont découpés en patches de taille b=1 → L patches
#   - Les couplings sont découpés de la même façon → L patches
#   - Deux matrices d'embedding séparées :
#       W_spin : ℝ^1 → ℝ^(d_model/2)   (16 dims)
#       W_coup : ℝ^1 → ℝ^(d_model/2)   (16 dims)
#   - Concaténation : x_i = [W_spin · σ_patch_i, W_coup · γ_patch_i] ∈ ℝ^d_model
#   - Positional encoding ajouté
#
# Étape 2 — Transformer Encoder (num_layers=2) :
#   - Multi-Head Attention : heads=4, dim par tête = d_model/heads = 8
#   - PAS d'invariance translationnelle (transl_invariant=False pour le désordre)
#   - FFN + LayerNorm standard
#   - Après la 1ère couche, les représentations spin/coupling sont mélangées
#
# Étape 3 — Output :
#   - Somme des vecteurs de sortie : z = Σᵢ yᵢ
#   - Couche dense z → ℂ (amplitude log ψ)
#   - Seule cette dernière couche est complexe
#
# Nombre de paramètres estimé : ~30k-50k
#

vit_params = {
    "num_layers": 2,      # profondeur du Transformer
    "d_model": 32,        # dimension d'embedding (16 spin + 16 coupling)
    "heads": 4,           # têtes d'attention (dim/tête = 8)
    "b": 1,               # taille de patch = 1 site
    "L_eff": L,           # taille effective du système
}

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10 * max(h0_train_list))


def generate_multi_h0_disorder(h0_list, n_reps, system_size, sigma, rng=None):
    """Pour chaque h₀ : n_reps réalisations gaussiennes + 1 config homogène."""
    if rng is None:
        rng = np.random.default_rng()
    
    all_configs = []
    for h_m in h0_list:
        raw_configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        random_configs = np.abs(raw_configs)
        homogeneous_config = np.full((1, system_size), h_m)
        batch_configs = np.vstack([random_configs, homogeneous_config])
        all_configs.append(batch_configs)
        
    return np.vstack(all_configs)


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

sa = nk.sampler.MetropolisSampler(
    hi,
    rule=GlobalFlipRule(prob_global_flip),
    n_chains=n_chains,
)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed, chunk_size=chunk_size)

# Initialisation 50/50 up/down
sigma_orig = vs.sampler_state.σ
flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
half = flat_sigma.shape[0] // 2
flat_sigma = flat_sigma.at[:half, :L].set(1)
flat_sigma = flat_sigma.at[half:, :L].set(-1)
sigma_new = flat_sigma.reshape(sigma_orig.shape)
vs.sampler_state = vs.sampler_state.replace(σ=sigma_new)

params_list = generate_multi_h0_disorder(h0_train_list, n_replicas, hi.size, sigma=sigma_disorder, rng=rng)
print(f"Forme des paramètres de désordre : {params_list.shape}")  # (3000, L)
vs.parameter_array = params_list

# Opérateurs
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))

def create_operator(params):
    assert params.shape == (hi.size,)
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)


# === CALLBACK POUR LOGGER (sous-ensemble) ===
class ReplicaLogger(AbstractCallback):
    params_list: np.ndarray = struct.field(pytree_node=False)
    L: int = struct.field(pytree_node=False)
    eval_every: int = struct.field(pytree_node=False)
    # Indices des configs homogènes à logger (1 sur 50 h₀ ≈ 10 points)
    log_indices: list = struct.field(pytree_node=False)
    
    def __init__(self, params_list, L, n_replicas, n_h0, eval_every=20):
        self.params_list = params_list
        self.L = L
        self.eval_every = eval_every
        # On log ~10 configs homogènes réparties sur la plage de h₀
        step = max(1, n_h0 // 10)
        self.log_indices = [
            i * (n_replicas + 1) + n_replicas 
            for i in range(0, n_h0, step)
        ]
        
    def __call__(self, step, log_data, driver):
        if step % self.eval_every != 0:
            return True
            
        vs = driver.state
        hi = vs.hilbert
        sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=4)
        ham_dict = {}
        
        for idx, i in enumerate(self.log_indices):
            pars = self.params_list[i]
            _vs = vs.get_state(pars)
            
            mc_vs = nk.vqs.MCState(
                sampler=sa_eval, 
                model=_vs.model, 
                variables=_vs.variables, 
                n_samples=256,
                chunk_size=16,
            )
            mc_vs.reset()
            
            sigma_orig = np.array(mc_vs.sampler_state.σ)
            flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
            half_s = flat_sigma.shape[0] // 2
            flat_sigma[:half_s, :self.L] = 1
            flat_sigma[half_s:, :self.L] = -1
            sigma_new = jnp.array(flat_sigma.reshape(sigma_orig.shape))
            mc_vs.sampler_state = mc_vs.sampler_state.replace(**{'σ': sigma_new})
            
            H_op = create_operator(pars)
            stats = mc_vs.expect(H_op)
            
            ham_dict[str(idx)] = {
                "Mean": float(np.real(stats.Mean)),
                "Variance": float(stats.variance),
            }
            
        log_data["ham"] = ham_dict
        return True


# ==========================================
# 3. OPTIMISATION
# ==========================================

class SaveState(AbstractCallback):
    _path: str = struct.field(pytree_node=False)
    _prefix: str = struct.field(pytree_node=False)
    _save_every: int = struct.field(pytree_node=False)

    def __init__(self, path: str, save_every: int, prefix: str = "state"):
        self._path = path
        self._prefix = prefix
        self._save_every = save_every
        if nkpd.is_master_process():
            os.makedirs(self._path, exist_ok=True)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            if nkpd.is_master_process() and not os.path.exists(self._path):
                os.makedirs(self._path, exist_ok=True)
            path = os.path.join(self._path, f"{self._prefix}_{driver.step_count}.nk")
            driver.state.save(path)

learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=500)
optimizer = optax.sgd(learning_rate)

def cg_solver(A, b):
    return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]

gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift, linear_solver_fn=cg_solver, chunk_size_bwd=chunk_size_bwd, use_ntk=True)

log = nk.logging.JsonLog("log_data", save_params=False) 

meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {
        "type": "Ising Gaussian Disorder", 
        "J": J_val, 
        "h0_range": [0.8, 1.2],
        "n_h0": n_h0,
        "sigma_disorder": sigma_disorder,
    },
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "architecture_summary": {
        "input": f"concat([sigma, gamma]) de taille 2*{L}={2*L}",
        "patches": f"{L} patches de taille b=1 (spin) + b=1 (coupling)",
        "embedding": f"2 matrices séparées ℝ^1 → ℝ^{vit_params['d_model']//2}, concaténées → ℝ^{vit_params['d_model']}",
        "transformer": f"{vit_params['num_layers']} couches, {vit_params['heads']} têtes (dim/tête={vit_params['d_model']//vit_params['heads']})",
        "output": f"somme des {L} vecteurs → dense → ℂ",
        "translational_invariance": False,
    },
    "sampler": {
        "type": "MetropolisGlobalFlip", 
        "n_chains": n_chains, 
        "n_samples": n_samples,
        "prob_global_flip": prob_global_flip,
    },
    "optimizer": {
        "type": "SGD + SR (CG solver)", 
        "lr_init": lr_init, 
        "lr_end": lr_end, 
        "diag_shift": diag_shift,
        "use_ntk": True,
    },
    "n_iter": n_iter,
    "n_replicas_per_h0": n_replicas,
    "total_configs_train": total_configs_train,
    "M_total": n_samples,
    "M_per_system": n_samples / total_configs_train,
    "seed": seed,
}

try:
    # Inclure L dans le run_dir pour éviter les collisions entre tailles
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    explicit_run_dir = os.path.join(logs_path, f"run_L{L}_{timestamp}")
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path, run_dir=explicit_run_dir)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"

log = nk.logging.JsonLog(os.path.join(run_dir, "log_data.json"), save_params=False)

disorder_path = os.path.join(run_dir, "disorder_configs.npy")
np.save(disorder_path, params_list)

# Sauvegarder aussi la liste des h₀
h0_path = os.path.join(run_dir, "h0_train_list.npy")
np.save(h0_path, np.array(h0_train_list))

print(f"Configs désordre : {disorder_path}")
print(f"Liste h₀ : {h0_path}")

vs.chunk_size = chunk_size
start_time = time.time()

gs.run(
    n_iter,
    out=log,
    callback=[
        SaveState(run_dir, 20), 
        ReplicaLogger(params_list, L, n_replicas, n_h0, eval_every=30),
    ]
)

if "Energy" in log.data:
    log.data["ham"] = log.data["Energy"]

duration = time.time() - start_time
print(f"⏱️ Temps total d'entraînement : {duration:.2f} secondes")

meta["execution_time_seconds"] = duration
import json
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4, default=str)

# ==========================================
# 4. PLOTS DE SORTIE
# ==========================================
print('Génération des plots...')

# --- PLOT 1 : Convergence de la loss (énergie moyenne vs itérations) ---
fig, ax = plt.subplots(figsize=(9, 5))

# L'énergie totale (loss) est loggée par défaut par VMC_NG
if "Energy" in log.data:
    energy_log = log.data["Energy"]
    iters = np.array(energy_log.iters)
    e_mean = np.real(np.array(energy_log.Mean))
    e_err = np.array(energy_log.Sigma) if hasattr(energy_log, 'Sigma') else None
    
    ax.plot(iters, e_mean, "b-", linewidth=1.2, label="⟨E⟩ (loss)")
    if e_err is not None:
        ax.fill_between(iters, e_mean - e_err, e_mean + e_err, alpha=0.2, color="b")
    
    ax.set_xlabel("Itérations", fontsize=13)
    ax.set_ylabel("Énergie moyenne (loss)", fontsize=13)
    ax.set_title(f"Convergence — R={total_configs_train}, M={n_samples}, L={L}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "Pas de données d'énergie dans le log", 
            transform=ax.transAxes, ha="center", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "convergence_loss.pdf"), dpi=150)
plt.savefig(os.path.join(run_dir, "convergence_loss.png"), dpi=150)
plt.close()
print("  ✓ convergence_loss.pdf")

# --- PLOT 2 : Scatter V-score vs h₀ ---
print("Calcul des V-scores sur configs homogènes...")

# Évaluer ~25 configs homogènes réparties sur [0.8, 1.2]
eval_step = max(1, n_h0 // 25)
eval_indices = [i * (n_replicas + 1) + n_replicas for i in range(0, n_h0, eval_step)]
eval_h0 = [h0_train_list[i] for i in range(0, n_h0, eval_step)]

v_scores = []
energies = []

for idx in tqdm(eval_indices):
    pars = params_list[idx]
    _vs = vs.get_state(pars)
    
    vs_mc = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hi, n_chains=16), 
        model=_vs.model, 
        variables=_vs.variables, 
        n_samples=1024, 
        chunk_size=64,
    )
    
    _e = vs_mc.expect(create_operator(pars))
    v_scores.append(float(_e.variance / (_e.Mean.real**2 + 1e-12)))
    energies.append(float(np.real(_e.Mean)))

# Sauvegarde CSV
df = pd.DataFrame({"h0": eval_h0, "v_score": v_scores, "energy": energies})
df.to_csv(os.path.join(run_dir, "quick_eval.csv"), index=False)

# Scatter plot
fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(eval_h0, v_scores, c=eval_h0, cmap="coolwarm", 
                     s=60, edgecolors="k", linewidths=0.5, zorder=3)
ax.set_yscale("log")
ax.set_xlabel("h₀ / J", fontsize=13)
ax.set_ylabel("V-score", fontsize=13)
ax.set_title(f"V-score par h₀ (configs homogènes) — L={L}", fontsize=14)
ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="h₀/J = 1")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("h₀", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "v_score_scatter.pdf"), dpi=150)
plt.savefig(os.path.join(run_dir, "v_score_scatter.png"), dpi=150)
plt.close()
print("  ✓ v_score_scatter.pdf")

print(f"\n✅ Terminé ! Résultats dans : {run_dir}")
