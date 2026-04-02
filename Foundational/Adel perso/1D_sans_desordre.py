import os
import jax

# Initialisation du cluster JAX
try:
    jax.distributed.initialize()
    print(f"JAX Cluster initialisé : Processus {jax.process_index()} / {jax.process_count()}")
except Exception as e:
    print(f"Simple exécution locale ou erreur d'init : {e}")

print(f"👋 Bonjour depuis le noeud {jax.process_index()} sur {jax.process_count()} !")

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)

# Décommenter pour L supérieur à 16 ou 20
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk
import netket_foundational as nkf

from src.nqs_psc.utils import save_run

import time
import json
import pandas as pd
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax
from netket.utils import struct
import matplotlib.pyplot as plt
from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks import AbstractCallback
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

# Liste des champs transverses d'entraînement (SANS désordre)
# Densité accrue entre 0.5 et 1.5 pour couvrir la transition de phase
h0_train_list = [
    0.1, 0.3,
    0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 0.97, 0.995,
    1.0, 1.01, 1.025, 1.03, 1.05, 1.06, 1.08,
    1.1, 1.25, 1.3, 1.4, 1.5,
    2.0, 3.0
]
J_val = 1.0

# Pas de réplicas de désordre : chaque h0 donne exactement 1 configuration homogène
# => total_configs_train = len(h0_train_list)
total_configs_train = len(h0_train_list)

# --- PARAMÈTRES MONTE CARLO ---
chains_per_config = 4
samples_per_chain = 2
n_chains = total_configs_train * chains_per_config
n_samples = n_chains * samples_per_chain
prob_global_flip = 0.05

# --- PARAMÈTRES D'OPTIMISATION ---
n_iter = 600
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = "/users/eleves-a/2024/adel.mana/Documents/NeuralNetworkQuantumStates/Foundational/Adel perso"

# --- CALCUL AUTOMATIQUE DU CHUNK_SIZE ---
TARGET_CHUNK = 10
if n_samples <= TARGET_CHUNK:
    chunk_size = n_samples
else:
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples % i == 0:
            chunk_size = i
            break

chunk_size_bwd = 4

print(f"🔹 Configuration : {n_samples} samples total.")
print(f"🔹 Chunk size auto-calculé : {chunk_size} (Diviseur optimal <= {TARGET_CHUNK})")

# Paramètres du modèle ViT
vit_params = {
    "num_layers": 4,
    "d_model": 60,
    "heads": 10,
    "b": 4,
    "L_eff": L // 4,
}

# ==========================================
# 2. DÉFINITION DU SYSTÈME
# ==========================================

hi = nk.hilbert.Spin(0.5, L)
# ParameterSpace toujours nécessaire pour l'architecture FoundationalQuantumState
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10 * max(h0_train_list))

# --- GÉNÉRATION DES CONFIGURATIONS HOMOGÈNES (PAS DE DÉSORDRE) ---
# Chaque configuration est un vecteur constant (h0, h0, ..., h0)
def generate_clean_configs(h0_list, system_size):
    """
    Génère une configuration homogène par valeur de h0.
    Shape de sortie : (len(h0_list), system_size)
    """
    configs = []
    for h in h0_list:
        configs.append(np.full(system_size, h))
    return np.array(configs)

# Modèle ViT (identique au script d'origine, sans disorder=False car
# l'architecture accepte quand même le ParameterSpace)
ma = ViTFNQS(
    num_layers=vit_params["num_layers"],
    d_model=vit_params["d_model"],
    heads=vit_params["heads"],
    b=vit_params["b"],
    L_eff=vit_params["L_eff"],
    n_coups=ps.size,
    complex=True,
    disorder=True,       # L'architecture reste identique pour pouvoir charger les poids
    transl_invariant=False,
    two_dimensional=False,
)

# Sampler & État Variationnel
sa = nk.sampler.MetropolisSampler(
    hi,
    rule=GlobalFlipRule(prob_global_flip),
    n_chains=n_chains
)
vs = nkf.FoundationalQuantumState(
    sa, ma, ps,
    n_replicas=total_configs_train,
    n_samples=n_samples,
    seed=seed,
    chunk_size=chunk_size
)

# Initialisation 50/50 UP/DOWN (même astuce que l'original)
sigma_orig = vs.sampler_state.σ
flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
half = flat_sigma.shape[0] // 2
flat_sigma = flat_sigma.at[:half, :L].set(1)
flat_sigma = flat_sigma.at[half:, :L].set(-1)
sigma_new = flat_sigma.reshape(sigma_orig.shape)
vs.sampler_state = vs.sampler_state.replace(σ=sigma_new)

# Injection des configurations HOMOGÈNES (pas de désordre)
params_list = generate_clean_configs(h0_train_list, hi.size)
print(f"Forme des paramètres (sans désordre) : {params_list.shape}")
vs.parameter_array = params_list

# Opérateurs
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1.0 / float(hi.size))

def create_operator(params):
    assert params.shape == (hi.size,)
    ha_X  = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
                for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

ha_p  = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p  = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)

# ==========================================
# 3. CALLBACKS
# ==========================================

class ReplicaLogger(AbstractCallback):
    """
    Évalue l'énergie et la variance pour chaque configuration h0
    toutes les `eval_every` itérations et sauvegarde en .npy.
    """
    params_list: np.ndarray = struct.field(pytree_node=False)
    L: int = struct.field(pytree_node=False)
    eval_every: int = struct.field(pytree_node=False)
    run_dir: str = struct.field(pytree_node=False)

    iters:     list = struct.field(pytree_node=False, default_factory=list)
    energies:  list = struct.field(pytree_node=False, default_factory=list)
    variances: list = struct.field(pytree_node=False, default_factory=list)

    def __init__(self, params_list, L, run_dir, eval_every=10):
        self.params_list = params_list
        self.L = L
        self.run_dir = run_dir
        self.eval_every = eval_every
        self.iters = []
        self.energies = []
        self.variances = []

    def __call__(self, step, log_data, driver):
        if step % self.eval_every != 0:
            return True

        vs = driver.state
        hi = vs.hilbert
        sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=4)

        step_energies = []
        step_variances = []

        for i, pars in enumerate(self.params_list):
            _vs = vs.get_state(pars)

            mc_vs = nk.vqs.MCState(
                sampler=sa_eval,
                model=_vs.model,
                variables=_vs.variables,
                n_samples=256,
                chunk_size=16
            )
            mc_vs.reset()

            # Astuce 50/50 pour briser la symétrie Z2 lors de la mesure
            sigma_orig = np.array(mc_vs.sampler_state.σ)
            flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
            half = flat_sigma.shape[0] // 2
            flat_sigma[:half, :self.L] = 1
            flat_sigma[half:, :self.L] = -1
            sigma_new = jnp.array(flat_sigma.reshape(sigma_orig.shape))
            mc_vs.sampler_state = mc_vs.sampler_state.replace(**{'σ': sigma_new})

            H_op = create_operator(pars)
            stats = mc_vs.expect(H_op)

            step_energies.append(float(np.real(stats.Mean)))
            step_variances.append(float(stats.variance))

        self.iters.append(step)
        self.energies.append(step_energies)
        self.variances.append(step_variances)

        if nkpd.is_master_process():
            np.save(os.path.join(self.run_dir, "replica_iters.npy"),    np.array(self.iters))
            np.save(os.path.join(self.run_dir, "replica_energies.npy"), np.array(self.energies))
            np.save(os.path.join(self.run_dir, "replica_variances.npy"),np.array(self.variances))

        return True


class SaveState(AbstractCallback):
    _path:       str = struct.field(pytree_node=False)
    _prefix:     str = struct.field(pytree_node=False)
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

# ==========================================
# 4. LOGGING ET OPTIMISATION
# ==========================================

learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300)
optimizer = optax.sgd(learning_rate)

def cg_solver(A, b):
    return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]

gs = nkf.VMC_NG(
    ha_p, optimizer,
    variational_state=vs,
    diag_shift=diag_shift,
    linear_solver_fn=cg_solver,
    chunk_size_bwd=chunk_size_bwd,
    use_ntk=True
)

log = nk.logging.JsonLog("log_data", save_params=False)

meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {
        "type": "Ising Clean (no disorder)",
        "J": J_val,
        "h0_train_list": h0_train_list,
        "sigma": 0.0
    },
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "sampler": {
        "type": "MetropolisGlobalFlip",
        "n_chains": n_chains,
        "n_samples": n_samples
    },
    "optimizer": {
        "type": "SGD",
        "lr_init": lr_init,
        "lr_end": lr_end,
        "diag_shift": diag_shift
    },
    "n_iter": n_iter,
    "total_configs_train": total_configs_train,
    "seed": seed,
}

try:
    os.makedirs(logs_path, exist_ok=True)
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = os.path.join(logs_path, f"run_L={L}_fallback")
    os.makedirs(run_dir, exist_ok=True)

log = nk.logging.JsonLog(os.path.join(run_dir, "log_data.json"), save_params=False)

# Sauvegarde des configurations (toutes homogènes ici)
configs_path = os.path.join(run_dir, "h0_configs.npy")
np.save(configs_path, params_list)
print(f"Configurations h0 sauvegardées dans : {configs_path}")

vs.chunk_size = chunk_size

start_time = time.time()

gs.run(
    n_iter,
    out=log,
    callback=[
        SaveState(run_dir, 10),
        ReplicaLogger(params_list, L, run_dir=run_dir, eval_every=10)
    ]
)

if "Energy" in log.data:
    log.data["ham"] = log.data["Energy"]

duration = time.time() - start_time
print(f"⏱️ Temps total d'entraînement : {duration:.2f} secondes")

meta["execution_time_seconds"] = duration
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

# ==========================================
# 5. ANALYSE FINALE (MASTER PROCESS ONLY)
# ==========================================
DERIVATIVE_STEP  = 3
N_EVAL_DRAWS     = 20    # Répétitions MC par h0 pour les barres d'erreur
N_EVAL_SAMPLES   = 1024
N_EVAL_CHAINS    = 16
N_DISCARD        = 50
H0_TRANS_MIN_EVAL = 0.5
H0_TRANS_MAX_EVAL = 1.5
N_KEEP_BEFORE    = 5
N_KEEP_TRANS     = N_EVAL_DRAWS
N_KEEP_AFTER     = 5
MAX_KEEP         = max(N_KEEP_BEFORE, N_KEEP_TRANS, N_KEEP_AFTER)

def robust_derivative(y, x, step=1):
    """Dérivée centrée avec un écartement `step` pour filtrer le bruit MC."""
    dy = np.zeros_like(y, dtype=float)
    n  = len(y)
    for i in range(n):
        left  = max(0, i - step)
        right = min(n - 1, i + step)
        if right > left:
            dy[i] = (y[right] - y[left]) / (x[right] - x[left])
    return np.abs(dy)

if nkpd.is_master_process():
    print('\n' + '='*60)
    print('  ANALYSE FINALE — génération de tous les graphiques')
    print('='*60)

    # ----------------------------------------------------------
    # A. Collecte des métriques de CONVERGENCE (sur h0_train_list)
    # ----------------------------------------------------------
    train_results = {"h0": [], "energy": [], "energy_err": [], "v_score": [], "r_hat": []}

    sa_eval_train = nk.sampler.MetropolisLocal(hi, n_chains=N_EVAL_CHAINS)
    for r in tqdm(range(total_configs_train), desc="Évaluation finale (train points)"):
        pars  = params_list[r]
        _vs   = vs.get_state(pars)
        vs_mc = nk.vqs.MCState(
            sampler=sa_eval_train,
            model=_vs.model, variables=_vs.variables,
            n_samples=N_EVAL_SAMPLES, chunk_size=64
        )
        _e    = vs_mc.expect(create_operator(pars))
        h0_val = float(pars[0])
        train_results["h0"].append(h0_val)
        train_results["energy"].append(float(np.real(_e.Mean)))
        train_results["energy_err"].append(float(np.sqrt(_e.variance / _e.n_samples)))
        train_results["v_score"].append(float(_e.variance / (_e.Mean.real**2 + 1e-12)))
        train_results["r_hat"].append(float(getattr(_e, 'R_hat', np.nan)))

    df = pd.DataFrame(train_results).sort_values("h0")
    df.to_csv(os.path.join(run_dir, "train_results.csv"), index=False)

    # ----------------------------------------------------------
    # B. Calcul de Mz² sur une grille dense (0 → 3)
    # ----------------------------------------------------------
    # Grille dense avec concentration autour de hc=1
    H0_DENSE = sorted(set(h0_train_list) | {
        0.0, 0.1, 0.3,
        0.5, 0.6, 0.7, 0.8, 0.85, 0.90,
        0.95, 0.97, 0.975, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997,
        1.0,
        1.003, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.04,
        1.05, 1.06, 1.08, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35,
        1.4,1.45, 1.5, 2.0, 2.5, 3.0
    })
    H0_DENSE = [h for h in H0_DENSE if 0.0 <= h <= 3.0]

    mz2_data_path = os.path.join(run_dir, f"mz2_data_L={L}.npz")
    mz2_raw_arr    = np.full((len(H0_DENSE), MAX_KEEP), np.nan)
    energy_raw_arr = np.full((len(H0_DENSE), MAX_KEEP), np.nan)
    computed_mask  = np.zeros(len(H0_DENSE), dtype=bool)

    # Reprise si déjà calculé
    if os.path.exists(mz2_data_path):
        loaded = np.load(mz2_data_path)
        saved_h0 = loaded['h0_grid'].tolist()
        if len(saved_h0) == len(H0_DENSE) and np.allclose(saved_h0, H0_DENSE):
            mz2_raw_arr    = loaded['mz2_raw']
            energy_raw_arr = loaded['energy_raw']
            computed_mask  = ~np.isnan(mz2_raw_arr[:, 0])
            print(f"🔄 Reprise Mz² : {computed_mask.sum()} / {len(H0_DENSE)} points déjà calculés.")

    sa_dense = nk.sampler.MetropolisLocal(hi, n_chains=N_EVAL_CHAINS)
    dummy_pars  = np.full(L, 1.0)
    _vs_dummy   = vs.get_state(dummy_pars)
    mc_dense    = nk.vqs.MCState(
        sampler=sa_dense,
        model=_vs_dummy.model, variables=_vs_dummy.variables,
        n_samples=N_EVAL_SAMPLES, n_discard_per_chain=N_DISCARD
    )
    Mz_op  = sum(nkf.operator.sigmaz(hi, i) for i in range(L)) * (1.0 / L)
    Mz2_op = Mz_op @ Mz_op

    for idx_h, h0 in enumerate(tqdm(H0_DENSE, desc="Calcul Mz² grille dense")):
        if computed_mask[idx_h]:
            continue
        if   h0 <= H0_TRANS_MIN_EVAL: n_keep = N_KEEP_BEFORE
        elif h0 >= H0_TRANS_MAX_EVAL: n_keep = N_KEEP_AFTER
        else:                         n_keep = N_KEEP_TRANS

        pars = np.full(L, h0)
        mc_dense.variables = vs.get_state(pars).variables

        for draw in range(n_keep):
            mc_dense.reset()
            # Astuce 50/50 Z2
            s_orig  = np.array(mc_dense.sampler_state.σ)
            flat    = s_orig.reshape(-1, s_orig.shape[-1])
            half    = flat.shape[0] // 2
            flat[:half, :L] = 1
            flat[half:, :L] = -1
            mc_dense.sampler_state = mc_dense.sampler_state.replace(
                **{'σ': jnp.array(flat.reshape(s_orig.shape))}
            )
            st_mz2 = mc_dense.expect(Mz2_op)
            st_en  = mc_dense.expect(create_operator(pars))
            mz2_raw_arr[idx_h, draw]    = float(st_mz2.Mean.real)
            energy_raw_arr[idx_h, draw] = float(st_en.Mean.real)

        computed_mask[idx_h] = True
        np.savez(mz2_data_path,
                 h0_grid=np.array(H0_DENSE),
                 mz2_raw=mz2_raw_arr,
                 energy_raw=energy_raw_arr,
                 L=L, N_SAMPLES_MC=N_EVAL_SAMPLES)

    h0_arr      = np.array(H0_DENSE)
    mz2_mean    = np.nanmean(mz2_raw_arr, axis=1)
    mz2_sem     = np.nanstd(mz2_raw_arr, axis=1, ddof=1) / np.sqrt(
                      np.sum(~np.isnan(mz2_raw_arr), axis=1).clip(1))
    en_mean     = np.nanmean(energy_raw_arr, axis=1)
    en_sem      = np.nanstd(energy_raw_arr, axis=1, ddof=1) / np.sqrt(
                      np.sum(~np.isnan(energy_raw_arr), axis=1).clip(1))
    dmz2_dh     = robust_derivative(mz2_mean, h0_arr, step=DERIVATIVE_STEP)
    hc_est      = h0_arr[np.argmax(dmz2_dh)]

    micro_max_list = []
    for draw in range(MAX_KEEP):
        curve = mz2_raw_arr[:, draw]
        valid = ~np.isnan(curve)
        if np.sum(valid) > 2 * DERIVATIVE_STEP:
            micro_max_list.append(np.max(
                robust_derivative(curve[valid], h0_arr[valid], step=DERIVATIVE_STEP)
            ))
    micro_mean_max = np.mean(micro_max_list)
    micro_sem_max  = np.std(micro_max_list, ddof=1) / np.sqrt(len(micro_max_list)) if len(micro_max_list) > 1 else 0.0

    print(f"\n🎯 Position estimée du point critique : hc ≈ {hc_est:.4f}  (exact : 1.0)")
    print(f"   Erreur relative : {abs(hc_est - 1.0)*100:.2f} %")

    # ----------------------------------------------------------
    # FIGURE 1 : Grille de convergence V-score (par h0 d'entraînement)
    # ----------------------------------------------------------
    num_h0 = len(h0_train_list)
    cols   = 4
    rows   = (num_h0 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    colors_cv = plt.cm.plasma(np.linspace(0, 0.9, num_h0))

    try:
        hist_iters     = np.load(os.path.join(run_dir, "replica_iters.npy"))
        hist_energies  = np.load(os.path.join(run_dir, "replica_energies.npy"))
        hist_variances = np.load(os.path.join(run_dir, "replica_variances.npy"))

        for idx, h0 in enumerate(h0_train_list):
            ax = axes[idx // cols, idx % cols]
            c  = colors_cv[idx]
            v_scores = hist_variances[:, idx] / (hist_energies[:, idx]**2 + 1e-12)
            ax.plot(hist_iters, v_scores, linewidth=1.2, color=c)
            ax.set_yscale('log')
            ax.set_title(rf"$h_0 = {h0}$", color=c, fontweight='bold')
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"V-score")
            ax.grid(True, which="both", ls="--", alpha=0.3)
    except FileNotFoundError:
        print("⚠️ Fichiers .npy du ReplicaLogger introuvables.")

    for idx in range(num_h0, rows * cols):
        fig.delaxes(axes[idx // cols, idx % cols])
    plt.suptitle(f"V-score Convergence — Clean Ising L={L}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"vscore_convergence_grid_L={L}.pdf"), bbox_inches='tight')
    plt.clf()

    # ----------------------------------------------------------
    # FIGURE 2 : Transition de phase — 4 panneaux (0 → 3)
    # ----------------------------------------------------------
    from matplotlib.gridspec import GridSpec
    CRIT_COLOR = 'crimson'
    LW = 1.5
    MS = 4

    fig2 = plt.figure(figsize=(22, 5.5))
    gs   = GridSpec(1, 4, figure=fig2, wspace=0.35)

    # P1 : Mz²
    ax1 = fig2.add_subplot(gs[0])
    ax1.errorbar(h0_arr, mz2_mean, yerr=mz2_sem,
                 fmt='o-', color='steelblue', markersize=MS, linewidth=LW,
                 capsize=3, elinewidth=0.8, label=r"$\langle M_z^2 \rangle$")
    ax1.fill_between(h0_arr, mz2_mean - mz2_sem, mz2_mean + mz2_sem, color='steelblue', alpha=0.15)
    ax1.axvline(x=1.0,    color=CRIT_COLOR, linestyle='--', linewidth=1.2, alpha=0.7, label=r"$h_c^{\rm exact}=1$")
    ax1.axvline(x=hc_est, color='orange',   linestyle=':',  linewidth=1.2, alpha=0.8, label=rf"$h_c^{{\rm NQS}}\approx{hc_est:.3f}$")
    ax1.set_xlim(0, 3)
    ax1.set_xlabel(r"Transverse Field $h_0$", fontsize=12)
    ax1.set_ylabel(r"$\langle M_z^2 \rangle$", fontsize=12)
    ax1.set_title("Mean Squared Magnetization", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, ls="--", alpha=0.3)

    # P2 : Susceptibilité |dMz²/dh|
    ax2 = fig2.add_subplot(gs[1])
    ax2.plot(h0_arr, dmz2_dh, 'o-', color='darkorchid', markersize=MS, linewidth=LW)
    ax2.axvline(x=1.0,    color=CRIT_COLOR, linestyle='--', linewidth=1.2, alpha=0.7, label=r"$h_c=1$")
    ax2.axvline(x=hc_est, color='orange',   linestyle=':',  linewidth=1.2, alpha=0.8, label=rf"$h_c^{{\rm NQS}}\approx{hc_est:.3f}$")
    ax2.set_xlim(0, 3)
    ax2.set_xlabel(r"Transverse Field $h_0$", fontsize=12)
    ax2.set_ylabel(r"$\left|\partial \langle M_z^2\rangle / \partial h_0\right|$", fontsize=12)
    ax2.set_title(f"Susceptibility (span={DERIVATIVE_STEP})", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, ls="--", alpha=0.3)

    # P3 : Énergie par site
    ax3 = fig2.add_subplot(gs[2])
    ax3.errorbar(h0_arr, en_mean / L, yerr=en_sem / L,
                 fmt='s-', color='teal', markersize=MS, linewidth=LW,
                 capsize=3, elinewidth=0.8)
    ax3.axvline(x=1.0, color=CRIT_COLOR, linestyle='--', linewidth=1.2, alpha=0.7, label=r"$h_c=1$")
    ax3.set_xlim(0, 3)
    ax3.set_xlabel(r"Transverse Field $h_0$", fontsize=12)
    ax3.set_ylabel(r"Energy per site $E/L$", fontsize=12)
    ax3.set_title("Ground State Energy", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, ls="--", alpha=0.3)

    # P4 : Fluctuations MC du pic de susceptibilité
    ax4 = fig2.add_subplot(gs[3])
    if micro_max_list:
        ax4.hist(micro_max_list, bins=max(5, len(micro_max_list) // 3),
                 color='slategray', edgecolor='white', alpha=0.85)
        ax4.axvline(x=micro_mean_max, color=CRIT_COLOR, linewidth=2,
                    label=rf"$\mu = {micro_mean_max:.3f}$")
        ax4.axvspan(micro_mean_max - micro_sem_max,
                    micro_mean_max + micro_sem_max,
                    color=CRIT_COLOR, alpha=0.15)
    ax4.set_xlabel(r"$\max_h \left|\partial \langle M_z^2\rangle / \partial h_0\right|$", fontsize=11)
    ax4.set_ylabel("Count", fontsize=12)
    ax4.set_title("MC Fluctuations of Peak Susceptibility", fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, ls="--", alpha=0.3)

    plt.suptitle(rf"Clean 1D Transverse-Field Ising — $L={L}$, NQS (ViT)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(run_dir, f"phase_transition_L={L}.pdf"), bbox_inches='tight')
    plt.clf()

    # ----------------------------------------------------------
    # FIGURE 3 : V-score et R-hat sur la grille d'entraînement
    # ----------------------------------------------------------
    fig3, (axV, axR) = plt.subplots(1, 2, figsize=(14, 5))

    axV.scatter(df["h0"], df["v_score"], color='royalblue', marker='o', s=60, label='V-score (MC)')
    axV.set_yscale('log')
    axV.axvline(x=1.0, color='red', linestyle='--', alpha=0.6, label=r"$h_c = 1$")
    axV.set_xlim(0, 3)
    axV.set_xlabel(r"Transverse Field $h_0$", fontsize=12)
    axV.set_ylabel(r"V-score $(Var(E)/E^2)$", fontsize=12)
    axV.set_title(f"Accuracy Landscape (L={L})", fontsize=13)
    axV.legend(fontsize=11)
    axV.grid(True, which='both', ls='--', alpha=0.4)

    axR.scatter(df["h0"], df["r_hat"], color='darkorange', marker='s', s=60, label=r"$\hat{R}$")
    axR.axhline(y=1.05, color='black', linestyle=':', linewidth=2, label='Seuil (1.05)')
    axR.axvline(x=1.0, color='red', linestyle='--', alpha=0.6, label=r"$h_c = 1$")
    axR.set_xlim(0, 3)
    axR.set_xlabel(r"Transverse Field $h_0$", fontsize=12)
    axR.set_ylabel(r"Gelman-Rubin $\hat{R}$", fontsize=12)
    axR.set_title(f"Convergence Diagnostics (L={L})", fontsize=13)
    axR.legend(fontsize=11)
    axR.grid(True, ls='--', alpha=0.4)

    plt.suptitle(f"Training Points Diagnostics — Clean Ising L={L}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"diagnostics_L={L}.pdf"), bbox_inches='tight')
    plt.clf()

    # ----------------------------------------------------------
    # RÉSUMÉ NUMÉRIQUE
    # ----------------------------------------------------------
    print("\n" + "="*55)
    print(f"  RÉSUMÉ — Clean Ising L={L}")
    print("="*55)
    print(f"  hc estimé (pic |dMz²/dh|) : {hc_est:.4f}  (exact : 1.0000)")
    print(f"  Erreur relative            : {abs(hc_est - 1.0)*100:.2f} %")
    print(f"  Mz² à h=0.5               : {mz2_mean[np.argmin(np.abs(h0_arr - 0.5))]:.4f}")
    print(f"  Mz² à h=1.0               : {mz2_mean[np.argmin(np.abs(h0_arr - 1.0))]:.4f}")
    print(f"  Mz² à h=2.0               : {mz2_mean[np.argmin(np.abs(h0_arr - 2.0))]:.4f}")
    print(f"  Max susceptibilité         : {np.max(dmz2_dh):.4f}  ± {micro_sem_max:.4f}")
    print("="*55)
    print("\n✅ Run terminé. Tous les graphiques et CSV ont été générés avec succès !")
    print(f"📁 Dossier de sortie : {run_dir}")