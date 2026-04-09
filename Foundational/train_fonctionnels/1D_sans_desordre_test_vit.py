import os
import re
import jax

# Initialisation du cluster JAX (multi-GPU via SLURM)
if 'SLURM_NTASKS' in os.environ and int(os.environ['SLURM_NTASKS']) > 1:
    node_list = os.environ.get('SLURM_STEP_NODELIST', os.environ.get('SLURM_JOB_NODELIST', ''))
    first_node = re.match(r'([a-z][-a-z]*)', node_list).group(1)
    jax.distributed.initialize(
        coordinator_address=f'{first_node}:1234',
        num_processes=int(os.environ['SLURM_NTASKS']),
        process_id=int(os.environ['SLURM_PROCID']),
    )
    print(f'JAX distributed: process {jax.process_index()}/{jax.process_count()}, devices={jax.device_count()}')
else:
    print(f'Mode local, {jax.device_count()} device(s)')

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)

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
import matplotlib.pyplot as plt
from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks import AbstractCallback
import netket_pro.distributed as nkpd
from netket.utils import struct
from flip_rules import GlobalFlipRule

# ==========================================
# 1. CONFIGURATIONS TESTÉES (index via argv[2])
# ==========================================
# Chaque config est un dictionnaire complet : architecture ViT + hyperparamètres MC/optim.
# Lancement : sbatch script.sh 48 <config_idx>

CONFIGS = [
    # Config 0 — référence, b=1, 7 h0
    {
        "name": "ref_d60_l4_h10_b2_s128_h7",
        "num_layers": 4, "d_model": 60, "heads": 10, "b": 2,
        "n_samples_per_config": 128, "n_h0": 7,
        "lr_init": 0.03, "lr_end": 0.005, "diag_shift": 2e-4,
    },
    # Config 1 — modèle petit, b=1, 7 h0
    {
        "name": "small_d32_l2_h4_b2_s128_h7",
        "num_layers": 2, "d_model": 32, "heads": 4, "b": 2,
        "n_samples_per_config": 128, "n_h0": 7,
        "lr_init": 0.03, "lr_end": 0.005, "diag_shift": 2e-4,
    },
    # Config 2 — grand modèle, b=1, 7 h0
    {
        "name": "large_d96_l6_h12_b2_s256_h7",
        "num_layers": 6, "d_model": 96, "heads": 12, "b": 2,
        "n_samples_per_config": 256, "n_h0": 7,
        "lr_init": 0.02, "lr_end": 0.003, "diag_shift": 2e-4,
    },
    # Config 3 — ref + plus de samples, b=1, 7 h0
    {
        "name": "ref_d60_l4_h10_b2_s512_h7",
        "num_layers": 4, "d_model": 60, "heads": 10, "b": 2,
        "n_samples_per_config": 512, "n_h0": 7,
        "lr_init": 0.03, "lr_end": 0.005, "diag_shift": 2e-4,
    },
    # Config 4 — ref + plus de samples, b=4, 7 h0 (comparaison patch size)
    {
        "name": "ref_d60_l4_h10_b4_s512_h7",
        "num_layers": 4, "d_model": 60, "heads": 10, "b": 4,
        "n_samples_per_config": 512, "n_h0": 7,
        "lr_init": 0.03, "lr_end": 0.005, "diag_shift": 2e-4,
    },
    # Config 5 — ref, b=1, 13 h0 (grille fine, référence complète)
    {
        "name": "ref_d60_l4_h10_b2_s128_h13",
        "num_layers": 4, "d_model": 60, "heads": 10, "b": 2,
        "n_samples_per_config": 128, "n_h0": 13,
        "lr_init": 0.03, "lr_end": 0.005, "diag_shift": 2e-4,
    },
]

# ==========================================
# 2. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
seed = 1
rng = np.random.default_rng(seed)

L          = int(sys.argv[1]) if len(sys.argv) > 1 else 48
cfg_idx    = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cfg        = CONFIGS[cfg_idx]

print(f"Config [{cfg_idx}] : {cfg['name']}")

# --- PARAMÈTRES PHYSIQUES ---
n_iter    = 600
J_val     = 1.0
h0_train_list       = list(np.round(np.linspace(0.7, 1.3, cfg["n_h0"]), 4))
total_configs_train = len(h0_train_list)

# --- PARAMÈTRES MONTE CARLO ---
n_samples_per_config = cfg["n_samples_per_config"]
samples_per_chain    = 2
chains_per_config    = n_samples_per_config // samples_per_chain
n_chains             = total_configs_train * chains_per_config
n_samples            = n_chains * samples_per_chain
prob_global_flip     = 0.05

# --- PARAMÈTRES D'OPTIMISATION ---
lr_init     = cfg["lr_init"]
lr_end      = cfg["lr_end"]
diag_shift  = cfg["diag_shift"]
logs_path   = os.path.join(foundational_dir, "logs")

# --- CHUNK SIZE ---
TARGET_CHUNK = 10
n_devices = jax.device_count()
n_samples_per_rank = n_samples // n_devices

if n_samples_per_rank <= TARGET_CHUNK:
    chunk_size = n_samples_per_rank
else:
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples_per_rank % i == 0:
            chunk_size = i
            break

chunk_size_bwd = 4

print(f"Configuration : {n_samples} samples, {n_devices} GPU(s), {n_samples_per_rank} samples/GPU, chunk_size={chunk_size}")

# --- PARAMÈTRES ViT ---
vit_params = {
    "num_layers": cfg["num_layers"],
    "d_model":    cfg["d_model"],
    "heads":      cfg["heads"],
    "b":          cfg["b"],
    "L_eff":      L // cfg["b"],
}

# ==========================================
# 3. DÉFINITION DU SYSTÈME
# ==========================================
hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10 * max(h0_train_list))

# Configurations homogènes (sans désordre) : une par valeur de h0
params_list = np.array([np.full(hi.size, h0) for h0 in h0_train_list])
print(f"Nombre de configs : {params_list.shape[0]} (une par h0, sans désordre)")

# Modèle
ma = ViTFNQS(
    num_layers=vit_params["num_layers"],
    d_model=vit_params["d_model"],
    heads=vit_params["heads"],
    b=vit_params["b"],
    L_eff=vit_params["L_eff"],
    n_coups=ps.size,
    complex=True,
    disorder=True,       # Le modèle reçoit h en entrée pour généraliser sur h0
    transl_invariant=False,
    two_dimensional=False,
)

# Sampler & État Variationnel
sa = nk.sampler.MetropolisSampler(
    hi,
    rule=GlobalFlipRule(prob_global_flip),
    n_chains=n_chains
)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed, chunk_size=chunk_size)

# Initialisation 50/50 UP/DOWN
sigma_orig = vs.sampler_state.σ
flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
half = flat_sigma.shape[0] // 2
flat_sigma = flat_sigma.at[:half, :L].set(1)
flat_sigma = flat_sigma.at[half:, :L].set(-1)
vs.sampler_state = vs.sampler_state.replace(σ=flat_sigma.reshape(sigma_orig.shape))

vs.parameter_array = params_list

# Opérateurs
def create_operator(params):
    assert params.shape == (hi.size,)
    ha_X  = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)

# ==========================================
# 4. CALLBACKS
# ==========================================

class ReplicaLogger(AbstractCallback):
    params_list: np.ndarray = struct.field(pytree_node=False)
    L: int                  = struct.field(pytree_node=False)
    eval_every: int         = struct.field(pytree_node=False)
    run_dir: str            = struct.field(pytree_node=False)
    iters: list             = struct.field(pytree_node=False, default_factory=list)
    energies: list          = struct.field(pytree_node=False, default_factory=list)
    variances: list         = struct.field(pytree_node=False, default_factory=list)

    def __init__(self, params_list, L, run_dir, eval_every=10):
        self.params_list = params_list
        self.L           = L
        self.run_dir     = run_dir
        self.eval_every  = eval_every
        self.iters       = []
        self.energies    = []
        self.variances   = []

    def on_step_end(self, step, log_data, driver):
        if step % self.eval_every != 0:
            return

        vs   = driver.state
        hi   = vs.hilbert
        sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=4)

        step_energies  = []
        step_variances = []

        for pars in self.params_list:
            _vs = vs.get_state(pars)
            mc_vs = nk.vqs.MCState(
                sampler=sa_eval, model=_vs.model, variables=_vs.variables,
                n_samples=256, chunk_size=16
            )
            mc_vs.reset()

            sigma_orig = np.array(mc_vs.sampler_state.σ)
            flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
            half = flat_sigma.shape[0] // 2
            flat_sigma[:half, :self.L] = 1
            flat_sigma[half:, :self.L] = -1
            mc_vs.sampler_state = mc_vs.sampler_state.replace(
                **{'σ': jnp.array(flat_sigma.reshape(sigma_orig.shape))}
            )

            H_op    = create_operator(pars)
            H_sq    = H_op @ H_op
            stats   = mc_vs.expect(H_op)
            stats_H2 = mc_vs.expect(H_sq)
            mean_val = float(np.real(stats.Mean))
            var_val  = float(np.real(stats_H2.Mean)) - mean_val**2
            step_energies.append(mean_val)
            step_variances.append(var_val)

        self.iters.append(step)
        self.energies.append(step_energies)
        self.variances.append(step_variances)

        if nkpd.is_master_process():
            np.save(os.path.join(self.run_dir, "replica_iters.npy"),     np.array(self.iters))
            np.save(os.path.join(self.run_dir, "replica_energies.npy"),  np.array(self.energies))
            np.save(os.path.join(self.run_dir, "replica_variances.npy"), np.array(self.variances))


class SaveState(AbstractCallback):
    _path: str       = struct.field(pytree_node=False)
    _prefix: str     = struct.field(pytree_node=False)
    _save_every: int = struct.field(pytree_node=False)

    def __init__(self, path: str, save_every: int, prefix: str = "state"):
        self._path       = path
        self._prefix     = prefix
        self._save_every = save_every
        if nkpd.is_master_process():
            os.makedirs(self._path, exist_ok=True)

    def on_step_end(self, step, log_data, driver):
        if step > 0 and step % self._save_every == 0:
            if nkpd.is_master_process() and not os.path.exists(self._path):
                os.makedirs(self._path, exist_ok=True)
            path = os.path.join(self._path, f"{self._prefix}_{driver.step_count}.nk")
            driver.state.save(path)

# ==========================================
# 5. LOGGING ET OPTIMISATION
# ==========================================
learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300)
optimizer     = optax.sgd(learning_rate)

def cg_solver(A, b):
    return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]

gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift,
                linear_solver_fn=cg_solver, chunk_size_bwd=chunk_size_bwd, use_ntk=True)

log  = nk.logging.JsonLog("log_data", save_params=False)

meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {"type": "Ising (sans désordre)", "J": J_val, "h0_train_list": h0_train_list},
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "config_name": cfg["name"],
    "config_idx": cfg_idx,
    "sampler": {"type": "MetropolisLocal+GlobalFlip", "n_chains": n_chains, "n_samples": n_samples,
                "n_samples_per_config": n_samples_per_config},
    "optimizer": {"type": "SGD+NG", "lr_init": lr_init, "lr_end": lr_end, "diag_shift": diag_shift},
    "n_iter": n_iter,
    "total_configs_train": total_configs_train,
    "seed": seed,
}

try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"

log = nk.logging.JsonLog(os.path.join(run_dir, "log_data.json"), save_params=False)
np.save(os.path.join(run_dir, "h0_configs.npy"), params_list)
print(f"Run dir : {run_dir}")

vs.chunk_size = chunk_size
start_time = time.time()

gs.run(
    n_iter,
    out=log,
    callback=[
        SaveState(run_dir, 50),
        ReplicaLogger(params_list, L, run_dir=run_dir, eval_every=10),
    ]
)

duration = time.time() - start_time
print(f"Temps total d'entraînement : {duration:.2f} secondes")

meta["execution_time_seconds"] = duration
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

# ==========================================
# 6. PLOTS ET ANALYSE FINALE
# ==========================================
if nkpd.is_master_process():
    print('Analyse finale et génération des graphiques...')
    train_results = {"v_score": [], "r_hat": []}

    for pars in tqdm(params_list):
        _vs   = vs.get_state(pars)
        vs_mc = nk.vqs.MCState(
            sampler=nk.sampler.MetropolisLocal(hi, n_chains=16),
            model=_vs.model, variables=_vs.variables,
            n_samples=1024, chunk_size=64
        )
        H_op     = create_operator(pars)
        H_sq     = H_op @ H_op
        stats    = vs_mc.expect(H_op)
        stats_H2 = vs_mc.expect(H_sq)
        mean_val = float(np.real(stats.Mean))
        var_val  = float(np.real(stats_H2.Mean)) - mean_val**2
        rhat_val = float(getattr(stats, 'R_hat', np.nan))
        train_results["v_score"].append(var_val / (mean_val**2 + 1e-12))
        train_results["r_hat"].append(rhat_val)

    df_train = pd.DataFrame({
        "h0":     h0_train_list,
        "v_score": train_results["v_score"],
        "r_hat":   train_results["r_hat"],
    })
    df_train.to_csv(os.path.join(run_dir, "train_results.csv"), index=False)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(h0_train_list)))

    # --- PLOT 1 : Scatter V-score (post-train) ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_train["h0"], df_train["v_score"], color='royalblue', marker='o', zorder=3, label='V-score')
    plt.plot(df_train["h0"], df_train["v_score"], color='royalblue', linestyle='--', linewidth=1)
    plt.yscale('log')
    plt.xlabel(r"Champ transverse $h_0$", fontsize=12)
    plt.ylabel(r"V-score $Var(E)/E^2$", fontsize=12)
    plt.title(f"V-score post-train — {cfg['name']} (L={L})", fontsize=13)
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"vscore_scatter_L={L}.pdf"))
    plt.clf()

    # --- PLOT 2 : Scatter R-hat (post-train) ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_train["h0"], df_train["r_hat"], color='royalblue', marker='o', zorder=3, label=r'$\hat{R}$')
    plt.plot(df_train["h0"], df_train["r_hat"], color='royalblue', linestyle='--', linewidth=1)
    plt.xlabel(r"Champ transverse $h_0$", fontsize=12)
    plt.ylabel(r"Gelman-Rubin $\hat{R}$", fontsize=12)
    plt.title(rf"$\hat{{R}}$ post-train — {cfg['name']} (L={L})", fontsize=13)
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"rhat_scatter_L={L}.pdf"))
    plt.clf()

    # --- PLOT 3 : Convergence de l'énergie avec zoom sur les 100 dernières itérations ---
    try:
        log_file = os.path.join(run_dir, "log_data.json.log")
        with open(log_file, 'r') as f:
            raw_log = json.load(f)
        energy_iters = np.array(raw_log["Energy"]["iters"])
        energy_mean  = np.array(raw_log["Energy"]["Mean"]["real"])

        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))
        ax_main.plot(energy_iters, energy_mean, color='royalblue', linewidth=1.2)
        ax_main.set_xlabel("Itération", fontsize=12)
        ax_main.set_ylabel(r"$\langle E \rangle$", fontsize=12)
        ax_main.set_title(f"Convergence de l'énergie — {cfg['name']} (L={L})", fontsize=12)
        ax_main.grid(True, ls='--', alpha=0.4)

        n_zoom = min(100, len(energy_iters))
        ax_zoom.plot(energy_iters[-n_zoom:], energy_mean[-n_zoom:], color='orangered', linewidth=1.2)
        ax_zoom.set_xlabel("Itération", fontsize=12)
        ax_zoom.set_ylabel(r"$\langle E \rangle$", fontsize=12)
        ax_zoom.set_title(f"Zoom — {n_zoom} dernières itérations", fontsize=12)
        ax_zoom.grid(True, ls='--', alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"energy_convergence_L={L}.pdf"))
        plt.clf()
    except (FileNotFoundError, KeyError) as e:
        print(f"⚠️ Plot de convergence d'énergie ignoré : {e}")

    # --- PLOT 4 : Courbes de V-score vs step pour chaque h0 ---
    try:
        hist_iters     = np.load(os.path.join(run_dir, "replica_iters.npy"))
        hist_energies  = np.load(os.path.join(run_dir, "replica_energies.npy"))
        hist_variances = np.load(os.path.join(run_dir, "replica_variances.npy"))

        plt.figure(figsize=(12, 6))
        for idx, h0 in enumerate(h0_train_list):
            e = hist_energies[:, idx]
            v = hist_variances[:, idx]
            plt.plot(hist_iters, v / (e**2 + 1e-12), color=colors[idx], linewidth=1.5, label=rf"$h_0={h0}$")

        plt.yscale('log')
        plt.xlabel("Optimization step", fontsize=12)
        plt.ylabel(r"V-score $Var(E)/E^2$", fontsize=12)
        plt.title(f"Convergence du V-score par $h_0$ — {cfg['name']} (L={L})", fontsize=12)
        plt.legend(fontsize=8, ncol=3, loc='upper right')
        plt.grid(True, which='both', ls='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"vscore_convergence_all_L={L}.pdf"))
        plt.clf()
    except FileNotFoundError:
        print("⚠️ Fichiers .npy introuvables. Le tracé de convergence du V-score a été ignoré.")

    print(f"✅ Run terminé [{cfg['name']}]. 4 graphiques générés avec succès !")
