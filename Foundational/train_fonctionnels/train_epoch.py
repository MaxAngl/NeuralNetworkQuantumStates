"""
Entrainement Epoch-based pour Foundational NQS (ViTFNQS, disorder=True).

Methode: a chaque epoch, on tire un sous-ensemble aleatoire de h0 parmi
la liste d'entrainement et on regenere le desordre (data augmentation).

Sauvegarde un vstate compatible avec plot_dMz_vs_sigma.py.

Usage:
    NETKET_EXPERIMENTAL_SHARDING=1 python Foundational/train_epoch.py
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax

import netket as nk
import netket_foundational as nkf

from netket_foundational._src.model.vit import ViTFNQS
from flip_rules import GlobalFlipRule

# ==========================================
# 1. PARAMETRES
# ==========================================

L = 16
seed = 1
J_val = 1.0

# Training h0 values
h0_train_list = [float(x) for x in np.round(np.linspace(0.1, 5.0, 20), 2)]
sigma_disorder = 0.1
n_replicas = 5  # replicas de desordre par h0

prob_global_flip = 0.05

# Epoch parameters
n_epochs = 12
n_iter_per_epoch = 100
n_h0_per_epoch = 10  # h0 tires a chaque epoch parmi les 20
total_configs_train = len(h0_train_list) * (n_replicas + 1)  # 120
n_reps_per_h0_epoch = total_configs_train // n_h0_per_epoch - 1  # 11 replicas + 1 homogene = 12 par h0

# Sampling
chains_per_replica = 4
samples_per_chain = 2
n_chains = total_configs_train * chains_per_replica
n_samples = n_chains * samples_per_chain

# Chunk size
TARGET_CHUNK = 64
if n_samples <= TARGET_CHUNK:
    chunk_size = n_samples
else:
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples % i == 0:
            chunk_size = i
            break
chunk_size_bwd = chunk_size

# Optimizer
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4

# ViT params
vit_params = {
    "num_layers": 2,
    "d_model": 16,
    "heads": 4,
    "b": 1,
    "L_eff": L,
}

# Logging
logs_path = os.path.join(foundational_dir, "logs")
save_every = n_iter_per_epoch  # sauvegarde a chaque fin d'epoch

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10 * max(h0_train_list))

Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1.0 / hi.size)


def create_operator(params):
    assert params.shape == (hi.size,)
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(
        nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
        for i in range(hi.size)
    )
    return -ha_X - J_val * ha_ZZ


ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)


def generate_multi_h0_disorder(h0_list, n_reps, system_size, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    all_configs = []
    for h_m in h0_list:
        random_configs = np.abs(rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size)))
        homogeneous_config = np.full((1, system_size), h_m)
        batch_configs = np.vstack([random_configs, homogeneous_config])
        all_configs.append(batch_configs)
    return np.vstack(all_configs)


def cg_solver(A, b):
    return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]


# ==========================================
# 3. LOGGING
# ==========================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(logs_path, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

n_iter_total = n_epochs * n_iter_per_epoch

meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {
        "type": "Ising Disorder",
        "J": J_val,
        "h0_train_list": h0_train_list,
        "sigma": sigma_disorder,
    },
    "training_method": "epoch",
    "epoch_config": {
        "n_epochs": n_epochs,
        "n_iter_per_epoch": n_iter_per_epoch,
        "n_h0_per_epoch": n_h0_per_epoch,
        "n_reps_per_h0_epoch": n_reps_per_h0_epoch,
        "h0_selection": "random subset without replacement",
        "disorder_refresh": "fresh abs(gaussian) + homogeneous each epoch",
    },
    "total_configs_train": total_configs_train,
    "n_iter_total": n_iter_total,
    "vit_config": vit_params,
    "sampler": {
        "type": "MetropolisSampler + GlobalFlipRule",
        "prob_global_flip": prob_global_flip,
        "chains_per_replica": chains_per_replica,
        "samples_per_chain": samples_per_chain,
        "n_chains": n_chains,
        "n_samples": n_samples,
    },
    "optimizer": {
        "type": "SGD + CG solver",
        "lr_init": lr_init,
        "lr_end": lr_end,
        "diag_shift": diag_shift,
    },
    "seed": seed,
}

with open(os.path.join(run_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"Run dir: {run_dir}")
print(f"Meta saved.")

# ==========================================
# 4. MODELE + SAMPLER + VSTATE
# ==========================================

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

vs = nkf.FoundationalQuantumState(
    sa, ma, ps,
    n_replicas=total_configs_train,
    n_samples=n_samples,
    seed=seed,
    chunk_size=chunk_size,
)

# Init avec premier sous-ensemble
rng_ep = np.random.default_rng(seed + 100)
h0_indices = rng_ep.choice(len(h0_train_list), size=n_h0_per_epoch, replace=False)
h0_subset = [h0_train_list[i] for i in h0_indices]
params_init = generate_multi_h0_disorder(
    h0_subset, n_reps_per_h0_epoch, hi.size, sigma_disorder, rng_ep
)
vs.parameter_array = params_init

print(f"Configs shape: {params_init.shape}")
print(f"Nombre total configs: {total_configs_train}")

# Sauvegarde des configs initiales
np.save(os.path.join(run_dir, "disorder_configs.npy"), params_init)

# ==========================================
# 5. OPTIMISEUR + DRIVER
# ==========================================

lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=n_iter_total)
optimizer = optax.sgd(lr)

gs = nkf.VMC_NG(
    ha_p, optimizer, variational_state=vs,
    diag_shift=diag_shift,
    linear_solver_fn=cg_solver,
    use_ntk=True,
    chunk_size_bwd=chunk_size_bwd,
)

log = nk.logging.RuntimeLog()

# ==========================================
# 6. ENTRAINEMENT PAR EPOCHS
# ==========================================

print(f"\n{'='*60}")
print(f"  EPOCH TRAINING")
print(f"  {n_h0_per_epoch}/{len(h0_train_list)} h0 par epoch, {n_reps_per_h0_epoch}+1 configs par h0")
print(f"  {n_epochs} epochs x {n_iter_per_epoch} iters = {n_iter_total} total")
print(f"  Sampler: MetropolisSampler + GlobalFlipRule(p={prob_global_flip})")
print(f"{'='*60}\n")

start_time = time.time()

for epoch in range(n_epochs):
    # Nouveau sous-ensemble de h0 + nouveau desordre
    h0_indices = rng_ep.choice(len(h0_train_list), size=n_h0_per_epoch, replace=False)
    h0_subset = [h0_train_list[i] for i in h0_indices]
    params_epoch = generate_multi_h0_disorder(
        h0_subset, n_reps_per_h0_epoch, hi.size, sigma_disorder, rng_ep
    )
    vs.parameter_array = params_epoch

    gs.run(n_iter_per_epoch, out=log, obs={"ham": ha_p, "mz": mz_p})

    # Sauvegarde du state a la fin de chaque epoch
    iter_count = (epoch + 1) * n_iter_per_epoch
    save_path = os.path.join(run_dir, f"state_{iter_count}.nk")
    vs.save(save_path)

    print(f"  Epoch {epoch + 1:2d}/{n_epochs} | "
          f"h0={[f'{h:.2f}' for h in sorted(h0_subset)]} | "
          f"state saved: state_{iter_count}.nk")

duration = time.time() - start_time
print(f"\nEntrainement termine en {duration:.1f}s")

# Sauvegarde finale
final_path = os.path.join(run_dir, f"state_{n_iter_total}.nk")
vs.save(final_path)
print(f"State final sauvegarde: {final_path}")

# Mise a jour meta avec temps d'execution
meta["execution_time_seconds"] = duration
with open(os.path.join(run_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print(f"  DONE - Run: {run_dir}")
print(f"  Pour tracer: python Foundational/plot_dMz_vs_sigma.py \\")
print(f"    --run_dir {run_dir} --checkpoint state_{n_iter_total}.nk")
print(f"{'='*60}")
