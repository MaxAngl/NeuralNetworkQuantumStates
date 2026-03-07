"""
Comparison: Standard training vs Epoch-based training
Same sampling/model as 1D_avec_desordre_pluri_h0.py (MetropolisSampler + GlobalFlipRule, ViTFNQS disorder=True)
Only difference: epoch-based regenere le desordre a chaque epoch (data augmentation).

8 epochs x 50 iters = 400 total for both methods.
"""

import os
import sys
# Ajouter le repertoire racine du projet au chemin Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax

import netket as nk
import netket_foundational as nkf

from netket_foundational._src.model.vit import ViTFNQS
from flip_rules import GlobalFlipRule

# ==========================================
# CONFIGURATION
# ==========================================

L = 16
seed = 1
J_val = 1.0

# Training h0 values — 20 valeurs de 0.1 a 5.0
h0_train_list = list(np.round(np.linspace(0.1, 5.0, 20), 2))
sigma_disorder = 0.1
n_replicas = 5  # Nombre de realisations de desordre par h0 (pour garder ~120 configs)
prob_global_flip = 0.01

# Total budget
n_iter_total = 450  # 6 x 75

# Standard training
total_configs_train = len(h0_train_list) * (n_replicas + 1)  # 110

# Epoch training
n_epochs = 6
n_iter_per_epoch = 75
n_h0_per_epoch = 10  # 10 h0 parmi 20 a chaque epoch
# Nombre de replicas par h0 en epoch pour garder total_configs_train constant
n_reps_per_h0_epoch = total_configs_train // n_h0_per_epoch - 1  # -1 pour la config homogene
# => 120 // 10 - 1 = 11 replicas + 1 homogene = 12 par h0, 10 * 12 = 120

# Sampling (identique a 1D_avec_desordre_pluri_h0.py)
chains_per_replica = 4
samples_per_chain = 2
n_chains = total_configs_train * chains_per_replica
n_samples = n_chains * samples_per_chain

# Chunk size (meme methode)
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

# ViT params (same as reference)
vit_params = {
    "num_layers": 2,
    "d_model": 16,
    "heads": 4,
    "b": 1,
    "L_eff": L,
}

# Test grid for evaluation
h0_test_list = np.linspace(0.1, 6.0, 15)
sigma_test_list = np.linspace(0.01, 0.3, 16)
N_test_per_point = 5

# MC evaluation config
eval_n_chains = 16
eval_n_samples = 1024
eval_chunk_size = 64

# Output
out_dir = os.path.join(project_root, "graphs", "epoch_comparison_desordre_pluri_v3")
os.makedirs(out_dir, exist_ok=True)

print(f"Configuration : {n_samples} samples total.")
print(f"Chunk size auto-calcule : {chunk_size} (Diviseur optimal <= {TARGET_CHUNK})")

# ==========================================
# META
# ==========================================

meta = {
    "L": L,
    "J": J_val,
    "h0_train_list": h0_train_list,
    "sigma_disorder": sigma_disorder,
    "model": "ViTFNQS",
    "vit_params": vit_params,
    "standard": {
        "n_replicas_per_h0": n_replicas,
        "total_configs_train": total_configs_train,
        "n_iter": n_iter_total,
        "disorder": "fixed throughout training, abs(gaussian) + homogeneous",
    },
    "epoch": {
        "n_h0_per_epoch": n_h0_per_epoch,
        "n_reps_per_h0_epoch": n_reps_per_h0_epoch,
        "total_configs_train": total_configs_train,
        "n_iter_per_epoch": n_iter_per_epoch,
        "n_epochs": n_epochs,
        "n_iter_total": n_epochs * n_iter_per_epoch,
        "disorder": "random h0 subset + fresh abs(gaussian) + homogeneous each epoch",
        "h0_selection": "random subset without replacement",
    },
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
    "evaluation": {
        "method": "MCState (Monte Carlo sampling)",
        "n_chains": eval_n_chains,
        "n_samples": eval_n_samples,
    },
    "test_grid": {
        "h0_range": [float(h0_test_list[0]), float(h0_test_list[-1])],
        "n_h0": len(h0_test_list),
        "sigma_range": [float(sigma_test_list[0]), float(sigma_test_list[-1])],
        "n_sigma": len(sigma_test_list),
        "N_test_per_point": N_test_per_point,
    },
    "seed": seed,
}

with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"Meta saved to {out_dir}/meta.json")

# ==========================================
# SYSTEM SETUP
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
    """
    Genere les configs de desordre: n_reps gaussiennes (abs) + 1 homogene par h0.
    Identique a 1D_avec_desordre_pluri_h0.py
    """
    if rng is None:
        rng = np.random.default_rng()

    all_configs = []
    for h_m in h0_list:
        # Replicas desordonnees (gaussienne + valeur absolue = repliement)
        raw_configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        random_configs = np.abs(raw_configs)

        # Configuration homogene
        homogeneous_config = np.full((1, system_size), h_m)

        # Empiler: n_reps desordre + 1 homogene
        batch_configs = np.vstack([random_configs, homogeneous_config])
        all_configs.append(batch_configs)

    return np.vstack(all_configs)


def make_model():
    return ViTFNQS(
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


def make_sampler():
    return nk.sampler.MetropolisSampler(
        hi,
        rule=GlobalFlipRule(prob_global_flip),
        n_chains=n_chains,
    )


def cg_solver(A, b):
    return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]


# ==========================================
# STANDARD TRAINING
# ==========================================

def train_standard():
    print("\n" + "=" * 60)
    print("  STANDARD TRAINING (all h0 at once, fixed disorder)")
    print(f"  {len(h0_train_list)} h0 x ({n_replicas}+1) = {total_configs_train} configs")
    print(f"  {n_iter_total} iterations")
    print("=" * 60)

    model = make_model()
    rng_std = np.random.default_rng(seed)
    params_train = generate_multi_h0_disorder(
        h0_train_list, n_replicas, hi.size, sigma_disorder, rng_std
    )
    print(f"  Training configs: {params_train.shape}")

    sa = make_sampler()
    vs = nkf.FoundationalQuantumState(
        sa, model, ps,
        n_replicas=total_configs_train,
        n_samples=n_samples,
        seed=seed,
        chunk_size=chunk_size,
    )
    vs.parameter_array = params_train

    lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=n_iter_total)
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(
        ha_p, optimizer, variational_state=vs,
        diag_shift=diag_shift,
        linear_solver_fn=cg_solver,
        chunk_size_bwd=chunk_size_bwd,
    )

    log = nk.logging.RuntimeLog()
    start = time.time()
    gs.run(n_iter_total, out=log, obs={"ham": ha_p})
    duration = time.time() - start
    print(f"  Done in {duration:.1f}s")

    return vs, duration


# ==========================================
# EPOCH TRAINING
# ==========================================

def train_epoch():
    print("\n" + "=" * 60)
    print("  EPOCH TRAINING (random h0 subset + fresh disorder each epoch)")
    print(f"  {n_h0_per_epoch}/{len(h0_train_list)} h0 per epoch, {n_reps_per_h0_epoch}+1 configs per h0")
    print(f"  {n_epochs} epochs x {n_iter_per_epoch} iters = {n_epochs * n_iter_per_epoch} total")
    print("=" * 60)

    model = make_model()
    rng_ep = np.random.default_rng(seed + 100)

    # Initialisation avec premier sous-ensemble
    h0_indices = rng_ep.choice(len(h0_train_list), size=n_h0_per_epoch, replace=False)
    h0_subset = [h0_train_list[i] for i in h0_indices]
    params_init = generate_multi_h0_disorder(
        h0_subset, n_reps_per_h0_epoch, hi.size, sigma_disorder, rng_ep
    )

    sa = make_sampler()
    vs = nkf.FoundationalQuantumState(
        sa, model, ps,
        n_replicas=total_configs_train,
        n_samples=n_samples,
        seed=seed,
        chunk_size=chunk_size,
    )
    vs.parameter_array = params_init

    lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=n_iter_total)
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(
        ha_p, optimizer, variational_state=vs,
        diag_shift=diag_shift,
        linear_solver_fn=cg_solver,
        chunk_size_bwd=chunk_size_bwd,
    )

    log = nk.logging.RuntimeLog()
    start = time.time()

    for epoch in range(n_epochs):
        # Nouveau sous-ensemble de h0 + nouveau desordre a chaque epoch
        h0_indices = rng_ep.choice(len(h0_train_list), size=n_h0_per_epoch, replace=False)
        h0_subset = [h0_train_list[i] for i in h0_indices]
        params_epoch = generate_multi_h0_disorder(
            h0_subset, n_reps_per_h0_epoch, hi.size, sigma_disorder, rng_ep
        )
        vs.parameter_array = params_epoch

        gs.run(n_iter_per_epoch, out=log, obs={"ham": ha_p})
        print(f"  Epoch {epoch + 1:2d}/{n_epochs} | h0={[f'{h:.2f}' for h in sorted(h0_subset)]}")

    duration = time.time() - start
    print(f"  Done in {duration:.1f}s")

    return vs, duration


# ==========================================
# MC-BASED EVALUATION ON 2D GRID
# ==========================================

def evaluate_grid_mc(vs, label):
    print(f"\n  Evaluating {label} on {len(h0_test_list)}x{len(sigma_test_list)} grid "
          f"(MC, {eval_n_samples} samples)...")

    rng_test = np.random.default_rng(seed + 2000)
    total = len(h0_test_list) * len(sigma_test_list) * N_test_per_point

    grid_vscores = np.zeros((len(h0_test_list), len(sigma_test_list), N_test_per_point))

    pbar = tqdm(total=total, desc=f"  {label}", leave=True)

    for i, h0_t in enumerate(h0_test_list):
        for j, sig_t in enumerate(sigma_test_list):
            # Desordre de test (abs pour rester coherent avec le training)
            test_params = np.abs(rng_test.normal(h0_t, sig_t, (N_test_per_point, hi.size)))

            for k, pars in enumerate(test_params):
                _vs = vs.get_state(pars)
                _ha = create_operator(pars)

                sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=eval_n_chains)
                mc_state = nk.vqs.MCState(
                    sampler=sa_eval,
                    model=_vs.model,
                    variables=_vs.variables,
                    n_samples=eval_n_samples,
                    seed=seed + i * 1000 + j * 100 + k,
                    chunk_size=eval_chunk_size,
                )
                _e = mc_state.expect(_ha)
                grid_vscores[i, j, k] = float(_e.variance / (_e.Mean.real ** 2 + 1e-12))
                pbar.update(1)

    pbar.close()
    return grid_vscores


# ==========================================
# MAIN
# ==========================================

print("=" * 60)
print("  Epoch vs Standard -- ViTFNQS + GlobalFlipRule + Disorder")
print(f"  L={L}, J={J_val}, sigma={sigma_disorder}")
print(f"  {len(h0_train_list)} h0 values, {n_replicas} replicas + 1 homogeneous")
print(f"  Total iters: {n_iter_total} (both methods)")
print(f"  Epoch: {n_epochs} x {n_iter_per_epoch} iters")
print(f"  Sampler: MetropolisSampler + GlobalFlipRule(p={prob_global_flip})")
print(f"  Eval: MCState with {eval_n_samples} samples")
print("=" * 60)

# Train both
vs_std, t_std = train_standard()
vs_ep, t_ep = train_epoch()

# Evaluate both on same grid
grid_std = evaluate_grid_mc(vs_std, "Standard")
grid_ep = evaluate_grid_mc(vs_ep, "Epoch")

# Compute medians
median_std = np.median(grid_std, axis=2)
median_ep = np.median(grid_ep, axis=2)

# Save raw results
np.savez(
    os.path.join(out_dir, "raw_results.npz"),
    h0_test=h0_test_list,
    sigma_test=sigma_test_list,
    grid_std=grid_std,
    grid_ep=grid_ep,
    median_std=median_std,
    median_ep=median_ep,
)
print(f"\nRaw results saved to {out_dir}/raw_results.npz")


# ==========================================
# PLOTTING (3-panel colormap)
# ==========================================

vmin = min(median_std[median_std > 0].min(), median_ep[median_ep > 0].min())
vmax = max(median_std.max(), median_ep.max())

sigma_grid, h0_grid = np.meshgrid(sigma_test_list, h0_test_list)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# --- Panel 1: Standard ---
ax = axes[0]
im1 = ax.pcolormesh(
    sigma_grid, h0_grid, median_std,
    norm=LogNorm(vmin=vmin, vmax=vmax),
    cmap="viridis", shading="auto",
)
for h in h0_train_list:
    ax.axhline(h, color="red", alpha=0.25, linewidth=0.5, linestyle="--")
ax.axvline(sigma_disorder, color="white", alpha=0.5, linewidth=1, linestyle="--")
ax.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax.set_ylabel(r"$h_0^{test}$", fontsize=12)
ax.set_title("Standard training\n(fixed disorder, 400 iters)", fontsize=12)
plt.colorbar(im1, ax=ax, label="Median V-score")

# --- Panel 2: Epoch ---
ax2 = axes[1]
im2 = ax2.pcolormesh(
    sigma_grid, h0_grid, median_ep,
    norm=LogNorm(vmin=vmin, vmax=vmax),
    cmap="viridis", shading="auto",
)
for h in h0_train_list:
    ax2.axhline(h, color="red", alpha=0.25, linewidth=0.5, linestyle="--")
ax2.axvline(sigma_disorder, color="white", alpha=0.5, linewidth=1, linestyle="--")
ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel(r"$h_0^{test}$", fontsize=12)
ax2.set_title(f"Epoch training\n({n_epochs} epochs x {n_iter_per_epoch} iters, {n_h0_per_epoch}/{len(h0_train_list)} h0 + fresh disorder)", fontsize=12)
plt.colorbar(im2, ax=ax2, label="Median V-score")

# --- Panel 3: Ratio ---
ax3 = axes[2]
ratio = median_ep / (median_std + 1e-15)
ratio_clipped = np.clip(ratio, 0.01, 100)
im3 = ax3.pcolormesh(
    sigma_grid, h0_grid, ratio_clipped,
    norm=TwoSlopeNorm(vcenter=1.0, vmin=min(0.1, ratio_clipped.min()), vmax=max(10.0, ratio_clipped.max())),
    cmap="RdBu_r", shading="auto",
)
for h in h0_train_list:
    ax3.axhline(h, color="gray", alpha=0.2, linewidth=0.5, linestyle="--")
ax3.axvline(sigma_disorder, color="black", alpha=0.3, linewidth=1, linestyle="--")
ax3.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax3.set_ylabel(r"$h_0^{test}$", fontsize=12)
ax3.set_title("Ratio (Epoch / Standard)\nBlue < 1 = Epoch better", fontsize=12)
plt.colorbar(im3, ax=ax3, label="V-score ratio")

# Global title
fig.suptitle(
    f"Epoch vs Standard -- ViTFNQS + GlobalFlipRule + Disorder\n"
    f"L={L}, J={J_val}, $\\sigma_{{train}}$={sigma_disorder}, {n_iter_total} iters total\n"
    f"Epoch: {n_epochs}x{n_iter_per_epoch} iters ({n_h0_per_epoch}/{len(h0_train_list)} h0 + fresh disorder each epoch) | "
    f"Red dashes = train h0",
    fontsize=12, y=1.02,
)

plt.tight_layout()

pdf_path = os.path.join(out_dir, "epoch_comparison_desordre_pluri_v3.pdf")
png_path = os.path.join(out_dir, "epoch_comparison_desordre_pluri_v3.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:")
print(f"  {pdf_path}")
print(f"  {png_path}")

# Summary
n_epoch_better = np.sum(ratio < 1)
n_total = ratio.size
print(f"\n{'=' * 60}")
print(f"  SUMMARY")
print(f"{'=' * 60}")
print(f"  Training time Standard: {t_std:.1f}s")
print(f"  Training time Epoch   : {t_ep:.1f}s")
print(f"  Median ratio (Ep/Std) : {np.median(ratio):.3f}")
print(f"  Mean ratio   (Ep/Std) : {np.mean(ratio):.3f}")
print(f"  Grid points Epoch wins: {n_epoch_better}/{n_total}")
print(f"{'=' * 60}")
print("Done.")
