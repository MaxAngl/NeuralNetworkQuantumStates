"""
Comparison: Standard training vs Epoch-based training (v3)
Both use DualEmbed ansatz. MC-based evaluation.

Changes from v2:
- h0 includes 0 and negative values: [-2, ..., 0, ..., 4]
- 70 iters per epoch
- Same total iters for both (420)

Saves to: graphs/epoch_comparison_v3/
"""

import os
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

from ansatz_dual_embed import ViTFNQS_DualEmbed


# ==========================================
# CONFIGURATION
# ==========================================

L = 16
seed = 42
J_val = 1.0 / np.e

# Training h0 values — 25 values including negatives and zero
h0_train_list = list(np.round(np.linspace(-2.0, 4.0, 25), 2))
sigma_train = 0.1

# Total budget: 420 iters for both
n_iter_total = 420

# Standard training
n_replicas_per_h0_std = 2          # 25 × 2 = 50 replicas
n_replicas_total = len(h0_train_list) * n_replicas_per_h0_std  # 50

# Epoch training
n_h0_per_epoch = 10                # 10 of 25 h0 per mini-epoch
n_reps_per_h0_epoch = n_replicas_total // n_h0_per_epoch  # 5
n_iter_per_epoch = 70
n_epochs = n_iter_total // n_iter_per_epoch  # 6

# Sampling (training)
chains_per_replica = 4
samples_per_chain = 2
n_chains = n_replicas_total * chains_per_replica
n_samples = n_chains * samples_per_chain

# Optimizer
lr_init = 0.03
lr_end = 0.005
diag_shift = 1e-4

# ViT
vit_params = {
    "num_layers": 1,
    "d_model": 16,
    "heads": 2,
    "b": 1,
}

# Test grid — includes negative h0
h0_test_list = np.linspace(-2.5, 4.5, 16)
sigma_test_list = np.linspace(0.01, 0.3, 16)
N_test_per_point = 5

# MC evaluation config
eval_n_chains = 16
eval_n_samples = 512

# Output
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
out_dir = os.path.join(project_root, "graphs", "epoch_comparison_v3")
os.makedirs(out_dir, exist_ok=True)


# ==========================================
# META
# ==========================================

meta = {
    "L": L,
    "J": J_val,
    "h0_train_list": h0_train_list,
    "sigma_train": sigma_train,
    "model": "ViTFNQS_DualEmbed",
    "vit_params": vit_params,
    "standard": {
        "n_replicas_per_h0": n_replicas_per_h0_std,
        "n_replicas_total": n_replicas_total,
        "n_iter": n_iter_total,
        "disorder": "fixed throughout training",
    },
    "epoch": {
        "n_h0_per_epoch": n_h0_per_epoch,
        "n_reps_per_h0": n_reps_per_h0_epoch,
        "n_iter_per_epoch": n_iter_per_epoch,
        "n_epochs": n_epochs,
        "n_iter_total": n_epochs * n_iter_per_epoch,
        "disorder": "fresh each epoch (data augmentation)",
        "h0_selection": "random subset without replacement",
    },
    "optimizer": {"type": "SGD", "lr_init": lr_init, "lr_end": lr_end, "diag_shift": diag_shift},
    "sampling_train": {"chains_per_replica": chains_per_replica, "samples_per_chain": samples_per_chain},
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

# ParameterSpace must cover negative h values
h_abs_max = max(abs(min(h0_train_list)), abs(max(h0_train_list)))
ps = nkf.ParameterSpace(N=hi.size, min=-10 * h_abs_max, max=10 * h_abs_max)

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


def make_model():
    return ViTFNQS_DualEmbed(
        num_layers=vit_params["num_layers"],
        d_model=vit_params["d_model"],
        heads=vit_params["heads"],
        b=vit_params["b"],
        L_eff=L,
        n_coups=ps.size,
        complex=True,
        transl_invariant=False,
        two_dimensional=False,
    )


def generate_disorder(h0_list, n_reps, sigma, rng):
    configs = []
    for h in h0_list:
        configs.append(rng.normal(h, sigma, (n_reps, hi.size)))
    return np.vstack(configs)


# ==========================================
# STANDARD TRAINING
# ==========================================

def train_standard():
    print("\n" + "=" * 60)
    print("  STANDARD TRAINING (all h0 at once, fixed disorder)")
    print(f"  {len(h0_train_list)} h0 x {n_replicas_per_h0_std} replicas = {n_replicas_total}")
    print(f"  {n_iter_total} iterations")
    print("=" * 60)

    model = make_model()
    rng = np.random.default_rng(seed)
    params_train = generate_disorder(h0_train_list, n_replicas_per_h0_std, sigma_train, rng)
    print(f"  Training configs: {params_train.shape}")

    sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vs = nkf.FoundationalQuantumState(
        sa, model, ps,
        n_replicas=n_replicas_total,
        n_samples=n_samples,
        seed=seed,
    )
    vs.parameter_array = params_train

    lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=n_iter_total)
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift)

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
    print("  EPOCH TRAINING (mini-batch h0 + fresh disorder each epoch)")
    print(f"  {n_h0_per_epoch} of {len(h0_train_list)} h0 per epoch, {n_reps_per_h0_epoch} reps each")
    print(f"  {n_epochs} epochs x {n_iter_per_epoch} iters = {n_epochs * n_iter_per_epoch} total")
    print("=" * 60)

    model = make_model()
    rng_ep = np.random.default_rng(seed + 100)

    # Initialize with first epoch params
    h0_indices = rng_ep.choice(len(h0_train_list), size=n_h0_per_epoch, replace=False)
    h0_subset = [h0_train_list[i] for i in h0_indices]
    params_init = generate_disorder(h0_subset, n_reps_per_h0_epoch, sigma_train, rng_ep)

    sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vs = nkf.FoundationalQuantumState(
        sa, model, ps,
        n_replicas=n_replicas_total,
        n_samples=n_samples,
        seed=seed,
    )
    vs.parameter_array = params_init

    lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=n_iter_total)
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift)

    log = nk.logging.RuntimeLog()
    start = time.time()

    for epoch in range(n_epochs):
        h0_indices = rng_ep.choice(len(h0_train_list), size=n_h0_per_epoch, replace=False)
        h0_subset = [h0_train_list[i] for i in h0_indices]

        params_epoch = generate_disorder(h0_subset, n_reps_per_h0_epoch, sigma_train, rng_ep)
        vs.parameter_array = params_epoch

        gs.run(n_iter_per_epoch, out=log, obs={"ham": ha_p})
        print(f"  Epoch {epoch+1:2d}/{n_epochs} | h0={[f'{h:.2f}' for h in h0_subset]}")

    duration = time.time() - start
    print(f"  Done in {duration:.1f}s")

    return vs, duration


# ==========================================
# MC-BASED EVALUATION ON 2D GRID
# ==========================================

def evaluate_grid_mc(vs, label):
    print(f"\n  Evaluating {label} on {len(h0_test_list)}x{len(sigma_test_list)} grid (MC, {eval_n_samples} samples)...")

    rng_test = np.random.default_rng(seed + 2000)
    total = len(h0_test_list) * len(sigma_test_list) * N_test_per_point

    grid_vscores = np.zeros((len(h0_test_list), len(sigma_test_list), N_test_per_point))

    pbar = tqdm(total=total, desc=f"  {label}", leave=True)

    for i, h0_t in enumerate(h0_test_list):
        for j, sig_t in enumerate(sigma_test_list):
            test_params = rng_test.normal(h0_t, sig_t, (N_test_per_point, hi.size))

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
print("  Epoch Training Comparison v3 (with h0<=0, MC eval)")
print(f"  L={L}, {len(h0_train_list)} h0_train values")
print(f"  h0_train range: [{min(h0_train_list)}, {max(h0_train_list)}]")
print(f"  sigma_train={sigma_train}, J={J_val:.4f}")
print(f"  Total iters: {n_iter_total} (both methods)")
print(f"  Epoch: {n_epochs} x {n_iter_per_epoch} iters")
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
# PLOTTING
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
ax.axhline(0, color="white", alpha=0.5, linewidth=1, linestyle="-")
ax.axvline(sigma_train, color="white", alpha=0.5, linewidth=1, linestyle="--")
ax.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax.set_ylabel(r"$h_0^{test}$", fontsize=12)
ax.set_title("Standard training\n(all h0 at once, fixed disorder)", fontsize=12)
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
ax2.axhline(0, color="white", alpha=0.5, linewidth=1, linestyle="-")
ax2.axvline(sigma_train, color="white", alpha=0.5, linewidth=1, linestyle="--")
ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel(r"$h_0^{test}$", fontsize=12)
ax2.set_title("Epoch training\n(mini-batch h0 + fresh disorder)", fontsize=12)
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
ax3.axhline(0, color="black", alpha=0.5, linewidth=1, linestyle="-")
ax3.axvline(sigma_train, color="black", alpha=0.3, linewidth=1, linestyle="--")
ax3.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax3.set_ylabel(r"$h_0^{test}$", fontsize=12)
ax3.set_title("Ratio (Epoch / Standard)\nBlue < 1 = Epoch better", fontsize=12)
plt.colorbar(im3, ax=ax3, label="V-score ratio")

# Global title
fig.suptitle(
    f"Epoch vs Standard — DualEmbed ViTFNQS (MC eval)\n"
    f"L={L}, $\\sigma_{{train}}$={sigma_train}, {n_iter_total} iters, "
    f"{len(h0_train_list)} h0 in [{min(h0_train_list)}, {max(h0_train_list)}]\n"
    f"Epoch: {n_epochs}x{n_iter_per_epoch} iters, {n_h0_per_epoch}/{len(h0_train_list)} h0/epoch | "
    f"White line = h0=0, Red dashes = train h0",
    fontsize=12, y=1.02,
)

plt.tight_layout()

pdf_path = os.path.join(out_dir, "epoch_comparison_v3.pdf")
png_path = os.path.join(out_dir, "epoch_comparison_v3.png")
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
print(f"  Grid points Epoch wins: {n_epoch_better}/{n_total}")
print(f"{'=' * 60}")
print("Done.")
