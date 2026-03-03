"""
Comparison: Original ViTFNQS (single embed) vs DualEmbed ViTFNQS
L=16, 1D disordered Ising, single h0=1.0, varying sigma

Saves results and plots to ../graphs/
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax

import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS

from ansatz_dual_embed import ViTFNQS_DualEmbed


# ==========================================
# CONFIGURATION
# ==========================================

L = 16
seed = 42
h0 = 1.0
J_val = 1.0 / np.e
n_replicas = 10

sigma_list = [0.01, 0.05, 0.1, 0.2, 0.5]

# ViT hyperparameters (identical for both models)
vit_params = {
    "num_layers": 1,
    "d_model": 16,
    "heads": 2,
    "b": 1,
}

# Training hyperparameters
n_iter = 200
lr_init = 0.03
lr_end = 0.005
diag_shift = 1e-4
chains_per_replica = 4
samples_per_chain = 2

# Output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
graphs_dir = os.path.join(project_root, "graphs")
os.makedirs(graphs_dir, exist_ok=True)


# ==========================================
# SYSTEM SETUP
# ==========================================

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10 * h0)

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


def generate_disorder(n_reps, system_size, h0_val, sigma, rng):
    return rng.normal(loc=h0_val, scale=sigma, size=(n_reps, system_size))


# ==========================================
# TRAIN AND EVALUATE
# ==========================================

def train_and_evaluate(model, sigma, rng_seed, model_name=""):
    """Train model on n_replicas disorder configs and return V-scores."""
    rng = np.random.default_rng(rng_seed)
    params_list = generate_disorder(n_replicas, hi.size, h0, sigma, rng)

    n_chains = n_replicas * chains_per_replica
    n_samples = n_chains * samples_per_chain

    sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vs = nkf.FoundationalQuantumState(
        sa, model, ps,
        n_replicas=n_replicas,
        n_samples=n_samples,
        seed=seed,
    )
    vs.parameter_array = params_list

    # Optimizer with linear schedule
    lr = optax.linear_schedule(
        init_value=lr_init, end_value=lr_end, transition_steps=300
    )
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift)

    # Train
    log = nk.logging.RuntimeLog()
    start = time.time()
    gs.run(n_iter, out=log, obs={"ham": ha_p, "mz": mz_p})
    duration = time.time() - start
    print(f"    {model_name} training: {duration:.1f}s")

    # Evaluate V-scores using FullSumState (exact, no MC noise)
    v_scores = []
    for r in tqdm(range(n_replicas), desc=f"    {model_name} eval", leave=False):
        pars = params_list[r]
        _vs = vs.get_state(pars)
        _ha = create_operator(pars)

        vs_fs = nk.vqs.FullSumState(
            hilbert=hi, model=_vs.model, variables=_vs.variables
        )
        _e = vs_fs.expect(_ha)
        v_score = float(_e.variance / (_e.Mean.real ** 2 + 1e-12))
        v_scores.append(v_score)

    return np.array(v_scores), duration


# ==========================================
# MAIN COMPARISON LOOP
# ==========================================

print("=" * 60)
print("  Embedding Comparison: Original vs DualEmbed")
print(f"  L={L}, h0={h0}, J={J_val:.4f}, n_replicas={n_replicas}")
print(f"  ViT: {vit_params}")
print(f"  Sigmas: {sigma_list}")
print("=" * 60)

all_results = []

for sigma in sigma_list:
    print(f"\n{'─' * 50}")
    print(f"  sigma = {sigma}")
    print(f"{'─' * 50}")

    # --- Original model (single embed, disorder=False) ---
    model_orig = ViTFNQS(
        num_layers=vit_params["num_layers"],
        d_model=vit_params["d_model"],
        heads=vit_params["heads"],
        b=vit_params["b"],
        L_eff=L,
        n_coups=ps.size,
        complex=True,
        disorder=False,
        transl_invariant=False,
        two_dimensional=False,
    )
    v_orig, t_orig = train_and_evaluate(model_orig, sigma, seed, "Original")

    # --- DualEmbed model ---
    model_dual = ViTFNQS_DualEmbed(
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
    v_dual, t_dual = train_and_evaluate(model_dual, sigma, seed, "DualEmbed")

    # Store results
    for r in range(n_replicas):
        all_results.append({
            "sigma": sigma,
            "replica": r,
            "v_score_original": v_orig[r],
            "v_score_dual": v_dual[r],
            "time_original": t_orig,
            "time_dual": t_dual,
        })

    print(f"\n  V-score Original : {np.mean(v_orig):.2e} +/- {np.std(v_orig):.2e}")
    print(f"  V-score DualEmbed: {np.mean(v_dual):.2e} +/- {np.std(v_dual):.2e}")


# ==========================================
# SAVE RAW RESULTS
# ==========================================

df = pd.DataFrame(all_results)
csv_path = os.path.join(graphs_dir, "vscore_comparison_embed.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")


# ==========================================
# PLOTTING
# ==========================================

# Prepare data for plotting
v_orig_by_sigma = [df[df["sigma"] == s]["v_score_original"].values for s in sigma_list]
v_dual_by_sigma = [df[df["sigma"] == s]["v_score_dual"].values for s in sigma_list]

mean_orig = [np.mean(v) for v in v_orig_by_sigma]
std_orig = [np.std(v) for v in v_orig_by_sigma]
mean_dual = [np.mean(v) for v in v_dual_by_sigma]
std_dual = [np.std(v) for v in v_dual_by_sigma]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- Left panel: Mean V-score with error bars ---
ax = axes[0]
x_pos = np.arange(len(sigma_list))
width = 0.35

bars1 = ax.bar(x_pos - width/2, mean_orig, width, yerr=std_orig,
               label="Original (single embed)", color="tab:blue", alpha=0.7,
               capsize=4, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, mean_dual, width, yerr=std_dual,
               label="DualEmbed (separate)", color="tab:orange", alpha=0.7,
               capsize=4, edgecolor="black", linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels([str(s) for s in sigma_list])
ax.set_xlabel(r"$\sigma$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (mean $\\pm$ std)", fontsize=12)
ax.set_yscale("log")
ax.set_title(f"V-score comparison — L={L}, $h_0$={h0}, {n_replicas} replicas", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# --- Right panel: Box plot ---
ax2 = axes[1]
positions_orig = np.arange(len(sigma_list)) * 3
positions_dual = positions_orig + 1

bp1 = ax2.boxplot(
    v_orig_by_sigma, positions=positions_orig, widths=0.8,
    patch_artist=True,
    boxprops=dict(facecolor="tab:blue", alpha=0.5),
    medianprops=dict(color="black"),
)
bp2 = ax2.boxplot(
    v_dual_by_sigma, positions=positions_dual, widths=0.8,
    patch_artist=True,
    boxprops=dict(facecolor="tab:orange", alpha=0.5),
    medianprops=dict(color="black"),
)

ax2.set_xticks(positions_orig + 0.5)
ax2.set_xticklabels([str(s) for s in sigma_list])
ax2.set_xlabel(r"$\sigma$ (disorder strength)", fontsize=12)
ax2.set_ylabel("V-score", fontsize=12)
ax2.set_yscale("log")
ax2.set_title("V-score distribution per model", fontsize=13)
ax2.legend(
    [bp1["boxes"][0], bp2["boxes"][0]],
    ["Original", "DualEmbed"],
    fontsize=10,
)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

pdf_path = os.path.join(graphs_dir, "vscore_comparison_embed.pdf")
png_path = os.path.join(graphs_dir, "vscore_comparison_embed.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:")
print(f"  {pdf_path}")
print(f"  {png_path}")
print(f"\nDone.")
