"""
DualEmbed ViTFNQS: Patch size (b) comparison at L=36
Protocol: Train with sigma_train for each patch size b, evaluate on sigma_test values.
Evaluation via MCState (FullSumState impossible for L=36).

L=36, 1D disordered Ising, h0=1.0
Train: 20 replicas, sigma_train=0.1, 200 iters
Test:  40 sigma values in [0.01, 0.3], 10 realizations each

b values tested: 1, 2, 3, 4, 6 (divisors of 36)
  b=1 -> L_eff=36  (36 tokens of 1 spin)
  b=2 -> L_eff=18  (18 tokens of 2 spins)
  b=3 -> L_eff=12  (12 tokens of 3 spins)
  b=4 -> L_eff=9   (9 tokens of 4 spins)
  b=6 -> L_eff=6   (6 tokens of 6 spins)

Saves: graphs/vscore_dualembed_patchsize_L36.{pdf,png,csv}
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import netket as nk
import netket_foundational as nkf

from ansatz_dual_embed import ViTFNQS_DualEmbed


# ==========================================
# CONFIGURATION
# ==========================================

L = 36
seed = 42
h0 = 1.0
J_val = 1.0 / np.e

# Training config
sigma_train = 0.1
n_replicas_train = 20
n_iter = 200
lr_init = 0.03
lr_end = 0.005
diag_shift = 1e-4
chains_per_replica = 4
samples_per_chain = 2

# Patch sizes to compare (must be divisors of L)
b_values = [1, 2, 3, 4, 6]

# Test config
sigma_test_list = np.linspace(0.01, 0.3, 40)
N_test_per_sigma = 10

# Evaluation MC config
n_eval_chains = 16
n_eval_samples = 1024

# ViT hyperparameters (b will be varied)
vit_base = {
    "num_layers": 1,
    "d_model": 16,
    "heads": 2,
}

# Output
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
graphs_dir = os.path.join(project_root, "graphs")
os.makedirs(graphs_dir, exist_ok=True)


# ==========================================
# HELPER: create fresh system (avoid JAX tracer leaks between runs)
# ==========================================

def make_system():
    """Create fresh Hilbert space, ParameterSpace, operators."""
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

    return hi, ps, ha_p, mz_p, create_operator


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_model(b):
    """Train DualEmbed model with patch size b."""
    L_eff = L // b

    # Fresh system objects
    hi, ps, ha_p, mz_p, create_operator = make_system()

    rng = np.random.default_rng(seed)
    params_train = rng.normal(loc=h0, scale=sigma_train, size=(n_replicas_train, hi.size))

    n_chains = n_replicas_train * chains_per_replica
    n_samples = n_chains * samples_per_chain

    model = ViTFNQS_DualEmbed(
        num_layers=vit_base["num_layers"],
        d_model=vit_base["d_model"],
        heads=vit_base["heads"],
        b=b,
        L_eff=L_eff,
        n_coups=ps.size,
        complex=True,
        transl_invariant=False,
        two_dimensional=False,
    )

    sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vs = nkf.FoundationalQuantumState(
        sa, model, ps,
        n_replicas=n_replicas_train,
        n_samples=n_samples,
        seed=seed,
    )
    vs.parameter_array = params_train

    lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300)
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift)

    log = nk.logging.RuntimeLog()

    print(f"\n  Training DualEmbed b={b} (L_eff={L_eff}, "
          f"{n_replicas_train} replicas, n_samples={n_samples})...",
          flush=True)
    start = time.time()
    gs.run(n_iter, out=log, obs={"ham": ha_p, "mz": mz_p})
    duration = time.time() - start
    print(f"  b={b} training done: {duration:.1f}s", flush=True)

    return vs, hi, create_operator, duration


# ==========================================
# EVALUATION FUNCTION (MCState)
# ==========================================

def evaluate_on_sigmas(vs, hi, create_operator, label):
    """Evaluate trained model on test disorder configs using MCState."""
    rng_test = np.random.default_rng(seed + 1000)

    results = []

    for sigma_test in tqdm(sigma_test_list, desc=f"  {label} test sigmas"):
        test_params = rng_test.normal(
            loc=h0, scale=sigma_test, size=(N_test_per_sigma, hi.size)
        )

        v_scores = []
        for pars in test_params:
            _vs = vs.get_state(pars)
            _ha = create_operator(pars)

            sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=n_eval_chains)
            vs_mc = nk.vqs.MCState(
                sa_eval, _vs.model,
                variables=_vs.variables,
                n_samples=n_eval_samples,
                seed=seed + 2000,
            )
            _e = vs_mc.expect(_ha)
            v_score = float(_e.variance / (_e.Mean.real ** 2 + 1e-12))
            v_scores.append(v_score)

        v_arr = np.array(v_scores)
        results.append({
            "sigma_test": sigma_test,
            "v_scores": v_arr,
            "median": np.median(v_arr),
            "q25": np.percentile(v_arr, 25),
            "q75": np.percentile(v_arr, 75),
            "mean": np.mean(v_arr),
            "std": np.std(v_arr),
        })

        # Clear GPU caches periodically to avoid OOM
        jax.clear_caches()

    return results


# ==========================================
# MAIN
# ==========================================

print("=" * 60)
print("  DualEmbed Patch Size Comparison — L=36 (MCState eval)")
print(f"  L={L}, h0={h0}, J={J_val:.4f}")
print(f"  Train: {n_replicas_train} replicas, sigma_train={sigma_train}, "
      f"{n_iter} iters")
print(f"  Test: {len(sigma_test_list)} sigmas, "
      f"{N_test_per_sigma} realizations each")
print(f"  Eval: MCState with {n_eval_samples} samples, "
      f"{n_eval_chains} chains")
print(f"  ViT base: {vit_base}")
print(f"  Patch sizes b: {b_values}")
print(f"  -> L_eff values: {[L // b for b in b_values]}")
print("=" * 60, flush=True)

all_results = {}
train_times = {}

for b in b_values:
    label = f"b={b}"
    L_eff = L // b

    vs, hi, create_operator, t = train_model(b)
    train_times[label] = t

    print(f"\n  Evaluating {label} (L_eff={L_eff})...", flush=True)
    results = evaluate_on_sigmas(vs, hi, create_operator, label)
    all_results[label] = results

    # Clear JAX caches between runs
    jax.clear_caches()


# ==========================================
# SAVE RAW RESULTS
# ==========================================

rows = []
for b in b_values:
    label = f"b={b}"
    for r in all_results[label]:
        for i in range(N_test_per_sigma):
            rows.append({
                "b": b,
                "L_eff": L // b,
                "sigma_test": r["sigma_test"],
                "replica": i,
                "v_score": r["v_scores"][i],
            })

df = pd.DataFrame(rows)
csv_path = os.path.join(graphs_dir, "vscore_dualembed_patchsize_L36.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}", flush=True)


# ==========================================
# PLOTTING
# ==========================================

sigmas = sigma_test_list
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Median V-score vs sigma for each patch size ---
ax = axes[0]
for b, color in zip(b_values, colors):
    label = f"b={b}"
    L_eff = L // b
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    q25 = np.array([r["q25"] for r in res])
    q75 = np.array([r["q75"] for r in res])

    ax.plot(sigmas, med, "o-", color=color,
            label=f"b={b} (L_eff={L_eff})",
            markersize=3, linewidth=1.5)
    ax.fill_between(sigmas, q25, q75, color=color, alpha=0.12)

ax.axvline(sigma_train, color="gray", linestyle="--", alpha=0.7,
           label=f"$\\sigma_{{train}}={sigma_train}$")
ax.set_xlabel(r"$\sigma_{test}$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (median)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    f"DualEmbed patch size comparison — L={L}, $h_0$={h0}\n"
    f"Trained on $\\sigma={sigma_train}$, {n_replicas_train} replicas, "
    f"{n_iter} iters\n"
    f"Eval: MCState ({n_eval_samples} samples) | Shaded: IQR",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Right: Ratio relative to b=1 ---
ax2 = axes[1]
ref_label = f"b={b_values[0]}"
med_ref = np.array([r["median"] for r in all_results[ref_label]])

for b, color in zip(b_values[1:], colors[1:]):
    label = f"b={b}"
    L_eff = L // b
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    ratio = med / (med_ref + 1e-30)
    ax2.plot(sigmas, ratio, "o-", color=color,
             label=f"b={b} / b={b_values[0]}",
             markersize=3, linewidth=1.5)

ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
ax2.axvline(sigma_train, color="gray", linestyle="--", alpha=0.5,
            label=f"$\\sigma_{{train}}={sigma_train}$")
ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel(f"V-score ratio (vs b={b_values[0]})", fontsize=12)
ax2.set_title(f"Relative performance (< 1 = larger patch helps)",
              fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ylims = ax2.get_ylim()
ax2.fill_between(sigmas, 0, 1, alpha=0.05, color="green",
                 label="Larger patch helps")
ax2.fill_between(sigmas, 1, max(ylims[1], 2), alpha=0.05, color="red",
                 label="b=1 sufficient")
ax2.legend(fontsize=9)

plt.tight_layout()

pdf_path = os.path.join(graphs_dir, "vscore_dualembed_patchsize_L36.pdf")
png_path = os.path.join(graphs_dir, "vscore_dualembed_patchsize_L36.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:")
print(f"  {pdf_path}")
print(f"  {png_path}", flush=True)

# Summary
print(f"\n{'=' * 60}")
print(f"  SUMMARY — L={L}, patch size comparison")
print(f"{'=' * 60}")
for b in b_values:
    label = f"b={b}"
    L_eff = L // b
    res = all_results[label]
    global_med = np.median([r["median"] for r in res])
    print(f"  b={b:>2d} (L_eff={L_eff:>2d}): "
          f"train={train_times[label]:.1f}s, "
          f"global median V-score={global_med:.6f}")

# Ratio summary
print(f"\n  Ratios vs b={b_values[0]}:")
med_ref_all = np.array([r["median"] for r in all_results[f"b={b_values[0]}"]])
for b in b_values[1:]:
    label = f"b={b}"
    med_b = np.array([r["median"] for r in all_results[label]])
    ratio = med_b / (med_ref_all + 1e-30)
    print(f"    b={b}: median ratio = {np.median(ratio):.4f}, "
          f"wins {np.sum(ratio < 1)}/{len(ratio)} sigmas")

print(f"{'=' * 60}")
print("Done.", flush=True)
