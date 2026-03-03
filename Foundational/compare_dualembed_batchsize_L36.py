"""
DualEmbed ViTFNQS: Batch size comparison at L=36
Protocol: Train with sigma_train for each batch size, evaluate on sigma_test values.
Evaluation via MCState (FullSumState impossible for L=36).

L=36, 1D disordered Ising, h0=1.0
Train: 20 replicas, sigma_train=0.1, 200 iters
Test:  40 sigma values in [0.01, 0.3], 10 realizations each

Saves: graphs/vscore_dualembed_batchsize_L36.{pdf,png,csv}
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

# Batch sizes to compare (total n_samples = n_replicas * chains_per_rep * samples_per_ch)
batch_configs = [
    {"label": "n=160",  "chains_per_replica": 4,  "samples_per_chain": 2},   # 160
    {"label": "n=320",  "chains_per_replica": 4,  "samples_per_chain": 4},   # 320
    {"label": "n=640",  "chains_per_replica": 8,  "samples_per_chain": 4},   # 640
    {"label": "n=1280", "chains_per_replica": 16, "samples_per_chain": 4},   # 1280
]

# Test config
sigma_test_list = np.linspace(0.01, 0.3, 40)
N_test_per_sigma = 10

# Evaluation MC config (reduced to avoid GPU OOM at L=36)
n_eval_chains = 16
n_eval_samples = 1024

# ViT hyperparameters
vit_params = {
    "num_layers": 1,
    "d_model": 16,
    "heads": 2,
    "b": 1,
}

# Output
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
graphs_dir = os.path.join(project_root, "graphs")
os.makedirs(graphs_dir, exist_ok=True)


# ==========================================
# HELPER: create fresh operators (avoid JAX tracer leak between runs)
# ==========================================

def make_system():
    """Create fresh Hilbert space, ParameterSpace, operators.

    Must be called for each batch config to avoid JAX tracer leaks
    when reusing ParametrizedOperator across multiple training runs.
    """
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

def train_model(batch_cfg):
    """Train DualEmbed model with specific batch configuration.

    Creates fresh system objects to avoid JAX tracer cache issues.
    """
    label = batch_cfg["label"]
    chains_per_rep = batch_cfg["chains_per_replica"]
    samples_per_ch = batch_cfg["samples_per_chain"]

    # Fresh system for each batch config (avoids JAX tracer leak)
    hi, ps, ha_p, mz_p, create_operator = make_system()

    rng = np.random.default_rng(seed)
    params_train = rng.normal(loc=h0, scale=sigma_train, size=(n_replicas_train, hi.size))

    n_chains = n_replicas_train * chains_per_rep
    n_samples = n_chains * samples_per_ch

    model = ViTFNQS_DualEmbed(
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

    print(f"\n  Training DualEmbed {label} "
          f"({n_replicas_train} replicas, n_samples={n_samples})...")
    start = time.time()
    gs.run(n_iter, out=log, obs={"ham": ha_p, "mz": mz_p})
    duration = time.time() - start
    print(f"  {label} training done: {duration:.1f}s")

    return vs, hi, create_operator, duration


# ==========================================
# EVALUATION FUNCTION (MCState)
# ==========================================

def evaluate_on_sigmas(vs, hi, create_operator, label):
    """Evaluate trained model on test disorder configs using MCState.

    MCState is required because L=36 makes FullSumState (2^36 states) impossible.
    We use n_eval_samples MC samples to estimate <H> and Var(H).
    """
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
print("  DualEmbed Batch Size Comparison — L=36 (MCState eval)")
print(f"  L={L}, h0={h0}, J={J_val:.4f}")
print(f"  Train: {n_replicas_train} replicas, sigma_train={sigma_train}, "
      f"{n_iter} iters")
print(f"  Test: {len(sigma_test_list)} sigmas, "
      f"{N_test_per_sigma} realizations each")
print(f"  Eval: MCState with {n_eval_samples} samples, "
      f"{n_eval_chains} chains")
print(f"  ViT: {vit_params}")
print(f"  Batch configs: {[c['label'] for c in batch_configs]}")
print("=" * 60)

all_results = {}
train_times = {}

for cfg in batch_configs:
    # Each run gets fresh system objects to avoid JAX tracer leaks
    vs, hi, create_operator, t = train_model(cfg)
    train_times[cfg["label"]] = t

    print(f"\n  Evaluating {cfg['label']}...")
    results = evaluate_on_sigmas(vs, hi, create_operator, cfg["label"])
    all_results[cfg["label"]] = results

    # Clear JAX caches between runs
    jax.clear_caches()


# ==========================================
# SAVE RAW RESULTS
# ==========================================

rows = []
for cfg in batch_configs:
    label = cfg["label"]
    for r in all_results[label]:
        for i in range(N_test_per_sigma):
            rows.append({
                "batch_config": label,
                "sigma_test": r["sigma_test"],
                "replica": i,
                "v_score": r["v_scores"][i],
            })

df = pd.DataFrame(rows)
csv_path = os.path.join(graphs_dir, "vscore_dualembed_batchsize_L36.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")


# ==========================================
# PLOTTING
# ==========================================

sigmas = sigma_test_list
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Median V-score vs sigma for each batch size ---
ax = axes[0]
for cfg, color in zip(batch_configs, colors):
    label = cfg["label"]
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    q25 = np.array([r["q25"] for r in res])
    q75 = np.array([r["q75"] for r in res])

    n_total = n_replicas_train * cfg["chains_per_replica"] * cfg["samples_per_chain"]
    ax.plot(sigmas, med, "o-", color=color,
            label=f"{label} (total={n_total})",
            markersize=3, linewidth=1.5)
    ax.fill_between(sigmas, q25, q75, color=color, alpha=0.12)

ax.axvline(sigma_train, color="gray", linestyle="--", alpha=0.7,
           label=f"$\\sigma_{{train}}={sigma_train}$")
ax.set_xlabel(r"$\sigma_{test}$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (median)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    f"DualEmbed batch size comparison — L={L}, $h_0$={h0}\n"
    f"Trained on $\\sigma={sigma_train}$, {n_replicas_train} replicas, "
    f"{n_iter} iters\n"
    f"Eval: MCState ({n_eval_samples} samples) | Shaded: IQR",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Right: Ratio relative to smallest batch ---
ax2 = axes[1]
ref_label = batch_configs[0]["label"]
med_ref = np.array([r["median"] for r in all_results[ref_label]])

for cfg, color in zip(batch_configs[1:], colors[1:]):
    label = cfg["label"]
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    ratio = med / (med_ref + 1e-30)
    ax2.plot(sigmas, ratio, "o-", color=color,
             label=f"{label} / {ref_label}",
             markersize=3, linewidth=1.5)

ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
ax2.axvline(sigma_train, color="gray", linestyle="--", alpha=0.5,
            label=f"$\\sigma_{{train}}={sigma_train}$")
ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel(f"V-score ratio (vs {ref_label})", fontsize=12)
ax2.set_title("Relative performance (< 1 = larger batch helps)",
              fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ylims = ax2.get_ylim()
ax2.fill_between(sigmas, 0, 1, alpha=0.05, color="green",
                 label="Larger batch helps")
ax2.fill_between(sigmas, 1, max(ylims[1], 2), alpha=0.05, color="red",
                 label="Smallest batch sufficient")
ax2.legend(fontsize=9)

plt.tight_layout()

pdf_path = os.path.join(graphs_dir, "vscore_dualembed_batchsize_L36.pdf")
png_path = os.path.join(graphs_dir, "vscore_dualembed_batchsize_L36.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:")
print(f"  {pdf_path}")
print(f"  {png_path}")

# Summary
print(f"\n{'=' * 60}")
print(f"  SUMMARY — L={L}")
print(f"{'=' * 60}")
for cfg in batch_configs:
    label = cfg["label"]
    n_total = (n_replicas_train * cfg["chains_per_replica"]
               * cfg["samples_per_chain"])
    res = all_results[label]
    global_med = np.median([r["median"] for r in res])
    print(f"  {label:>8s} (n_samples={n_total:>5d}): "
          f"train={train_times[label]:.1f}s, "
          f"global median V-score={global_med:.6f}")

print(f"{'=' * 60}")
print("Done.")
