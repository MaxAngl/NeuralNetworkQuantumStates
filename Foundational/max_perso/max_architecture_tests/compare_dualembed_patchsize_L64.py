"""
DualEmbed ViTFNQS: Patch size (b) comparison at L=64
Compare DualEmbed with different patch sizes b = 2, 4, 8, 16.
Evaluation via MCState (FullSumState impossible for L=64).

L=64, 1D disordered Ising, h0=1.0
Train: 20 replicas, sigma_train=0.1, 200 iters
Test:  40 sigma values in [0.01, 0.3], 10 realizations each

b=2  -> L_eff=32  (32 tokens of 2 spins)
b=4  -> L_eff=16  (16 tokens of 4 spins)
b=8  -> L_eff=8   ( 8 tokens of 8 spins)  <-- predicted optimal (L_eff ~ 9)
b=16 -> L_eff=4   ( 4 tokens of 16 spins)

Saves: graphs/vscore_dualembed_patchsize_L64.{pdf,png,csv}
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

L = 64
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

# Patch sizes to compare (divisors of 64, skip b=1)
patch_configs = [
    {"b": 2,  "L_eff": 32},
    {"b": 4,  "L_eff": 16},
    {"b": 8,  "L_eff": 8},
    {"b": 16, "L_eff": 4},
]

# Test config
sigma_test_list = np.linspace(0.01, 0.3, 40)
N_test_per_sigma = 10

# Evaluation MC config
n_eval_chains = 16
n_eval_samples = 1024

# ViT hyperparameters (shared except b and L_eff)
d_model = 16
num_layers = 1
heads = 2

# Output
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
graphs_dir = os.path.join(project_root, "graphs")
os.makedirs(graphs_dir, exist_ok=True)


# ==========================================
# HELPER: create fresh operators
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

def train_model(pcfg):
    """Train DualEmbed model with specific patch size."""
    b = pcfg["b"]
    L_eff = pcfg["L_eff"]

    hi, ps, ha_p, mz_p, create_operator = make_system()

    rng = np.random.default_rng(seed)
    params_train = rng.normal(loc=h0, scale=sigma_train,
                              size=(n_replicas_train, hi.size))

    n_chains = n_replicas_train * chains_per_replica
    n_samples = n_chains * samples_per_chain

    model = ViTFNQS_DualEmbed(
        num_layers=num_layers,
        d_model=d_model,
        heads=heads,
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

    lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end,
                               transition_steps=300)
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs,
                     diag_shift=diag_shift)

    log = nk.logging.RuntimeLog()

    print(f"\n  Training DualEmbed b={b} "
          f"(L_eff={L_eff}, {n_replicas_train} replicas)...",
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

        jax.clear_caches()

    return results


# ==========================================
# MAIN
# ==========================================

print("=" * 60, flush=True)
print("  DualEmbed Patch Size (b) Comparison — L=64 (MCState eval)",
      flush=True)
print(f"  L={L}, h0={h0}, J={J_val:.4f}", flush=True)
print(f"  Train: {n_replicas_train} replicas, sigma_train={sigma_train}, "
      f"{n_iter} iters", flush=True)
print(f"  Test: {len(sigma_test_list)} sigmas, "
      f"{N_test_per_sigma} realizations each", flush=True)
print(f"  Eval: MCState ({n_eval_samples} samples, "
      f"{n_eval_chains} chains)", flush=True)
print(f"  d_model={d_model}, num_layers={num_layers}, heads={heads}",
      flush=True)
print(f"  Patch sizes: {[c['b'] for c in patch_configs]}", flush=True)
print(f"  -> L_eff:     {[c['L_eff'] for c in patch_configs]}",
      flush=True)
print("=" * 60, flush=True)

all_results = {}
train_times = {}

for pcfg in patch_configs:
    label = f"b={pcfg['b']}"

    vs, hi, create_operator, t = train_model(pcfg)
    train_times[label] = t

    print(f"\n  Evaluating {label} (L_eff={pcfg['L_eff']})...",
          flush=True)
    results = evaluate_on_sigmas(vs, hi, create_operator, label)
    all_results[label] = results

    jax.clear_caches()


# ==========================================
# SAVE RAW RESULTS
# ==========================================

rows = []
for pcfg in patch_configs:
    label = f"b={pcfg['b']}"
    for r in all_results[label]:
        for i in range(N_test_per_sigma):
            rows.append({
                "patch_size_b": pcfg["b"],
                "L_eff": pcfg["L_eff"],
                "sigma_test": r["sigma_test"],
                "replica": i,
                "v_score": r["v_scores"][i],
            })

df = pd.DataFrame(rows)
csv_path = os.path.join(graphs_dir, "vscore_dualembed_patchsize_L64.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}", flush=True)


# ==========================================
# PLOTTING
# ==========================================

sigmas = sigma_test_list
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Median V-score vs sigma for each patch size ---
ax = axes[0]
for pcfg, color in zip(patch_configs, colors):
    label = f"b={pcfg['b']}"
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    q25 = np.array([r["q25"] for r in res])
    q75 = np.array([r["q75"] for r in res])

    ax.plot(sigmas, med, "o-", color=color,
            label=f"b={pcfg['b']} (L_eff={pcfg['L_eff']})",
            markersize=3, linewidth=1.5)
    ax.fill_between(sigmas, q25, q75, color=color, alpha=0.12)

ax.axvline(sigma_train, color="gray", linestyle="--", alpha=0.7,
           label=f"$\\sigma_{{train}}={sigma_train}$")
ax.set_xlabel(r"$\sigma_{test}$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (median)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    f"DualEmbed patch size comparison — L={L}, $h_0$={h0}\n"
    f"d_model={d_model}, {num_layers} layer, {heads} heads, "
    f"{n_iter} iters\n"
    f"Eval: MCState ({n_eval_samples} samples) | Shaded: IQR",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Right: Ratio relative to b=2 (smallest tested) ---
ax2 = axes[1]
ref_label = f"b={patch_configs[0]['b']}"
med_ref = np.array([r["median"] for r in all_results[ref_label]])

for pcfg, color in zip(patch_configs[1:], colors[1:]):
    label = f"b={pcfg['b']}"
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    ratio = med / (med_ref + 1e-30)
    ax2.plot(sigmas, ratio, "o-", color=color,
             label=f"b={pcfg['b']} / b={patch_configs[0]['b']}",
             markersize=3, linewidth=1.5)

ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
ax2.axvline(sigma_train, color="gray", linestyle="--", alpha=0.5,
            label=f"$\\sigma_{{train}}={sigma_train}$")
ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel(f"V-score ratio (vs b={patch_configs[0]['b']})",
               fontsize=12)
ax2.set_title("Relative performance (< 1 = larger patch helps)",
              fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ylims = ax2.get_ylim()
ax2.fill_between(sigmas, 0, 1, alpha=0.05, color="green",
                 label="Larger b helps")
ax2.fill_between(sigmas, 1, max(ylims[1], 2), alpha=0.05, color="red",
                 label=f"b={patch_configs[0]['b']} better")
ax2.legend(fontsize=9)

plt.tight_layout()

pdf_path = os.path.join(graphs_dir, "vscore_dualembed_patchsize_L64.pdf")
png_path = os.path.join(graphs_dir, "vscore_dualembed_patchsize_L64.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:", flush=True)
print(f"  {pdf_path}", flush=True)
print(f"  {png_path}", flush=True)

# Summary
print(f"\n{'=' * 60}", flush=True)
print(f"  SUMMARY — L={L}, DualEmbed patch size comparison",
      flush=True)
print(f"{'=' * 60}", flush=True)
for pcfg in patch_configs:
    label = f"b={pcfg['b']}"
    res = all_results[label]
    global_med = np.median([r["median"] for r in res])
    print(f"  b={pcfg['b']:>2d} (L_eff={pcfg['L_eff']:>2d}): "
          f"train={train_times[label]:.1f}s, "
          f"global median V-score={global_med:.6f}",
          flush=True)

# Ratio summary
ref_b = patch_configs[0]["b"]
ref_label = f"b={ref_b}"
med_ref_all = np.array([r["median"] for r in all_results[ref_label]])
print(f"\n  Ratios vs b={ref_b}:", flush=True)
for pcfg in patch_configs[1:]:
    label = f"b={pcfg['b']}"
    med_b = np.array([r["median"] for r in all_results[label]])
    ratio = med_b / (med_ref_all + 1e-30)
    mask_interp = sigmas <= sigma_train
    mask_extrap = sigmas > sigma_train
    print(f"    b={pcfg['b']:>2d}: median ratio={np.median(ratio):.4f}, "
          f"interp={np.median(ratio[mask_interp]):.4f}, "
          f"extrap={np.median(ratio[mask_extrap]):.4f}",
          flush=True)

print(f"{'=' * 60}", flush=True)
print("Done.", flush=True)
