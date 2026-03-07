"""
DualEmbed: Symmetric vs Asymmetric embedding split comparison.
Compare different allocations of d_model between spins and couplings.

L=16, b=4, 1D disordered Ising, h0=1.0
Train: 20 replicas, sigma_train=0.1, 200 iters
Test:  40 sigma values in [0.01, 0.3], 10 realizations each

Splits tested (d_spin / d_coup, total d_model=16):
  4/12, 6/10, 8/8 (baseline), 10/6, 12/4

Saves: graphs/vscore_dualembed_split.{pdf,png,csv}
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
from netket_foundational._src.model.vit import (
    Embed,
    Encoder,
    OuputHead,
    extract_patches1d,
)


# ==========================================
# ASYMMETRIC DUALEMBED MODEL
# ==========================================

class ViTFNQS_DualEmbed_Asym(nn.Module):
    """DualEmbed with configurable split between spin and coupling dims.

    d_spin + d_coup = d_model.
    Standard DualEmbed uses d_spin = d_coup = d_model/2.
    """
    num_layers: int
    d_model: int
    d_spin: int       # embedding dim for spins
    d_coup: int       # embedding dim for couplings
    heads: int
    L_eff: int
    b: int
    n_coups: int
    complex: bool = False
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        assert self.d_spin + self.d_coup == self.d_model, \
            f"d_spin({self.d_spin}) + d_coup({self.d_coup}) != d_model({self.d_model})"

        self.embed_spins = Embed(
            self.d_spin, self.b, two_dimensional=self.two_dimensional
        )
        self.embed_coups = Embed(
            self.d_coup, self.b, two_dimensional=self.two_dimensional
        )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
        )

        self.output = OuputHead(self.d_model, complex=self.complex)

    def __call__(self, x):
        n_coups = self.n_coups
        spins = x[..., :-n_coups]
        hvals = x[..., -n_coups:]

        spins = jnp.atleast_2d(spins)
        hvals = jnp.atleast_2d(hvals)
        hvals = jnp.broadcast_to(hvals, spins.shape)

        x_spins = self.embed_spins(spins)    # (batch, L_eff, d_spin)
        x_coups = self.embed_coups(hvals)    # (batch, L_eff, d_coup)

        x = jnp.concatenate((x_spins, x_coups), axis=-1)  # (batch, L_eff, d_model)

        x = self.encoder(x)
        return self.output(x)


# ==========================================
# CONFIGURATION
# ==========================================

L = 16
b = 4
L_eff = L // b  # = 4
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

# Split configurations (d_spin, d_coup)
d_model = 16
split_configs = [
    {"d_spin": 4,  "d_coup": 12, "label": "4/12"},
    {"d_spin": 6,  "d_coup": 10, "label": "6/10"},
    {"d_spin": 8,  "d_coup": 8,  "label": "8/8 (baseline)"},
    {"d_spin": 10, "d_coup": 6,  "label": "10/6"},
    {"d_spin": 12, "d_coup": 4,  "label": "12/4"},
]

# Test config
sigma_test_list = np.linspace(0.01, 0.3, 40)
N_test_per_sigma = 10

# ViT hyperparameters
num_layers = 1
heads = 2

# Output
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


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_model(scfg):
    """Train DualEmbed model with specific split configuration."""
    label = scfg["label"]
    d_spin = scfg["d_spin"]
    d_coup = scfg["d_coup"]

    rng = np.random.default_rng(seed)
    params_train = rng.normal(loc=h0, scale=sigma_train,
                              size=(n_replicas_train, hi.size))

    n_chains = n_replicas_train * chains_per_replica
    n_samples = n_chains * samples_per_chain

    model = ViTFNQS_DualEmbed_Asym(
        num_layers=num_layers,
        d_model=d_model,
        d_spin=d_spin,
        d_coup=d_coup,
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

    print(f"\n  Training split={label} "
          f"(d_spin={d_spin}, d_coup={d_coup})...", flush=True)
    start = time.time()
    gs.run(n_iter, out=log, obs={"ham": ha_p, "mz": mz_p})
    duration = time.time() - start
    print(f"  {label} training done: {duration:.1f}s", flush=True)

    return vs, duration


# ==========================================
# EVALUATION FUNCTION (FullSumState — exact for L=16)
# ==========================================

def evaluate_on_sigmas(vs, label):
    """Evaluate trained model on test disorder configs."""
    rng_test = np.random.default_rng(seed + 1000)

    results = []

    for sigma_test in tqdm(sigma_test_list,
                           desc=f"  {label} test sigmas"):
        test_params = rng_test.normal(
            loc=h0, scale=sigma_test, size=(N_test_per_sigma, hi.size)
        )

        v_scores = []
        for pars in test_params:
            _vs = vs.get_state(pars)
            _ha = create_operator(pars)

            vs_fs = nk.vqs.FullSumState(
                hilbert=hi, model=_vs.model, variables=_vs.variables
            )
            _e = vs_fs.expect(_ha)
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

    return results


# ==========================================
# MAIN
# ==========================================

print("=" * 60, flush=True)
print("  DualEmbed Split Comparison — Symmetric vs Asymmetric",
      flush=True)
print(f"  L={L}, b={b}, L_eff={L_eff}, h0={h0}, J={J_val:.4f}",
      flush=True)
print(f"  d_model={d_model}, num_layers={num_layers}, heads={heads}",
      flush=True)
print(f"  Train: {n_replicas_train} replicas, sigma_train={sigma_train}, "
      f"{n_iter} iters", flush=True)
print(f"  Test: {len(sigma_test_list)} sigmas, "
      f"{N_test_per_sigma} realizations each", flush=True)
print(f"  Eval: FullSumState (exact)", flush=True)
print(f"  Splits: {[c['label'] for c in split_configs]}", flush=True)
print("=" * 60, flush=True)

all_results = {}
train_times = {}

for scfg in split_configs:
    label = scfg["label"]

    vs, t = train_model(scfg)
    train_times[label] = t

    print(f"\n  Evaluating {label}...", flush=True)
    results = evaluate_on_sigmas(vs, label)
    all_results[label] = results


# ==========================================
# SAVE RAW RESULTS
# ==========================================

rows = []
for scfg in split_configs:
    label = scfg["label"]
    for r in all_results[label]:
        for i in range(N_test_per_sigma):
            rows.append({
                "d_spin": scfg["d_spin"],
                "d_coup": scfg["d_coup"],
                "split": label,
                "sigma_test": r["sigma_test"],
                "replica": i,
                "v_score": r["v_scores"][i],
            })

df = pd.DataFrame(rows)
csv_path = os.path.join(graphs_dir, "vscore_dualembed_split.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}", flush=True)


# ==========================================
# PLOTTING
# ==========================================

sigmas = sigma_test_list
colors = ["tab:purple", "tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Median V-score vs sigma for each split ---
ax = axes[0]
for scfg, color in zip(split_configs, colors):
    label = scfg["label"]
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    q25 = np.array([r["q25"] for r in res])
    q75 = np.array([r["q75"] for r in res])

    ax.plot(sigmas, med, "o-", color=color,
            label=f"d_spin/d_coup = {label}",
            markersize=3, linewidth=1.5)
    ax.fill_between(sigmas, q25, q75, color=color, alpha=0.12)

ax.axvline(sigma_train, color="gray", linestyle="--", alpha=0.7,
           label=f"$\\sigma_{{train}}={sigma_train}$")
ax.set_xlabel(r"$\sigma_{test}$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (median)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    f"DualEmbed split comparison — L={L}, b={b}, L_eff={L_eff}\n"
    f"d_model={d_model}, {num_layers} layer, {heads} heads, "
    f"{n_iter} iters\n"
    f"Spins are binary (±0.5), h_i are continuous ~N(h0, σ)",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Right: Ratio relative to 8/8 baseline ---
ax2 = axes[1]
ref_label = "8/8 (baseline)"
med_ref = np.array([r["median"] for r in all_results[ref_label]])

for scfg, color in zip(split_configs, colors):
    label = scfg["label"]
    if label == ref_label:
        continue
    res = all_results[label]
    med = np.array([r["median"] for r in res])
    ratio = med / (med_ref + 1e-30)
    ax2.plot(sigmas, ratio, "o-", color=color,
             label=f"{label} / 8|8",
             markersize=3, linewidth=1.5)

ax2.axhline(1.0, color="tab:orange", linestyle="--", alpha=0.7,
            linewidth=2, label="8/8 baseline")
ax2.axvline(sigma_train, color="gray", linestyle="--", alpha=0.5,
            label=f"$\\sigma_{{train}}={sigma_train}$")
ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel("V-score ratio (vs 8/8 baseline)", fontsize=12)
ax2.set_title("< 1 = better than symmetric split", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ylims = ax2.get_ylim()
ax2.fill_between(sigmas, 0, 1, alpha=0.05, color="green")
ax2.fill_between(sigmas, 1, max(ylims[1], 3), alpha=0.05, color="red")

plt.tight_layout()

pdf_path = os.path.join(graphs_dir, "vscore_dualembed_split.pdf")
png_path = os.path.join(graphs_dir, "vscore_dualembed_split.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:", flush=True)
print(f"  {pdf_path}", flush=True)
print(f"  {png_path}", flush=True)

# Summary
print(f"\n{'=' * 60}", flush=True)
print(f"  SUMMARY — DualEmbed split comparison", flush=True)
print(f"{'=' * 60}", flush=True)
for scfg in split_configs:
    label = scfg["label"]
    res = all_results[label]
    global_med = np.median([r["median"] for r in res])
    print(f"  {label:>15s}: train={train_times[label]:.1f}s, "
          f"global median V-score={global_med:.6f}", flush=True)

# Ratios
ref_med = np.array([r["median"] for r in all_results[ref_label]])
print(f"\n  Ratios vs {ref_label}:", flush=True)
for scfg in split_configs:
    label = scfg["label"]
    if label == ref_label:
        continue
    med = np.array([r["median"] for r in all_results[label]])
    ratio = med / (ref_med + 1e-30)
    mask_interp = sigmas <= sigma_train
    mask_extrap = sigmas > sigma_train
    print(f"    {label:>5s}: median ratio={np.median(ratio):.4f}, "
          f"wins {np.sum(ratio < 1)}/{len(ratio)}, "
          f"interp={np.median(ratio[mask_interp]):.4f}, "
          f"extrap={np.median(ratio[mask_extrap]):.4f}", flush=True)

# Find best split
best_label = None
best_med = float("inf")
for scfg in split_configs:
    label = scfg["label"]
    res = all_results[label]
    global_med = np.median([r["median"] for r in res])
    if global_med < best_med:
        best_med = global_med
        best_label = label

print(f"\n  Best split: {best_label} "
      f"(global median V-score = {best_med:.6f})", flush=True)
print(f"{'=' * 60}", flush=True)
print("Done.", flush=True)
