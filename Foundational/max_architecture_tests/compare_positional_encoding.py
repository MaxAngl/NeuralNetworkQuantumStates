"""
Comparison: Standard ViTFNQS vs ViTFNQS with Positional Encoding
Protocol: Train ONCE with sigma_train, evaluate on 40 sigma_test values.

L=16, 1D disordered Ising, h0=1.0
Train: 20 replicas, sigma_train=0.1
Test:  40 sigma values in [0.01, 0.3], 10 realizations each

Saves: graphs/vscore_comparison_posenc.{pdf,png,csv}
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
    ViTFNQS,
    Embed,
    Encoder,
    OuputHead,
    extract_patches1d,
    extract_patches2d,
    log_cosh,
)


# ==========================================
# POSITIONAL ENCODING MODEL
# ==========================================

class ViTFNQS_PosEnc(nn.Module):
    """ViTFNQS with learned positional embedding added after token embedding.

    Identical to the standard ViTFNQS (disorder=False path) except for
    a learned positional embedding of shape (1, L_eff, d_model) that is
    added to the token representations before entering the encoder.
    """
    num_layers: int
    d_model: int
    heads: int
    L_eff: int
    b: int
    n_coups: int
    complex: bool = False
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

        # Learned positional embedding: one vector per token position
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.L_eff, self.d_model),
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
        n_spin = self.n_coups
        spins = x[..., :-n_spin]
        hvals = x[..., -n_spin:]

        x = jnp.atleast_2d(spins)

        if self.two_dimensional:
            x = extract_patches2d(x, self.b)
        else:
            x = extract_patches1d(x, self.b)

        hvals = jnp.broadcast_to(
            hvals.reshape(*hvals.shape[:-1], 1, -1), (*x.shape[:-1], n_spin)
        )
        x = jnp.concatenate((x, hvals), axis=-1)
        x = self.embed(x)

        # --- Positional encoding: the only difference vs standard ---
        x = x + self.pos_embed

        x = self.encoder(x)
        return self.output(x)


# ==========================================
# CONFIGURATION
# ==========================================

L = 16
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

# Test config
sigma_test_list = np.linspace(0.01, 0.3, 40)
N_test_per_sigma = 10

# ViT hyperparameters (identical for both models)
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

def train_model(model, model_name):
    """Train model once on n_replicas_train disorder realizations."""
    rng = np.random.default_rng(seed)
    params_train = rng.normal(loc=h0, scale=sigma_train, size=(n_replicas_train, hi.size))

    n_chains = n_replicas_train * chains_per_replica
    n_samples = n_chains * samples_per_chain

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

    print(f"\n  Training {model_name} ({n_replicas_train} replicas, sigma_train={sigma_train})...")
    start = time.time()
    gs.run(n_iter, out=log, obs={"ham": ha_p, "mz": mz_p})
    duration = time.time() - start
    print(f"  {model_name} training done: {duration:.1f}s")

    return vs, duration


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_on_sigmas(vs, model_name):
    """Evaluate trained model on test disorder configs for each sigma_test."""
    rng_test = np.random.default_rng(seed + 1000)

    results = []

    for sigma_test in tqdm(sigma_test_list, desc=f"  {model_name} test sigmas"):
        test_params = rng_test.normal(loc=h0, scale=sigma_test, size=(N_test_per_sigma, hi.size))

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

print("=" * 60)
print("  Positional Encoding Comparison")
print(f"  L={L}, h0={h0}, J={J_val:.4f}")
print(f"  Train: {n_replicas_train} replicas, sigma_train={sigma_train}")
print(f"  Test: {len(sigma_test_list)} sigmas in [{sigma_test_list[0]:.2f}, {sigma_test_list[-1]:.2f}]")
print(f"         {N_test_per_sigma} realizations per sigma")
print(f"  ViT: {vit_params}")
print("=" * 60)

# --- Train Standard (no positional encoding) ---
model_std = ViTFNQS(
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
vs_std, t_std = train_model(model_std, "Standard")

# --- Train with Positional Encoding ---
model_pos = ViTFNQS_PosEnc(
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
vs_pos, t_pos = train_model(model_pos, "PosEnc")

# --- Evaluate both ---
print("\n" + "=" * 60)
print("  Evaluating on test sigma values...")
print("=" * 60)

results_std = evaluate_on_sigmas(vs_std, "Standard")
results_pos = evaluate_on_sigmas(vs_pos, "PosEnc")

# ==========================================
# SAVE RAW RESULTS
# ==========================================

rows = []
for rs, rp in zip(results_std, results_pos):
    sigma_t = rs["sigma_test"]
    for i in range(N_test_per_sigma):
        rows.append({
            "sigma_test": sigma_t,
            "replica": i,
            "v_score_standard": rs["v_scores"][i],
            "v_score_posenc": rp["v_scores"][i],
        })

df = pd.DataFrame(rows)
csv_path = os.path.join(graphs_dir, "vscore_comparison_posenc.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")


# ==========================================
# PLOTTING
# ==========================================

sigmas = np.array([r["sigma_test"] for r in results_std])
med_std = np.array([r["median"] for r in results_std])
q25_std = np.array([r["q25"] for r in results_std])
q75_std = np.array([r["q75"] for r in results_std])
med_pos = np.array([r["median"] for r in results_pos])
q25_pos = np.array([r["q25"] for r in results_pos])
q75_pos = np.array([r["q75"] for r in results_pos])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Median V-score vs sigma with IQR bands ---
ax = axes[0]
ax.plot(sigmas, med_std, "o-", color="tab:blue", label="Standard (no pos. enc.)", markersize=3, linewidth=1.5)
ax.fill_between(sigmas, q25_std, q75_std, color="tab:blue", alpha=0.2)
ax.plot(sigmas, med_pos, "s-", color="tab:red", label="+ Positional Encoding", markersize=3, linewidth=1.5)
ax.fill_between(sigmas, q25_pos, q75_pos, color="tab:red", alpha=0.2)

ax.axvline(sigma_train, color="gray", linestyle="--", alpha=0.7, label=f"$\\sigma_{{train}}={sigma_train}$")
ax.set_xlabel(r"$\sigma_{test}$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (median)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    f"Median V-score vs disorder — L={L}, $h_0$={h0}\n"
    f"Trained on $\\sigma={sigma_train}$, {n_replicas_train} replicas, {n_iter} iters\n"
    f"Shaded: IQR (25th-75th percentile)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Right: Ratio PosEnc / Standard ---
ax2 = axes[1]
ratio = med_pos / (med_std + 1e-30)
ax2.plot(sigmas, ratio, "k-o", markersize=3, linewidth=1.5)
ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
ax2.axvline(sigma_train, color="gray", linestyle="--", alpha=0.5, label=f"$\\sigma_{{train}}={sigma_train}$")

ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel("V-score ratio (PosEnc / Standard)", fontsize=12)
ax2.set_title("Relative performance (< 1 means PosEnc is better)", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Color the background
ylims = ax2.get_ylim()
ax2.fill_between(sigmas, 0, 1, alpha=0.05, color="green", label="PosEnc better")
ax2.fill_between(sigmas, 1, max(ylims[1], 2), alpha=0.05, color="red", label="Standard better")
ax2.legend(fontsize=9)

plt.tight_layout()

pdf_path = os.path.join(graphs_dir, "vscore_comparison_posenc.pdf")
png_path = os.path.join(graphs_dir, "vscore_comparison_posenc.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:")
print(f"  {pdf_path}")
print(f"  {png_path}")

# Summary
print(f"\n{'=' * 60}")
print(f"  SUMMARY")
print(f"{'=' * 60}")
print(f"  Training time Standard: {t_std:.1f}s")
print(f"  Training time PosEnc  : {t_pos:.1f}s")
print(f"  Median ratio (PosEnc/Std): {np.median(ratio):.4f}")
print(f"  Mean ratio (PosEnc/Std)  : {np.mean(ratio):.4f}")
print(f"  Sigmas where PosEnc wins  : {np.sum(ratio < 1)}/{len(ratio)}")
print(f"  Sigmas where Standard wins: {np.sum(ratio > 1)}/{len(ratio)}")

# Per-region analysis
mask_interp = sigmas <= sigma_train
mask_extrap = sigmas > sigma_train
if np.any(mask_interp):
    print(f"  Interpolation (sigma <= {sigma_train}): median ratio = {np.median(ratio[mask_interp]):.4f}")
if np.any(mask_extrap):
    print(f"  Extrapolation (sigma > {sigma_train}) : median ratio = {np.median(ratio[mask_extrap]):.4f}")

print(f"{'=' * 60}")
print("Done.")
