"""
Comparison: DualEmbed vs DualEmbed + FiLM conditioning
FiLM = Feature-wise Linear Modulation: h_i modulate features at each encoder layer.

L=16, b=4, 1D disordered Ising, h0=1.0
Train: 20 replicas, sigma_train=0.1, 200 iters
Test:  40 sigma values in [0.01, 0.3], 10 realizations each

Saves: graphs/vscore_comparison_dualembed_film.{pdf,png,csv}
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
    EncoderBlock,
    OuputHead,
    extract_patches1d,
    extract_patches2d,
)

from ansatz_dual_embed import ViTFNQS_DualEmbed


# ==========================================
# DUALEMBED + FiLM MODEL
# ==========================================

class ViTFNQS_DualEmbed_FiLM(nn.Module):
    """DualEmbed ViTFNQS with FiLM conditioning.

    At each encoder layer, the disorder field h_i modulates
    the features via learned scale (gamma) and shift (beta):
        x = gamma(h) * x + beta(h)

    gamma and beta are initialized so that FiLM is identity at init
    (gamma=1, beta=0), ensuring training starts from the same point
    as the non-FiLM model.
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
        assert self.d_model % 2 == 0
        half_d = self.d_model // 2

        # Same embeddings as DualEmbed
        self.embed_spins = Embed(
            half_d, self.b, two_dimensional=self.two_dimensional
        )
        self.embed_coups = Embed(
            half_d, self.b, two_dimensional=self.two_dimensional
        )

        # Encoder blocks (manual list instead of Encoder wrapper,
        # so we can apply FiLM between blocks)
        self.encoder_blocks = [
            EncoderBlock(
                d_model=self.d_model,
                h=self.heads,
                L_eff=self.L_eff,
                transl_invariant=self.transl_invariant,
                two_dimensional=self.two_dimensional,
            )
            for _ in range(self.num_layers)
        ]

        # FiLM: separate embedding of h patches for conditioning
        self.film_embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

        # Per-layer gamma (scale) and beta (shift) generators
        # Init: kernel=0, bias=1 for gamma -> gamma(h)=1 at init
        # Init: kernel=0, bias=0 for beta  -> beta(h)=0 at init
        # => FiLM is identity at initialization
        self.film_gammas = [
            nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.ones,
                param_dtype=jnp.float64,
                dtype=jnp.float64,
            )
            for _ in range(self.num_layers)
        ]
        self.film_betas = [
            nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                param_dtype=jnp.float64,
                dtype=jnp.float64,
            )
            for _ in range(self.num_layers)
        ]

        self.output = OuputHead(self.d_model, complex=self.complex)

    def __call__(self, x):
        n_coups = self.n_coups
        spins = x[..., :-n_coups]
        hvals = x[..., -n_coups:]

        spins = jnp.atleast_2d(spins)
        hvals = jnp.atleast_2d(hvals)
        hvals = jnp.broadcast_to(hvals, spins.shape)

        # Main embeddings (same as DualEmbed)
        x_spins = self.embed_spins(spins)    # (batch, L_eff, d_model/2)
        x_coups = self.embed_coups(hvals)    # (batch, L_eff, d_model/2)
        x = jnp.concatenate((x_spins, x_coups), axis=-1)

        # FiLM conditioning: separate embedding of h patches
        if self.two_dimensional:
            h_patches = extract_patches2d(hvals, self.b)
        else:
            h_patches = extract_patches1d(hvals, self.b)
        h_cond = self.film_embed(h_patches)  # (batch, L_eff, d_model)

        # Encoder with FiLM modulation at each layer
        for block, gamma_layer, beta_layer in zip(
            self.encoder_blocks, self.film_gammas, self.film_betas
        ):
            x = block(x)
            gamma = gamma_layer(h_cond)   # (batch, L_eff, d_model)
            beta = beta_layer(h_cond)     # (batch, L_eff, d_model)
            x = gamma * x + beta

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

# Test config
sigma_test_list = np.linspace(0.01, 0.3, 40)
N_test_per_sigma = 10

# ViT hyperparameters
d_model = 16
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

def train_model(model, model_name):
    """Train model once on n_replicas_train disorder realizations."""
    rng = np.random.default_rng(seed)
    params_train = rng.normal(loc=h0, scale=sigma_train,
                              size=(n_replicas_train, hi.size))

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

    lr = optax.linear_schedule(init_value=lr_init, end_value=lr_end,
                               transition_steps=300)
    optimizer = optax.sgd(lr)
    gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs,
                     diag_shift=diag_shift)

    log = nk.logging.RuntimeLog()

    print(f"\n  Training {model_name} ({n_replicas_train} replicas, "
          f"b={b}, L_eff={L_eff})...", flush=True)
    start = time.time()
    gs.run(n_iter, out=log, obs={"ham": ha_p, "mz": mz_p})
    duration = time.time() - start
    print(f"  {model_name} training done: {duration:.1f}s", flush=True)

    return vs, duration


# ==========================================
# EVALUATION FUNCTION (FullSumState — exact for L=16)
# ==========================================

def evaluate_on_sigmas(vs, model_name):
    """Evaluate trained model on test disorder configs."""
    rng_test = np.random.default_rng(seed + 1000)

    results = []

    for sigma_test in tqdm(sigma_test_list,
                           desc=f"  {model_name} test sigmas"):
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
print("  DualEmbed vs DualEmbed + FiLM Conditioning", flush=True)
print(f"  L={L}, b={b}, L_eff={L_eff}, h0={h0}, J={J_val:.4f}",
      flush=True)
print(f"  Train: {n_replicas_train} replicas, sigma_train={sigma_train}, "
      f"{n_iter} iters", flush=True)
print(f"  Test: {len(sigma_test_list)} sigmas, "
      f"{N_test_per_sigma} realizations each", flush=True)
print(f"  Eval: FullSumState (exact)", flush=True)
print(f"  d_model={d_model}, num_layers={num_layers}, heads={heads}",
      flush=True)
print("=" * 60, flush=True)

# --- DualEmbed (baseline) ---
model_dual = ViTFNQS_DualEmbed(
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
vs_dual, t_dual = train_model(model_dual, "DualEmbed")

# --- DualEmbed + FiLM ---
model_film = ViTFNQS_DualEmbed_FiLM(
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
vs_film, t_film = train_model(model_film, "DualEmbed+FiLM")

# --- Evaluate both ---
print("\n" + "=" * 60, flush=True)
print("  Evaluating on test sigma values...", flush=True)
print("=" * 60, flush=True)

results_dual = evaluate_on_sigmas(vs_dual, "DualEmbed")
results_film = evaluate_on_sigmas(vs_film, "DualEmbed+FiLM")

# ==========================================
# SAVE RAW RESULTS
# ==========================================

rows = []
for rd, rf in zip(results_dual, results_film):
    sigma_t = rd["sigma_test"]
    for i in range(N_test_per_sigma):
        rows.append({
            "sigma_test": sigma_t,
            "replica": i,
            "v_score_dualembed": rd["v_scores"][i],
            "v_score_dualembed_film": rf["v_scores"][i],
        })

df = pd.DataFrame(rows)
csv_path = os.path.join(graphs_dir, "vscore_comparison_dualembed_film.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}", flush=True)


# ==========================================
# PLOTTING
# ==========================================

sigmas = np.array([r["sigma_test"] for r in results_dual])
med_dual = np.array([r["median"] for r in results_dual])
q25_dual = np.array([r["q25"] for r in results_dual])
q75_dual = np.array([r["q75"] for r in results_dual])
med_film = np.array([r["median"] for r in results_film])
q25_film = np.array([r["q25"] for r in results_film])
q75_film = np.array([r["q75"] for r in results_film])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Median V-score vs sigma ---
ax = axes[0]
ax.plot(sigmas, med_dual, "o-", color="tab:orange",
        label="DualEmbed", markersize=3, linewidth=1.5)
ax.fill_between(sigmas, q25_dual, q75_dual, color="tab:orange", alpha=0.2)
ax.plot(sigmas, med_film, "s-", color="tab:green",
        label="DualEmbed + FiLM", markersize=3, linewidth=1.5)
ax.fill_between(sigmas, q25_film, q75_film, color="tab:green", alpha=0.2)

ax.axvline(sigma_train, color="gray", linestyle="--", alpha=0.7,
           label=f"$\\sigma_{{train}}={sigma_train}$")
ax.set_xlabel(r"$\sigma_{test}$ (disorder strength)", fontsize=12)
ax.set_ylabel("V-score (median)", fontsize=12)
ax.set_yscale("log")
ax.set_title(
    f"DualEmbed vs FiLM — L={L}, b={b}, L_eff={L_eff}\n"
    f"d_model={d_model}, {num_layers} layer, {heads} heads, "
    f"{n_iter} iters\n"
    f"Shaded: IQR (25th-75th percentile)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Right: Ratio FiLM / DualEmbed ---
ax2 = axes[1]
ratio = med_film / (med_dual + 1e-30)
ax2.plot(sigmas, ratio, "k-o", markersize=3, linewidth=1.5)
ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
ax2.axvline(sigma_train, color="gray", linestyle="--", alpha=0.5,
            label=f"$\\sigma_{{train}}={sigma_train}$")

ax2.set_xlabel(r"$\sigma_{test}$", fontsize=12)
ax2.set_ylabel("V-score ratio (FiLM / DualEmbed)", fontsize=12)
ax2.set_title("Relative performance (< 1 means FiLM helps)",
              fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ylims = ax2.get_ylim()
ax2.fill_between(sigmas, 0, 1, alpha=0.05, color="green",
                 label="FiLM helps")
ax2.fill_between(sigmas, 1, max(ylims[1], 2), alpha=0.05, color="red",
                 label="DualEmbed alone better")
ax2.legend(fontsize=9)

plt.tight_layout()

pdf_path = os.path.join(graphs_dir,
                        "vscore_comparison_dualembed_film.pdf")
png_path = os.path.join(graphs_dir,
                        "vscore_comparison_dualembed_film.png")
plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")

print(f"\nPlots saved to:", flush=True)
print(f"  {pdf_path}", flush=True)
print(f"  {png_path}", flush=True)

# Summary
print(f"\n{'=' * 60}", flush=True)
print(f"  SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"  Training time DualEmbed     : {t_dual:.1f}s", flush=True)
print(f"  Training time DualEmbed+FiLM: {t_film:.1f}s", flush=True)
print(f"  Median ratio (FiLM/Dual): {np.median(ratio):.4f}", flush=True)
print(f"  Mean ratio (FiLM/Dual)  : {np.mean(ratio):.4f}", flush=True)
print(f"  Sigmas where FiLM helps    : "
      f"{np.sum(ratio < 1)}/{len(ratio)}", flush=True)
print(f"  Sigmas where DualEmbed wins: "
      f"{np.sum(ratio > 1)}/{len(ratio)}", flush=True)

mask_interp = sigmas <= sigma_train
mask_extrap = sigmas > sigma_train
if np.any(mask_interp):
    print(f"  Interpolation (sigma <= {sigma_train}): "
          f"median ratio = {np.median(ratio[mask_interp]):.4f}",
          flush=True)
if np.any(mask_extrap):
    print(f"  Extrapolation (sigma > {sigma_train}) : "
          f"median ratio = {np.median(ratio[mask_extrap]):.4f}",
          flush=True)

print(f"{'=' * 60}", flush=True)
print("Done.", flush=True)
