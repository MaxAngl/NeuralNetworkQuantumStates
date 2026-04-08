"""
Susceptibilité de fidélité χ(h₀)/N par autodiff + importance sampling.

Échantillonnage Metropolis à ~20 références dans [0.8, 1.2],
repondération SNIS sur une grille fine de 500 points.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk
import netket_foundational as nkf
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from netket_foundational._src.model.vit import ViTFNQS
from flip_rules import GlobalFlipRule

# ==========================================
# 1. CONFIGURATION
# ==========================================

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=str, help="Path to run directory")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file (default: latest)")
args = parser.parse_args()

RUN_DIR = args.run_dir

# Lire L et sigma depuis meta.json
with open(os.path.join(RUN_DIR, "meta.json"), "r") as f:
    meta = json.load(f)
L = meta["L"]
sigma_disorder = meta["hamiltonian"].get("sigma_disorder", meta["hamiltonian"].get("sigma", 0.1))

# Trouver le dernier checkpoint si non spécifié
if args.checkpoint:
    CHECKPOINT = args.checkpoint
else:
    import glob
    state_files = sorted(glob.glob(os.path.join(RUN_DIR, "state_*.nk")),
                         key=lambda x: int(os.path.basename(x).replace("state_", "").replace(".nk", "")))
    CHECKPOINT = os.path.basename(state_files[-1])
    print(f"Auto-selected checkpoint: {CHECKPOINT}")

J_val = 1.0
seed = 42

# ~20 points de référence pour le Metropolis, répartis dans [0.8, 1.2]
# Espacement ≈ 0.02, donc chaque point de la grille est à max ~0.01 d'une ref
h0_references = np.linspace(0.80, 1.20, 21).tolist()

# Grille fine = les mêmes 500 points du training
h0_grid = np.linspace(0.80, 1.20, 500)

# Monte Carlo pour l'échantillonnage aux références
n_mc_samples = 4096
n_mc_chains = 32
mc_chunk_size = 64
n_thermalization = 100

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
#
# Architecture (doit correspondre exactement au training) :
#   ViT-FNQS, disorder=True
#   - num_layers = 2
#   - d_model = 32 (16 spin + 16 coupling après embedding)
#   - heads = 4 (dim/tête = 8)
#   - b = 1 (patch = 1 site)
#   - transl_invariant = False
#   - complex = True (dernière couche complexe)
#
#   Input : concat([σ, γ]) ∈ ℝ^{2L}
#   Output : log ψ_θ(σ|γ) ∈ ℂ
#

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10 * 1.2)

vit_config = meta["vit_config"]
ma = ViTFNQS(
    num_layers=vit_config["num_layers"],
    d_model=vit_config["d_model"],
    heads=vit_config["heads"],
    b=vit_config["b"],
    L_eff=vit_config["L_eff"],
    n_coups=ps.size,
    complex=True,
    disorder=True,
    transl_invariant=False,
    two_dimensional=False,
)

sa = nk.sampler.MetropolisSampler(
    hi,
    rule=GlobalFlipRule(0.05),
    n_chains=n_mc_chains,
)
vs = nkf.FoundationalQuantumState(
    sa, ma, ps,
    n_replicas=1,
    n_samples=n_mc_samples,
    seed=seed,
    chunk_size=mc_chunk_size,
)

checkpoint_path = os.path.join(RUN_DIR, CHECKPOINT)
vs.load(checkpoint_path)
print(f"✅ Modèle chargé depuis {checkpoint_path}")

model = ma
variables = vs.variables

# ==========================================
# 3. FONCTIONS AUTODIFF
# ==========================================

def log_psi_of_gamma(gamma, sigma):
    """log ψ_θ(σ|γ) — input = concat([sigma, gamma]) ∈ ℝ^{2L}"""
    x = jnp.concatenate([sigma, gamma])
    return model.apply(variables, x)


@jax.jit
def compute_log_psi_batch(gamma, samples):
    """log ψ pour un batch de samples à gamma fixé."""
    return jax.vmap(lambda s: log_psi_of_gamma(gamma, s))(samples)


@jax.jit
def compute_O_and_logpsi(gamma, samples):
    """
    Pour chaque sample :
      - O(σ) = Σᵢ ∂log ψ / ∂γᵢ  (complexe, dérivée par rapport au param scalaire h₀)
      - log ψ(σ|γ)               (pour les poids IS)
    """
    def single_sample(sigma):
        lp = log_psi_of_gamma(gamma, sigma)
        
        # ∂log ψ / ∂γ = ∂Re(log ψ)/∂γ + i · ∂Im(log ψ)/∂γ
        grad_re = jax.grad(lambda g: jnp.real(log_psi_of_gamma(g, sigma)))(gamma)
        grad_im = jax.grad(lambda g: jnp.imag(log_psi_of_gamma(g, sigma)))(gamma)
        
        # Somme sur les sites → dérivée par rapport au paramètre scalaire h₀
        O = jnp.sum(grad_re + 1j * grad_im)
        return O, lp
    
    return jax.vmap(single_sample)(samples)


def compute_chi_snis(O_values, log_weights, N):
    """χ/N par SNIS."""
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights)
    
    mean_O = jnp.sum(weights * O_values)
    mean_OO = jnp.sum(weights * jnp.conj(O_values) * O_values)
    
    chi = jnp.real(mean_OO - jnp.conj(mean_O) * mean_O)
    return chi / N


def effective_sample_size(log_weights):
    """ESS — diagnostic qualité IS."""
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights)
    return 1.0 / jnp.sum(weights**2)


# ==========================================
# 4. ÉCHANTILLONNAGE AUX RÉFÉRENCES
# ==========================================

print(f"Échantillonnage Metropolis à {len(h0_references)} références...")

reference_data = {}

for h0_ref in h0_references:
    gamma_ref = np.full(L, h0_ref)
    
    _vs = vs.get_state(gamma_ref)
    mc_vs = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hi, n_chains=n_mc_chains),
        model=_vs.model,
        variables=_vs.variables,
        n_samples=n_mc_samples,
        chunk_size=mc_chunk_size,
    )
    mc_vs.reset()
    
    for _ in range(n_thermalization):
        mc_vs.sample()
    
    samples = mc_vs.sample().reshape(-1, L)
    
    gamma_ref_jax = jnp.full(L, h0_ref)
    log_psi_ref = compute_log_psi_batch(gamma_ref_jax, samples)
    
    reference_data[h0_ref] = {
        "samples": samples,
        "log_psi_ref": log_psi_ref,
    }
    
    print(f"  h₀ = {h0_ref:.3f} : {samples.shape[0]} échantillons")

# ==========================================
# 5. CALCUL DE χ SUR LA GRILLE FINE
# ==========================================

n_h0_points = len(h0_grid)
print(f"\nCalcul de χ(h₀)/N sur {n_h0_points} points par IS...")

chi_results = np.zeros(n_h0_points)
ess_results = np.zeros(n_h0_points)

for idx, h0_target in enumerate(h0_grid):
    # Référence la plus proche (distance max ~0.01)
    ref_idx = np.argmin(np.abs(np.array(h0_references) - h0_target))
    h0_ref = h0_references[ref_idx]
    
    samples = reference_data[h0_ref]["samples"]
    log_psi_ref = reference_data[h0_ref]["log_psi_ref"]
    
    gamma_target = jnp.full(L, h0_target)
    
    O_values, log_psi_target = compute_O_and_logpsi(gamma_target, samples)
    
    # Poids IS : w_k = |ψ_cible|² / |ψ_ref|²
    log_weights = 2.0 * jnp.real(log_psi_target - log_psi_ref)
    
    chi_val = compute_chi_snis(O_values, log_weights, L)
    ess_val = effective_sample_size(log_weights)
    
    chi_results[idx] = float(chi_val)
    ess_results[idx] = float(ess_val)
    
    if (idx + 1) % 100 == 0:
        print(f"  [{idx+1}/{n_h0_points}] h₀={h0_target:.4f}, χ/N={chi_val:.6f}, ESS={ess_val:.0f}/{samples.shape[0]}")

# ==========================================
# 6. RÉSULTATS ET PLOTS
# ==========================================

idx_max = np.argmax(chi_results)
h0_critical = h0_grid[idx_max]
print(f"\n🎯 Pic de χ/N à h₀ = {h0_critical:.4f} (χ/N = {chi_results[idx_max]:.6f})")

# Sauvegarder
output_data = np.column_stack([h0_grid, chi_results, ess_results])
np.savetxt(
    os.path.join(RUN_DIR, "fidelity_susceptibility.csv"),
    output_data,
    header="h0,chi_per_site,ess",
    delimiter=",",
    comments="",
)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                 gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(h0_grid, chi_results, "b-", linewidth=1.2)
ax1.axvline(h0_critical, color="r", linestyle="--", alpha=0.7, 
            label=f"Pic: h₀ = {h0_critical:.4f}")
for href in h0_references:
    ax1.axvline(href, color="gray", linestyle=":", alpha=0.2)
ax1.set_ylabel("χ(h₀) / N", fontsize=13)
ax1.set_title(f"Susceptibilité de fidélité — L={L}, σ_disorder={0.1}, J={J_val}", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.3)

ax2.plot(h0_grid, ess_results, "g-", linewidth=0.8, alpha=0.8)
ax2.axhline(n_mc_samples * 0.1, color="r", linestyle=":", alpha=0.5, label="10% seuil")
ax2.set_xlabel("h₀ / J", fontsize=13)
ax2.set_ylabel("ESS", fontsize=13)
ax2.set_yscale("log")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "fidelity_susceptibility.pdf"), dpi=150)
plt.savefig(os.path.join(RUN_DIR, "fidelity_susceptibility.png"), dpi=150)
print(f"✅ Figures sauvegardées dans {RUN_DIR}")
plt.show()
