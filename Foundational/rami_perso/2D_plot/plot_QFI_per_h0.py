from functools import partial
"""
Fidelity susceptibility chi_F vs h0 pour FNQS 2D.
chi_F = (1 - |<psi(h0)|psi(h0+delta)>|^2) / delta^2

Usage:
    python plot_QFI_per_h0.py <run_dir> [--delta 1e-3] [--n-samples 4096] [--n-rep 10]
"""
import os
import sys
import json
import argparse
import zipfile
import flax
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=str)
parser.add_argument("--delta", type=float, default=1e-3)
parser.add_argument("--n-samples", type=int, default=4096)
parser.add_argument("--n-rep", type=int, default=10)
parser.add_argument("--h0-min", type=float, default=0.5)
parser.add_argument("--h0-max", type=float, default=5.0)
parser.add_argument("--h0-npts", type=int, default=40)
args = parser.parse_args()

RUN_DIR = Path(args.run_dir)
DELTA_H = args.delta
N_SAMPLES = args.n_samples
N_REP = args.n_rep

# --- CHARGEMENT META ---
with open(RUN_DIR / "meta.json", "r") as f:
    meta = json.load(f)

L = meta["L"]
IS_2D = meta.get("n_dim", 1) == 2
n_spins = meta.get("nb_spins", L)
vit_params = meta["vit_config"]
sigma = meta["hamiltonian"]["sigma"]

print(f"System: {n_spins} spins, {'2D' if IS_2D else '1D'}, L={L}, sigma={sigma}")

# --- INITIALISATION ---
hi = nk.hilbert.Spin(0.5, n_spins)
ps = nkf.ParameterSpace(N=n_spins, min=0, max=10)

ma = ViTFNQS(
    num_layers=vit_params["num_layers"],
    d_model=vit_params["d_model"],
    heads=vit_params["heads"],
    b=vit_params["b"],
    L_eff=vit_params["L_eff"],
    n_coups=ps.size,
    complex=True,
    disorder=True,
    transl_invariant=False,
    two_dimensional=IS_2D,
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=N_SAMPLES)

# --- CHARGEMENT DES POIDS ---
state_files = sorted(RUN_DIR.glob("state_*.nk"), key=lambda x: int(x.stem.split("_")[1]))
CHECKPOINT_PATH = state_files[-1]
print(f"Loading: {CHECKPOINT_PATH}")

if not zipfile.is_zipfile(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(CHECKPOINT_PATH, "r") as zf:
        candidates = [f for f in zf.namelist() if f.endswith(".msgpack")]
        target_file = sorted(candidates, key=len)[-1]
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

vars_dict = state_dict.get("variables", state_dict.get("model", {}).get("variables", state_dict))
vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)

# --- OVERLAP CORRIGE ---
# Les samples FNQS ont shape (n_samples, n_spins + n_coups)
# x = [spins, hvals] concatenes
# Pour l overlap croise: on garde les spins d un etat mais on injecte les hvals de l autre

@partial(jax.jit, static_argnums=(3,))
def compute_fidelity(variables, samples_base, samples_shift, n_spins):
    """
    F = |<psi_base|psi_shift>|^2 via MC.
    
    samples_base: echantillonnes depuis psi(h0)
    samples_shift: echantillonnes depuis psi(h0+delta)
    
    On construit les entrees croisees en swappant les hvals.
    """
    # Extraire spins et hvals
    spins_base = samples_base[..., :n_spins]
    hvals_base = samples_base[..., n_spins:]
    spins_shift = samples_shift[..., :n_spins]
    hvals_shift = samples_shift[..., n_spins:]
    
    # Entrees croisees: spins de base avec hvals shift, et vice versa
    cross_base_shift = jnp.concatenate([spins_base, hvals_shift], axis=-1)
    cross_shift_base = jnp.concatenate([spins_shift, hvals_base], axis=-1)
    
    # log psi evaluations
    log_psi_base_base = ma.apply(variables, samples_base)      # psi_base(s_base)
    log_psi_shift_base = ma.apply(variables, cross_base_shift)  # psi_shift(s_base)
    log_psi_shift_shift = ma.apply(variables, samples_shift)    # psi_shift(s_shift)
    log_psi_base_shift = ma.apply(variables, cross_shift_base)  # psi_base(s_shift)
    
    # <psi_shift|psi_base> / <psi_base|psi_base> via samples de psi_base
    ratio_12 = jnp.exp(log_psi_shift_base - log_psi_base_base)
    O_12 = jnp.mean(ratio_12)
    
    # <psi_base|psi_shift> / <psi_shift|psi_shift> via samples de psi_shift
    ratio_21 = jnp.exp(log_psi_base_shift - log_psi_shift_shift)
    O_21 = jnp.mean(ratio_21)
    
    # Fidelite = O_12 * O_21
    return jnp.real(O_12 * O_21)

# --- CALCUL ---
h0_list = np.linspace(args.h0_min, args.h0_max, args.h0_npts)
rng = np.random.default_rng(42)

chi_F_mean = []
chi_F_std = []

for h0 in tqdm(h0_list, desc="Fidelity susceptibility"):
    chi_reps = []
    
    for _ in range(N_REP):
        epsilon = rng.normal(loc=0.0, scale=sigma, size=n_spins)
        pars_base = jnp.array(h0 + epsilon)
        pars_shift = jnp.array(h0 + DELTA_H + epsilon)
        
        vs_base = vs.get_state(pars_base)
        vs_shift = vs.get_state(pars_shift)
        
        samples_base = vs_base.sample().reshape(-1, n_spins + ps.size)
        samples_shift = vs_shift.sample().reshape(-1, n_spins + ps.size)
        
        F = compute_fidelity(vs.variables, samples_base, samples_shift, n_spins)
        chi = float((1 - F) / (DELTA_H ** 2))
        chi_reps.append(chi)
    
    chi_F_mean.append(np.mean(chi_reps))
    chi_F_std.append(np.std(chi_reps))

chi_F_mean = np.array(chi_F_mean)
chi_F_std = np.array(chi_F_std)

# --- PLOT ---
plt.figure(figsize=(10, 6))
plt.errorbar(h0_list, chi_F_mean, yerr=chi_F_std, fmt="-o", color="red", capsize=3, markersize=4)
if IS_2D:
    plt.axvline(x=3.04, color="gray", ls="--", alpha=0.5, label=r"$h_c \approx 3.04$")
else:
    plt.axvline(x=1.0, color="gray", ls="--", alpha=0.5, label=r"$h_c = 1.0$")
plt.xlabel(r"Transverse field $h_0$", fontsize=13)
plt.ylabel(r"Fidelity susceptibility $\chi_F$", fontsize=13)
plt.title(f"Fidelity Susceptibility ($\sigma={sigma}$) — {'2D' if IS_2D else '1D'} L={L} ({n_spins} spins)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out = RUN_DIR / f"fidelity_susceptibility_L={L}.pdf"
plt.savefig(out, dpi=150)
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150)
print(f"Saved: {out}")
