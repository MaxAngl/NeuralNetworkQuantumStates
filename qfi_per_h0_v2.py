"""
QFI (Quantum Fisher Information) / Fidelity susceptibility per h0 pour FNQS.
Methode: derivee de log psi par differences finies sur les MEMES echantillons.

chi_F(h0) = 4 * Var_{sigma ~ |psi(h0)|^2} [ d log psi(sigma; h0) / dh0 ]

avec d log psi / dh0 ~ [log psi(sigma; h0+delta) - log psi(sigma; h0-delta)] / (2*delta)

Usage:
    python qfi_per_h0_v2.py <run_dir> [--delta 0.01] [--n-samples 4096] [--n-rep 10]
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
from functools import partial

import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=str)
parser.add_argument("--delta", type=float, default=0.01)
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
print(f"Delta for finite diff: {DELTA_H}")

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

# --- CALCUL chi_F PAR DERIVEE DE LOG PSI ---
@partial(jax.jit, static_argnums=(4,))
def compute_chi_F(variables, samples, hvals_plus, hvals_minus, n_spins, delta):
    """
    chi_F = 4 * Var[ d log psi / dh0 ]

    d log psi / dh0 approx [log psi(sigma; h+delta) - log psi(sigma; h-delta)] / (2*delta)

    Les echantillons sigma sont les MEMES — on change seulement les hvals injectes.
    """
    spins = samples[..., :n_spins]  # (n_samples, n_spins)

    # Construire les entrees avec h0+delta et h0-delta
    hvals_plus_broad = jnp.broadcast_to(hvals_plus, spins.shape)
    hvals_minus_broad = jnp.broadcast_to(hvals_minus, spins.shape)

    x_plus = jnp.concatenate([spins, hvals_plus_broad], axis=-1)
    x_minus = jnp.concatenate([spins, hvals_minus_broad], axis=-1)

    # Evaluer log psi aux deux points
    log_psi_plus = ma.apply(variables, x_plus)    # (n_samples,) complex
    log_psi_minus = ma.apply(variables, x_minus)   # (n_samples,) complex

    # Derivee par differences finies centrales
    dlog_psi = (log_psi_plus - log_psi_minus) / (2 * delta)

    # chi_F = 4 * Var[dlog_psi] = 4 * (E[|dlog|^2] - |E[dlog]|^2)
    mean_dlog = jnp.mean(dlog_psi)
    var_dlog = jnp.mean(jnp.abs(dlog_psi)**2) - jnp.abs(mean_dlog)**2

    return 4.0 * jnp.real(var_dlog)


# --- BOUCLE SUR h0 ---
h0_list = np.linspace(args.h0_min, args.h0_max, args.h0_npts)
rng = np.random.default_rng(42)

chi_F_mean = []
chi_F_std = []

for h0 in tqdm(h0_list, desc="QFI per h0"):
    chi_reps = []

    for _ in range(N_REP):
        # Tirage du desordre
        epsilon = rng.normal(loc=0.0, scale=sigma, size=n_spins)
        pars = jnp.array(h0 + epsilon)

        # Echantillonner depuis |psi(h0)|^2
        vs_h = vs.get_state(pars)
        samples = vs_h.sample().reshape(-1, n_spins + ps.size)

        # hvals shiftes
        hvals_plus = jnp.array(h0 + DELTA_H + epsilon)
        hvals_minus = jnp.array(h0 - DELTA_H + epsilon)

        chi = float(compute_chi_F(vs.variables, samples, hvals_plus, hvals_minus, n_spins, DELTA_H))
        chi_reps.append(chi)

    chi_F_mean.append(np.mean(chi_reps))
    chi_F_std.append(np.std(chi_reps) / np.sqrt(N_REP))

chi_F_mean = np.array(chi_F_mean)
chi_F_std = np.array(chi_F_std)

# --- SAUVEGARDE DONNEES ---
np.savez(RUN_DIR / f"qfi_data_L={L}.npz",
         h0=h0_list, chi_F_mean=chi_F_mean, chi_F_std=chi_F_std,
         delta=DELTA_H, n_samples=N_SAMPLES, n_rep=N_REP, sigma=sigma)

# --- PLOT ---
plt.figure(figsize=(10, 6))
plt.errorbar(h0_list, chi_F_mean, yerr=chi_F_std, fmt="-o", color="red", capsize=3, markersize=4)
if IS_2D:
    plt.axvline(x=3.04, color="gray", ls="--", alpha=0.5, label=r"$h_c \approx 3.04$")
else:
    plt.axvline(x=1.0, color="gray", ls="--", alpha=0.5, label=r"$h_c = 1.0$")
plt.xlabel(r"Transverse field $h_0$", fontsize=13)
plt.ylabel(r"Fidelity susceptibility $\chi_F$", fontsize=13)
plt.title(rf"QFI / Fidelity Susceptibility ($\sigma={sigma}$) — {'2D' if IS_2D else '1D'} L={L} ({n_spins} spins)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out = RUN_DIR / f"QFI_per_h0_L={L}.pdf"
plt.savefig(out, dpi=150)
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150)
print(f"Saved: {out}")
print(f"Data: {RUN_DIR / f'qfi_data_L={L}.npz'}")
