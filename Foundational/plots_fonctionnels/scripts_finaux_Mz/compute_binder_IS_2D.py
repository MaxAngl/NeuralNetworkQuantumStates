"""
Calcul du Binder cumulant via IS pour toutes les sigmas, 2D TFIM.

Pour chaque (h0, sigma, k) :
  - Réutilise les samples MCMC déjà enregistrés (is_samples_2D_L{L}_chunk*.npz)
  - Pour sigma=0   : IS weights = 1 (h_target = h_ref), calcul direct
  - Pour sigma>0   : génère N_DISORDER tirages h^(k) ~ |h0 + N(0,sigma)|,
                     calcule les poids IS via forward pass du modèle,
                     puis mz2, mz4, Binder pour chaque tirage

Sorties (dans le même dossier que le run) :
  binder_data_2D_L{L}.npz :
    h0_grid    : (n_h0,)
    sigma_grid : (n_sigma,)
    mz2_raw    : (n_sigma, n_h0, n_disorder)   — n_disorder=1 pour sigma=0
    mz4_raw    : (n_sigma, n_h0, n_disorder)
    binder_raw : (n_sigma, n_h0, n_disorder)

Usage:
    python compute_binder_IS_2D.py --L 8 [--ndisorder 100] [--run-dir PATH]
"""
import os
import sys
import glob
import json
import argparse
import time

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import zipfile
from tqdm import tqdm

PROJECT = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "Foundational"))
from flip_rules import GlobalFlipRule

# ==========================================
# ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--L",          type=int, required=True)
parser.add_argument("--ndisorder",  type=int, default=100)
parser.add_argument("--run-dir",    type=str, default=None)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

L          = args.L
N_DISORDER = args.ndisorder
SEED       = args.seed

BASE_2D = os.path.join(PROJECT, "Foundational/logs/Trains_autour_transi_2D")
RUN_DIR = args.run_dir or os.path.join(BASE_2D, f"L={L}")

SIGMA_GRID_FULL = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0])

print(f"=== Binder IS 2D — L={L}, N_DISORDER={N_DISORDER} ===")
print(f"Run dir : {RUN_DIR}")

# Charge les données existantes si dispo, et ne calcule que les sigmas manquants
out_path_check = os.path.join(RUN_DIR, f"binder_data_2D_L{L}.npz")
existing_data = {}
if os.path.exists(out_path_check):
    d = np.load(out_path_check)
    existing_sigmas = list(d["sigma_grid"])
    existing_data = dict(d)
    print(f"Données existantes : sigma={existing_sigmas}")
else:
    existing_sigmas = []

SIGMA_GRID = np.array([s for s in SIGMA_GRID_FULL if not any(abs(s - es) < 1e-9 for es in existing_sigmas)])
print(f"Sigmas à calculer  : {list(SIGMA_GRID)}")

# ==========================================
# CHARGEMENT DU MODELE
# ==========================================
meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path) as f:
    meta = json.load(f)

nb_spins  = meta["nb_spins"]
vit_p     = meta["vit_config"]
prob_flip = meta.get("sampler", {}).get("prob_global_flip", 0.05)

hi = nk.hilbert.Spin(0.5, nb_spins)
ps = nkf.ParameterSpace(N=nb_spins, min=0, max=50.0)

ma = ViTFNQS(
    num_layers=vit_p["num_layers"],
    d_model=vit_p["d_model"],
    heads=vit_p["heads"],
    b=vit_p["b"],
    L_eff=vit_p["L_eff"],
    n_coups=ps.size,
    complex=True,
    disorder=True,
    transl_invariant=False,
    two_dimensional=True,
)

sa = nk.sampler.MetropolisSampler(hi, rule=GlobalFlipRule(prob_flip), n_chains=1)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

# Dernier checkpoint
checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
last_ck = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
print(f"Checkpoint : {last_ck}")

if not zipfile.is_zipfile(last_ck):
    with open(last_ck, "rb") as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(last_ck, "r") as zf:
        candidates = [f for f in zf.namelist() if f.endswith(".msgpack")]
        target_file = sorted(candidates, key=len)[-1]
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

vars_dict = state_dict.get(
    "variables",
    state_dict.get("model", {}).get("variables",
    state_dict.get("vqs", {}).get("variables", state_dict))
)
vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
print("Poids chargés.")

nn_params = vs.parameters

def make_vars(h_disorder):
    return {
        "foundational": {"parameters": jnp.asarray(h_disorder, dtype=jnp.float32)},
        "params": nn_params,
    }

# Forward pass compilé une fois
_dummy_vars = make_vars(np.zeros(nb_spins))
_vs_init    = vs.get_state(np.zeros(nb_spins))
sa_tmp      = nk.sampler.MetropolisSampler(hi, rule=GlobalFlipRule(prob_flip), n_chains=1)
mc_tmp      = nk.vqs.MCState(sampler=sa_tmp, model=_vs_init.model,
                              variables=_dummy_vars, n_samples=1)
apply_fn    = jax.jit(mc_tmp.model.apply)

# Warm-up JIT
_ = apply_fn(_dummy_vars, jnp.zeros((1, nb_spins)))
print("JIT compilé.")

# ==========================================
# CHARGEMENT DES SAMPLES
# ==========================================
chunk_files = sorted(
    glob.glob(os.path.join(RUN_DIR, f"is_samples_2D_L{L}_chunk*.npz")),
    key=lambda fn: int(fn.split("chunk")[-1].replace(".npz", ""))
)
if not chunk_files:
    print("ERREUR : aucun fichier is_samples trouvé.")
    sys.exit(1)

samples_by_h0 = {}   # idx_h0 -> (h0_val, samples (N,nb_spins), log_psi_ref (N,))
for fn in chunk_files:
    f = np.load(fn)
    h_ref_arr  = f["h_ref"]       # (n_h0_chunk, nb_spins)
    samp_arr   = f["samples"]     # (n_h0_chunk, N_samples, nb_spins)
    lpsi_arr   = f["log_psi_ref"] # (n_h0_chunk, N_samples)
    h0_idx_arr = f["h0_indices"]  # (n_h0_chunk,)
    h0_grid    = f["h0_grid"]     # (n_h0_total,)
    for j in range(len(h0_idx_arr)):
        idx = int(h0_idx_arr[j])
        if idx not in samples_by_h0:
            h0_val = float(h_ref_arr[j, 0])
            samples_by_h0[idx] = (h0_val, samp_arr[j], lpsi_arr[j])

h0_indices_sorted = sorted(samples_by_h0.keys())
h0_grid_out = np.array([samples_by_h0[i][0] for i in h0_indices_sorted])
n_h0        = len(h0_indices_sorted)
n_sigma     = len(SIGMA_GRID)
print(f"Samples chargés : {n_h0} points h0, h0 ∈ [{h0_grid_out.min():.3f}, {h0_grid_out.max():.3f}]")

# ==========================================
# TABLEAUX DE SORTIE
# ==========================================
# sigma=0 : n_disorder_eff = 1  (résultat déterministe)
# sigma>0 : n_disorder_eff = N_DISORDER
n_dis_sigma0 = 1
mz2_raw    = np.full((n_sigma, n_h0, N_DISORDER), np.nan)
mz4_raw    = np.full((n_sigma, n_h0, N_DISORDER), np.nan)
binder_raw = np.full((n_sigma, n_h0, N_DISORDER), np.nan)
ess_raw    = np.full((n_sigma, n_h0, N_DISORDER), np.nan)  # ESS/N_samples ∈ [0,1]

# Pour sigma=0, on n'a qu'une seule réalisation, on la met dans la case 0
mz2_sigma0    = np.full((n_h0,), np.nan)
mz4_sigma0    = np.full((n_h0,), np.nan)
binder_sigma0 = np.full((n_h0,), np.nan)

rng = np.random.default_rng(SEED)

# ==========================================
# BOUCLE PRINCIPALE
# ==========================================
t0 = time.time()

for i_h, idx_h in enumerate(tqdm(h0_indices_sorted, desc=f"h0 (L={L})")):
    h0_val, samples, log_psi_ref = samples_by_h0[idx_h]
    # samples : (N_samples, nb_spins), log_psi_ref : (N_samples,)
    samples_j = jnp.asarray(samples, dtype=jnp.float32)
    lpsi_ref_j = jnp.asarray(log_psi_ref, dtype=jnp.float32)

    # Observables par sample (indépendant de h)
    mz_per_sample  = jnp.mean(samples_j, axis=1)   # (N,)
    mz2_per_sample = mz_per_sample ** 2             # (N,)
    mz4_per_sample = mz_per_sample ** 4             # (N,)

    for i_s, sigma in enumerate(SIGMA_GRID):

        if sigma == 0.0:
            # Pas de repondération : IS weights = 1
            N_samp     = mz2_per_sample.shape[0]
            mz2_val    = float(jnp.mean(mz2_per_sample))
            mz4_val    = float(jnp.mean(mz4_per_sample))
            binder_val = 1.0 - mz4_val / (3.0 * mz2_val ** 2)
            # Stocké dans la case k=0 uniquement
            mz2_raw[i_s, i_h, 0]    = mz2_val
            mz4_raw[i_s, i_h, 0]    = mz4_val
            binder_raw[i_s, i_h, 0] = binder_val
            ess_raw[i_s, i_h, 0]    = 1.0   # poids uniformes → ESS = N

        else:
            # Génère N_DISORDER tirages de désordre
            N_samp = mz2_per_sample.shape[0]
            noise = rng.normal(loc=0.0, scale=sigma, size=(N_DISORDER, nb_spins))
            h_disorders = np.abs(h0_val + noise)   # (N_DISORDER, nb_spins)

            for k in range(N_DISORDER):
                target_vars = make_vars(h_disorders[k])
                log_psi_tgt = apply_fn(target_vars, samples_j)  # (N,) complex

                # Poids IS : |ψ(h_target)|² / |ψ(h_ref)|²
                log_w = 2.0 * (jnp.real(log_psi_tgt) - lpsi_ref_j)
                log_w = log_w - jnp.max(log_w)   # stabilisation
                w = jnp.exp(log_w)
                w_norm = w / jnp.sum(w)

                # ESS = 1/Σw_norm² normalisé par N → ∈ [1/N, 1]
                ess_ratio = float(1.0 / (N_samp * jnp.sum(w_norm ** 2)))

                mz2_val    = float(jnp.sum(w_norm * mz2_per_sample))
                mz4_val    = float(jnp.sum(w_norm * mz4_per_sample))
                binder_val = 1.0 - mz4_val / (3.0 * mz2_val ** 2)

                mz2_raw[i_s, i_h, k]    = mz2_val
                mz4_raw[i_s, i_h, k]    = mz4_val
                binder_raw[i_s, i_h, k] = binder_val
                ess_raw[i_s, i_h, k]    = ess_ratio

elapsed = time.time() - t0
print(f"\nTemps total : {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Pour sigma=0 (si calculé maintenant), répliquer dans toutes les colonnes k>0
if len(SIGMA_GRID) > 0 and SIGMA_GRID[0] == 0.0:
    for k in range(1, N_DISORDER):
        mz2_raw[0, :, k]    = mz2_raw[0, :, 0]
        mz4_raw[0, :, k]    = mz4_raw[0, :, 0]
        binder_raw[0, :, k] = binder_raw[0, :, 0]
        ess_raw[0, :, k]    = ess_raw[0, :, 0]

# Affichage ESS pour les grandes sigmas
for i_s, sigma in enumerate(SIGMA_GRID):
    if sigma >= 0.7:
        ess_mean = np.nanmean(ess_raw[i_s])
        ess_min  = np.nanmin(ess_raw[i_s])
        print(f"  σ={sigma:.1f} : ESS/N moyen={ess_mean:.3f}, min={ess_min:.3f}"
              f"  ({'OK' if ess_mean > 0.05 else 'ATTENTION bas'})")

# ==========================================
# FUSION AVEC LES DONNÉES EXISTANTES
# ==========================================
if existing_data:
    old_sigma_grid = existing_data["sigma_grid"]
    old_mz2    = existing_data["mz2_raw"]
    old_mz4    = existing_data["mz4_raw"]
    old_binder = existing_data["binder_raw"]
    old_ess    = existing_data.get("ess_raw", np.ones_like(old_binder))  # compat anciens fichiers

    # Concatène et retrie par sigma
    merged_sigma  = np.concatenate([old_sigma_grid, SIGMA_GRID])
    merged_mz2    = np.concatenate([old_mz2,    mz2_raw],    axis=0)
    merged_mz4    = np.concatenate([old_mz4,    mz4_raw],    axis=0)
    merged_binder = np.concatenate([old_binder, binder_raw], axis=0)
    merged_ess    = np.concatenate([old_ess,    ess_raw],    axis=0)

    sort_idx      = np.argsort(merged_sigma)
    sigma_grid_out  = merged_sigma[sort_idx]
    mz2_out         = merged_mz2[sort_idx]
    mz4_out         = merged_mz4[sort_idx]
    binder_out      = merged_binder[sort_idx]
    ess_out         = merged_ess[sort_idx]
    print(f"Fusion : {list(old_sigma_grid)} + {list(SIGMA_GRID)} → {list(sigma_grid_out)}")
else:
    sigma_grid_out = SIGMA_GRID
    mz2_out, mz4_out, binder_out, ess_out = mz2_raw, mz4_raw, binder_raw, ess_raw

# ==========================================
# SAUVEGARDE
# ==========================================
out_path = os.path.join(RUN_DIR, f"binder_data_2D_L{L}.npz")
np.savez(
    out_path,
    h0_grid        = h0_grid_out,
    sigma_grid     = sigma_grid_out,
    mz2_raw        = mz2_out,      # (n_sigma, n_h0, N_DISORDER)
    mz4_raw        = mz4_out,
    binder_raw     = binder_out,
    ess_raw        = ess_out,      # ESS/N_samples ∈ [0,1]
    N_DISORDER     = N_DISORDER,
    L              = L,
    nb_spins       = nb_spins,
)
print(f"Sauvegardé : {out_path}")
print(f"  sigma_grid : {list(sigma_grid_out)}")
print(f"  mz2_raw shape : {mz2_out.shape}")
print(f"  Taille fichier estimée : {3 * mz2_out.nbytes / 1e6:.1f} MB")
