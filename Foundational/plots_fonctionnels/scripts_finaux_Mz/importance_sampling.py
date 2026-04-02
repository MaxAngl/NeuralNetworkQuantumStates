"""
Script unifie d'importance sampling pour FNQS 1D et 2D.

Usage:
    python importance_sampling.py --dim <1|2> --L <taille> [--chunk CHUNK_ID] [--nchunks N_CHUNKS]
                                  [--nsamples N] [--nchains N] [--burnin N] [--ndisorder N] [--ntrials N]
                                  [--run-dir PATH] [--h0-grid 1d|2d|custom]

Exemples:
    # 1D, L=49, pas de chunking
    python importance_sampling.py --dim 1 --L 49

    # 2D, L=8, chunk 3 sur 20
    python importance_sampling.py --dim 2 --L 8 --chunk 3 --nchunks 20

    # 1D, L=25, burn-in court
    python importance_sampling.py --dim 1 --L 25 --burnin 100 --chunk 0 --nchunks 20
"""
import os
import sys
import time
import gc
import argparse

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import glob
import json
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import zipfile
from tqdm import tqdm

# ==========================================
# GRILLES H0 PRE-DEFINIES
# ==========================================
H0_GRID_1D = sorted(set(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # loin de la transition
    + [round(0.7 + i * 0.02, 3) for i in range(31)]  # [0.7, 1.3] pas de 0.02 = 31 pts
    + [1.4, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0]  # loin de la transition
))

H0_GRID_2D = [
    0.0, 0.2, 0.5, 0.8, 1.0, 1.3, 1.5,
    1.7, 1.8, 1.9, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45,
    2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95,
    3.0, 3.05, 3.1, 3.15, 3.2, 3.3, 3.4, 3.5,
    4.0, 5.0, 6.0
]

SIGMA_TEST_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# ==========================================
# CHEMINS PAR DEFAUT
# ==========================================
PROJECT_ROOT = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
RUN_DIR_1D = os.path.join(PROJECT_ROOT, "Foundational/logs/Trains_finaux_disordered_1D/run_L={L}")
RUN_DIR_2D = os.path.join(PROJECT_ROOT, "Foundational/rami_perso/2D_FNQS/Run_2D_L{L}_FNQS")

# ==========================================
# PARSING ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(description="Importance Sampling pour FNQS 1D/2D")
parser.add_argument("--dim", type=int, required=True, choices=[1, 2], help="Dimension (1 ou 2)")
parser.add_argument("--L", type=int, required=True, help="Taille lineaire du systeme")
parser.add_argument("--chunk", type=int, default=0, help="Index du chunk (defaut: 0)")
parser.add_argument("--nchunks", type=int, default=1, help="Nombre total de chunks (defaut: 1)")
parser.add_argument("--nsamples", type=int, default=16384, help="Nombre de samples IS (defaut: 16384)")
parser.add_argument("--nchains", type=int, default=256, help="Nombre de chaines MCMC (defaut: 256)")
parser.add_argument("--burnin", type=int, default=None, help="Burn-in par chaine (defaut: auto)")
parser.add_argument("--ndisorder", type=int, default=80, help="Nombre de realisations de desordre (defaut: 80)")
parser.add_argument("--ntrials", type=int, default=3, help="Nombre de tirages MCMC best-of (defaut: 3)")
parser.add_argument("--run-dir", type=str, default=None, help="Chemin du run (defaut: auto)")
parser.add_argument("--h0-grid", type=str, default=None, help="Grille h0: '1d', '2d', ou chemin vers fichier")
parser.add_argument("--prob-flip", type=float, default=None, help="Probabilite global flip (defaut: depuis meta.json ou 0.08)")
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================
DIM = args.dim
L = args.L
CHUNK_ID = args.chunk
N_CHUNKS = args.nchunks
N_SAMPLES_IS = args.nsamples
n_chains_is = args.nchains
N_DISORDER = args.ndisorder
N_MCMC_TRIALS = args.ntrials

# Run directory
if args.run_dir:
    RUN_DIR = args.run_dir
elif DIM == 1:
    RUN_DIR = RUN_DIR_1D.format(L=L)
else:
    RUN_DIR = RUN_DIR_2D.format(L=L)

# Grille h0
if args.h0_grid == "1d" or (args.h0_grid is None and DIM == 1):
    H0_TEST_LIST = H0_GRID_1D
elif args.h0_grid == "2d" or (args.h0_grid is None and DIM == 2):
    H0_TEST_LIST = H0_GRID_2D
else:
    H0_TEST_LIST = list(np.loadtxt(args.h0_grid))

# ==========================================
# SETUP ET CHARGEMENT
# ==========================================
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Foundational"))
from flip_rules import GlobalFlipRule

class SafeGlobalFlipRule(GlobalFlipRule):
    def transition(self, sampler, machine, parameters, state, key, sigma):
        sigma_new, log_prob = super().transition(sampler, machine, parameters, state, key, sigma)
        return jnp.asarray(sigma_new, dtype=sigma.dtype), log_prob

meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

if DIM == 2:
    nb_spins = meta["nb_spins"]  # L*L
else:
    nb_spins = meta.get("nb_spins", meta["L"])  # En 1D, nb_spins = L

vit_params = meta["vit_config"]

# Prob global flip
if args.prob_flip is not None:
    prob_global_flip = args.prob_flip
else:
    prob_global_flip = meta.get("sampler", {}).get("prob_global_flip", 0.08)

# Burn-in auto
if args.burnin is not None:
    n_discard_per_chain_is = args.burnin
elif DIM == 2:
    n_discard_per_chain_is = 100 if L <= 5 else 300
else:
    n_discard_per_chain_is = 100 if nb_spins <= 36 else 300

print(f"dim={DIM}D, L={L}, nb_spins={nb_spins}, burn_in={n_discard_per_chain_is}")
print(f"RUN_DIR: {RUN_DIR}")

hi = nk.hilbert.Spin(0.5, nb_spins)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=50.0)

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
    two_dimensional=(DIM == 2),
)

sa = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=1)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

# Chargement du checkpoint (dernier disponible)
checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
print(f"Checkpoint: {last_checkpoint}")

if not zipfile.is_zipfile(last_checkpoint):
    with open(last_checkpoint, 'rb') as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(last_checkpoint, 'r') as zf:
        candidates = [f for f in zf.namelist() if f.endswith('.msgpack')]
        target_file = sorted(candidates, key=len)[-1]
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

vars_dict = state_dict.get('variables', state_dict.get('model', {}).get('variables', state_dict.get('vqs', {}).get('variables', state_dict)))
vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
print("Poids charges.")

# ==========================================
# Helper
# ==========================================
_nn_params = vs.parameters

def _make_vars(pars):
    return {
        "foundational": {"parameters": jnp.asarray(pars)},
        "params": _nn_params,
    }

# ==========================================
# IMPORTANCE SAMPLING — BEST OF N_MCMC_TRIALS
# ==========================================
_init_vars = _make_vars(np.zeros(nb_spins))
_vs_init = vs.get_state(np.zeros(nb_spins))

sa_is = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=n_chains_is)

mc_is = nk.vqs.MCState(
    sampler=sa_is,
    model=_vs_init.model,
    variables=_init_vars,
    n_samples=N_SAMPLES_IS,
    n_discard_per_chain=n_discard_per_chain_is,
)

_apply_fn = jax.jit(mc_is.model.apply)

# ==========================================
# BOUCLE PRINCIPALE
# ==========================================
all_h0_indices = list(range(len(H0_TEST_LIST)))
chunk_indices = np.array_split(all_h0_indices, N_CHUNKS)[CHUNK_ID].tolist()
print(f"\nChunk {CHUNK_ID}/{N_CHUNKS}: h0 indices {chunk_indices[0]}-{chunk_indices[-1]} ({len(chunk_indices)} points)")

dim_label = f"{DIM}D"
print("\n" + "="*70)
print(f"IMPORTANCE SAMPLING {dim_label} (L={L}, {nb_spins} spins) -- Best of {N_MCMC_TRIALS} MCMC trials")
print(f"IS config: {N_SAMPLES_IS} samples, {n_chains_is} chaines, {n_discard_per_chain_is} burn-in")
print(f"{len(chunk_indices)} h0 (chunk {CHUNK_ID}/{N_CHUNKS}) x {len(SIGMA_TEST_LIST)} sigma x {N_DISORDER} disorder configs")
print("="*70)

mz2_raw = np.full((len(SIGMA_TEST_LIST), len(H0_TEST_LIST), N_DISORDER), np.nan)
ess_raw = np.full((len(SIGMA_TEST_LIST), len(H0_TEST_LIST), N_DISORDER), np.nan)
time_mcmc_ref_array = np.full(len(H0_TEST_LIST), np.nan)
trial_chosen = np.full(len(H0_TEST_LIST), -1, dtype=int)

# Pre-generer le bruit
rng_dict = {}
for idx_s, sigma in enumerate(SIGMA_TEST_LIST):
    if sigma > 0:
        rng = np.random.default_rng(seed=42 + idx_s)
        rng_dict[idx_s] = rng.normal(loc=0.0, scale=sigma, size=(N_DISORDER, nb_spins))

t_total_start = time.time()

for idx_h in tqdm(chunk_indices, desc=f"h0 sweep (chunk {CHUNK_ID})"):
    h0 = H0_TEST_LIST[idx_h]
    pars_ref = np.full(nb_spins, h0)
    ref_vars = _make_vars(pars_ref)

    best_ess_mean = -1.0
    best_trial_data = None

    t0_mcmc = time.time()
    for trial in range(N_MCMC_TRIALS):
        mc_is.variables = ref_vars
        mc_is.reset()
        mc_is.sample()
        samples_flat = mc_is.samples.reshape(-1, nb_spins)
        log_psi_ref = _apply_fn(ref_vars, samples_flat)
        mz_vals = jnp.mean(samples_flat, axis=1)
        mz2_vals = mz_vals ** 2

        ess_check_list = []
        for check_s_idx, check_sigma in [(2, 0.1), (4, 0.2)]:
            if check_s_idx < len(SIGMA_TEST_LIST):
                for kk in range(min(5, N_DISORDER)):
                    check_pars = np.abs(h0 + rng_dict[check_s_idx][kk])
                    check_vars = _make_vars(check_pars)
                    log_psi_t = _apply_fn(check_vars, samples_flat)
                    lw = 2.0 * jnp.real(log_psi_t - log_psi_ref)
                    lw = lw - jnp.max(lw)
                    w = jnp.exp(lw)
                    ess_check_list.append(float(jnp.sum(w) ** 2 / jnp.sum(w ** 2)))

        mean_ess_trial = np.mean(ess_check_list) if ess_check_list else N_SAMPLES_IS

        if mean_ess_trial > best_ess_mean:
            best_ess_mean = mean_ess_trial
            best_trial_data = (samples_flat, log_psi_ref, mz_vals, mz2_vals)
            trial_chosen[idx_h] = trial

    t_mcmc = time.time() - t0_mcmc
    time_mcmc_ref_array[idx_h] = t_mcmc

    samples_flat, log_psi_ref, mz_vals, mz2_vals = best_trial_data
    print(f"  h0={h0:.3f}: best trial={trial_chosen[idx_h]}, ESS check={best_ess_mean:.0f}", flush=True)

    for idx_s, sigma in enumerate(SIGMA_TEST_LIST):
        if sigma == 0.0:
            configs = [np.full(nb_spins, h0)]
        else:
            configs = [np.abs(h0 + rng_dict[idx_s][k]) for k in range(N_DISORDER)]

        for k, pars in enumerate(configs):
            target_vars = _make_vars(pars)
            log_psi_target = _apply_fn(target_vars, samples_flat)
            log_weights = 2.0 * jnp.real(log_psi_target - log_psi_ref)
            log_weights = log_weights - jnp.max(log_weights)
            weights = jnp.exp(log_weights)
            mz2_raw[idx_s, idx_h, k] = float(jnp.sum(weights * mz2_vals) / jnp.sum(weights))
            ess_raw[idx_s, idx_h, k] = float(jnp.sum(weights) ** 2 / jnp.sum(weights ** 2))

    gc.collect()

t_total = time.time() - t_total_start

# ==========================================
# SAUVEGARDE
# ==========================================
prefix = f"is_data_{dim_label}_L{L}"
if N_CHUNKS > 1:
    output_path = os.path.join(RUN_DIR, f"{prefix}_chunk{CHUNK_ID}.npz")
else:
    output_path = os.path.join(RUN_DIR, f"{prefix}_full.npz")

np.savez(
    output_path,
    h0_grid=np.array(H0_TEST_LIST),
    sigma_grid=np.array(SIGMA_TEST_LIST),
    mz2_raw=mz2_raw,
    ess_raw=ess_raw,
    time_mcmc_ref=time_mcmc_ref_array,
    trial_chosen=trial_chosen,
    N_SAMPLES_IS=N_SAMPLES_IS,
    n_chains_is=n_chains_is,
    N_DISORDER=N_DISORDER,
    N_MCMC_TRIALS=N_MCMC_TRIALS,
    L=L,
    nb_spins=nb_spins,
    dim=DIM,
)
print(f"\nDonnees sauvegardees: {output_path}")

# ==========================================
# RESUME
# ==========================================
print("\n" + "="*70)
print(f"Temps total: {t_total:.1f}s")
print(f"ESS moyen: {np.nanmean(ess_raw):.0f} / {N_SAMPLES_IS}")
print(f"ESS min:   {np.nanmin(ess_raw):.0f}")
print("="*70)
for idx_s, sigma in enumerate(SIGMA_TEST_LIST):
    print(f"  sigma={sigma:.2f} : ESS moyen={np.nanmean(ess_raw[idx_s]):.0f}, min={np.nanmin(ess_raw[idx_s]):.0f}")
