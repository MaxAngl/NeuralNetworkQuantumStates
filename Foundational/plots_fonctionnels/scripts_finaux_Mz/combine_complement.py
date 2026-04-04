"""
Combine un full.npz (grille originale) avec les chunks d'une grille complementaire.

Usage:
    python combine_complement.py --dim 1 --L 64 --ncomp 9
    python combine_complement.py --run-dir PATH --ncomp 9 --prefix is_data_1D_L64

Le script:
  1. Charge {prefix}_full.npz  (grille originale, N_h0_orig points)
  2. Charge {prefix}_chunk0..chunk{ncomp-1}.npz  (grille complementaire, 1 point chacun)
  3. Fusionne en une grille triee de N_h0_orig + N_h0_comp points
  4. Ecrase {prefix}_full.npz avec le resultat combine
  5. Idem pour {samples_prefix}_full.npz si present
"""
import os, sys, argparse
import numpy as np

PROJECT_ROOT = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
RUN_DIR_1D = os.path.join(PROJECT_ROOT, "Foundational/logs/Trains_autour_transi_1D/run_L={L}")

parser = argparse.ArgumentParser()
parser.add_argument("--dim",     type=int, default=None, choices=[1, 2])
parser.add_argument("--L",       type=int, default=None)
parser.add_argument("--ncomp",   type=int, required=True, help="Nombre de chunks complementaires")
parser.add_argument("--run-dir", type=str, default=None)
parser.add_argument("--prefix",  type=str, default=None)
args = parser.parse_args()

if args.run_dir:
    RUN_DIR = args.run_dir
elif args.dim == 1 and args.L:
    RUN_DIR = os.path.join(PROJECT_ROOT, f"Foundational/logs/Trains_autour_transi_1D/L={args.L}")
else:
    print("Erreur: specifier --dim+--L ou --run-dir"); sys.exit(1)

if args.prefix:
    prefix = args.prefix
elif args.dim and args.L:
    prefix = f"is_data_{args.dim}D_L{args.L}"
else:
    print("Erreur: specifier --prefix ou --dim+--L"); sys.exit(1)

# ── Charger le full.npz original ──────────────────────────────────────────
full_path = os.path.join(RUN_DIR, f"{prefix}_full.npz")
print(f"Chargement original: {full_path}")
orig = np.load(full_path)
h0_orig   = orig["h0_grid"]            # (N_orig,)
mz2_orig  = orig["mz2_raw"]            # (n_sigma, N_orig, N_d)
ess_orig  = orig["ess_raw"]
err_orig  = orig["mz2_err_raw"] if "mz2_err_raw" in orig else None
time_orig = orig["time_mcmc_ref"] if "time_mcmc_ref" in orig else None
trial_orig = orig["trial_chosen"] if "trial_chosen" in orig else None
sigma_grid = orig["sigma_grid"]
N_orig = len(h0_orig)
print(f"  {N_orig} points originaux, sigma_grid={sigma_grid}")

# ── Charger les chunks complementaires ────────────────────────────────────
h0_comp_list   = []
mz2_comp_list  = []
ess_comp_list  = []
err_comp_list  = []
time_comp_list = []
trial_comp_list = []

for c in range(args.ncomp):
    path = os.path.join(RUN_DIR, f"{prefix}_chunk{c}.npz")
    if not os.path.exists(path):
        print(f"  ATTENTION: chunk complementaire {c} manquant ({path})")
        continue
    chunk = np.load(path)
    # Chaque chunk a h0_grid = grille complementaire complete, mz2_raw non-NaN uniquement pour son index
    h0_grid_c = chunk["h0_grid"]
    mz2_c     = chunk["mz2_raw"]       # (n_sigma, N_comp, N_d)

    # Trouver les indices non-NaN (les points calcules par ce chunk)
    valid_mask = ~np.isnan(mz2_c[0, :, 0])
    valid_idxs = np.where(valid_mask)[0]

    for idx in valid_idxs:
        h0_val = float(h0_grid_c[idx])
        # Verifier qu'il n'est pas deja dans la grille originale
        if np.any(np.abs(h0_orig - h0_val) < 1e-8):
            print(f"  h0={h0_val:.6f} deja dans grille originale, ignore")
            continue
        h0_comp_list.append(h0_val)
        mz2_comp_list.append(mz2_c[:, idx, :])   # (n_sigma, N_d)
        ess_comp_list.append(chunk["ess_raw"][:, idx, :])
        if "mz2_err_raw" in chunk:
            err_comp_list.append(chunk["mz2_err_raw"][:, idx, :])
        if "time_mcmc_ref" in chunk:
            time_comp_list.append(chunk["time_mcmc_ref"][idx] if chunk["time_mcmc_ref"].ndim > 0 else float(chunk["time_mcmc_ref"]))
        if "trial_chosen" in chunk:
            trial_comp_list.append(chunk["trial_chosen"][idx] if chunk["trial_chosen"].ndim > 0 else int(chunk["trial_chosen"]))

N_comp = len(h0_comp_list)
print(f"  {N_comp} nouveaux points complementaires charges")

if N_comp == 0:
    print("Aucun nouveau point, rien a faire.")
    sys.exit(0)

# ── Construire la grille combinee triee ───────────────────────────────────
h0_all = np.concatenate([h0_orig, h0_comp_list])
sort_idx = np.argsort(h0_all)
h0_combined = h0_all[sort_idx]

n_sigma, _, N_d = mz2_orig.shape
N_total = N_orig + N_comp

mz2_comb  = np.full((n_sigma, N_total, N_d), np.nan)
ess_comb  = np.full((n_sigma, N_total, N_d), np.nan)
err_comb  = np.full((n_sigma, N_total, N_d), np.nan) if err_orig is not None or err_comp_list else None
time_comb = np.full(N_total, np.nan) if time_orig is not None else None
trial_comb = np.full(N_total, -1, dtype=int) if trial_orig is not None else None

# Remplir originaux
for i_new, i_old in enumerate(sort_idx):
    if i_old < N_orig:
        mz2_comb[:, i_new, :] = mz2_orig[:, i_old, :]
        ess_comb[:, i_new, :] = ess_orig[:, i_old, :]
        if err_comb is not None and err_orig is not None:
            err_comb[:, i_new, :] = err_orig[:, i_old, :]
        if time_comb is not None and time_orig is not None:
            time_comb[i_new] = time_orig[i_old] if time_orig.ndim > 0 else float(time_orig)
        if trial_comb is not None and trial_orig is not None:
            trial_comb[i_new] = trial_orig[i_old] if trial_orig.ndim > 0 else int(trial_orig)
    else:
        j = i_old - N_orig
        mz2_comb[:, i_new, :] = mz2_comp_list[j]
        ess_comb[:, i_new, :] = ess_comp_list[j]
        if err_comb is not None and j < len(err_comp_list):
            err_comb[:, i_new, :] = err_comp_list[j]
        if time_comb is not None and j < len(time_comp_list):
            time_comb[i_new] = time_comp_list[j]
        if trial_comb is not None and j < len(trial_comp_list):
            trial_comb[i_new] = trial_comp_list[j]

# ── Sauvegarder ──────────────────────────────────────────────────────────
save_dict = dict(
    h0_grid=h0_combined,
    sigma_grid=sigma_grid,
    mz2_raw=mz2_comb,
    ess_raw=ess_comb,
    N_SAMPLES_IS=int(orig["N_SAMPLES_IS"]),
    n_chains_is=int(orig["n_chains_is"]),
    N_DISORDER=int(orig["N_DISORDER"]),
    N_MCMC_TRIALS=int(orig["N_MCMC_TRIALS"]),
    L=int(orig["L"]),
)
if err_comb is not None:
    save_dict["mz2_err_raw"] = err_comb
if time_comb is not None:
    save_dict["time_mcmc_ref"] = time_comb
if trial_comb is not None:
    save_dict["trial_chosen"] = trial_comb
for key in ["nb_spins", "dim"]:
    if key in orig:
        save_dict[key] = int(orig[key])

np.savez(full_path, **save_dict)
print(f"\nCombine: {full_path}")
print(f"  {N_orig} + {N_comp} = {N_total} points, h0 in [{h0_combined[0]:.4f}, {h0_combined[-1]:.4f}]")
print(f"  NaN restants dans mz2_raw[:,:,0]: {np.sum(np.isnan(mz2_comb[:, :, 0]))}")

# ── Combiner les samples ───────────────────────────────────────────────────
samples_prefix = prefix.replace("is_data_", "is_samples_")
samples_full_path = os.path.join(RUN_DIR, f"{samples_prefix}_full.npz")

if not os.path.exists(samples_full_path):
    print("\nPas de fichier samples original, skip.")
    sys.exit(0)

print(f"\nCombining samples: {samples_full_path}")
s_orig = np.load(samples_full_path)
N_S = int(s_orig["N_SAMPLES_IS"])
nb_spins = int(s_orig["nb_spins"])

samples_comb     = np.zeros((N_total, N_S, nb_spins), dtype=np.float32)
log_psi_comb     = np.zeros((N_total, N_S), dtype=np.float32)
h_ref_comb       = np.zeros((N_total, nb_spins), dtype=np.float32)

# Remplir originaux
for i_new, i_old in enumerate(sort_idx):
    if i_old < N_orig:
        samples_comb[i_new]  = s_orig["samples"][i_old]
        log_psi_comb[i_new]  = s_orig["log_psi_ref"][i_old]
        h_ref_comb[i_new]    = s_orig["h_ref"][i_old]

# Remplir complementaires depuis leurs chunks
for c in range(args.ncomp):
    path_s = os.path.join(RUN_DIR, f"{samples_prefix}_chunk{c}.npz")
    if not os.path.exists(path_s):
        continue
    cs = np.load(path_s)
    h0_grid_c = cs["h0_grid"]
    # h0_indices dans la grille complementaire
    comp_indices = cs["h0_indices"]
    for local_i, ci in enumerate(comp_indices):
        h0_val = float(h0_grid_c[ci])
        pos = np.where(np.abs(h0_combined - h0_val) < 1e-8)[0]
        if len(pos) == 0:
            continue
        i_new = pos[0]
        N_S_chunk = cs["samples"].shape[1]
        n = min(N_S, N_S_chunk)
        samples_comb[i_new, :n]  = cs["samples"][local_i, :n]
        log_psi_comb[i_new, :n]  = cs["log_psi_ref"][local_i, :n]
        h_ref_comb[i_new]         = cs["h_ref"][local_i]

np.savez(
    samples_full_path,
    samples=samples_comb,
    log_psi_ref=log_psi_comb,
    h_ref=h_ref_comb,
    h0_grid=h0_combined,
    N_SAMPLES_IS=N_S,
    L=int(s_orig["L"]),
    nb_spins=nb_spins,
    dim=int(s_orig["dim"]),
)
size_mb = os.path.getsize(samples_full_path) / 1e6
print(f"Samples combines: {samples_full_path} ({size_mb:.1f} MB)")
