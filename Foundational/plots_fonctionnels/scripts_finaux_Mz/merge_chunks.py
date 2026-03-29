"""
Script unifie de fusion des chunks d'importance sampling.

Usage:
    python merge_chunks.py --dim <1|2> --L <taille> --nchunks <N>
                           [--run-dir PATH] [--prefix PREFIX]

Exemples:
    python merge_chunks.py --dim 2 --L 8 --nchunks 20
    python merge_chunks.py --dim 1 --L 49 --nchunks 20
    python merge_chunks.py --run-dir /path/to/run --prefix is_data_2D_L8 --nchunks 20
"""
import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = r"/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates"
RUN_DIR_1D = os.path.join(PROJECT_ROOT, "Foundational/logs/Trains_finaux_disordered_1D/run_L={L}")
RUN_DIR_2D = os.path.join(PROJECT_ROOT, "Foundational/rami_perso/2D_FNQS/Run_2D_L{L}_FNQS")

parser = argparse.ArgumentParser(description="Fusion des chunks IS")
parser.add_argument("--dim", type=int, default=None, choices=[1, 2], help="Dimension")
parser.add_argument("--L", type=int, default=None, help="Taille lineaire")
parser.add_argument("--nchunks", type=int, required=True, help="Nombre de chunks a fusionner")
parser.add_argument("--run-dir", type=str, default=None, help="Chemin du run")
parser.add_argument("--prefix", type=str, default=None, help="Prefixe des fichiers (ex: is_data_2D_L8)")
args = parser.parse_args()

# Determiner le run directory et le prefixe
if args.run_dir:
    RUN_DIR = args.run_dir
elif args.dim == 1:
    RUN_DIR = RUN_DIR_1D.format(L=args.L)
elif args.dim == 2:
    RUN_DIR = RUN_DIR_2D.format(L=args.L)
else:
    print("Erreur: specifier --dim ou --run-dir"); sys.exit(1)

if args.prefix:
    prefix = args.prefix
elif args.dim and args.L:
    prefix = f"is_data_{args.dim}D_L{args.L}"
else:
    print("Erreur: specifier --prefix ou --dim + --L"); sys.exit(1)

N_CHUNKS = args.nchunks

# Charger le premier chunk
first_path = os.path.join(RUN_DIR, f"{prefix}_chunk0.npz")
print(f"Chargement: {first_path}")
first = np.load(first_path)

mz2_raw = first["mz2_raw"].copy()
ess_raw = first["ess_raw"].copy()
time_mcmc = first["time_mcmc_ref"].copy()
trial_chosen = first["trial_chosen"].copy()

# Fusionner les autres chunks
for i in range(1, N_CHUNKS):
    path = os.path.join(RUN_DIR, f"{prefix}_chunk{i}.npz")
    if not os.path.exists(path):
        print(f"ATTENTION: {path} manquant!")
        continue
    chunk = np.load(path)
    for s in range(mz2_raw.shape[0]):
        for h in range(mz2_raw.shape[1]):
            if not np.isnan(chunk["mz2_raw"][s, h, 0]):
                mz2_raw[s, h, :] = chunk["mz2_raw"][s, h, :]
                ess_raw[s, h, :] = chunk["ess_raw"][s, h, :]
    valid_t = ~np.isnan(chunk["time_mcmc_ref"])
    time_mcmc[valid_t] = chunk["time_mcmc_ref"][valid_t]
    valid_tr = chunk["trial_chosen"] >= 0
    trial_chosen[valid_tr] = chunk["trial_chosen"][valid_tr]
    print(f"Chunk {i}: fusionne")

# Sauvegarder
output_path = os.path.join(RUN_DIR, f"{prefix}_full.npz")

save_dict = dict(
    h0_grid=first["h0_grid"],
    sigma_grid=first["sigma_grid"],
    mz2_raw=mz2_raw,
    ess_raw=ess_raw,
    time_mcmc_ref=time_mcmc,
    trial_chosen=trial_chosen,
    N_SAMPLES_IS=int(first["N_SAMPLES_IS"]),
    n_chains_is=int(first["n_chains_is"]),
    N_DISORDER=int(first["N_DISORDER"]),
    N_MCMC_TRIALS=int(first["N_MCMC_TRIALS"]),
    L=int(first["L"]),
)
# Champs optionnels
for key in ["nb_spins", "dim"]:
    if key in first:
        save_dict[key] = int(first[key])

np.savez(output_path, **save_dict)
print(f"\nFusion terminee: {output_path}")
print(f"NaN restants dans mz2_raw[:,:,0]: {np.sum(np.isnan(mz2_raw[:, :, 0]))}")
