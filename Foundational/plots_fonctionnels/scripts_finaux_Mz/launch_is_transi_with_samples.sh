#!/bin/bash
# Relance l'IS pour L=16,24,36,48 (Trains_autour_transi_1D) en sauvegardant les samples.
# Grille h0 : 20 pts entre 0.8 et 1.2 (même que les runs existants).
# 10 chunks × 4 tailles = 40 jobs au total.
# Samples : 8192 (2^13), 80 réalisations de désordre.
#
# Usage: bash launch_is_transi_with_samples.sh
# Les sorties is_data_chunk* et is_samples_chunk* vont dans Trains_autour_transi_1D/L={L}/

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
SCRIPT="Foundational/plots_fonctionnels/scripts_finaux_Mz/importance_sampling.py"
GRID="$PROJECT/Foundational/plots_fonctionnels/scripts_finaux_Mz/h0_grid_transi_20pts.txt"
NCHUNKS=10
NSAMPLES=8192
NDISORDER=80
NCHAINS=256
NTRIALS=3

cd "$PROJECT"

for L in 16 24 36 48; do
    RUNDIR="Foundational/logs/Trains_autour_transi_1D/L=$L"
    for c in $(seq 0 $((NCHUNKS-1))); do
        sbatch --job-name="is_samp_L${L}_c${c}" \
               --output="Foundational/logs/slurm/is_samp_L${L}_c${c}_%j.log" \
               --ntasks=1 --cpus-per-task=4 --mem=60G \
               --time=08:00:00 --partition=SallesInfo \
               --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp,PYTHONPATH="$PROJECT:$PROJECT/Foundational" \
               --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate netket_env && python $SCRIPT --dim 1 --L $L --run-dir $RUNDIR --chunk $c --nchunks $NCHUNKS --ndisorder $NDISORDER --nsamples $NSAMPLES --nchains $NCHAINS --ntrials $NTRIALS --h0-grid $GRID"
    done
    echo "$NCHUNKS jobs soumis pour L=$L"
done

echo ""
echo "40 jobs soumis au total (10 chunks × 4 tailles)."
echo "Suivi: squeue -u \$USER | grep is_samp"
echo ""
echo "Après complétion, merge avec :"
echo "  python merge_chunks.py --run-dir Foundational/logs/Trains_autour_transi_1D/L=16 --prefix is_data_1D_L16 --nchunks 10"
echo "  (idem pour L=24, 36, 48)"
