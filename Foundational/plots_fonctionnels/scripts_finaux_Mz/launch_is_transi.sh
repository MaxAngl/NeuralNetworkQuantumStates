#!/bin/bash
# Lance l'IS pour les nouveaux trains (Trains_autour_transi_1D), sans désordre.
# Grille h0 : 40 pts entre 0.8 et 1.2. Calcul sigma=0 uniquement (ndisorder=1).
#
# Usage: bash launch_is_transi.sh <L> [nchunks]
# Exemple: bash launch_is_transi.sh 48 10

L=${1:?"Usage: $0 <L> [nchunks]"}
NCHUNKS=${2:-10}

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
SCRIPT="Foundational/plots_fonctionnels/scripts_finaux_Mz/importance_sampling.py"
GRID="$PROJECT/Foundational/plots_fonctionnels/scripts_finaux_Mz/h0_grid_transi_40pts.txt"
RUNDIR="Foundational/logs/Trains_autour_transi_1D/L=$L"

cd "$PROJECT"
for c in $(seq 0 $((NCHUNKS-1))); do
    sbatch --job-name="is_transi_L${L}_c${c}" \
           --output="Foundational/logs/slurm/is_transi_L${L}_c${c}_%j.log" \
           --ntasks=1 --cpus-per-task=4 --mem=60G \
           --time=04:00:00 --partition=SallesInfo \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp,PYTHONPATH="$PROJECT:$PROJECT/Foundational" \
           --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate netket_env && python $SCRIPT --dim 1 --L $L --run-dir $RUNDIR --chunk $c --nchunks $NCHUNKS --ndisorder 80 --nsamples 8192 --h0-grid $GRID"
done
echo "$NCHUNKS jobs soumis pour 1D L=$L (Trains_autour_transi_1D)"
echo "Suivi: squeue -u \$USER | grep is_transi_L${L}"
