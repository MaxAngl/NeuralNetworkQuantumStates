#!/bin/bash
# Lance l'importance sampling en parallele via SLURM
# Usage: bash launch_is.sh <dim> <L> <run-dir> [nchunks]
#
# Exemples:
#   bash launch_is.sh 1 48 Foundational/logs/Trains_v2_1D/run_L=48
#   bash launch_is.sh 1 64 Foundational/logs/run_2026-03-28_15-00-00 20
#   bash launch_is.sh 2 8 Foundational/rami_perso/2D_FNQS/Run_2D_L8_FNQS

DIM=${1:?"Usage: $0 <dim> <L> <run-dir> [nchunks]"}
L=${2:?"Usage: $0 <dim> <L> <run-dir> [nchunks]"}
RUNDIR=${3:?"Usage: $0 <dim> <L> <run-dir> [nchunks]"}
NCHUNKS=${4:-20}

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
SCRIPT="Foundational/plots_fonctionnels/scripts_finaux_Mz/importance_sampling.py"

cd "$PROJECT"
for c in $(seq 0 $((NCHUNKS-1))); do
    sbatch --job-name="is_L${L}_c${c}" \
           --output="Foundational/logs/slurm/is_${DIM}D_L${L}_c${c}_%j.log" \
           --ntasks=1 --cpus-per-task=4 --mem=60G \
           --time=1-00:00:00 --partition=SallesInfo \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp,PYTHONPATH="$PROJECT:$PROJECT/Foundational" \
           --wrap="python $SCRIPT --dim $DIM --L $L --run-dir $RUNDIR --chunk $c --nchunks $NCHUNKS"
done
echo "$NCHUNKS jobs soumis pour ${DIM}D L=$L"
echo "Suivi: squeue -u \$USER | grep is_L${L}"
