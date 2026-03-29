#!/bin/bash
# Lance un train via SLURM sur 1 GPU exclusif
# Usage: bash launch_train.sh <L> [n_iter] [node]
#
# Exemples:
#   bash launch_train.sh 64
#   bash launch_train.sh 80 400 bengali

L=${1:?"Usage: $0 <L> [n_iter] [node]"}
NITER=${2:-400}
NODE=${3:-bengali}

PROJECT="/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates"
SCRIPT="Foundational/1D_avec_desordre_pluri_h0_et_exact.py"

cd "$PROJECT"
sbatch --job-name="train_L${L}" \
       --output="Foundational/logs/slurm/train_L${L}_%j.out" \
       --ntasks=1 --nodelist=$NODE --exclusive \
       --cpus-per-task=8 --mem=60G \
       --time=3-00:00:00 --partition=SallesInfo \
       --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
       --wrap="python $SCRIPT --L $L --n-iter $NITER"
echo "Train L=$L soumis sur $NODE ($NITER iter)"
echo "Suivi: squeue -u \$USER | grep train_L${L}"
