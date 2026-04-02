#!/bin/bash
# Lance le train via SLURM, 1 GPU exclusif par taille
# Usage:
#   bash launch_train.sh 64            # une taille
#   bash launch_train.sh 48 64 80 100  # plusieurs en parallele

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
SCRIPT="Foundational/train_fonctionnels/1D_avec_desordre_pluri_h0.py"
NITER=600

# Noeuds avec GPU >= 20GB verifies
NODES=(bengali albatros autruche coucou epervier faisan gelinotte harpie hibou jabiru kamiche linotte mouette nandou)

cd "$PROJECT"
i=0
for L in "$@"; do
    NODE=${NODES[$((i % ${#NODES[@]}))]}
    sbatch --job-name="train_L${L}" \
           --output="Foundational/logs/slurm/train_L${L}_%j.out" \
           --ntasks=1 --nodelist=$NODE --exclusive \
           --cpus-per-task=8 --mem=60G \
           --time=3-00:00:00 --partition=SallesInfo \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
           --wrap="python $SCRIPT --L $L --n-iter $NITER"
    echo "L=$L -> $NODE"
    i=$((i+1))
done
echo "Suivi: squeue -u \$USER"
