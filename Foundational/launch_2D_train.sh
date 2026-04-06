#!/bin/bash
# Lance les trains 2D cette nuit via SLURM, 1 GPU exclusif par taille
# Usage:
#   bash Foundational/launch_2D_train.sh              # toutes les tailles (4 6 8 10)
#   bash Foundational/launch_2D_train.sh 8 10         # seulement ces tailles

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
SCRIPT="Foundational/train_fonctionnels/2D_avec_desordre_pluri_h0.py"
N_ITER=600
CONDA="source /users/eleves-b/2024/nathan.dupuy/miniconda3/etc/profile.d/conda.sh && conda activate netket_env"

# Noeuds avec GPU >= 20GB vérifiés
NODES=(bengali albatros autruche coucou epervier faisan gelinotte harpie hibou jabiru kamiche linotte mouette nandou)

SIZES=(${@:-4 6 8 10})

mkdir -p "$PROJECT/Foundational/logs/slurm"

cd "$PROJECT"
i=0
for L in "${SIZES[@]}"; do
    NODE=${NODES[$((i % ${#NODES[@]}))]}
    sbatch --job-name="2D_L${L}x${L}" \
           --output="Foundational/logs/slurm/2D_train_L${L}_%j.out" \
           --ntasks=1 --nodelist=$NODE --exclusive \
           --cpus-per-task=8 --mem=60G \
           --time=3-00:00:00 --partition=SallesInfo \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
           --wrap="$CONDA && python $PROJECT/$SCRIPT --L $L --n-iter $N_ITER"
    echo "L=${L}x${L} -> $NODE"
    i=$((i+1))
done
echo "Suivi: squeue -u \$USER"
