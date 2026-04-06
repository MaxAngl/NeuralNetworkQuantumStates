#!/bin/bash
# Lance l'importance sampling 2D après les trains, avec dépendance SLURM.
# 5 nœuds par taille (25 h0 points / 5 chunks = 5 points par nœud)
#
# Usage:
#   bash Foundational/launch_IS_2D.sh <job_L4> <job_L6> <job_L8> <job_L10>
#   Ex: bash Foundational/launch_IS_2D.sh 750 751 752 753

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
IS_SCRIPT="$PROJECT/Foundational/plots_fonctionnels/scripts_finaux_Mz/importance_sampling.py"
FIND_DIR="$PROJECT/Foundational/find_run_dir_2D.py"
H0_GRID="$PROJECT/Foundational/h0_grid_IS_2D.txt"

N_CHUNKS=5       # 5 nœuds par taille
CONDA="source /users/eleves-b/2024/nathan.dupuy/miniconda3/etc/profile.d/conda.sh && conda activate netket_env"
N_SAMPLES=8000
N_CHAINS=200     # 200 × 40 = 8000 samples exactement

SIZES=(4 6 8 10)
TRAIN_JOBS=($1 $2 $3 $4)

# Noeuds avec GPU >= 20GB vérifiés
NODES=(bengali albatros autruche coucou epervier faisan gelinotte harpie hibou jabiru kamiche linotte mouette nandou)

if [ ${#TRAIN_JOBS[@]} -lt 4 ] || [ -z "$1" ]; then
    echo "Usage: bash Foundational/launch_IS_2D.sh <job_L4> <job_L6> <job_L8> <job_L10>"
    echo "  Les job IDs sont ceux retournés par launch_2D_train.sh (ex: 750 751 752 753)"
    exit 1
fi

mkdir -p "$PROJECT/Foundational/logs/slurm"

i_node=0
for i_L in "${!SIZES[@]}"; do
    L=${SIZES[$i_L]}
    TRAIN_JOB=${TRAIN_JOBS[$i_L]}

    for chunk in $(seq 0 $((N_CHUNKS - 1))); do
        NODE=${NODES[$((i_node % ${#NODES[@]}))]}

        sbatch --dependency=afterok:$TRAIN_JOB \
               --job-name="IS_2D_L${L}_c${chunk}" \
               --output="$PROJECT/Foundational/logs/slurm/IS_2D_L${L}_c${chunk}_%j.out" \
               --ntasks=1 --nodelist=$NODE --exclusive \
               --cpus-per-task=8 --mem=60G \
               --time=1-00:00:00 --partition=SallesInfo \
               --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
               --wrap="
$CONDA
RUN_DIR=\$(python $FIND_DIR $L)
if [ -z \"\$RUN_DIR\" ]; then
    echo 'ERREUR: run_dir introuvable pour L=$L (2D)' >&2
    exit 1
fi
echo 'Run dir trouvé: '\$RUN_DIR
python $IS_SCRIPT \
    --dim 2 --L $L \
    --chunk $chunk --nchunks $N_CHUNKS \
    --nsamples $N_SAMPLES --nchains $N_CHAINS \
    --h0-grid $H0_GRID \
    --run-dir \$RUN_DIR
"
        echo "IS L=${L}x${L} chunk ${chunk}/${N_CHUNKS} -> $NODE (après job $TRAIN_JOB)"
        i_node=$((i_node + 1))
    done
done

echo ""
echo "Total: $((${#SIZES[@]} * N_CHUNKS)) jobs IS soumis en attente."
echo "Suivi: squeue -u \$USER"
