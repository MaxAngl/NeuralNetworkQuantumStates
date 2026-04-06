#!/bin/bash
# Lance l'importance sampling 2D directement (sans dépendance SLURM).
# À utiliser quand les trains sont déjà terminés.
#
# Usage:
#   bash Foundational/launch_IS_2D_now.sh
#   (détecte automatiquement les run_dirs via find_run_dir_2D.py)

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
IS_SCRIPT="$PROJECT/Foundational/plots_fonctionnels/scripts_finaux_Mz/importance_sampling.py"
FIND_DIR="$PROJECT/Foundational/find_run_dir_2D.py"
H0_GRID="$PROJECT/Foundational/h0_grid_IS_2D.txt"

N_CHUNKS=5
CONDA="source /users/eleves-b/2024/nathan.dupuy/miniconda3/etc/profile.d/conda.sh && conda activate netket_env"
N_SAMPLES=8192
N_CHAINS=256
PROB_FLIP=0.05

SIZES=(3 4 5 6 8 10)

NODES=(bengali albatros autruche coucou epervier faisan gelinotte harpie hibou jabiru kamiche linotte mouette nandou)

mkdir -p "$PROJECT/Foundational/logs/slurm"

i_node=0
for L in "${SIZES[@]}"; do
    RUN_DIR=$(python3 "$FIND_DIR" "$L" 2>/dev/null)
    if [ -z "$RUN_DIR" ]; then
        echo "SKIP L=$L : run_dir introuvable"
        continue
    fi
    echo "L=$L -> $RUN_DIR"

    for chunk in $(seq 0 $((N_CHUNKS - 1))); do
        NODE=${NODES[$((i_node % ${#NODES[@]}))]}

        sbatch --job-name="IS_2D_L${L}_c${chunk}" \
               --output="$PROJECT/Foundational/logs/slurm/IS_2D_L${L}_c${chunk}_%j.out" \
               --ntasks=1 --nodelist=$NODE --exclusive \
               --cpus-per-task=8 --mem=60G \
               --time=1-00:00:00 --partition=SallesInfo \
               --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
               --wrap="
$CONDA
python $IS_SCRIPT \
    --dim 2 --L $L \
    --chunk $chunk --nchunks $N_CHUNKS \
    --nsamples $N_SAMPLES --nchains $N_CHAINS \
    --prob-flip $PROB_FLIP \
    --h0-grid $H0_GRID \
    --run-dir $RUN_DIR
"
        echo "  IS L=${L}x${L} chunk ${chunk}/${N_CHUNKS} -> $NODE"
        i_node=$((i_node + 1))
    done
done

echo ""
echo "Total: $((${#SIZES[@]} * N_CHUNKS)) jobs IS soumis."
echo "Suivi: squeue -u \$USER"
