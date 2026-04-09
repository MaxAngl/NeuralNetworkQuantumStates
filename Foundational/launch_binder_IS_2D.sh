#!/bin/bash
# Lance le calcul du Binder cumulant IS pour tous les L 2D.
# Un job par taille (L=3,4,5,6,8,10), sur les noeuds GPU >= 20GB.
#
# Usage: bash Foundational/launch_binder_IS_2D.sh

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
SCRIPT="$PROJECT/Foundational/plots_fonctionnels/scripts_finaux_Mz/compute_binder_IS_2D.py"
CONDA="source /users/eleves-b/2024/nathan.dupuy/miniconda3/etc/profile.d/conda.sh && conda activate netket_env"

SIZES=(3 4 5 6 8 10)
N_DISORDER=100

# Noeuds avec GPU >= 20GB
NODES=(bengali albatros autruche coucou epervier faisan gelinotte harpie hibou jabiru)

mkdir -p "$PROJECT/Foundational/logs/slurm"

for i in "${!SIZES[@]}"; do
    L=${SIZES[$i]}
    NODE=${NODES[$((i % ${#NODES[@]}))]}
    RUN_DIR="$PROJECT/Foundational/logs/Trains_autour_transi_2D/L=$L"

    sbatch --job-name="binder_IS_L${L}" \
           --output="$PROJECT/Foundational/logs/slurm/binder_IS_2D_L${L}_%j.out" \
           --ntasks=1 --nodelist=$NODE --exclusive \
           --cpus-per-task=8 --mem=60G \
           --time=02:00:00 --partition=SallesInfo \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
           --wrap="
$CONDA
python $SCRIPT --L $L --ndisorder $N_DISORDER --run-dir $RUN_DIR
"
    echo "L=${L}x${L} -> $NODE"
done

echo ""
echo "${#SIZES[@]} jobs soumis."
echo "Suivi : squeue -u \$USER | grep binder_IS"
