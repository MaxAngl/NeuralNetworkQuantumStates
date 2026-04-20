#!/bin/bash
# Lance les tests ViT sans désordre
# Usage:
#   bash Foundational/launch_1D_sans_desordre_test_vit.sh <L> [cfg_idx...]
#   bash Foundational/launch_1D_sans_desordre_test_vit.sh 60 6        # Config 6, L=60
#   bash Foundational/launch_1D_sans_desordre_test_vit.sh 48          # toutes les configs (0-6), L=48
#   bash Foundational/launch_1D_sans_desordre_test_vit.sh 48 0 2 4    # configs 0,2,4, L=48

PROJECT="/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3"
SCRIPT="Foundational/train_fonctionnels/1D_sans_desordre_test_vit.py"
L=${1:-48}

# Configs: tous les args à partir du 2e, ou 0-6 par défaut
if [ $# -ge 2 ]; then
    CONFIGS=("${@:2}")
else
    CONFIGS=(0 1 2 3 4 5 6)
fi

# Noeuds avec GPU >= 20GB vérifiés
NODES=(bengali albatros autruche coucou epervier faisan gelinotte harpie hibou jabiru kamiche linotte mouette nandou)

cd "$PROJECT"
i=0
for CFG in "${CONFIGS[@]}"; do
    NODE=${NODES[$((i % ${#NODES[@]}))]}
    sbatch --job-name="vit1D_L${L}_cfg${CFG}" \
           --output="Foundational/logs/slurm/vit1D_L${L}_cfg${CFG}_%j.out" \
           --ntasks=1 --nodelist=$NODE --exclusive \
           --cpus-per-task=8 --mem=60G \
           --time=3-00:00:00 --partition=SallesInfo \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
           --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate netket_env && python $PROJECT/$SCRIPT $L $CFG"
    echo "Config $CFG -> $NODE (L=$L)"
    i=$((i+1))
done
echo "Suivi: squeue -u \$USER"
