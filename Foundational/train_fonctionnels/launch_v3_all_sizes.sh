#!/bin/bash
# Lance le script v3 pour toutes les tailles via SLURM
# 7 tailles × 12 GPUs = 84 noeuds (sur 97 disponibles)

PROJECT=/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates
SCRIPT=$PROJECT/Foundational/train_fonctionnels/1D_avec_desordre_pluri_h0_v3.py

# 97 noeuds disponibles, répartis en 7 groupes de ~12-15
# n_chains = 6000, diviseurs: 10, 12, 15, 20, 24, 25
# On utilise 12 GPUs par job (6000/12 = 500 chains/GPU)

ALL_NODES="acromion,ain,albatros,allier,apophyse,ardennes,astragale,atlas,autruche,axis,bengali,bentley,bugatti,cadillac,carmor,cher,chrysler,coccyx,corvette,cote,coucou,creuse,cubitus,cuboide,dindon,dordogne,doubs,epervier,essonne,faisan,femur,ferrari,fiat,ford,frontal,gelinotte,gironde,harpie,hibou,humerus,indre,jabiru,jaguar,jura,kamiche,lada,linotte,loire,loriol,malleole,manche,marne,maserati,mayenne,mazda,metacarpe,morbihan,moselle,mouette,nandou,nissan,niva,ombrette,parietal,perone,peugeot,phalange,pontiac,porsche,quetzal,quiscale,radius,renault,rolls,rotule,rouloul,rover,royce,sacrum,saone,sitelle,skoda,somme,sternum,tarse,temporal,test-252,test-253,tibia,traquet,urabu,vendee,venturi,verdier,volvo,vosges,xiphoide"

# Convertir en tableau
IFS=',' read -ra NODES <<< "$ALL_NODES"

SIZES=(16 24 36 48 64 80 100)
GPUS_PER_JOB=12

echo "=== Lancement v3 pour ${#SIZES[@]} tailles, ${GPUS_PER_JOB} GPUs chacune ==="
echo "Total noeuds utilisés: $((${#SIZES[@]} * GPUS_PER_JOB)) / ${#NODES[@]}"

for i in "${!SIZES[@]}"; do
    L=${SIZES[$i]}
    START=$((i * GPUS_PER_JOB))

    # Extraire le sous-ensemble de noeuds pour ce job
    JOB_NODES=""
    for j in $(seq $START $((START + GPUS_PER_JOB - 1))); do
        if [ -n "$JOB_NODES" ]; then
            JOB_NODES="${JOB_NODES},${NODES[$j]}"
        else
            JOB_NODES="${NODES[$j]}"
        fi
    done

    echo ""
    echo "--- L=$L : noeuds $JOB_NODES ---"

    sbatch --job-name="v3_L${L}" \
           --ntasks=${GPUS_PER_JOB} \
           --nodes=${GPUS_PER_JOB} \
           --nodelist="${JOB_NODES}" \
           --output="${PROJECT}/Foundational/train_fonctionnels/slurm_v3_L${L}_%j.out" \
           --error="${PROJECT}/Foundational/train_fonctionnels/slurm_v3_L${L}_%j.err" \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
           --wrap="srun python ${SCRIPT} ${L}"

    echo "Soumis L=$L"
done

echo ""
echo "=== Tous les jobs soumis ==="
echo "Vérifier avec: squeue -u \$USER"
