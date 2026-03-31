#!/bin/bash
# Lance le train pour chaque taille L sur 11 GPUs verifies
# Usage:
#   bash Foundational/launch_train_v2.sh           # toutes les tailles
#   bash Foundational/launch_train_v2.sh 48 64     # seulement ces tailles

PROJECT="/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates"
SCRIPT="Foundational/1D_avec_desordre_pluri_h0_et_exact.py"
NGPU=11  # diviseur de 440

# 97 noeuds verifies avec JAX GPU fonctionnel (>=20GB, excl charente,perdrix)
GOOD_NODES="acromion,ain,albatros,allier,apophyse,ardennes,astragale,atlas,autruche,axis,bengali,bentley,bugatti,cadillac,carmor,cher,chrysler,coccyx,corvette,cote,coucou,creuse,cubitus,cuboide,dindon,dordogne,doubs,epervier,essonne,faisan,femur,ferrari,fiat,ford,frontal,gelinotte,gironde,harpie,hibou,humerus,indre,jabiru,jaguar,jura,kamiche,lada,linotte,loire,loriol,malleole,manche,marne,maserati,mayenne,mazda,metacarpe,morbihan,moselle,mouette,nandou,nissan,niva,ombrette,parietal,perone,peugeot,phalange,pontiac,porsche,quetzal,quiscale,radius,renault,rolls,rotule,rouloul,rover,royce,sacrum,saone,sitelle,skoda,somme,sternum,tarse,temporal,test-252,test-253,tibia,traquet,urabu,vendee,venturi,verdier,volvo,vosges,xiphoide"

SIZES=(${@:-48 64 80 100})

cd "$PROJECT"
for L in "${SIZES[@]}"; do
    echo "==> L=$L on $NGPU GPUs"
    sbatch --job-name="v2_L${L}" \
           --output="Foundational/logs/slurm_v2_L${L}_%j.out" \
           --ntasks=$NGPU --nodes=$NGPU \
           --cpus-per-task=4 --mem=60G \
           --time=3-00:00:00 --partition=SallesInfo \
           --nodelist=$GOOD_NODES \
           --export=ALL,NETKET_EXPERIMENTAL_SHARDING=1,XLA_PYTHON_CLIENT_PREALLOCATE=false,TMPDIR=/var/tmp \
           --wrap="srun python $PROJECT/$SCRIPT --L $L --n-iter 400"
done
echo "Total: $((${#SIZES[@]} * NGPU)) GPUs"
echo "Suivi: squeue -u \$USER"
