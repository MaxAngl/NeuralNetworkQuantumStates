#!/bin/bash
#SBATCH --job-name=NetKet_HPC
#SBATCH --partition=SallesInfo
#SBATCH --nodes=4                  
#SBATCH --ntasks-per-node=1         
#SBATCH --time=24:00:00             
#SBATCH --output=logs_Multi_%j.txt       

# On garde ta liste d'élite pour ne pas tomber sur des petits GPU
#SBATCH --nodelist=faisan,kamiche,loriol,epervier,gelinotte,hibou,harpie,jabiru,linotte,ferrari,lada,maserati,cuboide,cubitus,jura,frontal,landes,manche,indre,marne,creuse,finistere,fiat,ford,malleole,essonne,loire,jaguar,gironde,femur,humerus,dordogne,mayenne,doubs

TAILLE_L=$1

# 1. Nettoyage et chargement
module purge
module load mpi/openmpi-x86_64

# 2. Environnement
source ~/.bashrc
conda activate netket_env

# On récupère l'adresse IP du premier nœud pour que les autres s'y connectent
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# On lance avec srun en propageant les variables JAX
srun python3 /users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/train_fonctionnels/1D_avec_desordre_pluri_h0.py $TAILLE_L