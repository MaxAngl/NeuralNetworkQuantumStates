#Sbatch --ntasks=1
#Sbatch --cpus-per-task=5
#Sbatch --nodes=1

echo"$(hostname)"
fuv python -c'print(1+1)'
#pour faire tourner python 

sbatch experiment.sh
#pour lancer le programme
# et pour voir si c'est fini
squeue -u $USER

#rajouter srun devant insctructions exécutés dans plusieurs machines 
pour dire à jax de faire parler les machines 

jax.distributed.initialize()

#Aller voir documentation Netket piur + de documentation cf zulip