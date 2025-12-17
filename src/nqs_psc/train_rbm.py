import netket as nk
from nqs_psc.utils import save_run
import numpy as np
from functools import partial
import copy
import flax.linen as nn
import jax
import jax.numpy as jnp
import nqxpack

# Taille du système

n_dim= 2
L = 3
J = -5
H = 1

#Paramètres RBM/optimisation

alpha = 3
lr= 0.01
diag_shift= 1e-3
n_chains = 300
n_samples =1000
n_iter =300


# Définition de l'hamiltonien

g = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ham = nk.operator.Ising(hi, g, J=J, h=H)


# Définition du Modèle RBM


model = nk.models.RBM(
    alpha=alpha,
    param_dtype=complex,
)


sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=451)

# Optimisation

optimizer = nk.optimizer.Sgd(learning_rate=lr)
gs = nk.driver.VMC_SR(
    ham,
    optimizer,
    variational_state=vstate,
    diag_shift=diag_shift,
)

# création du logger
log = nk.logging.RuntimeLog()

# meta identique
meta = {
    "L": L,
    "graph": "Hypercube",
    "n_dim": n_dim,
    "pbc": True,
    "hamiltonian": {"type": "Ising", "J": J, "h": H},
    "model": "RBM",
    "alpha": alpha,
    "sampler": {"type": "MetropolisLocal", "n_chains": n_chains, "n_samples": n_samples},
    "optimizer": {"type": "SGD", "lr": lr, "diag_shift": diag_shift},
    "n_iter": n_iter,
    "exact": "?",
}

# Créer le dossier de run avant l'entraînement
run_dir = save_run(log, meta, create_only=True)

# One or more logger objects must be passed to the keyword argument `out`.
def save_vstate(step, logdata, driver):
    if step % 50 == 0:
        nqxpack.save(driver.state, f"{run_dir}/vstate_step_{step}")
    return True

gs.run(n_iter=n_iter, out=log, callback=(save_vstate,))

# Sauvegarder les logs finaux
save_run(log, meta, run_dir=run_dir)
