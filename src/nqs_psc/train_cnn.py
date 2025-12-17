import netket as nk
from nqs_psc.utils import save_run
import nqs_psc.ansatz as ans
import numpy as np


# Taille du système
L = 5
a1 = np.array([1.0, 0.0])
a2 = np.array([0.0, 1.0])


# Définition de l'hamiltonien

g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ham = nk.operator.Heisenberg(hi, g)
lattice = nk.graph.Lattice(basis_vectors=[a1, a2], extent=(L, L), pbc=True)


# Création de l'état variationnel
t = (5, 5)
model = ans.CNN(
    lattice=lattice, kernel_size=((2, 2), (2, 2)), channels=t, param_dtype=complex
)
sampler = nk.sampler.MetropolisLocal(hi, n_chains=300)
vstate = nk.vqs.MCState(sampler, model, n_samples=1000, seed=12345)

# Optimisation
lr = 0.001
op = nk.optimizer.Sgd(learning_rate=lr)


gs = nk.driver.VMC_SR(
    hamiltonian=ham,
    optimizer=op,
    variational_state=vstate,
    diag_shift=1e-3
)



# création du logger

log = nk.logging.RuntimeLog()

# One or more logger objects must be passed to the keyword argument `out`.
n_iter = 300
gs.run(n_iter, out=log)


meta = {
    "L": L,
    "graph": "Hypercube",
    "n_dim": 2,
    "pbc": True,
    "hamiltonian": {"type": "Heisenberg"},
    "model": "CNN",
    "kernel_size": 1,
    "channels": t,
    "sampler": {"type": "MetropolisLocal", "n_chains": 300, "n_samples": 1000},
    "optimizer": {"type": "SGD", "lr": lr},
    "n_iter": n_iter,
}


run_dir = save_run(log, meta)
