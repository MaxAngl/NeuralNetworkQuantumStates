import netket as nk
from nqs_psc.utils import save_run
import numpy as np

# Taille du système
L = 3
a1 = np.array([1.0, 0.0])
a2 = np.array([0.0, 1.0])

# Définition de l'hamiltonien
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ham = nk.operator.Heisenberg(hi, g)
lattice = nk.graph.Lattice(basis_vectors=[a1, a2], extent=(3, 3), pbc=True)


from scipy.sparse.linalg import eigsh

e_gs, psi_gs = eigsh(ham.to_sparse(), k=2, which="SA")
e_gs = e_gs[0]
psi_gs = psi_gs.reshape(-1)
print(e_gs)

# ---- Seul changement ici ----
model = nk.models.RBM(alpha=1)
# -----------------------------

sampler = nk.sampler.MetropolisLocal(hi, n_chains=300)
vstate = nk.vqs.MCState(sampler, model, n_samples=1000)

# Optimisation
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
gs = nk.driver.VMC(ham, optimizer, variational_state=vstate)

# création du logger
log = nk.logging.RuntimeLog()

# One or more logger objects must be passed to the keyword argument `out`.
# gs.run(n_iter=300, out=log)

# meta identique
meta = {
    "L": L,
    "graph": "Hypercube",
    "n_dim": 2,
    "pbc": True,
    "hamiltonian": {"type": "Heisenberg"},
    "model": "RBM(alpha=1)",
    "sampler": {"type": "MetropolisLocal", "n_chains": 300, "n_samples": 1000},
    "optimizer": {"type": "SGD", "lr": 0.01},
    "n_iter": 300,
}

run_dir = save_run(log, meta)
