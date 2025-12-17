import netket as nk
from nqs_psc.utils import save_run
import numpy as np
from functools import partial
import copy
import flax.linen as nn
import jax
import jax.numpy as jnp

# Taille du système

L = 4
a1 = np.array([1.0, 0.0])
a2 = np.array([0.0, 1.0])
J = -5
H = 1
# Définition de l'hamiltonien
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ham = nk.operator.Ising(hi, g, J=J, h=H)
lattice = nk.graph.Lattice(basis_vectors=[a1, a2], extent=(3, 3), pbc=True)

# Fonction qui permet de log d'autres valeurs d'expectation


def expect_operator_callback(fs_state, operator_list):
    def aux_fn(step, logdata, driver, fs_state, operator_list):
        fs_state.variables = copy.deepcopy(driver.state.variables)
        for i, op in enumerate(operator_list):
            res = fs_state.expect(op)
            logdata[f"op_{i}"] = res
        return True

    return partial(aux_fn, fs_state=fs_state, operator_list=operator_list)


operator_list = [nk.operator.spin.sigmaz(hi, i) for i in range(g.n_nodes)]


# ---- Seul changement ici ----
alpha = 3

initializers = nn.initializers
BAD_COMPLEX_STDDEV = 1  # Grande stddev pour perturber (le "mauvais")
BIAS_POLARISATION = 0.07  # Petit biais réel positif pour briser la symétrie

# 1. Initialiseur du Kernel (W): Très grande variance complexe
mon_initialiseur_mauvais_complexe = initializers.normal(stddev=BAD_COMPLEX_STDDEV)


# 2. Initialiseur du Biais Visible (a): Petit biais constant positif (dans le domaine complexe)
def constant_positive_bias_complex(key, shape, dtype=jnp.complex128):
    """
    Initialise le tenseur avec une petite valeur réelle constante positive (0.1 + 0j).
    Ceci sert de graine pour forcer la brisure de symétrie vers l'état UP.
    """
    # Crée un tableau de nombres complexes (partie imaginaire = 0)
    return jnp.full(shape, BIAS_POLARISATION, dtype=dtype)


# 3. Définition du Modèle RBM
model = nk.models.RBM(
    alpha=1,
    param_dtype=complex,
)


sampler = nk.sampler.MetropolisLocal(hi, n_chains=300)
vstate = nk.vqs.MCState(sampler, model, n_samples=1000, seed=451)
#fs_state = nk.vqs.FullSumState(hi, model)


# Optimisation
lr = 0.002

optimizer = nk.optimizer.Sgd(learning_rate=lr)
gs = nk.driver.VMC_SR(
    ham,
    optimizer,
    variational_state=vstate,
    diag_shift=1e-4,
)

# création du logger
log = nk.logging.RuntimeLog()

# One or more logger objects must be passed to the keyword argument `out`.
gs.run(
    n_iter=300, out=log
)

# meta identique
meta = {
    "L": L,
    "graph": "Hypercube",
    "n_dim": 2,
    "pbc": True,
    "hamiltonian": {"type": "Ising", "J": J, "h": H},
    "model": "RBM",
    "alpha": alpha,
    "sampler": {"type": "MetropolisLocal", "n_chains": 300, "n_samples": 1000},
    "optimizer": {"type": "SGD", "lr": 0.01, "diag_shift": "?"},
    "n_iter": 300,
    "operators_list": "spins",
}

run_dir = save_run(log, meta)
