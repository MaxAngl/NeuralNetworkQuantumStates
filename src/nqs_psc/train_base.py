# Import necessary libraries
import platform
import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import ansatz as ans
from optimizer import optimize_NGD, optimize_SGD
from utils import save_runtime_log

L = 4
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=True)

# Build the Hamiltonian
hamiltonian = nk.operator.Ising(hi, g, h=3.044, J=1.0)

# Convert to JAX format
hamiltonian_jax = hamiltonian.to_jax_operator()

# Compute exact ground state for comparison
from scipy.sparse.linalg import eigsh

e_gs, psi_gs = eigsh(hamiltonian.to_sparse(), k=2, which="SA")
e_gs = e_gs[0]
psi_gs = psi_gs.reshape(-1)

L = [1, 2, 3, 4, 5, 6]

for x in L:
    alpha = x
    # Settings
    model = ans.BM(alpha=alpha)  # Try both MF() and Jastrow()
    n_chains = 20
    sampler = nk.sampler.MetropolisSampler(
        hi, nk.sampler.rules.LocalRule(), n_chains=n_chains
    )
    # n_iters = 300
    N_e = 1000
    chain_length = N_e // sampler.n_chains
    diag_shift = 0.01
    learning_rate = 5e-2
    n_iters = 300
    logger, model, parameters, sampler = optimize_NGD(
        model, sampler, hamiltonian, chain_length, diag_shift, n_iters, learning_rate
    )

    # auto save
    meta = dict(
        exact=float(e_gs),
        N_e=N_e,
        diag_shift=float(diag_shift),
        lr=float(learning_rate),
        hamiltonien="Hypercube - L = 4 - n = 2",
        model="jastrow",
        alpha=alpha,
    )
    run_dir = save_runtime_log(logger, meta=meta)
