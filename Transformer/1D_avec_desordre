# This example runs in ~2 minutes on 4 A100s + 5 minutes for the plotting exact values
import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket as nk
import netket_foundational as nkf

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax

from scipy.stats import gaussian_kde
from netket.utils import struct
import matplotlib.pyplot as plt

from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks.base import AbstractCallback
import netket_pro.distributed as nkpd

k = jax.random.key(1)
N = 100
h0 = 1.0
hi = nk.hilbert.Spin(0.5, 16)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=h0)
#N nombre de paramètres indépendants

def generate_disorder_realizations(N, system_size, h0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    #cela prend le generateur standard si aucun autre n'est proposé dans rng    

    # Shape: (N, system_size)
    return rng.uniform(0.0, h0, size=(N, system_size))
#produit une matrice de nombres aleatoires entre 0 et h0 de taille (N, system_size)

ma = ViTFNQS(
    num_layers=4,
    d_model=16,
    heads=8,
    L_eff=hi.size // 4, #taille effective du système
    n_coups=ps.size, #nombre de paramètres variables du hamiltonien, ie dimension de l'espace de paramètre
    b=4, #taille des patchs
    complex=True, #sortie complexe du modèle, sortie réelle possible aussi avec complex=False
    disorder=True, #specifie le type d'embedding adapté à un grand nombre de paramètres de hamiltonien
    transl_invariant=False, #pas d'invariance par translation entre les patchs car le modèle avec désordre ne l'est pas
    two_dimensional=False, #cas 1D ici, utile pour définir la dimension des patchs
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=800)

vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=N, n_samples=N*64, seed=1)

params_list = generate_disorder_realizations(N, hi.size, h0)
print(params_list.shape)
vs.parameter_array = params_list

Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))


def create_operator(params):
    # params: array of shape (system_size,)
    assert params.shape == (hi.size,)

    # Transverse field term: sum_i h_i sigma^x_i
    ha_X = sum(
        params[i] * nkf.operator.sigmax(hi, i)
        for i in range(hi.size)
    )

    # Ising interaction: sum_i sigma^z_i sigma^z_{i+1}
    ha_ZZ = sum(
        nkf.operator.sigmaz(hi, i)
        @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
        for i in range(hi.size)
    )

    return -ha_X - (1/np.exp(1)) * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
#hamiltonien paramétré
#pour un exemple concret on fait H=ha_p(params)
mz_p = nkf.operator.ParametrizedOperator(
    hi,
    ps,
    lambda _: sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size))
    * (1 / float(hi.size)),
)

# xs = vs.hilbert.random_state(k, 5)
# ha_p.get_conn_padded(xs)

class SaveState(AbstractCallback, mutable=True):
    _path: str = struct.field(pytree_node=False, serialize=False)
    _prefix: str = struct.field(pytree_node=False, serialize=False)
    _save_every: int = struct.field(pytree_node=False, serialize=False)

    def __init__(self, path: str, save_every: int, prefix: str = "state"):
        self._path = path
        self._prefix = prefix
        self._save_every = save_every

    def on_run_start(self, step, driver, callbacks):
        if nkpd.is_master_process() and not os.path.exists(self._path):
            os.makedirs(self._path)

        path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
        driver.state.save(path)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)

learning_rate = optax.linear_schedule(init_value=0.03, end_value=0.005, transition_steps=300) 
optimizer = optax.sgd(learning_rate)
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=1e-4)
#La parallélisation de l’optimisation pour plusieurs 
#h est prise en charge nativement par NetKet Foundational grâce à n_replicas=N.
#l'objet vs prend en charge en meme temps un batch de N (nb de réalisations ou de replicas )différents
#gs optimise en parallèle toutes les replicas et leurs Hamiltoniens respectifs

log = nk.logging.JsonLog("2")
#le "2" est le nom du dossier
gs.run(
    1000,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    step_size=10,
    callback=SaveState(
        "2",
        10,
    ),
)
#le champ obs permet de calculer les observables pendant la simulation
#call back  le 10 est pour le pas. Tous les 10 steps le log enregistre les données

# Convergence pour les valeurs de train
print('Plotting convergence curves...')
conv_data = []
for i, pars in tqdm(enumerate(vs.parameter_array)):
    #la boucle parcourt tous les paramètres vecteurs h de vs.parameter_array
    #i correspond au ième set de paramètres h
    _ha = create_operator(pars)
    ed = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()
    #le k=1 signifie qu'on veut la valeur la plus basse ie l'énergie fondamentale

    err_val = log.data["ham"][i].Mean - ed
    conv_data.append(
        {
            "h": h0,
            "e0": log.data["ham"][i].Mean,
            "energy": ed,
            "iters": log.data["ham"][i].iters,
            "err_val": log.data["ham"][i].Mean - ed,
        }
    )

for _data in conv_data:
    # print(max(np.abs(_data["err_val"] / _data["e0"])))
    plt.plot(
        _data["iters"],
        np.abs(_data["err_val"] / _data["e0"]),
        # label=f"h = {_data['h']:.2f}",
    )

plt.xlabel("Iteration")
plt.ylabel("Rel Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("convergence.pdf")
plt.clf()

# Générer 500 tirages pour le test:
N_test = 100
params_list = generate_disorder_realizations(N_test, hi.size, h0)
print(params_list.shape)
# vs.n_replicas = N_test
vs.parameter_array = params_list
Mz2 = Mz @ Mz
Mz2_mat = Mz2.to_sparse()
#convertit l'opérateur en un opérateur sparse (matrice creuse)
exact = {
    "h": vs.parameter_array,
    "Energy": [],
    "Mz2": [],
}
print('computing exact values on test set...')
for pars in tqdm(exact["h"]):
    _ha = create_operator(pars.reshape(-1))
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    E0 = E0.item()
    psi0 = psi0.reshape(-1)
    exact["Energy"].append(E0)
    exact["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0)).item())

exact = {
    "h": np.array(exact["h"]),
    "Energy": np.array(exact["Energy"]),
    "Mz2": jnp.real(np.array(exact["Mz2"])),
}

vmc_vals = {
    "Energy": [],
    "Mz2": [],
}


print('Computing the nqs predictions for the squared magnetizations on the test set...')
for pars in tqdm(vs.parameter_array):
    # Obtention de l'état variationnel et du hamiltonien correspondant à chaque valeur dans l'espace des paramètres
    _ha = create_operator(pars)
    _vs = vs.get_state(pars)
    _vs.reset()

    # Evaluation de la magnétisation et de l'énergie par échantillonage Monte Carlo
    # _vs.sample()
    # _vs.sample()
    # _e = _vs.expect(_ha)
    # _o = _vs.expect(Mz2)

    # Evaluation par sommation sur tout l'espace de Hilbert
    vs_fs = nk.vqs.FullSumState(hilbert = hi, model = _vs.model, chunk_size = _vs.chunk_size, variables = _vs.variables)
    #cet objet va ensuoite calculer ensuite les valeurs des observables en sommant sur tout l'espace de Hilbert
    #MAIS ATTENTION meme si calcul exact c'est la moyenne sur l'état fo,ndamental trouvé par l'alogrithe VMC
    _e = vs_fs.expect(_ha)
    _o = vs_fs.expect(Mz2)
    vmc_vals["Energy"].append(_e.Mean)
    vmc_vals["Mz2"].append(_o.Mean)

vmc_vals = {
    "h": np.array(vs.parameter_array),
    "Energy": np.array(vmc_vals["Energy"]),
    "Mz2": jnp.real(np.array(vmc_vals["Mz2"])),
}

# Plot des valeurs de magnetisation comparant valeurs estimées et valeurs provenant de la diagonalisation exacte
plt.scatter(exact['Mz2'], vmc_vals['Mz2'], alpha=0.7)
plt.plot(np.linspace(jnp.min(exact["Mz2"]), jnp.min(exact["Mz2"]), 100), np.linspace(jnp.min(exact["Mz2"]), jnp.min(exact["Mz2"]), 100), linestyle='--', color='black', label='id')
plt.xlabel("exact mag squared")
plt.ylabel("vmc mag squared")
plt.legend()
plt.savefig("mag_accordance.pdf")
plt.clf()

# Plot des distributions
# Exact
kde_exact = gaussian_kde(np.real(exact["Mz2"]))
x_exact = np.linspace(exact["Mz2"].min(), exact["Mz2"].max(), 500)
plt.plot(x_exact, kde_exact(x_exact),
         linestyle='--', color='black', label='Exact')

# VMC
kde_vmc = gaussian_kde(np.real(vmc_vals["Mz2"]))
x_vmc = np.linspace(vmc_vals["Mz2"].min(), vmc_vals["Mz2"].max(), 500)
plt.plot(x_vmc, kde_vmc(x_vmc),
         label='VMC')

plt.legend()
plt.xlabel("Mz2")
plt.ylabel("Distrib.")
plt.legend()
plt.savefig("mz2.pdf")
plt.clf()