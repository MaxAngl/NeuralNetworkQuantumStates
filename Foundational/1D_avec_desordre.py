import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket as nk
import netket_foundational as nkf
from nqs_psc.utils import save_run # Assure-toi que ce module est accessible

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

# ==========================================
# 1. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
# On définit tout ici pour que le 'meta' soit cohérent
seed = 1
k = jax.random.key(seed)
L = 4              # Taille du système
h0 = 1.0            # Champ moyen
sigma_disorder = 0.1 # Désordre
J_val = 1.0/np.e    # Couplage Ising (défini dans create_operator)
n_replicas = 100    # Nombre de réalisations de désordre
n_chains = 800      # Pour le sampler
n_samples = n_replicas * 64
n_iter = 1000       # Nombre d'étapes d'optimisation
lr_init = 0.03
lr_end = 0.005
diag_shift = 1e-4
logs_path = "logs"  # Dossier racine pour les logs

# Paramètres du modèle ViT
vit_params = {
    "num_layers": 4,
    "d_model": 16,
    "heads": 8,
    "b": 1,
    "L_eff": L,
}

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=h0)

def generate_disorder_realizations(n_replicas, system_size, h0, rng=None, sigma=0.1):
    if rng is None:
        rng = np.random.default_rng()
    # Shape: (N, system_size)
    return rng.normal(loc=h0, scale=sigma, size=(n_replicas, system_size))

# Modèle
ma = ViTFNQS(
    num_layers=vit_params["num_layers"],
    d_model=vit_params["d_model"],
    heads=vit_params["heads"],
    b=vit_params["b"],
    L_eff=vit_params["L_eff"], 
    n_coups=ps.size, 
    complex=True, 
    disorder=True, 
    transl_invariant=False, 
    two_dimensional=False, 
)

# Sampler & État Variationnel
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=n_replicas, n_samples=n_samples, seed=seed)

# Initialisation des paramètres (désordre)
params_list = generate_disorder_realizations(n_replicas, hi.size, h0, sigma=sigma_disorder)
print(f"Forme des paramètres de désordre : {params_list.shape}")
vs.parameter_array = params_list

# Opérateurs
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))

def create_operator(params):
    assert params.shape == (hi.size,)
    # Transverse field term: sum_i h_i sigma^x_i
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    # Ising interaction
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz)

# ==========================================
# 3. LOGGING ET OPTIMISATION
# ==========================================

# Callback de sauvegarde
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
        # Sauvegarde initiale optionnelle
        # path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
        # driver.state.save(path)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)

# Optimiseur
learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300) 
optimizer = optax.sgd(learning_rate)
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift)

# Dictionnaire META propre
meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {"type": "Ising Disorder", "J": J_val, "h0": h0, "sigma": sigma_disorder},
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "sampler": {"type": "MetropolisLocal", "n_chains": n_chains, "n_samples": n_samples},
    "optimizer": {"type": "SGD", "lr_init": lr_init, "lr_end": lr_end, "diag_shift": diag_shift},
    "n_iter": n_iter,
    "n_replicas": n_replicas,
}

# Logger
# On initialise le logger avec un dossier temporaire ou final
log = nk.logging.JsonLog("log_data", save_params=False) 

# Création de la structure de dossier via ta fonction utilitaire
# Note: Assure-toi que save_run renvoie bien le chemin créé si tu veux l'utiliser pour SaveState
try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"

# Lancement du run
gs.run(
    n_iter,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    callback=SaveState(run_dir, 10), # Sauvegarde dans le dossier créé
)

# ==========================================
# 4. ANALYSE ET PLOTS (CONVERGENCE)
# ==========================================
print('Plotting convergence curves...')
conv_data = []

# Attention: log.data["ham"] est une liste de loggers (un par replica) si nkf gère le logging shardé ainsi.
# Sinon, l'accès peut varier selon la version de nkf/nk.
# On suppose ici que log.data["ham"] est accessible par index [i].

for i, pars in tqdm(enumerate(vs.parameter_array)):
    _ha = create_operator(pars)
    # Calcul exact (Lanczos)
    ed = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()

    # Récupération des données du log
    # Note: Il faut s'assurer que log.data["ham"] contient bien une liste correspondant aux replicas
    # Si nkf agrège tout, cette boucle doit être adaptée. 
    # Supposons que log.data["ham"][i] existe :
    
    if hasattr(log.data["ham"], "__getitem__") and len(log.data["ham"]) > i:
        ham_log = log.data["ham"][i]
        err_val = ham_log.Mean - ed
        conv_data.append({
            "iters": ham_log.iters,
            "e0": ham_log.Mean,
            "err_val": err_val
        })

plt.figure()
for _data in conv_data:
    plt.plot(
        _data["iters"],
        np.abs(_data["err_val"] / _data["e0"]),
        alpha=0.3
    )

plt.xlabel("Iteration")
plt.ylabel("Rel Error")
plt.xscale("log")
plt.yscale("log")
plt.savefig("convergence.pdf")
plt.clf()

# ==========================================
# 5. TEST SUR NOUVEL ENSEMBLE (CORRIGÉ)
# ==========================================

N_test = 100
params_list_test = generate_disorder_realizations(N_test, hi.size, h0, sigma=sigma_disorder)
print(f"Test set shape: {params_list_test.shape}")

# Mise à jour de l'état avec les nouveaux paramètres
vs.parameter_array = params_list_test
Mz2 = Mz @ Mz
Mz2_mat = Mz2.to_sparse()

# --- Calcul Exact ---
exact = {"Energy": [], "Mz2": []}
print('Computing exact values on test set...')
for pars in tqdm(params_list_test):
    _ha = create_operator(pars.reshape(-1))
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    E0 = E0.item()
    psi0 = psi0.reshape(-1)
    
    exact["Energy"].append(E0)
    exact["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0)).item())

exact_vals = {
    "Energy": np.array(exact["Energy"]),
    "Mz2": jnp.real(np.array(exact["Mz2"])),
}

# --- Calcul VMC (FullSum) ---
vmc_vals = {
    "Energy": [],
    "Mz2": [],
    "V_score": [], # Correction de la majuscule pour matcher l'append
}

print('Computing NQS predictions on test set...')
for pars in tqdm(vs.parameter_array):
    _ha = create_operator(pars)
    _vs = vs.get_state(pars)
    _vs.reset()

    # Utilisation de FullSumState pour obtenir la valeur exacte de l'ansatz
    vs_fs = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, chunk_size=_vs.chunk_size, variables=_vs.variables)
    
    # --- CORRECTION MAJEURE ICI ---
    # 1. On calcule d'abord les espérances
    _e = vs_fs.expect(_ha)
    _o = vs_fs.expect(Mz2)
    
    # 2. Ensuite, on peut accéder à la variance
    variance_H = _e.variance
    energy_sq = (_e.Mean.real)**2 
    
    # Sécurité division par zéro
    if energy_sq > 1e-12:
        current_v_score = variance_H / energy_sq
    else:
        current_v_score = 0.0
    
    vmc_vals["Energy"].append(_e.Mean)
    vmc_vals["Mz2"].append(_o.Mean)
    vmc_vals["V_score"].append(current_v_score)

# Conversion en arrays
vmc_final = {
    "Energy": np.array(vmc_vals["Energy"]),
    "Mz2": jnp.real(np.array(vmc_vals["Mz2"])),
    "V_score": np.real(np.array(vmc_vals["V_score"]))
}

# ==========================================
# 6. PLOTS FINAUX (CORRIGÉS)
# ==========================================

# Plot Magnetization
plt.figure()
plt.scatter(exact_vals['Mz2'], vmc_final['Mz2'], alpha=0.7, label='Data')

# Correction de la ligne d'identité (min vers max, pas min vers min)
min_mz = min(jnp.min(exact_vals["Mz2"]), jnp.min(vmc_final["Mz2"]))
max_mz = max(jnp.max(exact_vals["Mz2"]), jnp.max(vmc_final["Mz2"]))
plt.plot([min_mz, max_mz], [min_mz, max_mz], linestyle='--', color='black', label='Ideal')

plt.xlabel("Exact $M_z^2$")
plt.ylabel("VMC $M_z^2$")
plt.legend()
plt.savefig("mag_accordance.pdf")
plt.clf()

# Plot Distributions (KDE)
plt.figure()
# Exact
kde_exact = gaussian_kde(exact_vals["Mz2"])
x_space = np.linspace(min_mz, max_mz, 500)
plt.plot(x_space, kde_exact(x_space), linestyle='--', color='black', label='Exact')

# VMC
kde_vmc = gaussian_kde(vmc_final["Mz2"])
plt.plot(x_space, kde_vmc(x_space), label='VMC')

plt.xlabel("$M_z^2$")
plt.ylabel("Distribution density")
plt.legend()
plt.savefig("mz2_distrib.pdf")
plt.clf()

print("Terminé. Résultats sauvegardés.")