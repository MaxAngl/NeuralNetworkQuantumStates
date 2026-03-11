import os
import sys
# Ajouter le répertoire racine du projet au chemin Python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)

#Décommenter cette ligne pour L supérieur à 16 ou 20
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
# Gestionnaire de mémoire plus efficace pour éviter la fragmentation
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
# Permet à JAX de ne pas allouer 90% de la VRAM au démarrage
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk
import netket_foundational as nkf

from src.nqs_psc.utils import save_run # Assure-toi que ce module est accessible

import time
import pandas as pd
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
from netket.sampler import rules
from flax import struct
from flip_rules import GlobalFlipRule

# ==========================================
# 1. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
# On définit tout ici pour que le 'meta' soit cohérent
seed = 1
rng = np.random.default_rng(seed)
k = jax.random.key(seed)

# --- PARAMÈTRES PHYSIQUES ---
L = 16                                     # Taille du système
# Si on passe un argument dans le terminal, on le prend pour L, sinon L=16 par défaut
if len(sys.argv) > 1:
    L = int(sys.argv[1])

h0_train_list = [ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 2, 3.5, 5.0 ]
sigma_disorder = 0.1 
J_val = 1.0    
n_replicas = 10                             # Nombre de réalisations de désordre

# --- PARAMÈTRES MONTE CARLO ---
total_configs_train = len(h0_train_list) * (n_replicas + 1)
chains_per_replica = 4      
samples_per_chain = 2       
n_chains = total_configs_train * chains_per_replica
n_samples = n_chains * samples_per_chain
prob_global_flip = 0.05  # Probabilité de flip global dans le sampler personnalisé

# --- PARAMÈTRES D'OPTIMISATION ---
n_iter = 400      
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = os.path.join(foundational_dir, "logs")

# --- CALCUL AUTOMATIQUE ET SYSTÉMATIQUE DU CHUNK_SIZE ---
TARGET_CHUNK = 16 

if n_samples <= TARGET_CHUNK:
    chunk_size = n_samples
else:
    # On cherche le plus grand diviseur de n_samples qui est <= TARGET_CHUNK
    chunk_size = 1
    for i in range(TARGET_CHUNK, 0, -1):
        if n_samples % i == 0:
            chunk_size = i
            break

chunk_size_bwd=4

print(f"🔹 Configuration : {n_samples} samples total.")
print(f"🔹 Chunk size auto-calculé : {chunk_size} (Diviseur optimal <= {TARGET_CHUNK})")


# Paramètres du modèle ViT
vit_params = {
    "num_layers": 2,
    "d_model": 32,
    "heads": 4,
    "b": 1,
    "L_eff": L,
}

# ==========================================
# 2. DEFINITION DU SYSTEME
# ==========================================

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(h0_train_list))

def generate_multi_h0_disorder(h0_list, n_reps, system_size, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    all_configs = []
    
    for h_m in h0_list:
        # 1. Génération des 'n_reps' configurations désordonnées (Aléatoire)
        # On tire d'abord selon la gaussienne (peut contenir des négatifs)
        raw_configs = rng.normal(loc=h_m, scale=sigma, size=(n_reps, system_size))
        
        # --- NOUVELLE OPTION : VALEUR ABSOLUE (Repliement) ---
        # Les valeurs négatives "rebondissent" en positif, gardant une distribution lisse
        random_configs = np.abs(raw_configs)
        
        # 2. Création de la configuration homogène (h_m, h_m, ..., h_m) (Exacte)
        # Shape (1, system_size)
        homogeneous_config = np.full((1, system_size), h_m)
        
        # 3. On empile les deux : on obtient (n_reps + 1) configurations pour ce h_m
        # L'homogène est ajoutée à la fin du bloc de ce h_m
        batch_configs = np.vstack([random_configs, homogeneous_config])
        
        all_configs.append(batch_configs)
        
    return np.vstack(all_configs)

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
sa = nk.sampler.MetropolisSampler(
    hi,
    rule=GlobalFlipRule(prob_global_flip),
    n_chains=n_chains
)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=total_configs_train, n_samples=n_samples, seed=seed, chunk_size=chunk_size)

# 1. On récupère le tableau d'états exact généré par NetKet (qui contient Spins + Couplings)
sigma_orig = vs.sampler_state.σ

# 2. On l'aplatit temporairement pour gérer n'importe quelle forme (réplicas/chaînes)
flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
half = flat_sigma.shape[0] // 2

# 3. On utilise .at[...].set(...) car les tableaux JAX sont immuables.
# On écrase UNIQUEMENT les L premières colonnes (qui correspondent aux spins)
# Moitié UP (+1)
flat_sigma = flat_sigma.at[:half, :L].set(1)
# Moitié DOWN (-1)
flat_sigma = flat_sigma.at[half:, :L].set(-1)

# 4. On lui redonne sa forme d'origine et on met à jour le sampler
sigma_new = flat_sigma.reshape(sigma_orig.shape)
vs.sampler_state = vs.sampler_state.replace(σ=sigma_new)

# Initialisation des paramètres (désordre)
params_list = generate_multi_h0_disorder(h0_train_list, n_replicas, hi.size, sigma=sigma_disorder)
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

# === NOUVEAU CALLBACK POUR LOGGER LES REPLICAS SANS CRASH XLA ===
class ReplicaLogger(AbstractCallback):
    # DÉCLARATION OBLIGATOIRE POUR JAX/NETKET :
    params_list: np.ndarray = struct.field(pytree_node=False)
    L: int = struct.field(pytree_node=False)
    eval_every: int = struct.field(pytree_node=False)
    
    def __init__(self, params_list, L, eval_every=10):
        self.params_list = params_list
        self.L = L
        self.eval_every = eval_every
        
    def __call__(self, step, log_data, driver):
        if step % self.eval_every != 0:
            return True
            
        vs = driver.state
        hi = vs.hilbert
        sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=4)
        ham_dict = {}
        
        # Évaluation séquentielle (1 réplica à la fois = 0 crash mémoire)
        for i, pars in enumerate(self.params_list):
            _vs = vs.get_state(pars)
            
            mc_vs = nk.vqs.MCState(
                sampler=sa_eval, 
                model=_vs.model, 
                variables=_vs.variables, 
                n_samples=256,  # Léger pour ne pas ralentir l'entraînement
                chunk_size=16
            )
            mc_vs.reset()
            
            # Application de ton astuce 50/50 pour la mesure
            sigma_orig = np.array(mc_vs.sampler_state.σ)
            flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
            half = flat_sigma.shape[0] // 2
            flat_sigma[:half, :self.L] = 1
            flat_sigma[half:, :self.L] = -1
            sigma_new = jnp.array(flat_sigma.reshape(sigma_orig.shape))
            mc_vs.sampler_state = mc_vs.sampler_state.replace(**{'σ': sigma_new})
            
            # Mesure
            H_op = create_operator(pars)
            stats = mc_vs.expect(H_op)
            
            # On recrée la structure exacte attendue par ton script de plot
            ham_dict[str(i)] = {
                "Mean": float(np.real(stats.Mean)),
                "Variance": float(stats.variance)
            }
            
        log_data["ham"] = ham_dict
        return True
    
# ==========================================
# 3. LOGGING ET OPTIMISATION
# ==========================================

# Callback de sauvegarde
class SaveState(AbstractCallback):
    _path: str = struct.field(pytree_node=False)
    _prefix: str = struct.field(pytree_node=False)
    _save_every: int = struct.field(pytree_node=False)

    def __init__(self, path: str, save_every: int, prefix: str = "state"):
        self._path = path
        self._prefix = prefix
        self._save_every = save_every
        
        # 1. On force la création du dossier dès l'initialisation si on est sur le master
        if nkpd.is_master_process():
            os.makedirs(self._path, exist_ok=True)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            
            # 2. Sécurité ultime : on vérifie que le dossier existe juste avant d'écrire
            # (Au cas où un nettoyage automatique l'aurait supprimé ou s'il y a un délai)
            if nkpd.is_master_process() and not os.path.exists(self._path):
                os.makedirs(self._path, exist_ok=True)
            
            # Construction du chemin
            path = os.path.join(self._path, f"{self._prefix}_{driver.step_count}.nk")
            
            # Sauvegarde
            driver.state.save(path)

# Optimiseur
learning_rate = optax.linear_schedule(init_value=lr_init, end_value=lr_end, transition_steps=300)
optimizer = optax.sgd(learning_rate)
def cg_solver(A, b):
    # jax.scipy.sparse.linalg.cg renvoie (x, info), on ne garde que x [0]
    return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=diag_shift, linear_solver_fn=cg_solver, chunk_size_bwd=chunk_size_bwd, use_ntk=True)

# Logger
# On initialise le logger avec un dossier temporaire ou final
log = nk.logging.JsonLog("log_data", save_params=False) 

# Dictionnaire META propre
meta = {
    "L": L,
    "graph": "Hypercube 1D",
    "n_dim": 1,
    "pbc": True,
    "hamiltonian": {
        "type": "Ising Disorder", 
        "J": J_val, 
        "h0_train_list": h0_train_list, # On enregistre la liste complète
        "sigma": sigma_disorder
    },
    "model": "ViTFNQS",
    "vit_config": vit_params,
    "sampler": {
        "type": "MetropolisLocal", 
        "n_chains": n_chains, 
        "n_samples": n_samples
    },
    "optimizer": {
        "type": "SGD", 
        "lr_init": lr_init, 
        "lr_end": lr_end, 
        "diag_shift": diag_shift
    },
    "n_iter": n_iter,
    "n_replicas_per_h0": n_replicas,
    "total_configs_train": total_configs_train,
    "seed": seed,
}


# Création de la structure de dossier via ta fonction utilitaire
try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"

# Initialisation du logger et création du dossier
log = nk.logging.JsonLog(os.path.join(run_dir, "log_data.json"), save_params=False)

# AJOUT : SAUVEGARDE DES CONFIGURATIONS DE DÉSORDRE
disorder_path = os.path.join(run_dir, "disorder_configs.npy")
np.save(disorder_path, params_list)
print(f"Configurations de désordre sauvegardées dans : {disorder_path}")

vs.chunk_size = chunk_size

start_time = time.time()

# Lancement du run
# Lancement du run
gs.run(
    n_iter,
    out=log,
    # On met les deux callbacks dans une liste, et SURTOUT pas de paramètre 'obs='
    callback=[
        SaveState(run_dir, 10), 
        ReplicaLogger(params_list, L, eval_every=10)
    ]
)

if "Energy" in log.data:
    log.data["ham"] = log.data["Energy"]

duration = time.time() - start_time
print(f"⏱️ Temps total d'entraînement : {duration:.2f} secondes")

# Mise à jour du meta.json avec le temps d'exécution final
meta["execution_time_seconds"] = duration
import json
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

# ==========================================
# 4. PLOTS ET ANALYSE FINALE
# ==========================================
print('Analyse et sauvegarde...')

# --- 1. Plot Convergence ---
conv_data = []
for i, pars in tqdm(enumerate(vs.parameter_array)):
    if hasattr(log.data["ham"], "__getitem__") and len(log.data["ham"]) > i:
        ham_log = log.data["ham"][i]
        # On prend la partie réelle pour éviter les warnings ComplexWarning
        conv_data.append({"iters": ham_log.iters, "e0": np.real(ham_log.Mean)})

plt.figure()
for _data in conv_data: 
    plt.plot(_data["iters"], _data["e0"], alpha=0.3)
plt.xlabel("Iterations")
plt.ylabel("Energy (Real)")
plt.savefig(os.path.join(run_dir, "convergence.pdf"))
plt.clf()

# --- 2. Calcul des V-scores sur le Train ---
print("Calcul des V-scores finaux sur le Train...")
train_results = {"v_score": []}

for r in tqdm(range(total_configs_train)):
    pars = params_list[r]
    _vs = vs.get_state(pars)
    
    # CORRECTION ICI : On retire l'argument 'hilbert=hi'
    # Le sampler contient déjà l'info sur Hilbert.
    vs_mc = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hi, n_chains=16), 
        model=_vs.model, 
        variables=_vs.variables, 
        n_samples=1024, 
        chunk_size=64
    )
    
    _e = vs_mc.expect(create_operator(pars))
    val = _e.variance / (_e.Mean.real**2 + 1e-12)
    train_results["v_score"].append(val)

v_train = np.array(train_results["v_score"])

# --- 3. Sauvegarde CSV ---
h_mean_train_full = []
for h_val in h0_train_list: 
    h_mean_train_full.extend([h_val] * (n_replicas + 1))

min_len = min(len(h_mean_train_full), len(v_train))

df_train = pd.DataFrame({
    "h_mean": h_mean_train_full[:min_len], 
    "v_score": v_train[:min_len]
})

output_csv = os.path.join(run_dir, "train_results.csv")
df_train.to_csv(output_csv, index=False)
print(f"✅ Terminé ! Résultats sauvegardés dans : {output_csv}")