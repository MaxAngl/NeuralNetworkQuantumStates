import os
import jax 

# Initialisation du cluster JAX
try:
    jax.distributed.initialize()
    print(f"JAX Cluster initialisé : Processus {jax.process_index()} / {jax.process_count()}")
except Exception as e:
    print(f"Simple exécution locale ou erreur d'init : {e}")

# Remplace ton ancien print MPI par celui-ci, plus moderne
print(f"👋 Bonjour depuis le noeud {jax.process_index()} sur {jax.process_count()} !")

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
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax
from scipy.stats import gaussian_kde
from netket.utils import struct
import matplotlib.pyplot as plt
from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks import AbstractCallback
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
L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 600

h0_train_list = [ 0.1, 0.4, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 2.5, 4.0 ]
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
lr_init = 0.03
lr_end = 0.005
diag_shift = 2e-4
logs_path = os.path.join(foundational_dir, "logs")

# --- CALCUL AUTOMATIQUE ET SYSTÉMATIQUE DU CHUNK_SIZE ---
TARGET_CHUNK = 10 

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
    "num_layers": 4,
    "d_model": 60,
    "heads": 10,
    "b": 4,
    "L_eff": L//4,  #L/b
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
    params_list: np.ndarray = struct.field(pytree_node=False)
    L: int = struct.field(pytree_node=False)
    eval_every: int = struct.field(pytree_node=False)
    run_dir: str = struct.field(pytree_node=False) # Ajout du dossier de sauvegarde
    
    # On stocke les historiques en interne
    iters: list = struct.field(pytree_node=False, default_factory=list)
    energies: list = struct.field(pytree_node=False, default_factory=list)
    variances: list = struct.field(pytree_node=False, default_factory=list)
    
    def __init__(self, params_list, L, run_dir, eval_every=10):
        self.params_list = params_list
        self.L = L
        self.run_dir = run_dir
        self.eval_every = eval_every
        self.iters = []
        self.energies = []
        self.variances = []
        
    def __call__(self, step, log_data, driver):
        if step % self.eval_every != 0:
            return True
            
        vs = driver.state
        hi = vs.hilbert
        sa_eval = nk.sampler.MetropolisLocal(hi, n_chains=4)
        
        step_energies = []
        step_variances = []
        
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
            
            # On stocke les valeurs brutes pour ce réplica
            step_energies.append(float(np.real(stats.Mean)))
            step_variances.append(float(stats.variance))
            
        # On ajoute la ligne de ce step à l'historique complet
        self.iters.append(step)
        self.energies.append(step_energies)
        self.variances.append(step_variances)
        
        # SAUVEGARDE PHYSIQUE DIRECTE (Le cœur de la solution)
        if nkpd.is_master_process():
            np.save(os.path.join(self.run_dir, "replica_iters.npy"), np.array(self.iters))
            np.save(os.path.join(self.run_dir, "replica_energies.npy"), np.array(self.energies))
            np.save(os.path.join(self.run_dir, "replica_variances.npy"), np.array(self.variances))
            
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
gs.run(
    n_iter,
    out=log,
    # On met les deux callbacks dans une liste, et SURTOUT pas de paramètre 'obs='
    callback=[
        SaveState(run_dir, 10), 
        ReplicaLogger(params_list, L, run_dir=run_dir, eval_every=10) # <-- MODIFICATION ICI
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
# Correction : on utilise nkpd au lieu de mpi
if nkpd.is_master_process():
    print('Analyse finale et génération des graphiques MCMC...')
    train_results = {"v_score": [], "r_hat": []}

    for r in tqdm(range(total_configs_train)):
        pars = params_list[r]
        _vs = vs.get_state(pars)
        
        vs_mc = nk.vqs.MCState(
            sampler=nk.sampler.MetropolisLocal(hi, n_chains=16), 
            model=_vs.model, variables=_vs.variables, 
            n_samples=1024, chunk_size=64
        )
        
        _e = vs_mc.expect(create_operator(pars))
        train_results["v_score"].append(float(_e.variance / (_e.Mean.real**2 + 1e-12)))
        train_results["r_hat"].append(float(getattr(_e, 'R_hat', np.nan)))

    v_train = np.array(train_results["v_score"])
    r_train = np.array(train_results["r_hat"])

    # Sauvegarde CSV
    h_mean_train_full = []
    for h_val in h0_train_list: h_mean_train_full.extend([h_val] * (n_replicas + 1))
    
    df_train = pd.DataFrame({
        "h_mean": h_mean_train_full[:len(v_train)], 
        "v_score": v_train, "r_hat": r_train
    })
    df_train.to_csv(os.path.join(run_dir, "train_results.csv"), index=False)

    # --- PLOT A : Grille de Convergence V-score (Basé sur les .npy du Callback) ---
    num_h0 = len(h0_train_list)
    cols = 3
    rows = (num_h0 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_h0))

    try:
        # On charge les historiques sauvegardés par notre ReplicaLogger
        hist_iters = np.load(os.path.join(run_dir, "replica_iters.npy"))
        hist_energies = np.load(os.path.join(run_dir, "replica_energies.npy"))
        hist_variances = np.load(os.path.join(run_dir, "replica_variances.npy"))
        
        for idx, h0 in enumerate(h0_train_list):
            ax = axes[idx // cols, idx % cols]
            c = colors[idx]
            start_idx = idx * (n_replicas + 1)
            end_idx = start_idx + (n_replicas + 1)
            
            for rep_i in range(start_idx, end_idx):
                rep_energies = hist_energies[:, rep_i]
                rep_variances = hist_variances[:, rep_i]
                
                v_scores = rep_variances / (rep_energies**2 + 1e-12)
                ax.plot(hist_iters, v_scores, alpha=0.4, linewidth=1.0, color=c)
                
            ax.set_yscale('log')
            ax.set_title(rf"$h_0 = {h0}$", color=c, fontweight='bold')
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"V-score $Var(E)/E^2$")
            ax.grid(True, which="both", ls="--", alpha=0.3)

    except FileNotFoundError:
        print("⚠️ Fichiers .npy introuvables. Le tracé de la grille de convergence a été ignoré.")

    for idx in range(num_h0, rows * cols): fig.delaxes(axes[idx // cols, idx % cols])
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"vscore_convergence_grid_L={L}.pdf"))
    plt.clf()

    # --- PLOT B : Scatter plot V-score ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_train["h_mean"], df_train["v_score"], alpha=0.3, color='royalblue', marker='^', label='Train Replicas')
    mean_vscores = df_train.groupby("h_mean")["v_score"].mean().reset_index()
    plt.plot(mean_vscores["h_mean"], mean_vscores["v_score"], marker='s', linestyle='--', color='mediumblue', label='Train Mean', markersize=7)
    plt.yscale('log')
    plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
    plt.ylabel(r"V-score (MC) $\left( Var(E)/E^2 \right)$", fontsize=12)
    plt.title(f"Accuracy Landscape (MC Est.): V-score (L={L})", fontsize=14)
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"vscore_scatter_L={L}.pdf"))
    plt.clf()

    # --- PLOT C : Scatter plot R-hat ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df_train["h_mean"], df_train["r_hat"], alpha=0.3, color='royalblue', marker='^', label='Train Replicas')
    mean_rhats = df_train.groupby("h_mean")["r_hat"].mean().reset_index()
    plt.plot(mean_rhats["h_mean"], mean_rhats["r_hat"], marker='s', linestyle='--', color='darkblue', label='Train Mean', markersize=7)
    
    plt.axhline(y=1.05, color='black', linestyle=':', linewidth=2, label='Convergence Threshold (1.05)')
    
    plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
    plt.ylabel(r"Gelman-Rubin $\hat{R}$", fontsize=12)
    plt.title(f"Convergence Diagnostics ($\hat{{R}}$): Train only (L={L})", fontsize=14)
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"rhat_scatter_L={L}.pdf"))
    plt.clf()

    print("✅ Run terminé à 100%. Graphiques et CSV générés avec succès !")