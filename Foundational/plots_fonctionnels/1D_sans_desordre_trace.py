import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

from matplotlib.pylab import svd
import netket as nk
import netket_foundational as nkf
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as  plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from netket.utils import struct
import netket_pro.distributed as nkpd
from netket_foundational._src.model.vit import ViTFNQS
import netket.utils.timing as timing
from advanced_drivers._src.callbacks import AbstractCallback
from netket.driver import AbstractVariationalDriver
from netket.stats import statistics
from netket.sampler import rules

# --- CONFIGURATION ---
output_dir = Path("/users/eleves-a/2024/adel.mana/Documents/NeuralNetworkQuantumStates")
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = output_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Liste des tailles de système à étudier
system_sizes = [32]  # Toutes divisibles par 2
h_range = jnp.array([0.03,0.05,0.07,0.10,0.15,0.2,0.3,0.4,0.5])
n_iterations = 300

class GlobalFlipRule(rules.MetropolisRule):
    prob_global: float = struct.field(pytree_node=False, default=0.1)
    def __init__(self, prob_global=0.4):
        """
        prob_global: probabilité de proposer un flip global vs local
        """
        self.prob_global = prob_global
        super().__init__()
    
    def transition(rule, sampler, machine, parameters, state, key, σ):
        # Avec probabilité prob_global, flip tous les spins
        # Sinon, flip local
        
        key, subkey = jax.random.split(key)
        #crée 2 keys à partir d'une seule
        do_global = jax.random.uniform(subkey) < rule.prob_global
        #c'est un booléen qui vaut true si le nombre aléatoire généré est inférieur à rule.prob_global
        def global_flip(σ):
            return -σ  # Flip tous les spins
        
        def local_flip(σ):
            # Flip un spin aléatoire
            key2, subkey2 = jax.random.split(key)
            site = jax.random.randint(subkey2, (), 0, σ.shape[-1])
            #choisit au hasard un site
            return σ.at[..., site].multiply(-1)
        
        σ_new = jax.lax.cond(do_global, global_flip, local_flip, σ)
        #syntaxe condensée pour faire un if/else
        return σ_new, None
colors = cm.viridis(np.linspace(0, 1, len(system_sizes)))

# Dictionnaires pour stocker tous les résultats
all_convergence = {}  # {L: {"iteration": [], "energy": [], "rel_error": []}}
all_magnetization = {}  # {L: {"h": [], "Mz2": [], "Mz2_err": []}}

# --- CALLBACK DE SAUVEGARDE ---
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
        if driver.step_count % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)

# --- BOUCLE PRINCIPALE SUR LES TAILLES ---
for L in tqdm(system_sizes, desc="Tailles de système"):
    print(f"\n{'='*60}")
    print(f"▶ Calcul VMC pour L={L}")
    print(f"{'='*60}")
    
    # Hilbert space
    hi = nk.hilbert.Spin(0.5, L)
    ps = nkf.ParameterSpace(N=1, min=0.0, max=4.0)
    
    # Modèle ViT
    ma = ViTFNQS(
        num_layers=2,
        d_model=12,
        heads=4,
        L_eff=hi.size // 2,
        n_coups=ps.size,
        b=2,
        complex=False,
        disorder=False,
        transl_invariant=True,
        two_dimensional=False,
    )
    
    # Configuration sampler
    n_replicas = len(h_range)
    n_chains = n_replicas * 4
    if n_chains % n_replicas != 0:
        n_chains = (n_chains // n_replicas) * n_replicas  # Assurer divisibilité
    n_samples=n_chains*2

    sa = nk.sampler.MetropolisSampler(
        hi,
        rule=GlobalFlipRule(prob_global=0.01), 
        n_chains=n_chains,
    )
    #sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains )
    vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=n_replicas,n_samples=n_samples, seed=1)
    vs.parameter_array = h_range.reshape(-1, 1)
    
    # Opérateurs
    def create_operator(params, hilbert=hi):
        assert params.shape == (1,)
        h = params[0]
        ha_X = sum(nkf.operator.sigmax(hilbert, i) for i in range(hilbert.size))
        ha_ZZ = sum(
            nkf.operator.sigmaz(hilbert, i) @ nkf.operator.sigmaz(hilbert, (i + 1) % hilbert.size)
            for i in range(hilbert.size)
        )
        return -h * ha_X - ha_ZZ
    
    ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
    
    Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))
    mz_p = nkf.operator.ParametrizedOperator(hi, ps, lambda _: Mz @ Mz)
    
    # --- OPTIMISATION VMC ---
    optimizer = optax.sgd(0.005)
    #def cg_solver(A, b):
    # jax.scipy.sparse.linalg.cg renvoie (x, info), on ne garde que x [0]
        #return jax.scipy.sparse.linalg.cg(A, b, tol=1e-4)[0]
    gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=1e-3)
    
    log_path = output_dir / f"log_data_L{L}"
    log = nk.logging.JsonLog(str(log_path))
    with timing.Timer() as timer:
        gs.run(
            n_iterations,
            out=log,
            obs={"ham": ha_p, "mz2": mz_p},
            step_size=10,
            #callback=SaveState(str(checkpoint_dir / f"L{L}"), save_every=50),
            timeit=True,
        )
        print(f"\n  Timing breakdown pour L={L}:")
        print(timer)
    
    # --- EXTRACTION CONVERGENCE ---
    # On prend la moyenne sur toutes les répliques pour la convergence
    iters = np.asarray(log.data["ham"][0].iters)  # Prendre les iters de la 1ère réplique (identiques pour toutes)
    energies_all_replicas = np.array([np.asarray(log.data["ham"][i].Mean) for i in range(n_replicas)])  # Shape: (n_replicas, n_iter)
    energy_mean = np.mean(energies_all_replicas, axis=0)  # Moyenne sur les répliques  # Shape: (n_iter, n_replicas)

    
    all_convergence[L] = {
        "iteration": iters,
        "energy": energy_mean,
    }
    
    # --- ÉVALUATION Mz² FINALE pour chaque h ---
    print(f"\n  → Évaluation Mz² pour L={L}")
    mz2_values = []
    mz2_errors = []
    h_eval = jnp.linspace(0, 4, 100)
    rhat_values = []
    for pars in tqdm(h_eval.reshape(-1, 1), desc=f"    Mz² L={L}"):
        _vs = vs.get_state(pars)
        n_chains_vs = _vs.sampler_state.σ.shape[0]
        σ_init = jnp.ones(_vs.sampler_state.σ.shape,dtype=jnp.int8)
        σ_init = σ_init.at[n_chains_vs//2:].multiply(-1)
        _vs.sampler_state = _vs.sampler_state.replace(σ=σ_init)
        n_therm = 40 if 0.7 < pars[0] < 1.3 else 10
        for _ in range(n_therm): 
            _vs.sample()
        result = _vs.expect(Mz @ Mz)
        samples = _vs.samples
        # Récupérer les samples
         # Convert to float to avoid NaN/integer dtype issues
        samples_float = jnp.asarray(samples, dtype=jnp.float32)
    
    # Reshape samples to 2D: (n_chains, -1)
        if samples_float.ndim > 2:
            samples_reshaped = samples_float.reshape(samples_float.shape[0], -1)
        else:
            samples_reshaped = samples_float
        rhat = statistics(samples_reshaped).R_hat # pour récupérer le R-hat
        mz2_values.append(float(result.Mean.real))
        rhat_values.append(float(rhat.max()))
        #Ceci récupère 
        mz2_errors.append(float(result.Sigma.real))
    
    all_magnetization[L] = {
        "h": np.array(h_eval),
        "Mz2": np.array(mz2_values),
        "Mz2_err": np.array(mz2_errors),
        "Rhat": np.array(rhat_values),
    }
    
    print(f"✓ L={L} terminé\n")

# --- PLOTTING : 2 SUBPLOTS DANS UN SEUL PDF ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1 : Convergence de l'énergie
ax1 = axes[0]
for i, L in enumerate(system_sizes):
    data = all_convergence[L]
    ax1.plot(
        data["iteration"],
        data["energy"],
        color=colors[i],
        linewidth=2,
        label=f"L={L}",
    )

ax1.set_xlabel("Itération", fontsize=12)
ax1.set_ylabel("Énergie moyenne", fontsize=12)
ax1.set_title("Convergence de l'énergie en fonction des itérations", fontsize=13, fontweight="bold")
ax1.legend(loc="best", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale("log")

# Subplot 2 : Magnétisation Mz² vs h
ax2 = axes[1]
for i, L in enumerate(system_sizes):
    data = all_magnetization[L]
    ax2.errorbar(
        data["h"],
        data["Mz2"],
        yerr=data["Mz2_err"],
        color=colors[i],
        marker="o",
        markersize=4,
        linewidth=1.5,
        capsize=2,
        label=f"L={L}",
    )

ax2.set_xlabel("Champ magnétique H", fontsize=12)
ax2.set_ylabel(r"Magnétisation $M_z^2$", fontsize=12)
ax2.set_title(r"$M_z^2$ en fonction du champ magnétique", fontsize=13, fontweight="bold")
ax2.legend(loc="best", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 4)
ax2.set_ylim(0, 1.05)


# Subplot 3 : R-hat vs h
ax3 = axes[2]
for i, L in enumerate(system_sizes):
    data = all_magnetization[L]
    ax3.plot(
        data["h"],
        data["Rhat"],
        color=colors[i],
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=f"L={L}",
    )

ax3.axhline(y=1.1, color='red', linestyle='--', label='Seuil R̂=1.1')
ax3.set_xlabel("Champ magnétique H", fontsize=12)
ax3.set_ylabel(r"$\hat{R}$ (convergence)", fontsize=12)
ax3.set_title("Convergence de l'échantillonnage", fontsize=13, fontweight="bold")
ax3.legend(loc="best", fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 4)

plt.tight_layout()
plt.savefig(output_dir / "resultats.pdf", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "resultats.png", dpi=150, bbox_inches="tight")
# Vérification que les fichiers sont bien créés
pdf_path = output_dir / "resultats.pdf"
png_path = output_dir / "resultats.png"

print(f"\n{'='*60}")
print("📊 FICHIERS GÉNÉRÉS :")
print(f"   PDF existe : {pdf_path.exists()} → {pdf_path}")
print(f"   PNG existe : {png_path.exists()} → {png_path}")
if pdf_path.exists():
    print(f"   Taille PDF : {pdf_path.stat().st_size / 1024:.1f} KB")
if png_path.exists():
    print(f"   Taille PNG : {png_path.stat().st_size / 1024:.1f} KB")
print(f"{'='*60}")

# --- SAUVEGARDE CSV ---
# Convergence
#conv_list = []
#for L in system_sizes:
 #   data = all_convergence[L]
  #  temp_df = pd.DataFrame({
   #     "L": L,
    #    "iteration": data["iteration"],
     #   "energy": data["energy"],
    #})
    #conv_list.append(temp_df)

#full_conv_df = pd.concat(conv_list, ignore_index=True)
#full_conv_df.to_csv(output_dir / "convergence_energie.csv", index=False)

# Magnétisation
#mag_list = []
#for L in system_sizes:
 #   data = all_magnetization[L]
  #  temp_df = pd.DataFrame({
   #     "L": L,
    #    "h": data["h"],
     #   "Mz2": data["Mz2"],
      #  "Mz2_err": data["Mz2_err"],
    #})
    #mag_list.append(temp_df)

#full_mag_df = pd.concat(mag_list, ignore_index=True)
#full_mag_df.to_csv(output_dir / "magnetisation.csv", index=False)

#print(f"\n{'='*60}")
#print(f"✅ Simulation terminée pour toutes les tailles !")
#print(f"   Résultats dans : {output_dir}")
#print(f"   - Graphiques : resultats_complets.pdf/.png")
#print(f"   - Données : convergence_energie.csv, magnetisation.csv")
#print(f"{'='*60}")