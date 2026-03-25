import os
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
import warnings
import glob

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 👇 MODIFIER LE CHEMIN ICI 👇
RUN_DIR = r"/users/eleves-a/2024/rami.chagnaud/Documents/logs/2D_FNQS/Run_2D_L8_FNQS"
IS_2D = True

# Paramètres de test : h0 fixés et balayage de sigma
H0_FIXED_LIST = [1.0, 3.0, 5.0]  # Ex: Phase ordonnée, critique, paramagnétique
SIGMA_TEST_LIST = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
N_TEST_PER_CONFIG = 5

# ==========================================
# 2. SETUP ET CHARGEMENT DU MODÈLE
# ==========================================
if not os.path.exists(RUN_DIR):
    print(f"❌ Erreur : Le dossier {RUN_DIR} n'existe pas.")
    sys.exit(1)

meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
n_spins = L**2 if IS_2D else L
vit_params = meta["vit_config"]

print(f"🔹 Configuration QFI : {'2D' if IS_2D else '1D'}, L={L}, n_spins={n_spins}")

hi = nk.hilbert.Spin(0.5, n_spins)
ps = nkf.ParameterSpace(N=n_spins, min=0, max=10*max(H0_FIXED_LIST))

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
    two_dimensional=IS_2D,
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=16)

checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

with zipfile.ZipFile(last_checkpoint, 'r') as zf:
    target = [name for name in zf.namelist() if name.endswith('.msgpack')][-1]
    state_dict = flax.serialization.msgpack_restore(zf.read(target))

vars_dict = state_dict.get('variables', state_dict.get('model', {}).get('variables', state_dict))
vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
print(f"✅ Poids chargés depuis : {os.path.basename(last_checkpoint)}")

# ==========================================
# 3. CALCUL DE L'INFORMATION DE FISHER
# ==========================================
def compute_qfi_simplified(state_obj, disorder_configs_batch):
    batch_size, n_spins = disorder_configs_batch.shape
    trace_qfi = []
    max_eig_qfi = []
    h_epsilon = 1e-3

    for b in range(batch_size):
        W = disorder_configs_batch[b]
        qfi = np.zeros((n_spins, n_spins))

        _vs_base = state_obj.get_state(W)
        samples = _vs_base.sample().reshape(-1, n_spins)

        for i in range(n_spins):
            for j in range(i, n_spins):
                W_pp = W.copy(); W_pp[i] += h_epsilon; W_pp[j] += h_epsilon
                W_pm = W.copy(); W_pm[i] += h_epsilon; W_pm[j] -= h_epsilon
                W_mp = W.copy(); W_mp[i] -= h_epsilon; W_mp[j] += h_epsilon
                W_mm = W.copy(); W_mm[i] -= h_epsilon; W_mm[j] -= h_epsilon

                log_pp = np.mean(state_obj.get_state(W_pp).log_value(samples).real)
                log_pm = np.mean(state_obj.get_state(W_pm).log_value(samples).real)
                log_mp = np.mean(state_obj.get_state(W_mp).log_value(samples).real)
                log_mm = np.mean(state_obj.get_state(W_mm).log_value(samples).real)

                mixed_deriv = (log_pp - log_pm - log_mp + log_mm) / (4 * h_epsilon ** 2)

                qfi[i, j] = mixed_deriv
                qfi[j, i] = mixed_deriv

        qfi = 0.5 * (qfi + qfi.T)
        eigvals = np.linalg.eigvalsh(qfi)
        eigvals = np.maximum(eigvals, 1e-10) 

        trace_qfi.append(np.sum(eigvals))
        max_eig_qfi.append(np.max(eigvals))

    return np.array(trace_qfi), np.array(max_eig_qfi)

# ==========================================
# 4. BOUCLE PRINCIPALE ET AGREGATION
# ==========================================
rng = np.random.default_rng(42)
results = {h0: {"trace_mean": [], "trace_std": [], "eig_mean": [], "eig_std": []} for h0 in H0_FIXED_LIST}

for h0 in H0_FIXED_LIST:
    print(f"\n▶ Calculs pour h0 = {h0}")
    for sigma in tqdm(SIGMA_TEST_LIST, desc=f"Balayage sigma"):
        configs = np.abs(rng.normal(loc=h0, scale=sigma, size=(N_TEST_PER_CONFIG, n_spins)))
        
        tr_qfi, max_qfi = compute_qfi_simplified(vs, configs)
        
        results[h0]["trace_mean"].append(np.mean(tr_qfi))
        results[h0]["trace_std"].append(np.std(tr_qfi))
        results[h0]["eig_mean"].append(np.mean(max_qfi))
        results[h0]["eig_std"].append(np.std(max_qfi))

# ==========================================
# 5. TRACÉS
# ==========================================
print("\n📈 Génération des graphiques QFI vs Sigma...")

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
colors = plt.cm.plasma(np.linspace(0, 0.8, len(H0_FIXED_LIST)))

for idx, h0 in enumerate(H0_FIXED_LIST):
    c = colors[idx]
    
    # Trace Tr(g)
    axes[0].errorbar(SIGMA_TEST_LIST, results[h0]["trace_mean"], yerr=results[h0]["trace_std"], 
                     fmt='o-', color=c, linewidth=2, capsize=4, label=rf"$h_0 = {h0}$")
    
    # Valeur propre max λ_max(g)
    axes[1].errorbar(SIGMA_TEST_LIST, results[h0]["eig_mean"], yerr=results[h0]["eig_std"], 
                     fmt='s-', color=c, linewidth=2, capsize=4, label=rf"$h_0 = {h0}$")

axes[0].set_xlabel(r"Disorder Strength $\sigma$", fontsize=12)
axes[0].set_ylabel(r"$\text{Tr}(g)$", fontsize=12)
axes[0].set_title(f"QFI Trace vs Disorder (L={L})", fontsize=14)
axes[0].grid(True, which="both", ls="--", alpha=0.3)
axes[0].legend()

axes[1].set_xlabel(r"Disorder Strength $\sigma$", fontsize=12)
axes[1].set_ylabel(r"$\lambda_{\max}(g)$", fontsize=12)
axes[1].set_title(f"QFI Max Eigenvalue vs Disorder (L={L})", fontsize=14)
axes[1].grid(True, which="both", ls="--", alpha=0.3)
axes[1].legend()

plt.tight_layout()
out_path = os.path.join(RUN_DIR, f"QFI_vs_sigma_L={L}.pdf")
plt.savefig(out_path)
print(f"✅ Graphe sauvegardé : {out_path}")