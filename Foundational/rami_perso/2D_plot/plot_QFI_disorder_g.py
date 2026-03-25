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
RUN_DIR = r"/users/eleves-a/2024/rami.chagnaud/Documents/logs/2D_FNQS/Run_2D_L6_FNQS"
IS_2D = True

# Paramètres de test : balayage de la transition de phase
H0_TEST_LIST = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
N_TEST_PER_H0 = 5

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
sigma = meta["hamiltonian"]["sigma"]
vit_params = meta["vit_config"]

print(f"🔹 Configuration QFI : {'2D' if IS_2D else '1D'}, L={L}, n_spins={n_spins}, sigma={sigma}")

hi = nk.hilbert.Spin(0.5, n_spins)
ps = nkf.ParameterSpace(N=n_spins, min=0, max=10*max(H0_TEST_LIST))

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
# 3. GENERATION DES CONFIGS DE DESORDRE
# ==========================================
rng = np.random.default_rng(42)
test_configs = []
h0_labels = []

for h0 in H0_TEST_LIST:
    # Valeur absolue pour correspondre au domaine positif du champ
    configs = np.abs(rng.normal(loc=h0, scale=sigma, size=(N_TEST_PER_H0, n_spins)))
    test_configs.append(configs)
    h0_labels.extend([h0] * N_TEST_PER_H0)

test_configs = np.vstack(test_configs)
h0_labels = np.array(h0_labels)

# ==========================================
# 4. CALCUL DE L'INFORMATION DE FISHER
# ==========================================
def compute_qfi_simplified(state_obj, disorder_configs_batch):
    batch_size, n_spins = disorder_configs_batch.shape
    trace_qfi = []
    max_eig_qfi = []
    h_epsilon = 1e-3

    for b in tqdm(range(batch_size), desc="Calcul de la Matrice QFI"):
        W = disorder_configs_batch[b]
        qfi = np.zeros((n_spins, n_spins))

        _vs_base = state_obj.get_state(W)
        # 🔹 CORRECTION : Aplatissement du tenseur (n_chains, n_samples, n_spins) -> (batch, n_spins)
        samples = _vs_base.sample().reshape(-1, n_spins)

        for i in range(n_spins):
            for j in range(i, n_spins):
                # Différences finies centrées (4 points)
                W_pp = W.copy(); W_pp[i] += h_epsilon; W_pp[j] += h_epsilon
                W_pm = W.copy(); W_pm[i] += h_epsilon; W_pm[j] -= h_epsilon
                W_mp = W.copy(); W_mp[i] -= h_epsilon; W_mp[j] += h_epsilon
                W_mm = W.copy(); W_mm[i] -= h_epsilon; W_mm[j] -= h_epsilon

                # Évaluation sur le batch aplati
                log_pp = np.mean(state_obj.get_state(W_pp).log_value(samples).real)
                log_pm = np.mean(state_obj.get_state(W_pm).log_value(samples).real)
                log_mp = np.mean(state_obj.get_state(W_mp).log_value(samples).real)
                log_mm = np.mean(state_obj.get_state(W_mm).log_value(samples).real)

                mixed_deriv = (log_pp - log_pm - log_mp + log_mm) / (4 * h_epsilon ** 2)

                qfi[i, j] = mixed_deriv
                qfi[j, i] = mixed_deriv

        # Symétrisation et régularisation (semi-définie positive)
        qfi = 0.5 * (qfi + qfi.T)
        eigvals = np.linalg.eigvalsh(qfi)
        eigvals = np.maximum(eigvals, 1e-10) 

        trace_qfi.append(np.sum(eigvals))
        max_eig_qfi.append(np.max(eigvals))

    return np.array(trace_qfi), np.array(max_eig_qfi)

trace_qfi, max_eig_qfi = compute_qfi_simplified(vs, test_configs)

# ==========================================
# 5. AGREGATION STATISTIQUE ET PLOT
# ==========================================
print("\n📈 Génération des graphiques QFI...")

trace_mean = [np.mean(trace_qfi[h0_labels == h]) for h in H0_TEST_LIST]
trace_std = [np.std(trace_qfi[h0_labels == h]) for h in H0_TEST_LIST]
eig_mean = [np.mean(max_eig_qfi[h0_labels == h]) for h in H0_TEST_LIST]
eig_std = [np.std(max_eig_qfi[h0_labels == h]) for h in H0_TEST_LIST]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# --- Trace Tr(g) ---
axes[0].errorbar(H0_TEST_LIST, trace_mean, yerr=trace_std, fmt='o-', color='#1f77b4', linewidth=2, capsize=5)
axes[0].set_xlabel(r"Transverse Field $h_0$", fontsize=12)
axes[0].set_ylabel(r"$\text{Tr}(g)$", fontsize=12)
axes[0].set_title(f"QFI Trace (System sensitivity) - L={L}", fontsize=14)
axes[0].grid(True, which="both", ls="--", alpha=0.3)

# --- Valeur propre max λ_max(g) ---
axes[1].errorbar(H0_TEST_LIST, eig_mean, yerr=eig_std, fmt='s-', color='#ff7f0e', linewidth=2, capsize=5)
axes[1].set_xlabel(r"Transverse Field $h_0$", fontsize=12)
axes[1].set_ylabel(r"$\lambda_{\max}(g)$", fontsize=12)
axes[1].set_title(f"QFI Max Eigenvalue (Critical direction) - L={L}", fontsize=14)
axes[1].grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
out_path = os.path.join(RUN_DIR, f"QFI_disorder_L={L}.pdf")
plt.savefig(out_path)
print(f"✅ Graphe sauvegardé : {out_path}")