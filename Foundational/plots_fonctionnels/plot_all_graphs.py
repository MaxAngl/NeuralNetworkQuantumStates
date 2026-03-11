import os
import sys
import glob
import json
import numpy as np
import jax
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import msgpack
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import zipfile
import jax.numpy as jnp

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 👇 MODIFIEZ LE CHEMIN ICI 👇
RUN_DIR = r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/Foundational/logs/run_2026-03-09_22-03-41"

H0_TEST_LIST = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.5, 3.5, 4.5] 
N_TEST_PER_H0 = 20 
prob_global_flip = 0.05

# ==========================================
# 2. SETUP ET CHARGEMENT
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)
from flip_rules import GlobalFlipRule

# On crée une "enveloppe" sécurisée autour de votre règle
class SafeGlobalFlipRule(GlobalFlipRule):
    def transition(self, sampler, machine, parameters, state, key, sigma):
        # On appelle votre règle originale
        sigma_new, log_prob = super().transition(sampler, machine, parameters, state, key, sigma)
        # On force la conversion vers le type attendu par l'échantillonneur (int8)
        return jnp.asarray(sigma_new, dtype=sigma.dtype), log_prob

if not os.path.exists(RUN_DIR):
    print(f"❌ Erreur : Le dossier {RUN_DIR} n'existe pas.")
    sys.exit(1)

meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
J_val = meta["hamiltonian"]["J"]
sigma_disorder = meta["hamiltonian"]["sigma"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
vit_params = meta["vit_config"]

# 🔥 OPTION AUTOMATIQUE POUR LES GRANDS SYSTÈMES 🔥
# Désactive la diagonalisation exacte et le plot d'erreur relative si L > 20
PLOT_RELATIVE_ERROR = True if L <= 20 else False

print(f"🔹 Configuration : L={L}, J={J_val:.4f}")
print(f"🔹 Calcul Erreur Relative (Exact) : {'ACTIVÉ' if PLOT_RELATIVE_ERROR else 'DÉSACTIVÉ (L trop grand)'}")

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(H0_TEST_LIST + h0_train_list))

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

sa = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=1)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
if not checkpoints:
    print("❌ Aucun checkpoint (.nk) trouvé.")
    sys.exit(1)
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
print(f"🔹 Chargement des poids depuis : {os.path.basename(last_checkpoint)}")

state_dict = None
if not zipfile.is_zipfile(last_checkpoint):
    with open(last_checkpoint, 'rb') as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(last_checkpoint, 'r') as zf:
        file_list = zf.namelist()
        candidates = [f for f in file_list if f.endswith('.msgpack')]
        target_file = sorted(candidates, key=len)[-1]
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

vars_dict = None
if 'variables' in state_dict: vars_dict = state_dict['variables']
elif 'model' in state_dict and 'variables' in state_dict['model']: vars_dict = state_dict['model']['variables']
elif 'params' in state_dict: vars_dict = state_dict
elif 'vqs' in state_dict and 'variables' in state_dict['vqs']: vars_dict = state_dict['vqs']['variables']
if vars_dict is None: vars_dict = state_dict

try:
    vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
    print("✅ Poids injectés avec succès !")
except Exception as e:
    print(f"❌ Erreur injection Flax : {e}")
    sys.exit(1)

# ==========================================
# 3. PREPARATION DATASETS
# ==========================================
print("\n📦 Préparation des datasets...")

# TRAIN
disorder_file = os.path.join(RUN_DIR, "disorder.configs.npy")
if not os.path.exists(disorder_file):
    disorder_file = os.path.join(RUN_DIR, "disorder_configs.npy")

train_params = []
train_h0 = []
if os.path.exists(disorder_file):
    train_params = np.load(disorder_file)
    n_h0_train = len(h0_train_list)
    replicas_per_h0_train = len(train_params) // n_h0_train
    for h0 in h0_train_list:
        train_h0.extend([h0] * replicas_per_h0_train)
else:
    print("⚠️  Train : Fichier disorder.configs.npy introuvable.")

# TEST
rng_gen = np.random.default_rng(seed=42)
test_params = []
test_h0 = []
for h0 in H0_TEST_LIST:
    configs = np.abs(rng_gen.normal(loc=h0, scale=sigma_disorder, size=(N_TEST_PER_H0, L)))
    test_params.append(configs)
    test_h0.extend([h0] * N_TEST_PER_H0)
test_params = np.vstack(test_params)


# ==========================================
# 4. FONCTIONS DE CALCUL (VMC ONLY)
# ==========================================
def get_hamiltonian_op(params):
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L))
    return -ha_X - J_val * ha_ZZ

def compute_metrics(params_batch, desc="metrics"):
    rel_errors = []
    v_scores_mc = [] 
    r_hats = []
    tau_corrs = []
    
    sa_multi = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=16)

    # Initialisation MCState une seule fois
    _vs_init = vs.get_state(params_batch[0])
    mc_vs = nk.vqs.MCState(
        sampler=sa_multi,
        model=_vs_init.model,
        variables=_vs_init.variables,
        n_samples=2048, 
        n_discard_per_chain=500,
        chunk_size=32
    )

    for pars in tqdm(params_batch, desc=desc):
        H = get_hamiltonian_op(pars)
        
        mc_vs.variables = vs.get_state(pars).variables
        mc_vs.reset()

        # 👇 ON INTERCEPTE ET ON FORCE LE 50/50 POUR LE TEST 👇
        sigma_orig = mc_vs.sampler_state.σ
        flat_sigma = sigma_orig.reshape(-1, sigma_orig.shape[-1])
        half = flat_sigma.shape[0] // 2
        
        flat_sigma = flat_sigma.at[:half, :L].set(1)
        flat_sigma = flat_sigma.at[half:, :L].set(-1)
        
        sigma_new = flat_sigma.reshape(sigma_orig.shape)
        mc_vs.sampler_state = mc_vs.sampler_state.replace(σ=sigma_new)
        # 👆 FIN DE L'ASTUCE 👆

        stats_mc = mc_vs.expect(H)
        
        E_vmc_est = stats_mc.Mean.real
        
        # 1. EXACT (Seulement si la taille le permet)
        if PLOT_RELATIVE_ERROR:
            E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
            err = abs(E_vmc_est - E_exact) / abs(E_exact)
            rel_errors.append(err)
            
        # 2. MONTE CARLO
        r_hats.append(stats_mc.R_hat)
        tau_corrs.append(stats_mc.tau_corr)
        v_mc = stats_mc.variance / (E_vmc_est**2 + 1e-12)
        v_scores_mc.append(v_mc)

    return {
        "rel_error": np.array(rel_errors) if PLOT_RELATIVE_ERROR else None,
        "v_score_mc": np.array(v_scores_mc),
        "r_hat": np.array(r_hats),
        "tau_corr": np.array(tau_corrs)
    }

# ==========================================
# 5. EXECUTION CALCULS
# ==========================================
print("\n🚀 Lancement des calculs...")
metrics_test = compute_metrics(test_params, desc="Calculs Test")

metrics_train = {}
if len(train_params) > 0:
    metrics_train = compute_metrics(train_params, desc="Calculs Train")


# ==========================================
# 6. PLOTS
# ==========================================


# --- PLOT 1 : RELATIVE ERROR (OPTIONNEL) ---
if PLOT_RELATIVE_ERROR:
    plt.figure(figsize=(10, 6))
    plt.scatter(test_h0, metrics_test["rel_error"], color='crimson', alpha=0.3, s=40, label='Test Replicas', edgecolor='none', marker='o', zorder=1)
    if metrics_train:
        plt.scatter(train_h0, metrics_train["rel_error"], color='royalblue', alpha=0.3, s=40, label='Train Replicas', edgecolor='none', marker='^', zorder=1)

    df_test = pd.DataFrame({"h0": test_h0, "err": metrics_test["rel_error"]})
    mean_test = df_test.groupby("h0")["err"].mean()
    plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
    plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

    if metrics_train:
        df_train = pd.DataFrame({"h0": train_h0, "err": metrics_train["rel_error"]})
        mean_train = df_train.groupby("h0")["err"].mean()
        plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
        plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

    plt.yscale('log')
    plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
    plt.ylabel(r"Relative Energy Error $|E_{VMC} - E_{exact}| / |E_{exact}|$", fontsize=12)
    plt.title(f"Energy Accuracy: Train vs Test (L={L})", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='best', frameon=True, fontsize=10)
    out1 = os.path.join(RUN_DIR, f"rel_error_scatter_train_test_L={L}.pdf")
    plt.tight_layout()
    plt.savefig(out1)
    print(f"✅ Plot RelError sauvegardé : {out1}")
    plt.close()


# --- PLOT 2 : V-SCORE (MONTE CARLO) ---
plt.figure(figsize=(10, 6))
plt.scatter(test_h0, metrics_test["v_score_mc"], color='crimson', alpha=0.3, s=40, label='Test Replicas', edgecolor='none', marker='o', zorder=1)
if metrics_train:
    plt.scatter(train_h0, metrics_train["v_score_mc"], color='royalblue', alpha=0.3, s=40, label='Train Replicas', edgecolor='none', marker='^', zorder=1)

df_test = pd.DataFrame({"h0": test_h0, "v": metrics_test["v_score_mc"]})
mean_test = df_test.groupby("h0")["v"].mean()
plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

if metrics_train:
    df_train = pd.DataFrame({"h0": train_h0, "v": metrics_train["v_score_mc"]})
    mean_train = df_train.groupby("h0")["v"].mean()
    plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

plt.yscale('log')
plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"V-score (MC) ($\text{Var}(E) / E^2$)", fontsize=12)
plt.title(f"Accuracy Landscape (MC Est.): Train vs Test (L={L})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)
out2 = os.path.join(RUN_DIR, f"vscore_mc_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(out2)
print(f"✅ Plot V-score (MC) sauvegardé : {out2}")
plt.close()


# --- PLOT 3 : R-HAT ---
plt.figure(figsize=(10, 6))
plt.scatter(test_h0, metrics_test["r_hat"], color='crimson', alpha=0.3, s=40, label='Test Replicas', edgecolor='none', marker='o', zorder=1)
if metrics_train:
    plt.scatter(train_h0, metrics_train["r_hat"], color='royalblue', alpha=0.3, s=40, label='Train Replicas', edgecolor='none', marker='^', zorder=1)

df_test = pd.DataFrame({"h0": test_h0, "r": metrics_test["r_hat"]})
mean_test = df_test.groupby("h0")["r"].mean()
plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

if metrics_train:
    df_train = pd.DataFrame({"h0": train_h0, "r": metrics_train["r_hat"]})
    mean_train = df_train.groupby("h0")["r"].mean()
    plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

plt.yscale('linear')
plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"Gelman-Rubin $\hat{R}$", fontsize=12)
plt.title(f"Convergence Diagnostics ($\hat{{R}}$): Train vs Test (L={L})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)
out3 = os.path.join(RUN_DIR, f"rhat_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(out3)
print(f"✅ Plot R-hat sauvegardé : {out3}")
plt.close()


# --- PLOT 4 : TAU CORR ---
plt.figure(figsize=(10, 6))
plt.scatter(test_h0, metrics_test["tau_corr"], color='crimson', alpha=0.3, s=40, label='Test Replicas', edgecolor='none', marker='o', zorder=1)
if metrics_train:
    plt.scatter(train_h0, metrics_train["tau_corr"], color='royalblue', alpha=0.3, s=40, label='Train Replicas', edgecolor='none', marker='^', zorder=1)

df_test = pd.DataFrame({"h0": test_h0, "t": metrics_test["tau_corr"]})
mean_test = df_test.groupby("h0")["t"].mean()
plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

if metrics_train:
    df_train = pd.DataFrame({"h0": train_h0, "t": metrics_train["tau_corr"]})
    mean_train = df_train.groupby("h0")["t"].mean()
    plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

plt.yscale('linear')
plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"Correlation Time $\tau_{corr}$", fontsize=12)
plt.title(rf"Sampling Efficiency ($\tau_{{corr}}$): Train vs Test (L={L})", fontsize=14)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5) 
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='upper left', frameon=True, fontsize=10)
out4 = os.path.join(RUN_DIR, f"tau_corr_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(out4)
print(f"✅ Plot Tau-Corr sauvegardé : {out4}")
plt.close()



# --- PLOT 5 : CONVERGENCE HISTORY (ORIGINAL) ---
print("\n📈 Génération du Plot de Convergence (Historique)...")
candidates = ["log_data.json.log", "log_data.log", "log_data.json"]
log_path = None
for name in candidates:
    p = os.path.join(RUN_DIR, name)
    if os.path.exists(p):
        log_path = p
        break

if log_path:
    with open(log_path, 'r') as f:
        log_json = json.load(f)
    ham_data = log_json.get("ham", log_json.get("Energy", []))
    
    n_h0 = len(h0_train_list)
    cols = 3
    rows = (n_h0 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, n_h0)]
    
    n_rep_per_h0_log = len(train_params) // n_h0
    
    for idx, h0_val in enumerate(h0_train_list):
        start_idx = idx * n_rep_per_h0_log
        ax = axes[idx]
        current_color = colors[idx]
        
        for rep_offset in range(n_rep_per_h0_log):
            global_idx = start_idx + rep_offset
            
            # 1. Récupération sécurisée (gère les listes ET les dictionnaires JSON)
            data = None
            if isinstance(ham_data, dict):
                data = ham_data.get(str(global_idx), ham_data.get(global_idx))
            elif isinstance(ham_data, list) and global_idx < len(ham_data):
                data = ham_data[global_idx]
            
            # 2. LA SÉCURITÉ ANTI-CRASH (Si la donnée n'existe pas, on passe au suivant)
            if data is None:
                print(f"⚠️ Index {global_idx} introuvable dans le log, on l'ignore.")
                continue
            
            # 3. Extraction (ne s'exécute que si data existe vraiment)
            iters = np.array(data.get("iters", []))
            
            # Extraction Mean
            means_raw = data.get("Mean", {})
            if isinstance(means_raw, dict):
                means = np.array(means_raw.get("real", means_raw.get("value", [])))
            else:
                means = np.real(np.array(means_raw))
            
            # Extraction Variance
            vars_raw = data.get("Variance", {})
            if isinstance(vars_raw, dict):
                variances = np.array(vars_raw.get("real", vars_raw.get("value", [])))
            else:
                variances = np.real(np.array(vars_raw))
            
            if len(iters) > 0 and len(means) > 0 and len(variances) > 0:
                plot_metric = variances / (means**2 + 1e-12)
                ax.plot(iters, plot_metric, color=current_color, alpha=0.5, linewidth=1)

        ax.set_yscale('log')
        ax.set_title(f"$h_0 = {h0_val}$", fontweight='bold', color=current_color)
        ax.set_xlabel("Iteration")
        # On garde le ylabel exact d'avant comme demandé
        ax.set_ylabel("V-score $\\text{Var}(E)/E^2$", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    for idx in range(n_h0, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    # On garde le nom de fichier exact d'avant comme demandé
    out5 = os.path.join(RUN_DIR, f"vscore_convergence_L={L}.pdf")
    plt.savefig(out5)
    print(f"✅ Plot Convergence sauvegardé : {out5}")
    plt.close()

else:
    print("⚠️ Pas de fichier log trouvé, plot de convergence ignoré.")

print("\n✨ TERMINÉ ! Tous les graphiques sont générés.")