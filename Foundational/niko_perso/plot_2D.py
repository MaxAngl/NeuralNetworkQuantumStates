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
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import zipfile
import warnings

# Suppression du warning obsolÃ¨te si persistant
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================
IS_2D = True 
RUN_DIR = r"/users/eleves-a/2024/nikola.audit/NeuralNetworkQuantumStates/logs/2D_FNQS/run_2026-03-11_01-19-42"

H0_TEST_LIST = [0.5, 1.5, 2.2, 2.5, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.5, 4.0, 5.0]
N_TEST_PER_H0 = 20 
nb_steps_thermalization = 10 

# ==========================================
# 2. SETUP ET CHARGEMENT
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if not os.path.exists(RUN_DIR):
    print(f"âŒ Erreur : Le dossier {RUN_DIR} n'existe pas.")
    sys.exit(1)

meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
n_spins = L**2 if IS_2D else L
J_val = meta["hamiltonian"]["J"]
sigma_disorder = meta["hamiltonian"]["sigma"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
vit_params = meta["vit_config"]

print(f"ðŸ”¹ Configuration : {'2D' if IS_2D else '1D'}, Grille={L}x{L}, n_spins={n_spins}, b={vit_params['b']}, L_eff={vit_params['L_eff']}")

hi = nk.hilbert.Spin(0.5, n_spins)
ps = nkf.ParameterSpace(N=n_spins, min=0, max=10*max(H0_TEST_LIST + h0_train_list))

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

sa = nk.sampler.MetropolisLocal(hi, n_chains=1) 
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
if not checkpoints:
    print("âŒ Aucun checkpoint (.nk) trouvÃ©.")
    sys.exit(1)
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

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

vars_dict = state_dict.get('variables', state_dict.get('model', {}).get('variables', state_dict))
vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)

# ==========================================
# 3. PREPARATION DATASETS
# ==========================================
print("\nðŸ“¦ PrÃ©paration des datasets...")
disorder_file = os.path.join(RUN_DIR, "disorder_configs.npy")
train_params = np.load(disorder_file) if os.path.exists(disorder_file) else []
train_h0 = []
if len(train_params) > 0:
    n_h0_train = len(h0_train_list)
    replicas_per_h0_train = len(train_params) // n_h0_train
    for h0 in h0_train_list:
        train_h0.extend([h0] * replicas_per_h0_train)

rng_gen = np.random.default_rng(seed=42)
test_params = np.vstack([rng_gen.normal(loc=h0, scale=sigma_disorder, size=(N_TEST_PER_H0, n_spins)) for h0 in H0_TEST_LIST])
test_h0 = [h0 for h0 in H0_TEST_LIST for _ in range(N_TEST_PER_H0)]

# ==========================================
# 4. FONCTIONS DE CALCUL (1D/2D UNIFIÃ‰ES)
# ==========================================
def get_hamiltonian_op(params):
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(n_spins))
    
    if IS_2D:
        ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i % L + 1) % L + (i // L) * L) for i in range(n_spins))
        ha_ZZ += sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + L) % n_spins) for i in range(n_spins))
    else:
        ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + 1) % n_spins) for i in range(n_spins))
        
    return -ha_X - J_val * ha_ZZ



def compute_metrics(params_batch, desc="metrics"):
    rel_errors, v_scores, r_hats, tau_corrs = [], [], [], []
    sa_multi = nk.sampler.MetropolisLocal(hi, n_chains=16)

    # --- SEUIL DE SÉCURITÉ ---
    # On désactive l'Exact Diagonalization si n_spins > 20
    RUN_EXACT = True if n_spins <= 20 else False
    if not RUN_EXACT:
        print(f"⚠️ n_spins={n_spins} trop grand. Calcul Exact (ED) désactivé pour éviter l'OOM.")

    for pars in tqdm(params_batch, desc=desc):
        H = get_hamiltonian_op(pars)
        _vs = vs.get_state(pars)
        
        # 1. Calcul Exact (ED) uniquement si possible
        if RUN_EXACT:
            try:
                E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
                res_fs = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, variables=_vs.variables).expect(H)
                rel_errors.append(abs(res_fs.Mean.real - E_exact) / abs(E_exact))
            except Exception as e:
                print(f"Erreur ED : {e}")
                rel_errors.append(np.nan)
        else:
            rel_errors.append(np.nan) # On met NaN pour ne pas casser les plots

        # 2. V-Score (FullSum ou MC)
        # Attention : FullSumState explose aussi en mémoire pour n_spins > 20 !
        if n_spins <= 20:
            res_val = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, variables=_vs.variables).expect(H)
            v_scores.append(res_val.variance / (res_val.Mean.real**2 + 1e-12))
        else:
            # Pour n_spins > 20, on utilise UNIQUEMENT l'estimation Monte Carlo
            mc_vs = nk.vqs.MCState(sampler=sa_multi, model=_vs.model, variables=_vs.variables, n_samples=4096)
            res_mc = mc_vs.expect(H)
            v_scores.append(res_mc.variance / (res_mc.Mean.real**2 + 1e-12))
            r_hats.append(res_mc.R_hat)
            tau_corrs.append(res_mc.tau_corr)

    return {
        "rel_error": np.array(rel_errors), 
        "v_score": np.array(v_scores), 
        "r_hat": np.array(r_hats) if r_hats else np.zeros(len(params_batch)), 
        "tau_corr": np.array(tau_corrs) if tau_corrs else np.zeros(len(params_batch))
    }



# ==========================================
# 5. EXECUTION ET PLOTS
# ==========================================
print("\nðŸš€ Lancement des calculs...")
metrics_test = compute_metrics(test_params, desc="Calculs Test")
metrics_train = compute_metrics(train_params, desc="Calculs Train") if len(train_params) > 0 else {}

# --- PLOT 1 : RELATIVE ERROR ---
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
plt.title(f"Energy Accuracy: Train vs Test (Grid={L}x{L})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)
out1 = os.path.join(RUN_DIR, f"rel_error_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(out1)
print(f"âœ… Plot RelError sauvegardÃ© : {out1}")
plt.close()

# --- PLOT 2 : V-SCORE ---
plt.figure(figsize=(10, 6))
plt.scatter(test_h0, metrics_test["v_score"], color='crimson', alpha=0.3, s=40, label='Test Replicas', edgecolor='none', marker='o', zorder=1)
if metrics_train:
    plt.scatter(train_h0, metrics_train["v_score"], color='royalblue', alpha=0.3, s=40, label='Train Replicas', edgecolor='none', marker='^', zorder=1)

df_test = pd.DataFrame({"h0": test_h0, "v": metrics_test["v_score"]})
mean_test = df_test.groupby("h0")["v"].mean()
plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

if metrics_train:
    df_train = pd.DataFrame({"h0": train_h0, "v": metrics_train["v_score"]})
    mean_train = df_train.groupby("h0")["v"].mean()
    plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

plt.yscale('log')
plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"V-score ($\text{Var}(E) / E^2$)", fontsize=12)
plt.title(f"Accuracy Landscape: Train vs Test (Grid={L}x{L})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)
out2 = os.path.join(RUN_DIR, f"vscore_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(out2)
print(f"âœ… Plot V-score sauvegardÃ© : {out2}")
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
plt.title(f"Convergence Diagnostics ($\hat{{R}}$): Train vs Test (Grid={L}x{L})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)
out3 = os.path.join(RUN_DIR, f"rhat_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(out3)
print(f"âœ… Plot R-hat sauvegardÃ© : {out3}")
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
plt.title(rf"Sampling Efficiency ($\tau_{{corr}}$): Train vs Test (Grid={L}x{L})", fontsize=14)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.5) 
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='upper left', frameon=True, fontsize=10)
out4 = os.path.join(RUN_DIR, f"tau_corr_scatter_train_test_L={L}.pdf")
plt.tight_layout()
plt.savefig(out4)
print(f"âœ… Plot Tau-Corr sauvegardÃ© : {out4}")
plt.close()

# --- PLOT 5 : CONVERGENCE HISTORY ---
print("\nðŸ“ˆ GÃ©nÃ©ration du Plot de Convergence (Historique)...")
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
    
    exact_energies_train = []
    if len(train_params) > 0:
        for pars in train_params:
            H = get_hamiltonian_op(pars)
            E = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
            exact_energies_train.append(E)

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
            if global_idx < len(ham_data) and global_idx < len(exact_energies_train):
                data = ham_data[global_idx]
                iters = np.array(data.get("iters", []))
                means_raw = data.get("Mean", {})
                if isinstance(means_raw, dict):
                    means = np.array(means_raw.get("real", means_raw.get("value", [])))
                else:
                    means = np.real(np.array(means_raw))
                e_ex = exact_energies_train[global_idx]
                if len(iters) > 0:
                    rel_err = np.abs((means - e_ex) / e_ex)
                    ax.plot(iters, rel_err, color=current_color, alpha=0.5, linewidth=1)

        ax.set_yscale('log')
        ax.set_title(f"$h_0 = {h0_val}$", fontweight='bold', color=current_color)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rel Error $|E_{VMC} - E_{ex}|/|E_{ex}|$", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    for idx in range(n_h0, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    out5 = os.path.join(RUN_DIR, f"relative_error_energy_convergence_L={L}.pdf")
    plt.savefig(out5)
    print(f"âœ… Plot Convergence sauvegardÃ© : {out5}")
    plt.close()
else:
    print("âš ï¸ Pas de fichier log trouvÃ©, plot de convergence ignorÃ©.")

print("\n TERMINÃ‰ ! Tous les graphiques sont gÃ©nÃ©rÃ©s.")