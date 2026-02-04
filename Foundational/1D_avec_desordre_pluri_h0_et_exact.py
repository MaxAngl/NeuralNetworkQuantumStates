import os
import json
import netket as nk
import netket_foundational as nkf
import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from pathlib import Path
from netket_foundational._src.model.vit import ViTFNQS

# ==========================================
# 1. HYPERPARAMÈTRES ET CONFIGURATION
# ==========================================
# On définit tout ici pour que le 'meta' soit cohérent
seed = 1
k = jax.random.key(seed)
L = 4              # Taille du système
h0_train_list = [ 0.2, 0.8, 1.0, 1.2, 5.0 ]          # Champ moyen
sigma_disorder = 0.1 # Désordre
J_val = 1.0/np.e    # Couplage Ising (défini dans create_operator)
n_replicas = 10    # Nombre de réalisations de désordre
total_configs_train = len(h0_train_list) * n_replicas
chains_per_replica = 4      # <--- ICI : Chaque réplica aura 4 chaînes indépendantes
samples_per_chain = 2      # Nombre de points récoltés par chaque chaîne
n_chains = total_configs_train * chains_per_replica 
n_samples = n_chains * samples_per_chain             
n_iter = 300       # Nombre d'étapes d'optimisation
lr_init = 0.03
lr_end = 0.005
diag_shift = 1e-4
logs_path = "logs"  # Dossier racine pour les logs

h0_test_list = [ 0.85, 1.05, 1.3, 1.5, 3] # Valeurs d'interpolation et d'extrapolation
N_test_per_h0 = 10  # Nombre de configurations de désordre par h0 de test

# Paramètres du modèle ViT
vit_params = {
    "num_layers": 2,
    "d_model": 16,
    "heads": 4,
    "b": 1,
    "L_eff": L,
}

L = meta["L"]
h0_train_list = np.array(meta["hamiltonian"]["h0_train_list"])
n_reps = meta["n_replicas_per_h0"]
n_h0 = len(h0_train_list)
J_val = meta["hamiltonian"]["J"]
sigma_disorder = meta["hamiltonian"]["sigma"]
n_iter = meta["n_iter"]

# Chargement du log NetKet
log_nk = nk.logging.JsonLog(str(log_path).replace(".json", ""))
print(f"✅ Log et Meta chargés pour L={L}")

# Détection de la clé d'énergie (souvent 'ham' d'après ton gs.run)
energy_key = "ham" if "ham" in log_nk.data else "Energy"

# --- 2. RECONSTRUCTION DU SYSTÈME ET CALCUL EXACT ---
hi = nk.hilbert.Spin(0.5, L)

def create_operator(params):
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

# Régénération des paramètres d'entraînement (même seed)
rng = np.random.default_rng(meta["seed"])
params_train = []
for h in h0_train_list:
    params_train.append(rng.normal(h, sigma_disorder, (n_reps, L)))
params_train = np.vstack(params_train)

print("Calcul des énergies exactes pour la convergence...")
exact_energies = []
for p in tqdm(params_train):
    res = nk.exact.lanczos_ed(create_operator(p), k=1)
    exact_energies.append(res[0].item())
exact_energies = np.array(exact_energies)

# --- 3. EXTRACTION DES DONNÉES ---
convergence_errors = [] 
final_metrics = {
    "h0": [], "rel_err": [], "variance": [], 
    "r_hat": [], "v_score": [], "tau": []
}


# Création de la structure de dossier via ta fonction utilitaire
# Note: Assure-toi que save_run renvoie bien le chemin créé si tu veux l'utiliser pour SaveState
try:
    run_dir = save_run(log, meta, create_only=True, base_dir=logs_path)
except Exception as e:
    print(f"Warning: save_run issue ({e}), using default path.")
    run_dir = "checkpoints"



# Initialisation du logger et création du dossier
log = nk.logging.JsonLog(os.path.join(run_dir, "log_data.json"), save_params=False)

start_time = time.time()

# Lancement du run
gs.run(
    n_iter,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    callback=SaveState(run_dir, 10), # Sauvegarde dans le dossier créé
)

duration = time.time() - start_time
print(f"⏱️ Temps total d'entraînement : {duration:.2f} secondes")

# Mise à jour du meta.json avec le temps d'exécution final
meta["execution_time_seconds"] = duration
import json
with open(os.path.join(run_dir, "meta.json"), 'w') as f:
    json.dump(meta, f, indent=4)

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
plt.savefig(os.path.join(run_dir, f"Found_disordered_pluri_h0_L={L}_convergence.pdf"))
plt.clf()

# ==========================================
# 5. TEST SUR NOUVEL ENSEMBLE (CORRIGÉ)
# ==========================================


params_list_test = generate_multi_h0_disorder(h0_test_list, N_test_per_h0, hi.size, sigma_disorder)
N_test_total = params_list_test.shape[0]

vmc_vals = {"Energy": [], "Mz2": [], "V_score": []}

print(f'Computing NQS predictions on test set ({N_test_total} samples)...')

# On avance par paquets de n_replicas (ici 10)
"""for i in tqdm(range(0, N_test_total, total_configs_train)):
    # 1. On prend un lot de 10 paramètres
    batch_params = params_list_test[i : i + total_configs_train]
    
    convergence_errors.append(errs)
    
    # Métriques à la dernière itération
    final_metrics["h0"].append(h_val)
    final_metrics["rel_err"].append(errs[-1])
    final_metrics["variance"].append(replica_data.Variance[-1])
    final_metrics["v_score"].append(replica_data.Variance[-1] / (means[-1].real**2 + 1e-12))
    # R_hat et TauCorr peuvent être optionnels selon le sampler
    final_metrics["r_hat"].append(getattr(replica_data, 'R_hat', [1.0])[-1])
    final_metrics["tau"].append(getattr(replica_data, 'TauCorr', [0.0])[-1])

# --- 4. GRAPHES DE CONVERGENCE PAR FENÊTRE (H0) ---
n_cols = 3
n_rows = (n_h0 + n_cols - 1) // n_cols
fig1, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()
cmap_tab = cm.get_cmap("tab10")

for i in range(n_h0):
    ax = axes[i]
    for r in range(n_reps):
        idx = i * n_reps + r
        ax.plot(iters, convergence_errors[idx], color=cmap_tab(i % 10), alpha=0.4)
    ax.set_yscale('log')
    ax.set_title(f"$h_0 = {h0_train_list[i]}$")
    ax.grid(True, which="both", alpha=0.2)
    if i % n_cols == 0: ax.set_ylabel("Relative Error (Energy)")

for j in range(i + 1, len(axes)): fig1.delaxes(axes[j])
fig1.tight_layout()
fig1.savefig(run_dir / f"Found_L={L}_convergence_per_h0.pdf")

# --- 5. SUPERPOSITION AVEC GRADIENT MAGMA ---
plt.figure(figsize=(10, 6))
norm = plt.Normalize(h0_train_list.min(), h0_train_list.max())
sm = cm.ScalarMappable(cmap="magma", norm=norm)

for i in range(len(convergence_errors)):
    h_val = h0_train_list[i // n_reps]
    plt.plot(iters, convergence_errors[i], color=cm.magma(norm(h_val)), alpha=0.3)

plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Relative Error")
plt.title(f"Superposed Convergence Errors (L={L})")
plt.colorbar(sm, label="$h_0$")
plt.savefig(run_dir / f"Found_L={L}_convergence_magma.pdf")

# --- 6. SCATTER PLOTS DES MÉTRIQUES FINALES ---


    # 2. On injecte le lot dans le vstate
    vs.parameter_array = batch_params
    
    # 3. On évalue chaque réplica du lot individuellement
    for r in range(total_in_batch):
        pars = batch_params[r]
        _vs = vs.get_state(pars) # Récupère l'état spécifique au paramètre r
        
        # Passage en FullSum pour la précision (puisque L=4)
        vs_fs = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, variables=_vs.variables)
        
        _ha = create_operator(pars)
        _e = vs_fs.expect(_ha)
        _o = vs_fs.expect(Mz @ Mz)
        
        # Calcul du V-score
        v_score = _e.variance / (_e.Mean.real**2 + 1e-12)
        
        vmc_vals["Energy"].append(_e.Mean)
        vmc_vals["Mz2"].append(_o.Mean)
        vmc_vals["V_score"].append(v_score) """
        
        
        
for i in tqdm(range(0, N_test_total)):
    pars = params_list_test[i]
    _vs = vs.get_state(pars)

    vs_fs = nk.vqs.FullSumState(
        hilbert=hi,
        model=_vs.model,
        variables=_vs.variables
    )

    _ha = create_operator(pars)
    _e = vs_fs.expect(_ha)
    _o = vs_fs.expect(Mz @ Mz)

    v_score = _e.variance / (_e.Mean.real**2 + 1e-12)

    vmc_vals["Energy"].append(_e.Mean)
    vmc_vals["Mz2"].append(_o.Mean)
    vmc_vals["V_score"].append(v_score)


# --- Calcul Exact de Test (Indispensable pour comparer) ---
exact_vals = {"Energy": [], "Mz2": []}
print('Computing exact values on test set...')
Mz2_op = Mz @ Mz
Mz2_mat = Mz2_op.to_sparse()

for pars in tqdm(params_list_test):
    _ha = create_operator(pars)
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    exact_vals["Energy"].append(E0.item())
    exact_vals["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item().real)

# --- Conversion des résultats VMC en numpy arrays ---
vmc_final = {
    "Energy": np.array([np.real(e) for e in vmc_vals["Energy"]]),
    "Mz2": np.array([np.real(m) for m in vmc_vals["Mz2"]]),
    "V_score": np.array(vmc_vals["V_score"])
}

# --- Conversion des résultats EXACTS en numpy arrays (INDISPENSABLE) ---
ex_energy = np.array(exact_vals["Energy"])
ex_mz2 = np.array(exact_vals["Mz2"])

# Maintenant le calcul d'erreur ne plantera plus
err_test = np.abs(vmc_final['Mz2'] - ex_mz2) / (np.abs(ex_mz2) + 1e-12)

# ==========================================
# 5bis. SAUVEGARDE DES RESULTATS (CSV)
# ==========================================
# On crée une liste des h_mean correspondant à chaque point de test
h_mean_test_full = []
for h_val in h0_test_list:
    h_mean_test_full.extend([h_val] * N_test_per_h0)

df_results = pd.DataFrame({
    "h_mean": h_mean_test_full, # Utilise la liste étendue
    "exact_energy": exact_vals["Energy"],
    "vmc_energy": vmc_final["Energy"],
    "exact_mz2": exact_vals["Mz2"],
    "vmc_mz2": vmc_final["Mz2"],
    "v_score": vmc_final["V_score"]
})

csv_path = os.path.join(run_dir, "test_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"✅ Données de test sauvegardées dans : {csv_path}")

# ==========================================
# 6. ANALYSE COMPARATIVE : TRAIN VS TEST
# ==========================================

print("Ré-évaluation des points de Train pour comparaison...")
vs.parameter_array = params_list  # On remet les 10 points originaux

train_results = {"V_score": [], "Mz2": [], "Ex_Mz2": []}

for r in range(total_configs_train):
    pars = params_list[r]
    _vs = vs.get_state(pars)
    vs_fs = nk.vqs.FullSumState(hilbert=hi, model=_vs.model, variables=_vs.variables)
    
    # VMC
    _e = vs_fs.expect(create_operator(pars))
    _o = vs_fs.expect(Mz @ Mz)
    
    # Exact
    E0, psi0 = nk.exact.lanczos_ed(create_operator(pars), k=1, compute_eigenvectors=True)
    ex_mz2_val = (psi0.reshape(-1).T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item().real
    
    train_results["V_score"].append(_e.variance / (_e.Mean.real**2 + 1e-12))
    train_results["Mz2"].append(_o.Mean.real)
    train_results["Ex_Mz2"].append(ex_mz2_val)

# --- CONVERSION EN ARRAYS POUR CALCULS ---
v_train = np.array(train_results["V_score"])
err_train = np.abs(np.array(train_results["Mz2"]) - np.array(train_results["Ex_Mz2"])) / (np.abs(np.array(train_results["Ex_Mz2"])) + 1e-12)

v_test = vmc_final['V_score']
# err_test a déjà été calculé plus haut avec ex_mz2 converti

# --- TRACÉ DES HISTOGRAMMES ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogramme V-scores
all_v = np.concatenate([v_train, v_test])
bins_v = np.logspace(np.log10(all_v.min() + 1e-18), np.log10(all_v.max() + 1e-2), 25)
ax1.hist(v_test, bins=bins_v, alpha=0.5, label='Test', color='orange', edgecolor='darkorange')
ax1.hist(v_train, bins=bins_v, alpha=0.8, label='Train', color='red', edgecolor='black')
ax1.set_xscale('log')
ax1.set_title("Distribution du V-score")
ax1.legend()

# Histogramme Erreurs Relatives
all_err = np.concatenate([err_train, err_test])
bins_e = np.logspace(np.log10(all_err.min() + 1e-18), np.log10(all_err.max() + 1e-1), 25)
ax2.hist(err_test, bins=bins_e, alpha=0.5, label='Test', color='blue', edgecolor='darkblue')
ax2.hist(err_train, bins=bins_e, alpha=0.8, label='Train', color='cyan', edgecolor='black')
ax2.set_xscale('log')
ax2.set_title("Distribution de l'Erreur Relative $M_z^2$")
ax2.legend()

fig3.delaxes(axes3[-1]) # Supprime le dernier subplot vide
plt.tight_layout()
plt.savefig(os.path.join(run_dir, f"Found_disordered_pluri_h0_L={L}_extrapolation_analysis.pdf"))
print(f"✅ Analyse terminée dans : {run_dir}")

# ==========================================
# 7. CALCUL ET ENREGISTREMENT DES ENERGIES EXACTES
# ==========================================
print("\nComputing and saving exact energies for each training replica...")
exact_energies_train = {}

for i, pars in tqdm(enumerate(params_list)):
    _ha = create_operator(pars)
    E0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()
    exact_energies_train[str(i)] = float(np.real(E0))

# Charger le fichier log_data.log existant
log_data_file = "log_data.log"
with open(log_data_file, 'r') as f:
    log_json = json.load(f)

# Ajouter les énergies exactes au JSON du log
log_json["exact_energies"] = exact_energies_train

# Sauvegarder le fichier log_data.log mis à jour
with open(log_data_file, 'w') as f:
    json.dump(log_json, f, indent=4)
print(f"✅ Exact energies saved to: {log_data_file}")
# 7. CALCUL ET ENREGISTREMENT DES ENERGIES EXACTES
# ==========================================
print("\nComputing and saving exact energies for each training replica...")
exact_energies_train = {}

for i, pars in tqdm(enumerate(params_list)):
    _ha = create_operator(pars)
    E0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()
    exact_energies_train[str(i)] = float(np.real(E0))

# Charger le fichier log_data.log existant
log_data_file = os.path.join(run_dir, "log_data.log")
with open(log_data_file, 'r') as f:
    log_json = json.load(f)

# Ajouter les énergies exactes au JSON du log
log_json["exact_energies"] = exact_energies_train

# Sauvegarder le fichier log_data.log mis à jour
with open(log_data_file, 'w') as f:
    json.dump(log_json, f, indent=4)
print(f"✅ Exact energies saved to: {log_data_file}")
