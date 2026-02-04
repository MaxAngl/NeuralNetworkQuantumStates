import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
logs_path = "logs"
run_dir = None

# Trouver le dernier run
if os.path.exists(logs_path):
    runs = [d for d in os.listdir(logs_path) if d.startswith("run_")]
    if runs:
        run_dir = os.path.join(logs_path, sorted(runs)[-1])
        print(f"✅ Using run: {run_dir}")

if run_dir is None:
    print("❌ No run directory found!")
    sys.exit(1)

# Charger la configuration depuis meta.json
meta_path = os.path.join(run_dir, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
n_replicas = meta["n_replicas_per_h0"]
total_configs_train = meta["total_configs_train"]

print(f"L = {L}")
print(f"h0_train_list = {h0_train_list}")
print(f"n_replicas_per_h0 = {n_replicas}")
print(f"total_configs_train = {total_configs_train}")

# ==========================================
# CHARGER LES DONNEES DE LOG
# ==========================================
# Charger le fichier log_data.log depuis la racine
log_path = os.path.join(os.path.dirname(run_dir), "..", "log_data.log")
if not os.path.exists(log_path):
    print(f"❌ Log file not found: {log_path}")
    sys.exit(1)

print(f"Loading log data from: {log_path}")

# Charger le JSON
with open(log_path, 'r') as f:
    log_json = json.load(f)

# Extraire les données d'énergie (ham)
if "ham" not in log_json:
    print("❌ 'ham' key not found in log data")
    print(f"Available keys: {list(log_json.keys())}")
    sys.exit(1)

ham_data = log_json["ham"]
print(f"Number of replicas in log: {len(ham_data)}")


# ==========================================
# PRÉPARER LES DONNÉES
# ==========================================
# Créer un mapping: h0_value -> list of replica indices
h0_to_replicas = {}
for idx, h0_val in enumerate(h0_train_list):
    h0_to_replicas[h0_val] = list(range(idx * n_replicas, (idx + 1) * n_replicas))

print(f"h0 to replicas mapping: {h0_to_replicas}")

# ==========================================
# CRÉER LES PLOTS
# ==========================================
n_h0 = len(h0_train_list)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Colormap pour les replicas
cmap = plt.colormaps['tab10']
colors = [cmap(i / n_replicas) for i in range(n_replicas)]

for h0_idx, h0_val in enumerate(h0_train_list):
    ax = axes[h0_idx]
    replica_indices = h0_to_replicas[h0_val]
    
    # Tracer une courbe par réplica
    for replica_num, replica_idx in enumerate(replica_indices):
        if replica_idx < len(ham_data):
            energy_data = ham_data[replica_idx]
            
            # Extraire les itérations et les énergies moyennes
            iters = np.array(energy_data.get("iters", []))
            means_data = energy_data.get("Mean", {})
            
            if isinstance(means_data, dict):
                means = np.array(means_data.get("real", []))
            else:
                means = np.array(means_data)
            
            # Prendre la partie réelle si complexe
            means = np.real(np.array(means))
            
            if len(iters) > 0 and len(means) > 0:
                ax.plot(iters, means, 
                       label=f"Replica {replica_num}", 
                       color=colors[replica_num],
                       linewidth=1.5,
                       alpha=0.8)
    
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Energy", fontsize=10)
    ax.set_title(f"$h_0 = {h0_val}$", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

# Masquer les subplots inutilisés
for idx in range(n_h0, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
output_path = os.path.join(run_dir, f"energy_convergence_by_h0_L={L}.pdf")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Plot saved: {output_path}")
plt.close()

# ==========================================
# PLOT ALTERNATIF : TOUTES LES ÉNERGIES SUR UN SEUL GRAPHE AVEC LÉGENDE COULEUR
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))

# Créer une colormap étendue pour tous les replicas de tous les h0
n_total = total_configs_train
cmap_full = plt.colormaps['tab20c']
colors_full = [cmap_full(i % 20 / 20) for i in range(n_total)]

# Tracer toutes les énergies
for replica_idx in range(min(len(ham_data), total_configs_train)):
    energy_data = ham_data[replica_idx]
    
    iters = np.array(energy_data.get("iters", []))
    means_data = energy_data.get("Mean", {})
    
    if isinstance(means_data, dict):
        means = np.array(means_data.get("real", []))
    else:
        means = np.array(means_data)
    
    means = np.real(np.array(means))
    
    if len(iters) > 0 and len(means) > 0:
        # Déterminer h0 et replica_num correspondants
        h0_idx = replica_idx // n_replicas
        replica_num = replica_idx % n_replicas
        if h0_idx < len(h0_train_list):
            h0_val = h0_train_list[h0_idx]
            label = f"h₀={h0_val}, r={replica_num}"
            ax.plot(iters, means, 
                   label=label,
                   color=colors_full[replica_idx],
                   linewidth=1.2,
                   alpha=0.7)

ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Energy", fontsize=12)
ax.set_title("Energy Convergence for All Replicas", fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=3, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path_all = os.path.join(run_dir, f"energy_convergence_all_replicas_L={L}.pdf")
plt.savefig(output_path_all, dpi=150, bbox_inches='tight')
print(f"✅ Combined plot saved: {output_path_all}")

plt.show()
