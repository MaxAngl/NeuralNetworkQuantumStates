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

# Charger les énergies exactes depuis le JSON du log
exact_energies = log_json.get("exact_energies", {})
if not exact_energies:
    print(f"⚠️  Warning: 'exact_energies' not found in log data")

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

print(f"✅ Energy convergence plot saved successfully!")

# ==========================================
# PLOT DES ERREURS RELATIVES PAR H0
# ==========================================
if exact_energies:
    print("\nCreating relative error plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for h0_idx, h0_val in enumerate(h0_train_list):
        ax = axes[h0_idx]
        replica_indices = h0_to_replicas[h0_val]
        
        # Tracer une courbe par réplica
        for replica_num, replica_idx in enumerate(replica_indices):
            if replica_idx < len(ham_data):
                energy_data = ham_data[replica_idx]
                
                # Extraire les itérations et les énergies moyennes VMC
                iters = np.array(energy_data.get("iters", []))
                means_data = energy_data.get("Mean", {})
                
                if isinstance(means_data, dict):
                    means = np.array(means_data.get("real", []))
                else:
                    means = np.array(means_data)
                
                means = np.real(np.array(means))
                
                # Récupérer l'énergie exacte pour ce réplica
                replica_str = str(replica_idx)
                if replica_str in exact_energies:
                    exact_E = exact_energies[replica_str]
                    
                    # Calculer l'erreur relative
                    if len(iters) > 0 and len(means) > 0:
                        rel_error = np.abs(means - exact_E) / (np.abs(exact_E) + 1e-12)
                        ax.semilogy(iters, rel_error, 
                                   label=f"Replica {replica_num}", 
                                   color=colors[replica_num],
                                   linewidth=1.5,
                                   alpha=0.8)
        
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Relative Error |E_VMC - E_exact| / |E_exact|", fontsize=10)
        ax.set_title(f"$h_0 = {h0_val}$", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2, loc='best')
        ax.grid(True, alpha=0.3, which='both')
    
    # Masquer les subplots inutilisés
    for idx in range(len(h0_train_list), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path_error = os.path.join(run_dir, f"relative_error_by_h0_L={L}.pdf")
    plt.savefig(output_path_error, dpi=150, bbox_inches='tight')
    print(f"✅ Relative error plot saved: {output_path_error}")
    plt.close()
else:
    print("⚠️  exact_energies.json not available, skipping relative error plot")