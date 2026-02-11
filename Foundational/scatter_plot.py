import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
run_dir = "/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates-3/logs/tests_pluri_h0_L=4/run_2026-02-11_08-56-14"  # À adapter
csv_train_path = os.path.join(run_dir, "train_results.csv")
csv_test_path = os.path.join(run_dir, "test_results.csv")

# Chargement des données
df_train = pd.read_csv(csv_train_path)
df_test = pd.read_csv(csv_test_path)

plt.figure(figsize=(12, 7))

# ==========================================
# 1. DONNÉES DE TRAIN (ROUGE)
# ==========================================
# Points individuels
plt.scatter(df_train["h_mean"], df_train["v_score"], 
            color='red', alpha=0.4, edgecolors='none', label='Réplicas (Train)')

# Ligne moyenne
train_summary = df_train.groupby("h_mean")["v_score"].mean().reset_index()
plt.plot(train_summary["h_mean"], train_summary["v_score"], 
         color='darkred', marker='o', markersize=8, linestyle='-', linewidth=2, label='Moyenne Train')

# ==========================================
# 2. DONNÉES DE TEST (BLEU)
# ==========================================
# Points individuels
plt.scatter(df_test["h_mean"], df_test["v_score"], 
            color='royalblue', alpha=0.4, edgecolors='none', label='Réplicas (Test)')

# Ligne moyenne
test_summary = df_test.groupby("h_mean")["v_score"].mean().reset_index()
plt.plot(test_summary["h_mean"], test_summary["v_score"], 
         color='navy', marker='s', markersize=8, linestyle='--', linewidth=2, label='Moyenne Test')

# ==========================================
# CONFIGURATION DES AXES ET STYLE
# ==========================================
plt.yscale('log')  # Échelle log cruciale pour le V-score

# On combine les h_mean pour avoir des graduations claires
all_h = sorted(list(set(df_train["h_mean"].unique()) | set(df_test["h_mean"].unique())))
plt.xticks(all_h, rotation=45)

plt.xlabel(r'Champ transverse moyen $h_{mean}$', fontsize=12)
plt.ylabel('V-score (échelle log)', fontsize=12)
plt.title(f'Comparaison V-score : Train vs Test (L=4)', fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(frameon=True, loc='best')

# Sauvegarde
plot_path = os.path.join(run_dir, "v_score_comparison_train_test.pdf")
plt.tight_layout()
plt.savefig(plot_path)

print(f"✅ Graphique comparatif sauvegardé : {plot_path}")
plt.show()