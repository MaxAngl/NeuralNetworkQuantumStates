"""
Script pour tracer le temps d'exécution par itération en fonction du nombre de paramètres RBM
Pour une RBM: nb_params = L * (2*alpha + 1)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from scipy.optimize import curve_fit

# Configuration - chercher dans tous les logs
log_dirs = [
    Path("logs"),
]

# Collecter les données
data_points = []

for log_dir in log_dirs:
    if not log_dir.exists():
        continue
    
    # Parcourir récursivement pour trouver tous les meta.json
    for meta_file in log_dir.rglob("meta.json"):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                
                L = meta.get('L')
                model_type = meta.get('model', 'RBM')
                exec_time = meta.get('execution_time_seconds')
                n_iter = meta.get('n_iter')
                
                # Calculer le nombre de paramètres selon le type de modèle
                nb_params = None
                alpha = None
                
                if model_type == 'RBM':
                    alpha = meta.get('alpha')
                    if L and alpha:
                        # RBM: L visible units, alpha*L hidden units
                        # params = L (bias visible) + alpha*L (bias hidden) + L*alpha*L (weights)
                        nb_params = L * (2 * alpha + 1)
                
                elif model_type == 'CNN':
                    # Pour CNN, essayer de lire les paramètres du modèle
                    # Si disponible dans meta, sinon on peut estimer
                    nb_params = meta.get('nb_params')  # Si stocké directement
                    if not nb_params and L:
                        # Estimation pour un CNN simple (à ajuster selon votre architecture)
                        # Par exemple: conv layers + dense layers
                        # Ceci est une estimation, idéalement il faudrait le nombre exact
                        features = meta.get('features', [16, 32])  # valeurs par défaut
                        if isinstance(features, list) and len(features) > 0:
                            # Estimation simplifiée (à adapter selon votre CNN)
                            nb_params = sum(features) * L  # très approximatif
                
                if L and nb_params and exec_time and n_iter and n_iter > 0:
                    time_per_iter = exec_time / n_iter
                    
                    data_points.append({
                        'L': L,
                        'alpha': alpha if alpha else 0,
                        'model': model_type,
                        'nb_params': nb_params,
                        'time_per_iter': time_per_iter,
                        'file': str(meta_file)
                    })
        except Exception as e:
            print(f"Erreur lecture {meta_file}: {e}")

print(f"Total de {len(data_points)} runs trouvés")

# Grouper par (L, alpha) et calculer moyennes
from collections import defaultdict
grouped = defaultdict(list)

for point in data_points:
    key = (point['L'], point['alpha'], point['nb_params'])
    grouped[key].append(point['time_per_iter'])

# Filtrer les outliers et calculer moyennes
L_list = []
alpha_list = []
nb_params_list = []
time_mean_list = []
time_std_list = []

for (L, alpha, nb_params), times in sorted(grouped.items()):
    times_array = np.array(times)
    
    # Filtrage outliers par IQR si assez de données
    if len(times_array) >= 3:
        q1 = np.percentile(times_array, 25)
        q3 = np.percentile(times_array, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        times_filtered = times_array[(times_array >= lower) & (times_array <= upper)]
    else:
        times_filtered = times_array
    
    # Filtrage spécial pour L=100 (garder groupe à ~4s)
    if L == 100 and len(times_filtered) > 1:
        times_filtered = times_filtered[times_filtered > 3.0]
    
    if len(times_filtered) > 0:
        L_list.append(L)
        alpha_list.append(alpha)
        nb_params_list.append(nb_params)
        time_mean_list.append(np.mean(times_filtered))
        time_std_list.append(np.std(times_filtered) if len(times_filtered) > 1 else 0)
        print(f"L={L}, alpha={alpha}, nb_params={nb_params}: {len(times_filtered)} runs, "
              f"temps moyen = {np.mean(times_filtered):.4f} ± {np.std(times_filtered):.4f} s")

# Convertir en arrays
nb_params_array = np.array(nb_params_list)
time_array = np.array(time_mean_list)
time_std_array = np.array(time_std_list)
alpha_array = np.array(alpha_list)

# Définir les fonctions de fit
def linear_fit(n, a, b):
    """Fit linéaire: t = a*n + b"""
    return a * n + b

def power_fit(n, a, b):
    """Fit en loi de puissance: t = a*n^b"""
    return a * n**b

# Effectuer les fits - garder seulement loi de puissance
fits = {}

try:
    popt_power, _ = curve_fit(power_fit, nb_params_array, time_array, p0=[1e-6, 1.5])
    fits['Loi de puissance'] = (power_fit, popt_power, f"t = {popt_power[0]:.2e}·n^{popt_power[1]:.2f}")
    print(f"\nFit loi de puissance: t = {popt_power[0]:.2e}·n^{popt_power[1]:.2f}")
except Exception as e:
    print(f"Erreur fit loi de puissance: {e}")

# Calculer les R²
r_squared_dict = {}
for fit_name, (fit_func, popt, _) in fits.items():
    y_pred = fit_func(nb_params_array, *popt)
    ss_res = np.sum((time_array - y_pred)**2)
    ss_tot = np.sum((time_array - np.mean(time_array))**2)
    r_squared = 1 - (ss_res / ss_tot)
    r_squared_dict[fit_name] = r_squared

# Créer la figure avec deux subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Points sans distinction de couleur par alpha
ax1.errorbar(nb_params_array, time_array, yerr=time_std_array,
            fmt='o', markersize=6, capsize=4, label='Données',
            color='black', alpha=0.7)

# Tracer le fit
n_fit = np.linspace(nb_params_array.min(), nb_params_array.max(), 200)
for fit_name, (fit_func, popt, label) in fits.items():
    r2 = r_squared_dict[fit_name]
    ax1.plot(n_fit, fit_func(n_fit, *popt), '--',
            label=f'{fit_name} (R²={r2:.4f}): {label}', color='red', linewidth=2)

ax1.set_xlabel('Nombre de paramètres RBM', fontsize=14, fontweight='bold')
ax1.set_ylabel('Temps par itération (s)', fontsize=14, fontweight='bold')
ax1.set_title('Temps d\'exécution vs nombre de paramètres RBM\n(échelle linéaire)',
             fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', framealpha=0.9, fontsize=9)

# Graphique 2: Log-log
# Erreurs relatives pour échelle log
rel_err = np.minimum(time_std_array / (time_array + 1e-10), 0.5)
yerr_lower = time_array * rel_err
yerr_upper = time_array * rel_err

ax2.errorbar(nb_params_array, time_array, yerr=[yerr_lower, yerr_upper],
            fmt='o', markersize=6, capsize=4, label='Données',
            color='black', alpha=0.7, elinewidth=1.5)

for fit_name, (fit_func, popt, label) in fits.items():
    r2 = r_squared_dict[fit_name]
    ax2.plot(n_fit, fit_func(n_fit, *popt), '--',
            label=f'{fit_name} (R²={r2:.4f})', color='red', linewidth=2)

ax2.set_xlabel('Nombre de paramètres RBM', fontsize=14, fontweight='bold')
ax2.set_ylabel('Temps par itération (s)', fontsize=14, fontweight='bold')
ax2.set_title('Temps d\'exécution vs nombre de paramètres RBM\n(échelle log-log)',
             fontsize=16, fontweight='bold', pad=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, linestyle='--', which='both')
ax2.legend(loc='best', framealpha=0.9, fontsize=9)

plt.tight_layout()

# Sauvegarder
output_file = "execution_time_vs_nb_params.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé : {output_file}")

# Afficher les R²
print("\n=== Qualité des fits (R²) ===")
for fit_name, r2 in r_squared_dict.items():
    print(f"{fit_name}: R² = {r2:.6f}")

plt.show()
