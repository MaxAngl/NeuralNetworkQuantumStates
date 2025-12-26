
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from scipy.optimize import curve_fit

# Configuration
data_dir = Path("logs/Data_courbes_Mz_1D")

# Extraire toutes les valeurs de L disponibles
L_values = []
for folder in sorted(data_dir.iterdir()):
    if folder.is_dir() and folder.name.startswith("L="):
        L = int(folder.name.split("=")[1])
        L_values.append(L)

L_values.sort()
print(f"Valeurs de L trouvées : {L_values}")

# Collecter les temps d'exécution pour chaque L
L_list = []
time_per_iter_list = []
time_per_iter_std_list = []

for L in L_values:
    runs_dir = data_dir / f"L={L}" / "Runs"
    
    if runs_dir.exists():
        times_per_iter = []
        
        # Parcourir tous les runs pour ce L
        for run_folder in runs_dir.iterdir():
            if run_folder.is_dir():
                meta_file = run_folder / "meta.json"
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                        exec_time = meta.get('execution_time_seconds')
                        n_iter = meta.get('n_iter')
                        
                        if exec_time is not None and n_iter is not None and n_iter > 0:
                            time_per_iter = exec_time / n_iter
                            times_per_iter.append(time_per_iter)
        
        if len(times_per_iter) > 0:
            # Pour L=100, garder seulement les valeurs autour de 4s (groupe lent)
            if L == 100:
                times_filtered = [t for t in times_per_iter if t > 3.0]
                print(f"L={L}: Filtré pour garder le groupe à ~4s/iter ({len(times_filtered)} runs sur {len(times_per_iter)})")
            else:
                # Filtrage automatique des outliers par méthode IQR pour les autres L
                times_array = np.array(times_per_iter)
                q1 = np.percentile(times_array, 25)
                q3 = np.percentile(times_array, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Filtrer les valeurs dans l'intervalle [lower_bound, upper_bound]
                times_filtered = [t for t in times_per_iter if lower_bound <= t <= upper_bound]
                
                if len(times_filtered) < len(times_per_iter):
                    print(f"L={L}: Filtré {len(times_per_iter) - len(times_filtered)} outliers (gardé {len(times_filtered)} runs)")
            
            if len(times_filtered) > 0:
                mean_time = np.mean(times_filtered)
                std_time = np.std(times_filtered)
                L_list.append(L)
                time_per_iter_list.append(mean_time)
                time_per_iter_std_list.append(std_time)
                print(f"L={L}: {len(times_filtered)} runs, temps moyen par itération = {mean_time:.4f} ± {std_time:.4f} s")
            else:
                print(f"L={L}: Aucune donnée valide après filtrage")
        else:
            print(f"L={L}: Aucune donnée valide trouvée")
    else:
        print(f"Répertoire Runs non trouvé pour L={L}")

# Convertir en arrays numpy
L_array = np.array(L_list)
time_array = np.array(time_per_iter_list)
time_std_array = np.array(time_per_iter_std_list)

# Définir les fonctions de fit
def linear_fit(L, a, b):
    """Fit linéaire: t = a*L + b"""
    return a * L + b

def quadratic_fit(L, a, b, c):
    """Fit quadratique: t = a*L^2 + b*L + c"""
    return a * L**2 + b * L + c

def power_fit(L, a, b):
    """Fit en loi de puissance: t = a*L^b"""
    return a * L**b

def exponential_fit(L, a, b, c):
    """Fit exponentiel: t = a*exp(b*L) + c"""
    return a * np.exp(b * L) + c

# Effectuer différents fits (sans le linéaire)
fits = {}

try:
    popt_quad, _ = curve_fit(quadratic_fit, L_array, time_array, sigma=time_std_array, absolute_sigma=True)
    fits['Quadratique'] = (quadratic_fit, popt_quad, f"t = {popt_quad[0]:.2e}·L² + {popt_quad[1]:.2e}·L + {popt_quad[2]:.2e}")
    print(f"\nFit quadratique: t = {popt_quad[0]:.2e}·L² + {popt_quad[1]:.2e}·L + {popt_quad[2]:.2e}")
except Exception as e:
    print(f"Erreur fit quadratique: {e}")

try:
    popt_power, _ = curve_fit(power_fit, L_array, time_array, sigma=time_std_array, absolute_sigma=True, p0=[0.001, 2])
    fits['Loi de puissance'] = (power_fit, popt_power, f"t = {popt_power[0]:.2e}·L^{popt_power[1]:.2f}")
    print(f"Fit loi de puissance: t = {popt_power[0]:.2e}·L^{popt_power[1]:.2f}")
except Exception as e:
    print(f"Erreur fit loi de puissance: {e}")

# Créer la figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Calculer les R² pour chaque fit
r_squared_dict = {}
for fit_name, (fit_func, popt, _) in fits.items():
    y_pred = fit_func(L_array, *popt)
    ss_res = np.sum((time_array - y_pred)**2)
    ss_tot = np.sum((time_array - np.mean(time_array))**2)
    r_squared = 1 - (ss_res / ss_tot)
    r_squared_dict[fit_name] = r_squared

# Graphique 1: Échelle linéaire
ax1.errorbar(L_array, time_array, yerr=time_std_array, 
            fmt='o', markersize=8, capsize=5, capthick=2,
            label='Données', color='black', linewidth=2)

# Tracer les fits
L_fit = np.linspace(L_array.min(), L_array.max(), 200)
colors = ['blue', 'green']
for (fit_name, (fit_func, popt, label)), color in zip(fits.items(), colors):
    r2 = r_squared_dict[fit_name]
    ax1.plot(L_fit, fit_func(L_fit, *popt), '--', 
            label=f'{fit_name} (R²={r2:.4f}): {label}', color=color, linewidth=2)

ax1.set_xlabel('Nombre de spins L', fontsize=14, fontweight='bold')
ax1.set_ylabel('Temps moyen par itération (s)', fontsize=14, fontweight='bold')
ax1.set_title('Temps d\'exécution par itération vs L\n(échelle linéaire)', 
             fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', framealpha=0.9, fontsize=10)

# Graphique 2: Échelle log-log avec erreurs mieux proportionnées
# Calculer les erreurs relatives pour mieux voir en échelle log
relative_error = time_std_array / time_array  # Erreur relative
# Limiter l'erreur relative à maximum 50% pour éviter les barres trop grandes
relative_error_capped = np.minimum(relative_error, 0.5)

# Calculer les erreurs asymétriques basées sur l'erreur relative limitée
yerr_lower = time_array * relative_error_capped
yerr_upper = time_array * relative_error_capped

ax2.errorbar(L_array, time_array, yerr=[yerr_lower, yerr_upper], 
            fmt='o', markersize=8, capsize=5, capthick=2,
            label='Données', color='black', linewidth=2, elinewidth=1.5, alpha=0.8)

for (fit_name, (fit_func, popt, label)), color in zip(fits.items(), colors):
    r2 = r_squared_dict[fit_name]
    ax2.plot(L_fit, fit_func(L_fit, *popt), '--', 
            label=f'{fit_name} (R²={r2:.4f})', color=color, linewidth=2)

ax2.set_xlabel('Nombre de spins L', fontsize=14, fontweight='bold')
ax2.set_ylabel('Temps moyen par itération (s)', fontsize=14, fontweight='bold')
ax2.set_title('Temps d\'exécution par itération vs L\n(échelle log-log)', 
             fontsize=16, fontweight='bold', pad=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, linestyle='--', which='both')
ax2.legend(loc='best', framealpha=0.9, fontsize=10)

plt.tight_layout()

# Sauvegarder le graphique
output_file = "execution_time_vs_L_with_fit.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé : {output_file}")

# Calculer et afficher les R² pour chaque fit
print("\n=== Qualité des fits (R²) ===")
for fit_name, (fit_func, popt, _) in fits.items():
    y_pred = fit_func(L_array, *popt)
    ss_res = np.sum((time_array - y_pred)**2)
    ss_tot = np.sum((time_array - np.mean(time_array))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"{fit_name}: R² = {r_squared:.6f}")

plt.show()
