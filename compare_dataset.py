import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Lire le dataset de référence (avec virgule comme séparateur décimal)
# CORRECTION: échanger x et y car les colonnes sont inversées dans le CSV
default_data = pd.read_csv('Default Dataset.csv', sep=';', header=None, names=['y', 'x'], decimal=',')

# Lire les données pour L=10 et L=9
data_L10 = pd.read_csv('logs/Data_courbes_Mz_2D/L=10/Résultats.csv')
data_L9 = pd.read_csv('logs/Data_courbes_Mz_2D/L=9/Résultats.csv')

# Définir le modèle de fit : mz^2 = A*(h_c - h)^beta pour h < h_c, et 0 pour h >= h_c
def fit_model_before_hc(h, A, h_c, beta):
    """Modèle pour h < h_c uniquement"""
    return A * (h_c - h)**beta

def full_model(h, A, h_c, beta):
    """Modèle complet avec transition à h_c"""
    result = np.zeros_like(h)
    mask = h < h_c
    result[mask] = A * (h_c - h[mask])**beta
    return result

# Fonction pour fitter les données
def fit_magnetization(H, Mz_sq, errors):
    try:
        # Trouver une estimation de h_c (autour du point où Mz² devient minimal)
        h_c_init = 3.0 # Légèrement au-dessus du minimum
        
        # Fitter uniquement les points avant h_c estimé
        mask = H < h_c_init
        if np.sum(mask) < 3:
            print("Pas assez de points pour le fit")
            return None, None
            
        H_fit = H[mask]
        Mz_sq_fit = Mz_sq[mask]
        errors_fit = errors[mask]
        
        # Estimation initiale : beta ~ 0.125 pour Ising 2D
        p0 = [1.0, h_c_init, 0.125]  # [A, h_c, beta]
        
        # Fit avec pondération par les erreurs
        popt, pcov = curve_fit(fit_model_before_hc, H_fit, Mz_sq_fit, p0=p0, 
                               sigma=errors_fit, absolute_sigma=True,
                               bounds=([0, H.min(), 0], [np.inf, H.max(), 1]))
        perr = np.sqrt(np.diag(pcov))  # Erreurs sur les paramètres
        
        return popt, perr
    except Exception as e:
        print(f"Erreur lors du fit: {e}")
        return None, None

# Fitter les données L=10
print("\n" + "="*50)
print("FIT DES DONNÉES L=10")
print("="*50)
params_L10, errors_L10 = fit_magnetization(data_L10['H'].values, 
                                            data_L10['Magnetization_Sq'].values,
                                            data_L10['Magnetization_Sq_Error'].values)

if params_L10 is not None:
    A_L10, h_c_L10, beta_L10 = params_L10
    A_err_L10, h_c_err_L10, beta_err_L10 = errors_L10
    print(f"A = {A_L10:.4f} ± {A_err_L10:.4f}")
    print(f"h_c = {h_c_L10:.4f} ± {h_c_err_L10:.4f}")
    print(f"beta = {beta_L10:.4f} ± {beta_err_L10:.4f}")
    
    # Calculer R² pour évaluer la qualité du fit (sur les points h < h_c)
    H_fit = data_L10['H'].values
    mask_fit = H_fit < h_c_L10
    Mz_sq_fit = full_model(H_fit[mask_fit], A_L10, h_c_L10, beta_L10)
    residuals = data_L10['Magnetization_Sq'].values[mask_fit] - Mz_sq_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_L10['Magnetization_Sq'].values[mask_fit] - np.mean(data_L10['Magnetization_Sq'].values[mask_fit]))**2)
    r_squared_L10 = 1 - (ss_res / ss_tot)
    print(f"R² = {r_squared_L10:.4f}")

# Fitter les données L=9
print("\n" + "="*50)
print("FIT DES DONNÉES L=9")
print("="*50)
params_L9, errors_L9 = fit_magnetization(data_L9['H'].values, 
                                          data_L9['Magnetization_Sq'].values,
                                          data_L9['Magnetization_Sq_Error'].values)

if params_L9 is not None:
    A_L9, h_c_L9, beta_L9 = params_L9
    A_err_L9, h_c_err_L9, beta_err_L9 = errors_L9
    print(f"A = {A_L9:.4f} ± {A_err_L9:.4f}")
    print(f"h_c = {h_c_L9:.4f} ± {h_c_err_L9:.4f}")
    print(f"beta = {beta_L9:.4f} ± {beta_err_L9:.4f}")
    
    # Calculer R² (sur les points h < h_c)
    H_fit = data_L9['H'].values
    mask_fit = H_fit < h_c_L9
    Mz_sq_fit = full_model(H_fit[mask_fit], A_L9, h_c_L9, beta_L9)
    residuals = data_L9['Magnetization_Sq'].values[mask_fit] - Mz_sq_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_L9['Magnetization_Sq'].values[mask_fit] - np.mean(data_L9['Magnetization_Sq'].values[mask_fit]))**2)
    r_squared_L9 = 1 - (ss_res / ss_tot)
    print(f"R² = {r_squared_L9:.4f}")

# Créer la figure avec deux subplots côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Comparaison avec L=10
ax1.scatter(default_data['x'], default_data['y'], label='Dataset de référence', 
            color='red', s=100, marker='o', alpha=0.7, zorder=3)
ax1.errorbar(data_L10['H'], data_L10['Magnetization_Sq'], yerr=data_L10['Magnetization_Sq_Error'],
             label='Magnétisation² L=10 (données)', marker='s', linestyle='', 
             capsize=3, alpha=0.7)

# Ajouter la courbe de fit pour L=10
if params_L10 is not None:
    H_smooth = np.linspace(data_L10['H'].min(), data_L10['H'].max(), 200)
    Mz_sq_smooth = full_model(H_smooth, A_L10, h_c_L10, beta_L10)
    ax1.plot(H_smooth, Mz_sq_smooth, 'b-', linewidth=2, 
             label=f'Fit: A={A_L10:.2f}, $h_c$={h_c_L10:.2f}, β={beta_L10:.3f}')
    ax1.axvline(h_c_L10, color='blue', linestyle='--', alpha=0.5, linewidth=2, label=f'$h_c$ = {h_c_L10:.2f}')

ax1.set_xlabel('Champ magnétique H')
ax1.set_ylabel('Magnétisation²')
ax1.set_title(f'Comparaison: Dataset vs Magnétisation² (L=10)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: Comparaison avec L=9
ax2.scatter(default_data['x'], default_data['y'], label='Dataset de référence', 
            color='red', s=100, marker='o', alpha=0.7, zorder=3)
ax2.errorbar(data_L9['H'], data_L9['Magnetization_Sq'], yerr=data_L9['Magnetization_Sq_Error'],
             label='Magnétisation² L=9 (données)', marker='s', linestyle='', 
             capsize=3, alpha=0.7)

# Ajouter la courbe de fit pour L=9
if params_L9 is not None:
    H_smooth = np.linspace(data_L9['H'].min(), data_L9['H'].max(), 200)
    Mz_sq_smooth = full_model(H_smooth, A_L9, h_c_L9, beta_L9)
    ax2.plot(H_smooth, Mz_sq_smooth, 'g-', linewidth=2,
             label=f'Fit: A={A_L9:.2f}, $h_c$={h_c_L9:.2f}, β={beta_L9:.3f}')
    ax2.axvline(h_c_L9, color='green', linestyle='--', alpha=0.5, linewidth=2, label=f'$h_c$ = {h_c_L9:.2f}')

ax2.set_xlabel('Champ magnétique H')
ax2.set_ylabel('Magnétisation²')
ax2.set_title(f'Comparaison: Dataset vs Magnétisation² (L=9)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/comparaison_dataset_magnetization_squared.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé dans graphs/comparaison_dataset_magnetization_squared.png")

# Créer une deuxième figure pour les racines carrées (magnétisation)
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

# Calculer les racines carrées et leurs erreurs
# Pour Mz = sqrt(Mz²), l'erreur est : dMz = dMz²/(2*sqrt(Mz²))
data_L10['Mz_from_sq'] = np.sqrt(data_L10['Magnetization_Sq'])
data_L10['Mz_from_sq_Error'] = data_L10['Magnetization_Sq_Error'] / (2 * data_L10['Mz_from_sq'])

data_L9['Mz_from_sq'] = np.sqrt(data_L9['Magnetization_Sq'])
data_L9['Mz_from_sq_Error'] = data_L9['Magnetization_Sq_Error'] / (2 * data_L9['Mz_from_sq'])

# Dataset de référence en racine
default_data['y_sqrt'] = np.sqrt(default_data['y'])

# Subplot 3: Racine pour L=10
ax3.scatter(default_data['x'], default_data['y_sqrt'], label='Dataset de référence (√M²)', 
            color='red', s=100, marker='o', alpha=0.7, zorder=3)
ax3.errorbar(data_L10['H'], data_L10['Mz_from_sq'], yerr=data_L10['Mz_from_sq_Error'],
             label='√(Magnétisation²) L=10 (données)', marker='s', linestyle='', 
             capsize=3, alpha=0.7)

# Ajouter la courbe de fit en racine pour L=10
if params_L10 is not None:
    H_smooth = np.linspace(data_L10['H'].min(), data_L10['H'].max(), 200)
    Mz_sq_smooth = full_model(H_smooth, A_L10, h_c_L10, beta_L10)
    Mz_smooth = np.sqrt(Mz_sq_smooth)
    ax3.plot(H_smooth, Mz_smooth, 'b-', linewidth=2, 
             label=f'Fit √M²: $h_c$={h_c_L10:.2f}, β={beta_L10:.3f}')
    ax3.axvline(h_c_L10, color='blue', linestyle='--', alpha=0.5, linewidth=2, label=f'$h_c$ = {h_c_L10:.2f}')

ax3.set_xlabel('Champ magnétique H')
ax3.set_ylabel('√(Magnétisation²) = |Magnétisation|')
ax3.set_title(f'Comparaison: Dataset vs √M² (L=10)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: Racine pour L=9
ax4.scatter(default_data['x'], default_data['y_sqrt'], label='Dataset de référence (√M²)', 
            color='red', s=100, marker='o', alpha=0.7, zorder=3)
ax4.errorbar(data_L9['H'], data_L9['Mz_from_sq'], yerr=data_L9['Mz_from_sq_Error'],
             label='√(Magnétisation²) L=9 (données)', marker='s', linestyle='', 
             capsize=3, alpha=0.7)

# Ajouter la courbe de fit en racine pour L=9
if params_L9 is not None:
    H_smooth = np.linspace(data_L9['H'].min(), data_L9['H'].max(), 200)
    Mz_sq_smooth = full_model(H_smooth, A_L9, h_c_L9, beta_L9)
    Mz_smooth = np.sqrt(Mz_sq_smooth)
    ax4.plot(H_smooth, Mz_smooth, 'g-', linewidth=2,
             label=f'Fit √M²: $h_c$={h_c_L9:.2f}, β={beta_L9:.3f}')
    ax4.axvline(h_c_L9, color='green', linestyle='--', alpha=0.5, linewidth=2, label=f'$h_c$ = {h_c_L9:.2f}')

ax4.set_xlabel('Champ magnétique H')
ax4.set_ylabel('√(Magnétisation²) = |Magnétisation|')
ax4.set_title(f'Comparaison: Dataset vs √M² (L=9)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/comparaison_dataset_magnetization_sqrt.png', dpi=300, bbox_inches='tight')
print("Graphique racine sauvegardé dans graphs/comparaison_dataset_magnetization_sqrt.png")

# Afficher les statistiques
print("\n=== Dataset de référence (après correction) ===")
print(f"Nombre de points: {len(default_data)}")
print(f"Plage x (H): [{default_data['x'].min():.3f}, {default_data['x'].max():.3f}]")
print(f"Plage y (M²): [{default_data['y'].min():.3f}, {default_data['y'].max():.3f}]")

print("\n=== Données L=10 ===")
print(f"Nombre de points: {len(data_L10)}")
print(f"Plage H: [{data_L10['H'].min():.3f}, {data_L10['H'].max():.3f}]")
print(f"Plage Magnetization: [{data_L10['Magnetization'].min():.3f}, {data_L10['Magnetization'].max():.3f}]")
print(f"Plage Magnetization_Sq: [{data_L10['Magnetization_Sq'].min():.3f}, {data_L10['Magnetization_Sq'].max():.3f}]")

print("\n=== Données L=9 ===")
print(f"Nombre de points: {len(data_L9)}")
print(f"Plage H: [{data_L9['H'].min():.3f}, {data_L9['H'].max():.3f}]")
print(f"Plage Magnetization: [{data_L9['Magnetization'].min():.3f}, {data_L9['Magnetization'].max():.3f}]")
print(f"Plage Magnetization_Sq: [{data_L9['Magnetization_Sq'].min():.3f}, {data_L9['Magnetization_Sq'].max():.3f}]")

plt.show()
