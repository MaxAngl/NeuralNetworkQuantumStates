import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Lire les données pour L=100 et L=81
data_L100 = pd.read_csv('logs/Data_courbes_Mz_1D/L=100/Résultats.csv')
data_L81 = pd.read_csv('logs/Data_courbes_Mz_1D/L=81/Résultats.csv')

# Ignorer le premier point
data_L100 = data_L100.iloc[1:]
data_L81 = data_L81.iloc[1:]

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
        # Estimation automatique de h_c (80% de la valeur max de H)
        h_c_init = 1.0
        
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

# Fitter les données L=100
print("\n" + "="*50)
print("FIT DES DONNÉES L=100")
print("="*50)
params_L100, errors_L100 = fit_magnetization(data_L100['H'].values, 
                                              data_L100['Magnetization_Sq'].values,
                                              data_L100['Magnetization_Sq_Error'].values)

if params_L100 is not None:
    A_L100, h_c_L100, beta_L100 = params_L100
    A_err_L100, h_c_err_L100, beta_err_L100 = errors_L100
    print(f"A = {A_L100:.4f} ± {A_err_L100:.4f}")
    print(f"h_c = {h_c_L100:.6f} ± {h_c_err_L100:.6f}")
    print(f"beta = {beta_L100:.4f} ± {beta_err_L100:.4f}")
    
    # Calculer R²
    H_fit = data_L100['H'].values
    mask_fit = H_fit < h_c_L100
    Mz_sq_fit = full_model(H_fit[mask_fit], A_L100, h_c_L100, beta_L100)
    residuals = data_L100['Magnetization_Sq'].values[mask_fit] - Mz_sq_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_L100['Magnetization_Sq'].values[mask_fit] - np.mean(data_L100['Magnetization_Sq'].values[mask_fit]))**2)
    r_squared_L100 = 1 - (ss_res / ss_tot)
    print(f"R² = {r_squared_L100:.4f}")

# Fitter les données L=81
print("\n" + "="*50)
print("FIT DES DONNÉES L=81")
print("="*50)
params_L81, errors_L81 = fit_magnetization(data_L81['H'].values, 
                                            data_L81['Magnetization_Sq'].values,
                                            data_L81['Magnetization_Sq_Error'].values)

if params_L81 is not None:
    A_L81, h_c_L81, beta_L81 = params_L81
    A_err_L81, h_c_err_L81, beta_err_L81 = errors_L81
    print(f"A = {A_L81:.4f} ± {A_err_L81:.4f}")
    print(f"h_c = {h_c_L81:.6f} ± {h_c_err_L81:.6f}")
    print(f"beta = {beta_L81:.4f} ± {beta_err_L81:.4f}")
    
    # Calculer R²
    H_fit = data_L81['H'].values
    mask_fit = H_fit < h_c_L81
    Mz_sq_fit = full_model(H_fit[mask_fit], A_L81, h_c_L81, beta_L81)
    residuals = data_L81['Magnetization_Sq'].values[mask_fit] - Mz_sq_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_L81['Magnetization_Sq'].values[mask_fit] - np.mean(data_L81['Magnetization_Sq'].values[mask_fit]))**2)
    r_squared_L81 = 1 - (ss_res / ss_tot)
    print(f"R² = {r_squared_L81:.4f}")

# Créer la figure avec deux subplots côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: L=100
ax1.errorbar(data_L100['H'], data_L100['Magnetization_Sq'], yerr=data_L100['Magnetization_Sq_Error'],
             label='Magnétisation² L=100 (données)', marker='s', linestyle='', 
             capsize=3, alpha=0.7)

if params_L100 is not None:
    H_smooth = np.linspace(data_L100['H'].min(), data_L100['H'].max(), 200)
    Mz_sq_smooth = full_model(H_smooth, A_L100, h_c_L100, beta_L100)
    ax1.plot(H_smooth, Mz_sq_smooth, 'b-', linewidth=2, 
             label=f'Fit: $M_z^2 = A(h_c - h)^\\beta$\n$h_c$ = {h_c_L100:.4f} ± {h_c_err_L100:.1e}\nβ = {beta_L100:.4f} ± {beta_err_L100:.4f}')
    ax1.axvline(h_c_L100, color='blue', linestyle='--', alpha=0.5, linewidth=2)

ax1.set_xlabel('Champ magnétique H')
ax1.set_ylabel('Magnétisation²')
ax1.set_title('Fit de la Magnétisation² (L=100)')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# Subplot 2: L=81
ax2.errorbar(data_L81['H'], data_L81['Magnetization_Sq'], yerr=data_L81['Magnetization_Sq_Error'],
             label='Magnétisation² L=81 (données)', marker='s', linestyle='', 
             capsize=3, alpha=0.7)

if params_L81 is not None:
    H_smooth = np.linspace(data_L81['H'].min(), data_L81['H'].max(), 200)
    Mz_sq_smooth = full_model(H_smooth, A_L81, h_c_L81, beta_L81)
    ax2.plot(H_smooth, Mz_sq_smooth, 'g-', linewidth=2,
             label=f'Fit: $M_z^2 = A(h_c - h)^\\beta$\n$h_c$ = {h_c_L81:.4f} ± {h_c_err_L81:.1e}\nβ = {beta_L81:.4f} ± {beta_err_L81:.4f}')
    ax2.axvline(h_c_L81, color='green', linestyle='--', alpha=0.5, linewidth=2)

ax2.set_xlabel('Champ magnétique H')
ax2.set_ylabel('Magnétisation²')
ax2.set_title('Fit de la Magnétisation² (L=81)')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/comparaison_dataset_magnetization_squared.png', dpi=300, bbox_inches='tight')
print("\nGraphique sauvegardé dans graphs/comparaison_dataset_magnetization_squared.png")

print("\n=== Données L=100 ===")
print(f"Nombre de points: {len(data_L100)}")
print(f"Plage H: [{data_L100['H'].min():.3f}, {data_L100['H'].max():.3f}]")

print("\n=== Données L=81 ===")
print(f"Nombre de points: {len(data_L81)}")
print(f"Plage H: [{data_L81['H'].min():.3f}, {data_L81['H'].max():.3f}]")

plt.show()
