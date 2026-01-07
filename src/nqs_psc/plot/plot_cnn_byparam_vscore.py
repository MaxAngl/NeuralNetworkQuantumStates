import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode sans fenêtre pour serveur/SSH
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ==============================================================================
# 1. PARSING ROBUSTE
# ==============================================================================
def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x: flatten(x[a], name + a + '_')
        else: out[name[:-1]] = x
    flatten(y)
    return out

def get_variance_curve(log_path):
    """
    Extrait la courbe de Variance (proxy du V-score) depuis le log.json
    """
    try:
        with open(log_path, 'r') as f: log_data = json.load(f)
    except: return None, None

    if "Energy" not in log_data: return None, None
    
    # On cherche la Variance (parfois nommée 'Variance', 'var', ou 'Sigma')
    # C'est la donnée qui permet de calculer le V-score
    data_block = None
    keys_to_check = ["Variance", "var", "Sigma"]
    
    for k in keys_to_check:
        if k in log_data["Energy"]:
            data_block = log_data["Energy"][k]
            break
            
    if data_block is None: return None, None
    
    # Extraction NetKet (gestion formats dict/list)
    if isinstance(data_block, dict) and "real" in data_block:
        # Format NetKet récent
        var_vals = np.array(data_block["real"], dtype=float)
    elif isinstance(data_block, list):
        # Format liste de dicts ou liste de valeurs
        vals = [x['real'] if isinstance(x, dict) else x for x in data_block]
        var_vals = np.array(vals, dtype=float)
    else: 
        return None, None

    # Nettoyage des NaNs/Infs
    var_vals[np.isinf(var_vals)] = np.nan
    
    # Si tout est NaN ou vide
    if np.isnan(var_vals).all() or len(var_vals) < 2: return None, None 

    # Création de l'axe X (itérations)
    iters = np.arange(len(var_vals))
    
    return iters, var_vals

def get_param_val_from_meta(meta_path, target_name):
    try:
        with open(meta_path, 'r') as f: meta = json.load(f)
        flat_meta = flatten_json(meta)
        for key, val in flat_meta.items():
            # Cherche la clé exacte ou une fin de clé (ex: "model_kernel_size")
            if key == target_name or key.endswith(f"_{target_name}"):
                return val
    except: pass
    return "?"

# ==============================================================================
# 2. FONCTION PRINCIPALE
# ==============================================================================
def main(folder_path):
    root = Path(folder_path)
    if not root.exists(): return print(f"Erreur : Dossier introuvable {root}")

    param_name = root.name # Le nom du paramètre est le nom du dossier parent
    print(f"--- Paramètre ciblé : {param_name} ---")

    # Gestion structure dossier (Parfois c'est direct, parfois dans "Runs")
    search_dir = root / "Runs" if (root / "Runs").exists() else root
    curves = []

    # --- CHARGEMENT ---
    print("Lecture des fichiers...")
    for sub in sorted(search_dir.iterdir()):
        if not sub.is_dir(): continue
        
        log_f = sub / "log.json"
        meta_f = sub / "meta.json"

        if log_f.exists():
            iters, variance = get_variance_curve(log_f)
            
            if variance is not None:
                # Récupération de la valeur du paramètre
                val = get_param_val_from_meta(meta_f, param_name) if meta_f.exists() else "?"
                if val == "?":
                    # Fallback : essayer de trouver la valeur dans le nom du dossier (ex: L=5)
                    try:
                        val = float(sub.name.split('=')[-1])
                    except:
                        val = sub.name # Garde le nom du dossier si pas de nombre

                # Stockage
                curves.append({
                    "x": iters, 
                    "y": variance, 
                    "val": val, 
                    "label": f"{param_name}={val}"
                })
                print(f"  [OK] {sub.name} -> {param_name}={val}")

    if not curves: return print("Erreur : Aucune courbe de variance valide trouvée.")

    # --- TRI ROBUSTE ---
    # Pour que la légende soit ordonnée numériquement si possible
    def robust_sort_key(d):
        try: return (0, float(d["val"]))
        except: return (1, str(d["val"]))
    curves.sort(key=robust_sort_key)

    # ==========================================================================
    # 3. PRÉPARATION DES COULEURS (MAGMA)
    # ==========================================================================
    cmap = colormaps.get_cmap("magma")
    N = len(curves)
    
    # On évite le noir pur (0.0) et le blanc pur (1.0)
    if N > 1:
        color_indices = np.linspace(0.15, 0.85, N)
    else:
        color_indices = [0.5]

    linestyles = ['-', '--', '-.', ':'] 

    for i, c in enumerate(curves):
        c["color"] = cmap(color_indices[i])
        c["style"] = linestyles[i % 4]

    # ==========================================================================
    # 4. TRACÉ PRINCIPAL
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    for c in curves:
        ax.plot(c["x"], c["y"], 
                color=c["color"], 
                linestyle=c["style"], 
                label=c["label"], 
                linewidth=1.5, 
                alpha=0.9)

    # LÉGENDE & ESTHÉTIQUE
    ax.legend(title=param_name, fontsize='medium', loc='best', frameon=True)
    
    ax.set_xlabel("Nombre d'itérations", fontsize=12)
    ax.set_ylabel("Variance de l'Énergie (Log scale)", fontsize=12)
    ax.set_title(f"Convergence du V-score (Variance) vs {param_name}", fontweight='bold', fontsize=14)
    
    # IMPORTANT : Échelle Logarithmique pour le V-score/Variance
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.4)

    # ==========================================================================
    # 5. ZOOM INSET (Convergence finale)
    # ==========================================================================
    # Position de l'encart : [x, y, width, height] (coordonnées relatives 0-1)
    axins = ax.inset_axes([0.5, 0.4, 0.45, 0.45]) 

    for c in curves:
        axins.plot(c["x"], c["y"], 
                   color=c["color"], 
                   linestyle=c["style"], 
                   linewidth=2)

    # Paramètres du zoom : les 150 dernières itérations
    max_len = max([len(c["y"]) for c in curves])
    start_zoom = max(0, max_len - 150)
    
    axins.set_xlim(start_zoom, max_len)
    
    # Ajustement Y du zoom (Log scale aussi)
    axins.set_yscale('log')
    axins.grid(True, which="both", linestyle=':', alpha=0.5)
    axins.set_title("Zoom (Fin de convergence)", fontsize=9)
    
    # Masquer les labels des ticks de l'encart pour alléger
    axins.tick_params(axis='both', which='both', labelsize=8)

    # Lignes reliant le zoom
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.5)

    # SAUVEGARDE
    out_file = root / f"convergence_variance_{param_name}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nSUCCÈS : Graphique sauvegardé -> {out_file}")

# ==============================================================================
if __name__ == "__main__":
    path_logs = "/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates/logs/rami/CNN_2D/L=5/diag_shift"
    
    main(path_logs)