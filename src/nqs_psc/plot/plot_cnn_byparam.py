import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode sans fenêtre pour SSH
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def get_energy_curve(log_path):
    try:
        with open(log_path, 'r') as f: log_data = json.load(f)
    except: return None, None

    if "Energy" not in log_data: return None, None
    data_block = log_data["Energy"]["Mean"]
    
    # Extraction NetKet
    if isinstance(data_block, dict) and "real" in data_block:
        energy = np.array(data_block["real"], dtype=float)
    elif isinstance(data_block, list):
        vals = [x['real'] if isinstance(x, dict) else x for x in data_block]
        energy = np.array(vals, dtype=float)
    else: return None, None

    # Nettoyage
    energy[np.isinf(energy)] = np.nan
    if np.isnan(energy).all() or len(energy) < 2: return None, None 

    # Création de l'axe X propre
    iters = np.arange(len(energy))
    return iters, energy

def get_param_val_from_meta(meta_path, target_name):
    try:
        with open(meta_path, 'r') as f: meta = json.load(f)
        flat_meta = flatten_json(meta)
        for key, val in flat_meta.items():
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

    param_name = root.name
    print(f"--- Paramètre ciblé : {param_name} ---")

    search_dir = root / "Runs" if (root / "Runs").exists() else root
    curves = []

    # --- CHARGEMENT ---
    for sub in sorted(search_dir.iterdir()):
        if not sub.is_dir(): continue
        log_f, meta_f = sub / "log.json", sub / "meta.json"

        if log_f.exists():
            iters, energy = get_energy_curve(log_f)
            if energy is None: continue
            
            val = get_param_val_from_meta(meta_f, param_name) if meta_f.exists() else "?"
            
            # On stocke les données
            curves.append({
                "x": iters, "y": energy, "val": val, 
                "label": f"{param_name}={val}"
            })
            print(f"  [OK] {sub.name} (val={val})")

    if not curves: return print("Erreur : Aucune courbe valide.")

    # --- TRI ROBUSTE ---
    def robust_sort_key(d):
        try: return (0, float(d["val"]))
        except: return (1, str(d["val"]))
    curves.sort(key=robust_sort_key)

    # ==========================================================================
    # 3. PRÉPARATION DES COULEURS (DISCRÈTES)
    # ==========================================================================
    # On veut des couleurs bien séparées dans la palette Magma
    # On évite le tout début (trop noir) et la toute fin (trop jaune clair)
    # np.linspace(0.1, 0.85, N) nous donne N points équidistants
    
    cmap = colormaps.get_cmap("magma")
    N = len(curves)
    
    # Génération des indices de couleurs
    if N > 1:
        color_indices = np.linspace(0.1, 0.85, N)
    else:
        color_indices = [0.5] # Une seule courbe = couleur milieu

    linestyles = ['-', '--', '-.', ':'] # Cycle de styles

    # On assigne définitivement la couleur et le style à chaque courbe
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
                label=c["label"], # C'est ceci qui s'affichera dans la légende
                linewidth=1.8, 
                alpha=0.9)

    # --- REMPLACEMENT DU GRADIENT PAR LA LÉGENDE ---
    ax.legend(title=param_name, fontsize='medium', loc='best', frameon=True)
    # -----------------------------------------------

    ax.set_xlabel("Itérations")
    ax.set_ylabel("Énergie")
    ax.set_title(f"Convergence vs {param_name}", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # ==========================================================================
    # 5. ZOOM INSET
    # ==========================================================================
    axins = ax.inset_axes([0.45, 0.35, 0.45, 0.45]) 

    # On retrace avec exactement les mêmes attributs (stockés dans le dict)
    for c in curves:
        axins.plot(c["x"], c["y"], 
                   color=c["color"], 
                   linestyle=c["style"], 
                   linewidth=2)

    # Calcul limites Zoom (100 derniers points)
    max_len = max([len(c["y"]) for c in curves])
    start_zoom = max(0, max_len - 100)
    
    # Axe X
    axins.set_xlim(start_zoom, max_len)
    
    # Axe Y (automatique sur la zone visible)
    y_vals_in_zoom = []
    for c in curves:
        if len(c["y"]) > start_zoom:
            y_vals_in_zoom.extend(c["y"][start_zoom:])
            
    if y_vals_in_zoom:
        z_min, z_max = min(y_vals_in_zoom), max(y_vals_in_zoom)
        margin = (z_max - z_min) * 0.1 if z_max != z_min else 0.01
        axins.set_ylim(z_min - margin, z_max + margin)

    axins.grid(True, linestyle=':', alpha=0.5)
    axins.set_title("Zoom (Fin)", fontsize=9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # SAUVEGARDE
    out_file = root / f"compare_{param_name}.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nSUCCÈS : Graphique sauvegardé -> {out_file}")

# ==============================================================================
if __name__ == "__main__":
    # CHANGE LE CHEMIN ICI
    path_logs = "/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates/logs/rami/CNN_2D/L=5/kernel_size"
    main(path_logs)