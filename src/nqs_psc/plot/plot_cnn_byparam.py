import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force le mode "sans fenêtre" pour SSH/VSCode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ==============================================================================
# 1. GESTION DES COULEURS (Magma Sombre)
# ==============================================================================
class TruncatedNormalize(mcolors.Normalize):
    """Normalisateur pour éviter le jaune illisible de la colormap Magma."""
    def __init__(self, vmin=None, vmax=None, clip=False, c_start=0.1, c_stop=0.85):
        super().__init__(vmin, vmax, clip)
        self.c_start, self.c_stop = c_start, c_stop
    def __call__(self, value, clip=None):
        return self.c_start + super().__call__(value, clip) * (self.c_stop - self.c_start)

# ==============================================================================
# 2. FONCTIONS DE PARSING (Extraction des données)
# ==============================================================================
def flatten_json(y):
    """Aplatit les dictionnaires imbriqués."""
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x: flatten(x[a], name + a + '_')
        else: out[name[:-1]] = x
    flatten(y)
    return out

def get_energy_curve(log_path):
    """Lit le log.json et renvoie (iterations, energy)."""
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    except:
        return None, None

    if "Energy" not in log_data: return None, None
    data_block = log_data["Energy"]["Mean"]
    
    # Gestion des formats NetKet (Dict of Lists vs List of Dicts)
    if isinstance(data_block, dict) and "real" in data_block:
        energy = np.array(data_block["real"], dtype=float)
    elif isinstance(data_block, list):
        vals = [x['real'] if isinstance(x, dict) else x for x in data_block]
        energy = np.array(vals, dtype=float)
    else: return None, None

    # Nettoyage des NaN/Infini
    energy[np.isinf(energy)] = np.nan
    if np.isnan(energy).all() or len(energy) < 2:
        return None, None 

    # On génère l'axe X manuellement pour éviter le bug 0-1
    iters = np.arange(len(energy))
    return iters, energy

def get_param_val_from_meta(meta_path, target_name):
    """Cherche la valeur du paramètre dans meta.json."""
    try:
        with open(meta_path, 'r') as f: meta = json.load(f)
        flat_meta = flatten_json(meta)
        for key, val in flat_meta.items():
            if key == target_name or key.endswith(f"_{target_name}"):
                return val
    except:
        pass
    return "?"

# ==============================================================================
# 3. FONCTION PRINCIPALE
# ==============================================================================
def main(folder_path):
    root = Path(folder_path)
    if not root.exists():
        print(f"Erreur : Le dossier {root} n'existe pas.")
        return

    param_name = root.name
    print(f"--- Paramètre ciblé : {param_name} ---")

    # Gestion de la structure de dossier (avec ou sans /Runs)
    search_dir = root / "Runs" if (root / "Runs").exists() else root
    curves = []

    # --- A. CHARGEMENT ---
    for sub in sorted(search_dir.iterdir()):
        if not sub.is_dir(): continue
        
        log_f = sub / "log.json"
        meta_f = sub / "meta.json"

        if log_f.exists():
            iters, energy = get_energy_curve(log_f)
            
            if energy is None:
                print(f"  [X] {sub.name} : Données corrompues ou NaN.")
                continue
            
            val = get_param_val_from_meta(meta_f, param_name) if meta_f.exists() else "?"
            mean_e = np.nanmean(energy)
            
            # Affichage console pour vérification
            print(f"  [OK] {sub.name} : {len(energy)} pts | E_moy={mean_e:.4f} | {param_name}={val}")
            
            curves.append({"x": iters, "y": energy, "val": val, "label": f"{param_name}={val}"})

    if not curves:
        print("ERREUR : Aucune courbe valide à tracer.")
        return

    # --- B. TRI ROBUSTE (Correction du bug TypeError) ---
    def robust_sort_key(d):
        v = d["val"]
        try:
            return (0, float(v)) # Priorité aux nombres
        except ValueError:
            return (1, str(v))   # Les textes ensuite

    curves.sort(key=robust_sort_key)

    # --- C. PRÉPARATION DU PLOT ---
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = colormaps.get_cmap("magma")
    
    # Styles de lignes pour différencier les courbes superposées
    linestyles = ['-', '--', '-.', ':']

    # Déterminer si on peut utiliser la colormap numérique
    try:
        vals = [float(c["val"]) for c in curves if isinstance(robust_sort_key(c)[1], float)]
        if not vals: vals = [0, 1]
        v_min, v_max = min(vals), max(vals)
        if v_min == v_max: v_min -= 0.1; v_max += 0.1
        
        norm = TruncatedNormalize(vmin=v_min, vmax=v_max, c_start=0.1, c_stop=0.85)
        use_cmap = True
    except:
        use_cmap = False
        colors = [cmap(i) for i in np.linspace(0.1, 0.85, len(curves))]

    # --- D. TRACÉ PRINCIPAL ---
    for i, c in enumerate(curves):
        # Choix couleur
        if use_cmap:
            try: color = cmap(norm(float(c["val"])))
            except: color = "gray"
        else:
            color = colors[i % len(colors)]
        
        style = linestyles[i % 4] # Cycle les styles
        
        ax.plot(c["x"], c["y"], color=color, linestyle=style, label=c["label"], linewidth=1.5, alpha=0.9)

    # Légende et Colorbar
    if use_cmap:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(param_name)
    else:
        ax.legend()

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy")
    ax.set_title(f"Convergence vs {param_name}", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- E. TRACÉ DU ZOOM (INSET) ---
    # Position: [x, y, width, height] en % de la figure
    axins = ax.inset_axes([0.45, 0.35, 0.45, 0.45]) 

    # On retrace les mêmes courbes dans l'encart
    for i, c in enumerate(curves):
        if use_cmap:
            try: color = cmap(norm(float(c["val"])))
            except: color = "gray"
        else:
            color = colors[i % len(colors)]
        style = linestyles[i % 4]
        
        axins.plot(c["x"], c["y"], color=color, linestyle=style, linewidth=2)

    # Calcul des limites pour zoomer sur les 100 dernières itérations
    max_len = max([len(c["y"]) for c in curves])
    start_zoom = max(0, max_len - 100)
    end_zoom = max_len
    
    axins.set_xlim(start_zoom, end_zoom)
    
    # Ajustement vertical automatique sur la zone de zoom
    y_vals_in_zoom = []
    for c in curves:
        relevant = c["y"][start_zoom:] if len(c["y"]) > start_zoom else []
        y_vals_in_zoom.extend(relevant)
    
    if y_vals_in_zoom:
        z_min, z_max = min(y_vals_in_zoom), max(y_vals_in_zoom)
        margin = (z_max - z_min) * 0.1 if z_max != z_min else 0.01
        axins.set_ylim(z_min - margin, z_max + margin)

    axins.grid(True, linestyle=':', alpha=0.5)
    axins.set_title("Zoom (Fin)", fontsize=9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # --- F. SAUVEGARDE ---
    out_file = root / f"compare_{param_name}.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nSUCCÈS : Graphique sauvegardé ici -> {out_file}")

# ==============================================================================
# EXECUTION (Mets ton chemin ici)
# ==============================================================================
if __name__ == "__main__":
    path_logs = "/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates/logs/rami/CNN_2D/L=5/kernel_size"
    main(path_logs)