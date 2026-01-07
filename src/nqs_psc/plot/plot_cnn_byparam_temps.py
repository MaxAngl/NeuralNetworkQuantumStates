import json
import matplotlib
matplotlib.use('Agg') # Mode sans affichage (pour cluster/ssh)
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
import numpy as np

# ==============================================================================
# 1. OUTILS DE PARSING
# ==============================================================================
def flatten_json(y):
    """
    Aplatit un dictionnaire imbriqué.
    Ex: {"optimizer": {"diag_shift": 0.01}} devient {"optimizer_diag_shift": 0.01}
    C'est crucial pour trouver vos paramètres imbriqués.
    """
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x: flatten(x[a], name + a + '_')
        else: out[name[:-1]] = x
    flatten(y)
    return out

def get_run_data(folder_path, param_name_target):
    """
    Lit meta.json pour extraire :
    1. La valeur du paramètre ciblé (ex: diag_shift)
    2. Le temps d'exécution (execution_time_seconds)
    """
    meta_path = folder_path / "meta.json"
    
    if not meta_path.exists():
        return None, None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        # 1. RÉCUPÉRATION DU TEMPS
        # C'est la clé exacte que vous m'avez donnée
        time_val = meta.get("execution_time_seconds")
        
        # 2. RÉCUPÉRATION DU PARAMÈTRE
        # On aplatit le json pour trouver "diag_shift" même s'il est dans "optimizer"
        flat_meta = flatten_json(meta)
        
        param_val = None
        
        # On cherche une clé qui FINIT par le nom du paramètre
        # Ex: si target="diag_shift", on accepte "optimizer_diag_shift" ou "diag_shift"
        for key, val in flat_meta.items():
            if key == param_name_target or key.endswith(f"_{param_name_target}"):
                param_val = val
                break
        
        # Si on ne trouve pas dans le json, on regarde le nom du dossier (Secours)
        if param_val is None:
            # Ex: run_diag_shift=0.001
            import re
            match = re.search(r"([0-9]+\.?[0-9]*e?-?[0-9]*)", folder_path.name)
            if match:
                try: param_val = float(match.group(1))
                except: param_val = folder_path.name
            else:
                param_val = folder_path.name

        return param_val, time_val

    except Exception as e:
        print(f"Erreur lecture {meta_path} : {e}")
        return None, None

# ==============================================================================
# 2. FONCTION PRINCIPALE
# ==============================================================================
def main(folder_path):
    root = Path(folder_path)
    if not root.exists(): return print(f"Erreur : Dossier introuvable {root}")

    # Le nom du dossier parent définit quel paramètre on étudie (ex: "diag_shift")
    param_name = root.name 
    print(f"--- Paramètre variable : {param_name} ---")

    search_dir = root / "Runs" if (root / "Runs").exists() else root
    
    data_points = []

    print("Lecture des fichiers meta.json...")
    
    for sub in sorted(search_dir.iterdir()):
        if not sub.is_dir(): continue
        
        # Extraction des infos
        val, runtime = get_run_data(sub, param_name)
        
        if val is not None and runtime is not None:
            data_points.append({
                "val": val, 
                "time": runtime,
                "label": str(val)
            })
            print(f"  [OK] {sub.name} -> {param_name}={val} | Time={runtime:.2f}s")
        else:
            # Si un run a échoué ou n'a pas fini (pas de meta.json complet)
            pass

    if not data_points: return print("Erreur : Aucune donnée valide trouvée.")

    # --- TRI (Numérique si possible) ---
    try:
        data_points.sort(key=lambda d: float(d["val"]))
    except:
        data_points.sort(key=lambda d: str(d["val"]))

    x_labels = [d["label"] for d in data_points]
    y_values = [d["time"] for d in data_points]

    # ==========================================================================
    # 3. TRACÉ (HISTOGRAMME)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Couleurs basées sur la durée (Magma : sombre=court, clair=long)
    cmap = colormaps.get_cmap("magma")
    norm = plt.Normalize(min(y_values), max(y_values))
    colors = cmap(norm(y_values))

    bars = ax.bar(x_labels, y_values, color=colors, edgecolor='black', alpha=0.9, width=0.6)

    ax.set_xlabel(f"Valeur du paramètre : {param_name}", fontsize=12)
    ax.set_ylabel("Temps d'exécution (s)", fontsize=12)
    ax.set_title(f"Temps d'exécution vs {param_name}", fontweight='bold', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Affichage de la valeur exacte au-dessus de la barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    out_file = root / f"histogramme_temps_{param_name}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"\nSUCCÈS : Graphique sauvegardé -> {out_file}")

if __name__ == "__main__":
    # Chemin vers votre dossier contenant les "Runs"
    path_logs = "/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates/logs/rami/CNN_2D/L=5/channel"
    
    main(path_logs)