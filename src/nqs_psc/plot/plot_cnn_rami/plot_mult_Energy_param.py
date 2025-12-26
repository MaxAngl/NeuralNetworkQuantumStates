import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Import nécessaire pour la classe perso
from matplotlib import colormaps
from pathlib import Path

# ==============================================================================
# CLASSE PERSONNALISÉE POUR RESTREINDRE LES COULEURS
# ==============================================================================
class TruncatedNormalize(mcolors.Normalize):
    """
    Normalisateur personnalisé qui mappe les données non pas sur [0, 1],
    mais sur un sous-intervalle [c_start, c_stop] de la colormap.
    """
    def __init__(self, vmin=None, vmax=None, clip=False, c_start=0.0, c_stop=1.0):
        super().__init__(vmin, vmax, clip)
        # Limites de la fraction de colormap à utiliser (entre 0 et 1)
        self.c_start = c_start 
        self.c_stop = c_stop

    def __call__(self, value, clip=None):
        # 1. Normalisation standard vers [0, 1]
        norm_val = super().__call__(value, clip)
        # 2. Redimensionnement vers [c_start, c_stop]
        # Si c_start=0.1 et c_stop=0.8, une donnée au max sera mappée à 0.8 de la colormap
        return self.c_start + norm_val * (self.c_stop - self.c_start)

# ==============================================================================
# FONCTIONS UTILITAIRES (Identiques à la version précédente)
# ==============================================================================
def flatten_dict(d, parent_key='', sep='_'):
    """Aplatit le dictionnaire meta pour comparer les parametres."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def extract_energy_robust(energy_section):
    """Extrait l'energie : gere les formats Dict of Lists et List of Dicts."""
    if "Mean" not in energy_section: return None
    mean_data = energy_section["Mean"]
    # CAS 1 : Format Dict of Lists {"real": [...], "imag": [...]}
    if isinstance(mean_data, dict) and "real" in mean_data:
        return np.array(mean_data["real"], dtype=float)
    # CAS 2 : Format List of Dicts [{"real": x, "imag": y}, ...]
    if isinstance(mean_data, list):
        clean_list = []
        for x in mean_data:
            if isinstance(x, dict): clean_list.append(x.get("real", 0.0))
            elif isinstance(x, (int, float)): clean_list.append(float(x))
        return np.array(clean_list, dtype=float)
    return None

def detect_variable(all_flat_metas):
    """Detecte automatiquement quel parametre varie entre les dossiers."""
    if not all_flat_metas: return "run"
    keys = all_flat_metas[0].keys()
    varying = []
    for k in keys:
        vals = [str(m.get(k)) for m in all_flat_metas]
        if len(set(vals)) > 1 and k not in ["execution_time_seconds", "n_iter"]:
            varying.append(k)
    # Priorites pour la legende
    for p in ["hamiltonian_h", "h", "optimizer_lr", "lr", "kernel_size", "channels"]:
        if p in varying: return p
    return varying[0] if varying else "run"

# ==============================================================================
# FONCTION PRINCIPALE DE PLOT
# ==============================================================================
def main_plot(path):
    root = Path(path)
    data_list = []

    # 1. Chargement et parsing des fichiers
    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        log_f, meta_f = sub / "log.json", sub / "meta.json"
        if not log_f.exists() or not meta_f.exists(): continue
        try:
            with open(meta_f, "r") as f: meta = json.load(f)
            with open(log_f, "r") as f: logger = json.load(f)
            if "Energy" in logger:
                iters = np.array(logger["Energy"]["iters"])
                energy = extract_energy_robust(logger["Energy"])
                if energy is not None and len(energy) > 0:
                    length = min(len(iters), 450, len(energy))
                    data_list.append({
                        "iters": iters[:length],
                        "energy": energy[:length],
                        "flat_meta": flatten_dict(meta),
                        "name": sub.name
                    })
        except Exception as e:
            print(f"Erreur lecture {sub.name}: {e}")
            continue

    if not data_list:
        print("Erreur : Aucun run valide n'a pu etre traite.")
        return

    # 2. Identification de la variable
    var_key = detect_variable([d["flat_meta"] for d in data_list])
    print(f"Graphique genere en fonction de : {var_key}")
    for d in data_list:
        d["val"] = d["flat_meta"].get(var_key, 0)

    # 3. Tri des donnees
    data_list.sort(key=lambda x: x["val"] if isinstance(x["val"], (int, float)) else str(x["val"]))
    
    # 4. Trace avec gestion des COULEURS personnalisée
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # --- CHOIX DE LA COLORMAP ET RESTRICTION ---
    # On utilise 'magma' comme demandé
    cmap = colormaps.get_cmap("magma")
    
    try:
        v_vals = [float(d["val"]) for d in data_list]
        v_min, v_max = min(v_vals), max(v_vals)
        
        # --- UTILISATION DU NORMALISATEUR TRONQUÉ ---
        # c_start=0.1 : commence par du violet foncé, pas du noir complet
        # c_stop=0.85 : s'arrête avant que magma ne devienne jaune vif/blanc
        norm = TruncatedNormalize(vmin=v_min, vmax=v_max, c_start=0.1, c_stop=0.85)
        
        for d in data_list:
            # On récupère la couleur spécifique pour cette valeur
            color_val = cmap(norm(float(d["val"])))
            ax.plot(d["iters"], d["energy"], color=color_val, 
                    label=f"{var_key}={d['val']}", linewidth=1.5)
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax, label=var_key)
        # Astuce : on peut aussi restreindre les ticks de la colorbar visuellement
        # cbar.ax.set_ylim(v_min, v_max)

    except Exception as e:
        print(f"Fallback sur couleurs par défaut (erreur numérique: {e})")
        # Fallback legende standard si texte
        for d in data_list:
            ax.plot(d["iters"], d["energy"], label=f"{var_key}={d['val']}")

    # Labels en texte brut
    ax.set_title(f"Convergence Energie par {var_key}", fontweight='bold')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energie <H>")
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if len(data_list) < 15:
        ax.legend(fontsize='x-small', ncol=1, loc='best')

    plt.tight_layout()
    
    # Sauvegarde
    output_path = root / "convergence_energy.png"
    plt.savefig(output_path, dpi=200)
    print(f"Fichier sauvegarde : {output_path}")
    plt.show()

# --- EXECUTION ---
path_runs = "/users/eleves-a/2024/rami.chagnaud/Documents/NeuralNetworkQuantumStates/logs/rami/CNN_2D/L=5/h"
main_plot(path_runs)