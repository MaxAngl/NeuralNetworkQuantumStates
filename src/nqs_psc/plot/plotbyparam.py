import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from pathlib import Path


# ----------------------------------------------------------
# Sous-titre propre, adapté à votre format de meta
# ----------------------------------------------------------


def make_subtitle_clean(meta, meta_key=None):
    """
    Construit un sous-titre lisible à partir du meta fourni :
    - enlève meta_key s'il existe
    - enlève tous les champs dont la valeur est "?"
    """

    m = dict(meta)  # copie
    if meta_key in m:
        m.pop(meta_key)

    # ---------- LIGNE 1 ----------
    parts1 = []

    # Model
    model = m.get("model")
    if model not in (None, "?"):
        parts1.append(f"Model: {model}")

    # Graph
    graph = m.get("graph")
    if graph not in (None, "?"):
        parts1.append(f"Graph: {graph}")

    # L
    L = m.get("L")
    if L not in (None, "?"):
        parts1.append(f"L={L}")

    # dim
    dim = m.get("n_dim")
    if dim not in (None, "?"):
        parts1.append(f"dim={dim}")

    # pbc
    pbc = m.get("pbc")
    if isinstance(pbc, bool):
        parts1.append("PBC" if pbc else "OBC")

    line1 = " | ".join(parts1)

    # ---------- LIGNE 2 ----------
    parts2 = []

    # Hamiltonien
    H = m.get("hamiltonian", {})
    H_type = H.get("type")
    H_J = H.get("J")
    H_h = H.get("h")

    if H_type not in (None, "?"):
        txt = H_type
        items = []
        if H_J not in (None, "?"):
            items.append(f"J={H_J}")
        if H_h not in (None, "?"):
            items.append(f"h={H_h}")
        if items:
            txt += " (" + ", ".join(items) + ")"
        parts2.append(txt)

    # Sampler
    sampler = m.get("sampler", {})
    samp_type = sampler.get("type")
    n_chains = sampler.get("n_chains")
    n_samples = sampler.get("n_samples")

    if samp_type not in (None, "?"):
        txt = samp_type
        if n_chains not in (None, "?") and n_samples not in (None, "?"):
            txt += f" ({n_chains}×{n_samples})"
        parts2.append("Sampler: " + txt)

    # Optimizer
    opt = m.get("optimizer", {})
    opt_type = opt.get("type")
    lr = opt.get("lr")
    diag_shift = opt.get("diag_shift")

    if opt_type not in (None, "?"):
        txt = opt_type
        items = []
        if lr not in (None, "?"):
            items.append(f"lr={lr}")
        if diag_shift not in (None, "?"):
            items.append(f"diag={diag_shift}")
        if items:
            txt += " (" + ", ".join(items) + ")"
        parts2.append("Optim: " + txt)

    line2 = " | ".join(parts2)

    return line1 + "\n" + line2


# ----------------------------------------------------------
# Fonction principale intelligente de plot
# ----------------------------------------------------------


def plot_energy_by_meta(run_dir, meta_key="magma", cmap_name="viridis"):
    """
    Analyse un dossier contenant plusieurs runs (run_*/log.json, meta.json)

    - Coloration selon meta_key
    - Sous-titre = tout le meta sauf meta_key et les champs '?'
    - Légende = uniquement meta_key = valeur
    
    meta_key peut être:
    - Une clé simple: "alpha", "L", etc.
    - Une clé imbriquée avec point: "hamiltonian.h", "optimizer.lr"
    """

    run_dir = Path(run_dir)
    runs = []
    metas = []
    values = []
    
    def get_nested_value(d, key):
        """Récupère une valeur potentiellement imbriquée (ex: 'optimizer.lr')"""
        keys = key.split('.')
        val = d
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return None
        return val
    
    def format_value_for_display(val):
        """Convertit les valeurs complexes en chaînes lisibles"""
        if isinstance(val, (list, tuple)):
            # Pour kernel_size comme [[1,1],[1,1]] ou channels comme [5,5]
            if all(isinstance(v, (list, tuple)) for v in val):
                # Nested list: [[1,1],[1,1]] -> "1x1_1x1"
                return "_".join("x".join(map(str, v)) for v in val)
            else:
                # Simple list: [5,5] -> "5_5"
                return "_".join(map(str, val))
        elif isinstance(val, dict):
            # Dict -> "key1=val1_key2=val2"
            return "_".join(f"{k}={v}" for k, v in val.items())
        else:
            return val

    # ------------------- Lecture des sous-dossiers -------------------
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue

        logger_file = sub / "log.json"
        meta_file = sub / "meta.json"

        if not logger_file.exists() or not meta_file.exists():
            continue

        with open(logger_file, "r") as f:
            logger = json.load(f)
        with open(meta_file, "r") as f:
            meta = json.load(f)

        # Le paramètre doit exister dans meta
        param_value = get_nested_value(meta, meta_key)
        if param_value is None:
            continue
        
        # Convertir en format affichable si nécessaire
        display_value = format_value_for_display(param_value)

        # -------- Extraction énergie --------
        energy_block = logger["Energy"]

        # Format NetKet : {"iters": [...], "Mean": [...]}
        if (
            isinstance(energy_block, dict)
            and "iters" in energy_block
            and "Mean" in energy_block
        ):
            it = np.asarray(energy_block["iters"])
            E = np.asarray(energy_block["Mean"])

        # Format PSC : [[iter, E], ...]
        elif isinstance(energy_block, list):
            arr = np.asarray(energy_block)
            it = arr[:, 0].astype(int)
            E = arr[:, 1].astype(float)

        else:
            print(f"[ERROR] Format Energy inconnu pour {sub.name}")
            continue

        runs.append((it, E))
        metas.append(meta)
        values.append(display_value)

    if len(runs) == 0:
        print("Aucun run valide trouvé.")
        return

    # Sous-titre basé sur le premier run
    subtitle = make_subtitle_clean(metas[0], meta_key)

    # Valeurs numériques ?
    is_numeric = all(isinstance(v, (int, float)) for v in values)

    # Figure + Axes explicites (corrige l’erreur Colorbar)
    fig, ax = plt.subplots(figsize=(10, 6))

    # ------------------- Tracé -------------------
    if is_numeric:
        vmin, vmax = min(values), max(values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = get_cmap(cmap_name)

        for (it, E), v in zip(runs, values):
            ax.plot(it, E, linewidth=2, color=cmap(norm(v)), label=f"{meta_key} = {v}")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(meta_key)

    else:
        categories = sorted(set(values))
        cmap = get_cmap(cmap_name)
        cols = cmap(np.linspace(0, 1, len(categories)))
        cmap_dict = {cat: cols[i] for i, cat in enumerate(categories)}

        for (it, E), v in zip(runs, values):
            ax.plot(it, E, linewidth=2, color=cmap_dict[v], label=f"{meta_key} = {v}")

    # ------------------- Labels & mise en forme -------------------
    ax.set_title("Énergie en fonction des itérations")
    fig.suptitle(subtitle, fontsize=9, y=0.96)

    ax.set_xlabel("Itérations")
    ax.set_ylabel("Énergie")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()
 # ----------- Sauvegarde -----------

    plt.tight_layout()
    plt.savefig(run_dir / "energy_plot_byparam.png", dpi=200)
    plt.close()

    print(f"Graphique sauvegardé dans {run_dir / 'energy_plot_byparam.png'}")

plot_energy_by_meta(r"/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates/logs/CNN_optimisation/Kernel_size", meta_key="model.kernel_size", cmap_name="viridis")