import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    - Affiche deux graphiques: énergie et erreur relative
    
    meta_key peut être:
    - Une clé simple: "alpha", "L", etc.
    - Une clé imbriquée avec point: "hamiltonian.h", "optimizer.lr"
    """

    run_dir = Path(run_dir)
    runs = []
    metas = []
    values = []
    exact_values = []
    
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
        """Convertit les valeurs complexes en chaînes lisibles (toujours string)"""
        if isinstance(val, (list, tuple)):
            if all(isinstance(v, (list, tuple)) for v in val):
                return "_".join("x".join(map(str, v)) for v in val)
            else:
                return "_".join(map(str, val))
        elif isinstance(val, dict):
            return "_".join(f"{k}={v}" for k, v in val.items())
        else:
            return str(val)  # Toujours convertir en string

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
        
        # Récupérer la valeur exacte
        exact = meta.get("exact")
        if exact is None:
            print(f"[WARNING] Pas de valeur 'exact' dans {sub.name}, skip pour l'erreur relative")
            continue
        
        # Convertir en format affichable si nécessaire
        display_value = format_value_for_display(param_value)

        # -------- Extraction énergie --------
        energy_block = logger["Energy"]

        # Format NetKet : {"iters": [...], "Mean": {"real": [...], "imag": [...]}}
        # ou {"iters": [...], "Mean": [...]}
        if (
            isinstance(energy_block, dict)
            and "iters" in energy_block
            and "Mean" in energy_block
        ):
            it = np.asarray(energy_block["iters"])
            mean_data = energy_block["Mean"]
            
            # Mean peut être un dict {"real": [...], "imag": [...]} ou directement une liste
            if isinstance(mean_data, dict):
                # Format: {"real": [...], "imag": [...]}
                E = np.asarray(mean_data.get("real", mean_data.get("Real", [])))
            elif isinstance(mean_data, list) and len(mean_data) > 0:
                if isinstance(mean_data[0], dict):
                    # Format: [{"real": x, "imag": y}, ...]
                    E = np.array([m.get("real", m.get("Real", 0)) for m in mean_data])
                else:
                    E = np.asarray(mean_data)
            else:
                E = np.asarray(mean_data)
            
            # S'assurer que E est bien un array de floats
            E = E.astype(float)

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
        exact_values.append(exact)

    if len(runs) == 0:
        print("Aucun run valide trouvé.")
        return

    # Sous-titre basé sur le premier run
    subtitle = make_subtitle_clean(metas[0], meta_key)

    # Valeurs numériques ?
    is_numeric = all(isinstance(v, (int, float)) for v in values)

    # Figure avec 2 subplots côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ------------------- Préparation colormap -------------------
    if is_numeric:
        vmin, vmax = min(values), max(values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
        get_color = lambda v: cmap(norm(v))
    else:
        categories = sorted(set(values))
        cmap = plt.get_cmap(cmap_name)
        cols = cmap(np.linspace(0, 1, len(categories)))
        cmap_dict = {cat: cols[i] for i, cat in enumerate(categories)}
        get_color = lambda v: cmap_dict[v]

    # ------------------- Tracé énergie (ax1) -------------------
    for (it, E), v in zip(runs, values):
        ax1.plot(it, E, linewidth=2, color=get_color(v), label=f"{meta_key} = {v}")

    ax1.set_title("Énergie en fonction des itérations")
    ax1.set_xlabel("Itérations")
    ax1.set_ylabel("Énergie")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ------------------- Tracé erreur relative (ax2) -------------------
    for (it, E), v, exact in zip(runs, values, exact_values):
        # Erreur relative : |E - E_exact| / |E_exact|
        rel_error = np.abs(E - exact) / np.abs(exact)
        ax2.plot(it, rel_error, linewidth=2, color=get_color(v), label=f"{meta_key} = {v}")

    ax2.set_title("Erreur relative par rapport à l'énergie exacte")
    ax2.set_xlabel("Itérations")
    ax2.set_ylabel(r"$|E - E_{\mathrm{exact}}| / |E_{\mathrm{exact}}|$")
    ax2.set_yscale("log")  # Échelle log pour mieux voir la convergence
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend()

    # ------------------- Colorbar si numérique -------------------
    if is_numeric:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=[ax1, ax2], location='right', shrink=0.8)
        cbar.set_label(meta_key)

    # ------------------- Titre global -------------------
    fig.suptitle(subtitle, fontsize=10, y=0.98)

    fig.tight_layout()
    plt.savefig(run_dir / "energy_plot_byparam.png", dpi=200)
    plt.close()

    print(f"Graphique sauvegardé dans {run_dir / 'energy_plot_byparam.png'}")


# Exemple d'utilisation
if __name__ == "__main__":
    plot_energy_by_meta(
        r"/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates/logs/alpha_rapport",
        meta_key="alpha",
        cmap_name="viridis"
    )