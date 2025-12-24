import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def make_subtitle(meta, meta_key=None):
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
    N = L**2
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


def plot_energy_from_run(run_dir):
    run_dir = Path(run_dir)

    # ----------- Lecture fichiers JSON -----------
    with open(run_dir / "log.json", "r", encoding="utf-8") as f:
        logger_data = json.load(f)

    with open(run_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # ----------- Extraction énergie -----------
    
    energy_data = logger_data["Energy"]
    
    # 1. Extraction des valeurs (Mean)
    # On gère le format dictionnaire {"real": [...], "imag": [...]}
    raw_mean = energy_data["Mean"]

    if isinstance(raw_mean, dict) and "real" in raw_mean:
        energy_E = np.array(raw_mean["real"])
    else:
        # Ancien format ou liste simple
        temp = np.array(raw_mean)
        if np.iscomplexobj(temp):
            energy_E = temp.real
        else:
            energy_E = temp

    # 2. Extraction des itérations (Axe X)
    if "iters" in energy_data:
        iters_E = np.array(energy_data["iters"])
    else:
        iters_E = np.arange(len(energy_E))

    # 3. Sécurité (Force le format liste même s'il n'y a qu'une seule valeur)
    energy_E = np.atleast_1d(energy_E)
    iters_E = np.atleast_1d(iters_E)

    # Récupération de N (Nombre de sites) depuis les métadonnées
    L = meta.get("L", 1)
    n_dim = meta.get("n_dim", 1) # On récupère la dimension (1D ou 2D)
    
    # Calcul automatique du nombre de sites
    N = L ** n_dim

    # ----------- Plot énergie -----------

    # Figure principale
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(iters_E, energy_E / N, label="Énergie MC par site", linewidth=2)

    # Sous-titre riches en infos
    subtitle = make_subtitle(meta)

    ax.set_title("Énergie en fonction des itérations")
    fig.suptitle(subtitle, fontsize=9, y=0.96)

    ax.set_xlabel("Itérations")
    ax.set_ylabel("Énergie")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ----------- Zoom sur les 100 dernières itérations -----------

    n_zoom = 100
    iters_zoom = iters_E[-n_zoom:]
    energy_zoom = (energy_E / N)[-n_zoom:]
    
    axin = ax.inset_axes([0.65, 0.5, 0.3, 0.3])

    axin.plot(iters_zoom, energy_zoom, linewidth=2)
    axin.set_title("Zoom (100 dernières it.)", fontsize=8)

    # Limites automatiques avec petite marge
    margin_y = 0.05 * (energy_zoom.max() - energy_zoom.min())
    axin.set_xlim(iters_zoom[0], iters_zoom[-1])
    axin.set_ylim(
        energy_zoom.min() - margin_y,
        energy_zoom.max() + margin_y,
    )

    axin.grid(True, alpha=0.3)

    ax.indicate_inset_zoom(axin, edgecolor="black")

    # ----------- Sauvegarde -----------

    plt.tight_layout()
    plt.savefig(run_dir / "energy_plot.png", dpi=200)
    plt.close()

    print(f"Graphique sauvegardé dans {run_dir / 'energy_plot.png'}")

plot_energy_from_run(
    r"/users/eleves-b/2024/nathan.dupuy/NeuralNetworkQuantumStates/logs/Data_courbes_Mz_1D/L=81/Runs/run_2025-12-24_15-09-40"
)
