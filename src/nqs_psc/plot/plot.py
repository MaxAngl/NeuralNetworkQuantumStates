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
    # Format : Energy = [[iter, value], [iter, value], ...]
    energy_list = logger_data["Energy"]
    energy_list = np.asarray(energy_list)

    iters_E = energy_list[:, 0].astype(int)
    energy_E = energy_list[:, 1].astype(float)

    # ----------- Plot énergie -----------
    plt.figure(figsize=(9, 5))
    plt.plot(iters_E, energy_E, label="Énergie MC", linewidth=2)

    # Sous-titre riches en infos
    subtitle = make_subtitle(meta)

    plt.title("Énergie en fonction des itérations")
    plt.suptitle(subtitle, fontsize=9, y=0.96)

    plt.xlabel("Itérations")
    plt.ylabel("Énergie")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_energy_from_run(
    r"NeuralNetworkQuantumStates/logs/run_2026-02-03_16-00-43"
)


