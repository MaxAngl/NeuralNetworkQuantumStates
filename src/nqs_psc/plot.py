import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def make_subtitle(meta):
    # Ligne 1 : Infos système
    graph = meta.get("graph", "?")
    L = meta.get("L", "?")
    dim = meta.get("n_dim", "?")
    pbc = "PBC" if meta.get("pbc", False) else "OBC"
    model = meta.get("model", "?")

    line1 = f"Model: {model} | Graph: {graph} (L={L}, dim={dim}, {pbc})"

    # Ligne 2 : Hamiltonien + sampler + optimiser
    H = meta.get("hamiltonian", {})
    H_type = H.get("type", "?")
    H_J = H.get("J", "?")
    H_h = H.get("h", "?")

    sampler = meta.get("sampler", {})
    samp_type = sampler.get("type", "?")
    n_chains = sampler.get("n_chains", "?")
    n_samples = sampler.get("n_samples", "?")

    optim = meta.get("optimizer", {})
    opt_type = optim.get("type", "?")
    lr = optim.get("lr", "?")

    line2 = (
        f"{H_type}: J={H_J}, h={H_h} | "
        f"Sampler: {samp_type} ({n_chains}×{n_samples}) | "
        f"Optim: {opt_type} (lr={lr})"
    )

    return line1 + "\n" + line2


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
