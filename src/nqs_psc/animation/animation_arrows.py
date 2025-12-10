import json
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D



def get_subfolders(run_dir):
    return [p for p in Path(run_dir).iterdir() if p.is_dir()]

def load_runs(list_subfolders):
    loggers = []
    metas = []
    for path in list_subfolders:
        metrics_file = path / "metrics.runtime.json"
        meta_file = path / "meta.json"
        if metrics_file.exists() and meta_file.exists():
            with open(metrics_file, "r") as f:
                loggers.append(json.load(f))
            with open(meta_file, "r") as g:
                metas.append(json.load(g))
    return loggers, metas

# --- Load metadata JSON automatically ---
def load_metadata(run_path):
    metadata_file = os.path.join(run_path, "meta.json")  # chemin exact vers meta.json
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"{metadata_file} introuvable.")

def animate_spins(run_dir, grid_shape=None, fps=10):
    metadata = load_metadata(run_dir)
    p = Path(run_dir)
    subfolders = [p] if (p / "metrics.runtime.json").exists() else get_subfolders(run_dir)
    loggers, metas = load_runs(subfolders)
    if not loggers:
        raise RuntimeError(f"Aucun run trouvé dans {run_dir}")

    logger = loggers[0]

    # Données de magnétisation
    if "magnetization" not in logger:
        raise RuntimeError("Aucune magnétisation trouvée dans le logger.")
    
    mag_data = logger["magnetization"]
    n_iters = len(mag_data[0]["value"])
    n_sites = len(mag_data)
    spins_data = np.array([[spin["value"][i] for spin in mag_data] for i in range(n_iters)])

    # Détermination automatique de la grille si non fournie
    if grid_shape is None:
        L = int(np.sqrt(n_sites))
        if L*L != n_sites:
            raise ValueError(f"Impossible de déterminer une grille carrée pour {n_sites} spins. Fournir grid_shape.")
        grid_shape = (L, L)
    assert grid_shape[0]*grid_shape[1] == n_sites, "grid_shape incompatible avec le nombre de spins"

    # Données d'énergie
    if "Energy" not in logger or "Mean" not in logger["Energy"]:
        raise RuntimeError("Aucune donnée d'énergie disponible.")
    energy = np.array(logger["Energy"]["Mean"])

    # --- Figure ---
    fig, (ax_grid, ax_energy) = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={'height_ratios':[1,1]})
    
    # ---------- Build metadata text ----------
    metadata_text = (
        f"Model: {metadata['model']}\n"
        f"Hamiltonian: {metadata['hamiltonien']}\n"
        f"Learning rate: {metadata['lr']}\n"
        f"N_e: {metadata['N_e']}\n"
        f"Diag shift: {metadata['diag_shift']}\n"
        f"Exact energy: {metadata['exact']}\n"
        f"Iterations: {metadata['n_iters']}"
    )

    # ---------- Metadata box on the left ----------
    meta_box = ax_grid.text(
        -0.3, 0.5,
        metadata_text,
        ha="right",
        va="center",
        fontsize=9,
        transform=ax_grid.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            linewidth=1
        )
    )

    # Grille des spins
    grid_plot = ax_grid.imshow(
        spins_data[0].reshape(grid_shape),
        vmin=-1, vmax=1,
        cmap=cm.bwr,
        origin='lower'
    )
    ax_grid.set_xticks(np.arange(grid_shape[1]))
    ax_grid.set_yticks(np.arange(grid_shape[0]))
    ax_grid.set_xticklabels(np.arange(1, grid_shape[1]+1))
    ax_grid.set_yticklabels(np.arange(1, grid_shape[0]+1))
    ax_grid.set_xlabel("Site x")
    ax_grid.set_ylabel("Site y")
    cbar = fig.colorbar(grid_plot, ax=ax_grid, label="⟨σz⟩")

    # Barre de progression au-dessus de la grille
    progress_bar = ax_grid.barh(grid_shape[0]-0.5+0.15, grid_shape[0], height=0.3, left=-0.5, color='green')
    ax_grid.set_xlim(-0.5, grid_shape[1]-0.5)
    ax_grid.set_ylim(-0.5, grid_shape[0]-0.5+0.3)

    # Courbe d'énergie
    energy_line, = ax_energy.plot([], [], color='blue', label='Estimated energy')
    ax_energy.set_xlim(0, n_iters)
    margin = (np.max(energy)-np.min(energy))*0.05
    ax_energy.set_ylim(np.min(energy)-margin, np.max(energy)+margin)
    ax_energy.axhline(y=metadata['exact'], color='red', linestyle='--', label='Exact energy')
    ax_energy.legend()
    ax_energy.set_xlabel("Iteration")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy vs iteration")

    # créer le texte du titre une seule fois
    title_text = ax_grid.text(
        0.5, 1.05,
        "",
        ha="center", va="bottom",
        transform=ax_grid.transAxes,
        fontsize=12
    )

    # ---------- Flèche de magnétisation ----------
    norm = Normalize(vmin=-1, vmax=1)
    cmap = cm.bwr

    # position x à l'intérieur des axes
    x_arrow = grid_shape[1] - 0.5  # juste à l'intérieur de la limite
    y_center = grid_shape[0]/2

    # flèche initiale
    mag_arrow = ax_grid.arrow(
        x_arrow, y_center,
        0, 0,
        width=0.2,
        color=cmap(norm(0)),
        length_includes_head=True
    )

    # --- update ---
    def update(frame):
        nonlocal mag_arrow  # <<< important !
        
        # spins
        grid_plot.set_data(spins_data[frame].reshape(grid_shape))
        title_text.set_text(f"Spin configuration (iteration {frame+1}/{n_iters})")
        progress_bar[0].set_width((frame+1)/n_iters * grid_shape[1])
        energy_line.set_data(np.arange(frame+1), energy[:frame+1])

        # magnétisation
        m = spins_data[frame].mean()
        mag_arrow.remove()
        mag_arrow = ax_grid.arrow(
            x_arrow, y_center,
            0, m*grid_shape[0]/2,
            width=0.2,
            color=cmap(norm(m)),
            length_includes_head=True
        )

        return grid_plot, energy_line, progress_bar[0], title_text, mag_arrow
    ani = FuncAnimation(fig, update, frames=n_iters, interval=1000/fps, blit=True)
    plt.tight_layout()
    return ani


# --- Exemple ---
run_path = r'/Users/nathandupuy/Desktop/PSC/PSC_automatisation_du_tracé_des_courbes/runs/Variation_ansatz/model=RBM_alpha=1_hamiltonien=Hypercube_-_L_=_4_-_n_=_2_lr=0.1_n_iters=300_diag_shift=0.05_N_e=300'
ani = animate_spins(run_path, grid_shape=None, fps=20)

# ani.save("spin_evolution.mp4", fps=10, dpi=150)
plt.show()


