"""
Plot colormap du V-score (h0 vs sigma) pour un run Foundational NQS.

Charge un vstate depuis un checkpoint .nk, evalue le V-score sur une grille
(h0, sigma), et trace une colormap pcolormesh.

Usage:
    cd NeuralNetworkQuantumStates/Foundational
    NETKET_EXPERIMENTAL_SHARDING=1 python plot_vscore_cmap.py \
        --run_dir ../logs/run_2026-03-05_11-31-03 \
        --checkpoint state_400.nk

    # Options:
    NETKET_EXPERIMENTAL_SHARDING=1 python plot_vscore_cmap.py \
        --run_dir ../logs/run_2026-03-05_11-31-03 \
        --checkpoint state_400.nk \
        --h_min 0.1 --h_max 5.0 --n_h 15 \
        --sigma_min 0.01 --sigma_max 0.3 --n_sigma 12 \
        --n_disorder 5 --seed 42
"""

import os
import sys
import argparse
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, foundational_dir)

os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import netket as nk
import netket_foundational as nkf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from pathlib import Path
from spin_vmc.data.funcs import vscore
from flip_rules import GlobalFlipRule


def parse_args():
    parser = argparse.ArgumentParser(description="Plot V-score colormap (h0 vs sigma)")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--h_min", type=float, default=0.1)
    parser.add_argument("--h_max", type=float, default=5.0)
    parser.add_argument("--n_h", type=int, default=15)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=0.3)
    parser.add_argument("--n_sigma", type=int, default=12)
    parser.add_argument("--n_disorder", type=int, default=5,
                        help="Nombre de realisations de desordre par point (h0, sigma)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_path = Path(args.run_dir)
    checkpoint_path = str(run_path / args.checkpoint)

    # Chargement du vstate
    vs = nkf.FoundationalQuantumState.load(checkpoint_path)

    with open(run_path / "meta.json") as f:
        meta = json.load(f)

    L = meta["L"]
    J_val = meta["hamiltonian"]["J"]
    hi = nk.hilbert.Spin(0.5, L)
    h0_train_list = meta["hamiltonian"]["h0_train_list"]
    sigma_train = meta["hamiltonian"].get("sigma", None)

    def create_operator(params):
        ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
        ha_ZZ = sum(
            nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
            for i in range(hi.size)
        )
        return -ha_X - J_val * ha_ZZ

    # Grilles
    h0_test = np.linspace(args.h_min, args.h_max, args.n_h)
    sigma_test = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)
    rng = np.random.default_rng(args.seed)

    print(f"L={L}, J={J_val}, checkpoint={args.checkpoint}")
    print(f"Grille: {args.n_h} h0 x {args.n_sigma} sigma x {args.n_disorder} realisations")

    # Evaluation
    grid_vscores = np.zeros((args.n_h, args.n_sigma, args.n_disorder))
    total = args.n_h * args.n_sigma * args.n_disorder

    pbar = tqdm(total=total, desc="Evaluation")
    for i, h0 in enumerate(h0_test):
        for j, sig in enumerate(sigma_test):
            for k in range(args.n_disorder):
                pars = np.abs(rng.normal(loc=h0, scale=sig, size=L))
                _ha = create_operator(pars)
                _vs = vs.get_state(pars)
                _vs.sample()
                e = _vs.expect(_ha)
                grid_vscores[i, j, k] = vscore(e.mean.real, e.variance.real, L)
                pbar.update(1)
    pbar.close()

    # Median sur les realisations
    median_vs = np.median(grid_vscores, axis=2)

    # Sauvegarde donnees
    np.savez(
        run_path / "vscore_cmap_data.npz",
        h0_test=h0_test, sigma_test=sigma_test,
        grid_vscores=grid_vscores, median_vscores=median_vs,
        L=L, J=J_val,
    )
    print(f"Donnees sauvegardees: {run_path / 'vscore_cmap_data.npz'}")

    # Plot
    sigma_grid, h0_grid = np.meshgrid(sigma_test, h0_test)

    fig, ax = plt.subplots(figsize=(8, 6))

    vmin = median_vs[median_vs > 0].min()
    vmax = median_vs.max()

    im = ax.pcolormesh(
        sigma_grid, h0_grid, median_vs,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis", shading="auto",
    )
    for h in h0_train_list:
        ax.axhline(h, color="red", alpha=0.25, linewidth=0.5, linestyle="--")
    if sigma_train is not None:
        ax.axvline(sigma_train, color="white", alpha=0.5, linewidth=1, linestyle="--")

    ax.set_xlabel(r"$\sigma_{test}$", fontsize=13)
    ax.set_ylabel(r"$h_0^{test}$", fontsize=13)
    ax.set_title(
        f"Median V-score — L={L}, J={J_val}\n"
        f"checkpoint: {args.checkpoint}, {args.n_disorder} realisations/point\n"
        f"Red dashes = train h0"
        + (f", white dash = $\\sigma_{{train}}$={sigma_train}" if sigma_train else ""),
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="Median V-score")

    plt.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_path = str(run_path / f"vscore_cmap_L{L}.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Plot sauvegarde: {out_path}")


if __name__ == "__main__":
    main()
