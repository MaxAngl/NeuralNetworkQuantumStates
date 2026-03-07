"""
Plot: max(d<Mz^2>/dh) en fonction du desordre sigma.

Charge un etat foundational (mcstate) entraine sur tout (h, sigma),
evalue <Mz^2> sur une grille fine en h pour differentes valeurs de sigma,
calcule la derivee numerique, et trace le max de la derivee vs sigma.

La grille en h est NON-UNIFORME, divisee en 3 zones:
  - [h_min, h_trans_min]  : n_h_before points, zone avant transition
  - [h_trans_min, h_trans_max] : n_h_trans points, zone de transition (grille fine)
  - [h_trans_max, h_max]  : n_h_after points, zone apres transition
Cela permet de concentrer la resolution autour de la transition de phase
(typiquement h/J ~ 1 pour Ising 1D propre) tout en couvrant un large intervalle.

Le nombre de realisations de desordre varie aussi par zone de h:
  - Zone avant transition (h < h_trans_min) : n_disorder realisations
  - Zone de transition (h_trans_min <= h <= h_trans_max) : n_disorder_trans realisations
  - Zone apres transition (h > h_trans_max) : n_disorder_far realisations
On met plus de realisations dans la zone de transition car c est la que la
derivee est maximale et ou la statistique est la plus importante.

Usage (depuis NeuralNetworkQuantumStates/Foundational/):
    NETKET_EXPERIMENTAL_SHARDING=1 python plots_fonctionnels/plot_dMz_vs_sigma.py \
        --run_dir logs/run_XXXX/ \
        --checkpoint state_400.nk

    # Ajuster la grille en h autour de la transition (defaut: 0.5-1.5):
    NETKET_EXPERIMENTAL_SHARDING=1 python plots_fonctionnels/plot_dMz_vs_sigma.py \
        --run_dir logs/run_XXXX/ \
        --checkpoint state_400.nk \
        --h_min 0.1 --h_max 5.0 \
        --h_trans_min 0.5 --h_trans_max 1.5 \
        --n_h_before 4 --n_h_trans 20 --n_h_after 3

    # Plus de realisations dans la zone de transition:
    NETKET_EXPERIMENTAL_SHARDING=1 python plots_fonctionnels/plot_dMz_vs_sigma.py \
        --run_dir logs/run_XXXX/ \
        --checkpoint state_400.nk \
        --n_disorder 30 --n_disorder_trans 100 --n_disorder_far 10

    # Sigma plus large, plus de points:
    NETKET_EXPERIMENTAL_SHARDING=1 python plots_fonctionnels/plot_dMz_vs_sigma.py \
        --run_dir logs/run_XXXX/ \
        --checkpoint state_400.nk \
        --sigma_min 0.0 --sigma_max 3.0 --n_sigma 30

    # Calcul exact (full sum, pour petits L seulement):
    NETKET_EXPERIMENTAL_SHARDING=1 python plots_fonctionnels/plot_dMz_vs_sigma.py \
        --run_dir logs/run_XXXX/ \
        --checkpoint state_400.nk \
        --fullsum
"""

import os
import sys
import argparse
import json
import zipfile

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "1")

import netket as nk
import netket_foundational as nkf
import jax
import jax.numpy as jnp
import numpy as np
import flax
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from netket_foundational._src.model.vit import ViTFNQS

# Import GlobalFlipRule
sys.path.insert(0, foundational_dir)
from flip_rules import GlobalFlipRule


class SafeGlobalFlipRule(GlobalFlipRule):
    def transition(self, sampler, machine, parameters, state, key, sigma):
        sigma_new, log_prob = super().transition(sampler, machine, parameters, state, key, sigma)
        return jnp.asarray(sigma_new, dtype=sigma.dtype), log_prob


def parse_args():
    parser = argparse.ArgumentParser(description="Plot max(dMz2/dh) vs sigma")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Chemin vers le dossier du run (contient meta.json)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Nom du fichier checkpoint (ex: state_400.nk)")
    parser.add_argument("--n_h_before", type=int, default=4,
                        help="Nombre de points en h avant la transition")
    parser.add_argument("--n_h_trans", type=int, default=20,
                        help="Nombre de points en h dans la zone de transition")
    parser.add_argument("--n_h_after", type=int, default=3,
                        help="Nombre de points en h apres la transition")
    parser.add_argument("--n_sigma", type=int, default=20,
                        help="Nombre de points en sigma")
    parser.add_argument("--n_disorder", type=int, default=30,
                        help="Nombre de realisations de desordre par point (h, sigma)")
    parser.add_argument("--h_min", type=float, default=0.1,
                        help="h minimum de la grille")
    parser.add_argument("--h_max", type=float, default=5.0,
                        help="h maximum de la grille")
    parser.add_argument("--sigma_min", type=float, default=0.0,
                        help="sigma minimum")
    parser.add_argument("--sigma_max", type=float, default=1.5,
                        help="sigma maximum")
    parser.add_argument("--fullsum", action="store_true",
                        help="Utiliser FullSumState (exact, pour L petit)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed pour la generation du desordre")
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin du fichier de sortie (defaut: dans run_dir)")
    parser.add_argument("--n_sweeps_burn", type=int, default=100,
                        help="Nombre de sweeps de thermalisation (burn-in)")
    parser.add_argument("--n_mc_avg", type=int, default=10,
                        help="Nombre d evaluations MC independantes pour sigma=0")
    parser.add_argument("--h_trans_min", type=float, default=0.5,
                        help="Debut de la zone de transition (plus de realisations)")
    parser.add_argument("--h_trans_max", type=float, default=1.5,
                        help="Fin de la zone de transition")
    parser.add_argument("--n_disorder_trans", type=int, default=100,
                        help="Realisations dans la zone de transition")
    parser.add_argument("--n_disorder_far", type=int, default=10,
                        help="Realisations apres la transition (grand h)")
    return parser.parse_args()


def load_state(run_dir, checkpoint_name):
    """Charge le meta.json et reconstruit + charge l etat foundational via zipfile+flax."""
    run_path = Path(run_dir)
    meta_path = run_path / "meta.json"
    checkpoint_path = str(run_path / checkpoint_name)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    L = meta["L"]
    h0_train_list = meta["hamiltonian"]["h0_train_list"]

    hi = nk.hilbert.Spin(0.5, L)
    ps = nkf.ParameterSpace(N=hi.size, min=0, max=10 * max(h0_train_list))

    vit_cfg = meta["vit_config"]
    ma = ViTFNQS(
        num_layers=vit_cfg["num_layers"],
        d_model=vit_cfg["d_model"],
        heads=vit_cfg["heads"],
        b=vit_cfg["b"],
        L_eff=vit_cfg["L_eff"],
        n_coups=ps.size,
        complex=True,
        disorder=True,
        transl_invariant=False,
        two_dimensional=False,
    )

    # Init minimal (comme plot_vscore_scatter.py)
    sa = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(0.01), n_chains=1)
    vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

    # Chargement via zipfile + flax (methode qui donne les bons poids)
    state_dict = None
    if not zipfile.is_zipfile(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
    else:
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            candidates = [f for f in zf.namelist() if f.endswith('.msgpack')]
            target_file = sorted(candidates, key=len)[-1]
            with zf.open(target_file) as f:
                state_dict = flax.serialization.msgpack_restore(f.read())

    vars_dict = None
    if 'variables' in state_dict:
        vars_dict = state_dict['variables']
    elif 'model' in state_dict and 'variables' in state_dict['model']:
        vars_dict = state_dict['model']['variables']
    elif 'params' in state_dict:
        vars_dict = state_dict
    elif 'vqs' in state_dict and 'variables' in state_dict['vqs']:
        vars_dict = state_dict['vqs']['variables']
    if vars_dict is None:
        vars_dict = state_dict

    vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
    print(f"Poids injectes depuis : {checkpoint_path}")

    return vs, hi, ps, meta


def create_operator(hi, params, J_val):
    """Construit le Hamiltonien d Ising transverse avec champs locaux."""
    L = hi.size
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(
        nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + 1) % L)
        for i in range(L)
    )
    return -ha_X - J_val * ha_ZZ


def evaluate_mz2(vs, hi, params, use_fullsum=False, n_sweeps_burn=100):
    """Evalue <Mz^2> pour un vecteur de parametres donne."""
    L = hi.size
    Mz = sum(nk.operator.spin.sigmaz(hi, i) for i in range(L)) * (1.0 / float(L))
    Mz2_op = Mz @ Mz

    _vs = vs.get_state(params)

    if use_fullsum:
        vs_eval = nk.vqs.FullSumState(
            hilbert=hi, model=_vs.model, variables=_vs.variables
        )
    else:
        # Thermalisation: plusieurs sweeps de burn-in
        for _ in range(n_sweeps_burn):
            _vs.sample()
        vs_eval = _vs

    result = vs_eval.expect(Mz2_op)
    return np.real(result.Mean)


def compute_mz2_curve(vs, hi, h_grid, sigma, n_disorder, n_disorder_trans, n_disorder_far,
                      h_trans_min, h_trans_max, rng, J_val, use_fullsum, n_sweeps_burn, n_mc_avg):
    """
    Pour un sigma donne, calcule <Mz^2> moyen sur les realisations de desordre
    pour chaque h de la grille, avec plus de realisations autour de la transition.
    Retourne (moyenne, std, raw) pour chaque point h.
    """
    L = hi.size

    # Determiner le nombre de realisations par point h
    # avant transition: n_disorder, transition: n_disorder_trans, apres: n_disorder_far
    def get_n_real(h):
        if sigma < 1e-10:
            return n_mc_avg
        if h < h_trans_min:
            return n_disorder
        if h <= h_trans_max:
            return n_disorder_trans
        return n_disorder_far

    max_n_real = max(get_n_real(h) for h in h_grid)
    mz2_raw = np.full((len(h_grid), max_n_real), np.nan)

    for i, h in enumerate(h_grid):
        n_real = get_n_real(h)
        if sigma < 1e-10:
            params = np.full(L, h)
            for j in range(n_real):
                mz2_raw[i, j] = evaluate_mz2(vs, hi, params, use_fullsum, n_sweeps_burn)
        else:
            for j in range(n_real):
                params = rng.normal(loc=h, scale=sigma, size=L)
                params = np.abs(params)
                mz2_raw[i, j] = evaluate_mz2(vs, hi, params, use_fullsum, n_sweeps_burn)

    mz2_avg = np.nanmean(mz2_raw, axis=1)
    mz2_std = np.nanstd(mz2_raw, axis=1)

    return mz2_avg, mz2_std, mz2_raw


def main():
    args = parse_args()

    # Chargement
    vs, hi, ps, meta = load_state(args.run_dir, args.checkpoint)
    J_val = meta["hamiltonian"]["J"]
    L = meta["L"]

    # Grille non-uniforme en h : dense dans la transition, sparse ailleurs
    h_before = np.linspace(args.h_min, args.h_trans_min, args.n_h_before, endpoint=False)
    h_trans = np.linspace(args.h_trans_min, args.h_trans_max, args.n_h_trans, endpoint=False)
    h_after = np.linspace(args.h_trans_max, args.h_max, args.n_h_after + 1)  # +1 car endpoint=True
    h_grid = np.concatenate([h_before, h_trans, h_after])
    sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)
    rng = np.random.default_rng(args.seed)

    print(f"L={L}, J={J_val}")
    print(f"Grille h : {len(h_grid)} points ({args.n_h_before} avant + {args.n_h_trans} transi + {args.n_h_after+1} apres)")
    print(f"Grille sigma : {args.n_sigma} points dans [{args.sigma_min}, {args.sigma_max}]")
    print(f"Realisations de desordre : {args.n_disorder} (petit h) / {args.n_disorder_trans} (transition [{args.h_trans_min}, {args.h_trans_max}]) / {args.n_disorder_far} (grand h)")
    print(f"Evaluations MC pour sigma=0 : {args.n_mc_avg}")
    print(f"Thermalisation : {args.n_sweeps_burn} sweeps de burn-in")
    print(f"Mode evaluation : {'FullSum (exact)' if args.fullsum else 'Monte Carlo'}")

    # Calcul principal
    max_dMz2_dh = []
    max_dMz2_dh_err = []
    all_curves = {}
    all_stds = {}
    all_raw = {}

    for sigma in tqdm(sigma_grid, desc="Sweep sigma"):
        mz2_curve, mz2_std, mz2_raw = compute_mz2_curve(
            vs, hi, h_grid, sigma, args.n_disorder, args.n_disorder_trans, args.n_disorder_far,
            args.h_trans_min, args.h_trans_max, rng, J_val, args.fullsum,
            args.n_sweeps_burn, args.n_mc_avg
        )
        all_curves[sigma] = mz2_curve
        all_stds[sigma] = mz2_std
        all_raw[sigma] = mz2_raw
        dMz2_dh = np.gradient(mz2_curve, h_grid)
        max_dMz2_dh.append(np.max(np.abs(dMz2_dh)))
        # Erreur sur la derivee: n_real varie par point h
        n_real_arr = np.array([np.sum(~np.isnan(mz2_raw[i])) for i in range(len(h_grid))])
        sem = mz2_std / np.sqrt(n_real_arr)
        dh = h_grid[1] - h_grid[0]
        grad_err = np.sqrt(2) * sem / dh  # erreur sur derivee par differences finies
        idx_max = np.argmax(np.abs(dMz2_dh))
        max_dMz2_dh_err.append(grad_err[idx_max])

    max_dMz2_dh = np.array(max_dMz2_dh)
    max_dMz2_dh_err = np.array(max_dMz2_dh_err)

    # Sauvegarde des donnees
    run_path = Path(args.run_dir)
    data_path = run_path / "dMz2_vs_sigma_data.npz"
    # Convertir en arrays
    curves_array = np.array([all_curves[s] for s in sigma_grid])
    stds_array = np.array([all_stds[s] for s in sigma_grid])
    # raw: liste de arrays (n_h x n_real) par sigma, padding si n_real differe
    raw_list = [all_raw[s] for s in sigma_grid]
    max_n_real = max(r.shape[1] for r in raw_list)
    raw_array = np.full((len(sigma_grid), len(h_grid), max_n_real), np.nan)
    for i, r in enumerate(raw_list):
        raw_array[i, :, :r.shape[1]] = r
    np.savez(
        data_path,
        sigma_grid=sigma_grid,
        max_dMz2_dh=max_dMz2_dh,
        max_dMz2_dh_err=max_dMz2_dh_err,
        h_grid=h_grid,
        mz2_curves=curves_array,
        mz2_stds=stds_array,
        mz2_raw=raw_array,
        n_disorder=args.n_disorder,
        L=L,
    )
    print(f"Donnees sauvegardees : {data_path}")

    # ==========================================
    # PLOT
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.errorbar(sigma_grid, max_dMz2_dh, yerr=max_dMz2_dh_err,
                 fmt="o-", color="tab:blue", markersize=5, linewidth=1.5,
                 capsize=3, capthick=1.2, elinewidth=1.2)
    ax1.set_xlabel(r"$\sigma$ (disorder strength)", fontsize=13)
    ax1.set_ylabel(r"$\max_h \left| \frac{d\langle M_z^2 \rangle}{dh} \right|$", fontsize=13)
    eval_mode = 'FullSum' if args.fullsum else 'MC'
    ax1.set_title(
        f"Max derivative of magnetization vs disorder\n"
        f"L={L}, J={J_val}, {eval_mode}, "
        f"n_dis: {args.n_disorder}/{args.n_disorder_trans}/{args.n_disorder_far} "
        f"(h<{args.h_trans_min}/{args.h_trans_min}-{args.h_trans_max}/h>{args.h_trans_max})",
        fontsize=10,
    )
    ax1.grid(True, alpha=0.3)

    if "sigma" in meta.get("hamiltonian", {}):
        sigma_train = meta["hamiltonian"]["sigma"]
        ax1.axvline(
            sigma_train, color="red", linestyle="--", alpha=0.7,
            label=rf"$\sigma_{{train}}={sigma_train}$"
        )
        ax1.legend(fontsize=10)

    ax2 = axes[1]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid)))

    for idx, (sig, color) in enumerate(zip(sigma_grid, cmap)):
        curve = all_curves[sig]
        ax2.plot(h_grid, curve, "-", color=color, linewidth=1.5,
                 label=rf"$\sigma={sig:.2f}$")

    ax2.set_xlabel(r"$h$ (transverse field)", fontsize=13)
    ax2.set_ylabel(r"$\langle M_z^2 \rangle$", fontsize=13)
    ax2.set_title(f"Magnetization curves for all $\\sigma$\nL={L}", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_path = str(run_path / f"max_dMz2_vs_sigma_L{L}.pdf")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Plot sauvegarde : {out_path}")
    print(f"Plot sauvegarde : {out_path.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
