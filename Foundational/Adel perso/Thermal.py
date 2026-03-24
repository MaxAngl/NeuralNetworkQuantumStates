
import os
import sys
import glob
import json
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import msgpack
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile


def thermalizing(etat_foundation, sampler, operator, max_chain_length_init: int, plot: bool,nb_tentatives:int, seuil=1.05):

    current_max = max_chain_length_init
    stats, hist_data = None, None

    for tentative in range(nb_tentatives):
        print(f"Tentative {tentative+1} avec max_chain_length={current_max}")
        
        stats, hist_data = etat_foundation.check_mc_convergence(
            operator,
            min_chain_length=50,
            max_chain_length=current_max,
            plot=plot
        )

        errors = hist_data["error_of_mean"]
        rhat_values = hist_data["R_hat"]

        premier_indice = None
        for i, rhat in enumerate(rhat_values):
            if abs(rhat) < (seuil):
                premier_indice = i
                break

        if premier_indice is not None:
            break  

        print(f"  Seuil non atteint (rhat_finale = {rhat_values[-1]:.2e}), doublement de max_chain_length...")
        current_max *= 2

    else:
        raise RuntimeError(
            f"Thermalisation non atteinte après {nb_tentatives} tentatives.\n"
            f"  max_chain_length atteint = {current_max}\n"
            f"  error_of_mean finale     = {errors[-1]:.2e} > seuil = {seuil:.2e}\n"
            f"  tau_corr final           = {hist_data['tau_corr_acf'][-1]:.2e}\n"
            f"  R_hat final              = {hist_data['R_hat'][-1]:.4f}\n"
            f"  sweep_size final         = {hist_data['sweep_size'][-1]}\n"
            f"  → Vérifiez votre modèle ou assouplissez le seuil."
        )

    tau_corr   = hist_data["tau_corr_acf"]
    R_hat      = hist_data["R_hat"]
    sweep_size = hist_data["sweep_size"]

    print(f"Convergence atteinte à l'itération {premier_indice}")
    print(f"  error_of_mean au seuil = {errors[premier_indice]:.2e}")
    print(f"  tau_corr final         = {tau_corr[-1]:.2e}")
    print(f"  R_hat final            = {R_hat[-1]:.4f}")
    print(f"  sweep_size final       = {sweep_size[-1]}")

    vstate_thermalized = nk.vqs.MCState(
        sampler=sampler,
        model=etat_foundation.model,
        variables=etat_foundation.variables,
        n_samples=current_max,
        n_discard_per_chain=premier_indice
    )

    convergence_info = {
        "premier_indice" : premier_indice,
        "error_of_mean"  : errors[premier_indice],
        "tau_corr"       : tau_corr[-1],
        "R_hat"          : R_hat[-1],
        "sweep_size"     : sweep_size[-1],
        "max_chain_length_utilise" : current_max,
    }

    return stats, hist_data, convergence_info, vstate_thermalized

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_hist_data(errors, tau=None, R_hat=None, sweep=None):
    n = len(errors)
    return {
        "error_of_mean" : np.array(errors),
        "variance"      : np.ones(n) * 40.0,
        "tau_corr_acf"  : np.array(tau   or [3.5]  * n),
        "R_hat"         : np.array(R_hat or [1.001] * n),
        "sweep_size"    : np.array(sweep or [10]    * n),
    }

def make_etat_foundation(hist_data_list):
    mock = MagicMock()
    mock.model     = MagicMock()
    mock.variables = {"params": {}}
    mock.check_mc_convergence.side_effect = [
        (MagicMock(), hd) for hd in hist_data_list
    ]
    return mock

def run_test(name, fn):
    """Exécute un test et affiche ✅ ou ❌ selon le résultat."""
    try:
        fn()
        print(f"  ✅  {name}")
        return True
    except Exception as e:
        print(f"  ❌  {name}")
        print(f"       {e}")
        return False


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestThermalizing:

    def setup_method(self):
        self.sampler  = MagicMock()
        self.operator = MagicMock()

    def test_convergence_premier_appel(self):
        errors = [0.5, 0.1, 0.001, 0.0005, 0.0001]
        hd = make_hist_data(errors)
        etat = make_etat_foundation([hd])

        with patch("netket.vqs.MCState") as MockMCState:
            MockMCState.return_value = MagicMock()
            stats, hist, conv, vstate_th = thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=5,
            )

        assert conv["premier_indice"] == 3
        assert conv["error_of_mean"] < 1e-3
        assert conv["max_chain_length_utilise"] == 500
        assert etat.check_mc_convergence.call_count == 1

    def test_convergence_premier_appel_avec_plot(self):
        errors = [0.5, 0.1, 0.001, 0.0005, 0.0001]
        hd = make_hist_data(errors)
        etat = make_etat_foundation([hd])

        with patch("netket.vqs.MCState") as MockMCState:
            MockMCState.return_value = MagicMock()
            thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=True, seuil=1e-3, nb_tentatives=5,
            )

        _, kwargs = etat.check_mc_convergence.call_args
        assert kwargs["plot"] is True

    def test_convergence_apres_doublement(self):
        hd_echec  = make_hist_data([0.5, 0.2, 0.05])
        hd_succes = make_hist_data([0.05, 0.005, 0.0005])
        etat = make_etat_foundation([hd_echec, hd_succes])

        with patch("netket.vqs.MCState") as MockMCState:
            MockMCState.return_value = MagicMock()
            stats, hist, conv, vstate_th = thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=5,
            )

        assert etat.check_mc_convergence.call_count == 2
        assert conv["premier_indice"] == 2
        assert conv["max_chain_length_utilise"] == 1000
        _, kwargs_2 = etat.check_mc_convergence.call_args_list[1]
        assert kwargs_2["max_chain_length"] == 1000

    def test_echec_max_tentatives(self):
        hd_echec = make_hist_data([0.5, 0.2, 0.05])
        etat = make_etat_foundation([hd_echec] * 3)

        with pytest.raises(RuntimeError, match="Thermalisation non atteinte"):
            thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=3,
            )

        assert etat.check_mc_convergence.call_count == 3

    def test_echec_message_contient_diagnostics(self):
        hd_echec = make_hist_data(
            errors=[0.5, 0.2, 0.05],
            tau=[3.5, 4.0, 4.5], R_hat=[1.01, 1.02, 1.03], sweep=[10, 10, 10],
        )
        etat = make_etat_foundation([hd_echec])

        with pytest.raises(RuntimeError) as exc_info:
            thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=1,
            )

        msg = str(exc_info.value)
        assert "tau_corr" in msg
        assert "R_hat"    in msg
        assert "sweep"    in msg

    def test_n_discard_correct(self):
        errors = [0.5, 0.0005]
        hd = make_hist_data(errors)
        etat = make_etat_foundation([hd])

        with patch("netket.vqs.MCState") as MockMCState:
            MockMCState.return_value = MagicMock()
            thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=5,
            )
            _, kwargs = MockMCState.call_args
            assert kwargs["n_discard_per_chain"] == 1

    def test_n_samples_est_current_max(self):
        errors = [0.5, 0.0005]
        hd = make_hist_data(errors)
        etat = make_etat_foundation([hd])

        with patch("netket.vqs.MCState") as MockMCState:
            MockMCState.return_value = MagicMock()
            thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=5,
            )
            _, kwargs = MockMCState.call_args
            assert kwargs["n_samples"] == 500

    def test_convergence_info_valeurs_finales(self):
        errors = [0.5, 0.0005]
        hd = make_hist_data(errors, tau=[3.0, 2.0], R_hat=[1.01, 1.001], sweep=[10, 20])
        etat = make_etat_foundation([hd])

        with patch("netket.vqs.MCState") as MockMCState:
            MockMCState.return_value = MagicMock()
            _, _, conv, _ = thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=5,
            )

        assert conv["tau_corr"]   == pytest.approx(2.0)
        assert conv["R_hat"]      == pytest.approx(1.001)
        assert conv["sweep_size"] == 20

    def test_variables_transmises(self):
        errors = [0.5, 0.0005]
        hd = make_hist_data(errors)
        fake_variables = {"params": {"kernel": np.ones((4, 4))}}
        etat = make_etat_foundation([hd])
        etat.variables = fake_variables

        with patch("netket.vqs.MCState") as MockMCState:
            MockMCState.return_value = MagicMock()
            thermalizing(
                etat_foundation=etat, sampler=self.sampler, operator=self.operator,
                max_chain_length_init=500, plot=False, seuil=1e-3, nb_tentatives=5,
            )
            _, kwargs = MockMCState.call_args
            assert kwargs["variables"] is fake_variables


# ─────────────────────────────────────────────
# Lancement manuel avec ✅ / ❌
# ─────────────────────────────────────────────

if __name__ == "__main__":
    t = TestThermalizing()

    tests = [
        ("Convergence dès le 1er appel",                 t.test_convergence_premier_appel),
        ("Convergence 1er appel avec plot=True",          t.test_convergence_premier_appel_avec_plot),
        ("Convergence après doublement max_chain_length", t.test_convergence_apres_doublement),
        ("Échec après nb_tentatives",                     t.test_echec_max_tentatives),
        ("Message d'erreur contient tau, R_hat, sweep",   t.test_echec_message_contient_diagnostics),
        ("n_discard_per_chain correct",                   t.test_n_discard_correct),
        ("n_samples = current_max",                       t.test_n_samples_est_current_max),
        ("convergence_info valeurs finales [-1]",         t.test_convergence_info_valeurs_finales),
        ("variables transmises au MCState",               t.test_variables_transmises),
    ]

    print("\n===== Tests thermalizing =====")
    resultats = []
    for name, fn in tests:
        t.setup_method()
        resultats.append(run_test(name, fn))

    total  = len(resultats)
    reussi = sum(resultats)
    print(f"\n  {reussi}/{total} tests réussis")
    print("==============================\n")



