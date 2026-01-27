# This example runs in ~2 minutes on 4 A100s + 5 minutes for the plotting exact values

import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket as nk
import netket_foundational as nkf

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from netket_foundational._src.model.vit import ViTFNQS

k = jax.random.key(1)

hi = nk.hilbert.Spin(0.5, 10)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)

ma = ViTFNQS(
    num_layers=2,
    d_model=12,
    heads=4,
    L_eff=hi.size // 2,
    n_coups=ps.size,
    b=2,
    complex=False,
    disorder=False,
    transl_invariant=True,
    two_dimensional=False,
)


sa = nk.sampler.MetropolisLocal(hi, n_chains=5016)

vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=8, seed=1)
vs.parameter_array = jnp.linspace(0.8, 1.2, vs.n_replicas).reshape(-1, 1)
#le reshape est là pour en faire un vecteur

Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))

xs = vs.hilbert.random_state(k, 5)
# Rappel : génère 5 états aléatoirement

def create_operator(params):
    # print(params.shape, params)
    assert params.shape == (1,)
    h = params[0]

    ha_X = sum(nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(
        nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
        for i in range(hi.size)
    )
    return -h * ha_X - ha_ZZ


ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(
    hi,
    ps,
    lambda _: sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size))
    * (1 / float(hi.size)),
)

ha_p.get_conn_padded(xs)


from netket.utils import struct

import netket_pro.distributed as nkpd
from advanced_drivers._src.callbacks.base import AbstractCallback

import os


class SaveState(AbstractCallback, mutable=True):
    _path: str = struct.field(pytree_node=False, serialize=False)
    _prefix: str = struct.field(pytree_node=False, serialize=False)
    _save_every: int = struct.field(pytree_node=False, serialize=False)

    def __init__(self, path: str, save_every: int, prefix: str = "state"):
        self._path = path
        self._prefix = prefix
        self._save_every = save_every

    def on_run_start(self, step, driver, callbacks):
        if nkpd.is_master_process() and not os.path.exists(self._path):
            os.makedirs(self._path)

        path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
        driver.state.save(path)

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)


#
import optax

optimizer = optax.sgd(0.005)
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=1e-4)

log = nk.logging.JsonLog("2")
gs.run(
    1000,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    step_size=10,
    callback=SaveState(
        "2",
        10,
    ),
)

Mz2 = Mz @ Mz
Mz2_mat = Mz2.to_sparse()

exact = {
    "h": np.linspace(0.8, 1.2, 40),
    "Energy": [],
    "Mz2": [],
}
for pars in tqdm(exact["h"]):
    _ha = create_operator(pars.reshape(-1))
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    E0 = E0.item()
    psi0 = psi0.reshape(-1)
    exact["Energy"].append(E0)
    exact["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0)).item())

exact = {
    "h": np.array(exact["h"]),
    "Energy": np.array(exact["Energy"]),
    "Mz2": np.array(exact["Mz2"]),
}

vmc_vals = {
    "Energy": [],
    "Mz2": [],
}
for pars in tqdm(vs.parameter_array):
    _ha = create_operator(pars)
    # ed = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()
    _vs = vs.get_state(pars)
    _vs.reset()
    _vs.sample()
    _vs.sample()
    _e = _vs.expect(_ha)
    _o = _vs.expect(Mz2)
    vmc_vals["Energy"].append(_e.Mean)
    vmc_vals["Mz2"].append(_o.Mean)

vmc_vals = {
    "h": np.array(vs.parameter_array),
    "Energy": np.array(vmc_vals["Energy"]),
    "Mz2": np.array(vmc_vals["Mz2"]),
}

import matplotlib.pyplot as plt

plt.plot(exact["h"], exact["Energy"], label="Exact")
plt.plot(vmc_vals["h"], vmc_vals["Energy"], "x", label="VMC")
plt.xlabel("h")
plt.ylabel("Energy")
plt.legend()
plt.savefig("energy.pdf")
plt.clf()

plt.plot(exact["h"], exact["Mz2"], label="Exact")
plt.plot(vmc_vals["h"], vmc_vals["Mz2"], "x", label="VMC")
plt.xlabel("h")
plt.ylabel("Mz2")
plt.legend()
plt.savefig("mz2.pdf")
plt.clf()


# Convergence
conv_data = []
for i, pars in tqdm(enumerate(vs.parameter_array)):
    _ha = create_operator(pars)
    ed = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()

    err_val = log.data["ham"][i].Mean - ed
    conv_data.append(
        {
            "h": float(pars.item()),
            "e0": log.data["ham"][i].Mean,
            "energy": ed,
            "iters": log.data["ham"][i].iters,
            "err_val": log.data["ham"][i].Mean - ed,
        }
    )

for _data in conv_data:
    plt.plot(
        _data["iters"],
        np.abs(_data["err_val"] / _data["e0"]),
        label=f"h = {_data['h']:.2f}",
    )
plt.xlabel("Iteration")
plt.ylabel("Rel Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("convergence.pdf")
plt.clf()