import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket as nk
import netket_foundational as nkf
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from netket.utils import struct
import netket_pro.distributed as nkpd
from advanced_drivers._src.callbacks.base import AbstractCallback
from netket_foundational._src.model.vit import ViTFNQS

# --- CONFIGURATION ET DOSSIERS ---
output_dir = Path("/results_simulation")
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = output_dir / "checkpoints"

# --- INITIALISATION SYSTÈME ---
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

# --- OPÉRATEURS ---
def create_operator(params):
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
    hi, ps, lambda _: sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))
)

# --- CALLBACK DE SAUVEGARDE ---
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

# --- EXÉCUTION VMC ---
optimizer = optax.sgd(0.005)
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=1e-4)

log_path = output_dir / "log_data"
log = nk.logging.JsonLog(str(log_path))

gs.run(
    1000,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    step_size=10,
    callback=SaveState(str(checkpoint_dir), 10),
)

# --- CALCULS EXACTS ---
Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))
Mz2_mat = (Mz @ Mz).to_sparse()

h_exact_range = np.linspace(0.8, 1.2, 40)
exact_data = {"h": [], "Energy": [], "Mz2": []}

for h_val in tqdm(h_exact_range, desc="Calculs Exacts"):
    _ha = create_operator(h_val.reshape(-1))
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    psi0 = psi0.reshape(-1)
    exact_data["h"].append(h_val)
    exact_data["Energy"].append(E0.item())
    exact_data["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0)).item())

# --- ÉVALUATION VMC FINALE ---
vmc_final = {"h": [], "Energy": [], "Mz2": []}
for pars in tqdm(jnp.linspace(0.8, 1.2, 40).reshape(-1, 1), desc="Évaluation VMC"):
    _ha = create_operator(pars)
    _vs = vs.get_state(pars)
    _vs.sample()
    vmc_final["h"].append(pars[0].item())
    vmc_final["Energy"].append(_vs.expect(_ha).Mean.real)
    vmc_final["Mz2"].append(_vs.expect(Mz @ Mz).Mean.real)

# --- ANALYSE DE LA CONVERGENCE & SAUVEGARDE CSV ---
convergence_list = []
for i, h_val in enumerate(vs.parameter_array.flatten()):
    _ha_exact = create_operator(jnp.array([h_val]))
    e_exact = nk.exact.lanczos_ed(_ha_exact, k=1, compute_eigenvectors=False).item()
    
    iters = log.data["ham"].iters
    energies = log.data["ham"].Mean[:, i]
    
    temp_df = pd.DataFrame({
        "iteration": iters,
        "h": h_val,
        "energy_vmc": energies,
        "energy_exact": e_exact,
        "rel_error": np.abs((energies - e_exact) / e_exact)
    })
    convergence_list.append(temp_df)

full_conv_df = pd.concat(convergence_list)
full_conv_df.to_csv(output_dir / "convergence_history.csv", index=False)

# --- PLOTTING ---

# 1. Energie et Mz2
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(exact_data["h"], exact_data["Energy"], label="Exact")
ax[0].plot(vmc_final["h"], vmc_final["Energy"], "x", label="VMC")
ax[0].set_xlabel("h")
ax[0].set_ylabel("Energy")
ax[0].legend()

ax[1].plot(exact_data["h"], exact_data["Mz2"], label="Exact")
ax[1].plot(vmc_final["h"], vmc_final["Mz2"], "x", label="VMC")
ax[1].set_xlabel("h")
ax[1].set_ylabel("Mz2")
ax[1].legend()
plt.savefig(output_dir / "physics_results.pdf")

# 2. Convergence (Toutes les courbes ensemble)
plt.figure(figsize=(10, 6))
for h_val in full_conv_df["h"].unique():
    data = full_conv_df[full_conv_df["h"] == h_val]
    plt.plot(data["iteration"], data["rel_error"], label=f"h={h_val:.2f}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Relative Error")
plt.title("Convergence des répliques")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "convergence_all_h.pdf")

print(f"\n✅ Simulation terminée. Résultats sauvegardés dans le dossier : {output_dir}")