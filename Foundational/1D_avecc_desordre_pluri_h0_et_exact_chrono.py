"""
Training script for Foundational Neural Quantum States (FNQS) on disordered Ising model
Organised version with clear sections and modular structure
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Uncomment for L > 16 or 20
# os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax
from netket.utils import struct

import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
from advanced_drivers._src.callbacks.base import AbstractCallback
import netket_pro.distributed as nkpd

from src.nqs_psc.utils import save_run


# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    """Central configuration for the experiment"""
    # System parameters
    L = 4
    seed = 1
    J_val = 1.0 / np.e
    
    # Training disorder configurations
    h0_train_list = [0.0, 0.8, 1.0, 1.2, 2.0, 5.0]
    sigma_disorder = 0.1
    n_replicas = 2
    
    # Test disorder configurations
    h0_test_list = [0.4, 0.95, 1.05, 1.3, 1.5, 3]
    N_test_per_h0 = 2
    
    # Sampling parameters
    chains_per_replica = 4
    samples_per_chain = 2
    
    # Training parameters
    n_iter = 200
    lr_init = 0.03
    lr_end = 0.005
    diag_shift = 1e-4
    
    # ViT model parameters
    vit_params = {
        "num_layers": 2,
        "d_model": 16,
        "heads": 4,
        "b": 1,
        "L_eff": L,
    }
    
    # Logging
    logs_path = "logs"
    save_every = 50
    
    @property
    def total_configs_train(self):
        return len(self.h0_train_list) * self.n_replicas
    
    @property
    def n_chains(self):
        return self.total_configs_train * self.chains_per_replica
    
    @property
    def n_samples(self):
        return self.n_chains * self.samples_per_chain
    
    @property
    def N_test_total(self):
        return len(self.h0_test_list) * self.N_test_per_h0


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def generate_disorder_configs(h0_list, n_reps, system_size, sigma, rng=None):
    """Generate disorder configurations for multiple mean field values"""
    if rng is None:
        rng = np.random.default_rng()
    
    all_configs = []
    for h_mean in h0_list:
        configs = rng.normal(loc=h_mean, scale=sigma, size=(n_reps, system_size))
        all_configs.append(configs)
    
    return np.vstack(all_configs)  # Shape: (len(h0_list)*n_reps, system_size)


def create_hamiltonian(params, hilbert, J):
    """Create disordered transverse field Ising Hamiltonian"""
    assert params.shape == (hilbert.size,)
    
    # Transverse field term: sum_i h_i sigma^x_i
    ha_X = sum(params[i] * nkf.operator.sigmax(hilbert, i) 
               for i in range(hilbert.size))
    
    # Ising interaction: -J sum_i sigma^z_i sigma^z_{i+1}
    ha_ZZ = sum(nkf.operator.sigmaz(hilbert, i) @ nkf.operator.sigmaz(hilbert, (i + 1) % hilbert.size)
                for i in range(hilbert.size))
    
    return -ha_X - J * ha_ZZ


def get_metadata(config):
    """Generate metadata dictionary for the run"""
    return {
        "L": config.L,
        "graph": "Hypercube 1D",
        "n_dim": 1,
        "pbc": True,
        "hamiltonian": {
            "type": "Ising Disorder",
            "J": config.J_val,
            "h0_train_list": config.h0_train_list,
            "sigma": config.sigma_disorder
        },
        "model": "ViTFNQS",
        "vit_config": config.vit_params,
        "sampler": {
            "type": "MetropolisLocal",
            "n_chains": config.n_chains,
            "n_samples": config.n_samples
        },
        "optimizer": {
            "type": "SGD",
            "lr_init": config.lr_init,
            "lr_end": config.lr_end,
            "diag_shift": config.diag_shift
        },
        "n_iter": config.n_iter,
        "n_replicas_per_h0": config.n_replicas,
        "total_configs_train": config.total_configs_train,
        "seed": config.seed,
    }


# ==========================================
# CALLBACK CLASSES
# ==========================================

class SaveState(AbstractCallback, mutable=True):
    """Callback to periodically save variational state"""
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

    def on_step_end(self, step, log_data, driver):
        if step % self._save_every == 0:
            path = f"{self._path}/{self._prefix}_{driver.step_count}.nk"
            driver.state.save(path)


# ==========================================
# SYSTEM SETUP
# ==========================================

def setup_system(config):
    """Initialize Hilbert space, model, sampler, and variational state"""
    # Hilbert space
    hilbert = nk.hilbert.Spin(0.5, config.L)
    
    # Parameter space
    param_space = nkf.ParameterSpace(N=hilbert.size, min=0, max=10*max(config.h0_train_list))
    
    # Model (ViT FNQS)
    model = ViTFNQS(
        num_layers=config.vit_params["num_layers"],
        d_model=config.vit_params["d_model"],
        heads=config.vit_params["heads"],
        b=config.vit_params["b"],
        L_eff=config.vit_params["L_eff"],
        n_coups=param_space.size,
        complex=True,
        disorder=True,
        transl_invariant=False,
        two_dimensional=False,
    )
    
    # Sampler
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=config.n_chains)
    
    # Variational state
    vstate = nkf.FoundationalQuantumState(
        sampler, model, param_space,
        n_replicas=config.total_configs_train,
        n_samples=config.n_samples,
        seed=config.seed
    )
    
    return hilbert, param_space, model, sampler, vstate


def setup_operators(hilbert, param_space, J):
    """Create Hamiltonian and observable operators"""
    # Magnetization observable
    Mz = sum(nkf.operator.sigmaz(hilbert, i) for i in range(hilbert.size)) * (1.0 / hilbert.size)
    
    # Parametrized Hamiltonian
    ha_param = nkf.operator.ParametrizedOperator(
        hilbert, param_space,
        lambda params: create_hamiltonian(params, hilbert, J)
    )
    
    # Parametrized magnetization
    mz_param = nkf.operator.ParametrizedOperator(hilbert, param_space, lambda _: Mz)
    
    return ha_param, mz_param, Mz


# ==========================================
# TRAINING
# ==========================================

def train_model(config, vstate, ha_param, mz_param, run_dir):
    """Train the variational model"""
    # Setup optimizer
    learning_rate = optax.linear_schedule(
        init_value=config.lr_init,
        end_value=config.lr_end,
        transition_steps=300
    )
    optimizer = optax.sgd(learning_rate)
    
    # Setup driver
    driver = nkf.VMC_NG(ha_param, optimizer, variational_state=vstate, diag_shift=config.diag_shift)
    
    # Setup logger (pass base name; JsonLog appends '.log')
    log = nk.logging.JsonLog(os.path.join(run_dir, "log_data"), save_params=False)
    
    # Train
    print(f"\n🚀 Starting training for {config.n_iter} iterations...")
    start_time = time.time()

    # Run training (measure time with simple wall-clock)
    driver.run(
        n_iter=config.n_iter,
        out=log,
        obs={"ham": ha_param, "mz2": mz_param},
        step_size=10,
        callback=SaveState(str(Path(run_dir) / f"L{config.L}"), save_every=config.save_every),
        timeit=True,
    )

    duration = time.time() - start_time
    print(f"⏱️  Training completed in {duration:.2f} seconds")
    
    return log, duration


# ==========================================
# EVALUATION
# ==========================================

def compute_exact_energy(params, hilbert, J):
    """Compute exact ground state energy using Lanczos"""
    ha = create_hamiltonian(params, hilbert, J)
    E0 = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=False).item()
    return np.real(E0)


def compute_exact_observables(params, hilbert, J, Mz):
    """Compute exact ground state energy and magnetization"""
    ha = create_hamiltonian(params, hilbert, J)
    E0, psi0 = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
    
    Mz2_op = Mz @ Mz
    Mz2_mat = Mz2_op.to_sparse()
    mz2_val = (psi0.T.conj() @ (Mz2_mat @ psi0.reshape(-1))).item().real
    
    return E0.item(), mz2_val


def evaluate_vmc_state(vstate, params, hilbert, ha_operator, Mz):
    """Evaluate VMC predictions for given parameters"""
    _vs = vstate.get_state(params)
    vs_fs = nk.vqs.FullSumState(hilbert=hilbert, model=_vs.model, variables=_vs.variables)
    
    # Energy expectation
    _e = vs_fs.expect(ha_operator)
    
    # Mz^2 expectation
    _mz2 = vs_fs.expect(Mz @ Mz)
    
    # Variance score
    v_score = _e.variance / (_e.Mean.real**2 + 1e-12)
    
    return _e.Mean.real, _mz2.Mean.real, v_score


def evaluate_test_set(config, vstate, hilbert, J, Mz, params_test):
    """Evaluate model on test disorder configurations"""
    N_test = params_test.shape[0]
    
    vmc_results = {"Energy": [], "Mz2": [], "V_score": []}
    exact_results = {"Energy": [], "Mz2": []}
    
    print(f"\n📊 Evaluating on {N_test} test samples...")
    
    for i in tqdm(range(N_test)):
        pars = params_test[i]
        
        # VMC prediction
        ha_test = create_hamiltonian(pars, hilbert, J)
        e_vmc, mz2_vmc, v_score = evaluate_vmc_state(vstate, pars, hilbert, ha_test, Mz)
        vmc_results["Energy"].append(e_vmc)
        vmc_results["Mz2"].append(mz2_vmc)
        vmc_results["V_score"].append(v_score)
        
        # Exact calculation
        e_exact, mz2_exact = compute_exact_observables(pars, hilbert, J, Mz)
        exact_results["Energy"].append(e_exact)
        exact_results["Mz2"].append(mz2_exact)
    
    return vmc_results, exact_results


def save_test_results(config, vmc_results, exact_results, run_dir):
    """Save test results to CSV"""
    # Generate h_mean labels for each test point
    h_mean_labels = []
    for h_val in config.h0_test_list:
        h_mean_labels.extend([h_val] * config.N_test_per_h0)
    
    df = pd.DataFrame({
        "h_mean": h_mean_labels,
        "exact_energy": exact_results["Energy"],
        "vmc_energy": vmc_results["Energy"],
        "exact_mz2": exact_results["Mz2"],
        "vmc_mz2": vmc_results["Mz2"],
        "v_score": vmc_results["V_score"]
    })
    
    csv_path = os.path.join(run_dir, "test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Test results saved to: {csv_path}")


def save_exact_energies_to_log(params_train, hilbert, J, run_dir):
    """Compute and save exact energies for training set"""
    print("\n💾 Computing exact energies for training set...")
    
    exact_energies = {}
    for i, pars in tqdm(enumerate(params_train)):
        E0 = compute_exact_energy(pars, hilbert, J)
        exact_energies[str(i)] = float(E0)
    
    # Load existing log file if present, otherwise create new log data
    log_file = os.path.join(run_dir, "log_data.log")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        except Exception:
            log_data = {}
    else:
        log_data = {}

    # Add exact energies and save
    log_data["exact_energies"] = exact_energies
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"✅ Exact energies saved to: {log_file}")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    print("="*60)
    print("FNQS Training on Disordered Ising Model")
    print("="*60)
    print(f"System size: L = {config.L}")
    print(f"Training configurations: {config.total_configs_train}")
    print(f"Test configurations: {config.N_test_total}")
    print("="*60)
    
    # Setup system
    print("\n🔧 Setting up quantum system...")
    hilbert, param_space, model, sampler, vstate = setup_system(config)
    ha_param, mz_param, Mz = setup_operators(hilbert, param_space, config.J_val)
    
    # Generate disorder configurations
    print("\n🎲 Generating disorder configurations...")
    params_train = generate_disorder_configs(
        config.h0_train_list, config.n_replicas, hilbert.size, config.sigma_disorder
    )
    params_test = generate_disorder_configs(
        config.h0_test_list, config.N_test_per_h0, hilbert.size, config.sigma_disorder
    )
    
    print(f"Training disorder shape: {params_train.shape}")
    print(f"Test disorder shape: {params_test.shape}")
    
    # Set training parameters
    vstate.parameter_array = params_train
    
    # Setup logging directory
    print("\n📁 Setting up logging directory...")
    meta = get_metadata(config)
    
    try:
        run_dir = save_run(None, meta, create_only=True, base_dir=config.logs_path)
    except Exception as e:
        print(f"Warning: save_run issue ({e}), using default path.")
        run_dir = "checkpoints"
        os.makedirs(run_dir, exist_ok=True)
    
    # Train model
    log, duration = train_model(config, vstate, ha_param, mz_param, run_dir)
    
    # Update metadata with training time
    meta["execution_time_seconds"] = duration
    meta["training_iterations"] = config.n_iter
    meta["timestamp"] = time.time()
    
    # Evaluate on test set
    vmc_results, exact_results = evaluate_test_set(config, vstate, hilbert, config.J_val, Mz, params_test)
    
    # Save test results
    save_test_results(config, vmc_results, exact_results, run_dir)
    
    # Save exact energies for training set
    save_exact_energies_to_log(params_train, hilbert, config.J_val, run_dir)
    
    # Save final metadata
    print("\n💾 Saving final metadata...")
    with open(os.path.join(run_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=4)
    
    print("\n" + "="*60)
    print("✅ Training and evaluation completed successfully!")
    print(f"📂 Results saved in: {run_dir}")
    print("="*60)


if __name__ == "__main__":
    main()