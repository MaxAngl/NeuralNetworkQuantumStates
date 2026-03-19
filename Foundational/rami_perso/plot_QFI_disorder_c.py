#!/usr/bin/env python3
"""
Quantum Fisher Information plotter for disorder systems with FNQS

This script computes and visualizes the Quantum Fisher Information (QFI) matrix
as a function of disorder strength, enabling the detection of quantum phase transitions.

Usage:
    python plot_QFI_disorder.py <path_to_run_directory>

The script will:
1. Load the last trained model
2. Generate test disorder configurations not seen during training
3. Compute QFI matrix: g_ij = <∂_i log ψ | ∂_j log ψ>
4. Plot QFI measures (trace, max eigenvalue) vs disorder strength
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import warnings

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# JAX CONFIG
# ==========================================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def find_latest_state(run_dir):
    """Find the latest saved state in run directory."""
    state_files = sorted(
        Path(run_dir).glob("state_*.nk"),
        key=lambda x: int(x.stem.split("_")[1])
    )
    if not state_files:
        raise FileNotFoundError(f"No state files found in {run_dir}")
    return str(state_files[-1])


def load_model_and_state(run_dir, device="cpu"):
    """Load trained model and latest state from run directory."""
    
    # Load metadata
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Extract parameters
    L = meta.get("L", 16)
    if "nb_spins" in meta:
        n_spins = meta["nb_spins"]
    else:
        n_dim = meta.get("n_dim", 1)
        n_spins = L ** n_dim
    
    disorder_config = meta["hamiltonian"]
    h0_train_list = disorder_config["h0_train_list"]
    sigma_disorder = disorder_config["sigma"]
    J_val = disorder_config["J"]
    n_replicas = meta.get("n_replicas_per_h0", 10)
    vit_params = meta["vit_config"]
    seed = meta.get("seed", 1)
    
    print(f"📋 Configuration loaded:")
    print(f"   - System: {n_spins} spins ({n_dim}D, L={L})")
    print(f"   - ViT config: {vit_params}")
    print(f"   - h0_train: {h0_train_list}")
    print(f"   - Disorder σ: {sigma_disorder}")
    
    # Build Hilbert space and model
    hi = nk.hilbert.Spin(0.5, n_spins)
    ps = nkf.ParameterSpace(N=n_spins, min=0, max=10*max(h0_train_list))
    
    ma = ViTFNQS(
        num_layers=vit_params["num_layers"],
        d_model=vit_params["d_model"],
        heads=vit_params["heads"],
        b=vit_params["b"],
        L_eff=vit_params["L_eff"],
        n_coups=ps.size,
        complex=True,
        disorder=True,
        transl_invariant=False,
        two_dimensional=(n_dim == 2),
    )
    
    # Load latest checkpoint
    state_path = find_latest_state(run_dir)
    print(f"📦 Loading model from: {state_path}")
    
    state = nk.serialization.deserialize(state_path)
    
    return {
        "state": state,
        "meta": meta,
        "hilbert": hi,
        "param_space": ps,
        "model": ma,
        "n_spins": n_spins,
        "L": L,
        "n_dim": n_dim,
        "h0_train": h0_train_list,
        "sigma": sigma_disorder,
        "J": J_val,
        "seed": seed,
        "n_replicas": n_replicas,
    }


def generate_test_disorder_configs(h0_list, n_test_per_h0, n_spins, sigma, rng=None):
    """Generate fresh disorder configurations for testing (not used in training)."""
    if rng is None:
        rng = np.random.default_rng()
    
    test_configs = []
    for h_val in h0_list:
        configs = rng.normal(loc=h_val, scale=sigma, size=(n_test_per_h0, n_spins))
        configs = np.abs(configs)  # Ensure positive
        test_configs.append(configs)
    
    return np.vstack(test_configs)


def compute_log_psi_derivatives(state_obj, disorder_config):
    """
    Compute derivatives of log wavefunction w.r.t. disorder parameters.
    
    Returns:
        grad_log_psi: (n_spins,) array of derivatives
    """
    def log_psi_fn(h_disorder):
        """Wrapper computing log amplitude for given disorder config."""
        states = state_obj.sample(disorder_config)
        log_vals = state_obj.log_value(states, disorder_config)
        return jnp.sum(log_vals)
    
    # Compute gradients for each disorder parameter
    grads = grad(log_psi_fn)
    grad_log_psi = grads(disorder_config)
    
    return grad_log_psi


def compute_qfi_matrix(state_obj, disorder_configs_batch, sampler=None):
    """
    Compute Quantum Fisher Information matrix for a batch of disorder configs.
    
    QFI: g_ij = <∂_i log ψ | ∂_j log ψ> - <∂_i log ψ><∂_j log ψ>
    
    Args:
        state_obj: NetKet FoundationalQuantumState or similar
        disorder_configs_batch: (batch_size, n_spins) array
        sampler: Optional sampler for computing expectation values
    
    Returns:
        qfi_matrices: (batch_size, n_spins, n_spins) array
        trace_qfi: (batch_size,) trace of QFI
        max_eig_qfi: (batch_size,) largest eigenvalue of QFI
    """
    batch_size, n_spins = disorder_configs_batch.shape
    qfi_matrices = np.zeros((batch_size, n_spins, n_spins))
    
    for b in tqdm(range(batch_size), desc="Computing QFI matrices"):
        W = disorder_configs_batch[b]
        
        # Sample states for this disorder config
        samples = state_obj.sample(W)
        log_psi_vals = state_obj.log_value(samples, W)
        
        # Compute derivatives ∂_i log ψ
        def single_log_psi(h_disorder):
            vals = state_obj.log_value(samples, h_disorder)
            return jnp.mean(vals)
        
        grad_fn = grad(single_log_psi)
        grad_log_psi = grad_fn(W)  # (n_spins,)
        
        # Compute second derivatives for full matrix
        def batch_derivative_i(h_disorder, i):
            def log_psi_i(hi):
                h_temp = h_disorder.at[i].set(hi)
                return jnp.mean(state_obj.log_value(samples, h_temp))
            return grad(log_psi_i)(h_disorder[i])
        
        # QFI matrix: g_ij = <∂_i log ψ ∂_j log ψ>
        # Approximate with finite differences for stability
        h_epsilon = 1e-4
        qfi = np.zeros((n_spins, n_spins))
        
        for i in range(n_spins):
            for j in range(i, n_spins):
                # <∂_i log ψ | ∂_j log ψ>
                W_ij_pp = W.copy()
                W_ij_pp[i] += h_epsilon
                W_ij_pp[j] += h_epsilon
                
                W_ij_pm = W.copy()
                W_ij_pm[i] += h_epsilon
                W_ij_pm[j] -= h_epsilon
                
                W_ij_mp = W.copy()
                W_ij_mp[i] -= h_epsilon
                W_ij_mp[j] += h_epsilon
                
                W_ij_mm = W.copy()
                W_ij_mm[i] -= h_epsilon
                W_ij_mm[j] -= h_epsilon
                
                log_vals_pp = np.mean(state_obj.log_value(samples, W_ij_pp))
                log_vals_pm = np.mean(state_obj.log_value(samples, W_ij_pm))
                log_vals_mp = np.mean(state_obj.log_value(samples, W_ij_mp))
                log_vals_mm = np.mean(state_obj.log_value(samples, W_ij_mm))
                
                second_deriv = (log_vals_pp - log_vals_pm - log_vals_mp + log_vals_mm) / (4 * h_epsilon**2)
                qfi[i, j] = second_deriv
                qfi[j, i] = second_deriv
        
        qfi_matrices[b] = qfi
    
    # Compute trace and eigenvalues
    trace_qfi = np.array([np.trace(qfi_matrices[b]) for b in range(batch_size)])
    max_eig_qfi = np.array([np.max(np.linalg.eigvalsh(qfi_matrices[b])) for b in range(batch_size)])
    
    return qfi_matrices, trace_qfi, max_eig_qfi


def compute_qfi_simplified(state_obj, disorder_configs_batch, n_samples_per_config=1000):
    """
    Efficient QFI computation: g_ij = <∂_i log ψ | ∂_j log ψ>
    
    For FNQS, we compute derivatives w.r.t. disorder parameters using automatic differentiation.
    QFI matrix element: g_ij = Σ_σ |ψ(σ)|² ∂_i(log|ψ|) ∂_j(log|ψ|)
    
    Since we're working with normalized states, we compute:
    g_ij ≈ (1/N_samples) Σ ∂_i(log ψ) ∂_j(log ψ)  (for complex log of wavefunction)
    
    Args:
        state_obj: NetKet FoundationalQuantumState
        disorder_configs_batch: (batch_size, n_spins) disorder configurations
        n_samples_per_config: Number of samples for convergence
    """
    batch_size, n_spins = disorder_configs_batch.shape
    trace_qfi = []
    max_eig_qfi = []
    
    h_epsilon = 1e-3  # Step size for finite differences
    
    for b in tqdm(range(batch_size), desc="Computing QFI"):
        W = disorder_configs_batch[b]
        
        try:
            # Build QFI matrix using numerical differentiation
            # G_ij = ∂²(log|ψ|)/∂W_i∂W_j evaluated at test states
            
            qfi = np.zeros((n_spins, n_spins))
            
            # We compute the matrix using a finite-difference approach
            # that respects the structure of the FNQS
            
            for i in range(n_spins):
                for j in range(i, n_spins):
                    
                    # Four-point stencil for mixed derivatives
                    W_pp = W.copy()
                    W_pp[i] += h_epsilon
                    W_pp[j] += h_epsilon
                    
                    W_pm = W.copy()
                    W_pm[i] += h_epsilon
                    W_pm[j] -= h_epsilon
                    
                    W_mp = W.copy()
                    W_mp[i] -= h_epsilon
                    W_mp[j] += h_epsilon
                    
                    W_mm = W.copy()
                    W_mm[i] -= h_epsilon
                    W_mm[j] -= h_epsilon
                    
                    # Sample states (once, reused for all evaluations)
                    samples = state_obj.sample(W)
                    
                    # Evaluate log ψ at each perturbed config
                    log_pp = np.mean(state_obj.log_value(samples, W_pp))
                    log_pm = np.mean(state_obj.log_value(samples, W_pm))
                    log_mp = np.mean(state_obj.log_value(samples, W_mp))
                    log_mm = np.mean(state_obj.log_value(samples, W_mm))
                    
                    # Mixed second derivative
                    mixed_deriv = (log_pp - log_pm - log_mp + log_mm) / (4 * h_epsilon ** 2)
                    
                    qfi[i, j] = mixed_deriv
                    qfi[j, i] = mixed_deriv
            
            # Symmetrize for numerical stability
            qfi = 0.5 * (qfi + qfi.T)
            
            # Regularization: ensure positive-semi-definite
            eigvals, eigvecs = np.linalg.eigh(qfi)
            eigvals = np.maximum(eigvals, 1e-10)  # Floor negative eigenvalues
            qfi = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Compute invariants
            trace_val = np.trace(qfi)
            max_eig = np.max(np.linalg.eigvalsh(qfi))
            
            trace_qfi.append(trace_val)
            max_eig_qfi.append(max_eig)
            
        except Exception as e:
            print(f"⚠ Warning: QFI computation failed at config {b}: {e}")
            trace_qfi.append(np.nan)
            max_eig_qfi.append(np.nan)
    
    return np.array(trace_qfi), np.array(max_eig_qfi)


# ==========================================
# MAIN ANALYSIS
# ==========================================

def main(run_dir, n_test_per_h0=5, plot_output=None):
    """
    Main analysis pipeline:
    1. Load trained model
    2. Generate test disorder configs
    3. Compute QFI for each config
    4. Plot results
    """
    
    print(f"\n{'='*60}")
    print(f"Quantum Fisher Information Analysis")
    print(f"{'='*60}\n")
    
    # Load model and metadata
    print("Step 1: Loading model and metadata...")
    model_data = load_model_and_state(run_dir)
    
    state = model_data["state"]
    n_spins = model_data["n_spins"]
    h0_train = model_data["h0_train"]
    sigma = model_data["sigma"]
    seed = model_data["seed"]
    n_replicas = model_data["n_replicas"]
    
    # Generate test configs
    print(f"\nStep 2: Generating test disorder configurations...")
    rng = np.random.default_rng(seed + 1000)  # Different seed for test set
    test_configs = generate_test_disorder_configs(
        h0_train, n_test_per_h0, n_spins, sigma, rng=rng
    )
    print(f"   Generated {test_configs.shape[0]} test configurations")
    
    # Compute average disorder value for each h0
    h0_avg_disorder = []
    for h_val in h0_train:
        mask = np.all((test_configs > h_val - 0.15*sigma) & (test_configs < h_val + 0.15*sigma), axis=1)
        if np.any(mask):
            disorder_level = np.mean(np.std(test_configs[mask], axis=1))
            h0_avg_disorder.append((h_val, disorder_level))
    
    # Compute QFI
    print(f"\nStep 3: Computing Quantum Fisher Information...")
    trace_qfi, max_eig_qfi = compute_qfi_simplified(state, test_configs)
    
    # Group QFI by disorder strength
    h0_values = []
    qfi_trace_by_h0 = {h: [] for h in h0_train}
    qfi_eig_by_h0 = {h: [] for h in h0_train}
    
    idx = 0
    for h_val in h0_train:
        for _ in range(n_test_per_h0):
            h0_values.append(h_val)
            qfi_trace_by_h0[h_val].append(trace_qfi[idx])
            qfi_eig_by_h0[h_val].append(max_eig_qfi[idx])
            idx += 1
    
    h0_values = np.array(h0_values)
    
    # Compute statistics
    h0_unique = sorted(set(h0_values))
    trace_mean = np.array([np.mean(qfi_trace_by_h0[h]) for h in h0_unique])
    trace_std = np.array([np.std(qfi_trace_by_h0[h]) for h in h0_unique])
    eig_mean = np.array([np.mean(qfi_eig_by_h0[h]) for h in h0_unique])
    eig_std = np.array([np.std(qfi_eig_by_h0[h]) for h in h0_unique])
    
    # Plot
    print(f"\nStep 4: Creating visualizations...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Trace of QFI
    ax = axes[0]
    ax.errorbar(h0_unique, trace_mean, yerr=trace_std, fmt='o-', 
                linewidth=2, markersize=8, capsize=5, label='QFI Trace',
                color='#1f77b4', ecolor='#1f77b4', alpha=0.8)
    ax.fill_between(h0_unique, trace_mean - trace_std, trace_mean + trace_std, 
                     alpha=0.2, color='#1f77b4')
    ax.set_xlabel("Disorder strength $h_0$", fontsize=12)
    ax.set_ylabel("Tr(g) - QFI Trace", fontsize=12)
    ax.set_title("Quantum Fisher Information: Trace (sensitivity to all parameters)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # Plot 2: Maximum eigenvalue
    ax = axes[1]
    ax.errorbar(h0_unique, eig_mean, yerr=eig_std, fmt='s-', 
                linewidth=2, markersize=8, capsize=5, label='QFI Max Eigenvalue',
                color='#ff7f0e', ecolor='#ff7f0e', alpha=0.8)
    ax.fill_between(h0_unique, eig_mean - eig_std, eig_mean + eig_std, 
                     alpha=0.2, color='#ff7f0e')
    ax.set_xlabel("Disorder strength $h_0$", fontsize=12)
    ax.set_ylabel("λ_max(g) - Maximum eigenvalue", fontsize=12)
    ax.set_title("Quantum Fisher Information: Largest Eigenvalue (most sensitive direction)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    if plot_output is None:
        plot_output = os.path.join(run_dir, "QFI_disorder_analysis.png")
    
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_output}")
    
    # Save numerical results
    results_file = os.path.join(run_dir, "QFI_results.json")
    results = {
        "h0_values": h0_unique.tolist(),
        "trace_qfi": trace_mean.tolist(),
        "trace_qfi_std": trace_std.tolist(),
        "max_eig_qfi": eig_mean.tolist(),
        "max_eig_qfi_std": eig_std.tolist(),
        "n_spins": int(n_spins),
        "n_test_configs_per_h0": n_test_per_h0,
        "disorder_sigma": float(sigma),
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to: {results_file}")
    
    # Print summary and interpretation
    print(f"\n{'='*60}")
    print(f"Analysis Summary & Physical Interpretation")
    print(f"{'='*60}")
    print(f"System size: {n_spins} spins")
    print(f"h0 range: {min(h0_unique):.2f} - {max(h0_unique):.2f}")
    print(f"Disorder σ: {sigma:.3f}")
    
    print(f"\n📊 QFI Trace (total parameter sensitivity):")
    print(f"  Min: {trace_mean.min():.4f} (at h0={h0_unique[trace_mean.argmin()]:.2f})")
    print(f"  Max: {trace_mean.max():.4f} (at h0={h0_unique[trace_mean.argmax()]:.2f})")
    trace_ratio = trace_mean.max() / (trace_mean.min() + 1e-8)
    print(f"  Ratio Max/Min: {trace_ratio:.2f}x")
    
    print(f"\n📈 QFI Max Eigenvalue (most sensitive direction):")
    print(f"  Min: {eig_mean.min():.4f} (at h0={h0_unique[eig_mean.argmin()]:.2f})")
    print(f"  Max: {eig_mean.max():.4f} (at h0={h0_unique[eig_mean.argmax()]:.2f})")
    eig_ratio = eig_mean.max() / (eig_mean.min() + 1e-8)
    print(f"  Ratio Max/Min: {eig_ratio:.2f}x")
    
    print(f"\n🔍 Phase Transition Indicators:")
    # Find peaks/discontinuities
    trace_deriv = np.gradient(trace_mean)
    max_eig_deriv = np.gradient(eig_mean)
    
    trace_peak_idx = np.argmax(np.abs(trace_deriv))
    eig_peak_idx = np.argmax(np.abs(max_eig_deriv))
    
    print(f"  Trace steepest change at h0 ≈ {h0_unique[trace_peak_idx]:.3f}")
    print(f"  Max eigenvalue steepest change at h0 ≈ {h0_unique[eig_peak_idx]:.3f}")
    print(f"  These points may indicate phase transitions or critical regions.")
    
    print(f"\n💡 Interpretation:")
    print(f"  • QFI Trace: Total sensitivity to parameter variations")
    print(f"    → Larger values = state is more 'distinguishable' with parameter changes")
    print(f"  • QFI Max Eigenvalue: Sensitivity in the most sensitive direction")
    print(f"    → Peak indicates enhanced sensitivity (possible quantum phase transition)")
    print(f"  • Ratio Max/Min: Indicates strength of the transition signature")
    print(f"    → Large ratio (>>1) = strong transition signature")
    
    if trace_ratio > 2:
        print(f"  ✓ Strong variation detected (ratio = {trace_ratio:.2f})")
    elif trace_ratio > 1.2:
        print(f"  ⚠ Moderate variation (ratio = {trace_ratio:.2f})")
    else:
        print(f"  ~ Weak variation (ratio = {trace_ratio:.2f})")
    
    print(f"{'='*60}\n")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot Quantum Fisher Information for disorder systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/
  python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/ --n-test 8
        """
    )
    parser.add_argument(
        "run_dir",
        help="Path to the run directory containing trained models and metadata"
    )
    parser.add_argument(
        "--n-test", type=int, default=5,
        help="Number of test configurations per h0 value (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for plot (default: <run_dir>/QFI_disorder_analysis.png)"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"❌ Error: Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    try:
        main(args.run_dir, n_test_per_h0=args.n_test, plot_output=args.output)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
