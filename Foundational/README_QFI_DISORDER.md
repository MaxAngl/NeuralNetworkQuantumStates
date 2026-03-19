# Quantum Fisher Information Analysis for Disorder Systems

## 📌 Overview

This script computes and visualizes the **Quantum Fisher Information (QFI)** matrix as a function of disorder strength for FNQS models on lattice systems. The QFI serves as a sensitive probe for detecting quantum phase transitions and critical phenomena.

## 🎯 Physical Background

The Quantum Fisher Information quantifies the sensitivity of a quantum state to parameter variations:

$$g_{ij} = \left\langle \frac{\partial \log \psi}{\partial W_i} \bigg| \frac{\partial \log \psi}{\partial W_j} \right\rangle$$

where:
- $\psi$ is the FNQS wavefunction
- $W_i$ are disorder parameters (on-site fields)
- The expectation value is over configurations sampled from the wavefunction

### Key Interpretations:

- **Trace $\text{Tr}(g)$**: Total parameter sensitivity (sum of all eigenvalues)
  - Larger values → state is more "distinguishable" with parameter changes
  - Indicates regions of high information content
  
- **Maximum Eigenvalue $\lambda_{\max}(g)$**: Sensitivity in the most sensitive direction
  - Peaks often coincide with quantum phase transitions
  - Reveals the dominant direction of parameter variation

- **Ratio $\lambda_{\max} / \lambda_{\min}$**: Anisotropy of the QFI geometry
  - Large values indicate strong phase transition signatures
  - Indicates broken symmetry effects

## 📦 Usage

### Basic Usage

```bash
python plot_QFI_disorder.py <path_to_run_directory>
```

### With Options

```bash
# Increase number of test configurations per h0 value
python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/ --n-test 10

# Specify output file
python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/ --output results.png

# Combined
python plot_QFI_disorder.py logs/run_2026-02-13_22-44-00/ --n-test 8 --output my_qfi_plot.png
```

### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `run_dir` | str | - | **Required.** Path to run directory with trained models |
| `--n-test` | int | 5 | Number of test configurations per h0 value |
| `--output` | str | `<run_dir>/QFI_disorder_analysis.png` | Output path for plot |

## 📁 Input Requirements

Your run directory must contain:

```
run_dir/
├── state_*.nk           # Trained model checkpoints (latest will be loaded)
├── meta.json            # Configuration metadata
└── disorder_configs.npy # Training disorder configurations
```

## 📊 Output Files

The script generates:

1. **`QFI_disorder_analysis.png`** - Main visualization with:
   - Trace of QFI vs disorder strength
   - Maximum eigenvalue vs disorder strength
   - Error bars from multiple realizations

2. **`QFI_results.json`** - Numerical results containing:
   - h0 values
   - Trace QFI and uncertainties
   - Max eigenvalue and uncertainties
   - Metadata about the computation

## 🔬 How It Works

### 1. Model Loading
- Automatically finds and loads the latest checkpoint (highest index state_*.nk)
- Reconstructs FNQS architecture from meta.json

### 2. Test Configuration Generation
- Creates **fresh** disorder configurations not seen during training
- Different random seed ensures independence from training set
- Multiple realizations per h0 value for error estimation

### 3. QFI Computation
For each test configuration:
- Samples quantum states using the FNQS
- Computes derivatives using 4-point finite difference stencil: 
  $$g_{ij} \approx \frac{\partial^2 \log|\psi|}{\partial W_i \partial W_j}$$
- Ensures positive-semi-definite matrix through eigenvalue regularization
- Computes trace and maximum eigenvalue

### 4. Statistical Analysis
- Averages QFI over multiple test configurations per h0
- Computes uncertainties (standard deviation)
- Identifies phase transition signatures through derivative analysis

## 🎨 Interpreting the Plots

### Trace of QFI

```
     ╱╲
    ╱  ╲
   ╱    ╲
  ╱      ╲___
 ╱
```

- **Peak structure**: Indicates region of high parameter sensitivity
- **Sharp transitions**: May signal first-order phase transitions
- **Smooth variations**: Characteristic of crossovers or smooth transitions

### Maximum Eigenvalue

- **Sharper peaks than trace**: Often more pronounced at transition points
- **Indicates dominant mode**: The most sensitive parameter direction

## 💡 Physical Interpretation Examples

### Strong Disorder-Induced Localization (1D Anderson Localization)
- **Trace of QFI**: Sharp peak at mobility edge
- **Interpretation**: Maximum sensitivity near transition from extended to localized states

### 2D Disordered Ising Model
- **Behavior varies by dimension**: Different transition signatures
- **Quantum Critical Point**: QFI often peaks at critical coupling strength

### Many-Body Localization (MBL)
- **Low-disorder phase**: Smooth, small QFI
- **MBL transition region**: Large peaks in both trace and max eigenvalue
- **MBL phase**: Different scaling behavior at high disorder

## ⚙️ Advanced Usage

### Adjusting Numerical Precision

Edit the `h_epsilon` parameter in `compute_qfi_simplified()` (line with `h_epsilon = 1e-3`):
- **Smaller** (e.g., 1e-4): More accurate but noisier
- **Larger** (e.g., 1e-2): Smoother but less precise

### Processing Multiple Runs

```python
import glob
import json

run_dirs = glob.glob("logs/run_*")
results_all = []

for run_dir in run_dirs:
    main(run_dir)
    with open(f"{run_dir}/QFI_results.json") as f:
        results_all.append(json.load(f))

# Now compare across different training configurations
```

## 🐛 Troubleshooting

### Memory Issues
- Reduce `--n-test` value
- The script processes one configuration at a time to minimize memory

### Numerical Instability
- Check `h_epsilon` value (adjust if needed)
- Ensure `disorder_configs.npy` contains reasonable values

### Cannot Load Model
- Verify run directory path is correct
- Check that `state_*.nk` files exist
- Verify `meta.json` is valid JSON

## 📚 References

The Quantum Fisher Information is a fundamental concept in quantum metrology:
- Enhancing precision in quantum measurement
- Detecting quantum phase transitions
- Characterizing quantum critical points

Related work:
- Braunstein & Caves (1994) - QFI in parameter estimation
- Paris (2009) - Quantum Estimation for Quantum Metrology
- Various applications in trapped ions, photonics, and condensed matter

## 📝 Citation

If you use this analysis in your work, please cite:

```bibtex
@software{qfi_disorder,
  title={Quantum Fisher Information Analysis for FNQS Disorder Systems},
  author={NeuralNetworkQuantumStates},
  year={2026},
  url={https://github.com/MaxAngl/NeuralNetworkQuantumStates}
}
```

## ✅ Requirements

- NetKet (with Pro support)
- JAX
- NumPy, Matplotlib
- Python 3.8+

Install via:
```bash
pip install netket netket-pro jax matplotlib numpy tqdm
```

---

**Last Updated**: March 2026  
**Version**: 1.0
