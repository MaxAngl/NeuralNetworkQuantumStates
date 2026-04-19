"""
Comparison of two susceptibility estimators vs disorder sigma (1D, L=48):

    Blue  : max_{h0} |d/dh0  mean_{disorder}(Mz^2)|   (max of mean derivative)
    Red   : mean_{disorder}[ max_{h0} |d/dh0 Mz^2| ]  (mean of max derivative) ± stderr

Derivative computed with robust_derivative (span=1).
Data source: is_data_1D_L48_full.npz  (mz2_raw shape: sigma x h0 x configs)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR,
             "../../logs/Trains_autour_transi_1D/L=48/is_data_1D_L48_full.npz")
OUT_DIR    = os.path.join(SCRIPT_DIR,
             "../../logs/Trains_autour_transi_1D/L=48")

# ── parameters ───────────────────────────────────────────────────────────────
# h0_grid covers [0.8, 1.2] — use full range
H0_MIN = 0.8
H0_MAX = 1.2
SPAN   = 1     # derivative span (robust central difference)

# ── load ─────────────────────────────────────────────────────────────────────
data       = np.load(DATA_PATH)
sigma_grid = data["sigma_grid"]          # (n_sigma,)
h0_grid    = data["h0_grid"]             # (n_h0,)
mz2_raw    = data["mz2_raw"]             # (n_sigma, n_h0, n_configs)
L          = int(data["L"])

print(f"L={L}, sigma_grid={sigma_grid}")
print(f"h0_grid: [{h0_grid[0]:.3f}, {h0_grid[-1]:.3f}], {len(h0_grid)} points")
print(f"mz2_raw shape={mz2_raw.shape}")

# ── robust derivative (same as plot_IS.py) ───────────────────────────────────
def robust_derivative(y, x, step=1):
    dy = np.zeros_like(y)
    n  = len(y)
    for i in range(n):
        left  = max(0,     i - step)
        right = min(n - 1, i + step)
        if right == left:
            dy[i] = 0.0
        else:
            dy[i] = (y[right] - y[left]) / (x[right] - x[left])
    return np.abs(dy)

# ── transition window mask on h0_grid ────────────────────────────────────────
trans_mask = (h0_grid >= H0_MIN) & (h0_grid <= H0_MAX)

# ── compute both estimators for each sigma ───────────────────────────────────
max_of_mean = np.full(len(sigma_grid), np.nan)   # blue
mean_of_max = np.full(len(sigma_grid), np.nan)   # red
err_of_max  = np.full(len(sigma_grid), np.nan)   # red stderr

for idx_s in range(len(sigma_grid)):
    # ── blue: derivative of the disorder-averaged curve ──────────────────────
    mean_curve = np.nanmean(mz2_raw[idx_s], axis=1)          # (n_h0,)
    deriv_mean = robust_derivative(mean_curve, h0_grid, step=SPAN)
    max_of_mean[idx_s] = np.max(deriv_mean[trans_mask])

    # ── red: per-config max of derivative, then average ───────────────────────
    n_configs = mz2_raw.shape[2]
    max_list  = []
    for k in range(n_configs):
        curve_k    = mz2_raw[idx_s, :, k]
        valid_mask = ~np.isnan(curve_k)
        if np.sum(valid_mask) > 2 * SPAN:
            h0_v    = h0_grid[valid_mask]
            mz2_v   = curve_k[valid_mask]
            deriv_k = robust_derivative(mz2_v, h0_v, step=SPAN)
            win_mask = (h0_v >= H0_MIN) & (h0_v <= H0_MAX)
            if np.any(win_mask):
                max_list.append(np.max(deriv_k[win_mask]))

    if max_list:
        arr = np.array(max_list)
        mean_of_max[idx_s] = arr.mean()
        err_of_max[idx_s]  = arr.std(ddof=1) / np.sqrt(len(arr))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(sigma_grid, max_of_mean,
        color="blue", marker="o", markersize=4, linewidth=1.2,
        label=r"$\max_{h_0}\left|\frac{\partial}{\partial h_0}"
              r"\,\overline{\langle M_z^2\rangle}\right|$")

ax.errorbar(sigma_grid, mean_of_max, yerr=err_of_max,
            color="red", marker="s", markersize=4, linewidth=1.2,
            elinewidth=1.0, capsize=3,
            label=r"$\overline{\max_{h_0}\left|\frac{\partial}{\partial h_0}"
                  r"\,\langle M_z^2\rangle\right|}$")

ax.set_xlabel(r"Disorder strength $\sigma$", fontsize=12)
ax.set_ylabel(r"Peak susceptibility", fontsize=12)
ax.set_title(rf"Max-of-mean vs Mean-of-max  —  1D $L={L}$, span={SPAN}, "
             rf"$h_0 \in [{H0_MIN}, {H0_MAX}]$", fontsize=11)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()

base = "meanofmax_vs_maxofmean"
for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"{base}.{ext}")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")

plt.close()
