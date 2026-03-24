import numpy as np
import matplotlib.pyplot as plt

data = np.load("mz2_data_L=16.npz")

sigma_grid = data["sigma_grid"]
H0_TEST_LIST = data["h0_grid"].tolist()
L = int(data["L"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_grid)))
max_derivatives = []

for idx, sigma in enumerate(sigma_grid):
    c = colors[idx]
    mean_vals = data["mz2_mean"][idx]
    min_vals  = data["mz2_min"][idx]
    max_vals  = data["mz2_max"][idx]

    ax1.plot(H0_TEST_LIST, mean_vals, marker='o', markersize=3, color=c,
             linewidth=1, label=rf"$\sigma = {sigma:.2f}$", zorder=3)
    ax1.plot(H0_TEST_LIST, min_vals, linestyle='--', color=c, alpha=0.5, linewidth=0.5, zorder=2)
    ax1.plot(H0_TEST_LIST, max_vals, linestyle='--', color=c, alpha=0.5, linewidth=0.5, zorder=2)

    derivative = np.gradient(mean_vals, H0_TEST_LIST)
    max_derivatives.append(np.max(np.abs(derivative)))

ax1.set_xlabel(r"Transverse Field $h_0$", fontsize=12)
ax1.set_ylabel(r"Squared Magnetization $\langle M_z^2 \rangle$", fontsize=12)
ax1.set_title(f"Magnetization order parameter vs Transverse Field (L={L})", fontsize=14)
ax1.grid(True, which="both", ls="--", alpha=0.3)
ax1.legend(loc='upper right', frameon=True, fontsize=10, title="Disorder strength")

ax2.plot(sigma_grid, max_derivatives, marker='s', markersize=6,
         color='crimson', linewidth=1.5, linestyle='-')
ax2.set_xlabel(r"Disorder strength $\sigma$", fontsize=12)
ax2.set_ylabel(r"$\max \left| \frac{\partial \langle M_z^2 \rangle}{\partial h_0} \right|$", fontsize=12)
ax2.set_title(r"Maximum Susceptibility vs Disorder", fontsize=14)
ax2.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(f"mz2_replot_L={L}.pdf")
plt.show()