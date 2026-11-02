import numpy as np
import matplotlib.pyplot as plt


def P3(u):
    return u + u**3 / 3.0 + u**5 / 5.0


def phi(x, eps=0.1):
    x = np.asarray(x)
    y = np.empty_like(x, dtype=float)

    left = x < 0
    middle = (x >= 0) & (x <= 1)
    right = x > 1

    y[left] = eps * np.tanh(P3(x[left] / eps))
    y[middle] = x[middle]
    y[right] = 1.0 + eps * np.tanh(P3((x[right] - 1.0) / eps))
    return y


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

eps = 1.0e-3
x = np.linspace(-0.3, 1.3, 5000)

fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[2.25, 1.0])
ax = fig.add_subplot(gs[0, :])
ax_left = fig.add_subplot(gs[1, 0])
ax_right = fig.add_subplot(gs[1, 1])

ax.plot(x, phi(x, eps), linewidth=2.2, color="tab:blue", label=rf"$\phi(x)$")
ax.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.0, label="$y=x$ on $[0,1]$")
ax.axvline(0, color="0.55", linestyle="--", linewidth=1)
ax.axvline(1, color="0.55", linestyle="--", linewidth=1)
ax.axhline(-eps, color="tab:red", linestyle=":", linewidth=1.4)
ax.axhline(1 + eps, color="tab:red", linestyle=":", linewidth=1.4)
ax.text(0.5, -eps, r"$-\varepsilon$", color="tab:red", va="bottom", ha="left")
ax.text(0.5, 1 + eps, r"$1+\varepsilon$", color="tab:red", va="bottom", ha="left")
ax.set_title(r"$C^6$ masking function, $\varepsilon=10^{-3}$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\phi(x)$")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-0.1, 1.1)
ax.grid(True, alpha=0.24)
ax.legend(frameon=False, loc="upper left")

x_left = np.linspace(-0.004, 0.006, 1200)
ax_left.plot(x_left, phi(x_left, eps), color="tab:blue", linewidth=2)
ax_left.plot([0, 0.006], [0, 0.006], color="0.25", linestyle="--", linewidth=0.9)
ax_left.axhline(-eps, color="tab:red", linestyle=":", linewidth=1.2)
ax_left.axvline(0, color="0.55", linestyle="--", linewidth=0.9)
ax_left.set_title("Zoom near $x=0$")
ax_left.set_xlabel("$x$")
ax_left.set_ylabel(r"$\phi(x)$")
ax_left.set_xlim(-0.004, 0.006)
ax_left.set_ylim(-0.00125, 0.006)
ax_left.grid(True, alpha=0.24)

x_right = np.linspace(0.994, 1.004, 1200)
ax_right.plot(x_right, phi(x_right, eps), color="tab:blue", linewidth=2)
ax_right.plot([0.994, 1], [0.994, 1], color="0.25", linestyle="--", linewidth=0.9)
ax_right.axhline(1 + eps, color="tab:red", linestyle=":", linewidth=1.2)
ax_right.axvline(1, color="0.55", linestyle="--", linewidth=0.9)
ax_right.set_title("Zoom near $x=1$")
ax_right.set_xlabel("$x$")
ax_right.set_ylabel(r"$\phi(x)$")
ax_right.set_xlim(0.994, 1.004)
ax_right.set_ylim(0.994, 1.00125)
ax_right.grid(True, alpha=0.24)

fig.savefig("masking_function_phi.png", dpi=300)
fig.savefig("masking_function_phi.pdf")
