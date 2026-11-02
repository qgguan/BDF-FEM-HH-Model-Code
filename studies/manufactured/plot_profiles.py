"""Exact manufactured voltage profiles in the three-panel branch layout.

Run from python-bdf-fem:
    python -B -m studies.manufactured.plot_profiles
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from manufactured import ManufacturedSolution
from mesh import y_graph
from model import HHParameters
from studies.paths import MANUFACTURED_RESULTS, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", type=float, nargs="+", default=[0, 0.25, 0.5, 1])
    parser.add_argument("--output", type=Path, default=MANUFACTURED_RESULTS)
    args = parser.parse_args()
    times = np.array(args.times, dtype=float)
    if not len(times) or not np.all(np.isfinite(times)) or np.any(times < 0):
        parser.error("times must be finite and nonnegative")
    if len(times) > 6:
        parser.error("use at most six times to keep the figure legible")
    args.output.mkdir(parents=True, exist_ok=True)

    # The mesh supplies graph lengths only; the plotted curves are analytic
    # fields, not FE interpolants or outputs of a time-stepping method.
    mesh = y_graph(elements=1, degree=1)
    exact = ManufacturedSolution(mesh, HHParameters())
    ell = np.linspace(0, 1, 1001)
    profiles = np.stack([
        [exact.voltage(t, np.full(ell.shape, edge, dtype=int), ell)
         for edge in range(3)] for t in times])
    for index, t in enumerate(times):
        np.testing.assert_allclose(profiles[index, 0], profiles[index, 1], atol=1e-14)
        np.testing.assert_allclose(profiles[index, 0], profiles[index, 2], atol=1e-14)
        np.testing.assert_allclose(profiles[index, :, [0, -1]], 10 + 3*np.exp(-t), atol=1e-14)
        for edge in range(3):
            np.testing.assert_allclose(exact.derivative(t, edge, np.array([0, 1])), 0, atol=1e-13)

    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 12, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#17becf"]
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    for edge, (ax, name) in enumerate(zip(axes, ("Trunk", "Upper", "Lower"))):
        for index, t in enumerate(times):
            ax.plot(ell, profiles[index, edge], color=colors[index],
                    linestyle=styles[index], linewidth=2,
                    label=rf"$t={t:g}$ ms")
        ax.set(title=rf"{name} ($A_{edge}$)", xlim=(0, 1), ylim=(6.75, 13.25),
               xlabel=rf"Arc length $\ell_{edge}$ (cm)")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(range(7, 14))
        ax.grid(True, linewidth=0.6, alpha=0.24)
    axes[0].set_ylabel(r"Exact voltage $V^{\mathrm{ex}}$ (mV)")
    fig.suptitle("Manufactured voltage on the Y-shaped graph", fontsize=16, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.015),
               ncol=min(len(times), 4), frameon=True, handlelength=3,
               columnspacing=2)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.23, wspace=0.11)

    stem = args.output / "manufactured_y_profiles"
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Exact manufactured voltage profiles on the Y-shaped graph",
        "Subject": "Analytic HH verification field V=10+3*exp(-t)*cos(2*pi*ell) on each unit-length branch.",
    })
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(dict(
        field="exact manufactured voltage", formula="10 + 3*exp(-t)*cos(2*pi*ell)",
        branch_names=["trunk", "upper", "lower"], branch_lengths_cm=[1, 1, 1],
        times_ms=times.tolist(), arc_length_range_cm=[0, 1], samples_per_branch=len(ell),
        source="manufactured.py:ManufacturedSolution.voltage",
        coordinate="ell measured from trunk left endpoint or daughter junction",
        figure_size_inches=[13.5, 4.4], png_dpi=600,
        notes="Analytic fields; identical on all branches. No time stepping or FE approximation."
    ), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {stem.with_suffix('.pdf')} and 600-dpi PNG")


if __name__ == "__main__":
    main()
