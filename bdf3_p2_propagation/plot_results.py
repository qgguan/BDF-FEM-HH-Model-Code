"""Plot existing output: python plot_results.py outputs/y_axon."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_graph(ax, mesh, nodal_voltage, basis, segments, linewidth=7):
    """Color the connected graph using the P2 voltage on every edge."""
    from matplotlib.collections import LineCollection
    values = nodal_voltage[mesh.cells] @ basis.T
    # Flat ends keep color boundaries perpendicular to the 1D branch.
    # Avoid antialiasing each tiny segment, which fades diagonal branches.
    collection = LineCollection(segments, cmap="turbo", linewidth=linewidth,
                                capstyle="butt", antialiaseds=False)
    collection.set_array(((values[:, :-1] + values[:, 1:]) / 2).ravel())
    collection.set_clim(-20, 120)
    ax.add_collection(collection)
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], c="black", s=12, zorder=3)
    ax.autoscale()
    ax.margins(0.15)
    ax.set_aspect("equal")
    return collection


def _plot_y_propagation(result, directory, basis, segments):
    """Show time evolution on the connected Y, with a shared voltage scale."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt = pyplot()
    mesh, times, voltage = result.mesh, result.times, result.voltage
    junction = mesh.edges[0].end
    junction_peak = int(np.argmax(voltage[:, junction]))
    # Include the branching event, with snapshots before and after it. For a
    # run without an interior junction peak, use evenly spaced sample times.
    if 0 < junction_peak < len(times) - 1:
        targets = np.concatenate((np.linspace(times[0], times[junction_peak], 4),
                                  np.linspace(times[junction_peak], times[-1], 3)[1:]))
    else:
        targets = np.linspace(times[0], times[-1], 6)
    indices = np.unique([int(np.argmin(np.abs(times - t))) for t in targets])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True,
                             layout="constrained")
    active_axes = axes.ravel()[:len(indices)]
    for ax, index in zip(active_axes, indices):
        collection = _draw_graph(ax, mesh, voltage[index], basis, segments)
        ax.set_title(f"t = {times[index]:g} ms", fontweight="bold")
        ax.set_xticks([-1, -0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])
    for ax in axes.ravel()[len(indices):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("x (cm)")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("y (cm)")
    # Edge labels follow the supplied geometry, so the shared junction stays
    # visible and the daughters remain identifiable in the combined drawing.
    for edge in mesh.edges:
        midpoint = (mesh.vertices[edge.start, :2] + mesh.vertices[edge.end, :2]) / 2
        offset = (0, 12) if midpoint[1] >= 0 else (0, -17)
        active_axes[0].annotate(edge.name, midpoint, xytext=offset,
                                textcoords="offset points", ha="center", fontsize=9)
    fig.suptitle("Voltage propagation through the Y-shaped axon", fontsize=16)
    fig.colorbar(collection, ax=list(active_axes), label="Shifted voltage (mV)",
                 shrink=0.88, ticks=np.arange(-20, 121, 20))
    fig.savefig(directory / "branch_voltage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    collection = _draw_graph(ax, mesh, voltage[0], basis, segments, linewidth=9)
    ax.set(xlabel="x (cm)", ylabel="y (cm)")
    title = ax.set_title(f"t = {times[0]:g} ms", fontweight="bold")
    fig.suptitle("Voltage propagation through the Y-shaped axon", fontsize=14)
    fig.colorbar(collection, ax=ax, label="Shifted voltage (mV)",
                 ticks=np.arange(-20, 121, 20))

    def update(index):
        values = voltage[index, mesh.cells] @ basis.T
        collection.set_array(((values[:, :-1] + values[:, 1:]) / 2).ravel())
        title.set_text(f"t = {times[index]:g} ms")
        return collection, title

    # Sample uniformly in physical time; include both endpoints and keep GIF
    # size bounded even when the simulation stores a much finer history.
    frame_times = np.linspace(times[0], times[-1], min(161, len(times)))
    frames = np.unique([int(np.argmin(np.abs(times - t))) for t in frame_times])
    animation = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    animation.save(directory / "branch_voltage.gif", writer=PillowWriter(fps=20), dpi=110)
    plt.close(fig)


def plot_result(result, directory):
    plt = pyplot()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mesh, times, voltage = result.mesh, result.times, result.voltage
    # Graph snapshot at the largest sampled voltage. Resolve each P2 element
    # into small plotting segments so the drawing follows the quadratic field.
    peak = int(np.argmax(voltage.max(axis=1)))
    xi = np.linspace(-1, 1, 9)
    basis = np.stack((xi * (xi - 1) / 2, 1 - xi**2, xi * (xi + 1) / 2), axis=1)
    xy = np.einsum("qi,eij->eqj", basis, mesh.coordinates[mesh.cells, :2])
    segments = np.stack((xy[:, :-1], xy[:, 1:]), axis=2).reshape(-1, 2, 2)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    collection = _draw_graph(ax, mesh, voltage[peak], basis, segments, linewidth=5)
    ax.set(xlabel="x (cm)", ylabel="y (cm)", title=f"Voltage on the graph, t = {times[peak]:.2f} ms")
    fig.colorbar(collection, ax=ax, label="Shifted voltage (mV)")
    fig.savefig(directory / "graph_voltage.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    probes = [(mesh.edge_dofs[0][len(mesh.edge_dofs[0]) // 2], "Trunk midpoint"),
              (mesh.edges[0].end, "Junction"),
              (mesh.edge_dofs[1][len(mesh.edge_dofs[1]) // 2], "Upper midpoint"),
              (mesh.edges[1].end, "Upper tip")]
    for node, label in probes:
        axes[0].plot(times, voltage[:, node], label=label)
    axes[0].set(ylabel="Shifted voltage (mV)", title="Action-potential propagation")
    axes[0].legend(loc="upper right", fontsize=9)
    junction = mesh.edges[0].end
    for j, name in enumerate(("m", "h", "n")):
        axes[1].plot(times, result.gates[:, j, junction], label=name)
    axes[1].set(xlabel="Time (ms)", ylabel="Gate value at junction")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.axvspan(0.1, 0.3, color="orange", alpha=0.2)
    fig.savefig(directory / "time_traces.png", dpi=180)
    plt.close(fig)

    _plot_y_propagation(result, directory, basis, segments)


def load_result(directory):
    from branched_hh import Edge, GraphMesh, HHParameters, SimulationConfig, SimulationResult
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    info = metadata["mesh"]
    mesh = GraphMesh(info["vertices"], [Edge(**e) for e in info["edges"]],
                     info["elements_per_edge"], info["quadrature_order"])
    with np.load(directory / "solution.npz", allow_pickle=False) as data:
        return SimulationResult(mesh, SimulationConfig(**metadata["config"]),
                                HHParameters(**metadata["parameters"]), data["times"],
                                data["voltage"], data["gates"], metadata["diagnostics"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plot_result(load_result(args.directory), args.directory)
