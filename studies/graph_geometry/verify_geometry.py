"""Save and illustrate the existing multiple-junction manufactured HH check.

Run from python-bdf-fem:
    python -B -m studies.graph_geometry.verify_geometry
    python -B -m studies.graph_geometry.verify_geometry --plot-only
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from manufactured import ManufacturedSolution
from mesh import Edge, GraphMesh
from model import HHParameters
from solver import IMEXBDFSolver, SimulationConfig
from studies.paths import PROJECT_ROOT


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
VERTICES = [[0, 0], [1, 0], [2, 0], [1, 1], [3, 1], [2, -1]]
EDGES = [Edge(0, 1, "A0"), Edge(1, 2, "A1", length=1.3),
         Edge(1, 3, "A2"), Edge(2, 4, "A3"), Edge(2, 5, "A4")]
COUNTS = np.array([4, 5, 4, 5, 4])
ERROR_KEYS = ("voltage_error", "gate_error", "voltage_h1_error")


def run_case(factor, *, dt=0.005, quadrature=24):
    mesh = GraphMesh(VERTICES, EDGES, factor * COUNTS, degree=4,
                     quadrature_order=quadrature)
    parameters = HHParameters()
    exact = ManufacturedSolution(mesh, parameters)
    config = SimulationConfig(bdf_order=5, dt=dt, final_time=0.1,
                              sample_interval=0.025)
    result = IMEXBDFSolver(mesh, config, parameters).run(**exact.run_arguments())
    return result, exact


def field_difference(a, b):
    """Continuous graph L2 changes on identical spatial meshes/sample times."""
    if not (np.array_equal(a.mesh.cells, b.mesh.cells)
            and np.array_equal(a.mesh.edge_lengths, b.mesh.edge_lengths)
            and np.allclose(a.times, b.times, rtol=0, atol=1e-14)):
        raise ValueError("controls require identical meshes and sampled times")
    mesh = a.mesh
    rule = mesh.integration_rule(40)
    ev = mesh.at_quadrature(a.voltage - b.voltage, rule)
    ew = mesh.at_quadrature(a.gates - b.gates, rule)
    return dict(voltage_error=float(np.max(np.sqrt(np.sum(
                    rule.weights * ev**2, axis=(-2, -1))))),
                gate_error=float(np.max(np.sqrt(np.sum(
                    rule.weights * ew**2, axis=(-2, -1))))))


def assembly_checks(mesh):
    junction_dofs = {}
    for vertex in (1, 2):
        incident = [int(mesh.edge_dofs[i][0 if edge.start == vertex else -1])
                    for i, edge in enumerate(mesh.edges)
                    if vertex in (edge.start, edge.end)]
        if len(set(incident)) != 1:
            raise AssertionError("junction values do not share a degree of freedom")
        junction_dofs[f"v{vertex}"] = incident
    mass_asymmetry = mesh.mass - mesh.mass.T
    stiffness_asymmetry = mesh.stiffness - mesh.stiffness.T
    return dict(shared_junction_dofs=junction_dofs,
                mass_integral=float(mesh.mass.sum()),
                total_metric_length=float(mesh.edge_lengths.sum()),
                mass_symmetry_defect=float(np.max(np.abs(mass_asymmetry.data), initial=0)),
                stiffness_symmetry_defect=float(np.max(np.abs(stiffness_asymmetry.data), initial=0)),
                constant_stiffness_residual=float(np.max(np.abs(mesh.stiffness @ np.ones(mesh.ndof)))),
                minimum_nodal_weight=float(mesh.nodal_weights.min()))


def compute(directory):
    rows = []
    for factor in (1, 2, 4, 8):
        result, exact = run_case(factor)
        result.save(directory / "runs" / f"factor{factor}")
        row = dict(factor=factor, elements=int(result.mesh.nelements),
                   dofs=result.mesh.ndof, h=result.mesh.h, dt=result.config.dt,
                   **exact.errors(result, error_quadrature=40))
        for key in ERROR_KEYS:
            row[key.replace("error", "rate")] = (
                float(np.log2(rows[-1][key] / row[key])) if rows else None)
        row["assembly"] = assembly_checks(result.mesh)
        rows.append(row)
        print(f"factor={factor}: V L2={row['voltage_error']:.8e}, "
              f"gate L2={row['gate_error']:.8e}", flush=True)

    controls = {}
    for name, kwargs in (("dt_half", dict(dt=0.0025)),
                         ("quadrature40", dict(quadrature=40))):
        control, _ = run_case(8, **kwargs)
        control.save(directory / "controls" / name)
        change = field_difference(result, control)
        ratios = {key: change[key] / rows[-1][key] for key in change}
        controls[name] = dict(settings=kwargs, field_change=change,
                              change_over_primary_error=ratios,
                              below_five_percent=max(ratios.values()) < 0.05)
    error80 = exact.errors(result, error_quadrature=80)
    quadrature_relative = {key: abs(error80[key] - rows[-1][key]) / rows[-1][key]
                           for key in ERROR_KEYS}
    sources = ["mesh.py", "model.py", "solver.py", "manufactured.py",
               "studies/graph_geometry/verify_geometry.py", "tests/test_solver.py"]
    report = dict(
        description="Manufactured HH spatial verification on the existing two-junction test graph",
        bdf_order=5, degree=4, dt=0.005, final_time=0.1,
        sampled_times=result.times.tolist(), assembly_quadrature=24,
        error_quadrature=40, coarse_elements_per_edge=COUNTS.tolist(),
        graph=result.mesh.metadata(), rows=rows, controls=controls,
        error_quadrature40_vs80_relative_change=quadrature_relative,
        error_definitions=dict(
            voltage_error="maximum sampled continuous voltage L2 error",
            gate_error="maximum over gates and sampled times of continuous L2 error (including interpolation)",
            gate_nodal_error="maximum sampled positive nodal gate error; supplementary only",
            voltage_h1_error="maximum sampled voltage derivative L2 error, an H1 SEMINORM, not the full H1 norm"),
        scope="Manufactured forcing; exact vertex derivatives vanish edgewise. Not a successive-junction physical pulse test.",
        source_sha256={path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                       for path in sources})
    (directory / "study.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    columns = [key for key in rows[0] if key != "assembly"]
    with (directory / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    if not all(control["below_five_percent"] for control in controls.values()):
        raise RuntimeError("a finest-grid sensitivity control exceeds five percent; inspect study.json")
    return report


def plot(report, directory):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})

    # A schematic embedding; the second edge's metric length overrides its drawing length.
    vertices = np.array(report["graph"]["vertices"])
    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout="constrained")
    labels = [(0.5, -0.26), (1.5, 0.23), (0.58, 0.56), (3.10, 0.46), (2.40, -0.52)]
    lengths = (r"1", r"1.3", r"1", r"\sqrt{2}", r"1")
    for i, edge in enumerate(report["graph"]["edges"]):
        a, b = vertices[[edge["start"], edge["end"]], :2]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#195A85", lw=2.8)
        for t in np.arange(1, COUNTS[i]) / COUNTS[i]:
            point = (1-t)*a + t*b
            ax.plot(*point, "o", ms=3.7, color="#9BB8CB")
        ax.text(*labels[i], rf"$A_{i}:\ L_{i}={lengths[i]}$" + "\n" + rf"$N_{i}={COUNTS[i]}r$",
                ha="center", va="center", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.8, alpha=0.94))
    offsets = [(-0.09, 0.16), (0.16, -0.17), (-0.18, -0.17), (0.08, 0.16), (0.08, 0.16), (0.08, -0.18)]
    for i, (point, offset) in enumerate(zip(vertices[:, :2], offsets)):
        junction = i in (1, 2)
        ax.plot(*point, "o", color="#AB461D" if junction else "#17364A",
                ms=8 if junction else 6, zorder=5)
        ax.text(*(point + offset), rf"$v_{i}$", ha="center", va="center", fontsize=12)
    ax.set(xlim=(-0.30, 3.35), ylim=(-1.32, 1.30), aspect="equal")
    ax.axis("off")
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"two_junction_geometry.{suffix}", dpi=600)
    plt.close(fig)
    plot_convergence(report, directory)


def plot_convergence(report, directory):
    """Plot saved L2 and derivative errors without recomputing any fields."""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    rows = report["rows"]
    h = np.array([row["h"] for row in rows])
    panels = (
        ("voltage_error", "#156082", "(a) Voltage", r"$E_V$ ($L^2$)", 5),
        ("gate_error", "#B14720", "(b) Gates", r"$E_W$ (maximum gate $L^2$)", 5),
        ("voltage_h1_error", "#38734C", "(c) Voltage derivative",
         r"$E_{V,1}$ ($H^1$ seminorm)", 4),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.25), layout="constrained")
    fig.supxlabel(r"Maximum element length $h$", fontsize=11)
    for ax, (key, color, title, ylabel, expected_order) in zip(axes, panels):
        error = np.array([row[key] for row in rows])
        ax.loglog(h, error, "o-", color=color, lw=1.8, ms=6, label="BDF5 + $P_4$")
        ax.loglog(h, 0.35 * error[0] * (h / h[0])**expected_order, "--",
                  color="#646D75", lw=1.4, label=rf"$C h^{expected_order}$")
        for index, row in enumerate(rows[1:], 1):
            rate = row[key.replace("error", "rate")]
            ax.annotate(f"{rate:.3f}", (np.sqrt(h[index-1]*h[index]),
                        np.sqrt(error[index-1]*error[index])),
                        ha="center", va="center", color=color, fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.95))
        ax.set(ylabel=ylabel, title=title)
        ax.set_xticks(h, labels=[f"{x:.4f}" for x in h])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(True, which="major", alpha=0.18)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"two_junction_convergence.{suffix}", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-only", action="store_true", help="redraw saved study.json without solving")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = (json.loads((args.output / "study.json").read_text(encoding="utf-8"))
              if args.plot_only else compute(args.output))
    plot(report, args.output)
    print(f"Saved study and figures to {args.output}")


if __name__ == "__main__":
    main()
