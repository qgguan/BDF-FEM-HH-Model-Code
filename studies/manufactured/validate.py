"""Manufactured time, space, joint, and quadrature studies for BDF1--5/P1--4."""

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from mesh import y_graph
from model import HHParameters
from solver import IMEXBDFSolver, SimulationConfig
from manufactured import ManufacturedSolution
from studies.paths import MANUFACTURED_RESULTS, PROJECT_ROOT


ERROR_KEYS = ("voltage_error", "gate_error", "gate_nodal_error", "voltage_h1_error")


def manufactured_run(order, degree, elements, dt, final_time, sample_interval,
                     quadrature=None, profile="cosine"):
    mesh, parameters = y_graph(elements, degree, quadrature), HHParameters()
    exact = ManufacturedSolution(mesh, parameters, profile)
    config = SimulationConfig(bdf_order=order, dt=dt, final_time=final_time,
                              sample_interval=sample_interval)
    result = IMEXBDFSolver(mesh, config, parameters).run(**exact.run_arguments())
    return result, exact.errors(result)


def difference(a, b):
    if (a.mesh.degree != b.mesh.degree or not np.array_equal(a.mesh.cells, b.mesh.cells)
            or not np.allclose(a.mesh.coordinates, b.mesh.coordinates, rtol=0, atol=1e-14)
            or a.times.shape != b.times.shape or not np.allclose(a.times, b.times)):
        raise ValueError("comparison requires the same mesh and stored times")
    return dict(voltage_error=float(np.max(a.mesh.l2_norm(a.voltage-b.voltage))),
                gate_error=float(np.max(a.mesh.l2_norm(a.gates-b.gates))),
                gate_nodal_error=float(np.max(a.mesh.nodal_norm(a.gates-b.gates))))


def add_rates(rows, scale):
    for j, row in enumerate(rows):
        for key in ERROR_KEYS:
            rate = None
            if j and min(row[key], rows[j-1][key]) > 0:
                rate = float(np.log(rows[j-1][key]/row[key]) / np.log(rows[j-1][scale]/row[scale]))
            row[key.replace("error", "rate")] = rate
    return rows


def time_study(order, degree, *, levels=4, elements=6, base_dt=0.05, quadrature=None):
    rows = []
    for level in range(levels):
        dt = base_dt / 2**level
        result, errors = manufactured_run(order, degree, elements, dt, 0.5, base_dt,
                                           quadrature, profile="uniform")
        rows.append(dict(dt=dt, h=result.mesh.h, dofs=result.mesh.ndof, **errors))
    return add_rates(rows, "dt"), dict(
        profile="uniform", final_time=0.5, sample_interval=base_dt,
        elements_per_edge=elements, reference="analytic continuous fields",
        reason="Spatially constant exact fields are represented exactly by every P degree; no spatial floor or history mismatch.",
        expected_voltage_rate=order, expected_gate_rate=order)


def joint_study(order, degree, *, levels=4, elements=4, base_dt=0.025, quadrature=None):
    rows = []
    for level in range(levels):
        n, dt = elements * 2**level, base_dt / 2**level
        result, errors = manufactured_run(order, degree, n, dt, 0.5, 2*base_dt, quadrature)
        rows.append(dict(elements=n, h=result.mesh.h, dt=dt, dofs=result.mesh.ndof, **errors))
    expected = min(order, degree+1)
    return add_rates(rows, "h"), dict(
        profile="cosine", final_time=0.5, sample_interval=2*base_dt,
        reference="analytic continuous fields", refinement="h and dt halved together",
        expected_voltage_rate=expected, expected_gate_rate=expected)


def space_study(order, degree, *, levels=4, elements=4, dt=0.001, quadrature=None):
    rows = []
    for level in range(levels):
        n = elements * 2**level
        result, errors = manufactured_run(order, degree, n, dt, 0.2, 0.05, quadrature)
        rows.append(dict(elements=n, h=result.mesh.h, dt=dt, dofs=result.mesh.ndof, **errors))
    finer, _ = manufactured_run(order, degree, n, dt/2, 0.2, 0.05, quadrature)
    change = difference(result, finer)
    ratios = {key: change[key]/max(errors[key], 1e-13) for key in ("voltage_error", "gate_error")}
    return add_rates(rows, "h"), dict(
        profile="cosine", final_time=0.2, sample_interval=0.05, dt=dt,
        reference="analytic continuous fields", expected_voltage_rate=degree+1,
        expected_gate_rate=degree+1, expected_voltage_h1_rate=degree,
        time_refinement_difference=change, time_difference_over_total_error=ratios,
        time_error_control_passed=max(ratios.values()) < 0.05,
        time_error_control_note="Empirical sensitivity test at finest mesh, not a rigorous temporal error bound.")


def quadrature_study(order, degree, *, quadrature=None):
    low = y_graph(8, degree, quadrature).quadrature_order
    high = max(40, low+8)
    a, errors = manufactured_run(order, degree, 8, 0.005, 0.2, 0.05, low)
    b, _ = manufactured_run(order, degree, 8, 0.005, 0.2, 0.05, high)
    change = difference(a, b)
    ratios = {key: change[key]/max(errors[key], 1e-13) for key in ("voltage_error", "gate_error")}
    passed = max(ratios.values()) < 0.02
    return dict(bdf_order=order, degree=degree, orders=[low, high], elements_per_edge=8,
                dt=0.005, final_time=0.2, difference=change,
                difference_over_total_error=ratios, passed=passed,
                note="Sensitivity of this manufactured test only; masks and nonpolynomial sources are not integrated exactly.")


def assess(rows, info):
    # Use both of the last two refinement rates; nodal gate superconvergence
    # is reported separately and is not used as a continuous L2 claim.
    passed = True
    for key in ("voltage", "gate"):
        tail = [row[f"{key}_rate"] for row in rows[-2:]]
        passed &= all(rate is not None and rate > info[f"expected_{key}_rate"]-0.65 for rate in tail)
    if "expected_voltage_h1_rate" in info:
        passed &= all(row["voltage_h1_rate"] is not None and
                      row["voltage_h1_rate"] > info["expected_voltage_h1_rate"]-0.65 for row in rows[-2:])
    return bool(passed and info.get("time_error_control_passed", True))


def save_study(directory, name, rows, info):
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / f"{name}.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")


def plot_comparison(records, directory):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [study for study in ("time", "joint", "space") if any(r["study"] == study for r in records)]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups)+1, 4), squeeze=False,
                             layout="constrained")
    for ax, study in zip(axes.ravel(), groups):
        shown_orders = sorted({row["bdf_order"] for row in records if row["study"] == study})
        values = np.full((len(shown_orders), 4), np.nan)
        for row in records:
            if row["study"] == study:
                values[shown_orders.index(row["bdf_order"]), row["degree"]-1] = row["voltage_rate"]
        image = ax.imshow(values, vmin=0, vmax=5, cmap="viridis", aspect="auto")
        for i, j in zip(*np.where(np.isfinite(values))):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", color="white" if values[i,j] < 3 else "black")
        ax.set(xticks=range(4), xticklabels=["P1", "P2", "P3", "P4"],
               yticks=range(len(shown_orders)), yticklabels=[f"BDF{k}" for k in shown_orders],
               title=f"{study.capitalize()} refinement")
    fig.colorbar(image, ax=list(axes.ravel()), label="Observed voltage L2 order")
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("time", "space", "joint", "quadrature", "all"), default="all")
    parser.add_argument("--bdf-orders", "--bdf-order", type=int, nargs="+", choices=range(1, 6), default=[3])
    parser.add_argument("--degrees", "--degree", type=int, nargs="+", choices=range(1, 5), default=[2])
    parser.add_argument("--all-combinations", action="store_true", help="select all 20 pairs")
    parser.add_argument("--levels", type=int, choices=range(3, 9), default=4)
    parser.add_argument("--elements", type=int, default=4, help="coarsest spatial/joint elements per branch")
    parser.add_argument("--time-elements", type=int, default=6)
    parser.add_argument("--time-dt", type=float, default=0.05)
    parser.add_argument("--joint-dt", type=float, default=0.025)
    parser.add_argument("--space-dt", type=float, default=0.001)
    parser.add_argument("--quadrature", type=int)
    parser.add_argument("--output", type=Path, default=MANUFACTURED_RESULTS)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    orders = range(1, 6) if args.all_combinations else sorted(set(args.bdf_orders))
    degrees = range(1, 5) if args.all_combinations else sorted(set(args.degrees))
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    records, details = [], []
    for study in ("time", "joint", "space", "quadrature"):
        if args.study not in (study, "all"):
            continue
        # In a full suite, isolate spatial accuracy using BDF5. Selected
        # combinations are all exercised by the time and joint studies.
        study_orders = [5] if args.study == "all" and study in ("space", "quadrature") else orders
        for order in study_orders:
            for degree in degrees:
                name = f"{study}_bdf{order}_p{degree}"
                print(f"Running {name}...", flush=True)
                if study == "quadrature":
                    info = quadrature_study(order, degree, quadrature=args.quadrature)
                    (directory / f"{name}.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
                    details.append(dict(study=study, **info))
                    print(f"  quadrature passed={info['passed']}", flush=True)
                    continue
                options = dict(levels=args.levels, quadrature=args.quadrature)
                if study == "time":
                    rows, info = time_study(order, degree, elements=args.time_elements, base_dt=args.time_dt, **options)
                elif study == "joint":
                    rows, info = joint_study(order, degree, elements=args.elements, base_dt=args.joint_dt, **options)
                else:
                    rows, info = space_study(order, degree, elements=args.elements, dt=args.space_dt, **options)
                info.update(bdf_order=order, degree=degree, study=study,
                            assembly_quadrature=y_graph(1, degree, args.quadrature).quadrature_order,
                            error_quadrature=max(40, (args.quadrature or 24)+8), passed=assess(rows, info))
                save_study(directory, name, rows, info)
                last = rows[-1]
                record = dict(study=study, bdf_order=order, degree=degree,
                              expected_rate=info["expected_voltage_rate"],
                              voltage_rate=last["voltage_rate"], gate_rate=last["gate_rate"],
                              gate_nodal_rate=last["gate_nodal_rate"],
                              voltage_error=last["voltage_error"], gate_error=last["gate_error"], passed=info["passed"])
                records.append(record)
                details.append(info)
                print(f"  V rate={last['voltage_rate']:.4f}, gates={last['gate_rate']:.4f}, "
                      f"expected={info['expected_voltage_rate']}, passed={info['passed']}", flush=True)
    if records:
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    passed = all(info["passed"] for info in details)
    summary = dict(all_passed=passed, studies=details,
                   environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                   error_definition="Maximum over stored times. Primary V/gate errors are continuous graph L2; gate_nodal is a separate Lobatto norm.")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if not args.no_plots:
        plot_comparison(records, directory)
    if not passed:
        raise SystemExit("Accuracy checks failed. Inspect tables, refine dt/h, and check time/quadrature sensitivity before interpreting rates.")
    print(f"All requested accuracy checks passed. Results: {directory.resolve()}")


if __name__ == "__main__":
    main()
