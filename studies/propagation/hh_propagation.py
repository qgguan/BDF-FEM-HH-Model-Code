"""Full HH pulse comparisons with every-step probes and checked references."""

import argparse
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import scipy
from mesh import y_graph
from model import HHParameters
from solver import IMEXBDFSolver, SimulationConfig
from pulse_metrics import Probe, ProbeRecorder, y_probes, measure_pulses, branch_speeds
from studies.paths import PROJECT_ROOT, PROPAGATION_RESULTS


@dataclass(frozen=True)
class Case:
    name: str
    bdf_order: int
    degree: int
    elements: int
    dt: float
    final_time: float
    quadrature: int = 24


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numerical_source_hash():
    digest = hashlib.sha256()
    for name in ("mesh.py", "model.py", "solver.py"):
        digest.update((PROJECT_ROOT/name).read_bytes())
    return digest.hexdigest()


def run_case(case, output, thresholds, recompute=False, *, progress=True):
    directory = output/"runs"/case.name
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path, trace_path = directory/"metadata.json", directory/"probe_traces.npz"
    signature = numerical_source_hash()
    cached = False
    if metadata_path.exists() and trace_path.exists() and not recompute:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = (metadata.get("case") == asdict(case) and metadata.get("source_sha256") == signature
                  and metadata.get("status") == "complete")
    if cached:
        with np.load(trace_path, allow_pickle=False) as data:
            times, voltage = data["times_ms"], data["voltage_mV"]
        probes = [Probe(**probe) for probe in metadata["probes"]]
        if len(times) != round(case.final_time/case.dt)+1:
            raise ValueError("cached probe history is incomplete")
        print(f"Reusing {case.name}", flush=True)
    else:
        print(f"Running {case.name}: BDF{case.bdf_order}+P{case.degree}, "
              f"N={case.elements}, dt={case.dt:g}, Q={case.quadrature}", flush=True)
        mesh = y_graph(case.elements, case.degree, case.quadrature)
        probes = y_probes(mesh)
        recorder = ProbeRecorder(mesh, probes)
        # Store only endpoint field snapshots; the observer captures every
        # time step at fixed physical probes without storing the whole field.
        config = SimulationConfig(bdf_order=case.bdf_order, dt=case.dt,
                                  final_time=case.final_time, sample_interval=case.final_time)
        result = IMEXBDFSolver(mesh, config).run(progress=progress, observer=recorder)
        times, voltage = recorder.arrays()
        if len(times) != round(case.final_time/case.dt)+1:
            raise RuntimeError("observer missed a time level")
        np.savez_compressed(trace_path, times_ms=times, voltage_mV=voltage,
                            probe_names=np.array([probe.name for probe in probes]),
                            edge_indices=np.array([probe.edge for probe in probes]),
                            arc_lengths_cm=np.array([probe.s_cm for probe in probes]))
        metadata = dict(status="complete", case=asdict(case), config=asdict(config),
                        parameters=asdict(result.parameters), mesh=mesh.metadata(),
                        diagnostics=result.diagnostics, probes=recorder.metadata(),
                        source_sha256=signature,
                        trace_axes=["time_step", "probe"],
                        event_definition="First positive-slope upward crossing after t=0.3 ms.",
                        event_interpolation="Local polynomial of degree equal to BDF order; bracketed root.",
                        peak_interpolation="Stationary point of a local polynomial; any fallback is recorded.")
        write_json(metadata_path, metadata)
    events = measure_pulses(times, voltage, probes, case.bdf_order, thresholds)
    speeds = branch_speeds(events, thresholds)
    write_csv(directory/"arrivals_and_peaks.csv", events)
    write_csv(directory/"branch_speeds.csv", speeds)
    return dict(case=case, metadata=metadata, times=times, voltage=voltage,
                probes=probes, events=events, speeds=speeds)


def primary_events(run, threshold=20):
    return {row["probe"]: row for row in run["events"] if row["threshold_mV"] == threshold}


def primary_speeds(run, threshold=20):
    return {row["branch"]: row for row in run["speeds"] if row["threshold_mV"] == threshold}


def comparison_rows(run, reference):
    ref_events = {(row["probe"], row["threshold_mV"]): row for row in reference["events"]}
    ref_speeds = {(row["branch"], row["threshold_mV"]): row for row in reference["speeds"]}
    case = asdict(run["case"])
    event_rows, speed_rows = [], []
    for row in run["events"]:
        ref = ref_events[(row["probe"], row["threshold_mV"])]
        comparison = dict(case, **row)
        for key, label in (("arrival_ms", "arrival_error_ms"),
                           ("peak_time_ms", "peak_time_error_ms"),
                           ("peak_voltage_mV", "peak_voltage_error_mV")):
            a, b = row[key], ref[key]
            comparison[f"reference_{key}"] = b
            comparison[label] = None if a is None or b is None else a-b
        event_rows.append(comparison)
    for row in run["speeds"]:
        ref = ref_speeds[(row["branch"], row["threshold_mV"])]
        speed, reference_speed = row["speed_m_per_s"], ref["speed_m_per_s"]
        speed_rows.append(dict(case, **row, reference_speed_m_per_s=reference_speed,
                              relative_speed_error_percent=None if speed is None or reference_speed is None
                              else 100*(speed-reference_speed)/reference_speed))
    return event_rows, speed_rows


def max_absolute(rows, key):
    values = [abs(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def reference_change(reference, check):
    events, speeds = comparison_rows(check, reference)
    primary = [row for row in events if row["threshold_mV"] == 20]
    speed_rows = [row for row in speeds if row["threshold_mV"] == 20]
    arrival = max_absolute(primary, "arrival_error_ms")
    speed = max_absolute(speed_rows, "relative_speed_error_percent")
    peak = max_absolute(primary, "peak_voltage_error_mV")
    valid = all(row["arrival_status"] == "arrived" and row["peak_status"] == "complete"
                and row["crossing_count"] == 1 for row in primary)
    valid &= all(row["arrival_status"] == "arrived" and row["peak_status"] == "complete"
                 and row["crossing_count"] == 1 for row in primary_events(reference).values())
    return dict(check=check["case"].name, elements=check["case"].elements,
                dt=check["case"].dt, quadrature=check["case"].quadrature,
                max_arrival_change_ms=arrival, max_speed_change_percent=speed,
                max_peak_voltage_change_mV=peak,
                all_probes_complete=valid,
                passed=bool(valid and arrival is not None and arrival < 0.002
                            and speed is not None and speed < 0.01 and peak is not None and peak < 0.02))


def summarize(run, reference):
    event_rows, speed_rows = comparison_rows(run, reference)
    primary = [row for row in event_rows if row["threshold_mV"] == 20]
    speeds = [row for row in speed_rows if row["threshold_mV"] == 20]
    lookup, velocity = primary_events(run), primary_speeds(run)
    symmetry = []
    for s in (0.25, 0.5, 0.75, 0.9, 1):
        a, b = lookup[f"upper_{s:g}"]["arrival_ms"], lookup[f"lower_{s:g}"]["arrival_ms"]
        if a is not None and b is not None:
            symmetry.append(abs(a-b))
    spread = []
    for branch in ("trunk", "upper", "lower"):
        values = [row["speed_m_per_s"] for row in run["speeds"]
                  if row["branch"] == branch and row["speed_m_per_s"] is not None]
        if values:
            spread.append(100*(max(values)-min(values))/np.mean(values))
    summary = dict(asdict(run["case"]), dofs=run["metadata"]["mesh"]["dofs"],
                   arrived_probes=sum(row["arrival_status"] == "arrived" for row in primary),
                   complete_peaks=sum(row["peak_status"] == "complete" for row in primary),
                   probe_count=len(primary),
                   max_arrival_error_ms=max_absolute(primary, "arrival_error_ms"),
                   max_interior_arrival_error_ms=max_absolute([r for r in primary if r["role"] == "interior"], "arrival_error_ms"),
                   max_speed_error_percent=max_absolute(speeds, "relative_speed_error_percent"),
                   max_peak_time_error_ms=max_absolute(primary, "peak_time_error_ms"),
                   max_peak_voltage_error_mV=max_absolute(primary, "peak_voltage_error_mV"),
                   junction_arrival_ms=lookup["junction"]["arrival_ms"],
                   upper_tip_arrival_ms=lookup["upper_1"]["arrival_ms"],
                   lower_tip_arrival_ms=lookup["lower_1"]["arrival_ms"],
                   max_daughter_arrival_asymmetry_ms=max(symmetry) if symmetry else None,
                   max_threshold_speed_spread_percent=max(spread) if spread else None,
                   arrival_linear_fallbacks=sum(row["arrival_method"] == "linear_fallback" for row in run["events"]),
                   ambiguous_probes=sum(row["crossing_count"] > 1 for row in primary),
                   gate_mask_steps=run["metadata"]["diagnostics"]["gate_mask_steps"],
                   voltage_cutoff_steps=run["metadata"]["diagnostics"]["voltage_cutoff_steps"],
                   solver_seconds=run["metadata"]["diagnostics"]["elapsed_seconds"])
    for branch in ("trunk", "upper", "lower"):
        summary[f"{branch}_speed_m_per_s"] = velocity[branch]["speed_m_per_s"]
    return summary, event_rows, speed_rows


def plot_results(output, summaries, reference, sensitivity):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    orders = sorted({row["bdf_order"] for row in summaries})
    degrees = sorted({row["degree"] for row in summaries})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, key, title, units, floor in zip(
            axes, ("max_arrival_error_ms", "max_speed_error_percent"),
            ("Maximum arrival-time difference", "Maximum branch-speed difference"),
            ("ms", "%"), (sensitivity["arrival_ms"], sensitivity["speed_percent"])):
        values = np.full((len(orders), len(degrees)), np.nan)
        for row in summaries:
            if row[key] is not None:
                values[orders.index(row["bdf_order"]), degrees.index(row["degree"])] = row[key]
        finite = values[np.isfinite(values)]
        lower = max(float(min(finite)) if len(finite) else 1e-8, floor, 1e-10)
        upper = max(float(max(finite)) if len(finite) else 1.0, lower*10)
        image = ax.imshow(np.maximum(values, lower), cmap="viridis", norm=LogNorm(lower, upper), aspect="auto")
        for i, j in zip(*np.where(np.isfinite(values))):
            value = values[i, j]
            label = f"{value:.2g}"+("*" if value <= floor else "")
            color = "black" if max(value, lower) > np.sqrt(lower*upper) else "white"
            ax.text(j, i, label, ha="center", va="center", color=color)
        ax.set(xticks=range(len(degrees)), xticklabels=[f"P{p}" for p in degrees],
               yticks=range(len(orders)), yticklabels=[f"BDF{k}" for k in orders], title=title)
        fig.colorbar(image, ax=ax, label=units, shrink=0.85)
    fig.suptitle("HH pulse: 20 mV upward crossings vs refined reference")
    settings = {(row["elements"], row["dt"]) for row in summaries}
    caption = ""
    if len(settings) == 1:
        elements, dt = next(iter(settings))
        caption = f"{elements} elements per branch; time step = {dt:g} ms. "
    ref = reference["case"]
    caption += f"Reference: BDF{ref.bdf_order}/P{ref.degree}, {ref.elements} elements, time step = {ref.dt:g} ms."
    unresolved = any((row["max_arrival_error_ms"] is not None and row["max_arrival_error_ms"] <= sensitivity["arrival_ms"])
                     or (row["max_speed_error_percent"] is not None and row["max_speed_error_percent"] <= sensitivity["speed_percent"])
                     for row in summaries)
    if unresolved:
        caption += "\n* Difference comparable to or below the measured reference sensitivity"
    fig.supxlabel(caption, fontsize=10)
    fig.savefig(output/"arrival_speed_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    names = ["trunk_0.25", "trunk_0.75", "junction", "upper_0.25", "upper_0.75", "upper_1"]
    events = primary_events(reference)
    for name in names:
        j = next(i for i, probe in enumerate(reference["probes"]) if probe.name == name)
        probe = reference["probes"][j]
        label = "Junction" if name == "junction" else f"{name.split('_')[0].capitalize()}, s={probe.s_cm:g} cm"
        line, = ax.plot(reference["times"], reference["voltage"][:, j], label=label)
        arrival = events[name]["arrival_ms"]
        if arrival is not None:
            ax.plot(arrival, 20, "o", color=line.get_color(), markersize=4)
    ax.axhline(20, linestyle="--", color="black", linewidth=1, label="Arrival threshold")
    ax.set(xlabel="Time (ms)", ylabel="Shifted voltage (mV)",
           title="Refined HH reference: probe traces and 20 mV arrivals")
    ax.grid(alpha=0.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=4, fontsize=9)
    fig.savefig(output/"reference_probe_traces.png", dpi=180)
    plt.close(fig)


def write_readme(output, reference, checks, sensitivity, summaries):
    case = reference["case"]
    velocity = primary_speeds(reference)
    events = primary_events(reference)
    additional = ", ".join(f"{x:g}" for x in sorted({row["threshold_mV"] for row in reference["events"]}) if x != 20)
    lines = ["# HH pulse propagation results", "",
             "Arrival is the first upward crossing of 20 mV after the stimulus ends at 0.3 ms.",
             f"Voltages are shifted so that rest is 0 mV. Additional thresholds (mV): {additional or 'none'}.", "",
             "Probe values are FE evaluations at fixed physical arc lengths, recorded at every time step.",
             "Arrival roots use local degree-K time interpolation for BDFK. Peak times are interpolated stationary points.",
             "Missing or unbracketed events are marked explicitly; they are not assigned zero times.", "",
             "Speeds use the 0.5 cm arc-length separation between s=0.25 and s=0.75 on each branch.",
             "The conversion is 1 cm/ms = 10 m/s. Tip arrivals are reported separately from interior speeds.", "",
             "## Reference", "",
             f"BDF{case.bdf_order}+P{case.degree}, {case.elements} elements per branch, dt={case.dt:g} ms, "
             f"Q={case.quadrature}, T={case.final_time:g} ms.", "",
             f"Independent reference checks passed: {all(row['passed'] for row in checks)}.",
             f"Sum of measured arrival changes: {sensitivity['arrival_ms']:.6g} ms.",
             f"Sum of measured speed changes: {sensitivity['speed_percent']:.6g}%.",
             "These sensitivities are empirical estimates, not rigorous bounds on the reference error.", "",
             "| Branch | Arrival at 0.25 cm (ms) | Arrival at 0.75 cm (ms) | Speed (m/s) |",
             "| --- | ---: | ---: | ---: |"]
    for branch, row in velocity.items():
        def number(value):
            return "unavailable" if value is None else f"{value:.8f}"
        lines.append(f"| {branch} | {number(row['arrival_a_ms'])} | {number(row['arrival_b_ms'])} | {number(row['speed_m_per_s'])} |")
    lines += ["", f"Junction arrival: {events['junction']['arrival_ms']} ms.",
              f"Upper-tip arrival: {events['upper_1']['arrival_ms']} ms.", "",
              "## Files", "",
              "- `summary.csv`: one row per tested method, including timing errors, speed errors, peak errors, and diagnostics.",
              "- `arrival_comparison.csv`: all probes and thresholds, with signed differences from the reference.",
              "- `speed_comparison.csv`: all branch speeds and thresholds, with signed relative differences.",
              "- `reference_checks.csv`: independent mesh, time-step, and quadrature changes.",
              "- `study.json`: study settings, source fingerprint, reference sensitivities, and software versions.",
              "- `runs/<case>/probe_traces.npz`: `times_ms`, `voltage_mV` (time, probe), names, edges, and arc lengths.",
              "- `runs/<case>/arrivals_and_peaks.csv`, `branch_speeds.csv`, `metadata.json`: per-run metrics and settings.",
              "- `arrival_speed_comparison.png`, `reference_probe_traces.png`: comparison figures.", "",
              "All configurations use the same physical parameters and stimulus. Different FEM degrees have different DOF counts; "
              "the comparison is at fixed element count and dt, not fixed computational cost.", "",
              "A smaller error at one setting can result from cancellation between spatial and temporal errors. "
              "Separate refinement studies are needed to compare convergence orders.", "",
              f"Recorded comparison cases: {len(summaries)}. See event statuses and reference sensitivities before interpreting small differences.", ""]
    (output/"README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdf-orders", type=int, nargs="+", choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--degrees", type=int, nargs="+", choices=range(1, 5), default=list(range(1, 5)))
    parser.add_argument("--elements", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--final-time", type=float, default=90)
    parser.add_argument("--quadrature", type=int, default=24)
    parser.add_argument("--reference-elements", type=int, default=200)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--space-check-elements", type=int, default=300)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[10, 20, 40, 50])
    parser.add_argument("--output", type=Path, default=PROPAGATION_RESULTS)
    parser.add_argument("--recompute", action="store_true", help="rerun even when a matching checkpoint exists")
    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))
    if 20 not in thresholds or not all(np.isfinite(thresholds)) or min(thresholds) <= 0:
        parser.error("thresholds must be positive and finite and include the primary threshold 20")
    if args.space_check_elements <= args.reference_elements:
        parser.error("space-check-elements must exceed reference-elements")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    reference_case = Case("reference", 5, 4, args.reference_elements, args.reference_dt,
                          args.final_time, args.quadrature)
    reference = run_case(reference_case, output, thresholds, args.recompute)
    checks = []
    variants = [Case("reference_space", 5, 4, args.space_check_elements, args.reference_dt,
                     args.final_time, args.quadrature),
                Case("reference_time", 5, 4, args.reference_elements, args.reference_dt/2,
                     args.final_time, args.quadrature),
                Case("reference_quadrature", 5, 4, args.reference_elements, args.reference_dt,
                     args.final_time, max(40, args.quadrature+8))]
    for variant in variants:
        check = run_case(variant, output, thresholds, args.recompute)
        report = reference_change(reference, check)
        checks.append(report)
        write_csv(output/"reference_checks.csv", checks)
        print(f"{variant.name} check: {report}", flush=True)
        del check
    sensitivity = dict(arrival_ms=sum(row["max_arrival_change_ms"] or 0 for row in checks),
                       speed_percent=sum(row["max_speed_change_percent"] or 0 for row in checks),
                       peak_voltage_mV=sum(row["max_peak_voltage_change_mV"] or 0 for row in checks))
    summaries, arrivals, speeds = [], [], []
    for order in sorted(set(args.bdf_orders)):
        for degree in sorted(set(args.degrees)):
            name = f"bdf{order}_p{degree}_n{args.elements}_dt{args.dt:g}"
            case = Case(name, order, degree, args.elements, args.dt, args.final_time, args.quadrature)
            run = run_case(case, output, thresholds, args.recompute)
            summary, events, velocity = summarize(run, reference)
            summaries.append(summary)
            arrivals.extend(events)
            speeds.extend(velocity)
            write_csv(output/"summary.csv", summaries)
            write_csv(output/"arrival_comparison.csv", arrivals)
            write_csv(output/"speed_comparison.csv", speeds)
            print(f"Completed {name}: arrivals={summary['arrived_probes']}/{summary['probe_count']}, "
                  f"max arrival difference={summary['max_arrival_error_ms']} ms, "
                  f"max speed difference={summary['max_speed_error_percent']}%", flush=True)
            del run
    verified = all(row["passed"] for row in checks)
    write_json(output/"study.json", dict(
        reference=asdict(reference_case), reference_verified=verified,
        reference_checks=checks, reference_sensitivity_estimate=sensitivity,
        reference_check_criteria=dict(max_arrival_change_ms=0.002,
                                      max_speed_change_percent=0.01, max_peak_voltage_change_mV=0.02),
        primary_threshold_mV=20, thresholds_mV=thresholds, after_ms=0.3,
        speed_probe_arc_lengths_cm=[0.25, 0.75], comparisons=summaries,
        source_sha256=numerical_source_hash(),
        software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__)))
    write_readme(output, reference, checks, sensitivity, summaries)
    plot_results(output, summaries, reference, sensitivity)
    if not verified:
        raise SystemExit("Reference sensitivity criteria failed; inspect reference_checks.csv and refine the reference.")
    print(f"HH propagation results saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
