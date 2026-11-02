"""Physical HH field convergence for BDF1--5/P1--4; no manufactured forcing."""

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import scipy

from mesh import y_graph
from model import gate_mask as regularized_gate_mask
import solver as solver_module
from solver import IMEXBDFSolver, SimulationConfig, SimulationResult
from studies.paths import PROJECT_ROOT, HH_CONVERGENCE_RESULTS, relative_data_path
from studies.propagation.hh_propagation import write_csv, write_json
from studies.hh_convergence.field_comparison import compare_fields, ERROR_KEYS


@dataclass(frozen=True)
class FieldCase:
    order: int
    degree: int
    elements: int
    dt: float
    quadrature: int = 24

    @property
    def name(self):
        return f"bdf{self.order}_p{self.degree}_n{self.elements}_dt{self.dt:g}_q{self.quadrature}"


def source_hash():
    digest = hashlib.sha256()
    for name in ("mesh.py", "model.py", "solver.py"):
        digest.update((PROJECT_ROOT/name).read_bytes())
    return digest.hexdigest()


def write_result_readme(path, text, *, encoding="utf-8"):
    """Keep a curated saved-experiment guide when regenerating reports.

    Unmarked automatic READMEs continue to refresh on every analysis.
    Return whether the generated text was written.
    """
    path = Path(path)
    marker = "<!-- hh-convergence:curated-readme -->"
    if path.exists() and marker in path.read_text(encoding=encoding):
        return False
    path.write_text(text, encoding=encoding)
    return True


def field_signature(case, final_time, sample_interval):
    return dict(case=asdict(case), final_time=final_time, sample_interval=sample_interval,
                source_sha256=source_hash(), problem="physical HH with paper stimulus and exact resting histories")


def gate_mask_fast_path(values, epsilon=1e-3):
    """Exact identity shortcut; retain the original full-array mask otherwise.

    No saturation is evaluated when all values lie in [0,1]. If any value
    needs regularization, the original implementation receives the complete
    original array, preserving its floating-point evaluation as well.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    values = np.asarray(values, dtype=float)
    if np.all((values >= 0) & (values <= 1)):
        return values.copy()
    return regularized_gate_mask(values, epsilon)


def run_field_case(case, final_time, sample_interval, output, recompute=False, *, progress=False, read_only=False):
    """Save full fields, rather than the existing speed study's probe-only traces."""
    folder = Path(output)/"runs"/case.name
    signature = field_signature(case, final_time, sample_interval)
    marker = folder/"field_study.json"
    if marker.exists() and not recompute:
        cached = json.loads(marker.read_text(encoding="utf-8"))
        if cached == signature:
            result = SimulationResult.load(folder)
            expected = SimulationConfig(bdf_order=case.order, dt=case.dt,
                                        final_time=final_time, sample_interval=sample_interval)
            if (result.config != expected or result.mesh.degree != case.degree
                    or not np.all(result.mesh.elements_per_edge == case.elements)
                    or result.mesh.quadrature_order != case.quadrature):
                raise ValueError("cached field metadata does not match the requested case")
            steps, stride = round(final_time/case.dt), round(sample_interval/case.dt)
            sample_steps = list(range(0, steps+1, stride))
            if sample_steps[-1] != steps:
                sample_steps.append(steps)
            expected_times = np.array(sample_steps)*case.dt
            if (result.voltage.shape != (len(result.times), result.mesh.ndof)
                    or result.gates.shape != (len(result.times), 3, result.mesh.ndof)
                    or not np.all(np.isfinite(result.voltage)) or not np.all(np.isfinite(result.gates))
                    or result.times.shape != expected_times.shape
                    or not np.allclose(result.times, expected_times, rtol=0, atol=1e-10)):
                raise ValueError("cached field history is incomplete or nonfinite")
            return result
    if read_only:
        raise ValueError("saved reference history does not match; read-only reuse cannot run or overwrite it")
    config = SimulationConfig(bdf_order=case.order, dt=case.dt,
                              final_time=final_time, sample_interval=sample_interval)
    mesh = y_graph(case.elements, case.degree, case.quadrature)
    # No initial-field callbacks or added source terms: use the physical solver defaults.
    last_bucket = [-1]
    def observer(t, voltage, gates):
        bucket = min(10, int(10*t/final_time+1e-12))
        if bucket > last_bucket[0]:
            print(f"{case.name}: t={t:g}/{final_time:g} ms", flush=True)
            last_bucket[0] = bucket
    original_mask = solver_module.gate_mask
    solver_module.gate_mask = gate_mask_fast_path
    try:
        result = IMEXBDFSolver(mesh, config).run(observer=observer if progress else None)
    finally:
        solver_module.gate_mask = original_mask
    result.diagnostics["mask_evaluation"] = "exact identity fast path; original full-array regularization otherwise"
    result.save(folder)
    write_json(marker, signature)
    return result


def _worker(payload):
    case, final_time, sample_interval, output, force, progress = payload
    result = run_field_case(case, final_time, sample_interval, output, force, progress=progress)
    return case.name, result.diagnostics["elapsed_seconds"]


def compute_cases(cases, final_time, sample_interval, output, workers, force, progress=False):
    payloads = [(case, final_time, sample_interval, str(output), force, progress) for case in cases]
    if workers == 1:
        for i, payload in enumerate(payloads, 1):
            name, _ = _worker(payload)
            print(f"[{i}/{len(payloads)}] {name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, payload) for payload in payloads]
            for i, future in enumerate(as_completed(futures), 1):
                name, _ = future.result()
                print(f"[{i}/{len(payloads)}] {name}", flush=True)


def check_case(case, final_time, sample_interval):
    SimulationConfig(bdf_order=case.order, dt=case.dt, final_time=final_time, sample_interval=sample_interval)
    if (case.order-1)*case.dt > 0.1+1e-14:
        raise ValueError("resting histories require (BDF order-1)*dt <= 0.1")
    if int(case.elements) != case.elements or case.elements < 1:
        raise ValueError("elements must be a positive integer")
    if case.degree not in range(1, 5) or case.quadrature < 3*case.degree+1:
        raise ValueError("invalid degree or insufficient assembly quadrature")


def saved_reference_cases(folder, final_time, sample_interval):
    """Read and validate a checked P4 reference store without modifying it."""
    folder = Path(folder)
    plan = json.loads((folder/"design.json").read_text(encoding="utf-8"))
    roles = ("base", "space", "time", "quadrature")
    try:
        names = plan["global_reference"]
        if set(names) != set(roles):
            raise ValueError("saved reference needs base, space, time and quadrature histories")
        result = {role: FieldCase(**plan["cases"][names[role]]) for role in roles}
    except (KeyError, TypeError) as error:
        raise ValueError("invalid saved full-field reference design") from error
    for role, case in result.items():
        if case.name != names[role]:
            raise ValueError("saved reference case name does not match its settings")
        run_field_case(case, final_time, sample_interval, folder, read_only=True)
    return result


def validate_reference_cases(refs):
    if set(refs) != {"base", "space", "time", "quadrature"}:
        raise ValueError("reference roles must be base, space, time and quadrature")
    base, space, time, quad = (refs[k] for k in ("base", "space", "time", "quadrature"))
    if (base.order, base.degree) != (5, 4):
        raise ValueError("saved PDE reference must use BDF5/P4")
    if space.elements <= base.elements or space != replace(base, elements=space.elements):
        raise ValueError("reference space check must refine only the mesh")
    if time.dt >= base.dt or time != replace(base, dt=time.dt):
        raise ValueError("reference time check must refine only dt")
    if quad.quadrature <= base.quadrature or quad != replace(base, quadrature=quad.quadrature):
        raise ValueError("reference quadrature check must increase only the integration order")


def build_design(orders, degrees, study, levels, elements, joint_dt, time_elements,
                 time_dt, space_dt, quadrature, final_time, sample_interval,
                 reference_elements=None, reference_dt=None, space_check_elements=None,
                 reference_cases=None):
    if levels < 3 or not orders or not degrees:
        raise ValueError("at least three levels and nonempty method selections are required")
    if any(k not in range(1, 6) for k in orders) or any(p not in range(1, 5) for p in degrees):
        raise ValueError("BDF orders must be 1--5 and FEM degrees 1--4")
    if final_time <= 0.3:
        raise ValueError("final_time must extend beyond the physical stimulus ending at 0.3 ms")
    series, cases = [], {}
    kinds = ("time", "space", "joint") if study == "all" else (study,)
    if any(kind not in ("time", "space", "joint") for kind in kinds):
        raise ValueError("unknown refinement study")
    def add(case):
        check_case(case, final_time, sample_interval)
        if case.name in cases and case != cases[case.name]:
            raise ValueError("formatted case names collide; choose distinguishable time steps")
        cases[case.name] = case
        return case.name
    for kind in kinds:
        for order in sorted(set(orders)):
            for degree in sorted(set(degrees)):
                names = []
                for level in range(levels):
                    n = time_elements if kind == "time" else elements*2**level
                    dt = space_dt if kind == "space" else (time_dt if kind == "time" else joint_dt)/2**level
                    names.append(add(FieldCase(order, degree, n, dt, quadrature)))
                series.append(dict(study=kind, bdf_order=order, degree=degree, cases=names))
    global_refs, time_refs, controls = {}, {}, {}
    global_cases = [cases[name] for s in series if s["study"] != "time" for name in s["cases"]]
    if global_cases:
        if reference_cases is not None:
            validate_reference_cases(reference_cases)
            global_refs = {role: add(case) for role, case in reference_cases.items()}
        else:
            finest_n = max(c.elements for c in global_cases)
            finest_dt = min(c.dt for c in global_cases)
            nr = reference_elements if reference_elements is not None else 2*finest_n
            dr = reference_dt if reference_dt is not None else finest_dt/4
            ns = space_check_elements if space_check_elements is not None else 3*nr//2
            if nr <= finest_n or dr >= finest_dt or ns <= nr:
                raise ValueError("global reference must refine both h and dt, and its spatial check must be finer again")
            global_refs = dict(base=add(FieldCase(5, 4, nr, dr, quadrature)),
                               space=add(FieldCase(5, 4, ns, dr, quadrature)),
                               time=add(FieldCase(5, 4, nr, dr/2, quadrature)),
                               quadrature=add(FieldCase(5, 4, nr, dr, max(40, quadrature+8))))
    for s in series:
        fine = cases[s["cases"][-1]]
        key = f"{s['study']}_bdf{s['bdf_order']}_p{s['degree']}"
        if s["study"] == "time":
            degree = s["degree"]
            if degree not in time_refs:
                dr = reference_dt if reference_dt is not None else fine.dt/4
                if dr >= fine.dt:
                    raise ValueError("temporal reference step must be smaller than the finest tested step")
                time_refs[degree] = dict(base=add(FieldCase(5, degree, time_elements, dr, quadrature)),
                                         time=add(FieldCase(5, degree, time_elements, dr/2, quadrature)))
        else:
            controls[key] = dict(quadrature=add(FieldCase(fine.order, fine.degree, fine.elements,
                                                         fine.dt, max(40, quadrature+8))))
            if s["study"] == "space":
                controls[key]["time"] = add(FieldCase(fine.order, fine.degree, fine.elements, fine.dt/2, quadrature))
    return series, cases, global_refs, time_refs, controls


def observed_rate(coarse, fine, ratio=2):
    if (not np.isfinite(coarse) or not np.isfinite(fine) or min(coarse, fine) <= 1e-13):
        return None
    return float(np.log(coarse/fine)/np.log(ratio))


def assess_series(rows, uncertainty, control_changes, expected, fraction=0.05):
    """Assess finite-range slopes only where reference and controls resolve the errors."""
    fine = rows[-1]
    keys = ("voltage_error", "gate_error")
    resolved = all(fine[k] > 1e-13 and uncertainty[k] <= fraction*fine[k] for k in keys)
    controls = {label: all(change[k] <= fraction*fine[k] for k in keys)
                for label, change in control_changes.items()}
    rate_met = all(row[k.replace("_error", "_rate")] is not None
                   and abs(row[k.replace("_error", "_rate")]-expected) <= 0.65
                   for row in rows[-2:] for k in keys)
    if not resolved:
        status = "reference_or_roundoff_limited"
    elif not all(controls.values()):
        status = "control_limited"
    elif not rate_met:
        status = "preasymptotic_or_mixed"
    else:
        status = "supported"
    return dict(reference_resolved=bool(resolved), controls_passed=controls,
                rate_band_met=bool(rate_met), interpretation=status)


def plot_rates(summaries, output, *, evaluation=""):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = list(dict.fromkeys(row["study"] for row in summaries))
    fig, axes = plt.subplots(1, len(kinds), figsize=(5*len(kinds), 4.8), squeeze=False, layout="constrained")
    for ax, kind in zip(axes.ravel(), kinds):
        values = np.full((5, 4), np.nan)
        for row in summaries:
            if row["study"] == kind and row["voltage_rate"] is not None:
                values[row["bdf_order"]-1, row["degree"]-1] = row["voltage_rate"]
                marker = "" if row["interpretation"] == "supported" else "*"
                ax.text(row["degree"]-1, row["bdf_order"]-1, f"{row['voltage_rate']:.2f}{marker}",
                        ha="center", va="center", color="white" if row["voltage_rate"] > 2.75 else "black", fontsize=9)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=5, aspect="auto")
        ax.set(xticks=range(4), xticklabels=[f"P{p}" for p in range(1, 5)],
               yticks=range(5), yticklabels=[f"BDF{k}" for k in range(1, 6)], title=kind.capitalize()+" refinement")
    fig.colorbar(image, ax=list(axes.ravel()), label="Last observed voltage L2 rate")
    fig.suptitle("Physical HH field convergence"+(" -- "+evaluation if evaluation else "")
                 +"\n* Reference/control check or slope criterion not satisfied")
    fig.savefig(output/"rates.png", dpi=170)
    plt.close(fig)


def plot_error_curves(rows, info, output, *, evaluation):
    """Show the actual refinement errors alongside the nominal power law."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    xkey = "dt" if info["study"] == "time" else "h"
    x = np.array([row[xkey] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    for ax, components, ylabel in (
            (axes[0], [("voltage_error", "Voltage")], r"Voltage $L^2$ error (mV $\sqrt{\mathrm{cm}}$)"),
            (axes[1], [(f"{gate}_error", gate) for gate in ("m", "h", "n")],
             r"Gate $L^2$ error ($\sqrt{\mathrm{cm}}$)")):
        for key, label in components:
            values = np.array([row[key] for row in rows])
            ax.loglog(x, values, "o-", label=label)
        anchor = rows[-1]["voltage_error" if ax is axes[0] else "gate_error"]
        if anchor > 0:
            ax.loglog(x, anchor*(x/x[-1])**info["expected_l2_rate"], "--", color="gray",
                      label=f"Order {info['expected_l2_rate']} guide")
        ax.set(xlabel="dt (ms)" if xkey == "dt" else "h (cm)", ylabel=ylabel)
        ax.set_xticks(x, labels=[f"{value:g}" for value in x])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.2)
        ax.legend()
    for row in rows[1:]:
        if row["voltage_rate"] is not None:
            axes[0].annotate(f"p={row['voltage_rate']:.2f}", (row[xkey], row["voltage_error"]),
                             xytext=(-4, 8) if row is rows[-1] else (0, 8), textcoords="offset points",
                             ha="right" if row is rows[-1] else "center", fontsize=9)
    fig.suptitle(f"BDF{info['bdf_order']}/P{info['degree']}: physical HH {evaluation}")
    fig.savefig(output/"error_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("joint", "time", "space", "all"), default="joint")
    parser.add_argument("--bdf-orders", type=int, nargs="+", choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument("--degrees", type=int, nargs="+", choices=range(1, 5), default=list(range(1, 5)))
    parser.add_argument("--levels", type=int, choices=range(3, 9), default=3)
    parser.add_argument("--elements", type=int, default=50)
    parser.add_argument("--joint-dt", type=float, default=0.02)
    parser.add_argument("--time-elements", type=int, default=100)
    parser.add_argument("--time-dt", type=float, default=0.02)
    parser.add_argument("--space-dt", type=float, default=0.00125)
    parser.add_argument("--quadrature", type=int, default=24)
    parser.add_argument("--comparison-quadrature", type=int, default=8)
    parser.add_argument("--final-time", type=float, default=90)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--error-time", choices=("max", "final"), default="max",
                        help="sampled maximum in time, or the L2 error at final time only")
    parser.add_argument("--reference-elements", type=int)
    parser.add_argument("--reference-folder", type=Path,
                        help="reuse an existing checked full-field P4 reference store for a space/joint study")
    parser.add_argument("--reference-dt", type=float)
    parser.add_argument("--space-check-elements", type=int)
    parser.add_argument("--sensitivity-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=min(3, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=HH_CONVERGENCE_RESULTS)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--analyze-only", action="store_true", help="rebuild reports from matching full-field histories without simulations")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress", action="store_true", help="show per-case physical time during simulations")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any series lacks supported rates")
    args = parser.parse_args()
    if not np.isfinite(args.sensitivity_fraction) or not 0 < args.sensitivity_fraction < 1:
        parser.error("sensitivity-fraction must be between zero and one")
    if args.comparison_quadrature < 5:
        parser.error("comparison-quadrature must be at least 5 for P4 comparisons")
    try:
        if args.reference_folder is not None:
            if args.study not in ("space", "joint"):
                parser.error("reference-folder reuse supports space or joint studies; temporal references must match each FEM space")
            if any(v is not None for v in (args.reference_elements, args.reference_dt, args.space_check_elements)):
                parser.error("reference-folder cannot be combined with reference-resolution overrides")
            reused_refs = saved_reference_cases(args.reference_folder, args.final_time, args.sample_interval)
        else:
            reused_refs = None
        series, cases, global_refs, time_refs, controls = build_design(
            args.bdf_orders, args.degrees, args.study, args.levels, args.elements,
            args.joint_dt, args.time_elements, args.time_dt, args.space_dt,
            args.quadrature, args.final_time, args.sample_interval,
            args.reference_elements, args.reference_dt, args.space_check_elements, reused_refs)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    analysis_hash = hashlib.sha256(Path(__file__).read_bytes()
                                  + Path(__file__).with_name("field_comparison.py").read_bytes()).hexdigest()
    external_names = set(global_refs.values()) if reused_refs is not None else set()
    def case_store(name):
        return args.reference_folder if name in external_names else output
    design = dict(problem="physical HH; no manufactured fields or additional source terms",
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "output"},
                  source_sha256=source_hash(), analysis_sha256=analysis_hash,
                  series=series, cases={k: asdict(v) for k, v in cases.items()},
                  global_reference=global_refs, temporal_references=time_refs, controls=controls,
                  case_stores={name: relative_data_path(case_store(name), output) for name in cases},
                  error_time=args.error_time,
                  norms=("final-time continuous graph L2" if args.error_time == "final" else
                         "maximum over common snapshot times of continuous graph L2")+"; gates reported separately and by maximum",
                  comparison="union of both FE meshes; exact polynomial-product Gauss integration; no time interpolation",
                  software=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))
    if args.analyze_only:
        if args.recompute:
            parser.error("analyze-only cannot be combined with recompute")
        missing = []
        for case in cases.values():
            folder = case_store(case.name)/"runs"/case.name
            marker = folder/"field_study.json"
            if (not marker.exists() or not (folder/"solution.npz").exists()
                    or not (folder/"metadata.json").exists()
                    or json.loads(marker.read_text(encoding="utf-8")) != field_signature(case, args.final_time, args.sample_interval)):
                missing.append(case.name)
        if missing:
            parser.error(f"{len(missing)} cases lack matching full-field histories; run without analyze-only")
    write_json(output/"design.json", design)
    print(f"Physical HH: {len(series)} series, {len(cases)} distinct field runs, T={args.final_time:g} ms", flush=True)
    if not args.analyze_only:
        compute_cases([case for name, case in cases.items() if name not in external_names],
                      args.final_time, args.sample_interval, output,
                      args.workers, args.recompute, args.progress)
    def compare(a, b):
        return compare_fields(a, b, args.comparison_quadrature, final_only=args.error_time == "final")
    def load(name):
        return run_field_case(cases[name], args.final_time, args.sample_interval,
                              case_store(name), read_only=name in external_names)
    checks, summaries, details = [], [], []
    reference_cache = {}
    for s in series:
        kind, order, degree = s["study"], s["bdf_order"], s["degree"]
        key = f"{kind}_bdf{order}_p{degree}"
        refs = time_refs[degree] if kind == "time" else global_refs
        ref_name = refs["base"]
        if ref_name not in reference_cache:
            reference = load(ref_name)
            if reference.diagnostics["voltage_nodal_range"][1] <= 1:
                raise RuntimeError("reference did not excite a nontrivial physical pulse")
            uncertainty = {k: 0.0 for k in ERROR_KEYS}
            for label, name in refs.items():
                if label == "base":
                    continue
                difference, _ = compare(load(name), reference)
                for metric in ERROR_KEYS:
                    uncertainty[metric] += difference[metric]
                checks.append(dict(reference=ref_name, check=label, case=name, **difference))
            reference_cache[ref_name] = reference, uncertainty
        reference, uncertainty = reference_cache[ref_name]
        rows = []
        for level, name in enumerate(s["cases"]):
            result = load(name)
            errors, history = compare(result, reference)
            row = dict(study=kind, bdf_order=order, degree=degree, level=level,
                       case=name, elements=cases[name].elements, h=result.mesh.h,
                       dt=cases[name].dt, dofs=result.mesh.ndof, **errors)
            for metric in ERROR_KEYS:
                row[metric.replace("_error", "_rate")] = (None if not rows else observed_rate(rows[-1][metric], errors[metric]))
            rows.append(row)
            folder = output/"errors"/key
            folder.mkdir(parents=True, exist_ok=True)
            write_csv(folder/f"level_{level}.csv", history)
        fine = load(s["cases"][-1])
        changes = {}
        for label, name in controls.get(key, {}).items():
            changes[label], _ = compare(load(name), fine)
        expected = order if kind == "time" else degree+1 if kind == "space" else min(order, degree+1)
        assessment = assess_series(rows, uncertainty, changes, expected, args.sensitivity_fraction)
        info = dict(study=kind, bdf_order=order, degree=degree, reference=ref_name,
                    error_time=args.error_time,
                    reference_type="matched-mesh semidiscrete" if kind == "time" else "checked P4 PDE reference",
                    reference_sensitivity_estimate=uncertainty, finest_control_changes=changes,
                    sensitivity_fraction=args.sensitivity_fraction, expected_l2_rate=expected,
                    expected_h1_rate=order if kind == "time" else degree if kind == "space" else min(order, degree),
                    **assessment)
        write_csv(output/(key+".csv"), rows)
        write_json(output/(key+".json"), info)
        summaries.append(dict(study=kind, bdf_order=order, degree=degree, expected_l2_rate=expected,
                              voltage_rate=rows[-1]["voltage_rate"], gate_rate=rows[-1]["gate_rate"],
                              voltage_h1_rate=rows[-1]["voltage_h1_rate"],
                              fine_voltage_error=rows[-1]["voltage_error"], fine_gate_error=rows[-1]["gate_error"],
                              reference_resolved=assessment["reference_resolved"], interpretation=assessment["interpretation"]))
        details.append(info)
        write_csv(output/"summary.csv", summaries)
        print(f"{key}: V rate={rows[-1]['voltage_rate']}, gates={rows[-1]['gate_rate']}; {assessment['interpretation']}", flush=True)
    write_csv(output/"reference_checks.csv", checks)
    interpretations = ("supported", "reference_or_roundoff_limited", "control_limited", "preasymptotic_or_mixed")
    counts = {kind: {status: sum(row["study"] == kind and row["interpretation"] == status for row in summaries)
                      for status in interpretations}
              for kind in dict.fromkeys(row["study"] for row in summaries)}
    write_json(output/"summary.json", dict(design, studies=details,
               interpretation_counts=counts,
               all_rates_supported=all(r["interpretation"] == "supported" for r in summaries),
               note="Finite-reference sensitivities are empirical, not rigorous error bounds. Sampled maxima and finite-range slopes do not prove formal orders."))
    if not args.no_plots:
        evaluation = (f"at T={args.final_time:g} ms" if args.error_time == "final" else
                      f"sampled maximum through T={args.final_time:g} ms")
        plot_rates(summaries, output, evaluation=evaluation)
        if len(series) == 1:
            plot_error_curves(rows, info, output, evaluation=evaluation)
    readme_written = write_result_readme(output/"README.md",
        "# Physical HH field convergence results\n\n"
        f"{len(series)} refinement series through T={args.final_time:g} ms, sampled every {args.sample_interval:g} ms. "
        +(f"Errors are measured only at final time T={args.final_time:g} ms. " if args.error_time == "final" else
          "Errors are maxima over common snapshot times. ")+
        "The Y-graph HH equations use the physical stimulus and resting histories, without manufactured forcing.\n\n"
        "`summary.csv` contains observed rates and interpretation flags. Per-series CSV/JSON files retain all levels, "
        "reference sensitivities and finest-level control checks. `reference_checks.csv` records field changes under reference refinement. "
        "`errors/` contains per-time norms; `runs/` contains full voltage/gate snapshots and source fingerprints. "
        "`design.json` records every configuration and relative case-store location.\n\n"
        "`error_curves.png` shows the errors and nominal rate guide for single-series runs.\n\n"
        "Time studies compare with BDF5 on the same FEM mesh/degree and quadrature. Space/joint studies compare with a finer P4 reference. "
        "Errors integrate FE differences on the union of both meshes and use common physical snapshot times. "
        "Reference uncertainty is estimated by adding the separate measured refinement changes; it is not a rigorous bound. "
        "A starred plot entry or a non-supported interpretation needs reference/control refinement or a different asymptotic range.\n",
        encoding="utf-8")
    if readme_written:
        with (output/"README.md").open("a", encoding="utf-8") as stream:
            stream.write("\n## L2 rate assessments\n\n"
                         "Supported rates meet both sensitivity checks and a 0.65 band around the nominal order at both last intervals.\n\n"
                         "| Mode | Supported | Reference/roundoff limited | Control limited | Preasymptotic/mixed |\n"
                         "| --- | ---: | ---: | ---: | ---: |\n")
            for kind, values in counts.items():
                stream.write("| "+kind+" | "+" | ".join(str(values[status]) for status in interpretations)+" |\n")
    if args.strict and any(s["interpretation"] != "supported" for s in summaries):
        raise SystemExit("Some rates are unresolved or outside the expected range; inspect summary.json and refine the indicated controls.")


if __name__ == "__main__":
    main()
