"""Summarize fresh measurements, quadrature controls and HH work-precision curves."""

import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_records(plan, records):
    """Reject mixed identities, duplicate repeats and missing measurements."""
    expected = {(row["case"], row["repeat"]) for row in plan["schedule"]}
    actual = [(row["case_id"], row["repeat"]) for row in records]
    if len(actual) != 57 or set(actual) != expected or len(set(actual)) != 57:
        raise ValueError("A final benchmark requires exactly three distinct measurements of every case")
    if len({row["identity"] for row in records}) != 1:
        raise ValueError("Measurements use different benchmark identities")
    environments = {json.dumps(row["environment"], sort_keys=True) for row in records}
    if len(environments) != 1:
        raise ValueError("Hardware, software or numerical-thread settings changed")
    summaries = []
    for case in plan["cases"]:
        group = sorted((row for row in records if row["case_id"] == case["case"]), key=lambda r: r["repeat"])
        if any(row.get("status") != "completed" or not row.get("fresh_solve") for row in group):
            raise ValueError("Failed or cached solves cannot enter the timing summary")
        row = dict(case=case["case"], scheme=case["scheme"], bdf_order=case["bdf_order"],
                   degree=case["degree"], elements=case["elements_per_branch"], h=case["h_cm"],
                   dt=case["dt_ms"], dofs=case["dofs"], repeats=len(group))
        for key in ("solve_wall_seconds", "solve_cpu_seconds", "loop_wall_seconds"):
            values = [item["timing"][key] for item in group]
            row["mean_"+key] = statistics.fmean(values)
            row["median_"+key] = statistics.median(values)
            row["min_"+key], row["max_"+key] = min(values), max(values)
            for item, value in zip(group, values):
                row[f"repeat{item['repeat']}_{key}"] = value
        for key in group[0]["errors"]:
            values = [item["errors"][key] for item in group]
            if max(values)-min(values) > 1e-12 + 1e-7*max(abs(v) for v in values):
                raise ValueError(f"Repeated physical errors disagree: {case['case']} {key}")
            row[key] = values[0]
        row["wall_relative_span"] = (row["max_solve_wall_seconds"]-row["min_solve_wall_seconds"])/row["median_solve_wall_seconds"]
        row["repeatability_flag"] = row["wall_relative_span"] > .1
        row["max_historical_voltage_difference"] = max(x["historical_field_max_differences"]["voltage"] for x in group)
        row["max_historical_gate_difference"] = max(x["historical_field_max_differences"]["gates"] for x in group)
        row["gate_mask_steps"] = max(x["diagnostics"]["gate_mask_steps"] for x in group)
        row["voltage_cutoff_steps"] = max(x["diagnostics"]["voltage_cutoff_steps"] for x in group)
        summaries.append(row)
    return summaries


def plot_precision(rows, output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"BDF2/P1":"#156082", "BDF3/P2":"#e97132", "BDF4/P3":"#196b24", "BDF5/P4":"#0f9ed5"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    for ax, key, label in zip(axes, ("voltage_error", "gate_error"),
            ("Voltage L2 error (mV sqrt(cm))", "Maximum gate L2 error (sqrt(cm))")):
        for scheme, color in colors.items():
            group = [row for row in rows if row["scheme"] == scheme]
            x = [r["mean_solve_wall_seconds"] for r in group]
            y = [r[key] for r in group]
            ax.plot(x, y, color=color, linewidth=1.3, label=scheme)
            for row in group:
                t, error = row["mean_solve_wall_seconds"], row[key]
                quadrature_failed = (not row["reference_quadrature_pass"]
                                     or row["candidate_quadrature_status"] == "failed")
                ax.errorbar(t, error, xerr=[[t-row["min_solve_wall_seconds"]],
                    [row["max_solve_wall_seconds"]-t]], fmt="x" if quadrature_failed else "o", color=color,
                    markerfacecolor=color, markersize=5, capsize=2)
                last = row is group[-1]
                ax.annotate(str(row["elements"]), (t, error), xytext=(-5 if last else 4,9 if last else 4),
                    ha="right" if last else "left", textcoords="offset points", fontsize=7)
        ax.set(xscale="log", yscale="log", xlabel="Mean elapsed solve time (s)", ylabel=label)
        ax.grid(True, which="major", alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle("HH error versus mean runtime at 46 ms")
    caption = "Labels: N. Bars: min-max of three runs."
    if any(not row["reference_quadrature_pass"] or row["candidate_quadrature_status"] == "failed" for row in rows):
        caption += " Crosses: failed quadrature screen."
    fig.supxlabel(caption, fontsize=9)
    fig.savefig(output / "work_precision.png", dpi=600)
    fig.savefig(output / "work_precision.svg")
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(output / "work_precision.pdf", metadata={"Title": "HH error versus mean runtime at 46 ms"})
    plt.close(fig)


def analyze(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("status") != "completed" or not manifest.get("historical_inputs_unchanged"):
        raise ValueError("Final analysis requires a completed batch and verified historical inputs")
    plan = manifest["protocol"]
    timing_paths = sorted((output / "records").glob("timing_*.json"))
    records = [json.loads(path.read_text()) for path in timing_paths]
    if any(row.get("identity") != manifest["identity"] for row in records):
        raise ValueError("A measurement differs from the manifest identity")
    rows = summarize_records(plan, records)
    libraries = records[0]["environment"].get("queried_libraries", [])
    if ({library["package"] for library in libraries} != {"numpy", "scipy"}
            or any(library["threads"] != 1 for library in libraries)):
        raise ValueError("Direct single-thread verification is required for NumPy and SciPy")
    controls = [json.loads(path.read_text()) for path in sorted((output / "records").glob("quadrature_*.json"))]
    expected_controls = {r["case"] for r in plan["cases"] if r["quadrature_check_required"]} | {"REFERENCE"}
    expected_controls.update(row["case"] for row in rows if row["gate_mask_steps"])
    if (len(controls) != len(expected_controls)
            or {r["case_id"] for r in controls} != expected_controls
            or any(r.get("status") != "completed" for r in controls)):
        raise ValueError("Quadrature controls are incomplete")
    if any(r["identity"] != manifest["identity"] for r in controls):
        raise ValueError("A quadrature control uses different numerical settings")
    if any(r["environment"] != records[0]["environment"] for r in controls):
        raise ValueError("A quadrature control uses a different runtime environment")
    control_by_case = {r["case_id"]:r for r in controls}
    ref_quad = control_by_case["REFERENCE"]["quadrature_changes"]
    check_path = PROJECT / plan["reference"]["store"] / "reference_checks.csv"
    with check_path.open() as stream:
        reference_checks = list(csv.DictReader(stream))
    indicators = {key:sum(float(r[key]) for r in reference_checks) for key in ("voltage_error","gate_error")}
    qrows = []
    fraction = plan["quadrature"]["proposed_control_fraction_of_error"]
    for row in rows:
        v, g = row["voltage_error"], row["gate_error"]
        row["reference_voltage_indicator_percent"] = 100*indicators["voltage_error"]/v
        row["reference_gate_indicator_percent"] = 100*indicators["gate_error"]/g
        row["reference_indicator_flag"] = max(row["reference_voltage_indicator_percent"],row["reference_gate_indicator_percent"]) > 5
        row["reference_quadrature_voltage_percent"] = 100*ref_quad["voltage_error"]/v
        row["reference_quadrature_gate_percent"] = 100*ref_quad["gate_error"]/g
        row["reference_quadrature_pass"] = max(ref_quad["voltage_error"]/v,ref_quad["gate_error"]/g) <= fraction
        control = control_by_case.get(row["case"])
        row["candidate_quadrature_status"] = "not_directly_checked"
        if control:
            change = control["quadrature_changes"]
            vp, gp = 100*change["voltage_error"]/v,100*change["gate_error"]/g
            passed = max(vp,gp) <= 100*fraction
            row["candidate_quadrature_status"] = "passed" if passed else "failed"
            row["candidate_quadrature_voltage_percent"],row["candidate_quadrature_gate_percent"] = vp,gp
            qrows.append(dict(case=row["case"], voltage_change=change["voltage_error"],
                             gate_change=change["gate_error"], voltage_percent_of_error=vp,
                             gate_percent_of_error=gp, passed=passed))
    qrows.append(dict(case="REFERENCE", voltage_change=ref_quad["voltage_error"],
                      gate_change=ref_quad["gate_error"],
                      voltage_percent_of_error=max(r["reference_quadrature_voltage_percent"] for r in rows),
                      gate_percent_of_error=max(r["reference_quadrature_gate_percent"] for r in rows),
                      passed=all(r["reference_quadrature_pass"] for r in rows)))
    raw = []
    for record in records:
        raw.append(dict(sequence=record["sequence"], case=record["case_id"], repeat=record["repeat"],
                        process_id=record["process_id"], started_at_utc=record["started_at_utc"],
                        finished_solve_at_utc=record["finished_solve_at_utc"],
                        **record["timing"], **record["errors"]))
    write_csv(output / "timings.csv", raw)
    write_csv(output / "summary.csv", rows)
    write_csv(output / "quadrature_checks.csv", qrows)
    targets = []
    for tolerance in (1.,.1,.01,.001,.0001,.00001):
        for scheme in ("BDF2/P1","BDF3/P2","BDF4/P3","BDF5/P4"):
            matching = [r for r in rows if r["scheme"]==scheme and r["voltage_error"]<=tolerance]
            eligible = [r for r in matching if r["reference_quadrature_pass"]
                        and r["candidate_quadrature_status"] != "failed"]
            if not eligible:
                targets.append(dict(voltage_l2_target=tolerance, scheme=scheme,
                                    status="control_limited" if matching else "not_reached"))
                continue
            best = min(eligible,key=lambda r:r["mean_solve_wall_seconds"])
            targets.append(dict(voltage_l2_target=tolerance,scheme=scheme,status="measured",
                case=best["case"], voltage_error=best["voltage_error"],
                mean_solve_wall_seconds=best["mean_solve_wall_seconds"],
                median_solve_wall_seconds=best["median_solve_wall_seconds"],
                reference_indicator_flag=best["reference_indicator_flag"],
                repeatability_flag=best["repeatability_flag"],
                candidate_quadrature_status=best["candidate_quadrature_status"]))
    write_csv(output / "target_errors.csv",targets)
    summary = dict(status="completed",case_count=len(rows),measured_solves=len(records),
                   displayed_timing_statistic="arithmetic mean of three fresh repetitions",
                   quadrature_controls=len(controls),environment=records[0]["environment"],
                   analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_identity=manifest["identity"], reference=plan["reference"],
                   reference_neighbor_indicators=indicators, cases=rows,
                   quadrature_checks=qrows,target_errors=targets,
                   repeatability_flags=[r["case"] for r in rows if r["repeatability_flag"]],
                   quadrature_checks_passed=all(r["passed"] for r in qrows),
                   historical_inputs_unchanged=manifest.get("historical_inputs_unchanged",False))
    summary["verification"] = dict(
        distinct_measurement_process_ids=len({record["process_id"] for record in records}),
        measured_solve_intervals_do_not_overlap=all(
            a["finished_solve_at_utc"] <= b["started_at_utc"]
            for a,b in zip(sorted(records,key=lambda r:r["sequence"]),
                           sorted(records,key=lambda r:r["sequence"])[1:])),
        max_historical_voltage_difference=max(r["max_historical_voltage_difference"] for r in rows),
        max_historical_gate_difference=max(r["max_historical_gate_difference"] for r in rows),
        max_q8_q16_relative_difference=max(
            value/max(abs(record["errors"][key]),1e-300)
            for record in records for key,value in record["q8_q16_absolute_differences"].items()),
        all_numerical_libraries_single_threaded=all(
            library["threads"]==1 for record in records for library in record["environment"]["queried_libraries"]))
    by_case = {row["case"]:row for row in rows}
    summary["repetition_blocks"] = [dict(repeat=repeat,
        measured_cases=sum(record["repeat"]==repeat for record in records),
        total_solve_wall_seconds=sum(record["timing"]["solve_wall_seconds"] for record in records if record["repeat"]==repeat),
        median_wall_ratio_to_case_median=statistics.median(
            record["timing"]["solve_wall_seconds"]/by_case[record["case_id"]]["median_solve_wall_seconds"]
            for record in records if record["repeat"]==repeat)) for repeat in (1,2,3)]
    write_json(output / "summary.json",summary)
    plot_precision(rows,output)
    lines = ["# Controlled HH error-versus-runtime results", "",
        "Completed 57 fresh serial measurements (three for each of 19 candidates) and "
        f"{len(controls)} separate Q=40 controls through 46 ms. All measured solves use the optimized "
        "gate-mask path, Q=24 assembly and one verified numerical-library thread. No previous elapsed times enter this summary.","",
        "The primary elapsed/CPU intervals include mesh and solver setup, startup histories and time stepping. "
        "Imports, the fixed warmup, reference loading, field saving and error analysis are excluded. "
        "The separate loop timer retains the narrower solver-only definition.","",
        f"Environment: {summary['environment']['cpu_model']}; "
        f"{summary['environment']['system']} {summary['environment']['os_release']}; "
        f"Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}, "
        f"SciPy {summary['environment']['scipy']}. Both loaded numerical libraries reported one thread "
        "in every repetition. The manifest retains the unchanged power-scheme identifier.","",
        "## Measured cases", "",
        "Current plots and cost tables use the arithmetic mean of the three repetitions. "
        "The original measurement plan's medians remain in the CSV/JSON alongside the means. "
        "The repeatability diagnostic retains its original definition: elapsed span = (max-min)/median.", "",
        "| Scheme | N | Mean elapsed (s) | Mean CPU (s) | Voltage L2 error | Gate L2 error | Elapsed span (%) | Voltage reference indicator (%) | Gate reference indicator (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['scheme']} | {r['elements']} | {r['mean_solve_wall_seconds']:.3f} | "
            f"{r['mean_solve_cpu_seconds']:.3f} | {r['voltage_error']:.8g} | {r['gate_error']:.8g} | "
            f"{100*r['wall_relative_span']:.2f} | {r['reference_voltage_indicator_percent']:.4g} | "
            f"{r['reference_gate_indicator_percent']:.4g} |")
    lines += ["", "![L2 error versus elapsed time](work_precision.png)", "",
        "The reference is BDF5/P4 h=dt=1/1280. Its coarser-neighbor differences are empirical sensitivity "
        "indicators rather than bounds. The finest P3/P4 indicators are about 10.3% and 15.0% "
        "of their reported errors; these diagnostics remain in the tables. "
        "The 5% screening threshold is a chosen reporting convention, not a mathematical requirement. "
        "The reference itself is excluded from the candidate list. These are finite-reference comparisons along h=dt refinement paths.","",
        "## Example accuracy target", "",
        "For voltage L2 error at most 0.001 mV sqrt(cm), the fastest measured candidate of each scheme is:", "",
        "| Scheme | N | Voltage L2 error | Mean elapsed (s) | Status |",
        "| --- | ---: | ---: | ---: | --- |"]
    for target in (item for item in targets if item["voltage_l2_target"] == .001):
        if target["status"] != "measured":
            lines.append(f"| {target['scheme']} | — | — | — | {target['status']} |")
            continue
        case = by_case[target["case"]]
        notes = ["sampled quadrature policy"]
        if case["reference_indicator_flag"]:
            notes.append("reference indicator >5%")
        if case["repeatability_flag"]:
            notes.append("elapsed span >10%")
        lines.append(f"| {case['scheme']} | {case['elements']} | {case['voltage_error']:.8g} | "
                     f"{case['mean_solve_wall_seconds']:.3f} | {'; '.join(notes)} |")
    lines += ["", "These choices compare computed points only. They do not establish the cheapest possible "
        "combination of h and dt or performance with degree-dependent quadrature. Consult the full ranges "
        "and flags when interpreting timing differences.", "",
        "## Checks and interpretation", "",
        f"All sampled quadrature controls pass the 1% criterion: **{summary['quadrature_checks_passed']}**. "
        "Each scheme's finest candidate, any additional masked candidates, and the reference were checked at Q=40. "
        "Other candidates are explicitly marked not_directly_checked; a finest-grid check is not a pointwise certification of every coarser stimulus integral.","",
        "All repetitions independently evaluate Q=8 final-time continuous FEM errors on the union mesh, "
        "checked with Q=16, and verify the matching historical fields. Gate and voltage norms retain their distinct units.","",
        f"The largest Q=8/Q=16 relative norm difference is {summary['verification']['max_q8_q16_relative_difference']:.6g}. "
        f"Maximum historical snapshot differences are {summary['verification']['max_historical_voltage_difference']:.6g} "
        f"for voltage and {summary['verification']['max_historical_gate_difference']:.6g} for gates. "
        "The complete original input hash inventory is unchanged.","",
        "Cases with elapsed span above 10%: " + (", ".join(summary["repeatability_flags"]) or "none") + ". "
        "Every raw measurement is retained. This span is an observed range, not a confidence interval.","",
        "The median within-case timing ratios for repetition blocks 1/2/3 are "
        + "/".join(f"{block['median_wall_ratio_to_case_median']:.4f}" for block in summary["repetition_blocks"])
        + ". Each ratio divides a measurement by that candidate's three-run median. "
        "The software, thread and source checks are unchanged across blocks. CPU frequency and affinity were not locked or instrumented; "
        "the exact hardware or scheduling cause of residual timing variation is not identified. "
        "Use the complete ranges when comparing close costs; no slow measurements were removed.","",
        "`target_errors.csv` selects the fastest measured case of each scheme meeting each declared "
        "voltage-error target. It performs no extrapolation and keeps reference/repeatability/quadrature flags. "
        "Cases failing a quadrature screen are excluded from that target table's cost choices. "
        "A not_reached row means no tested case meets that target, not that the method cannot meet it.","",
        "## Files and reproduction", "",
        "- [timings.csv](timings.csv): all 57 fresh timing/error records.",
        "- [summary.csv](summary.csv), [summary.json](summary.json): means, medians, ranges, errors and flags.",
        "- [quadrature_checks.csv](quadrature_checks.csv): separate assembly-quadrature changes.",
        "- [target_errors.csv](target_errors.csv): measured costs at stated error targets.",
        "- [manifest.json](manifest.json): immutable source identities, environment policy and approved design.",
        "- `records/`: full per-repetition and control records; `fields/`: first-repetition fields; `quadrature_fields/`: Q=40 fields.",
        "- [work_precision.png](work_precision.png): 7800 by 3180 pixel scientific figure at 600 dpi.",
        "- [work_precision.svg](work_precision.svg): scalable vector scientific figure.","",
        "- [work_precision.pdf](work_precision.pdf): vector PDF with embedded TrueType fonts.","",
        "Run from python-bdf-fem to regenerate this analysis without new solves:","","```text",
        "python -B -m studies.hh_convergence.runtime_benchmark.run_benchmark --analyze-only", "```", "",
        "To resume the identical batch after interruption, use the same command with --resume instead. "
        "Source/settings changes are rejected to avoid mixing timing protocols. The original HH convergence run stores remain unchanged."]
    control_table = ["## Assembly-quadrature controls", "",
        "Each row compares Q=24 and Q=40 fields. Candidate percentages divide by that candidate's "
        "primary error; reference percentages are the largest ratios across the candidate set. "
        "Both percentages must be at most 1% to pass.", "",
        "| Control | Voltage L2 change | Gate L2 change | Voltage change / error (%) | Gate change / error (%) | Pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for check in qrows:
        control_table.append(f"| {check['case']} | {check['voltage_change']:.8g} | "
            f"{check['gate_change']:.8g} | {check['voltage_percent_of_error']:.6g} | "
            f"{check['gate_percent_of_error']:.6g} | {check['passed']} |")
    lines[lines.index("## Files and reproduction"):lines.index("## Files and reproduction")] = control_table + [""]
    if (output / "benchmark.xlsx").exists():
        lines.insert(lines.index("## Files and reproduction") + 2,
            "- [benchmark.xlsx](benchmark.xlsx): editable spreadsheet dashboard snapshot of this batch. "
            "Its entered timing means recalculate in the workbook; the Python analysis command regenerates the CSV/JSON/figures, not this spreadsheet.")
    (output / "README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary
