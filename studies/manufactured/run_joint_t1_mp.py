"""Run the cosine BDF5/P4 h=dt study using 50/80-digit arithmetic.

From python-bdf-fem: python -m studies.manufactured.run_joint_t1_mp
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import mpmath
from mpmath import mp

from studies.manufactured import multiprecision
from studies.manufactured.multiprecision import ERROR_KEYS, run
from studies.paths import MANUFACTURED_RESULTS


def decimal_data(value, ctx):
    if hasattr(value, "_mpf_"):
        return ctx.nstr(value, ctx.dps)
    if isinstance(value, dict):
        return {key: decimal_data(item, ctx) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [decimal_data(item, ctx) for item in value]
    return value


def save(directory, name, rows, metadata, ctx):
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(decimal_data(rows, ctx))
    (directory / f"{name}.json").write_text(
        json.dumps(decimal_data(metadata, ctx), indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", nargs="+", type=int, default=[40, 80, 160, 320])
    parser.add_argument("--dps", nargs="+", type=int, default=[50, 80])
    parser.add_argument("--final-time", default="1")
    parser.add_argument("--sample-interval", default="0.05")
    parser.add_argument("--quadrature", type=int, default=24)
    parser.add_argument("--error-quadrature", type=int, default=40)
    parser.add_argument("--full-graph", action="store_true", help="disable the exact half-branch symmetry reduction")
    parser.add_argument("--output", type=Path, default=MANUFACTURED_RESULTS)
    args = parser.parse_args()
    elements, precisions = sorted(set(args.elements)), sorted(set(args.dps))
    args.output.mkdir(parents=True, exist_ok=True)
    tag = args.final_time.replace(".", "p")
    base_name = f"joint_t{tag}_bdf5_p4_mp"
    source_hash = hashlib.sha256(Path(multiprecision.__file__).read_bytes()).hexdigest()
    reports = {}
    for digits in precisions:
        rows, details = [], []
        ctx = mp.clone()
        ctx.dps = digits
        name = f"{base_name}{digits}"
        info = dict(complete=False, passed=False, digits=digits, bdf_order=5, degree=4,
                    profile="cosine", final_time=args.final_time,
                    sample_interval=args.sample_interval, assembly_quadrature=args.quadrature,
                    error_quadrature=args.error_quadrature, symmetry_reduced=not args.full_graph,
                    symmetry_note="Three identical unit branches, each reflection symmetric, "
                                  "are represented by [0,1/2] with graph squared norms multiplied by six.",
                    arithmetic="mpmath throughout: rational BDF coefficients, exact unit metric lengths, "
                               "multiprecision Lobatto nodes, quadrature, Ritz initialization, "
                               "FEM assembly, sparse LDL solve, gates, sources and norms",
                    error_definition="Maximum over stored times. Voltage/gate errors are continuous "
                                     "graph L2; gate_nodal is the Lobatto norm; voltage_h1 is the "
                                     "voltage derivative L2 error. All mp values are decimal strings.",
                    source_sha256=source_hash,
                    environment=dict(python=sys.version, mpmath=mpmath.__version__, platform=platform.platform()),
                    runs=details)
        for n in elements:
            print(f"Running {name}: N={n}, h=dt=1/{n}, T={args.final_time}", flush=True)
            result = run(n, digits=digits, final_time=args.final_time,
                         sample_interval=args.sample_interval, quadrature=args.quadrature,
                         error_quadrature=args.error_quadrature, reduced=not args.full_graph, progress=True)
            row = dict(elements=n, h=result.mesh.h, dt=result.diagnostics["dt"],
                       dofs=result.diagnostics["full_graph_dofs"], solved_dofs=result.mesh.ndof,
                       digits=digits, **result.errors)
            for key in ERROR_KEYS:
                row[key.replace("error", "rate")] = (None if not rows else
                    ctx.log(rows[-1][key]/row[key])/ctx.log(rows[-1]["h"]/row["h"]))
            rows.append(row)
            details.append(dict(**result.diagnostics, errors=result.errors, sample_errors=result.sample_errors))
            save(args.output, name, rows, info, ctx)
            print(f"  V={ctx.nstr(row['voltage_error'], 10)}, gates={ctx.nstr(row['gate_error'], 10)}; "
                  f"constant projection={ctx.nstr(result.diagnostics['constant_projection_nodal_error'], 5)}", flush=True)
        available = rows[1:][-2:]
        info["complete"] = True
        info["passed"] = bool(available) and all(row[key] > ctx.mpf("4.35")
                                                for row in available for key in ("voltage_rate", "gate_rate"))
        save(args.output, name, rows, info, ctx)
        reports[digits] = {row["elements"]: decimal_data(row, ctx) for row in rows}
        print(f"Saved {name}.csv; fifth-order check passed={info['passed']}", flush=True)

    ctx = mp.clone()
    ctx.dps = max(precisions)
    comparisons = []
    if len(precisions) > 1:
        low, high = precisions[0], precisions[-1]
        for n in elements:
            relative = {key: abs(ctx.mpf(reports[low][n][key])-ctx.mpf(reports[high][n][key])) /
                        max(abs(ctx.mpf(reports[high][n][key])), ctx.mpf("1e-1000")) for key in ERROR_KEYS}
            comparisons.append(dict(elements=n, lower_digits=low, higher_digits=high,
                                    relative_error_changes=relative,
                                    agrees_to_20_relative_digits=max(relative.values()) < ctx.mpf("1e-20")))
    comparison = dict(precisions=precisions, elements=elements, final_time=args.final_time,
                      comparisons=comparisons, results=reports)
    (args.output / f"{base_name}_comparison.json").write_text(
        json.dumps(decimal_data(comparison, ctx), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Multiprecision results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
