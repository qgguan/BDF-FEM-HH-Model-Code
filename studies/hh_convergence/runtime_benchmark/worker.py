"""One fresh, single-threaded physical HH solve for the runtime benchmark."""

import os

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "1"

import argparse
import ctypes
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

import numpy as np
import scipy
from mesh import y_graph
import solver
from solver import IMEXBDFSolver, SimulationConfig, SimulationResult
from studies.hh_convergence.hh_convergence import (
    FieldCase, field_signature, gate_mask_fast_path, run_field_case, source_hash,
)
from studies.hh_convergence.field_comparison import compare_fields, ERROR_KEYS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def blas_threads():
    """Query the loaded OpenBLAS/MKL public thread-count functions on Windows."""
    libraries = []
    if os.name == "nt":
        symbols = ("scipy_openblas_get_num_threads", "scipy_openblas_get_num_threads64_",
                   "openblas_get_num_threads", "openblas_get_num_threads64_", "mkl_get_max_threads")
        for package in (np, scipy):
            directory = Path(package.__file__).parent.parent / (package.__name__ + ".libs")
            for path in sorted(directory.glob("*.dll")):
                library = ctypes.CDLL(str(path))
                for symbol in symbols:
                    try:
                        function = getattr(library, symbol)
                    except AttributeError:
                        continue
                    function.restype, function.argtypes = ctypes.c_int, []
                    count = function()
                    if count != 1:
                        raise RuntimeError(f"{package.__name__} reports {count} numerical threads")
                    libraries.append(dict(package=package.__name__, library=path.name,
                                          thread_query=symbol, threads=count))
                    break
    return libraries


def environment():
    cpu = platform.processor()
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return dict(cpu_model=cpu, logical_cpus=os.cpu_count(), system=platform.system(),
                os_release=platform.release(), os_version=platform.version(),
                architecture=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__,
                thread_environment={key: os.environ[key] for key in THREAD_KEYS},
                queried_libraries=blas_threads())


def fresh_solve(case, final_time, *, warmup=True):
    """The measured interval starts before mesh construction and ends at return."""
    original = solver.gate_mask
    solver.gate_mask = gate_mask_fast_path
    try:
        if warmup:
            warm_mesh = y_graph(20, 1, 24)
            warm_config = SimulationConfig(bdf_order=2, dt=.01, final_time=.5, sample_interval=.5)
            IMEXBDFSolver(warm_mesh, warm_config).run(progress=False, observer=None)
        config = SimulationConfig(bdf_order=case.order, dt=case.dt,
                                  final_time=final_time, sample_interval=final_time)
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        mesh = y_graph(case.elements, case.degree, case.quadrature)
        result = IMEXBDFSolver(mesh, config).run(progress=False, observer=None)
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
    finally:
        solver.gate_mask = original
    result.diagnostics["mask_evaluation"] = "exact identity fast path; original full-array regularization otherwise"
    timing = dict(solve_wall_seconds=wall_seconds, solve_cpu_seconds=cpu_seconds,
                  loop_wall_seconds=result.diagnostics["elapsed_seconds"])
    if min(timing.values()) <= 0 or timing["loop_wall_seconds"] > wall_seconds:
        raise RuntimeError("Invalid benchmark timing interval")
    return result, timing


def compare_checked(a, b):
    errors, _ = compare_fields(a, b, 8, final_only=True)
    check, _ = compare_fields(a, b, 16, final_only=True)
    differences = {key: abs(errors[key] - check[key]) for key in ERROR_KEYS}
    for key in ERROR_KEYS:
        if differences[key] > 1e-12 + 1e-7 * abs(errors[key]):
            raise RuntimeError(f"Error-integration check failed for {key}")
    return errors, differences


def execute(request):
    case = FieldCase(**request["field_case"])
    info = environment()
    started_at = datetime.now(timezone.utc).isoformat()
    result, timing = fresh_solve(case, request.get("final_time_ms", 46.))
    record = dict(status="completed", kind=request["kind"], case_id=request["case_id"],
                  process_id=os.getpid(),
                  repeat=request.get("repeat"), sequence=request.get("sequence"),
                  identity=request["identity"], started_at_utc=started_at,
                  finished_solve_at_utc=datetime.now(timezone.utc).isoformat(),
                  field_case=asdict(case), timing=timing, environment=info,
                  diagnostics=result.diagnostics, source_sha256=source_hash(),
                  fresh_solve=True, warmup_excluded=True, progress_disabled=True)
    if request["kind"] != "smoke":
        ref = FieldCase(5, 4, 1280, 1/1280, 24)
        reference = run_field_case(ref, 46., 46., PROJECT / request["reference_store"], read_only=True)
        errors, integration = compare_checked(result, reference)
        record.update(errors=errors, q8_q16_absolute_differences=integration)
        baseline_path = PROJECT / request["baseline_metadata"]
        baseline = SimulationResult.load(baseline_path.parent)
        if request["kind"] == "timing":
            np.testing.assert_allclose(result.voltage, baseline.voltage, rtol=1e-11, atol=1e-10)
            np.testing.assert_allclose(result.gates, baseline.gates, rtol=1e-11, atol=1e-12)
            record["historical_field_max_differences"] = dict(
                voltage=float(np.max(np.abs(result.voltage-baseline.voltage))),
                gates=float(np.max(np.abs(result.gates-baseline.gates))))
            for key, expected in request["historical_errors"].items():
                if not np.isclose(errors[key], expected, rtol=1e-7, atol=1e-12):
                    raise RuntimeError(f"New {key} differs from the matching saved result")
        else:
            changes, check = compare_checked(result, baseline)
            record.update(quadrature_changes=changes, quadrature_q8_q16_absolute_differences=check)
    if request.get("save_fields"):
        destination = Path(request["save_fields"])
        result.save(destination)
        write_json(destination / "field_study.json", field_signature(case, result.config.final_time,
                                                                    result.config.sample_interval))
        record["saved_field_directory"] = request["public_field_directory"]
    write_json(request["record_path"], record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        execute(request)
    except Exception as error:
        message = str(error).replace(str(PROJECT), "<project>").replace(str(Path.home()), "<user>")
        write_json(request["record_path"], dict(kind=request["kind"], case_id=request["case_id"],
                   identity=request["identity"], status="failed", error=message,
                   failed_at_utc=datetime.now(timezone.utc).isoformat()))
        raise SystemExit(message)


if __name__ == "__main__":
    main()
