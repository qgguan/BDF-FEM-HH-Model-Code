"""Run/resume the controlled HH benchmark serially, with one fresh process per solve."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def batch_lock():
    """An OS-held lock is released automatically if the controller exits."""
    path = HERE / ".benchmark.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another HH runtime benchmark is already running") from error
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def power_scheme():
    if os.name != "nt":
        return None
    result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
    match = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", result.stdout)
    return match.group(0).lower() if match else None


def validate_plan(plan):
    cases = {case["case"]: case for case in plan["cases"]}
    if len(cases) != 19 or len(plan["schedule"]) != 57:
        raise ValueError("Expected the approved 19-case, 57-measurement design")
    if [item["sequence"] for item in plan["schedule"]] != list(range(1, 58)):
        raise ValueError("Invalid measurement ordering")
    if Counter(item["case"] for item in plan["schedule"]) != Counter({key: 3 for key in cases}):
        raise ValueError("Each candidate must have three repetitions")
    for case in cases.values():
        if case["assembly_quadrature"] != 24 or case["dofs"] != 3*case["elements_per_branch"]*case["degree"]+1:
            raise ValueError("Changed approved numerical settings")
        if sha(PROJECT / case["historical_metadata"]) != case["metadata_sha256"]:
            raise ValueError("A historical metadata input changed")
    if sha(PROJECT / "studies/hh_convergence/hh_convergence.py") != plan["optimized_mask_module_sha256"]:
        raise ValueError("The approved optimized-mask implementation changed")
    return cases


def input_inventory():
    store = PROJECT / "studies/hh_convergence/results"
    return {path.relative_to(store).as_posix(): sha(path)
            for path in sorted(store.rglob("*")) if path.is_file()}


def run_one(request, output, name, identity, expected_power):
    if power_scheme() != expected_power:
        raise RuntimeError("Power scheme changed during the controlled batch")
    request_path = output / "requests" / (name + ".json")
    record = output / "records" / (name + ".json")
    if record.exists():
        saved = json.loads(record.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise RuntimeError("Saved measurement has a different benchmark identity")
        if saved.get("status") == "completed":
            return saved, True
        raise RuntimeError(f"An earlier attempt failed: {record.name}; inspect it before rerunning")
    request.update(identity=identity, record_path=record.relative_to(PROJECT).as_posix())
    if request.get("save_fields"):
        request["public_field_directory"] = request["save_fields"]
    write_json(request_path, request)
    write_json(output / "progress.json", dict(status="running", active=name,
               kind=request["kind"], case=request["case_id"], repeat=request.get("repeat"),
               started_at_utc=datetime.now(timezone.utc).isoformat()))
    env = dict(os.environ, **{key: "1" for key in THREAD_KEYS}, PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.run([sys.executable, "-B", str(HERE / "worker.py"), "--request",
                            str(request_path.relative_to(PROJECT))], cwd=PROJECT, env=env,
                           capture_output=True, text=True)
    if child.returncode:
        message = child.stderr.replace(str(PROJECT), "<project>").replace(str(Path.home()), "<user>")
        (output / (name + ".stderr.txt")).write_text(message, encoding="utf-8")
        raise RuntimeError(f"Worker {name} failed; inspect its record and error log")
    saved = json.loads(record.read_text(encoding="utf-8"))
    if saved.get("status") != "completed" or not saved.get("fresh_solve"):
        raise RuntimeError("Worker did not record a fresh completed solve")
    return saved, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/controlled_20261028")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE):
        parser.error("benchmark results must stay in their own benchmark folder")
    if args.analyze_only:
        from studies.hh_convergence.runtime_benchmark.analyze import analyze
        analyze(output)
        return
    plan_path = HERE / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cases = validate_plan(plan)
    numerical_hash = hashlib.sha256(b"".join((PROJECT / name).read_bytes()
                                   for name in ("mesh.py", "model.py", "solver.py"))).hexdigest()
    if any(case["numerical_source_sha256"] != numerical_hash for case in cases.values()):
        parser.error("The numerical source differs from the approved saved-field inputs")
    identity_info = dict(plan_sha256=sha(plan_path), worker_sha256=sha(HERE / "worker.py"),
                         numerical_source_sha256=numerical_hash,
                         mask_module_sha256=plan["optimized_mask_module_sha256"])
    identity = hashlib.sha256(json.dumps(identity_info, sort_keys=True).encode()).hexdigest()
    with batch_lock():
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not args.resume:
                parser.error("Output exists; use --resume to retain completed measurements")
            if manifest["identity"] != identity:
                parser.error("Resume would mix changed numerical settings or worker code")
        else:
            output.mkdir(parents=True, exist_ok=True)
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                 capture_output=True, text=True)
            manifest = dict(identity=identity, identity_inputs=identity_info,
                            created_at_utc=datetime.now(timezone.utc).isoformat(),
                            git_head=git.stdout.strip(), source_state="hashes include local benchmark code",
                            power_scheme_guid=power_scheme(), status="running",
                            protocol=plan, controller_sha256=sha(__file__))
            write_json(output / "historical_input_sha256.json", input_inventory())
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        base = dict(reference_store=plan["reference"]["store"], final_time_ms=46.)
        print("Controlled HH benchmark: 19 cases, 57 fresh serial measurements, then Q40 checks", flush=True)
        for item in plan["schedule"]:
            case = cases[item["case"]]
            tag = f"timing_{item['sequence']:03d}"
            request = dict(base, kind="timing", case_id=case["case"],
                sequence=item["sequence"], repeat=item["repeat"],
                field_case=dict(order=case["bdf_order"], degree=case["degree"],
                    elements=case["elements_per_branch"], dt=case["dt_ms"], quadrature=24),
                baseline_metadata=case["historical_metadata"],
                historical_errors=dict(voltage_error=case["historical_voltage_l2_error"],
                                       gate_error=case["historical_gate_l2_error"]))
            if item["repeat"] == 1:
                request["save_fields"] = (output / "fields" / case["case"]).relative_to(PROJECT).as_posix()
            print(f"START {item['sequence']}/57 {case['case']} repeat {item['repeat']}", flush=True)
            record, reused = run_one(request, output, tag, identity, manifest["power_scheme_guid"])
            t = record["timing"]
            print(f"DONE {item['sequence']}/57 {case['case']}: wall={t['solve_wall_seconds']:.3f}s "
                  f"CPU={t['solve_cpu_seconds']:.3f}s cached_record={reused}", flush=True)
        control_cases = [case for case in cases.values() if case["quadrature_check_required"]]
        # Include any newly masked cases even if the historical run did not mask.
        for path in sorted((output / "records").glob("timing_*.json")):
            row = json.loads(path.read_text())
            case = cases[row["case_id"]]
            if row["diagnostics"]["gate_mask_steps"] and case not in control_cases:
                control_cases.append(case)
        ref = plan["reference"]
        ref_meta = PROJECT / ref["store"] / "runs/bdf5_p4_n1280_dt0.00078125_q24/metadata.json"
        control_cases.append(dict(case="REFERENCE", bdf_order=5, degree=4,
            elements_per_branch=1280, dt_ms=1/1280,
            historical_metadata=ref_meta.relative_to(PROJECT).as_posix()))
        for index, case in enumerate(control_cases, 1):
            print(f"START Q40 {index}/{len(control_cases)} {case['case']}", flush=True)
            request = dict(base, kind="quadrature", case_id=case["case"],
                field_case=dict(order=case["bdf_order"], degree=case["degree"],
                    elements=case["elements_per_branch"], dt=case["dt_ms"], quadrature=40),
                baseline_metadata=case["historical_metadata"],
                save_fields=(output / "quadrature_fields" / case["case"]).relative_to(PROJECT).as_posix())
            record, reused = run_one(request, output, "quadrature_" + case["case"], identity,
                                     manifest["power_scheme_guid"])
            print(f"DONE Q40 {case['case']}: voltage change="
                  f"{record['quadrature_changes']['voltage_error']:.8g}", flush=True)
        prior = json.loads((output / "historical_input_sha256.json").read_text())
        if input_inventory() != prior:
            raise RuntimeError("Historical study inputs changed during the benchmark")
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        historical_inputs_unchanged=True)
        write_json(manifest_path, manifest)
        write_json(output / "progress.json", dict(status="completed", measurements=57,
                                                  quadrature_controls=len(control_cases)))
    from studies.hh_convergence.runtime_benchmark.analyze import analyze
    analyze(output)
    print("BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
