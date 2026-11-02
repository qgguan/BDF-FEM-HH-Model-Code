"""Run T=1 cosine-manufactured studies with h=dt=1/40,...,1/320.

Run from python-bdf-fem: python -m studies.manufactured.run_joint_t1
The requested four CSV tables and companion JSON settings go to results/.
"""

from dataclasses import asdict
import time

import numpy as np

from model import HHParameters
from studies.manufactured.validate import add_rates, assess, manufactured_run, save_study
from studies.paths import MANUFACTURED_RESULTS


PAIRS = ((2, 1), (3, 2), (4, 3), (5, 4))
ELEMENT_COUNTS = (40, 80, 160, 320)
FINAL_TIME = 1.0
SAMPLE_INTERVAL = 0.05
QUADRATURE = 24


def main():
    directory = MANUFACTURED_RESULTS
    directory.mkdir(parents=True, exist_ok=True)
    all_passed = True
    for order, degree in PAIRS:
        name = f"joint_t1_bdf{order}_p{degree}"
        rows, runs = [], []
        for elements in ELEMENT_COUNTS:
            dt = 1.0 / elements
            print(f"{name}: elements={elements}, h=dt={dt:g}, T={FINAL_TIME:g}", flush=True)
            started = time.perf_counter()
            result, errors = manufactured_run(
                order, degree, elements, dt, FINAL_TIME, SAMPLE_INTERVAL,
                quadrature=QUADRATURE, profile="cosine",
            )
            if not (np.isclose(result.mesh.h, dt, rtol=0, atol=1e-14)
                    and np.isclose(result.times[-1], FINAL_TIME, rtol=0, atol=1e-14)
                    and len(result.times) == 21
                    and all(np.isfinite(value) for value in errors.values())):
                raise RuntimeError(f"Unexpected mesh, sample times, or errors in {name}.")
            rows.append(dict(elements=elements, h=result.mesh.h, dt=dt,
                             dofs=result.mesh.ndof, **errors))
            runs.append(dict(elements_per_edge=elements, h=result.mesh.h,
                             dt=dt, final_time=float(result.times[-1]),
                             samples=len(result.times), config=asdict(result.config),
                             diagnostics=result.diagnostics))
            print(f"  V L2={errors['voltage_error']:.8e}, "
                  f"gate L2={errors['gate_error']:.8e}, "
                  f"elapsed={time.perf_counter() - started:.2f} s", flush=True)
        add_rates(rows, "h")
        expected = min(order, degree + 1)
        info = dict(
            study="joint", bdf_order=order, degree=degree, profile="cosine",
            final_time=FINAL_TIME, sample_interval=SAMPLE_INTERVAL,
            elements_per_edge=list(ELEMENT_COUNTS),
            time_steps=[1.0 / n for n in ELEMENT_COUNTS],
            refinement="h=dt; both halved together",
            reference="analytic continuous fields",
            error_definition="Maximum over stored times including t=0 and t=1. "
                             "Primary voltage/gate errors are continuous graph L2; "
                             "gate_nodal is the separate Gauss-Lobatto norm; "
                             "voltage_h1 is the derivative error seminorm.",
            initialization="Ritz-projected exact voltage and nodally interpolated "
                           "exact gates at all BDF starting levels",
            assembly_quadrature=QUADRATURE,
            error_quadrature=max(40, QUADRATURE + 8),
            expected_voltage_rate=expected, expected_gate_rate=expected,
            parameters=asdict(HHParameters()),
            leak_reversal_used=HHParameters().leak_reversal,
            runs=runs,
        )
        info["passed"] = assess(rows, info)
        all_passed = all_passed and info["passed"]
        save_study(directory, name, rows, info)
        print(f"Saved {name}.csv: last V rate={rows[-1]['voltage_rate']:.4f}, "
              f"gate rate={rows[-1]['gate_rate']:.4f}, "
              f"expected={expected}, passed={info['passed']}", flush=True)
    if not all_passed:
        raise SystemExit("All four tables were saved; some convergence checks failed. "
                         "Inspect the errors and rates before interpreting them.")
    print(f"All four joint studies passed. Results: {directory.resolve()}", flush=True)


if __name__ == "__main__":
    main()
