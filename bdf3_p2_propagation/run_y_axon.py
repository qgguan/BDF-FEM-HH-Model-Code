"""Run the paper's Y-axon example: python run_y_axon.py --help."""

import argparse
from pathlib import Path

from branched_hh import BDF3Solver, SimulationConfig, y_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=100, help="P2 elements per branch")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--final-time", type=float, default=76.0)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--voltage-limit", type=float, default=200.0)
    parser.add_argument("--amplitude", type=float, default=100.0)
    parser.add_argument("--quadrature", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("outputs/y_axon"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    mesh = y_graph(args.elements, args.quadrature)
    config = SimulationConfig(dt=args.dt, final_time=args.final_time,
                              sample_interval=args.sample_interval, epsilon=args.epsilon,
                              voltage_limit=args.voltage_limit, stimulus_amplitude=args.amplitude)
    print(f"Y graph: {mesh.nelements} P2 elements, {mesh.ndof} shared global nodes", flush=True)
    result = BDF3Solver(mesh, config).run(progress=True)
    result.save(args.output)
    if not args.no_plots:
        from plot_results import plot_result
        plot_result(result, args.output)
    d = result.diagnostics
    print(f"Saved {len(result.times)} snapshots to {args.output.resolve()}")
    print(f"Voltage range: {d['voltage_nodal_range']}; mask steps: {d['gate_mask_steps']}; "
          f"cutoff steps: {d['voltage_cutoff_steps']}")
    print(f"Solver time: {d['elapsed_seconds']:.2f} s")


if __name__ == "__main__":
    main()
