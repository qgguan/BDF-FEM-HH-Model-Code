"""Masked IMEX BDF1--BDF5, with voltage implicit and gates extrapolated."""

from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
import json
import time
import numpy as np
from scipy.sparse.linalg import spsolve

from mesh import Edge, GraphMesh, integer
from model import HHParameters, gate_mask, gate_rates, paper_stimulus


# a[0]*u^{n+1} + a[1]*u^n + ... = dt*f^{n+1}.
BDF_COEFFICIENTS = {
    1: (1, -1),
    2: (3/2, -2, 1/2),
    3: (11/6, -3, 3/2, -1/3),
    4: (25/12, -4, 3, -4/3, 1/4),
    5: (137/60, -5, 5, -10/3, 5/4, -1/5),
}


def time_coefficients(order):
    k = integer(order, "bdf_order", 1, 5)
    return (np.array(BDF_COEFFICIENTS[k], dtype=float),
            np.array([(-1)**j * comb(k, j + 1) for j in range(k)], dtype=float))


@dataclass(frozen=True)
class SimulationConfig:
    bdf_order: int = 3
    dt: float = 0.01
    final_time: float = 0.5
    sample_interval: float = 0.05
    epsilon: float = 1e-3
    voltage_limit: float = 200.0
    stimulus_amplitude: float = 100.0

    def __post_init__(self):
        object.__setattr__(self, "bdf_order", integer(self.bdf_order, "bdf_order", 1, 5))
        if not np.all(np.isfinite(list(asdict(self).values()))):
            raise ValueError("simulation settings must be finite")
        if min(self.dt, self.final_time, self.sample_interval,
               self.epsilon, self.voltage_limit) <= 0:
            raise ValueError("dt, final_time, sample_interval, epsilon and voltage_limit must be positive")
        for name in ("final_time", "sample_interval"):
            ratio = getattr(self, name) / self.dt
            if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
                raise ValueError(f"{name} must be an integer multiple of dt")
        if round(self.final_time / self.dt) < self.bdf_order:
            raise ValueError("final_time must allow at least one BDF step after the starting history")
        if self.sample_interval < self.dt:
            raise ValueError("sample_interval must be >= dt")


@dataclass
class SimulationResult:
    mesh: GraphMesh
    config: SimulationConfig
    parameters: HHParameters
    times: np.ndarray
    voltage: np.ndarray
    gates: np.ndarray
    diagnostics: dict

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "solution.npz", times=self.times,
                            voltage=self.voltage, gates=self.gates,
                            coordinates=self.mesh.coordinates, cells=self.mesh.cells,
                            node_edge=self.mesh.node_edge, node_s=self.mesh.node_s,
                            nodal_weights=self.mesh.nodal_weights)
        report = dict(config=asdict(self.config), parameters=asdict(self.parameters),
                      mesh=self.mesh.metadata(), diagnostics=self.diagnostics,
                      leak_reversal_used=self.parameters.leak_reversal,
                      gate_order=["m", "h", "n"],
                      scheme=f"IMEX BDF{self.config.bdf_order}+P{self.mesh.degree}")
        (directory / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        report = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        info = report["mesh"]
        mesh = GraphMesh(info["vertices"], [Edge(**e) for e in info["edges"]],
                         info["elements_per_edge"], info["degree"], info["quadrature_order"])
        with np.load(directory / "solution.npz", allow_pickle=False) as data:
            return cls(mesh, SimulationConfig(**report["config"]), HHParameters(**report["parameters"]),
                       data["times"], data["voltage"], data["gates"], report["diagnostics"])


def finite_field(callback, t, edges, s, shape, name):
    value = np.asarray(callback(t, edges, s), dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must return a finite field of shape {shape}")
    return value


class IMEXBDFSolver:
    def __init__(self, mesh, config=None, parameters=None):
        self.mesh = mesh
        self.config = config or SimulationConfig()
        self.parameters = parameters or HHParameters()
        self.a, self.extrapolation = time_coefficients(self.config.bdf_order)
        c, p = self.config, self.parameters
        if p.g_na * c.epsilon * (1 + c.epsilon)**3 > p.g_l / 2:
            raise ValueError("epsilon violates gNa*epsilon*(1+epsilon)^3 <= gL/2")
        self.constant_matrix = self.a[0] * p.capacitance / c.dt * mesh.mass + p.diffusion * mesh.stiffness

    def run(self, *, initial_voltage=None, initial_gates=None, initial_derivative=None,
            voltage_source=None, gate_source=None, progress=False, observer=None):
        """Histories are functions evaluated at 0,dt,...,(k-1)dt, all accurate.

        Default histories are exact rest before the stimulus. Other problems
        must supply both histories; no low-order startup is substituted.
        voltage_source replaces the paper stimulus; gate_source is additive.
        Every callback receives (t, edge_indices, edge_arc_lengths).
        Optional observer(t, voltage, gates) is called at every time level,
        including all starting histories, independently of saved snapshots.
        The observer must not modify the supplied solution arrays.
        """
        mesh, c, p = self.mesh, self.config, self.parameters
        k, rule = c.bdf_order, mesh.rule
        if (initial_voltage is None) != (initial_gates is None):
            raise ValueError("provide both initial_voltage and initial_gates")
        if initial_voltage is None:
            if any(x is not None for x in (initial_derivative, voltage_source, gate_source)):
                raise ValueError("custom forcing requires explicit accurate starting histories")
            if (k - 1) * c.dt > 0.1 + 1e-14:
                raise ValueError("resting history must finish before stimulus onset: (k-1)*dt <= 0.1")
            equilibrium = HHParameters(g_na=p.g_na, g_k=p.g_k, g_l=p.g_l,
                                       e_na=p.e_na, e_k=p.e_k).leak_reversal
            if not np.isclose(p.leak_reversal, equilibrium, rtol=0, atol=1e-12):
                raise ValueError("resting startup requires the equilibrium leak reversal")
            alpha, lam = gate_rates(0.0, c.voltage_limit)
            rest = alpha / lam
            initial_voltage = lambda t, e, s: np.zeros_like(s)
            initial_gates = lambda t, e, s: np.broadcast_to(rest[:, None], (3, s.size))
        if voltage_source is None:
            voltage_source = lambda t, e, s: paper_stimulus(t, e, s, c.stimulus_amplitude)

        vh, wh = [], []
        for j in range(k):
            t = j * c.dt
            nodal_v = finite_field(initial_voltage, t, mesh.node_edge, mesh.node_s,
                                   (mesh.ndof,), "initial_voltage")
            w = finite_field(initial_gates, t, mesh.node_edge, mesh.node_s,
                             (3, mesh.ndof), "initial_gates")
            # Validate both traces of every edge, including all incidences of
            # a junction whose global node has only one canonical edge index.
            for eid, edge in enumerate(mesh.edges):
                endpoint_s = np.array([0.0, mesh.edge_lengths[eid]])
                endpoint_e = np.full(2, eid, dtype=int)
                trace_v = finite_field(initial_voltage, t, endpoint_e, endpoint_s, (2,), "voltage traces")
                trace_w = finite_field(initial_gates, t, endpoint_e, endpoint_s, (3, 2), "gate traces")
                ids = [edge.start, edge.end]
                if not (np.allclose(trace_v, nodal_v[ids], rtol=0, atol=1e-10)
                        and np.allclose(trace_w, w[:, ids], rtol=0, atol=1e-10)):
                    raise ValueError("starting histories must have common vertex traces")
            if initial_derivative is None:
                v = nodal_v
            else:
                values = finite_field(initial_voltage, t, rule.edges, rule.s, rule.s.shape, "initial_voltage")
                derivative = finite_field(initial_derivative, t, rule.edges, rule.s,
                                           rule.s.shape, "initial_derivative")
                v = mesh.ritz_projection(values, derivative, p.diffusion)
            vh.append(v.copy())
            wh.append(w.copy())

        stride, nsteps = round(c.sample_interval / c.dt), round(c.final_time / c.dt)
        sample_t, sample_v, sample_w = [], [], []

        def sample(step, v, w):
            if step % stride == 0 or step == nsteps:
                sample_t.append(step * c.dt)
                sample_v.append(v.copy())
                sample_w.append(w.copy())

        for j in range(k):
            sample(j, vh[j], wh[j])
            if observer is not None:
                observer(j * c.dt, vh[j], wh[j])
        vmin, vmax = float(np.min(vh)), float(np.max(vh))
        wmin, wmax = np.min(wh, axis=(0, 2)), np.max(wh, axis=(0, 2))
        vh.reverse()
        wh.reverse()
        diagnostics = dict(gate_mask_steps=0, masked_quadrature_values=0,
                           voltage_cutoff_steps=0, min_conductance=float("inf"),
                           max_relative_linear_residual=0.0)
        started = time.perf_counter()
        for step in range(k, nsteps + 1):
            t = step * c.dt
            extrapolated = np.tensordot(self.extrapolation, np.asarray(wh), axes=1)
            wq = mesh.at_quadrature(extrapolated)
            outside = int(np.count_nonzero((wq < 0) | (wq > 1)))
            diagnostics["gate_mask_steps"] += int(outside > 0)
            diagnostics["masked_quadrature_values"] += outside
            conductance, reversal = p.coefficients(gate_mask(wq, c.epsilon))
            diagnostics["min_conductance"] = min(diagnostics["min_conductance"], float(conductance.min()))
            matrix = self.constant_matrix + mesh.reaction_matrix(conductance)
            history_v = -np.tensordot(self.a[1:], np.asarray(vh), axes=1) * p.capacitance / c.dt
            source = finite_field(voltage_source, t, rule.edges, rule.s, rule.s.shape, "voltage_source")
            rhs = mesh.mass @ history_v + mesh.load_vector(reversal + source)
            v = spsolve(matrix, rhs, use_umfpack=False)
            diagnostics["voltage_cutoff_steps"] += int(np.any(np.abs(v) > c.voltage_limit))
            alpha, lam = gate_rates(v, c.voltage_limit)
            source_w = (0.0 if gate_source is None else finite_field(
                gate_source, t, mesh.node_edge, mesh.node_s, (3, mesh.ndof), "gate_source"))
            history_w = -np.tensordot(self.a[1:], np.asarray(wh), axes=1) / c.dt
            w = (history_w + alpha + source_w) / (self.a[0] / c.dt + lam)
            if not (np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
                raise FloatingPointError(f"nonfinite solution at t={t:g}; reduce dt")
            residual = np.linalg.norm(matrix @ v - rhs) / max(np.linalg.norm(rhs), 1e-14)
            diagnostics["max_relative_linear_residual"] = max(
                diagnostics["max_relative_linear_residual"], float(residual))
            vmin, vmax = min(vmin, float(v.min())), max(vmax, float(v.max()))
            wmin, wmax = np.minimum(wmin, w.min(axis=1)), np.maximum(wmax, w.max(axis=1))
            vh, wh = [v] + vh[:k-1], [w] + wh[:k-1]
            sample(step, v, w)
            if observer is not None:
                observer(t, v, w)
            if progress and (step % max(1, nsteps // 10) == 0 or step == nsteps):
                print(f"BDF{k}+P{mesh.degree}: t={t:g}/{c.final_time:g}, V=[{v.min():.4g},{v.max():.4g}]", flush=True)
        diagnostics.update(steps=nsteps, bdf_solves=nsteps-k+1,
                           elapsed_seconds=time.perf_counter()-started,
                           voltage_nodal_range=[vmin, vmax],
                           gate_nodal_ranges=np.stack((wmin, wmax), axis=1).tolist())
        return SimulationResult(mesh, c, p, np.array(sample_t), np.array(sample_v), np.array(sample_w), diagnostics)
