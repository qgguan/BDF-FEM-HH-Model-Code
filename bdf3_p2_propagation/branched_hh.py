"""Conforming P2 FEM / masked IMEX BDF3 for Hodgkin--Huxley metric graphs.

The voltage uses a consistent mass matrix. Gates are updated at the shared
global P2 nodes, without clipping the gate solution. See README.md for the
relationship to the manuscript and the meaning of the regularizations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spsolve


Array = np.ndarray
# Field functions receive (time, edge indices, arc lengths measured from edge start).
Field = Callable[[float, Array, Array], Array]


@dataclass(frozen=True)
class Edge:
    start: int
    end: int
    name: str = ""
    length: float | None = None


class GraphMesh:
    """Straight-edge drawing of a metric graph, with shared vertex P2 DOFs.

    An optional Edge.length overrides drawing length for the PDE. All edges
    are parametrized from s=0 at start to s=L at end. Curvature of an embedded
    drawing does not enter the metric-graph equations.
    """

    def __init__(self, vertices, edges, elements=100, quadrature_order=24):
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("vertices must be an (n, 2) or (n, 3) array")
        if not np.all(np.isfinite(points)):
            raise ValueError("vertex coordinates must be finite")
        if points.shape[1] == 2:
            points = np.column_stack((points, np.zeros(len(points))))
        self.vertices = points.copy()
        self.edges = tuple(edges)
        if not self.edges:
            raise ValueError("at least one edge is required")
        counts = np.broadcast_to(elements, (len(self.edges),))
        if np.any(counts < 1) or np.any(counts != counts.astype(int)):
            raise ValueError("elements must be positive integers, one per edge")
        counts = counts.astype(int)
        if int(quadrature_order) != quadrature_order or quadrature_order < 3:
            raise ValueError("quadrature_order must be an integer >= 3")
        self.quadrature_order = int(quadrature_order)
        xyz = list(points.copy())
        node_edge = [-1] * len(points)
        node_s = [0.0] * len(points)
        cells, cell_edges, cell_s0, cell_lengths = [], [], [], []
        self.edge_dofs, self.edge_s, lengths = [], [], []
        used_vertices = set()
        for eid, (edge, count) in enumerate(zip(self.edges, counts)):
            if not (0 <= edge.start < len(points) and 0 <= edge.end < len(points)):
                raise ValueError("edge vertex index is out of range")
            if edge.start == edge.end:
                raise ValueError("split self-loops into at least two edges")
            length = (np.linalg.norm(points[edge.end] - points[edge.start])
                      if edge.length is None else float(edge.length))
            if not np.isfinite(length) or length <= 0:
                raise ValueError("every edge must have positive finite length")
            lengths.append(length)
            used_vertices.update((edge.start, edge.end))
            local = [edge.start]
            for j in range(1, 2 * count):
                local.append(len(xyz))
                ratio = j / (2 * count)
                xyz.append((1 - ratio) * points[edge.start] + ratio * points[edge.end])
                node_edge.append(eid)
                node_s.append(ratio * length)
            local.append(edge.end)
            for endpoint, s in ((edge.start, 0.0), (edge.end, length)):
                if node_edge[endpoint] == -1:
                    node_edge[endpoint], node_s[endpoint] = eid, s
            self.edge_dofs.append(np.asarray(local, dtype=int))
            self.edge_s.append(np.linspace(0, length, 2 * count + 1))
            for j in range(count):
                cells.append(local[2 * j:2 * j + 3])
                cell_edges.append(eid)
                cell_s0.append(j * length / count)
                cell_lengths.append(length / count)
        if used_vertices != set(range(len(points))):
            raise ValueError("isolated vertices are not supported")
        self.coordinates = np.asarray(xyz)
        self.node_edge = np.asarray(node_edge)
        self.node_s = np.asarray(node_s)
        self.cells = np.asarray(cells)
        self.element_edge = np.asarray(cell_edges)
        self.element_s0 = np.asarray(cell_s0)
        self.element_lengths = np.asarray(cell_lengths)
        self.edge_lengths = np.asarray(lengths)
        self.elements_per_edge = counts
        self.ndof = len(xyz)
        self.nelements = len(cells)
        self.h = float(np.max(self.element_lengths))
        xi, weights = leggauss(self.quadrature_order)
        self.basis = np.stack((xi * (xi - 1) / 2, 1 - xi**2,
                               xi * (xi + 1) / 2), axis=1)
        derivative = np.stack((xi - 0.5, -2 * xi, xi + 0.5), axis=1)
        self.qweights = self.element_lengths[:, None] * weights[None, :] / 2
        self.qs = (self.element_s0[:, None]
                   + self.element_lengths[:, None] * (xi[None, :] + 1) / 2)
        self.qedges = np.broadcast_to(self.element_edge[:, None], self.qs.shape)
        self._rows = np.repeat(self.cells, 3, axis=1).ravel()
        self._cols = np.tile(self.cells, (1, 3)).ravel()
        local_mass = np.einsum("eq,qi,qj->eij", self.qweights, self.basis, self.basis)
        local_stiffness = np.einsum("q,qi,qj,e->eij", weights, derivative,
                                    derivative, 2 / self.element_lengths)
        self.mass = self._matrix(local_mass)
        self.stiffness = self._matrix(local_stiffness)
        self.lumped_weights = np.zeros(self.ndof)
        np.add.at(self.lumped_weights, self.cells.ravel(),
                  (self.element_lengths[:, None] * np.array([1, 4, 1]) / 6).ravel())
        self._ritz_solve = None

    def _matrix(self, local):
        return coo_matrix((local.ravel(), (self._rows, self._cols)),
                          shape=(self.ndof, self.ndof)).tocsc()

    def at_quadrature(self, nodal):
        """Accept a scalar nodal field or a stack of fields (..., ndof)."""
        return np.einsum("...ei,qi->...eq", np.asarray(nodal)[..., self.cells], self.basis)

    def reaction_matrix(self, coefficient):
        local = np.einsum("eq,qi,qj->eij", self.qweights * coefficient,
                          self.basis, self.basis)
        return self._matrix(local)

    def load_vector(self, values):
        local = np.einsum("eq,qi->ei", self.qweights * values, self.basis)
        result = np.zeros(self.ndof)
        np.add.at(result, self.cells.ravel(), local.ravel())
        return result

    def l2_norm(self, nodal):
        values = self.at_quadrature(nodal)
        return np.sqrt(np.sum(self.qweights * values**2, axis=(-2, -1)))

    def nodal_norm(self, nodal):
        return np.sqrt(np.sum(self.lumped_weights * np.asarray(nodal)**2, axis=-1))

    def ritz_projection(self, values, derivative, diffusion):
        """Project exact quadrature values/arc derivatives with D*K + M."""
        if self._ritz_solve is None or self._ritz_solve[0] != diffusion:
            self._ritz_solve = (diffusion, splu(diffusion * self.stiffness + self.mass))
        xi, _ = leggauss(self.quadrature_order)
        db = np.stack((xi - 0.5, -2 * xi, xi + 0.5), axis=1)
        local = np.einsum("eq,qi,e->ei", self.qweights * derivative, db,
                          2 / self.element_lengths)
        rhs = self.load_vector(values)
        np.add.at(rhs, self.cells.ravel(), diffusion * local.ravel())
        return self._ritz_solve[1].solve(rhs)

    def metadata(self):
        return {"vertices": self.vertices.tolist(), "edges": [asdict(e) for e in self.edges],
                "elements_per_edge": self.elements_per_edge.tolist(),
                "edge_lengths": self.edge_lengths.tolist(), "dofs": self.ndof,
                "quadrature_order": self.quadrature_order, "h_max": self.h}


def y_graph(elements=100, quadrature_order=24):
    c = 1 / np.sqrt(2)
    return GraphMesh([[-1, 0], [0, 0], [c, c], [c, -c]],
                     [Edge(0, 1, "trunk"), Edge(1, 2, "upper"), Edge(1, 3, "lower")],
                     elements, quadrature_order)


def _saturation(displacement, width):
    # tanh(P3(4)) already rounds to 1. Cap before powers to avoid overflow.
    u = np.minimum(np.abs(displacement), 4 * width) / width
    return np.sign(displacement) * width * np.tanh(u + u**3 / 3 + u**5 / 5)


def gate_mask(values, epsilon=1e-3):
    """Appendix A's C6 mask, equal to the identity on [0, 1]."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    x = np.asarray(values, dtype=float)
    return np.where(x < 0, _saturation(x, epsilon),
                    np.where(x > 1, 1 + _saturation(x - 1, epsilon), x))


def voltage_cutoff(values, limit=200.0):
    """C6 symmetric cutoff: identity on [-M,M], bounded by M+1."""
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("voltage cutoff limit must be positive and finite")
    x = np.asarray(values, dtype=float)
    excess = np.maximum(np.abs(x) - limit, 0)
    return np.where(np.abs(x) <= limit, x, np.sign(x) * (limit + _saturation(excess, 1)))


def _vtrap(x, scale):
    """x / expm1(x/scale), with a Taylor expansion at the removable zero."""
    z = np.asarray(x, dtype=float) / scale
    small = np.abs(z) < 1e-4
    out = np.empty_like(z)
    zs = z[small]
    out[small] = scale * (1 - zs / 2 + zs**2 / 12 - zs**4 / 720)
    out[~small] = scale * z[~small] / np.expm1(z[~small])
    return out


def gate_rates(voltage, cutoff=200.0):
    """Return alpha and lambda=alpha+beta in gate order (m,h,n)."""
    v = voltage_cutoff(voltage, cutoff)
    alpha = np.stack((0.1 * _vtrap(25 - v, 10), 0.07 * np.exp(-v / 20),
                      0.01 * _vtrap(10 - v, 10)))
    beta = np.stack((4 * np.exp(-v / 18), 1 / (np.exp((30 - v) / 10) + 1),
                     0.125 * np.exp(-v / 80)))
    return alpha, alpha + beta


@dataclass(frozen=True)
class HHParameters:
    capacitance: float = 1.0
    radius: float = 0.01
    resistivity: float = 35.4
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 115.0
    e_k: float = -12.0
    e_l: float | None = None

    def __post_init__(self):
        numbers = [v for v in asdict(self).values() if v is not None]
        if not np.all(np.isfinite(numbers)):
            raise ValueError("model parameters must be finite")
        if min(self.capacitance, self.radius, self.resistivity, self.g_l) <= 0:
            raise ValueError("Cm, radius, resistivity and leak conductance must be positive")
        if min(self.g_na, self.g_k) < 0:
            raise ValueError("sodium/potassium conductances must be nonnegative")

    @property
    def diffusion(self):
        return self.radius / (2 * self.resistivity)

    @cached_property
    def leak_reversal(self):
        if self.e_l is not None:
            return self.e_l
        alpha, lam = gate_rates(0.0)
        m, h, n = alpha / lam
        return (-self.g_na * m**3 * h * self.e_na
                - self.g_k * n**4 * self.e_k) / self.g_l

    def coefficients(self, gates):
        m, h, n = gates
        sodium, potassium = self.g_na * m**3 * h, self.g_k * n**4
        return (sodium + potassium + self.g_l,
                sodium * self.e_na + potassium * self.e_k + self.g_l * self.leak_reversal)


def paper_stimulus(t, edge, s, amplitude=100.0, trunk_edge=0):
    """Paper's smooth bump, on the first 0.1 length units of the trunk."""
    # s=0 is the left tip (paper coordinate x=-1).
    if not 0.1 < t < 0.3:
        return np.zeros_like(s)
    temporal = np.exp(1 - 0.01 / (0.01 - (t - 0.2)**2))
    spatial = np.zeros_like(s)
    inside = (edge == trunk_edge) & (s > 0) & (s < 0.1)
    z = 2 * s[inside] - 0.1
    spatial[inside] = np.exp(1 - 0.01 / (0.01 - z**2))
    return amplitude * temporal * spatial


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.02
    final_time: float = 76.0
    sample_interval: float = 0.1
    epsilon: float = 1e-3
    voltage_limit: float = 200.0
    stimulus_amplitude: float = 100.0

    def __post_init__(self):
        if not np.all(np.isfinite(list(asdict(self).values()))):
            raise ValueError("simulation settings must be finite")
        if min(self.dt, self.final_time, self.sample_interval, self.epsilon,
               self.voltage_limit) <= 0:
            raise ValueError("time settings, epsilon and voltage limit must be positive")
        for label, value in (("final_time", self.final_time),
                             ("sample_interval", self.sample_interval)):
            ratio = value / self.dt
            if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
                raise ValueError(f"{label} must be an integer multiple of dt")
        if self.final_time < 2 * self.dt or self.sample_interval < self.dt:
            raise ValueError("need T >= 2*dt and sample_interval >= dt")


@dataclass
class SimulationResult:
    mesh: GraphMesh
    config: SimulationConfig
    parameters: HHParameters
    times: Array
    voltage: Array
    gates: Array
    diagnostics: dict

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "solution.npz", times=self.times,
                            voltage=self.voltage, gates=self.gates,
                            coordinates=self.mesh.coordinates, cells=self.mesh.cells,
                            node_edge=self.mesh.node_edge, node_s=self.mesh.node_s,
                            lumped_weights=self.mesh.lumped_weights)
        report = {"config": asdict(self.config), "parameters": asdict(self.parameters),
                  "leak_reversal_used": self.parameters.leak_reversal,
                  "mesh": self.mesh.metadata(), "diagnostics": self.diagnostics,
                  "gate_order": ["m", "h", "n"],
                  "history_axes": {"voltage": ["sample", "node"],
                                   "gates": ["sample", "gate", "node"]}}
        (directory / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


class BDF3Solver:
    def __init__(self, mesh, config=None, parameters=None):
        self.mesh = mesh
        self.config = config or SimulationConfig()
        self.parameters = parameters or HHParameters()
        p, c = self.parameters, self.config
        if p.g_na * c.epsilon * (1 + c.epsilon)**3 > p.g_l / 2:
            raise ValueError("epsilon violates gNa*epsilon*(1+epsilon)^3 <= gL/2")
        self._constant_matrix = 11 * p.capacitance / (6 * c.dt) * mesh.mass + p.diffusion * mesh.stiffness

    def run(self, *, initial_voltage: Field | None = None,
            initial_gates: Field | None = None, voltage_source: Field | None = None,
            gate_source: Field | None = None, initial_derivative: Field | None = None,
            progress=False):
        """Advance the scheme.

        By default initialize the exact resting history before the paper's
        stimulus. For other problems, provide exact/accurate histories as
        functions evaluated at t=0,dt,2dt, plus any manufactured sources.
        initial_derivative selects Ritz projection instead of interpolation.
        voltage_source REPLACES the paper stimulus. gate_source is an additive
        nodal source, zero in the original HH problem.
        """
        mesh, c, p = self.mesh, self.config, self.parameters
        if (initial_voltage is None) != (initial_gates is None):
            raise ValueError("provide both initial_voltage and initial_gates")
        if initial_voltage is None:
            if voltage_source is not None or gate_source is not None or initial_derivative is not None:
                raise ValueError("custom forcing requires explicit starting-history functions")
            if 2 * c.dt > 0.1:
                raise ValueError("resting startup requires 2*dt <= stimulus onset 0.1")
            if not np.isclose(p.leak_reversal, HHParameters(
                    g_na=p.g_na, g_k=p.g_k, g_l=p.g_l, e_na=p.e_na,
                    e_k=p.e_k).leak_reversal, rtol=0, atol=1e-12):
                raise ValueError("resting startup requires the equilibrium leak reversal")
            alpha, lam = gate_rates(0.0, c.voltage_limit)
            rest = alpha / lam
            initial_voltage = lambda t, e, s: np.zeros_like(s)
            initial_gates = lambda t, e, s: np.broadcast_to(rest[:, None], (3, s.size)).copy()
        if voltage_source is None:
            voltage_source = lambda t, e, s: paper_stimulus(t, e, s, c.stimulus_amplitude)
        vh, wh = [], []  # newest first after reversal
        for j in range(3):
            t = j * c.dt
            if initial_derivative is None:
                v = np.asarray(initial_voltage(t, mesh.node_edge, mesh.node_s), dtype=float)
            else:
                v = mesh.ritz_projection(initial_voltage(t, mesh.qedges, mesh.qs),
                                         initial_derivative(t, mesh.qedges, mesh.qs), p.diffusion)
            w = np.asarray(initial_gates(t, mesh.node_edge, mesh.node_s), dtype=float)
            if v.shape != (mesh.ndof,) or w.shape != (3, mesh.ndof):
                raise ValueError("starting fields must have shapes (nodes,) and (3,nodes)")
            if not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)):
                raise ValueError("starting fields must be finite")
            vh.append(v.copy())
            wh.append(w.copy())
        stride = int(round(c.sample_interval / c.dt))
        nsteps = int(round(c.final_time / c.dt))
        samples_t, samples_v, samples_w = [], [], []

        def sample(step, v, w):
            if step % stride == 0 or step == nsteps:
                samples_t.append(step * c.dt)
                samples_v.append(v.copy())
                samples_w.append(w.copy())

        for j in range(3):
            sample(j, vh[j], wh[j])
        initial_v = np.asarray(vh)
        initial_w = np.asarray(wh)
        vmin, vmax = float(initial_v.min()), float(initial_v.max())
        wmin = initial_w.min(axis=(0, 2))
        wmax = initial_w.max(axis=(0, 2))
        vh.reverse()
        wh.reverse()
        diagnostics = {"gate_mask_steps": 0, "masked_quadrature_values": 0,
                       "voltage_cutoff_steps": 0, "cutoff_nodal_values": 0,
                       "min_conductance": float("inf"),
                       "extrapolated_gate_range_at_quadrature": [float("inf"), -float("inf")],
                       "max_relative_linear_residual": 0.0}
        started = time.perf_counter()
        report_every = max(1, nsteps // 10)
        for step in range(3, nsteps + 1):
            t = step * c.dt
            extrapolated = 3 * wh[0] - 3 * wh[1] + wh[2]
            wq = mesh.at_quadrature(extrapolated)
            outside = int(np.count_nonzero((wq < 0) | (wq > 1)))
            diagnostics["gate_mask_steps"] += int(outside > 0)
            diagnostics["masked_quadrature_values"] += outside
            ran = diagnostics["extrapolated_gate_range_at_quadrature"]
            ran[0], ran[1] = min(ran[0], float(wq.min())), max(ran[1], float(wq.max()))
            conductance, reversal = p.coefficients(gate_mask(wq, c.epsilon))
            diagnostics["min_conductance"] = min(diagnostics["min_conductance"], float(conductance.min()))
            matrix = self._constant_matrix + mesh.reaction_matrix(conductance)
            history = (18 * vh[0] - 9 * vh[1] + 2 * vh[2]) * p.capacitance / (6 * c.dt)
            source = np.asarray(voltage_source(t, mesh.qedges, mesh.qs))
            if source.shape != mesh.qs.shape:
                raise ValueError("voltage_source must return the quadrature field shape")
            rhs = mesh.mass @ history + mesh.load_vector(reversal + source)
            v = spsolve(matrix, rhs, use_umfpack=False)
            cutoff_count = int(np.count_nonzero(np.abs(v) > c.voltage_limit))
            diagnostics["voltage_cutoff_steps"] += int(cutoff_count > 0)
            diagnostics["cutoff_nodal_values"] += cutoff_count
            alpha, lam = gate_rates(v, c.voltage_limit)
            source_w = 0.0 if gate_source is None else np.asarray(gate_source(t, mesh.node_edge, mesh.node_s))
            if np.ndim(source_w) and np.shape(source_w) != (3, mesh.ndof):
                raise ValueError("gate_source must have shape (3,nodes)")
            w = (18 * wh[0] - 9 * wh[1] + 2 * wh[2] + 6 * c.dt * (alpha + source_w)) / (11 + 6 * c.dt * lam)
            if not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)):
                raise FloatingPointError(f"nonfinite solution at step {step}, t={t}")
            residual = np.linalg.norm(matrix @ v - rhs) / max(np.linalg.norm(rhs), 1e-14)
            diagnostics["max_relative_linear_residual"] = max(diagnostics["max_relative_linear_residual"], float(residual))
            vmin, vmax = min(vmin, float(v.min())), max(vmax, float(v.max()))
            wmin, wmax = np.minimum(wmin, w.min(axis=1)), np.maximum(wmax, w.max(axis=1))
            vh = [v, vh[0], vh[1]]
            wh = [w, wh[0], wh[1]]
            sample(step, v, w)
            if progress and (step % report_every == 0 or step == nsteps):
                print(f"t={t:8.3f} / {c.final_time:g}, V=[{v.min():.3f},{v.max():.3f}]", flush=True)
        if nsteps == 2:
            diagnostics["min_conductance"] = None
            diagnostics["extrapolated_gate_range_at_quadrature"] = None
        diagnostics.update({"steps": nsteps, "bdf3_solves": nsteps - 2,
                            "elapsed_seconds": time.perf_counter() - started,
                            "voltage_nodal_range": [vmin, vmax],
                            "gate_nodal_ranges": np.stack((wmin, wmax), axis=1).tolist()})
        return SimulationResult(mesh, c, p, np.asarray(samples_t), np.asarray(samples_v),
                                np.asarray(samples_w), diagnostics)
