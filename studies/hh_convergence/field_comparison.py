"""Continuous graph norms between FE fields on different uniform edge meshes."""

from fractions import Fraction
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix


class FieldComparison:
    """Integrate on the union of both meshes, without interpolating either field.

    A Q-point Gauss rule with Q >= max(p_a, p_b)+1 exactly integrates the
    squared piecewise-polynomial difference on every common subelement.
    Graph edges are integrated separately, including their own arc lengths.
    """

    def __init__(self, mesh_a, mesh_b, quadrature=8):
        if (mesh_a.edges != mesh_b.edges
                or mesh_a.vertices.shape != mesh_b.vertices.shape
                or not np.allclose(mesh_a.vertices, mesh_b.vertices, rtol=0, atol=1e-13)
                or not np.allclose(mesh_a.edge_lengths, mesh_b.edge_lengths, rtol=0, atol=1e-13)):
            raise ValueError("field comparison requires the same graph and edge ordering")
        if int(quadrature) != quadrature or quadrature < max(mesh_a.degree, mesh_b.degree)+1:
            raise ValueError("comparison quadrature must integrate squared FE differences exactly")
        xi, gw = leggauss(int(quadrature))
        points, weights, edge_ids = [], [], []
        for eid, length in enumerate(mesh_a.edge_lengths):
            na, nb = int(mesh_a.elements_per_edge[eid]), int(mesh_b.elements_per_edge[eid])
            # Rational breakpoints merge coincident nodes even for nonnested meshes.
            fractions = sorted({Fraction(j, na) for j in range(na+1)}
                               | {Fraction(j, nb) for j in range(nb+1)})
            knots = np.array([float(f)*length for f in fractions])
            half = np.diff(knots)/2
            points.append((knots[:-1, None]+knots[1:, None])/2+half[:, None]*xi)
            weights.append(half[:, None]*gw)
            edge_ids.extend([eid]*(len(half)*len(xi)))
        self.s = np.concatenate([p.ravel() for p in points])
        self.weights = np.concatenate([w.ravel() for w in weights])
        self.edge_ids = np.array(edge_ids)
        self.a, self.da = self._operators(mesh_a)
        self.b, self.db = self._operators(mesh_b)

    def _operators(self, mesh):
        counts = mesh.elements_per_edge[self.edge_ids]
        lengths = mesh.edge_lengths[self.edge_ids]
        local = np.minimum((self.s/lengths*counts).astype(int), counts-1)
        offsets = np.r_[0, np.cumsum(mesh.elements_per_edge)[:-1]]
        cells = offsets[self.edge_ids]+local
        x = 2*(self.s-mesh.element_s0[cells])/mesh.element_lengths[cells]-1
        rows = np.repeat(np.arange(len(self.s)), mesh.degree+1)
        cols = mesh.cells[cells].ravel()
        basis = mesh.basis_at(x)
        derivative = mesh.basis_at(x, derivative=True)*(2/mesh.element_lengths[cells, None])
        shape = (len(self.s), mesh.ndof)
        return (coo_matrix((basis.ravel(), (rows, cols)), shape=shape).tocsr(),
                coo_matrix((derivative.ravel(), (rows, cols)), shape=shape).tocsr())

    def at_time(self, va, wa, vb, wb):
        dv = self.a@va-self.b@vb
        ddv = self.da@va-self.db@vb
        dw = self.a@wa.T-self.b@wb.T
        return dict(voltage_error=float(np.sqrt(self.weights@dv**2)),
                    voltage_h1_error=float(np.sqrt(self.weights@ddv**2)),
                    **{f"{name}_error": float(np.sqrt(self.weights@dw[:, j]**2))
                       for j, name in enumerate(("m", "h", "n"))})


ERROR_KEYS = ("voltage_error", "voltage_h1_error", "m_error", "h_error", "n_error", "gate_error")


def compare_fields(a, b, quadrature=8, *, final_only=False):
    """Return field errors and per-time norms, optionally at final time only.

    Snapshots must be at the same physical times; no time interpolation,
    nodal snapping, phase shifting or rescaling is used.
    """
    if a.parameters != b.parameters:
        raise ValueError("physical parameters must match")
    if (a.times.shape != b.times.shape
            or not np.allclose(a.times, b.times, rtol=0, atol=1e-10)):
        raise ValueError("snapshots must use common physical times; no time interpolation is allowed")
    op = FieldComparison(a.mesh, b.mesh, quadrature)
    history = []
    indices = [len(a.times)-1] if final_only else range(len(a.times))
    for j in indices:
        t = a.times[j]
        row = op.at_time(a.voltage[j], a.gates[j], b.voltage[j], b.gates[j])
        row.update(time_ms=float(t), gate_error=max(row[f"{name}_error"] for name in ("m", "h", "n")))
        history.append(row)
    errors = {key: max(row[key] for row in history) for key in ERROR_KEYS}
    errors.update({key+"_time_ms": history[int(np.argmax([row[key] for row in history]))]["time_ms"]
                   for key in ERROR_KEYS})
    return errors, history
