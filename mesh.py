"""Conforming P1--P4 elements on metric graphs, with shared vertex DOFs."""

from dataclasses import asdict, dataclass
import numpy as np
from numpy.polynomial import Polynomial, Legendre
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class Edge:
    start: int
    end: int
    name: str = ""
    length: float | None = None


def integer(value, name, lower, upper=None):
    if not np.isscalar(value) or not np.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} must be >= {lower}" + (f" and <= {upper}" if upper else ""))
    return value


def lobatto_nodes(degree):
    """Interpolation nodes and positive weights, exact through degree 2p-1."""
    p = integer(degree, "degree", 1, 4)
    polynomial = Legendre.basis(p)
    nodes = np.r_[-1.0, polynomial.deriv().roots(), 1.0]
    weights = 2 / (p * (p + 1) * polynomial(nodes)**2)
    return nodes, weights


class GraphMesh:
    def __init__(self, vertices, edges, elements=40, degree=2, quadrature_order=None):
        self.degree = integer(degree, "degree", 1, 4)
        # Unmasked HH conductance has degree 4p; two test/trial functions
        # give degree 6p. Q-point Gauss integrates through degree 2Q-1.
        self.minimum_quadrature = 3 * self.degree + 1
        self.quadrature_order = integer(
            max(24, self.minimum_quadrature) if quadrature_order is None else quadrature_order,
            "quadrature_order", self.minimum_quadrature)
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3) or not np.all(np.isfinite(points)):
            raise ValueError("vertices must be a finite (n,2) or (n,3) array")
        if points.shape[1] == 2:
            points = np.column_stack((points, np.zeros(len(points))))
        self.vertices = points.copy()
        self.edges = tuple(edges)
        if not self.edges:
            raise ValueError("at least one edge is required")
        counts = np.broadcast_to(elements, (len(self.edges),))
        counts = np.array([integer(n, "elements", 1) for n in counts])
        self.elements_per_edge = counts
        self.reference_nodes, self.reference_nodal_weights = lobatto_nodes(self.degree)
        self._polynomials = []
        for i, x in enumerate(self.reference_nodes):
            others = np.delete(self.reference_nodes, i)
            self._polynomials.append(Polynomial.fromroots(others) / np.prod(x - others))
        xyz = list(points.copy())
        node_edge, node_s = [-1] * len(points), [0.0] * len(points)
        cells, cell_edges, starts, lengths, edge_lengths = [], [], [], [], []
        self.edge_dofs, self.edge_s = [], []
        used = set()
        for eid, (edge, count) in enumerate(zip(self.edges, counts)):
            a = integer(edge.start, "start vertex", 0, len(points) - 1)
            b = integer(edge.end, "end vertex", 0, len(points) - 1)
            if a == b:
                raise ValueError("split self-loops into at least two edges")
            length = np.linalg.norm(points[b] - points[a]) if edge.length is None else float(edge.length)
            if not np.isfinite(length) or length <= 0:
                raise ValueError("edge lengths must be positive and finite")
            edge_lengths.append(length)
            used.update((a, b))
            local, arc = [a], [0.0]
            for j in range(count):
                for r in range(1, self.degree + 1):
                    ratio = (j + (self.reference_nodes[r] + 1) / 2) / count
                    if j == count - 1 and r == self.degree:
                        node = b
                    else:
                        node = len(xyz)
                        xyz.append((1 - ratio) * points[a] + ratio * points[b])
                        node_edge.append(eid)
                        node_s.append(ratio * length)
                    local.append(node)
                    arc.append(ratio * length)
                cells.append(local[self.degree * j:self.degree * (j + 1) + 1])
                cell_edges.append(eid)
                starts.append(j * length / count)
                lengths.append(length / count)
            for endpoint, s in ((a, 0.0), (b, length)):
                if node_edge[endpoint] == -1:
                    node_edge[endpoint], node_s[endpoint] = eid, s
            self.edge_dofs.append(np.asarray(local))
            self.edge_s.append(np.asarray(arc))
        if used != set(range(len(points))):
            raise ValueError("isolated vertices are not supported")
        self.coordinates = np.asarray(xyz)
        self.node_edge, self.node_s = np.asarray(node_edge), np.asarray(node_s)
        self.cells, self.element_edge = np.asarray(cells), np.asarray(cell_edges)
        self.element_s0, self.element_lengths = np.asarray(starts), np.asarray(lengths)
        self.edge_lengths = np.asarray(edge_lengths)
        self.ndof, self.nelements = len(xyz), len(cells)
        self.h = float(max(lengths))
        width = self.degree + 1
        self._rows = np.repeat(self.cells, width, axis=1).ravel()
        self._cols = np.tile(self.cells, (1, width)).ravel()
        self._rules, self._ritz_solve = {}, None
        self.rule = self.integration_rule(self.quadrature_order)
        # Reuse basis products in every reaction assembly. The matrix product
        # evaluates exactly the same quadrature sum as the three-way einsum.
        self._basis_products = np.einsum("qi,qj->qij", self.rule.basis,
                                         self.rule.basis).reshape(self.quadrature_order, -1)
        self.mass = self.reaction_matrix(np.ones_like(self.rule.s))
        local_k = np.einsum("eq,qi,qj,e->eij", self.rule.weights,
                            self.rule.derivative, self.rule.derivative,
                            (2 / self.element_lengths)**2)
        self.stiffness = self._matrix(local_k)
        # A positive Gauss--Lobatto nodal norm for every supported degree.
        # These weights are never substituted for the consistent mass solve.
        self.nodal_weights = np.zeros(self.ndof)
        local_weights = self.element_lengths[:, None] * self.reference_nodal_weights / 2
        np.add.at(self.nodal_weights, self.cells.ravel(), local_weights.ravel())

    def basis_at(self, xi, derivative=False):
        return np.stack([(p.deriv() if derivative else p)(xi) for p in self._polynomials], axis=-1)

    def integration_rule(self, order):
        order = integer(order, "integration order", self.degree + 1)
        if order not in self._rules:
            xi, weights = leggauss(order)
            s = self.element_s0[:, None] + self.element_lengths[:, None] * (xi + 1) / 2
            self._rules[order] = Quadrature(
                self.basis_at(xi), self.basis_at(xi, derivative=True),
                self.element_lengths[:, None] * weights / 2, s,
                np.broadcast_to(self.element_edge[:, None], s.shape))
        return self._rules[order]

    def _matrix(self, local):
        return coo_matrix((local.ravel(), (self._rows, self._cols)),
                          shape=(self.ndof, self.ndof)).tocsc()

    def at_quadrature(self, nodal, rule=None, derivative=False):
        rule = self.rule if rule is None else rule
        basis = rule.derivative if derivative else rule.basis
        values = np.asarray(nodal)[..., self.cells] @ basis.T
        return values * (2 / self.element_lengths[:, None]) if derivative else values

    def reaction_matrix(self, coefficient):
        local = (self.rule.weights * coefficient) @ self._basis_products
        return self._matrix(local)

    def load_vector(self, values):
        local = (self.rule.weights * values) @ self.rule.basis
        result = np.zeros(self.ndof)
        np.add.at(result, self.cells.ravel(), local.ravel())
        return result

    def l2_norm(self, nodal):
        return np.sqrt(np.sum(self.rule.weights * self.at_quadrature(nodal)**2, axis=(-2, -1)))

    def nodal_norm(self, nodal):
        return np.sqrt(np.sum(self.nodal_weights * np.asarray(nodal)**2, axis=-1))

    def ritz_projection(self, values, derivative, diffusion):
        if self._ritz_solve is None or self._ritz_solve[0] != diffusion:
            self._ritz_solve = (diffusion, splu(diffusion * self.stiffness + self.mass))
        local = np.einsum("eq,qi,e->ei", self.rule.weights * derivative,
                          self.rule.derivative, 2 / self.element_lengths)
        rhs = self.load_vector(values)
        np.add.at(rhs, self.cells.ravel(), diffusion * local.ravel())
        return self._ritz_solve[1].solve(rhs)

    def metadata(self):
        return dict(vertices=self.vertices.tolist(), edges=[asdict(e) for e in self.edges],
                    elements_per_edge=self.elements_per_edge.tolist(), degree=self.degree,
                    quadrature_order=self.quadrature_order, dofs=self.ndof, h_max=self.h,
                    reference_nodes=self.reference_nodes.tolist(), node_family="Gauss-Lobatto")


@dataclass(frozen=True)
class Quadrature:
    basis: np.ndarray
    derivative: np.ndarray
    weights: np.ndarray
    s: np.ndarray
    edges: np.ndarray


def y_graph(elements=40, degree=2, quadrature_order=None):
    c = 1 / np.sqrt(2)
    return GraphMesh([[-1, 0], [0, 0], [c, c], [c, -c]],
                     [Edge(0, 1, "trunk"), Edge(1, 2, "upper"), Edge(1, 3, "lower")],
                     elements, degree, quadrature_order)
