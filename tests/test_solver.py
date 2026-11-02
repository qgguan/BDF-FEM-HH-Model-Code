"""Numerical invariants for the configurable BDF/FEM family."""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from mesh import Edge, GraphMesh, y_graph
from model import HHParameters, gate_mask, gate_rates
from solver import IMEXBDFSolver, SimulationConfig, SimulationResult, time_coefficients
from manufactured import ManufacturedSolution


class FamilyTests(unittest.TestCase):
    def test_bdf_and_extrapolation_polynomial_exactness(self):
        for order in range(1, 6):
            a, extrapolation = time_coefficients(order)
            for power in range(order+1):
                derivative = float(a @ (-np.arange(order+1, dtype=float))**power)
                self.assertAlmostEqual(derivative, 1.0 if power == 1 else 0.0, places=10)
            for power in range(order):
                prediction = extrapolation @ (-np.arange(1, order+1, dtype=float))**power
                self.assertAlmostEqual(float(prediction), 1.0 if power == 0 else 0.0, places=10)

    def test_every_degree_has_conforming_positive_consistent_assembly(self):
        for degree in range(1, 5):
            with self.subTest(degree=degree):
                mesh = y_graph([2, 3, 4], degree)
                self.assertEqual(mesh.ndof, 1 + degree*9)
                junction = mesh.edge_dofs[0][-1]
                self.assertEqual(junction, mesh.edge_dofs[1][0])
                self.assertEqual(junction, mesh.edge_dofs[2][0])
                np.testing.assert_allclose(mesh.basis_at(mesh.reference_nodes), np.eye(degree+1), atol=1e-14)
                np.testing.assert_allclose(mesh.rule.basis.sum(axis=1), 1, atol=1e-14)
                np.testing.assert_allclose(mesh.stiffness @ np.ones(mesh.ndof), 0, atol=1e-11)
                self.assertAlmostEqual(float(mesh.mass.sum()), 3, places=12)
                self.assertTrue(np.all(mesh.nodal_weights > 0))
                self.assertTrue(np.all(np.linalg.eigvalsh(mesh.mass.toarray()) > 0))
                np.testing.assert_allclose(mesh.stiffness.toarray(), mesh.stiffness.toarray().T, atol=1e-13)

    def test_p2_local_matrices_and_ritz_reproduction(self):
        mesh = GraphMesh([[0, 0], [2, 0]], [Edge(0, 1)], 1, 2)
        ids = mesh.cells[0]
        np.testing.assert_allclose(mesh.mass.toarray()[np.ix_(ids, ids)],
                                   2/30*np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]]), atol=1e-14)
        np.testing.assert_allclose(mesh.stiffness.toarray()[np.ix_(ids, ids)],
                                   1/6*np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]]), atol=1e-14)
        for degree in range(1, 5):
            mesh = GraphMesh([[0, 0], [1, 0]], [Edge(0, 1)], 3, degree)
            projected = mesh.ritz_projection(mesh.rule.s**degree,
                                             degree*mesh.rule.s**(degree-1), 0.2)
            np.testing.assert_allclose(projected, mesh.node_s**degree, atol=2e-13)

    def test_all_twenty_combinations_preserve_rest(self):
        for order in range(1, 6):
            for degree in range(1, 5):
                with self.subTest(order=order, degree=degree):
                    config = SimulationConfig(bdf_order=order, dt=0.01, final_time=0.1, sample_interval=0.01)
                    result = IMEXBDFSolver(y_graph(3, degree), config).run()
                    self.assertLess(float(np.max(np.abs(result.voltage))), 2e-12)
                    np.testing.assert_allclose(result.gates, np.broadcast_to(result.gates[0], result.gates.shape), atol=1e-13)
                    self.assertEqual(result.diagnostics["bdf_solves"], 11-order)

    def test_regularizations_and_finite_field_validation(self):
        mesh = y_graph(2, 4)
        config = SimulationConfig(bdf_order=5, dt=0.01, final_time=0.06, sample_interval=0.01, voltage_limit=1)
        result = IMEXBDFSolver(mesh, config).run(
            initial_voltage=lambda t, e, s: np.full_like(s, 300),
            initial_gates=lambda t, e, s: np.broadcast_to(np.array([0.8, -0.2, 0.3])[:, None], (3, s.size)),
            voltage_source=lambda t, e, s: np.zeros_like(s))
        self.assertGreater(result.diagnostics["gate_mask_steps"], 0)
        self.assertGreater(result.diagnostics["voltage_cutoff_steps"], 0)
        self.assertGreaterEqual(result.diagnostics["min_conductance"], 0.15)
        self.assertTrue(np.all(np.isfinite(result.gates)))
        exact = ManufacturedSolution(mesh, HHParameters())
        arguments = exact.run_arguments()
        arguments["voltage_source"] = lambda t, e, s: np.full_like(s, np.nan)
        with self.assertRaises(ValueError):
            IMEXBDFSolver(mesh, config).run(**arguments)

    def test_nonuniform_multiple_junctions_and_continuous_gate_norm(self):
        errors = []
        for factor in (1, 2):
            mesh = GraphMesh([[0, 0], [1, 0], [2, 0], [1, 1], [3, 1], [2, -1]],
                             [Edge(0, 1), Edge(1, 2, length=1.3), Edge(1, 3), Edge(2, 4), Edge(2, 5)],
                             [factor*n for n in (4, 5, 4, 5, 4)], degree=4)
            exact = ManufacturedSolution(mesh, HHParameters())
            result = IMEXBDFSolver(mesh, SimulationConfig(bdf_order=5, dt=0.005,
                                   final_time=0.1, sample_interval=0.025)).run(**exact.run_arguments())
            errors.append(exact.errors(result))
        for key in ("voltage_error", "gate_error"):
            self.assertGreater(errors[0][key]/errors[1][key], 2**4)
        # Nodal interpolation error is zero initially, but its continuous
        # reconstruction still has spatial error; primary gate L2 includes it.
        initial_gates = exact.gates(0, mesh.node_edge, mesh.node_s)
        self.assertEqual(float(mesh.nodal_norm(initial_gates-initial_gates).max()), 0)
        continuous = mesh.at_quadrature(initial_gates) - exact.gates(0, mesh.rule.edges, mesh.rule.s)
        self.assertGreater(float(np.max(np.abs(continuous))), 1e-8)

    def test_invalid_settings_and_vertex_traces(self):
        for degree in range(1, 5):
            with self.assertRaises(ValueError):
                y_graph(2, degree, 3*degree)
        for order in (0, 6):
            with self.assertRaises(ValueError):
                SimulationConfig(bdf_order=order)
        with self.assertRaises(ValueError):
            IMEXBDFSolver(y_graph(), SimulationConfig(epsilon=0.1))
        with self.assertRaises(ValueError):
            IMEXBDFSolver(y_graph(), SimulationConfig(bdf_order=5, dt=0.05, final_time=0.5, sample_interval=0.05)).run()
        with self.assertRaises(ValueError):
            IMEXBDFSolver(y_graph(), SimulationConfig()).run(
                initial_voltage=lambda t, e, s: e.astype(float),
                initial_gates=lambda t, e, s: np.zeros((3, s.size)))

    def test_save_load_roundtrip_for_every_degree(self):
        for degree in range(1, 5):
            result = IMEXBDFSolver(y_graph(2, degree), SimulationConfig(bdf_order=2,
                                   dt=0.01, final_time=0.05, sample_interval=0.01)).run()
            with tempfile.TemporaryDirectory() as directory:
                result.save(directory)
                loaded = SimulationResult.load(directory)
            np.testing.assert_array_equal(loaded.voltage, result.voltage)
            np.testing.assert_array_equal(loaded.gates, result.gates)
            np.testing.assert_array_equal(loaded.mesh.cells, result.mesh.cells)
            np.testing.assert_allclose(loaded.mesh.node_s, result.mesh.node_s, atol=1e-15)

    def test_bdf3_p2_matches_original_solver(self):
        legacy = Path(__file__).resolve().parents[2] / "python-bdf3-fem"
        if not legacy.is_dir():
            self.skipTest("original solver folder is not present")
        sys.path.append(str(legacy))
        import branched_hh as old
        kwargs = dict(dt=0.01, final_time=0.2, sample_interval=0.05)
        baseline = old.BDF3Solver(old.y_graph(6), old.SimulationConfig(**kwargs)).run()
        result = IMEXBDFSolver(y_graph(6, 2), SimulationConfig(bdf_order=3, **kwargs)).run()
        np.testing.assert_allclose(result.voltage, baseline.voltage, atol=3e-12)
        np.testing.assert_allclose(result.gates, baseline.gates, atol=3e-13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
