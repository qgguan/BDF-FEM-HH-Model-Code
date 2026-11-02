"""Independent checks for the optional mpmath BDF5/P4 experiment."""

import unittest

import numpy as np

try:
    from mpmath import mp
except ImportError:
    mp = None

if mp is not None:
    from studies.manufactured.multiprecision import HHModel, P4Mesh, run


@unittest.skipIf(mp is None, "Install requirements-multiprecision.txt for these optional checks")
class MultiprecisionTests(unittest.TestCase):
    def context(self, digits=50):
        ctx = mp.clone()
        ctx.dps = digits
        return ctx

    def test_sparse_graph_solve_matches_dense_multiprecision_lu(self):
        ctx = self.context()
        mesh = P4Mesh(ctx, 2, reduced=False)
        local = [[m+ctx.mpf("0.2")*k for m, k in zip(mr, kr)]
                 for mr, kr in zip(mesh.mass, mesh.stiffness)]
        rhs = [[ctx.mpf(j+1)/(e+2) for j in range(5)] for e in range(len(mesh.cells))]
        actual = mesh.solve([local]*len(mesh.cells), rhs)
        matrix, load = ctx.matrix(mesh.ndof), ctx.matrix(mesh.ndof, 1)
        for cell, vector in zip(mesh.cells, rhs):
            for i, a in enumerate(cell):
                load[a] += vector[i]
                for j, b in enumerate(cell):
                    matrix[a, b] += local[i][j]
        expected = ctx.lu_solve(matrix, load)
        self.assertLess(max(abs(a-b) for a, b in zip(actual, expected)), ctx.mpf("1e-44"))

    def test_reaction_polynomial_integration_matches_gauss_quadrature(self):
        ctx = self.context()
        mesh, model = P4Mesh(ctx, 8), HHModel(ctx)
        gates = [[base+amplitude*mesh.node_cos[node] for node in mesh.cells[0]]
                 for base, amplitude in zip(model.base, model.amplitude)]
        a, b, masked = mesh.reaction_terms(model, gates)
        aq, bq, masked_q = mesh.reaction_terms(model, gates, force_quadrature=True)
        self.assertEqual(masked+masked_q, 0)
        self.assertLess(max(abs(x-y) for ar, br in zip(a, aq) for x, y in zip(ar, br)), ctx.mpf("1e-45"))
        self.assertLess(max(abs(x-y) for x, y in zip(b, bq)), ctx.mpf("1e-45"))

    def test_constant_projection_accuracy_improves_with_precision(self):
        for digits, tolerance in ((50, "1e-42"), (80, "1e-72")):
            ctx = self.context(digits)
            mesh, model = P4Mesh(ctx, 40), HHModel(ctx)
            one, _ = mesh.projections(model)
            self.assertLess(max(abs(value-1) for value in one), ctx.mpf(tolerance))
            self.assertLess(abs(sum(mesh.nodal_weights)*mesh.norm_factor-3), ctx.mpf(tolerance))
            self.assertEqual(model.rates(ctx.mpf(10))[0][2], ctx.mpf("0.1"))

    def test_half_branch_reduction_matches_full_y_graph(self):
        reduced = run(8, sample_interval="0.25")
        full = run(8, sample_interval="0.25", reduced=False)
        for key in reduced.errors:
            self.assertLess(abs(reduced.errors[key]-full.errors[key]), mp.mpf("1e-40"))
        self.assertEqual(len(reduced.sample_errors), len(full.sample_errors))
        self.assertEqual(full.mesh.ndof, 97)
        self.assertEqual(reduced.mesh.ndof, 17)

    def test_full_multiprecision_run_matches_existing_double_solver(self):
        from mesh import y_graph
        from model import HHParameters
        from solver import IMEXBDFSolver, SimulationConfig
        from manufactured import ManufacturedSolution
        high = run(8, sample_interval="0.25", reduced=False)
        mesh, parameters = y_graph(8, 4, 24), HHParameters()
        exact = ManufacturedSolution(mesh, parameters)
        low = IMEXBDFSolver(mesh, SimulationConfig(bdf_order=5, dt=1/8, final_time=1,
                                                  sample_interval=0.25)).run(**exact.run_arguments())
        np.testing.assert_allclose(np.array([float(x) for x in high.final_voltage]), low.voltage[-1],
                                   rtol=0, atol=3e-12)
        np.testing.assert_allclose(np.array([[float(x) for x in row] for row in high.final_gates]),
                                   low.gates[-1], rtol=0, atol=3e-13)
        for key, error in exact.errors(low).items():
            self.assertLess(abs(float(high.errors[key])-error), 3e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
