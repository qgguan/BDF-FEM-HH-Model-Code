"""Continuous manufactured HH equations for independent accuracy studies."""

import numpy as np
from model import gate_rates


class ManufacturedSolution:
    def __init__(self, mesh, parameters, profile="cosine", voltage_limit=200.0):
        if profile not in ("uniform", "cosine"):
            raise ValueError("profile must be uniform or cosine")
        self.mesh, self.parameters = mesh, parameters
        self.profile, self.voltage_limit = profile, voltage_limit

    def _q(self, t, edge, s):
        spatial = (np.ones_like(s) if self.profile == "uniform" else
                   np.cos(2 * np.pi * s / self.mesh.edge_lengths[edge]))
        return np.exp(-t) * spatial

    def voltage(self, t, edge, s):
        return 10 + 3 * self._q(t, edge, s)

    def derivative(self, t, edge, s):
        if self.profile == "uniform":
            return np.zeros_like(s)
        k = 2 * np.pi / self.mesh.edge_lengths[edge]
        return -3 * np.exp(-t) * k * np.sin(k * s)

    def gates(self, t, edge, s):
        q = self._q(t, edge, s)
        return np.stack((0.1 + 0.02*q, 0.6 + 0.03*q, 0.3 + 0.02*q))

    def voltage_source(self, t, edge, s):
        p, q = self.parameters, self._q(t, edge, s)
        k2 = 0.0 if self.profile == "uniform" else (2*np.pi/self.mesh.edge_lengths[edge])**2
        conductance, reversal = p.coefficients(self.gates(t, edge, s))
        return -3*p.capacitance*q + 3*p.diffusion*k2*q + conductance*self.voltage(t, edge, s) - reversal

    def gate_source(self, t, edge, s):
        q = self._q(t, edge, s)
        derivative = np.stack((-0.02*q, -0.03*q, -0.02*q))
        alpha, lam = gate_rates(self.voltage(t, edge, s), self.voltage_limit)
        return derivative - alpha + lam*self.gates(t, edge, s)

    def run_arguments(self):
        return dict(initial_voltage=self.voltage, initial_gates=self.gates,
                    initial_derivative=self.derivative,
                    voltage_source=self.voltage_source, gate_source=self.gate_source)

    def errors(self, result, error_quadrature=None):
        """Max over stored times; continuous L2 gates include interpolation error."""
        mesh = self.mesh
        rule = mesh.integration_rule(error_quadrature or max(40, mesh.quadrature_order + 8))
        errors = dict(voltage_error=0.0, gate_error=0.0,
                      gate_nodal_error=0.0, voltage_h1_error=0.0)
        for t, v, w in zip(result.times, result.voltage, result.gates):
            ev = mesh.at_quadrature(v, rule) - self.voltage(t, rule.edges, rule.s)
            ew = mesh.at_quadrature(w, rule) - self.gates(t, rule.edges, rule.s)
            ed = mesh.at_quadrature(v, rule, derivative=True) - self.derivative(t, rule.edges, rule.s)
            en = w - self.gates(t, mesh.node_edge, mesh.node_s)
            current = dict(voltage_error=np.sqrt(np.sum(rule.weights * ev**2)),
                           gate_error=np.max(np.sqrt(np.sum(rule.weights * ew**2, axis=(-2, -1)))),
                           gate_nodal_error=np.max(mesh.nodal_norm(en)),
                           voltage_h1_error=np.sqrt(np.sum(rule.weights * ed**2)))
            for key in errors:
                errors[key] = max(errors[key], float(current[key]))
        return errors
