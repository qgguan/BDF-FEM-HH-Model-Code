"""Independent mpmath BDF5/P4 verification of the cosine manufactured problem.

All model, FEM, quadrature, history, solve and norm arithmetic uses a private
mpmath context. No NumPy/SciPy arrays or binary64 constants enter the solve.
The optional half-branch reduction is exact for this symmetric experiment:
three identical unit branches, each symmetric about its midpoint, give six
copies of [0, 1/2]. Graph L2 integrals and squared nodal norms are scaled by 6.
The full Y-graph mode is retained for independent symmetry checks.
"""

from collections import deque
from dataclasses import dataclass
import time

from mpmath import mp


ERROR_KEYS = ("voltage_error", "gate_error", "gate_nodal_error", "voltage_h1_error")


def polynomial_product(a, b):
    result = [a[0] * 0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += x*y
    return result


def polynomial_value(coefficients, x):
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value*x + coefficient
    return value


def legendre_rule(ctx, order):
    nodes, weights = ctx.gauss_quadrature(order, "legendre")
    return list(nodes), list(weights)


class HHModel:
    def __init__(self, ctx):
        self.ctx = ctx
        f = ctx.mpf
        self.cm, self.diffusion = f(1), f("0.01")/(2*f("35.4"))
        self.gna, self.gk, self.gl = f(120), f(36), f("0.3")
        self.ena, self.ek = f(115), f(-12)
        self.base = (f("0.1"), f("0.6"), f("0.3"))
        self.amplitude = (f("0.02"), f("0.03"), f("0.02"))
        self.epsilon, self.voltage_limit = f("0.001"), f(200)
        alpha, lam = self.rates(f(0))
        m, h, n = [a/l for a, l in zip(alpha, lam)]
        self.el = (-self.gna*m**3*h*self.ena-self.gk*n**4*self.ek)/self.gl
        self.bdf = (f(137)/60, f(-5), f(5), -f(10)/3, f(5)/4, -f(1)/5)
        self.extrapolation = (f(5), f(-10), f(10), f(-5), f(1))
        gates = [[b, a] for b, a in zip(self.base, self.amplitude)]
        g, reversal = self.coefficient_polynomials(gates)
        source = polynomial_product(g, [f(10), f(3)])
        for j, value in enumerate(reversal):
            source[j] -= value
        source[1] += -3*self.cm+3*self.diffusion*(2*ctx.pi)**2
        self.source_coefficients = source

    def saturation(self, displacement, width):
        u = displacement/width
        return width*self.ctx.tanh(u+u**3/3+u**5/5)

    def mask(self, x):
        if x < 0:
            return self.saturation(x, self.epsilon)
        if x > 1:
            return 1+self.saturation(x-1, self.epsilon)
        return x

    def rates(self, voltage):
        ctx, f = self.ctx, self.ctx.mpf
        if abs(voltage) > self.voltage_limit:
            sign = 1 if voltage > 0 else -1
            voltage = sign*(self.voltage_limit+self.saturation(abs(voltage)-self.voltage_limit, f(1)))

        def vtrap(x):
            return f(10) if x == 0 else x/ctx.expm1(x/10)

        alpha = (vtrap(25-voltage)/10, f("0.07")*ctx.exp(-voltage/20), vtrap(10-voltage)/100)
        beta = (4*ctx.exp(-voltage/18), 1/(ctx.exp((30-voltage)/10)+1),
                ctx.exp(-voltage/80)/8)
        return alpha, tuple(a+b for a, b in zip(alpha, beta))

    def coefficient_polynomials(self, gates):
        m, h, n = gates
        m2, n2 = polynomial_product(m, m), polynomial_product(n, n)
        sodium = [self.gna*x for x in polynomial_product(polynomial_product(m2, m), h)]
        potassium = [self.gk*x for x in polynomial_product(n2, n2)]
        g = [a+b for a, b in zip(sodium, potassium)]
        reversal = [self.ena*a+self.ek*b for a, b in zip(sodium, potassium)]
        g[0] += self.gl
        reversal[0] += self.gl*self.el
        return g, reversal


class SparseFactor:
    """Sparse SPD LDL^T factorization in a prescribed elimination order."""
    def __init__(self, ctx, rows, neighbors):
        self.ctx, self.neighbors = ctx, neighbors
        self.diagonal, self.columns = [], []
        for i, adjacent in enumerate(neighbors):
            diagonal = rows[i][i]
            if diagonal <= 0:
                raise ArithmeticError("Nonpositive pivot in multiprecision SPD solve")
            self.diagonal.append(diagonal)
            column = [rows[i][j]/diagonal for j in adjacent]
            self.columns.append(column)
            for position, (j, multiplier) in enumerate(zip(adjacent, column)):
                for k in adjacent[position:]:
                    rows[j][k] -= multiplier*rows[i][k]

    def solve(self, rhs):
        y = list(rhs)
        for i, (adjacent, column) in enumerate(zip(self.neighbors, self.columns)):
            for j, multiplier in zip(adjacent, column):
                y[j] -= multiplier*y[i]
        x = [self.ctx.mpf(0)]*len(y)
        for i in range(len(y)-1, -1, -1):
            x[i] = y[i]/self.diagonal[i]-self.ctx.fdot(self.columns[i], [x[j] for j in self.neighbors[i]])
        return x


class P4Mesh:
    def __init__(self, ctx, elements, quadrature=24, error_quadrature=40, reduced=True):
        if not isinstance(elements, int) or elements < 2 or elements % 2:
            raise ValueError("elements must be a positive even integer >= 2")
        if quadrature < 13 or error_quadrature < 13:
            raise ValueError("quadrature orders must be >= 13")
        self.ctx, self.elements, self.reduced = ctx, elements, reduced
        self.quadrature_order, self.error_quadrature_order = quadrature, error_quadrature
        f, zero = ctx.mpf, ctx.mpf(0)
        self.h, self.norm_factor = f(1)/elements, f(6 if reduced else 1)
        root = ctx.sqrt(f(3)/7)
        self.nodes = [-f(1), -root, zero, root, f(1)]
        self.lobatto_weights = [f(1)/10, f(49)/90, f(32)/45, f(49)/90, f(1)/10]
        self.polynomials = []
        for i, x in enumerate(self.nodes):
            polynomial, denominator = [f(1)], f(1)
            for j, y in enumerate(self.nodes):
                if i != j:
                    polynomial = polynomial_product(polynomial, [-y, f(1)])
                    denominator *= x-y
            self.polynomials.append([v/denominator for v in polynomial])
        self.derivatives = [[j*p[j] for j in range(1, 5)] for p in self.polynomials]
        self.coefficient_columns = list(zip(*self.polynomials))
        self.cells, self.cell_positions = [], []
        self.node_s = [zero, f("0.5")] if reduced else [zero, f(1), f(1), f(1)]
        edges = [(0, 1)] if reduced else [(0, 1), (1, 2), (1, 3)]
        count = elements//2 if reduced else elements
        for start, end in edges:
            local = [start]
            for e in range(count):
                for r in range(1, 5):
                    s = (f(e)+(self.nodes[r]+1)/2)/elements
                    if e == count-1 and r == 4:
                        node = end
                    else:
                        node = len(self.node_s)
                        self.node_s.append(s)
                    local.append(node)
                self.cells.append(local[4*e:4*e+5])
                self.cell_positions.append(e)
        self.ndof = len(self.node_s)
        self.nodal_weights = [zero]*self.ndof
        for cell in self.cells:
            for node, weight in zip(cell, self.lobatto_weights):
                self.nodal_weights[node] += self.h*weight/2

        # Eliminate the three interior DOFs of every element first. The
        # remaining endpoint graph is a tree, eliminated from leaves to root.
        order = [node for cell in self.cells for node in cell[1:4]]
        endpoints = set(node for cell in self.cells for node in (cell[0], cell[4]))
        adjacency = {node: set() for node in endpoints}
        for cell in self.cells:
            adjacency[cell[0]].add(cell[4])
            adjacency[cell[4]].add(cell[0])
        leaves = deque(sorted(node for node in endpoints if len(adjacency[node]) <= 1))
        removed = set()
        while leaves:
            node = leaves.popleft()
            if node in removed:
                continue
            removed.add(node)
            order.append(node)
            for other in list(adjacency[node]):
                adjacency[other].remove(node)
                if len(adjacency[other]) <= 1:
                    leaves.append(other)
            adjacency[node].clear()
        if len(order) != self.ndof or len(set(order)) != self.ndof:
            raise ValueError("Invalid graph elimination order")
        self.order = order
        self.rank = {node: i for i, node in enumerate(order)}
        self.ranked_cells = [[self.rank[node] for node in cell] for cell in self.cells]
        pattern = [set() for _ in order]
        for cell in self.ranked_cells:
            for i in cell:
                for j in cell:
                    if j > i:
                        pattern[i].add(j)
        self.neighbors = []
        for i, adjacent in enumerate(pattern):
            adjacent = sorted(adjacent)
            self.neighbors.append(adjacent)
            for position, j in enumerate(adjacent):
                pattern[j].update(adjacent[position+1:])

        def integral(poly):
            return ctx.fsum(2*poly[j]/(j+1) for j in range(0, len(poly), 2))

        self.mass = [[self.h*integral(polynomial_product(a, b))/2
                      for b in self.polynomials] for a in self.polynomials]
        self.stiffness = [[2*integral(polynomial_product(a, b))/self.h
                           for b in self.derivatives] for a in self.derivatives]
        self.load_moments = [[self.h*integral([zero]*k+p)/2 for k in range(17)]
                             for p in self.polynomials]
        self.pairs = [(i, j) for i in range(5) for j in range(i, 5)]
        self.reaction_moments = []
        for i, j in self.pairs:
            product = polynomial_product(self.polynomials[i], self.polynomials[j])
            self.reaction_moments.append([self.h*integral([zero]*k+product)/2 for k in range(17)])
        self.quad_nodes, self.quad_weights = legendre_rule(ctx, quadrature)
        self.basis = [[polynomial_value(p, x) for p in self.polynomials] for x in self.quad_nodes]
        self.derivative_basis = [[polynomial_value(p, x) for p in self.derivatives] for x in self.quad_nodes]
        self.error_nodes, self.error_weights = legendre_rule(ctx, error_quadrature)
        self.error_basis = [[polynomial_value(p, x) for p in self.polynomials] for x in self.error_nodes]
        self.error_derivatives = [[2*polynomial_value(p, x)/self.h for p in self.derivatives] for x in self.error_nodes]
        self.node_cos = [self.cosine(s) for s in self.node_s]
        self.source_moments, self.ritz_cos_rhs = [], []
        self.error_cos, self.error_derivative = [], []
        for e in range(count):
            spatial_cos = [self.cosine((f(e)+(x+1)/2)/elements) for x in self.quad_nodes]
            spatial_derivative = [-2*ctx.pi*ctx.sin(2*ctx.pi*(f(e)+(x+1)/2)/elements) for x in self.quad_nodes]
            self.source_moments.append([
                [self.h*ctx.fsum(w*b[i]*c**k for w, b, c in zip(self.quad_weights, self.basis, spatial_cos))/2
                 for k in range(6)] for i in range(5)])
            self.ritz_cos_rhs.append((
                [self.h*ctx.fsum(w*b[i]*c for w, b, c in zip(self.quad_weights, self.basis, spatial_cos))/2 for i in range(5)],
                [ctx.fsum(w*b[i]*d for w, b, d in zip(self.quad_weights, self.derivative_basis, spatial_derivative)) for i in range(5)]))
            spatial_s = [(f(e)+(x+1)/2)/elements for x in self.error_nodes]
            self.error_cos.append([self.cosine(s) for s in spatial_s])
            self.error_derivative.append([-2*ctx.pi*ctx.sin(2*ctx.pi*s) for s in spatial_s])

    def cosine(self, s):
        if s == 0 or s == 1:
            return self.ctx.mpf(1)
        if s == self.ctx.mpf("0.5"):
            return self.ctx.mpf(-1)
        return self.ctx.cos(2*self.ctx.pi*s)

    def factor(self, local_matrices):
        rows = [{i: self.ctx.mpf(0), **{j: self.ctx.mpf(0) for j in adjacent}}
                for i, adjacent in enumerate(self.neighbors)]
        for cell, matrix in zip(self.ranked_cells, local_matrices):
            for i, j in self.pairs:
                a, b = sorted((cell[i], cell[j]))
                rows[a][b] += matrix[i][j]
        return SparseFactor(self.ctx, rows, self.neighbors)

    def assemble_rhs(self, local_rhs):
        rhs = [self.ctx.mpf(0)]*self.ndof
        for cell, values in zip(self.ranked_cells, local_rhs):
            for node, value in zip(cell, values):
                rhs[node] += value
        return rhs

    def unpermute(self, x):
        result = [self.ctx.mpf(0)]*self.ndof
        for node, value in zip(self.order, x):
            result[node] = value
        return result

    def solve(self, local_matrices, local_rhs):
        return self.unpermute(self.factor(local_matrices).solve(self.assemble_rhs(local_rhs)))

    def projections(self, model):
        matrix = [[model.diffusion*k+m for k, m in zip(kr, mr)]
                  for kr, mr in zip(self.stiffness, self.mass)]
        factor = self.factor([matrix]*len(self.cells))
        constant_rhs = [row[0] for row in self.load_moments]
        projected_one = self.unpermute(factor.solve(self.assemble_rhs([constant_rhs]*len(self.cells))))
        cosine_rhs = [[a+model.diffusion*b for a, b in zip(*self.ritz_cos_rhs[e])]
                      for e in self.cell_positions]
        projected_cos = self.unpermute(factor.solve(self.assemble_rhs(cosine_rhs)))
        return projected_one, projected_cos

    def reaction_terms(self, model, gate_values, force_quadrature=False):
        ctx, zero = self.ctx, self.ctx.mpf(0)
        polynomials = [[ctx.fdot(values, column) for column in self.coefficient_columns] for values in gate_values]
        bounded = all(0 <= p[0]-ctx.fsum(abs(x) for x in p[1:]) and
                      p[0]+ctx.fsum(abs(x) for x in p[1:]) <= 1 for p in polynomials)
        if bounded and not force_quadrature:
            g, reversal = model.coefficient_polynomials(polynomials)
            upper = [ctx.fdot(g, moments) for moments in self.reaction_moments]
            rhs = [ctx.fdot(reversal, moments) for moments in self.load_moments]
            masked = 0
        else:
            coefficients, reversals, masked = [], [], 0
            for basis in self.basis:
                gates = [ctx.fdot(values, basis) for values in gate_values]
                masked += sum(x < 0 or x > 1 for x in gates)
                m, h, n = [model.mask(x) for x in gates]
                sodium, potassium = model.gna*m**3*h, model.gk*n**4
                coefficients.append(sodium+potassium+model.gl)
                reversals.append(sodium*model.ena+potassium*model.ek+model.gl*model.el)
            upper = [self.h*ctx.fsum(w*g*b[i]*b[j] for w, g, b in zip(self.quad_weights, coefficients, self.basis))/2
                     for i, j in self.pairs]
            rhs = [self.h*ctx.fsum(w*f*b[i] for w, f, b in zip(self.quad_weights, reversals, self.basis))/2 for i in range(5)]
        matrix = [[zero]*5 for _ in range(5)]
        for (i, j), value in zip(self.pairs, upper):
            matrix[i][j] = matrix[j][i] = value
        return matrix, rhs, masked

    def errors(self, voltage, gates, decay, model):
        ctx, zero = self.ctx, self.ctx.mpf(0)
        ev, ed, ew = zero, zero, [zero]*3
        for cell, e in zip(self.cells, self.cell_positions):
            v = [voltage[node] for node in cell]
            w = [[gate[node] for node in cell] for gate in gates]
            for weight, basis, derivative, cosine, exact_derivative in zip(
                    self.error_weights, self.error_basis, self.error_derivatives,
                    self.error_cos[e], self.error_derivative[e]):
                q = decay*cosine
                ev += weight*(ctx.fdot(v, basis)-(10+3*q))**2
                ed += weight*(ctx.fdot(v, derivative)-3*decay*exact_derivative)**2
                for j in range(3):
                    ew[j] += weight*(ctx.fdot(w[j], basis)-(model.base[j]+model.amplitude[j]*q))**2
        en = [ctx.fsum(weight*(value-(base+amplitude*decay*cosine))**2
                       for weight, value, cosine in zip(self.nodal_weights, gate, self.node_cos))
              for gate, base, amplitude in zip(gates, model.base, model.amplitude)]
        scale = self.norm_factor*self.h/2
        return dict(voltage_error=ctx.sqrt(scale*ev), gate_error=ctx.sqrt(scale*max(ew)),
                    gate_nodal_error=ctx.sqrt(self.norm_factor*max(en)), voltage_h1_error=ctx.sqrt(scale*ed))


@dataclass
class MPResult:
    mesh: P4Mesh
    errors: dict
    diagnostics: dict
    sample_errors: list
    final_voltage: list
    final_gates: list


def run(elements, *, digits=50, final_time="1", sample_interval="0.05",
        quadrature=24, error_quadrature=40, reduced=True, progress=False):
    if not isinstance(digits, int) or digits < 25:
        raise ValueError("digits must be an integer >= 25")
    ctx = mp.clone()
    ctx.dps = digits
    started = time.perf_counter()
    mesh = P4Mesh(ctx, elements, quadrature, error_quadrature, reduced)
    model = HHModel(ctx)
    dt, end, interval = ctx.mpf(1)/elements, ctx.mpf(final_time), ctx.mpf(sample_interval)
    ratios = (end/dt, interval/dt)
    nsteps, stride = [int(ctx.nint(ratio)) for ratio in ratios]
    if (nsteps < 5 or stride < 1 or any(abs(ratio-n) > 100*ctx.eps*max(1, abs(ratio))
                                     for ratio, n in zip(ratios, (nsteps, stride)))):
        raise ValueError("T and sample interval must be positive multiples of dt, with T >= 5*dt")
    one, cosine = mesh.projections(model)
    errors = {key: ctx.mpf(0) for key in ERROR_KEYS}
    samples = []

    def sample(step, voltage, gates):
        if step % stride == 0 or step == nsteps:
            current = mesh.errors(voltage, gates, ctx.exp(-step*dt), model)
            for key in ERROR_KEYS:
                errors[key] = max(errors[key], current[key])
            samples.append(dict(time=step*dt, **current))

    vh, wh = [], []
    for step in range(5):
        decay = ctx.exp(-step*dt)
        v = [10*a+3*decay*b for a, b in zip(one, cosine)]
        w = [[base+amplitude*decay*c for c in mesh.node_cos]
             for base, amplitude in zip(model.base, model.amplitude)]
        vh.append(v)
        wh.append(w)
        sample(step, v, w)
    vh.reverse()
    wh.reverse()
    constant_matrix = [[model.bdf[0]*model.cm*m/dt+model.diffusion*k for m, k in zip(mr, kr)]
                       for mr, kr in zip(mesh.mass, mesh.stiffness)]
    masks, cutoffs = 0, 0
    for step in range(5, nsteps+1):
        decay = ctx.exp(-step*dt)
        source_coefficients = [coefficient*decay**j for j, coefficient in enumerate(model.source_coefficients)]
        extrapolated = [[ctx.fdot(model.extrapolation, [history[j][node] for history in wh])
                         for node in range(mesh.ndof)] for j in range(3)]
        history_v = [-model.cm*ctx.fdot(model.bdf[1:], [history[node] for history in vh])/dt
                     for node in range(mesh.ndof)]
        local_matrices, local_rhs = [], []
        for cell, e in zip(mesh.cells, mesh.cell_positions):
            reaction, reversal, masked = mesh.reaction_terms(model, [[gate[node] for node in cell] for gate in extrapolated])
            masks += masked
            local_matrices.append([[a+b for a, b in zip(ar, br)] for ar, br in zip(constant_matrix, reaction)])
            history = [history_v[node] for node in cell]
            local_rhs.append([ctx.fdot(mass, history)+force+ctx.fdot(moment, source_coefficients)
                              for mass, force, moment in zip(mesh.mass, reversal, mesh.source_moments[e])])
        v = mesh.solve(local_matrices, local_rhs)
        w = [[ctx.mpf(0)]*mesh.ndof for _ in range(3)]
        for node, (voltage, spatial_cos) in enumerate(zip(v, mesh.node_cos)):
            cutoffs += int(abs(voltage) > model.voltage_limit)
            q = decay*spatial_cos
            alpha, lam = model.rates(voltage)
            alpha_exact, lam_exact = model.rates(10+3*q)
            for j, (base, amplitude) in enumerate(zip(model.base, model.amplitude)):
                source = -amplitude*q-alpha_exact[j]+lam_exact[j]*(base+amplitude*q)
                history = -ctx.fdot(model.bdf[1:], [old[j][node] for old in wh])/dt
                w[j][node] = (history+alpha[j]+source)/(model.bdf[0]/dt+lam[j])
        vh, wh = [v]+vh[:4], [w]+wh[:4]
        sample(step, v, w)
        if progress and (step % max(1, nsteps//10) == 0 or step == nsteps):
            print(f"  {digits} digits N={elements}: t={ctx.nstr(step*dt, 5)}/{ctx.nstr(end, 5)} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    constant_error = ctx.sqrt(mesh.norm_factor*ctx.fsum(weight*(10*value-10)**2
                                                      for weight, value in zip(mesh.nodal_weights, one)))
    return MPResult(mesh, errors, dict(
        digits=digits, elements_per_edge=elements, dt=dt, h=mesh.h, final_time=end,
        sample_interval=interval, samples=len(samples), steps=nsteps,
        full_graph_dofs=12*elements+1, solved_dofs=mesh.ndof,
        symmetry_reduced=reduced, norm_factor=mesh.norm_factor,
        masked_quadrature_values=masks, cutoff_nodal_values=cutoffs,
        constant_projection_nodal_error=constant_error, epsilon=ctx.eps,
        leak_reversal=model.el, elapsed_seconds=time.perf_counter()-started),
        samples, v, w)
