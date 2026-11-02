"""Physical probes and time-interpolated HH pulse arrivals and peaks."""

from dataclasses import asdict, dataclass
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq


@dataclass(frozen=True)
class Probe:
    name: str
    edge: int
    s_cm: float
    role: str = "interior"


def y_probes(mesh):
    if len(mesh.edges) != 3:
        raise ValueError("default probes require the three-edge Y graph")
    probes = [Probe(f"trunk_{s:g}", 0, s) for s in (0.25, 0.5, 0.75)]
    probes.append(Probe("junction", 0, float(mesh.edge_lengths[0]), "junction"))
    for eid, name in ((1, "upper"), (2, "lower")):
        probes.extend(Probe(f"{name}_{s:g}", eid, s, "tip" if s == 1 else "interior")
                      for s in (0.25, 0.5, 0.75, 0.9, 1.0))
    return probes


class ProbeRecorder:
    """Evaluate the FE polynomial at fixed arc lengths, without node snapping."""
    def __init__(self, mesh, probes):
        self.probes = tuple(probes)
        cells, basis = [], []
        for probe in probes:
            if not 0 <= probe.edge < len(mesh.edges):
                raise ValueError("probe edge is outside the graph")
            length = mesh.edge_lengths[probe.edge]
            if not 0 <= probe.s_cm <= length + 1e-12:
                raise ValueError("probe arc length is outside its edge")
            count = int(mesh.elements_per_edge[probe.edge])
            local = min(int(probe.s_cm / length * count), count - 1)
            element = int(np.sum(mesh.elements_per_edge[:probe.edge])) + local
            xi = 2*(probe.s_cm-mesh.element_s0[element])/mesh.element_lengths[element]-1
            cells.append(mesh.cells[element])
            basis.append(mesh.basis_at(np.array([xi]))[0])
        self.cells, self.basis = np.asarray(cells), np.asarray(basis)
        self.times, self.voltages = [], []

    def evaluate(self, voltage):
        return np.sum(np.asarray(voltage)[self.cells]*self.basis, axis=1)

    def __call__(self, t, voltage, gates):
        self.times.append(float(t))
        self.voltages.append(self.evaluate(voltage))

    def arrays(self):
        return np.asarray(self.times), np.asarray(self.voltages)

    def metadata(self):
        return [asdict(probe) for probe in self.probes]


def _check_trace(times, values):
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if (times.ndim != 1 or values.shape != times.shape or len(times) < 2
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(values))
            or not np.all(np.diff(times) > 0)):
        raise ValueError("a trace needs matching finite arrays and strictly increasing times")
    return times, values


def _local_polynomial(times, values, center, degree, *, centered=False):
    degree = min(int(degree), len(times)-1)
    count = degree+1
    offset = count//2 if centered else (count-2)//2
    start = max(0, min(center-offset, len(times)-count))
    ids = slice(start, start+count)
    scale = float(times[min(center+1, len(times)-1)]-times[max(center-1, 0)]) / (2 if center else 1)
    x = (times[ids]-times[center])/scale
    polynomial = Polynomial.fit(x, values[ids], degree).convert()
    return polynomial, float(times[center]), scale


def first_upcrossing(times, values, threshold=20.0, order=3, after=0.3):
    """First positive-slope crossing; degree-K interpolation for BDF order K."""
    times, values = _check_trace(times, values)
    if not np.isfinite(threshold) or not 1 <= order <= 5:
        raise ValueError("threshold must be finite and order must be between 1 and 5")
    candidates = np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold)
                               & (times[1:] > after))
    crossings = []
    for j in candidates:
        polynomial, origin, scale = _local_polynomial(times, values, int(j), order)
        left, right = (times[j]-origin)/scale, (times[j+1]-origin)/scale
        method = f"polynomial_degree_{min(order, len(times)-1)}"
        try:
            root = brentq(lambda x: float(polynomial(x)-threshold), left, right,
                           xtol=1e-12, rtol=1e-12)
            slope = float(polynomial.deriv()(root)/scale)
            if slope <= 0:
                raise ValueError("interpolated crossing is not rising")
            arrival = origin+scale*root
        except ValueError:
            slope = float((values[j+1]-values[j])/(times[j+1]-times[j]))
            arrival = float(times[j]+(threshold-values[j])/slope)
            method = "linear_fallback"
        if arrival > after:
            crossings.append((float(arrival), slope, method))
    if crossings:
        arrival, slope, method = crossings[0]
        return dict(arrival_ms=arrival, arrival_slope_mV_per_ms=slope,
                    arrival_method=method, crossing_count=len(crossings), arrival_status="arrived")
    eligible = values[times > after]
    status = "not_reached" if not len(eligible) or max(eligible) < threshold else "unbracketed_crossing"
    return dict(arrival_ms=None, arrival_slope_mV_per_ms=None, arrival_method=None,
                crossing_count=0, arrival_status=status)


def pulse_peak(times, values, order=3, after=0.3):
    """Largest post-stimulus peak, with a stationary-point interpolation."""
    times, values = _check_trace(times, values)
    eligible = np.flatnonzero(times > after)
    if not len(eligible):
        return dict(peak_time_ms=None, peak_voltage_mV=None, peak_method=None, peak_status="outside_window")
    j = int(eligible[np.argmax(values[eligible])])
    if j == 0 or j == len(times)-1 or j == eligible[0]:
        return dict(peak_time_ms=float(times[j]), peak_voltage_mV=float(values[j]),
                    peak_method="sample", peak_status="peak_not_bracketed")
    degree = min(max(2, order), 5, len(times)-1)
    for selected_degree in dict.fromkeys((degree, 2)):
        polynomial, origin, scale = _local_polynomial(times, values, j, selected_degree, centered=True)
        derivative = polynomial.deriv()
        left, right = (times[j-1]-origin)/scale, (times[j+1]-origin)/scale
        # Bracket the derivative root directly. Finding all roots through a
        # companion matrix is inaccurate for a nearly lower-degree fit.
        if derivative(left) >= 0 and derivative(right) <= 0:
            root = brentq(lambda x: float(derivative(x)), left, right,
                           xtol=1e-12, rtol=1e-12)
            if polynomial.deriv(2)(root) >= 0:
                continue
            return dict(peak_time_ms=float(origin+scale*root), peak_voltage_mV=float(polynomial(root)),
                        peak_method=f"polynomial_degree_{selected_degree}", peak_status="complete")
    return dict(peak_time_ms=float(times[j]), peak_voltage_mV=float(values[j]),
                peak_method="sample_fallback", peak_status="complete")


def measure_pulses(times, voltage, probes, order, thresholds=(10, 20, 40, 50), after=0.3):
    voltage = np.asarray(voltage)
    if voltage.shape != (len(times), len(probes)):
        raise ValueError("voltage must have shape (times, probes)")
    rows = []
    for j, probe in enumerate(probes):
        peak = pulse_peak(times, voltage[:, j], order, after)
        for threshold in thresholds:
            event = first_upcrossing(times, voltage[:, j], float(threshold), order, after)
            rows.append(dict(probe=probe.name, edge=probe.edge, s_cm=probe.s_cm, role=probe.role,
                             threshold_mV=float(threshold), **event, **peak))
    return rows


def branch_speeds(events, thresholds=(10, 20, 40, 50)):
    lookup = {(row["probe"], row["threshold_mV"]): row for row in events}
    rows = []
    for name in ("trunk", "upper", "lower"):
        for threshold in thresholds:
            a, b = lookup[(f"{name}_0.25", threshold)], lookup[(f"{name}_0.75", threshold)]
            ta, tb = a["arrival_ms"], b["arrival_ms"]
            complete = ta is not None and tb is not None and tb > ta
            speed = (b["s_cm"]-a["s_cm"])/(tb-ta) if complete else None
            rows.append(dict(branch=name, threshold_mV=float(threshold), s_a_cm=a["s_cm"],
                             s_b_cm=b["s_cm"], arrival_a_ms=ta, arrival_b_ms=tb,
                             speed_cm_per_ms=speed, speed_m_per_s=None if speed is None else 10*speed,
                             status="complete" if complete else "unavailable"))
    return rows
