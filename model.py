"""HH parameters, smooth regularizations, and stimulus shared by all orders."""

from dataclasses import asdict, dataclass
from functools import cached_property
import numpy as np


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
