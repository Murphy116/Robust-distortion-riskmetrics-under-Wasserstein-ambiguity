import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


def normal_mass(a, b):
    return ndtr(-a) - ndtr(-b) if a > 0 else ndtr(b) - ndtr(a)


def normal_density(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


class Benchmark:
    def __init__(self, name, pareto_shape=3):
        self.name = name
        self.pareto_shape = pareto_shape
        if name == "pareto" and pareto_shape <= 2:
            raise ValueError("Pareto shape must exceed 2 for finite variance.")

    def quantile(self, u):
        if self.name == "uniform":
            return np.asarray(u)
        if self.name == "triangular":
            return np.sqrt(np.asarray(u))
        if self.name == "normal":
            return ndtri(u)
        if self.name == "lognormal":
            return np.exp(ndtri(u))
        return np.power(1 - np.asarray(u), -1 / self.pareto_shape)

    def cdf(self, x):
        if self.name == "uniform":
            return np.clip(x, 0, 1)
        if self.name == "triangular":
            return np.clip(x, 0, 1) ** 2
        if self.name == "normal":
            return ndtr(x)
        if self.name == "lognormal":
            return ndtr(np.log(x)) if x > 0 else 0.0
        return 1 - x ** (-self.pareto_shape) if x > 1 else 0.0

    def moment(self, a, b, order=1):
        if b <= a:
            return 0.0
        if order == 0:
            return b - a
        if self.name == "uniform":
            if order == 1:
                return (b - a) * (a + b) / 2
            return (b - a) * (a * a + a * b + b * b) / 3
        if self.name == "triangular":
            exponent = 1 + order / 2
            if a == 0:
                return b ** exponent / exponent
            return a ** exponent * np.expm1(exponent * np.log1p((b - a) / a)) / exponent
        if self.name == "pareto":
            exponent = 1 - order / self.pareto_shape
            if b == 1:
                return (1 - a) ** exponent / exponent
            return ((1 - a) ** exponent
                    * -np.expm1(exponent * np.log1p(-(b - a) / (1 - a))) / exponent)
        za, zb = ndtri([a, b])
        if self.name == "lognormal":
            return np.exp(order * order / 2) * normal_mass(za - order, zb - order)
        if order == 1:
            return normal_density(za) - normal_density(zb)
        za_pdf = za * normal_density(za) if np.isfinite(za) else 0.0
        zb_pdf = zb * normal_density(zb) if np.isfinite(zb) else 0.0
        return b - a + za_pdf - zb_pdf

    def square_cost(self, level, a, b):
        if b <= a:
            return 0.0
        first, second = self.moment(a, b), self.moment(a, b, 2)
        value = level * level * (b - a) - 2 * level * first + second
        scale = abs(level * level * (b - a)) + abs(2 * level * first) + abs(second)
        if self.name == "normal" and (b - a < 1e-4 or value <= 1e-8 * scale):
            # Integrate the positive integrand when moment subtraction loses precision.
            left, right = ndtri([a, b])
            if np.isfinite(left) and np.isfinite(right):
                width = right - left
                return width * quad(
                    lambda t: (level - left - width * t) ** 2
                    * normal_density(left + width * t),
                    0, 1, epsabs=0, epsrel=1e-10)[0]
            return quad(lambda z: (level - z) ** 2 * normal_density(z),
                        left, right, epsabs=0, epsrel=1e-10)[0]
        if self.name == "lognormal" and (b - a < 1e-3 or value <= 1e-6 * scale):
            # X=exp(Z): use a positive integrand, not cancellation of moments.
            left, right = ndtri([a, b])
            if np.isfinite(left) and np.isfinite(right):
                width = right - left
                base = np.exp(left)
                offset = level - base
                return width * quad(
                    lambda t: (offset - base * np.expm1(width * t)) ** 2
                    * normal_density(left + width * t),
                    0, 1, epsabs=0, epsrel=1e-10)[0]
            # Absorb the Gaussian factor before squaring to avoid overflow
            # when an endpoint in the normal coordinate is infinite.
            return quad(lambda z: (level * np.exp(-z * z / 4)
                                   - np.exp(z - z * z / 4)) ** 2 / np.sqrt(2 * np.pi),
                        left, right, epsabs=0, epsrel=1e-10)[0]
        return value

    def expected_shortfall(self, alpha):
        return self.moment(alpha, 1) / (1 - alpha)


def worst_var(benchmark, alpha, radius):
    lower = float(benchmark.quantile(alpha))
    if radius == 0:
        return lower
    upper = benchmark.expected_shortfall(alpha) + radius / np.sqrt(1 - alpha)

    def residual(level):
        end = float(benchmark.cdf(level))
        return benchmark.square_cost(level, alpha, end) - radius * radius

    return brentq(residual, lower, upper, xtol=2e-12, rtol=2e-13)


def rvar_at_shift(benchmark, alpha, beta, shift):
    gb = float(benchmark.quantile(beta))

    def balance(level):
        start = max(alpha, min(beta, float(benchmark.cdf(level - shift))))
        end = max(beta, float(benchmark.cdf(level)))
        return (shift * (beta - start) - level * (end - start)
                + benchmark.moment(start, end))

    level = brentq(balance, gb, gb + shift, xtol=2e-12, rtol=2e-13)
    start = max(alpha, min(beta, float(benchmark.cdf(level - shift))))
    end = max(beta, float(benchmark.cdf(level)))
    cost = shift * shift * (start - alpha) + benchmark.square_cost(level, start, end)
    value = (benchmark.moment(alpha, start) + shift * (start - alpha)
             + level * (beta - start)) / (beta - alpha)
    return value, cost, level, start, end


def worst_rvar(benchmark, alpha, beta, radius):
    upper = radius / np.sqrt(beta - alpha)
    while rvar_at_shift(benchmark, alpha, beta, upper)[1] < radius * radius:
        upper *= 2

    def residual(shift):
        if shift == 0:
            return -radius * radius
        return rvar_at_shift(benchmark, alpha, beta, shift)[1] - radius * radius

    shift = brentq(residual, 0, upper, xtol=2e-12, rtol=2e-13)
    value, cost, level, start, end = rvar_at_shift(benchmark, alpha, beta, shift)
    return value, shift, level, start, end, cost


def isotonic(values, weights, return_blocks=False):
    means = np.empty(len(values))
    masses = np.empty(len(values))
    counts = np.empty(len(values), dtype=int)
    size = 0
    for value, weight in zip(values, weights):
        means[size], masses[size], counts[size] = value, weight, 1
        size += 1
        while size > 1 and means[size - 2] > means[size - 1]:
            mass = masses[size - 2] + masses[size - 1]
            means[size - 2] += ((means[size - 1] - means[size - 2])
                                * masses[size - 1] / mass)
            masses[size - 2] = mass
            counts[size - 2] += counts[size - 1]
            size -= 1
    fitted = np.repeat(means[:size], counts[:size])
    return (fitted, counts[:size].copy()) if return_blocks else fitted


def tk(t, gamma=0.7):
    return t ** gamma / (t ** gamma + (1 - t) ** gamma) ** (1 / gamma)


def tk_derivative(t, gamma=0.7):
    total = t ** gamma + (1 - t) ** gamma
    return (gamma * t ** (gamma - 1) * total
            - t ** gamma * (t ** (gamma - 1) - (1 - t) ** (gamma - 1))) / total ** (1 + 1 / gamma)


def tk_constants(benchmark):
    contact = brentq(lambda t: tk_derivative(t) - (1 - tk(t)) / (1 - t),
                     0.01, 0.7, xtol=2e-14)
    cut = 1 - contact
    slope = (1 - tk(contact)) / cut
    density_norm = np.sqrt(cut * slope ** 2 + quad(
        lambda t: tk_derivative(t) ** 2, 0, contact,
        epsabs=2e-10, epsrel=2e-10, limit=300)[0])
    bound = quad(lambda u: float(benchmark.quantile(u))
                 * (slope - tk_derivative(1 - u)), 0, cut,
                 epsabs=2e-10, epsrel=2e-10, limit=300)[0]
    variance = benchmark.moment(0, cut, 2) - benchmark.moment(0, cut) ** 2 / cut
    return dict(contact=contact, cut=cut, slope=slope, norm=density_norm,
                bound=bound, variance=variance)


def tk_grid(benchmark, cells, cut):
    if not isinstance(cells, (int, np.integer)) or cells < 2:
        raise ValueError("cells must be an integer of at least 2.")
    if not np.isfinite(cut) or not 0 < cut < 1:
        raise ValueError("cut must lie strictly between 0 and 1.")
    sine_squared = np.sin(np.linspace(0, np.pi / 2, cells + 1)) ** 2
    knots = sine_squared ** 2 / (sine_squared ** 2 + (1 - sine_squared) ** 2)
    knots[0], knots[-1] = 0.0, 1.0
    # Move an interior knot only; 0 and 1 must remain probability endpoints.
    knots[1 + np.argmin(abs(knots[1:-1] - cut))] = cut
    weights = np.diff(knots)
    if not np.all(weights > 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-14):
        raise ValueError("Grid masses must be positive and sum to one.")
    means = np.array([benchmark.moment(a, b) / (b - a)
                      for a, b in zip(knots[:-1], knots[1:])])
    density = (tk(1 - knots[:-1]) - tk(1 - knots[1:])) / weights
    envelope, counts = isotonic(density, weights, return_blocks=True)
    concentrated = means.copy()
    start = 0
    for count in counts:
        end = start + count
        if count > 1:
            concentrated[start:end] = np.dot(weights[start:end], means[start:end]) / weights[start:end].sum()
        start = end
    norm = np.sqrt(np.dot(weights, envelope ** 2))
    bound = np.dot(weights, means * (envelope - density))
    return weights, means, density, envelope, concentrated, norm, bound


def grid_optimum(means, density, weights, radius):
    # A zero objective does not require exhausting the transport budget.
    if radius == 0 or np.all(np.asarray(density) == 0):
        return means.copy(), 0.0

    def residual(scale):
        quantile = isotonic(means + scale * density, weights)
        return np.dot(weights, (quantile - means) ** 2) - radius ** 2

    upper = radius / np.sqrt(np.dot(weights, density ** 2))
    while residual(upper) < 0:
        upper *= 2
    scale = brentq(residual, 0, upper, xtol=2e-13, rtol=2e-13)
    return isotonic(means + scale * density, weights), scale


