from pathlib import Path
import argparse
import csv
import json

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri

from portfolio_experiment import covariance, simplex
from risk_numerics import Benchmark, grid_optimum, normal_density, worst_rvar


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results'
ALPHA = 0.75
CORRELATION = 0.3
RADII = (0.005, 0.05, 0.20)
ETAS = (0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
SCALAR_NODES = 961
NORMAL = Benchmark('normal')


def save_csv(name, rows):
    DATA.mkdir(exist_ok=True)
    with (DATA / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    with (DATA / name).open(newline='', encoding='utf-8') as stream:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def upper_cost(level):
    level = np.asarray(level, dtype=float)
    z = ndtri(ALPHA)
    mass = ndtr(level) - ALPHA
    value = ((level * level + 1) * mass + (z - 2 * level) * normal_density(z)
             + level * normal_density(level))
    delta = level - z
    small = (delta > 0) & (delta < 1e-3)
    if np.any(small):
        value = np.array(value, copy=True)
        for index in zip(*np.nonzero(small)) if level.ndim else [()]:
            d = float(delta[index])
            # x = z + d*t avoids cancellation in the short-interval cost.
            value[index] = d ** 3 * quad(
                lambda t: (1 - t) ** 2 * normal_density(z + d * t),
                0, 1, epsabs=0, epsrel=1e-12)[0]
    return np.where(delta <= 0, 0.0, value)


def original_scalar(radius):
    radius = np.asarray(radius, dtype=float)
    lower = np.full_like(radius, ndtri(ALPHA))
    upper = (normal_density(ndtri(ALPHA)) / (1 - ALPHA)
             + radius / np.sqrt(2 * (1 - ALPHA)))
    for _ in range(65):
        middle = (lower + upper) / 2
        below = upper_cost(middle) < radius * radius / 2
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return np.where(radius == 0, 2 * ndtri(ALPHA), lower + upper)


def regularized_scalar(eta, radius):
    if radius == 0:
        return 2 * NORMAL.moment(ALPHA, ALPHA + eta) / eta
    # Symmetry gives equal transport costs in the two tails.
    return 2 * worst_rvar(NORMAL, ALPHA, ALPHA + eta, radius / np.sqrt(2))[0]


def geometry(weights):
    sigma = covariance(CORRELATION)
    scales = np.sqrt(np.einsum('ij,jk,ik->i', weights, sigma, weights))
    norms = np.linalg.norm(weights, axis=1)
    return scales, norms, float(np.sqrt(np.trace(sigma)))


def build_scalar_tables(reuse=False):
    assert 0.5 < ALPHA < 1 and max(ETAS) < (1 - ALPHA) / 2
    weights = simplex(400)
    scales, norms, trace = geometry(weights)
    lower = 0.99 * min(RADII) * trace * np.min(norms / scales)
    upper = 1.01 * max(RADII) * trace * np.max(norms / scales)
    nodes = np.geomspace(lower, upper, SCALAR_NODES)
    if reuse:
        records = read_csv('iqr_scalar.csv')
        assert {row['eta'] for row in records} == set(ETAS)
        assert all(row['alpha'] == ALPHA for row in records)
    else:
        records = []
        for eta in ETAS:
            for radius in nodes:
                records.append(dict(alpha=ALPHA, eta=eta, radius=radius,
                                    value=regularized_scalar(eta, radius)))
            print(f'IQR scalar values: eta={eta:g}', flush=True)
        save_csv('iqr_scalar.csv', records)
    tables = {}
    for eta in ETAS:
        rows = [row for row in records if row['eta'] == eta]
        assert len(rows) == len(nodes)
        x = np.array([row['radius'] for row in rows])
        assert x[0] <= lower + 1e-12 and x[-1] >= upper - 1e-12
        tables[eta] = PchipInterpolator(x, [row['value'] for row in rows], extrapolate=False)
    return tables


def decisions(tables, denominator=200):
    weights = simplex(denominator)
    scales, norms, trace = geometry(weights)
    records = []
    for c in RADII:
        epsilon = c * trace
        radii = epsilon * norms / scales
        original = scales * original_scalar(radii)
        original_index = int(np.argmin(original))
        for eta in ETAS + (0.0,):
            interpolation_error = 0.0
            if eta == 0:
                index, value = original_index, float(original[original_index])
            else:
                estimates = scales * tables[eta](radii)
                assert np.all(np.isfinite(estimates))
                candidates = np.argpartition(estimates, 32)[:32]
                exact = np.array([scales[i] * regularized_scalar(eta, radii[i]) for i in candidates])
                interpolation_error = float(np.max(abs(exact - estimates[candidates])))
                index = int(candidates[np.argmin(exact)])
                value = float(np.min(exact))
                assert interpolation_error < 2e-8
                # S is nondecreasing in radius. A left node therefore bounds
                # every portfolio from below, unlike the top-32 interpolation errors.
                nodes = tables[eta].x
                left = np.searchsorted(nodes, radii, side='right') - 1
                lower_bounds = scales * tables[eta](nodes[left])
                unchecked = lower_bounds <= value + 2e-10
                unchecked[candidates] = False
                for i in np.flatnonzero(unchecked):
                    direct = scales[i] * regularized_scalar(eta, radii[i])
                    if direct < value - 2e-10:
                        raise RuntimeError(
                            'Top-32 selection missed a grid minimum; increase the '
                            'candidate count or refine the scalar grid before reporting results.')
            w = weights[index]
            assert abs(w.sum() - 1) < 1e-12 and np.min(w) >= 0
            assert value >= original[index] - 2e-10
            records.append(dict(c=c, epsilon=epsilon, alpha=ALPHA, correlation=CORRELATION,
                                eta=eta, weight1=w[0], weight2=w[1], weight3=w[2],
                                value=value, original_at_weights=float(original[index]),
                                original_minimum=float(original[original_index]),
                                scalar_radius=float(radii[index]), portfolio_scale=float(scales[index]),
                                weight_step=1 / denominator,
                                interpolation_error=interpolation_error))
        group = [row for row in records if row['c'] == c]
        assert np.max(np.diff([row['value'] for row in group])) <= 2e-10
        assert all(row['value'] >= row['original_minimum'] - 2e-10 for row in group)
    return records


def quantile_grid(eta, cells):
    band_cells = cells // 64
    outer_cells = (cells - 2 * band_cells) // 4
    middle_cells = cells - 2 * outer_cells - 2 * band_cells
    breaks = (0, 1 - ALPHA - eta, 1 - ALPHA, ALPHA, ALPHA + eta, 1)
    counts = (outer_cells, band_cells, middle_cells, band_cells, outer_cells)
    knots = np.concatenate([np.linspace(a, b, n + 1)[:-1]
                            for a, b, n in zip(breaks[:-1], breaks[1:], counts)] + [np.array([1.0])])
    mass = np.diff(knots)
    means = np.array([NORMAL.moment(a, b) / (b - a) for a, b in zip(knots[:-1], knots[1:])])
    mid = (knots[:-1] + knots[1:]) / 2
    density = (np.asarray((mid > ALPHA) & (mid < ALPHA + eta), dtype=float)
               - np.asarray((mid > 1 - ALPHA - eta) & (mid < 1 - ALPHA), dtype=float)) / eta
    assert len(mass) == cells and np.min(mass) > 0
    assert abs(np.dot(mass, density)) < 1e-12
    assert abs(np.dot(mass, abs(density)) - 2) < 1e-12
    return means, mass, density


def verify(tables, records):
    grid_checks = []
    for eta in ETAS:
        for cells in (4000, 8000):
            means, mass, density = quantile_grid(eta, cells)
            for row in (r for r in records if r['eta'] == eta):
                radius = row['scalar_radius']
                quantile, shift = grid_optimum(means, density, mass, radius)
                cost = float(np.dot(mass, (quantile - means) ** 2))
                residual = means + shift * density - quantile
                cumulative = np.cumsum(mass * residual)
                kkt_error = max(abs(cumulative[-1]), max(0, -np.min(cumulative)),
                                abs(np.dot(mass * residual, quantile)))
                assert np.min(np.diff(quantile)) >= -1e-11
                assert abs(np.sqrt(cost) - radius) < 2e-9 and kkt_error < 2e-9
                value = row['portfolio_scale'] * float(np.dot(mass * density, quantile))
                grid_checks.append(dict(c=row['c'], eta=eta, cells=cells,
                                        band_cells=cells // 64, value=value,
                                        continuous_value=row['value'],
                                        absolute_error=abs(value - row['value']),
                                        radius_error=abs(np.sqrt(cost) - radius), kkt_error=kkt_error))
        print(f'IQR grid checks: eta={eta:g}', flush=True)
    save_csv('iqr_grid_checks.csv', grid_checks)
    fine = decisions(tables, 400)
    weight_checks = []
    for row in fine:
        coarse = next(r for r in records if r['c'] == row['c'] and r['eta'] == row['eta'])
        weight_checks.append(dict(c=row['c'], eta=row['eta'],
                                  weight1=row['weight1'], weight2=row['weight2'], weight3=row['weight3'],
                                  coarse_value=coarse['value'], fine_value=row['value'],
                                  value_change=abs(row['value'] - coarse['value']),
                                  maximum_weight_change=max(abs(row[f'weight{i}'] - coarse[f'weight{i}'])
                                                            for i in (1, 2, 3))))
    save_csv('iqr_weight_checks.csv', weight_checks)
    direct_checks = []
    for row in (r for r in records if r['eta'] == 0):
        radius = row['scalar_radius']
        level = float(original_scalar(radius)) / 2
        integral = quad(lambda z: (level - z) ** 2 * normal_density(z),
                        ndtri(ALPHA), level, epsabs=1e-13, epsrel=1e-13)[0]
        error = abs(2 * integral - radius ** 2)
        assert error < 2e-12
        z = ndtri(ALPHA)
        area = level * (ndtr(level) - ALPHA) - normal_density(z) + normal_density(level)
        threshold = area / (level - z)
        recovery_error = max(abs(regularized_scalar(eta, radius) - 2 * level)
                             for eta in ETAS if eta <= threshold)
        assert recovery_error < 2e-9
        direct_checks.append(dict(c=row['c'], radius=radius, lower_quantile=-level,
                                  upper_quantile=level, total_squared_cost=2 * integral,
                                  squared_radius=radius ** 2, cost_error=error,
                                  recovery_eta_threshold=threshold,
                                  recovery_scalar_error=recovery_error))
    save_csv('iqr_reference_checks.csv', direct_checks)
    changes = [abs(row['value'] - next(r['value'] for r in grid_checks
                                     if r['c'] == row['c'] and r['eta'] == row['eta'] and r['cells'] == 4000))
               for row in grid_checks if row['cells'] == 8000]
    return dict(alpha=ALPHA, correlation=CORRELATION, c_values=RADII, eta_values=ETAS,
                scalar_nodes=SCALAR_NODES, coarse_weight_step=0.005, fine_weight_step=0.0025,
                quantile_cells=[4000, 8000],
                maximum_interpolation_error=max(r['interpolation_error'] for r in records + fine),
                maximum_grid_refinement_change=max(changes),
                maximum_grid_error_4000=max(r['absolute_error'] for r in grid_checks if r['cells'] == 4000),
                maximum_grid_error_8000=max(r['absolute_error'] for r in grid_checks if r['cells'] == 8000),
                maximum_weight_refinement_value_change=max(r['value_change'] for r in weight_checks),
                maximum_weight_change=max(r['maximum_weight_change'] for r in weight_checks),
                maximum_radius_error=max(r['radius_error'] for r in grid_checks),
                maximum_kkt_error=max(r['kkt_error'] for r in grid_checks),
                maximum_reference_cost_error=max(r['cost_error'] for r in direct_checks),
                maximum_recovery_scalar_error=max(r['recovery_scalar_error'] for r in direct_checks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-data', action='store_true')
    parser.add_argument('--skip-verification', action='store_true')
    args = parser.parse_args()
    tables = build_scalar_tables(args.reuse_data)
    records = decisions(tables)
    save_csv('iqr_portfolio.csv', records)
    if not args.skip_verification:
        report = verify(tables, records)
        (DATA / 'iqr_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    for row in records:
        w = tuple(row[f'weight{i}'] for i in (1, 2, 3))
        print(f"c={row['c']:.3f}, eta={row['eta']:g}, weights={w}, value={row['value']:.9f}", flush=True)


if __name__ == '__main__':
    main()
