from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np
from scipy.optimize import brentq

from risk_numerics import Benchmark, grid_optimum, isotonic, tk, tk_derivative


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results'
MU = np.array([0.080, 0.075, 0.070])
SIGMA = np.array([0.12, 0.20, 0.32])
CORRELATIONS = (0.0, 0.3, 0.6, 0.9)
RADII = np.linspace(0, 0.20, 41)
METHODS = ('original', 'envelope', 'candidate')


def save_csv(name, rows):
    DATA.mkdir(exist_ok=True)
    with (DATA / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    with (DATA / name).open(newline='', encoding='utf-8') as stream:
        return [{key: value if key == 'method' else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def simplex(denominator=200):
    return np.array([(i, j, denominator - i - j)
                     for i in range(denominator + 1)
                     for j in range(denominator - i + 1)], dtype=float) / denominator


def covariance(correlation):
    corr = np.full((3, 3), correlation)
    np.fill_diagonal(corr, 1.0)
    return SIGMA[:, None] * corr * SIGMA[None, :]


class ScalarModel:
    def __init__(self, cells=4000):
        self.cells = cells
        self.cut = brentq(lambda u: tk_derivative(u) - tk(u) / u, 0.05, 0.99)
        sine = np.sin(np.linspace(0, np.pi / 2, cells + 1)) ** 2
        knots = sine ** 2 / (sine ** 2 + (1 - sine) ** 2)
        knots[0], knots[-1] = 0.0, 1.0
        knots[np.argmin(abs(knots - self.cut))] = self.cut
        self.mass = np.diff(knots)
        benchmark = Benchmark('normal')
        self.quantile = np.array([benchmark.moment(a, b) / (b - a)
                                  for a, b in zip(knots[:-1], knots[1:])])
        self.density = np.diff(tk(knots)) / self.mass
        self.envelope, counts = isotonic(self.density, self.mass, return_blocks=True)
        self.concentrated = self.quantile.copy()
        start = 0
        for count in counts:
            end = start + count
            if count > 1:
                self.concentrated[start:end] = np.dot(
                    self.mass[start:end], self.quantile[start:end]) / self.mass[start:end].sum()
            start = end
        self.nominal = float(np.dot(self.mass, self.quantile * self.density))
        self.nominal_envelope = float(np.dot(self.mass, self.quantile * self.envelope))
        self.norm = float(np.sqrt(np.dot(self.mass, self.envelope ** 2)))
        self.variance = float(np.dot(self.mass, (self.concentrated - self.quantile) ** 2))
        self.benchmark_gap = self.nominal_envelope - self.nominal

    def exact(self, radius, check=False):
        q, scale = grid_optimum(self.quantile, self.density, self.mass, radius)
        if check:
            assert np.min(np.diff(q)) >= -1e-10
            assert abs(np.sqrt(np.dot(self.mass, (q - self.quantile) ** 2)) - radius) < 2e-8
            residual = self.quantile + scale * self.density - q
            cumulative = np.cumsum(self.mass * residual)
            assert cumulative.min() >= -2e-8
            assert abs(cumulative[-1]) < 2e-8
            assert abs(np.dot(self.mass * residual, q)) < 2e-8
        return float(np.dot(self.mass * self.density, q))

    def candidate(self, radius):
        radius = np.asarray(radius)
        return (self.nominal + radius * (self.benchmark_gap + radius * self.norm)
                / np.sqrt(self.variance + radius ** 2))

    def prepare(self):
        started = time.perf_counter()
        self.radii = np.linspace(0, 1.2, 481)
        self.values = np.array([self.exact(radius) for radius in self.radii])
        self.precompute_seconds = time.perf_counter() - started
        save_csv('portfolio_scalar.csv', [dict(radius=r, value=v)
                                           for r, v in zip(self.radii, self.values)])
        print(f'Scalar calculation: {self.precompute_seconds:.2f} seconds', flush=True)

    def lookup(self, radii):
        if np.max(radii) > self.radii[-1] + 1e-12:
            raise ValueError('Increase the scalar-radius grid before running this scenario.')
        return np.interp(radii, self.radii, self.values)


def geometry(weights, mean, cov):
    return (weights @ mean, np.sqrt(np.einsum('ij,jk,ik->i', weights, cov, weights)),
            np.linalg.norm(weights, axis=1))


def objectives(model, means, scales, norms, epsilon):
    radius = epsilon * norms / scales
    return {
        'original': means + scales * model.lookup(radius),
        'envelope': means + scales * model.nominal_envelope + epsilon * model.norm * norms,
        'candidate': means + scales * model.candidate(radius),
    }


def decision_records(model, weights, correlations=CORRELATIONS, radii=RADII):
    rows = []
    for correlation in correlations:
        cov = covariance(correlation)
        means, scales, norms = geometry(weights, MU, cov)
        for c in radii:
            epsilon = c * np.sqrt(np.trace(cov))
            values = objectives(model, means, scales, norms, epsilon)
            indices = {method: int(np.argmin(values[method])) for method in METHODS}
            optimum = values['original'][indices['original']]
            lower_minimum = np.min(means + model.nominal * scales + epsilon * model.norm * norms)
            for method in METHODS:
                index = indices[method]
                gap = values['original'][index] - optimum
                if method == 'envelope':
                    bound = values['envelope'][index] - lower_minimum
                elif method == 'candidate':
                    bound = values['envelope'][index] - values['candidate'][index]
                else:
                    bound = 0.0
                assert gap >= -1e-10
                assert gap <= bound + 2e-7
                rows.append(dict(correlation=correlation, c=c, epsilon=epsilon, method=method,
                                 weight1=weights[index, 0], weight2=weights[index, 1],
                                 weight3=weights[index, 2], value=values['original'][index],
                                 optimum=optimum, decision_loss=gap, upper_bound=bound))
        print(f'Portfolio decisions: correlation={correlation}', flush=True)
    return rows


def verify(model, weights, records):
    fine_model = ScalarModel(8000)
    scalar_checks = []
    selected = [r for r in records if any(abs(r['c'] - c) < 1e-12 for c in (0.05, 0.10, 0.20))]
    for row in selected:
        w = np.array([row['weight1'], row['weight2'], row['weight3']])
        scale = np.sqrt(w @ covariance(row['correlation']) @ w)
        radius = row['epsilon'] * np.linalg.norm(w) / scale
        coarse_value = model.exact(radius, check=True)
        fine_value = fine_model.exact(radius, check=True)
        direction = model.concentrated - model.quantile + radius * model.envelope / model.norm
        candidate = model.quantile + radius * direction / np.sqrt(np.dot(model.mass, direction ** 2))
        assert np.min(np.diff(candidate)) >= -1e-9
        assert abs(np.sqrt(np.dot(model.mass, (candidate - model.quantile) ** 2)) - radius) < 2e-9
        assert abs(np.dot(model.mass * model.density, candidate) - model.candidate(radius)) < 2e-9
        scalar_checks.append(dict(correlation=row['correlation'], c=row['c'], method=row['method'],
                                  radius=radius,
                                  interpolation_error=scale * abs(model.lookup(radius) - coarse_value),
                                  inner_refinement_error=scale * abs(fine_value - coarse_value),
                                  candidate_refinement_error=scale * abs(
                                      fine_model.candidate(radius) - model.candidate(radius))))
    save_csv('portfolio_inner_checks.csv', scalar_checks)
    fine_records = decision_records(model, simplex(400), radii=(0.05, 0.10, 0.20))
    refinement = []
    for row in fine_records:
        coarse = next(r for r in records if r['correlation'] == row['correlation']
                      and r['method'] == row['method'] and abs(r['c'] - row['c']) < 1e-12)
        refinement.append(dict(correlation=row['correlation'], c=row['c'], method=row['method'],
                               maximum_weight_change=max(abs(row[key] - coarse[key])
                                                         for key in ('weight1', 'weight2', 'weight3')),
                               decision_loss_change=abs(row['decision_loss'] - coarse['decision_loss'])))
    save_csv('portfolio_outer_checks.csv', refinement)
    max_interpolation = max(r['interpolation_error'] for r in scalar_checks)
    max_inner = max(r['inner_refinement_error'] for r in scalar_checks)
    max_candidate = max(r['candidate_refinement_error'] for r in scalar_checks)
    assert max_interpolation < 2e-6
    assert max_inner < 2e-6
    assert max_candidate < 2e-6
    means, scales, norms = geometry(weights, MU, covariance(0.3))
    epsilon = 0.1 * np.sqrt(np.trace(covariance(0.3)))
    timing = {}
    for method in METHODS:
        durations = []
        for _ in range(50):
            started = time.perf_counter()
            if method == 'original':
                values = means + scales * model.lookup(epsilon * norms / scales)
            elif method == 'envelope':
                values = means + scales * model.nominal_envelope + epsilon * model.norm * norms
            else:
                values = means + scales * model.candidate(epsilon * norms / scales)
            np.argmin(values)
            durations.append(time.perf_counter() - started)
        timing[method] = float(np.median(durations))
    report = dict(cells=4000, fine_cells=8000, mesh_denominator=200, fine_mesh_denominator=400,
                  scalar_radius_nodes=481, scalar_max_radius=1.2,
                  original_scalar_precompute_seconds=model.precompute_seconds,
                  median_outer_search_seconds=timing,
                  max_interpolation_error=max_interpolation, max_inner_refinement_error=max_inner,
                  max_candidate_refinement_error=max_candidate,
                  max_outer_weight_change=max(r['maximum_weight_change'] for r in refinement),
                  max_outer_loss_change=max(r['decision_loss_change'] for r in refinement),
                  nominal=model.nominal, nominal_envelope=model.nominal_envelope,
                  envelope_density_norm=model.norm, concentration_variance=model.variance)
    print(json.dumps(report, indent=2), flush=True)
    return report


def write_tables(model, records):
    rows = []
    for record in records:
        weights = np.array([[record[f'weight{i}'] for i in (1, 2, 3)]])
        assert np.min(weights) >= 0 and abs(weights.sum() - 1) < 1e-12
        means, scales, norms = geometry(weights, MU, covariance(record['correlation']))
        values = objectives(model, means, scales, norms, record['epsilon'])
        values = {key: float(value[0]) for key, value in values.items()}
        assert abs(values['original'] - record['value']) < 2e-10
        assert values['candidate'] <= values['original'] + 2e-7
        assert values['original'] <= values['envelope'] + 2e-7
        rows.append(dict(correlation=record['correlation'], c=record['c'],
                         epsilon=record['epsilon'], method=record['method'],
                         weight1=record['weight1'], weight2=record['weight2'],
                         weight3=record['weight3'], V_h=values['original'],
                         V_hstar=values['envelope'], V_hat_h=values['candidate']))
    save_csv('portfolio_objective_values.csv', rows)
    selected = []
    for c in (0.00, 0.05, 0.10, 0.15, 0.20):
        group = []
        for method in METHODS:
            matches = [row for row in rows if abs(row['correlation'] - 0.3) < 1e-12
                       and abs(row['c'] - c) < 1e-12 and row['method'] == method]
            assert len(matches) == 1
            group.append(matches[0])
        for row, field in zip(group, ('V_h', 'V_hstar', 'V_hat_h')):
            assert row[field] <= min(item[field] for item in group) + 2e-10
        selected.extend((group[0], group[2], group[1]))
    fields = ('correlation', 'c', 'epsilon', 'method', 'weight1', 'weight2', 'weight3',
              'V_h', 'V_hat_h', 'V_hstar')
    save_csv('portfolio_table.csv', [{key: row[key] for key in fields} for row in selected])
    print(f'Portfolio table data: {len(selected)} rows. Full objective values: {len(rows)} rows.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-data', action='store_true')
    parser.add_argument('--skip-verification', action='store_true')
    args = parser.parse_args()
    model = ScalarModel()
    if args.reuse_data:
        records = read_csv('portfolio_decisions.csv')
        scalar = read_csv('portfolio_scalar.csv')
        model.radii = np.array([row['radius'] for row in scalar])
        model.values = np.array([row['value'] for row in scalar])
    else:
        model.prepare()
        weights = simplex()
        records = decision_records(model, weights)
        save_csv('portfolio_decisions.csv', records)
        if not args.skip_verification:
            report = verify(model, weights, records)
            (DATA / 'portfolio_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_tables(model, records)


if __name__ == '__main__':
    main()
