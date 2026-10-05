from pathlib import Path
import argparse
import csv
import json
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

from risk_numerics import (Benchmark, grid_optimum, tk, tk_derivative,
                           tk_constants, tk_grid, worst_rvar)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results'
TOLERANCE = 0.05
ALPHA = 0.99
JUMP = 0.006
RVAR_LEVELS = {'uniform': (0.95, 0.975), 'triangular': (0.95, 0.975)}
VAR_RADII = np.unique(np.r_[0, np.geomspace(1e-8, 1e-5, 25),
                            np.linspace(1e-5, 0.002, 220), 0.0001, 0.001])
RVAR_RADII = np.unique(np.r_[0, np.geomspace(1e-5, 0.001, 25),
                             np.linspace(0.001, 0.1, 180), 0.01, 0.05])
TK_REFERENCES = ('normal', 'pareto')
TK_PARETO_SHAPE = 10
TK_EPSILON_MAX = 2.3
TK_CHECK_RADII = (0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 2.3)
TK_RADII = np.unique(np.r_[0, np.geomspace(1e-4, 0.1, 51),
                            np.linspace(0.1, TK_EPSILON_MAX, 177), TK_CHECK_RADII])


class StandardizedBenchmark(Benchmark):
    def __init__(self, name, pareto_shape=3):
        super().__init__(name, pareto_shape)
        self.base = Benchmark(name, pareto_shape)
        self.center = self.base.moment(0, 1)
        self.sd = np.sqrt(self.base.moment(0, 1, 2) - self.center ** 2)

    def quantile(self, u):
        return (self.base.quantile(u) - self.center) / self.sd

    def cdf(self, x):
        return self.base.cdf(self.center + self.sd * x)

    def moment(self, a, b, order=1):
        if b <= a:
            return 0.0
        if order == 0:
            return b - a
        first = self.base.moment(a, b)
        if order == 1:
            return (first - self.center * (b - a)) / self.sd
        return (self.base.moment(a, b, 2) - 2 * self.center * first
                + self.center ** 2 * (b - a)) / self.sd ** 2

    def square_cost(self, level, a, b):
        if b <= a:
            return 0.0
        if self.name == 'uniform':
            left = level - float(self.quantile(a))
            right = level - float(self.quantile(b))
            return (b - a) * (left * left + left * right + right * right) / 3
        value = super().square_cost(level, a, b)
        if value < 1e-8:
            return quad(lambda u: (level - float(self.quantile(u))) ** 2,
                        a, b, epsabs=1e-16, epsrel=1e-9)[0]
        return value


class JumpBenchmark(StandardizedBenchmark):
    def __init__(self):
        self.name = 'jump'
        self.center = 0.5 + JUMP * (1 - ALPHA)
        self.sd = np.sqrt(1 / 12 + JUMP * ALPHA * (1 - ALPHA)
                          + JUMP ** 2 * ALPHA * (1 - ALPHA))

    def quantile(self, u):
        u = np.asarray(u)
        return (u + JUMP * (u > ALPHA) - self.center) / self.sd

    def moment(self, a, b, order=1):
        total = 0.0
        for left, right, shift in ((a, min(b, ALPHA), 0),
                                    (max(a, ALPHA), b, JUMP)):
            if right <= left:
                continue
            x = (left + shift - self.center) / self.sd
            y = (right + shift - self.center) / self.sd
            if order == 0:
                total += right - left
            elif order == 1:
                total += (right - left) * (x + y) / 2
            else:
                total += (right - left) * (x * x + x * y + y * y) / 3
        return total


def write_csv(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    with (DATA / name).open(newline='', encoding='utf-8') as stream:
        return [{k: v if k in ('distribution', 'metric') else float(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def calculate_var():
    rows = []
    for name, benchmark in (('uniform', StandardizedBenchmark('uniform')),
                             ('jump', JumpBenchmark())):
        left = float(benchmark.quantile(ALPHA))
        jump = JUMP / benchmark.sd if name == 'jump' else 0.0
        right = left + jump
        tail = 1 - ALPHA
        width = tail / benchmark.sd
        bound = width / 2
        assert bound < TOLERANCE
        for radius in VAR_RADII:
            increment = (3 * radius ** 2 / benchmark.sd) ** (1 / 3)
            if increment > width:
                increment = width / 2 + np.sqrt(radius ** 2 / tail - width ** 2 / 12)
            worst_plus = right + increment
            worst_left = worst_plus if radius > 0 else left
            var_gap = jump + increment if radius > 0 else 0.0
            plus_gap = increment - radius / np.sqrt(tail)
            cost = (benchmark.sd * increment ** 3 / 3 if increment <= width else
                    tail * (increment ** 2 - increment * width + width ** 2 / 3))
            assert abs(cost - radius ** 2) < 1e-14
            assert -1e-12 <= plus_gap <= bound + 1e-12
            rows.append(dict(distribution=name, p=2, alpha=ALPHA, epsilon=radius,
                             var_gap=var_gap, var_plus_gap=plus_gap,
                             proposition4_bound=bound,
                             var_adjusted_bound=bound + jump + radius / np.sqrt(tail),
                             quantile_jump=jump, worst_var=worst_left,
                             worst_var_plus=worst_plus, candidate_var=left,
                             candidate_var_plus=right + radius / np.sqrt(tail),
                             radius_squared_error=abs(cost - radius ** 2)))
        print(f'VaR and VaR+: {name}, reference bound={bound:.8f}', flush=True)
    write_csv('var_epsilon.csv', rows)
    return rows


def calculate_rvar():
    rows = []
    for name, (alpha, beta) in RVAR_LEVELS.items():
        benchmark = StandardizedBenchmark(name)
        nominal = benchmark.moment(alpha, beta) / (beta - alpha)
        bound = benchmark.expected_shortfall(alpha) - nominal
        assert 0 <= bound < TOLERANCE
        for radius in RVAR_RADII:
            if radius == 0:
                value, cost = nominal, 0.0
            else:
                value, _, _, _, _, cost = worst_rvar(benchmark, alpha, beta, radius)
            candidate = nominal + radius / np.sqrt(1 - alpha)
            gap = value - candidate
            assert -2e-9 <= gap <= bound + 2e-9
            assert abs(np.sqrt(max(cost, 0)) - radius) < 2e-8
            rows.append(dict(distribution=name, p=2, alpha=alpha, beta=beta,
                             epsilon=radius, gap=gap, proposition4_bound=bound,
                             worst=value, candidate=candidate,
                             radius_error=abs(np.sqrt(max(cost, 0)) - radius)))
        print(f'RVaR: {name}, reference bound={bound:.8f}', flush=True)
    write_csv('rvar_epsilon.csv', rows)
    return rows


def continuous_constants(benchmark):
    constants = tk_constants(benchmark.base)
    constants['bound'] /= benchmark.sd
    constants['variance'] /= benchmark.sd ** 2
    pieces = [quad(lambda t: tk_derivative(t) ** 2, a, b,
                   epsabs=2e-9, epsrel=2e-9, limit=300)
              for a, b in ((0, 0.5), (0.5, 1))]
    constants['density_norm'] = np.sqrt(sum(v for v, _ in pieces))
    constants['quadrature_error'] = sum(e for _, e in pieces)
    return constants


def continuous_bounds(radius, constants):
    denominator = np.sqrt(constants['variance'] + radius ** 2)
    increment = radius * (constants['bound'] + radius * constants['norm']) / denominator
    first = (1 - radius / denominator) * (constants['bound'] + radius * constants['norm'])
    second = radius * constants['density_norm'] - increment
    return first, second


def calculate_tk(cells=4000, radii=TK_RADII, save=True):
    rows = []
    for name in TK_REFERENCES:
        benchmark = StandardizedBenchmark(name, TK_PARETO_SHAPE)
        assert abs(benchmark.moment(0, 1)) < 1e-12
        assert abs(benchmark.moment(0, 1, 2) - 1) < 1e-12
        constants = continuous_constants(benchmark)
        weights, means, density, envelope, concentrated, norm, reference_bound = tk_grid(
            benchmark, cells, constants['cut'])
        nominal = np.dot(weights, means * density)
        for radius in radii:
            quantile, scale = grid_optimum(means, density, weights, radius)
            direction = concentrated - means + radius * envelope / norm
            length = np.sqrt(np.dot(weights, direction ** 2))
            candidate = means + radius * direction / length
            candidate_value = np.dot(weights, candidate * density)
            worst = np.dot(weights, quantile * density)
            gap = worst - candidate_value
            first, second = continuous_bounds(radius, constants)
            candidate_increment = radius * (reference_bound + radius * norm) / length
            grid_first = reference_bound + radius * norm - candidate_increment
            grid_second = radius * np.sqrt(np.dot(weights, density ** 2)) - candidate_increment
            radius_error = abs(np.sqrt(np.dot(weights, (quantile - means) ** 2)) - radius)
            candidate_radius_error = abs(np.sqrt(np.dot(weights, (candidate - means) ** 2)) - radius)
            formula_error = abs(candidate_value - nominal - candidate_increment)
            residual = weights * (means + scale * density - quantile)
            cumulative = np.cumsum(residual)
            kkt_error = max(abs(cumulative[-1]), max(0, -cumulative.min()),
                            abs(np.dot(residual, quantile)))
            assert np.min(np.diff(quantile)) >= -2e-10
            assert np.min(np.diff(candidate)) >= -2e-10
            assert max(radius_error, candidate_radius_error, formula_error) < 2e-8
            assert kkt_error < 2e-7
            assert -2e-8 <= gap <= min(grid_first, grid_second) + 2e-8
            assert gap <= min(first, second) + 2e-5
            rows.append(dict(distribution=name, epsilon=radius, p=2, gamma=0.7,
                             cells=cells, grid_candidate_gap=gap,
                             theorem3_i_bound=first, theorem3_ii_bound=second,
                             grid_theorem3_i_bound=grid_first,
                             grid_theorem3_ii_bound=grid_second,
                             grid_worst=worst, grid_candidate_value=candidate_value,
                             radius_error=radius_error,
                             candidate_radius_error=candidate_radius_error,
                             candidate_formula_error=formula_error, kkt_error=kkt_error))
        print(f'TK: {name}, N={cells}', flush=True)
    if save:
        write_csv('tk_epsilon.csv', rows)
    return rows


def summarize(var, rvar, tk_rows):
    report = dict(tolerance=TOLERANCE, loss_standard_deviation=1,
                  tk_pareto_shape=TK_PARETO_SHAPE, tk_epsilon_max=TK_EPSILON_MAX,
                  python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, tk={}, rvar={})
    report['quantile_jump'] = JUMP / JumpBenchmark().sd
    for name in TK_REFERENCES:
        constants = continuous_constants(
            StandardizedBenchmark(name, TK_PARETO_SHAPE))
        crossing = constants['bound'] / (constants['density_norm'] - constants['norm'])
        residual = lambda r: min(continuous_bounds(r, constants)) - TOLERANCE
        knots = np.geomspace(1e-8, 1000, 2000)
        roots = [brentq(residual, a, b) for a, b in zip(knots[:-1], knots[1:])
                 if residual(a) * residual(b) < 0]
        series = [r for r in tk_rows if r['distribution'] == name]
        peak = max(series, key=lambda r: r['grid_candidate_gap'])
        report['tk'][name] = dict(constants=constants, bound_crossing=crossing,
                                  tolerance_crossings=roots,
                                  max_grid_gap=peak['grid_candidate_gap'],
                                  radius_at_grid_max=peak['epsilon'],
                                  at_001=next(r for r in series if r['epsilon'] == 0.01),
                                  at_1=next(r for r in series if r['epsilon'] == 1),
                                  at_max_radius=next(r for r in series if r['epsilon'] == TK_EPSILON_MAX))
    for name in RVAR_LEVELS:
        series = [r for r in rvar if r['distribution'] == name]
        report['rvar'][name] = dict(alpha=series[0]['alpha'], beta=series[0]['beta'],
                                    reference_bound=series[0]['proposition4_bound'],
                                    at_001=next(r for r in series if r['epsilon'] == 0.01),
                                    at_01=next(r for r in series if r['epsilon'] == 0.1))
    report['var_at_0001'] = [r for r in var if r['epsilon'] == 0.001]
    (DATA / 'experiment_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def verify(var, rvar, tk_rows, verify_tk=True):
    checks = []
    for name, (alpha, beta) in RVAR_LEVELS.items():
        benchmark = StandardizedBenchmark(name)
        assert abs(benchmark.moment(0, 1)) < 1e-12
        assert abs(benchmark.moment(0, 1, 2) - 1) < 1e-12
        for radius in (0.001, 0.01, 0.05, 0.1):
            value = worst_rvar(benchmark, alpha, beta, radius)[0]
            for cells in (2000, 8000):
                knots = np.unique(np.r_[np.linspace(0, 1, cells + 1), alpha, beta])
                mass = np.diff(knots)
                means = np.array([benchmark.moment(a, b) / (b - a)
                                  for a, b in zip(knots[:-1], knots[1:])])
                mid = (knots[:-1] + knots[1:]) / 2
                density = ((mid > alpha) & (mid < beta)) / (beta - alpha)
                quantile, _ = grid_optimum(means, density, mass, radius)
                error = abs(np.dot(mass * density, quantile) - value)
                assert error < (2e-4 if cells == 8000 else 2e-3)
                checks.append(dict(distribution=name, epsilon=radius, cells=cells,
                                   value_difference=error))
    jump = JumpBenchmark()
    assert abs(jump.moment(0, 1)) < 1e-12
    assert abs(jump.moment(0, 1, 2) - 1) < 1e-12
    for row in var:
        if row['epsilon'] > 0:
            expected = (row['var_plus_gap'] + row['quantile_jump']
                        + row['epsilon'] / np.sqrt(1 - row['alpha']))
            assert abs(row['var_gap'] - expected) < 1e-12
    refined = (calculate_tk(cells=8000, radii=TK_CHECK_RADII, save=False)
               if verify_tk else read_csv('tk_refinement.csv'))
    differences = []
    for fine in refined:
        coarse = next(r for r in tk_rows if r['distribution'] == fine['distribution']
                      and r['epsilon'] == fine['epsilon'])
        difference = abs(fine['grid_candidate_gap'] - coarse['grid_candidate_gap'])
        assert difference < 2e-5
        fine['gap_refinement_difference'] = difference
        differences.append(difference)
    write_csv('rvar_refinement.csv', checks)
    if verify_tk:
        write_csv('tk_refinement.csv', refined)
    report = dict(max_rvar_value_difference_8000=max(
        r['value_difference'] for r in checks if r['cells'] == 8000),
        max_tk_gap_refinement_difference=max(differences),
        max_tk_radius_error=max(r['radius_error'] for r in tk_rows + refined),
        max_tk_candidate_radius_error=max(r['candidate_radius_error'] for r in tk_rows + refined),
        max_tk_candidate_formula_error=max(r['candidate_formula_error'] for r in tk_rows + refined),
        max_tk_kkt_error=max(r['kkt_error'] for r in tk_rows + refined),
        tk_refinement_points=len(refined), rvar_refinement_points=len(checks))
    (DATA / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def plot(var, rvar, tk_rows, tk_only=False, rvar_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 11,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    blue, orange, green, purple = '#0072B2', '#D55E00', '#009E73', '#8A4F9E'
    dashed, dashdot = (0, (5, 2)), (0, (5, 2, 1, 2))

    def decorate(ax, tolerance_label_above=True, tolerance_x=0.025, tolerance_offset=0):
        ax.tick_params(axis='y', labelleft=True)
        ax.set_xlabel(r'$\epsilon$ (SD units)')
        ax.set_ylabel('Gap (SD units)')
        ax.grid(alpha=0.16, lw=0.6)
        ax.axhline(TOLERANCE, color='#777777', ls=':', lw=1.1, zorder=0)
        if tolerance_label_above:
            ax.text(tolerance_x, TOLERANCE + tolerance_offset, 'Tolerance 0.05', color='#666666', fontsize=8,
                    transform=ax.get_yaxis_transform(), va='bottom',
                    ha='right' if tolerance_x > 0.5 else 'left')
        else:
            ax.annotate('Tolerance 0.05', xy=(0.025, TOLERANCE),
                        xycoords=('axes fraction', 'data'), xytext=(0, -3),
                        textcoords='offset points', color='#666666', fontsize=8, va='top')

    def save(fig, name):
        fig.savefig(ROOT / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(ROOT / (name + '.png'), bbox_inches='tight', dpi=240)
        plt.close(fig)

    if not tk_only and not rvar_only:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
        for ax, name, title in zip(axes, ('uniform', 'jump'),
                                    ('Continuous quantile', r'Quantile jump at $\alpha$')):
            rows = [r for r in var if r['distribution'] == name]
            positive = [r for r in rows if r['epsilon'] > 0]
            ax.plot([r['epsilon'] for r in positive], [r['var_gap'] for r in positive],
                    color=blue, lw=1.8)
            ax.plot([r['epsilon'] for r in rows], [r['var_plus_gap'] for r in rows],
                    color=orange, lw=1.8)
            ax.plot([r['epsilon'] for r in rows], [r['proposition4_bound'] for r in rows],
                    color=green, ls=dashed, lw=1.4)
            ax.plot([r['epsilon'] for r in rows], [r['var_adjusted_bound'] for r in rows],
                    color=purple, ls=dashdot, lw=1.4)
            ax.scatter([0], [0], color=blue, s=22, clip_on=False, zorder=5)
            if name == 'jump':
                ax.scatter([0], [rows[0]['quantile_jump']], facecolors='white',
                           edgecolors=blue, s=26, clip_on=False, zorder=6)
            ax.set_title(title)
            ax.set_xlim(0, 0.002)
            ax.set_ylim(0, 0.061)
            ax.set_xticks((0, 0.0005, 0.001, 0.0015, 0.002))
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, -3), useMathText=True)
            decorate(ax)
        handles = [Line2D([], [], color=blue, lw=1.8, label=r'$S_{\mathrm{VaR}_\alpha}-\mathrm{VaR}_\alpha(F^*)$'),
                   Line2D([], [], color=orange, lw=1.8, label=r'$S_{\mathrm{VaR}^+_\alpha}-\mathrm{VaR}^+_\alpha(F^*)$'),
                   Line2D([], [], color=green, ls=dashed, lw=1.4, label=r'Proposition 5 bound ($\mathrm{VaR}^+_\alpha$)'),
                   Line2D([], [], color=purple, ls=dashdot, lw=1.4, label=r'Example 3 bound ($\mathrm{VaR}_\alpha$)')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 1.04),
                   ncol=2, frameon=False, fontsize=9.1, columnspacing=2.2)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.75, wspace=0.38)
        save(fig, 'var_comparison_epsilon')

    if not tk_only:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharey=True)
        for ax, name, title in zip(axes, ('uniform', 'triangular'),
                                    ('Standardized uniform', 'Standardized triangular')):
            rows = [r for r in rvar if r['distribution'] == name]
            ax.plot([r['epsilon'] for r in rows], [r['gap'] for r in rows], color=blue, lw=1.8)
            ax.plot([r['epsilon'] for r in rows], [r['proposition4_bound'] for r in rows],
                    color=green, ls=dashed, lw=1.5)
            alpha, beta = RVAR_LEVELS[name]
            ax.set_title(title + '\n' + rf'$\alpha={alpha},\ \beta={beta}$')
            ax.set_xlim(0, 0.1)
            ax.set_ylim(0, 0.055)
            decorate(ax)
        handles = [Line2D([], [], color=blue, lw=1.8,
                          label=r'$S_{\mathrm{RVaR}_{\alpha,\beta}}-\mathrm{RVaR}_{\alpha,\beta}(F^*)$'),
                   Line2D([], [], color=green, ls=dashed, lw=1.5, label='Proposition 5 bound')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 1.01),
                   ncol=2, frameon=False, fontsize=10)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.75, wspace=0.38)
        save(fig, 'rvar_gap_epsilon')

    if rvar_only:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, name, title in zip(axes, TK_REFERENCES,
                                ('Standard normal', rf'Standardized Pareto ($a={TK_PARETO_SHAPE}$)')):
        rows = [r for r in tk_rows if r['distribution'] == name]
        for field, color, style in (('grid_candidate_gap', blue, '-'),
                                    ('theorem3_i_bound', orange, dashed),
                                    ('theorem3_ii_bound', green, dashdot)):
            ax.plot([r['epsilon'] for r in rows], [r[field] for r in rows],
                    color=color, ls=style, lw=1.8 if field == 'grid_candidate_gap' else 1.5)
        ax.set_title(title)
        ax.set_xlim(0, TK_EPSILON_MAX)
        ax.set_ylim(0, 0.5)
        ax.set_xticks(np.arange(0, 2.1, 0.5))
        ax.set_yticks(np.arange(0, 0.51, 0.1))
        ax.tick_params(axis='x', labelbottom=True)
        ax.minorticks_off()
        decorate(ax, tolerance_x=0.98, tolerance_offset=0.007)
    handles = [Line2D([], [], color=blue, lw=1.8,
                      label=r'$S_h(\epsilon)-\rho_h(P_\epsilon)$ (grid)'),
               Line2D([], [], color=orange, ls=dashed, lw=1.5, label='Theorem 3(i) upper bound'),
               Line2D([], [], color=green, ls=dashdot, lw=1.5, label='Theorem 3(ii) upper bound')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 1.01),
               ncol=3, frameon=False, fontsize=8.7, columnspacing=1.2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.78, wspace=0.38)
    save(fig, 'tk_candidate_gap_bounds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-data', action='store_true')
    parser.add_argument('--skip-verification', action='store_true')
    args = parser.parse_args()
    var = read_csv('var_epsilon.csv') if args.reuse_data else calculate_var()
    rvar = read_csv('rvar_epsilon.csv') if args.reuse_data else calculate_rvar()
    tk_rows = read_csv('tk_epsilon.csv') if args.reuse_data else calculate_tk()
    summarize(var, rvar, tk_rows)
    if not args.skip_verification:
        verify(var, rvar, tk_rows)
    plot(var, rvar, tk_rows)


if __name__ == '__main__':
    main()
