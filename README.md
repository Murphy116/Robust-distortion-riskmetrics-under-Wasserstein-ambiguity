# Robust distortion riskmetrics under Wasserstein ambiguity

Python code for the VaR, RVaR, TK distortion, and portfolio experiments in the paper *Robust distortion riskmetrics under Wasserstein ambiguity*.

This repository contains code and running instructions only. Figures and numerical results are generated locally by the scripts.

## Setup

The experiments were run with Python 3.11.9. Use Python 3.11 with the versions in `requirements.txt`.

```text
python -m venv .venv
```

Activate the environment on Windows:

```text
.venv\Scripts\activate
```

On macOS or Linux:

```text
source .venv/bin/activate
```

Then install the dependencies:

```text
python -m pip install -r requirements.txt
```

The scripts use NumPy, SciPy, and Matplotlib. MATLAB and a LaTeX installation are not required. Paths are relative to the scripts, and plots are saved without opening a window.

## Run the experiments

Recompute all experiments, including the numerical checks:

```text
python run_experiments.py
```

After a full run has created the intermediate data, regenerate figures and portfolio results with:

```text
python run_experiments.py --reuse-data --skip-verification
```

The second command does not rerun the full numerical checks. It reads saved worst-case values. The IQR script still recalculates portfolio choices and directly evaluates the best candidates.

The scripts can also be run separately:

```text
python paper_experiments.py
python portfolio_experiment.py
python iqr_portfolio_experiment.py
```

Each script accepts `--reuse-data` and `--skip-verification`. Recomputing overwrites the corresponding files in `results/` and the figure files. If parameters are changed, recompute the data rather than using `--reuse-data`.

## Files and paper results

| File | Purpose |
|---|---|
| `risk_numerics.py` | Quantile integrals, worst-case RVaR values, the TK envelope, and weighted isotonic regression |
| `paper_experiments.py` | VaR, VaR+, RVaR, and TK approximation experiments |
| `portfolio_experiment.py` | TK portfolio selection using the original, explicit approximate, and envelope objectives |
| `iqr_portfolio_experiment.py` | Regularized IQR portfolio selection and the independent original-IQR calculation |
| `var_comparison_epsilon.pdf` | Figure 1: VaR and VaR+ |
| `rvar_gap_epsilon.pdf` | Figure 2: RVaR |
| `tk_candidate_gap_bounds.pdf` | Figure 3: TK |
| `results/portfolio_table.csv` | Table 2 data |
| `results/iqr_portfolio.csv` | Table 3 data, including the additional radius discussed in the text |

The files listed for Figures 1-3 and Tables 2-3 are generated outputs, not files supplied in the repository. Each figure is saved as PDF and PNG, and table data are saved as CSV. The scripts do not edit the manuscript.

## Experimental settings

All experiments use Wasserstein order `p=2`.

### VaR, RVaR, and TK

The reference losses are standardized to mean zero and variance one. Radii, gaps, and bounds are expressed in units of the original reference loss standard deviation. The common error tolerance is `0.05`.

- **VaR and VaR+:** `alpha=0.99`, with radii from `0` to `0.002`. The original references are `U[0,1]` and the mixture `0.99 U[0,0.99] + 0.01 U[0.996,1.006]`. The second has a quantile jump at the selected probability level.
- **RVaR:** `alpha=0.95` and `beta=0.975` for both references, with radii from `0` to `0.1`. The original references are `U[0,1]` and the triangular distribution with density `2y` on `[0,1]`.
- **TK:** `gamma=0.7`, with radii from `0` to `2.3`. The references are standard normal and standardized Pareto with minimum `1` and shape `10`. The numerical worst-case calculation uses `4,000` quantile cells. Both bounds in Theorem 3 are plotted against the error of the same candidate distribution.

### TK portfolio selection

The three asset losses have means `(0.080, 0.075, 0.070)` and standard deviations `(0.12, 0.20, 0.32)`. Pairwise correlations are `0`, `0.3`, `0.6`, and `0.9`. The radius is `epsilon = c * sqrt(trace(Sigma))`, where `c` ranges from `0` to `0.20` in steps of `0.005`. The distortion is the upper-tail reflection of TK with `gamma=0.7`.

The weight grid contains all nonnegative multiples of `0.005` that sum to one, giving `20,301` portfolios. Inner values use `4,000` quantile cells, with more cells near the probability endpoints. A grid endpoint is placed at the envelope transition. Values are stored at `481` equally spaced standardized radii from `0` to `1.2`, and linear interpolation is used during the weight search.

Table 2 uses correlation `0.3` and `c=0, 0.05, 0.10, 0.15, 0.20`. The `method` entries `original`, `candidate`, and `envelope` identify the objective used to select the weights. The paper's final column is `V_h` evaluated at those selected weights. Only the `original` method minimizes that column on the weight grid. The CSV also retains `V_hat_h` and `V_hstar` at each selected portfolio for checking the calculations.

### IQR portfolio selection

The IQR experiment uses the same assets and weight grid, correlation `0.3`, and `alpha=0.75`. The three radius parameters are `c=0.005, 0.05, 0.20`. The positive regularization parameters are `eta=0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001`.

For each positive `eta`, worst-case values are stored at `961` geometrically spaced standardized radii. PCHIP interpolation is used during the search, followed by direct calculation at the `32` portfolios with the lowest interpolated values. The original IQR is calculated independently using normal quantile transport costs and one-dimensional root finding.

In `results/iqr_portfolio.csv`, `eta=0` identifies the original IQR problem. It is not substituted into the positive-eta regularization formula. Table 3 uses `c=0.005` and `c=0.05`. Results for `c=0.20` are also included in the CSV.

## Numerical accuracy

The portfolio minima are numerical results on the stated weight grid. They are not asserted to be exact minima over all continuous portfolio weights. Stored check results include quantile-grid comparisons, radius constraints, optimality conditions, interpolation errors, and weight-grid comparisons. These are checks at the reported points, not uniform mathematical error bounds.

The calculations use specified reference distributions and deterministic grids. No random samples or external datasets are needed. Small floating-point differences across systems are possible, and regenerated PDFs can have different metadata and file hashes.
