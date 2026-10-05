from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Run the paper experiments.')
    parser.add_argument('--reuse-data', action='store_true')
    parser.add_argument('--skip-verification', action='store_true')
    args = parser.parse_args()
    options = []
    if args.reuse_data:
        options.append('--reuse-data')
    if args.skip_verification:
        options.append('--skip-verification')
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    for script in ('paper_experiments.py', 'portfolio_experiment.py',
                   'iqr_portfolio_experiment.py'):
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(root / script), *options],
                       cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
