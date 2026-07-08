"""
Plots extrinsic, intrinsic, and total return curves for the key_corridor environment
comparing R2D2, DEMIR, RND, and NGU algorithms across environment steps.

Author: Kajetan Frąckowiak
Date: 2026

Usage:
    python utils/plot_key_corridor.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_utils import load_alg, make_plot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data_to_plots', 'key_corridor')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots', 'key_corridor')

os.makedirs(OUT_DIR, exist_ok=True)

SMOOTHING = 0.99

COLORS = {
    'R2D2': '#1f77b4',
    'DEMIR': '#ff7f0e',
    'RND': '#d62728',
    'NGU': '#9467bd',
}

metric = 'mean_ext_return_100'
r2d2 = load_alg(f'r2d2_{metric}.csv', metric, DATA_DIR, SMOOTHING)
demir = load_alg(f'demir_{metric}.csv', metric, DATA_DIR, SMOOTHING)
rnd = load_alg(f'rnd_{metric}.csv', metric, DATA_DIR, SMOOTHING)
ngu = load_alg(f'ngu_{metric}.csv', metric, DATA_DIR, SMOOTHING)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
make_plot(ax, [r2d2, demir, rnd, ngu],
          ['R2D2', 'DEMIR', 'RND', 'NGU'],
          [COLORS['R2D2'], COLORS['DEMIR'], COLORS['RND'], COLORS['NGU']],
          'Environment Steps', 'Extrinsic Return (100-episode average)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{metric}.png'), bbox_inches='tight')
print(f'Zapisano {metric}.png')

metric = 'mean_int_return_100'
demir_i = load_alg(f'demir_{metric}.csv', metric, DATA_DIR, SMOOTHING)
rnd_i = load_alg(f'rnd_{metric}.csv', metric, DATA_DIR, SMOOTHING)
ngu_i = load_alg(f'ngu_{metric}.csv', metric, DATA_DIR, SMOOTHING)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
make_plot(ax, [demir_i, rnd_i, ngu_i],
          ['DEMIR', 'RND', 'NGU'],
          [COLORS['DEMIR'], COLORS['RND'], COLORS['NGU']],
          'Environment Steps', 'Intrinsic Return (100-episode average)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{metric}.png'), bbox_inches='tight')
print(f'Zapisano {metric}.png')

metric = 'mean_total_return_100'
demir_t = load_alg(f'demir_{metric}.csv', metric, DATA_DIR, SMOOTHING)
rnd_t = load_alg(f'rnd_{metric}.csv', metric, DATA_DIR, SMOOTHING)
ngu_t = load_alg(f'ngu_{metric}.csv', metric, DATA_DIR, SMOOTHING)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
make_plot(ax, [demir_t, rnd_t, ngu_t],
          ['DEMIR', 'RND', 'NGU'],
          [COLORS['DEMIR'], COLORS['RND'], COLORS['NGU']],
          'Environment Steps', 'Total Return (100-episode average)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{metric}.png'), bbox_inches='tight')
print(f'Zapisano {metric}.png')
