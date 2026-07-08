"""
Plots extrinsic/intrinsic episode rewards and return curves for the logic_env environment,
comparing DEMIR (scales 0.1, 0.5, 1.0), QMIX, RND, and NGU algorithms.

Author: Kajetan Frąckowiak
Date: 2026

Usage:
    python utils/plot_logic_env.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_utils import load_alg, make_plot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data_to_plots', 'logic_env')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots', 'logic_env')

os.makedirs(OUT_DIR, exist_ok=True)

SMOOTHING = 0.99
COLORS = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd', '#8c564b']
LABELS = ['DEMIR_01', 'DEMIR_05', 'QMIX', 'RND', 'NGU', 'DEMIR_1_0']

for sub in ['mean', 'max', 'min']:
    metric = f'extrinsic_episode_reward_{sub}'
    d01 = load_alg(f'demir_01_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    d05 = load_alg(f'demir_05_exstrinsic_episode_reward_{sub}.csv', metric, DATA_DIR, SMOOTHING)
    q = load_alg(f'qmix_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    rnd = load_alg(f'rnd_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    ngu = load_alg(f'ngu_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    d10 = load_alg(f'demir_1_0_{metric}.csv', metric, DATA_DIR, SMOOTHING)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    make_plot(ax, [d01, d05, q, rnd, ngu, d10], LABELS, COLORS,
              'Environment Steps', f'Extrinsic Episode Reward {sub.capitalize()}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'extrinsic_episode_reward_{sub}.png'), bbox_inches='tight')
    print(f'Zapisano extrinsic_episode_reward_{sub}.png')

for sub in ['mean', 'max', 'min']:
    metric = f'intrinsic_episode_reward_{sub}'
    d01 = load_alg(f'demir_01_instrinsic_episode_reward_{sub}.csv', metric, DATA_DIR, SMOOTHING)
    d05 = load_alg(f'demir_05_instrinsic_episode_reward_{sub}.csv', metric, DATA_DIR, SMOOTHING)
    rnd = load_alg(f'rnd_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    ngu = load_alg(f'ngu_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    d10 = load_alg(f'demir_1_0_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    algos = [d01, d05, rnd, ngu, d10]
    lbls = ['DEMIR_01', 'DEMIR_05', 'RND', 'NGU', 'DEMIR_1_0']
    clrs = [COLORS[0], COLORS[1], COLORS[3], COLORS[4], COLORS[5]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    make_plot(ax, algos, lbls, clrs,
              'Environment Steps', f'Intrinsic Episode Reward {sub.capitalize()}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'intrinsic_episode_reward_{sub}.png'), bbox_inches='tight')
    print(f'Zapisano intrinsic_episode_reward_{sub}.png')

for ret_type, ret_label in [('ext', 'Extrinsic'), ('int', 'Intrinsic'), ('total', 'Total')]:
    metric = f'mean_{ret_type}_return_100'
    d01 = load_alg(f'demir_01_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    d05 = load_alg(f'demir_05_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    rnd = load_alg(f'rnd_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    ngu = load_alg(f'ngu_{metric}.csv', metric, DATA_DIR, SMOOTHING)
    d10 = load_alg(f'demir_1_0_{metric}.csv', metric, DATA_DIR, SMOOTHING)

    if ret_type == 'ext':
        q = load_alg(f'qmix_{metric}.csv', metric, DATA_DIR, SMOOTHING)
        algos = [d01, d05, q, rnd, ngu, d10]
        lbls = LABELS
        clrs = COLORS
    else:
        algos = [d01, d05, rnd, ngu, d10]
        lbls = ['DEMIR_01', 'DEMIR_05', 'RND', 'NGU', 'DEMIR_1_0']
        clrs = [COLORS[0], COLORS[1], COLORS[3], COLORS[4], COLORS[5]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    make_plot(ax, algos, lbls, clrs,
              'Environment Steps', f'{ret_label} Return (100-episode average)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{metric}.png'), bbox_inches='tight')
    print(f'Zapisano {metric}.png')
