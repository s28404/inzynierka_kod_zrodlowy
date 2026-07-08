"""
Plots extrinsic, intrinsic, and total return ablation curves for the key_corridor
environment comparing DEMIR variants: b1=0,b2=1, b1=1,beta2=0, no Barlow, and MLP predictor.

Author: Kajetan Frąckowiak
Date: 2026

Usage:
    python utils/plot_key_corridor_ablations.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_utils import load_alg, make_plot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data_to_plots', 'key_corridor_ablations')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'plots', 'key_corridor_ablations')

os.makedirs(OUT_DIR, exist_ok=True)

SMOOTHING = 0.99

COLORS = {
    'b1=0,b2=1': '#2ca02c',
    'b1=1,beta2=0': '#8c564b',
    'no barlow': '#e377c2',
    'mlp': '#17becf',
}

LABELS = {
    'b1=0,b2=1': r'DEMIR ($\beta_1$=0, $\beta_2$=1)',
    'b1=1,beta2=0': r'DEMIR ($\beta_1$=1, $\beta_2$=0)',
    'no barlow': 'DEMIR (no Barlow)',
    'mlp': 'DEMIR (MLP)',
}

FILE_NAMES = {
    'b1=0,b2=1': 'beta1_0_beta2_1',
    'b1=1,beta2=0': 'beta1_1_beta2_0',
    'no barlow': 'no_barlow',
    'mlp': 'mlp',
}


def plot_ablations(metric, ylabel):
    keys = ['b1=0,b2=1', 'b1=1,beta2=0', 'no barlow', 'mlp']

    dfs = [
        load_alg(f'{FILE_NAMES[k]}_{metric}.csv', metric, DATA_DIR, SMOOTHING)
        for k in keys
    ]
    labels = [LABELS[k] for k in keys]
    colors = [COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    make_plot(ax, dfs, labels, colors, 'Environment Steps', ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{metric}.png'), bbox_inches='tight')
    print(f'Zapisano {metric}.png')


plot_ablations('mean_total_return_100', 'Total Return (100-episode average)')
plot_ablations('mean_ext_return_100', 'Extrinsic Return (100-episode average)')
plot_ablations('mean_int_return_100', 'Intrinsic Return (100-episode average)')
