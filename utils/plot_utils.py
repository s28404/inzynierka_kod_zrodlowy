"""
Shared plotting utilities for loading and visualising training curves.

Provides:
  - load_alg: reads CSV logs, applies smoothing, returns mean/min/max curves.
  - make_plot: plots multiple algorithm curves with shaded std bands on a given axis.

Author: Kajetan Frąckowiak
Date: 2026
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def load_alg(filename, metric_name, data_dir, smoothing):
    path = os.path.join(data_dir, filename)

    patterns = [
        f'Grouped runs - {metric_name}',
        f'Grouped runs - collection/reward/{metric_name}',
    ]

    df = pd.read_csv(path)

    mean_col = None
    for p in patterns:
        if p in df.columns:
            mean_col = p
            break

    if mean_col is None:
        raise KeyError(f'Column for metric "{metric_name}" not found in {filename}. '
                       f'Available columns: {list(df.columns)}')

    min_col = f'{mean_col}__MIN'
    max_col = f'{mean_col}__MAX'

    df = df.rename(columns={mean_col: 'mean', min_col: 'min', max_col: 'max'})
    df['mean'] = df['mean'].ewm(alpha=1 - smoothing).mean()
    df['min'] = df['min'].ewm(alpha=1 - smoothing).mean()
    df['max'] = df['max'].ewm(alpha=1 - smoothing).mean()

    df['plot_mean'] = df['mean']
    df['lower_std'] = df['min']
    df['upper_std'] = df['max']
    return df[['Step', 'plot_mean', 'lower_std', 'upper_std']]


def make_plot(ax, dfs, labels, colors, xlabel, ylabel):
    for df, label, color in zip(dfs, labels, colors):
        ax.plot(df['Step'], df['plot_mean'], label=label, linewidth=2.5, color=color)
        ax.fill_between(df['Step'], df['lower_std'], df['upper_std'],
                        alpha=0.15, color=color)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(250000))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _:
            f'{int(x/1e6)}M' if x >= 1e6 and x % 1e6 == 0
            else f'{x/1e6}M' if x >= 1e6
            else f'{int(x/1000)}k'))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    sns.despine()
