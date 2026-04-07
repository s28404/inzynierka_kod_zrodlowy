#!/usr/bin/env python3
#  Copyright (c) 2026 Kajetan Frąckowiak, s28404
#
#  Projekt: Algorytm DEMIR dla SMACv2 i Custom Logic Environment
#  Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
#  Praca Inżynierska (2026)
#
#  Opis: Skrypt do generowania ogólnych wykresów z CSV logów.
#  Format publikacyjny (High-quality figures do IEEE/ACM publikacji).

"""
Rysuje krzywe uczenia z CSV logów zbieranych przez thesis_csv_logger.

Użycie:
    python plot_thesis_results.py                              # all algorithms
    python plot_thesis_results.py --algo qmix demir            # select algos
    python plot_thesis_results.py --env smacv2                 # filter environment
    python plot_thesis_results.py --task protoss_5_vs_5        # specific task
    python plot_thesis_results.py --metric eval_return_mean    # specific metric
    python plot_thesis_results.py --out plots/thesis/          # output directory

Format CSV loga:
    frame,step,timestamp,eval_return_mean,eval_return_std,eval_win_rate,...

Automatycznie szuka: logs_thesis/**/*.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Publication style settings ---
sns.set_style("whitegrid", {"grid.alpha": 0.3})
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "lines.linewidth": 2.5,
        "lines.markersize": 6,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)

# --- Algorithm color/style mapping ---
ALGO_STYLE = {
    "qmix": {"color": "#555555", "label": "QMIX", "linestyle": "-", "marker": "o"},
    "rnd": {
        "color": "#E88C30",
        "label": "QMIX + RND",
        "linestyle": "--",
        "marker": "s",
    },
    "ngu": {
        "color": "#4C9BE8",
        "label": "QMIX + NGU",
        "linestyle": "--",
        "marker": "^",
    },
    "demir": {
        "color": "#D62728",
        "label": "QMIX + DEMIR",
        "linestyle": "-",
        "marker": "D",
    },
}

# --- Metric configuration ---
METRIC_CONFIGS = {
    "eval_return_mean": {
        "title": "Episode Return (Mean)",
        "ylabel": "Return",
        "xlabel": "Total Frames",
        "log_x": False,
    },
    "eval_win_rate": {
        "title": "Win Rate (SMACv2)",
        "ylabel": "Win Rate (%)",
        "xlabel": "Total Frames",
        "log_x": False,
    },
    "eval_success_rate": {
        "title": "Success Rate (Logic Env)",
        "ylabel": "Success Rate (%)",
        "xlabel": "Total Frames",
        "log_x": False,
    },
}


def load_csv_logs(logs_dir: Path = Path("logs_thesis")) -> dict:
    """
    Zbiera wszystkie CSV logi.

    Returns:
        dict: {(algo, env, task, seed): DataFrame}
    """
    data = {}

    for csv_path in sorted(logs_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)

            # Ustal algorytm z nazwy pliku
            filename = csv_path.stem

            # Priorytet dla autorskich modułów/wariantów umieszczonych w nazwie
            algo_from_name = None
            for specific in ["demir", "ngu", "rnd"]:
                if specific in filename:
                    algo_from_name = specific
                    break
            if algo_from_name is None and "qmix" in filename:
                algo_from_name = "qmix"

            # Zawsze ufamy nazwie pliku bardziej ze względu na "algorithm=qmix" wewnątrz CSV
            algo = algo_from_name if algo_from_name else "unknown"

            if len(df) == 0:
                continue

            # Zmienne z kolumn
            env = (
                df.iloc[0]["environment"] if "environment" in df.columns else "unknown"
            )
            task = df.iloc[0]["task"] if "task" in df.columns else "unknown"
            seed = int(df.iloc[0]["seed"]) if "seed" in df.columns else 1

            key = (str(algo).lower(), str(env).lower(), str(task).lower(), seed)
            data[key] = df

            print(f"[✓] Loaded: {key} ({len(df)} evaluations)")
        except Exception as e:
            print(f"[✗] Error loading {csv_path}: {e}")

    return data


def plot_metric(
    data: dict,
    metric: str,
    output_dir: Path,
    algo_filter: list = None,
    env_filter: str = None,
    task_filter: str = None,
):
    """
    Rysuje metrykę z wielu runów (mean ± std po seedach).

    Args:
        data: Dict z załadowanymi CSV'ami
        metric: Nazwa metryki do wykreślenia
        output_dir: Gdzie zapis plik
        algo_filter: Filtr algorytmów (None = wszystkie)
        env_filter: Filtr środowiska
        task_filter: Filtr tasku
    """

    # Grupuj po algorytmie, środowisku, tasku
    grouped = defaultdict(lambda: defaultdict(list))  # (algo, env, task) -> seeds_data

    for (algo, env, task, seed), df in data.items():
        if metric not in df.columns:
            continue
        if algo_filter and algo not in algo_filter:
            continue
        if env_filter and env_filter.lower() not in env.lower():
            continue
        if task_filter and task_filter.lower() not in task.lower():
            continue

        grouped[(algo, env, task)][seed].append(df)

    if not grouped:
        print(f"[!] No data found for metric={metric}")
        return

    # Rysuj
    fig, axes = plt.subplots(figsize=(12, 6))

    for (algo, env, task), seed_data in sorted(grouped.items()):
        # Zbiór wszystkich x (frames) ze wszystkich seedów
        all_frames = set()
        for seed, dfs in seed_data.items():
            for df in dfs:
                all_frames.update(df.get("frame", []))
        all_frames = sorted(all_frames)

        # Interpoluj każdy seed na wspólną siatkę
        seed_means = []
        seed_stds = []
        for seed, dfs in seed_data.items():
            for df in dfs:
                if len(df) > 0:
                    # Sort by frame
                    df = df.sort_values("frame")
                    x = np.array(df["frame"])
                    y = np.array(df[metric])

                    # Interpoluj na wszystkie frames
                    y_interp = np.interp(all_frames, x, y, left=np.nan, right=np.nan)
                    seed_means.append(y_interp)

        if not seed_means:
            continue

        # Mean ± std across seeds
        seed_means = np.array(seed_means)
        mean_vals = np.nanmean(seed_means, axis=0)
        std_vals = np.nanstd(seed_means, axis=0)

        # Style
        style = ALGO_STYLE.get(algo, {"color": "gray", "label": algo, "linestyle": "-"})

        # Plot
        axes.plot(
            all_frames,
            mean_vals,
            label=style["label"],
            **{
                "color": style["color"],
                "linestyle": style["linestyle"],
                "linewidth": 2.5,
                "marker": style.get("marker", "o"),
                "markersize": 5,
                "markevery": max(1, len(all_frames) // 10),
            },
        )

        # Shaded region (±2 std errors)
        axes.fill_between(
            all_frames,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=style["color"],
            alpha=0.2,
        )

    # Formatting
    cfg = METRIC_CONFIGS.get(metric, {})
    axes.set_xlabel(cfg.get("xlabel", "Frames"))
    axes.set_ylabel(cfg.get("ylabel", metric))
    axes.set_title(cfg.get("title", metric))
    if cfg.get("log_x"):
        axes.set_xscale("log")
    axes.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    axes.grid(True, alpha=0.3)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    task_str = f"_{task_filter}" if task_filter else ""
    env_str = f"_{env_filter}" if env_filter else ""
    save_path = output_dir / f"{metric}{env_str}{task_str}.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot thesis results from CSV logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs_thesis"),
        help="Directory with CSV logs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/thesis"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--algo", nargs="+", default=None, help="Algorithm filter (e.g., qmix demir)"
    )
    parser.add_argument(
        "--env", type=str, default=None, help="Environment filter (e.g., smacv2)"
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Task filter (e.g., protoss_5_vs_5)"
    )
    parser.add_argument(
        "--metric", type=str, default="eval_return_mean", help="Metric to plot"
    )
    parser.add_argument(
        "--all-metrics", action="store_true", help="Plot all available metrics"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading CSV files from {args.logs_dir}...")
    data = load_csv_logs(args.logs_dir)

    if not data:
        print("[!] No CSV data found!")
        return

    # Collect available metrics
    all_metrics = set()
    for df in data.values():
        for col in df.columns:
            if "eval_" in col or "loss" in col:
                all_metrics.add(col)

    print(f"Found {len(data)} runs with {len(all_metrics)} metrics")

    # Plot
    if args.all_metrics:
        metrics_to_plot = sorted(all_metrics)
    else:
        metrics_to_plot = [args.metric]

    for metric in metrics_to_plot:
        print(f"\nPlotting {metric}...")
        plot_metric(
            data=data,
            metric=metric,
            output_dir=args.out,
            algo_filter=args.algo,
            env_filter=args.env,
            task_filter=args.task,
        )

    print(f"\n✓ All plots saved to {args.out}")


if __name__ == "__main__":
    main()
