"""
Plots learning curves from CSV logs collected by thesis_csv_logger.

Usage:
    python plot_thesis_results.py                              # all algorithms
    python plot_thesis_results.py --algo qmix demir            # select specific algorithms
    python plot_thesis_results.py --env smacv2                 # filter by environment
    python plot_thesis_results.py --task protoss_5_vs_5        # specific task
    python plot_thesis_results.py --metric eval_return_mean    # specific metric
    python plot_thesis_results.py --out plots/thesis/          # output directory

CSV log format:
    frame,step,timestamp,eval_return_mean,eval_return_std,eval_win_rate,...

Automatically searches for: logs_thesis/**/*.csv
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
    Collects all CSV logs from the specified directory.

    Returns:
        dict: {(algo, env, task, seed): DataFrame}
    """
    data = {}

    for csv_path in sorted(logs_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)

            # Determine algorithm from filename
            filename = csv_path.stem

            # Priority for custom modules/variants in the filename
            algo_from_name = None
            for specific in ["demir", "ngu", "rnd"]:
                if specific in filename:
                    algo_from_name = specific
                    break
            if algo_from_name is None and "qmix" in filename:
                algo_from_name = "qmix"

            # We trust the filename more (due to "algorithm=qmix" possibly being inside CSV)
            algo = algo_from_name if algo_from_name else "unknown"

            if len(df) == 0:
                continue

            # Extract variables from columns
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
    Plots a metric from multiple runs (mean ± std across seeds).

    Args:
        data: Dict with loaded CSVs
        metric: Name of the metric to plot
        output_dir: Directory to save the plot
        algo_filter: List of algorithms to include (None = all)
        env_filter: Environment filter
        task_filter: Task filter
    """

    # Group by algorithm, environment, and task
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

    # Plot
    fig, axes = plt.subplots(figsize=(12, 6))

    for (algo, env, task), seed_data in sorted(grouped.items()):
        # Collect all unique frames across all seeds
        all_frames = set()
        for seed, dfs in seed_data.items():
            for df in dfs:
                all_frames.update(df.get("frame", []))
        all_frames = sorted(all_frames)

        # Interpolate each seed onto the common frame grid
        seed_means = []
        for seed, dfs in seed_data.items():
            for df in dfs:
                if len(df) > 0:
                    # Sort by frame
                    df = df.sort_values("frame")
                    x = np.array(df["frame"])
                    y = np.array(df[metric])

                    # Interpolate onto all_frames
                    y_interp = np.interp(all_frames, x, y, left=np.nan, right=np.nan)
                    seed_means.append(y_interp)

        if not seed_means:
            continue

        # Compute mean ± std across seeds
        seed_means = np.array(seed_means)
        mean_vals = np.nanmean(seed_means, axis=0)
        std_vals = np.nanstd(seed_means, axis=0)

        # Get plotting style
        style = ALGO_STYLE.get(algo, {"color": "gray", "label": algo, "linestyle": "-"})

        # Plot mean line
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

        # Shaded region (±1 std)
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

    # Save plot
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