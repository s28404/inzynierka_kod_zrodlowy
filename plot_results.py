# Autor: Kajetan Frąckowiak, s28404 (2026)
# Plik napisany od podstaw w ramach pracy inżynierskiej
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Skrypt do generowania wykresów krzywych uczenia (mean ± std po seedach)
#        z plików JSON produkowanych przez BenchMARL. Format publikacyjny.

"""
Rysuje krzywe uczenia z rozmyciem mean ± std po seedach.

Użycie:
    python plot_results.py                        # wszystkie taski
    python plot_results.py --task simple_spread   # konkretny task
    python plot_results.py --env smacv2           # filtrowanie po środowisku
    python plot_results.py --metric return        # konkretna metryka (domyślnie: return)
    python plot_results.py --out plots/           # folder zapisu (domyślnie: plots/)

Skrypt szuka automatycznie wszystkich plików *.json w outputs/
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # bez GUI (działa przez SSH / headless)
import matplotlib.pyplot as plt
import numpy as np

# ── Kolory i nazwy algorytmów ────────────────────────────────────────────────
ALGO_STYLE = {
    "qmix":  {"color": "#555555", "label": "QMIX (baseline)", "ls": "-"},
    "rnd":   {"color": "#E88C30", "label": "QMIX + RND",      "ls": "--"},
    "ngu":   {"color": "#4C9BE8", "label": "QMIX + NGU",      "ls": "--"},
    "demir": {"color": "#D62728", "label": "QMIX + DEMIR",    "ls": "-"},
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_runs(outputs_dir: Path) -> dict:
    """
    Zwraca słownik:
      runs[(env, task, algo)][seed] = lista (step_count, mean_val)
    """
    runs = defaultdict(lambda: defaultdict(list))

    for json_path in sorted(outputs_dir.rglob("*.json")):
        # pomijamy pliki wandb
        if "wandb" in str(json_path):
            continue
        try:
            data = load_json(json_path)
        except Exception:
            continue

        for env, env_data in data.items():
            for task, task_data in env_data.items():
                for algo, algo_data in task_data.items():
                    for seed_str, seed_data in algo_data.items():
                        steps = []
                        for key, val in seed_data.items():
                            if not key.startswith("step_"):
                                continue
                            step_count = val.get("step_count", None)
                            if step_count is None:
                                continue
                            steps.append((key, step_count, val))
                        steps.sort(key=lambda x: x[1])
                        runs[(env, task, algo)][seed_str].extend(steps)

    return runs


def aggregate(seed_data: dict, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dla każdego seeda zbiera serię (step_count, mean_return).
    Interpoluje na wspólną siatkę x i zwraca mean ± std po seedach.
    """
    seed_series = {}
    for seed, steps in seed_data.items():
        xs, ys = [], []
        for _, step_count, val in steps:
            if metric not in val:
                continue
            raw = val[metric]
            if isinstance(raw, list) and len(raw) > 0:
                ys.append(float(np.mean(raw)))
                xs.append(step_count)
        if xs:
            seed_series[seed] = (np.array(xs), np.array(ys))

    if not seed_series:
        return None, None, None

    # Wspólna siatka x — punkty oceny z pierwszego seeda
    ref_seed = next(iter(seed_series))
    x_grid = seed_series[ref_seed][0]

    interp_ys = []
    for xs, ys in seed_series.values():
        if len(xs) < 2:
            continue
        yi = np.interp(x_grid, xs, ys)
        interp_ys.append(yi)

    if not interp_ys:
        return None, None, None

    arr = np.array(interp_ys)
    return x_grid, arr.mean(axis=0), arr.std(axis=0)


def plot_task(task_runs: dict, task: str, env: str, metric: str, out_dir: Path):
    """Tworzy jeden wykres dla (env, task) ze wszystkimi algorytmami."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"{env} — {task}", fontsize=13)
    ax.set_xlabel("Frames collectées", fontsize=11)
    ax.set_ylabel(f"Moyenne {metric}", fontsize=11)

    plotted = False
    for algo, seed_data in sorted(task_runs.items()):
        style = ALGO_STYLE.get(algo, {"color": "black", "label": algo, "ls": "-"})
        x, mean, std = aggregate(seed_data, metric)
        if x is None:
            continue
        n_seeds = sum(1 for s in seed_data.values() if any(metric in v for _, _, v in s))
        label = f"{style['label']} (n={n_seeds})"
        ax.plot(x / 1e6, mean, color=style["color"], ls=style["ls"], lw=2, label=label)
        ax.fill_between(x / 1e6, mean - std, mean + std,
                        color=style["color"], alpha=0.15)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Klatki [×10⁶]", fontsize=11)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{env}__{task}__{metric}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Zapisano: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs", help="Folder z wynikami")
    parser.add_argument("--out", default="plots", help="Folder zapisu wykresów")
    parser.add_argument("--task", default=None, help="Filtruj po nazwie taska")
    parser.add_argument("--env", default=None, help="Filtruj po środowisku (vmas/smacv2)")
    parser.add_argument("--metric", default="return",
                        help="Metryka do rysowania (domyślnie: return)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out)

    print(f"Szukam JSON-ów w: {outputs_dir.resolve()}")
    runs = collect_runs(outputs_dir)

    if not runs:
        print("Nie znaleziono żadnych danych. Sprawdź ścieżkę --outputs_dir.")
        return

    # Grupowanie po (env, task)
    tasks = defaultdict(dict)
    for (env, task, algo), seed_data in runs.items():
        if args.env and env != args.env:
            continue
        if args.task and task != args.task:
            continue
        tasks[(env, task)][algo] = seed_data

    print(f"Znalezione kombinacje (env, task): {list(tasks.keys())}")

    for (env, task), task_runs in tasks.items():
        algos = list(task_runs.keys())
        print(f"  {env}/{task}: algorytmy = {algos}, "
              f"metryki próbka = {_sample_metrics(task_runs)}")
        plot_task(task_runs, task, env, args.metric, out_dir)

    print(f"\nGotowe. Wykresy zapisane w: {out_dir.resolve()}/")


def _sample_metrics(task_runs: dict) -> list:
    for seed_data in task_runs.values():
        for steps in seed_data.values():
            for _, _, val in steps:
                return [k for k in val.keys() if k != "step_count"]
    return []


if __name__ == "__main__":
    main()
