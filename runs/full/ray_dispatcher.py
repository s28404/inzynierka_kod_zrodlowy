#!/usr/bin/env python3
"""Minimal Ray dispatcher: submit SMACv2 runs as Ray tasks.

Usage:
  python3 ray_dispatcher.py --head 172.17.107.3

This script expects Ray to be running (head on the provided address, workers connected).
Each Ray task runs the training command locally on the worker using subprocess.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List

try:
    import ray
except Exception as e:
    print(
        "Ray is not installed. Install it with: python3 -m pip install ray[default]",
        file=sys.stderr,
    )
    raise

TASKS = ["smacv2/protoss_5_vs_5", "smacv2/protoss_10_vs_11", "smacv2/terran_10_vs_10"]
ALGOS = ["qmix", "rnd", "ngu", "demir"]


@dataclass
class Job:
    task: str
    algo: str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head", required=True, help="Ray head IP or address (e.g. 172.17.107.3)"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_cmd(job: Job, seed: int) -> List[str]:
    return [
        sys.executable,
        "-u",
        "fine_tuned/smacv2/smacv2_run.py",
        f"task={job.task}",
        f"algorithm={job.algo}",
        f"seed={seed}",
        "experiment.sampling_device=cpu",
        "experiment.train_device=cpu",
        "experiment.lr=0.00005",
        "experiment.max_n_frames=10000000",
        "experiment.off_policy_memory_size=150000",
        "experiment.off_policy_collected_frames_per_batch=500",
        "experiment.off_policy_train_batch_size=64",
        "experiment.evaluation_interval=250000",
        "experiment.evaluation_episodes=32",
        "experiment.buffer_device=cpu",
        "experiment.checkpoint_interval=1000000",
        "experiment.clip_grad_val=10.0",
        "experiment.parallel_collection=false",
        "experiment.render=false",
    ]


@ray.remote
def run_job_remote(cmd: List[str]) -> int:
    print(f"Running: {' '.join(shlex.quote(p) for p in cmd)}")
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    ray.init(address=f"{args.head}:6379")

    jobs = [Job(task=t, algo=a) for t in TASKS for a in ALGOS]

    futures = []
    for job in jobs:
        cmd = build_cmd(job, args.seed)
        if args.dry_run:
            print("DRY-RUN:", " ".join(shlex.quote(p) for p in cmd))
            continue
        futures.append(run_job_remote.remote(cmd))

    if not args.dry_run:
        results = ray.get(futures)
        for r in results:
            if r != 0:
                print("One job failed (exit!=0)", file=sys.stderr)
                return 2

    print("All Ray-submitted jobs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
