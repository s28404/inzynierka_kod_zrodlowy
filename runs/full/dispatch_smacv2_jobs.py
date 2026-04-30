#!/usr/bin/env python3

"""Dispatch SMACv2 experiments across multiple machines via SSH.

Each host runs at most one experiment at a time. As soon as a host finishes,
the next queued experiment is assigned to it.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import List


TASKS = ["smacv2/protoss_5_vs_5", "smacv2/protoss_10_vs_11", "smacv2/terran_10_vs_10"]
ALGOS = ["qmix", "rnd", "ngu", "demir"]


@dataclass(frozen=True)
class Job:
    task: str
    algo: str

    @property
    def label(self) -> str:
        return f"{self.algo} on {self.task}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        required=True,
        help="Path to a text file with one SSH host per line, for example user@192.168.1.10",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository directory on every machine.",
    )
    parser.add_argument(
        "--python-bin",
        default="uv run python",
        help="Command used to start Python inside the repo on each host.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed passed to every experiment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def load_hosts(hosts_file: str) -> List[str]:
    hosts: List[str] = []
    for raw_line in Path(hosts_file).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        hosts.append(line)
    if not hosts:
        raise SystemExit(f"No hosts found in {hosts_file}")
    return hosts


def build_jobs() -> List[Job]:
    return [Job(task=task, algo=algo) for task in TASKS for algo in ALGOS]


def build_remote_command(repo_dir: str, python_bin: str, seed: int, job: Job) -> str:
    args = [
        *shlex.split(python_bin),
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
    return f"cd {shlex.quote(repo_dir)} && {' '.join(shlex.quote(arg) for arg in args)}"


def run_job(
    host: str, repo_dir: str, python_bin: str, seed: int, job: Job, dry_run: bool
) -> int:
    remote_command = build_remote_command(repo_dir, python_bin, seed, job)
    ssh_command = ["ssh", host, remote_command]
    print(f"[{host}] START {job.label}", flush=True)
    if dry_run:
        print(" ".join(shlex.quote(part) for part in ssh_command), flush=True)
        return 0
    completed = subprocess.run(ssh_command, check=False)
    print(f"[{host}] END   {job.label} (exit={completed.returncode})", flush=True)
    return completed.returncode


def main() -> int:
    args = parse_args()
    hosts = load_hosts(args.hosts)
    jobs = build_jobs()

    if len(hosts) > len(jobs):
        hosts = hosts[: len(jobs)]

    pending_jobs = jobs.copy()
    running = {}

    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        for host in hosts:
            if not pending_jobs:
                break
            job = pending_jobs.pop(0)
            running[
                executor.submit(
                    run_job,
                    host,
                    args.repo_dir,
                    args.python_bin,
                    args.seed,
                    job,
                    args.dry_run,
                )
            ] = (host, job)

        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                host, job = running.pop(future)
                exit_code = future.result()
                if exit_code != 0:
                    print(
                        f"[{host}] Job failed: {job.label}", file=sys.stderr, flush=True
                    )
                    return exit_code
                if pending_jobs:
                    next_job = pending_jobs.pop(0)
                    running[
                        executor.submit(
                            run_job,
                            host,
                            args.repo_dir,
                            args.python_bin,
                            args.seed,
                            next_job,
                            args.dry_run,
                        )
                    ] = (host, next_job)

    print("All experiments finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
