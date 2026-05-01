#!/usr/bin/env python3
"""
Ray dispatcher for MiniGrid BabyAI experiments.
"""

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import tempfile
import time
import shutil
from dataclasses import dataclass
from typing import List, Optional

try:
    import ray

    remote_decorator = ray.remote
except ImportError:
    ray = None
    remote_decorator = lambda x: x


@dataclass
class Job:
    group: str
    script_path: str
    args: List[str]
    name: str


def build_minigrid_cmd(env_id, variant, seed, extras=None) -> Job:
    cmd = [
        sys.executable,
        "-u",
        "fine_tuned/minigrid/r2d2_run.py",
        f"--env-id={env_id}",
        f"--variant={variant}",
        f"--seed={seed}",
    ]
    if extras:
        cmd.extend(extras)

    name = f"minigrid_{env_id}_{variant}_seed{seed}"
    if extras:
        name += "_" + "_".join([x.replace("--", "").replace("=", "") for x in extras])

    return Job("minigrid", "fine_tuned/minigrid/r2d2_run.py", cmd, name)


def generate_jobs() -> List[Job]:
    jobs = []
    seeds = [100, 101, 102]

    # 1. RL MiniGrid
    # KeyCorridorS4R3
    for algo, seed in itertools.product(
        ["r2d2", "r2d2_demir", "r2d2_rnd", "r2d2_ngu"], seeds
    ):
        jobs.append(build_minigrid_cmd("BabyAI-KeyCorridorS4R3-v0", algo, seed))
    # GoToObj
    for algo, seed in itertools.product(["r2d2", "r2d2_demir", "r2d2_rnd"], seeds):
        jobs.append(build_minigrid_cmd("BabyAI-GoToObj-v0", algo, seed))
    # UnlockPickup
    for algo, seed in itertools.product(["r2d2", "r2d2_demir"], seeds):
        jobs.append(build_minigrid_cmd("BabyAI-UnlockPickup-v0", algo, seed))

    # 2. MiniGrid ablations
    for seed in seeds:
        # beta_1=0.0 beta_2=1.0
        jobs.append(
            build_minigrid_cmd(
                "BabyAI-KeyCorridorS4R3-v0",
                "r2d2_demir",
                seed,
                ["--demir-beta1=0.0", "--demir-beta2=1.0"],
            )
        )
        # beta_1=1.0 beta_2=0.0
        jobs.append(
            build_minigrid_cmd(
                "BabyAI-KeyCorridorS4R3-v0",
                "r2d2_demir",
                seed,
                ["--demir-beta1=1.0", "--demir-beta2=0.0"],
            )
        )
        # MLP instead of IDM
        jobs.append(
            build_minigrid_cmd(
                "BabyAI-KeyCorridorS4R3-v0",
                "r2d2_demir",
                seed,
                ["--demir-encoder-type=mlp"],
            )
        )
        # without barlow twins
        jobs.append(
            build_minigrid_cmd(
                "BabyAI-KeyCorridorS4R3-v0",
                "r2d2_demir",
                seed,
                ["--demir-encoder-type=idm_no_barlow"],
            )
        )

    return jobs


@remote_decorator
def run_job_remote(job: Job) -> dict:
    """Runs a single job and returns zipped outputs and status."""
    import os
    import shutil
    import subprocess
    import tempfile

    work_dir = tempfile.mkdtemp(prefix=f"ray_worker_{job.name}_")
    cmd = list(job.args)

    if job.group in ["smacv2", "logic_env"]:
        cmd.append(f"hydra.run.dir={work_dir}")
    elif job.group == "minigrid":
        cmd.append(f"--save-dir={work_dir}")

    print(f"Executing {job.name} -> {' '.join(cmd)}")

    # Run the subprocess
    start_t = time.time()
    res = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    end_t = time.time()

    # Write stdout/err to the work_dir so it gets zipped
    with open(os.path.join(work_dir, "stdout.log"), "w") as f:
        f.write(res.stdout)
    with open(os.path.join(work_dir, "stderr.log"), "w") as f:
        f.write(res.stderr)

    # Zip the working directory
    zip_path = shutil.make_archive(work_dir, "zip", work_dir)

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    return {
        "name": job.name,
        "returncode": res.returncode,
        "time": end_t - start_t,
        "zip_bytes": zip_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--head", type=str, default="127.0.0.1", help="IP of ray head node"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_thesis_minigrid",
        help="Local directory for saving results",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print jobs and exit")
    args = parser.parse_args()

    jobs = generate_jobs()
    print(f"Generated {len(jobs)} MiniGrid jobs.")

    if args.dry_run:
        for job in jobs:
            print(" ".join(job.args))
        sys.exit(0)

    if ray is None:
        print("Ray not installed. Cannot run.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    ray.init(
        address=f"{args.head}:6379",
        runtime_env={
            "working_dir": ".",
            "excludes": [
                "logs_thesis",  # Stare wyniki
                "results_thesis",  # Środowisko wynikowe SMACv2
                "results_thesis_minigrid",  # Opcjonalne wykluczenie drugiego folderu
                ".git",  # Git database
                ".venv",  # Virtual env
                ".uv",  # UV files
                "__pycache__",  # Skompilowany cache
                "outputs",  # Stare outputy Hydra
                "runs",  # Niepotrzebne katalogi do logowania
                "benchmarl_demir.egg-info",
            ],
        },
    )

    # Queue them up
    futures = [run_job_remote.remote(j) for j in jobs]

    print("Submitted all MiniGrid jobs to Ray queue. Waiting for completion...")

    successes = 0
    failures = 0

    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        for done_ref in done:
            result = ray.get(done_ref)

            # Save the zip contents
            zip_dest = os.path.join(args.out_dir, f"{result['name']}.zip")
            with open(zip_dest, "wb") as f:
                f.write(result["zip_bytes"])

            if result["returncode"] == 0:
                print(
                    f"[OK] {result['name']} returned 0 in {result['time']:.2f}s. Saved to {zip_dest}"
                )
                successes += 1
            else:
                print(
                    f"[FAIL] {result['name']} returned {result['returncode']}. Saved logs to {zip_dest}"
                )
                failures += 1

    print(f"All done! Successes: {successes}, Failures: {failures}")


if __name__ == "__main__":
    main()
