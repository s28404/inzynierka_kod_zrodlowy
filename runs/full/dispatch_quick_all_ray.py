#!/usr/bin/env python3
import os, sys, time, shutil, tempfile, argparse, subprocess
from dataclasses import dataclass
try:
    import ray
    remote_decorator = ray.remote
except ImportError:
    ray = None
    remote_decorator = lambda x: x

@dataclass
class Job:
    group: str
    name: str
    args: list

def generate_jobs():
    jobs = []
    
    # BabyAI
    envs = ["BabyAI-GoToObj-v0", "BabyAI-KeyCorridorS4R3-v0", "BabyAI-UnlockPickup-v0"]
    for env in envs:
        cmd = [
            sys.executable, "-u", "fine_tuned/minigrid/r2d2_run.py",
            "--variant", "r2d2_demir", "--env-id", env, "--seed", "2",
            "--total-steps", "4000", "--warmup-steps", "500", "--train-every", "4",
            "--batch-size", "16", "--replay-capacity-sequences", "2000",
            "--burn-in", "8", "--unroll-len", "16", "--n-step", "3",
            "--num-threads", "2", "--log-interval", "1000",
            "--eval-interval", "4000", "--eval-episodes", "1",
            "--checkpoint-interval", "1000"
        ]
        jobs.append(Job("minigrid", f"quick_babyai_{env}_demir_seed2", cmd))
        
    # Logic Env
    cmd = [
        sys.executable, "-u", "fine_tuned/logic_env/logic_env_run.py",
        "task=logic_env/synchronized", "algorithm=demir", "experiment.max_n_frames=6000",
        "experiment.evaluation_interval=6000", "experiment.evaluation_episodes=1",
        "experiment.checkpoint_interval=6000", "experiment.checkpoint_at_end=true",
        "experiment.keep_checkpoints_num=3", "seed=2"
    ]
    jobs.append(Job("logic_env", "quick_logic_env_synchronized_demir_seed2", cmd))
    
    # SMACv2
    maps = ["protoss_5_vs_5", "protoss_10_vs_11", "terran_10_vs_10"]
    for m in maps:
        cmd = [
            sys.executable, "-u", "fine_tuned/smacv2/smacv2_run.py",
            f"task=smacv2/{m}", "algorithm=demir", "experiment.off_policy_n_envs_per_worker=1",
            "experiment.off_policy_collected_frames_per_batch=1000", "experiment.off_policy_train_batch_size=64",
            "experiment.parallel_collection=false", "experiment.render=false", "experiment.buffer_device=cpu",
            "experiment.off_policy_memory_size=50000", "experiment.max_n_frames=1000",
            "experiment.evaluation_interval=1000", "experiment.evaluation_episodes=1",
            "experiment.checkpoint_interval=1000", "experiment.checkpoint_at_end=true",
            "experiment.keep_checkpoints_num=3", "seed=2"
        ]
        jobs.append(Job("smacv2", f"quick_smacv2_{m}_demir_seed2", cmd))

    return jobs

@remote_decorator
def run_job_remote(job: Job) -> dict:
    work_dir = tempfile.mkdtemp(prefix=f"ray_worker_{job.name}_")
    cmd = list(job.args)
    if job.group in ["smacv2", "logic_env"]: cmd.append(f"hydra.run.dir={work_dir}")
    elif job.group == "minigrid": cmd.append(f"--save-dir={work_dir}")
    print(f"Executing {job.name}")
    start_t = time.time()
    res = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    end_t = time.time()
    with open(os.path.join(work_dir, "stdout.log"), "w") as f: f.write(res.stdout)
    with open(os.path.join(work_dir, "stderr.log"), "w") as f: f.write(res.stderr)
    zip_path = shutil.make_archive(work_dir, "zip", work_dir)
    with open(zip_path, "rb") as f: zip_bytes = f.read()
    shutil.rmtree(work_dir, ignore_errors=True)
    if os.path.exists(zip_path): os.remove(zip_path)
    return {"name": job.name, "returncode": res.returncode, "time": end_t - start_t, "zip_bytes": zip_bytes}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=str, default="127.0.0.1")
    parser.add_argument("--out-dir", type=str, default="results_quick_tests")
    args = parser.parse_args()
    jobs = generate_jobs()
    print(f"Generated {len(jobs)} total quick jobs.")
    os.makedirs(args.out_dir, exist_ok=True)
    ray.init(address=f"{args.head}:6379", runtime_env={"working_dir": ".", "excludes": [".git", "logs_thesis", "results_thesis", "results_thesis_minigrid", "results_quick_tests", "outputs", ".venv", ".uv", "__pycache__", "runs", "benchmarl_demir.egg-info"]})
    futures = [run_job_remote.remote(j) for j in jobs]
    for done_ref in futures:
        done, _ = ray.wait([done_ref], num_returns=1)
        res = ray.get(done[0])
        zpath = os.path.join(args.out_dir, f"{res['name']}.zip")
        with open(zpath, "wb") as f: f.write(res["zip_bytes"])
        status = "OK" if res["returncode"] == 0 else "FAIL"
        print(f"[{status}] {res['name']} returned {res['returncode']} in {res['time']:.2f}s. Saved to {zpath}")

if __name__ == "__main__": main()
