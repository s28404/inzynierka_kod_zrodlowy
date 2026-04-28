# Author: Kajetan Frąckowiak, s28404 (2026)
# File written from scratch as part of an engineering thesis
# Description: Script for recording episodes of trained agents
#              to MP4 files (supports VMAS/MPE and SMACv2).

"""
Loads a training checkpoint and records episodes to an MP4 video file.

VMAS (MPE):  works headless (no monitor required)
SMACv2:      requires DISPLAY (e.g. Xvfb) — see instructions below

Usage:
    python record_video.py outputs/.../checkpoint.pt
    python record_video.py outputs/.../checkpoint.pt --episodes 3 --out video.mp4 --fps 20

For SMACv2 (without a monitor):
    Xvfb :99 -screen 0 1024x768x24 &
    DISPLAY=:99 python record_video.py ...
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

# ── imageio / imageio-ffmpeg ────────────────────────────────────────────────
try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.hydra_config import reload_experiment_from_file


def record(
    checkpoint_file: str,
    n_episodes: int = 1,
    out_path: str | None = None,
    fps: int = 20,
    deterministic: bool = True,
):
    """
    Records one or more episodes from a trained agent and saves them as an MP4 video.
    """
    if not _HAS_IMAGEIO:
        raise ImportError(
            "imageio and imageio-ffmpeg are required:\n"
            "    pip install imageio imageio-ffmpeg"
        )

    checkpoint_file = str(Path(checkpoint_file).resolve())
    print(f"Loading checkpoint: {checkpoint_file}")

    # ── Load experiment from checkpoint ─────────────────────────────────────
    experiment = reload_experiment_from_file(checkpoint_file)

    # Force rendering to be enabled (it might be disabled in the saved config)
    experiment.config.render = True
    # Reset test environment so it picks up the new config
    if hasattr(experiment, "_test_env"):
        del experiment._test_env

    env = experiment.test_env

    if not experiment.task.has_render(env):
        raise RuntimeError(
            f"Environment {type(env).__name__} does not support rendering. "
            "For SMACv2 make sure you have a DISPLAY set up (e.g. Xvfb)."
        )

    # ── Collect frames ──────────────────────────────────────────────────────
    all_frames = []
    exploration = (
        ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    )

    print(f"Recording {n_episodes} episode(s)...")
    with set_exploration_type(exploration):
        for ep in range(n_episodes):
            ep_frames = []

            def _callback(e, td):
                frame = experiment.task.__class__.render_callback(experiment, e, td)
                ep_frames.append(frame)

            env.set_seed(experiment.seed + ep)
            env.rollout(
                max_steps=experiment.max_steps,
                policy=experiment.policy,
                callback=_callback,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )
            all_frames.extend(ep_frames)
            print(f"  Episode {ep+1}/{n_episodes}: {len(ep_frames)} frames")

    if not all_frames:
        raise RuntimeError("No frames were collected. Check if rendering is working correctly.")

    # ── Save as MP4 ─────────────────────────────────────────────────────────
    if out_path is None:
        ckpt_path = Path(checkpoint_file)
        # Use experiment directory for output
        exp_dir = ckpt_path.parent
        out_path = str(exp_dir / f"video_{ckpt_path.stem}.mp4")

    # Ensure frames are uint8 RGB format
    frames_np = []
    for f in all_frames:
        if not isinstance(f, np.ndarray):
            f = np.array(f)
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)
        # Ensure shape is (H, W, 3)
        if f.ndim == 2:
            f = np.stack([f, f, f], axis=-1)
        elif f.shape[-1] == 4:
            f = f[..., :3]
        frames_np.append(f)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()

    total_s = len(frames_np) / fps
    print(f"\nSaved {len(frames_np)} frames ({total_s:.1f}s) → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record episodes of a trained agent to MP4 video."
    )
    parser.add_argument(
        "checkpoint_file",
        type=str,
        help="Path to the .pt checkpoint file (e.g. outputs/.../checkpoint_100000.pt)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to record (default: 1)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output MP4 file path (default: next to the checkpoint)",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Video FPS (default: 20)",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic policy instead of deterministic",
    )
    args = parser.parse_args()

    record(
        checkpoint_file=args.checkpoint_file,
        n_episodes=args.episodes,
        out_path=args.out,
        fps=args.fps,
        deterministic=not args.stochastic,
    )