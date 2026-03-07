# Autor: Kajetan Frąckowiak, s28404 (2026)
# Plik napisany od podstaw w ramach pracy inżynierskiej
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Skrypt do nagrywania epizodów wytrenowanych agentów
#        do pliku MP4 (VMAS/MPE i SMACv2).

"""
Wczytuje checkpoint z treningu i nagrywa epizod do pliku MP4.

VMAS (MPE):  działa headless, bez monitora
SMACv2:      wymaga DISPLAY (np. Xvfb) — patrz instrukcja poniżej

Użycie:
    python record_video.py outputs/.../checkpoint.pt
    python record_video.py outputs/.../checkpoint.pt --episodes 3 --out video.mp4 --fps 20

Dla SMACv2 (bez monitora):
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
    if not _HAS_IMAGEIO:
        raise ImportError(
            "imageio i imageio-ffmpeg są wymagane:\n"
            "    pip install imageio imageio-ffmpeg"
        )

    checkpoint_file = str(Path(checkpoint_file).resolve())
    print(f"Wczytuję checkpoint: {checkpoint_file}")

    # ── Wczytanie eksperymentu ───────────────────────────────────────────────
    experiment = reload_experiment_from_file(checkpoint_file)

    # Wymuszamy render=True (może być False w zapisanym configu)
    experiment.config.render = True
    # Resetujemy środowisko testowe żeby użyło nowego configu
    if hasattr(experiment, "_test_env"):
        del experiment._test_env

    env = experiment.test_env

    if not experiment.task.has_render(env):
        raise RuntimeError(
            f"Środowisko {type(env).__name__} nie wspiera renderowania. "
            "Dla SMACv2 upewnij się że masz DISPLAY (Xvfb)."
        )

    # ── Zbieranie klatek ────────────────────────────────────────────────────
    all_frames = []
    exploration = (
        ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    )

    print(f"Nagrywam {n_episodes} epizod(ów)...")
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
            print(f"  Epizod {ep+1}/{n_episodes}: {len(ep_frames)} klatek")

    if not all_frames:
        raise RuntimeError("Nie zebrano żadnych klatek. Sprawdź czy render działa.")

    # ── Zapis MP4 ───────────────────────────────────────────────────────────
    if out_path is None:
        ckpt_path = Path(checkpoint_file)
        # Szukamy nazwy eksperymentu w ścieżce
        exp_dir = ckpt_path.parent
        out_path = str(exp_dir / f"video_{ckpt_path.stem}.mp4")

    # Upewniamy się że klatki są uint8 RGB
    frames_np = []
    for f in all_frames:
        if not isinstance(f, np.ndarray):
            f = np.array(f)
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)
        # Upewnij się że kształt to (H, W, 3)
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
    print(f"\nZapisano {len(frames_np)} klatek ({total_s:.1f}s) → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nagrywa epizod wytrenowanego agenta do MP4."
    )
    parser.add_argument(
        "checkpoint_file",
        type=str,
        help="Ścieżka do pliku .pt checkpointu (np. outputs/.../checkpoint_100000.pt)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Liczba epizodów do nagrania (domyślnie: 1)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Plik wyjściowy MP4 (domyślnie: obok checkpointu)",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="FPS wideo (domyślnie: 20)",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Użyj stochastycznej polityki zamiast deterministycznej",
    )
    args = parser.parse_args()

    record(
        checkpoint_file=args.checkpoint_file,
        n_episodes=args.episodes,
        out_path=args.out,
        fps=args.fps,
        deterministic=not args.stochastic,
    )
