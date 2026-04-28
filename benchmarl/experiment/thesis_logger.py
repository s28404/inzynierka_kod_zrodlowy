"""
Module implementing own thesis logger callback for BenchMARL.

Author: Kajetan Frąckowiak
Date: 2026

Description: This module defines the `ThesisJSONLoggerCallback`, a custom callback for logging training and evaluation metrics 
in a structured JSON format. It is designed to be used with BenchMARL experiments, 
allowing for easy aggregation and analysis of results for the engineering thesis. The callback collects metadata 
at the start of training and appends evaluation metrics after each evaluation loop, saving everything to a JSON 
file named after the experiment. This approach ensures that all relevant information is captured in a single, 
easily accessible format for later analysis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from benchmarl.experiment.callback import Callback


class ThesisJSONLoggerCallback(Callback):
    """
    Custom JSON logger for BenchMARL used for the engineering thesis.
    Collects training and evaluation logs per evaluation interval and safely overwrites the JSON file.
    """

    def __init__(self, log_dir: str = "logs"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.file_path: Optional[Path] = None

        self.data_registry = {"metadata": {}, "metrics": []}

    def on_train_step(self, batch, group: str):
        """Standard BenchMARL callback - not used in this implementation."""
        pass

    def on_setup(self):
        """Called on startup - ensures the callback is properly loaded."""
        print("[THESIS_LOGGER] Callback loaded successfully!")

    def on_train_start(self, experiment) -> None:
        """
        Called once at the start of the experiment. Gathers metadata here.
        """
        print("[THESIS_LOGGER] on_train_start called!")

        # Extract key information from the BenchMARL experiment
        algo_name = experiment.algorithm.name.lower()

        # If DEMIR/NGU algorithms inherit from QMIX, algo_name will be "qmix".
        # Try to extract the actual name from command line arguments.
        for arg in sys.argv:
            if arg.startswith("algorithm="):
                algo_name = arg.split("=")[1]
                break

        # Fallback: try to get algorithm name from Hydra config
        if algo_name in ("qmix", ""):
            try:
                from hydra.core.hydra_config import HydraConfig
                choices = HydraConfig.get().runtime.choices
                if "algorithm" in choices:
                    algo_name = choices["algorithm"]
            except Exception:
                pass

        task_name = experiment.task.name
        seed = experiment.seed

        # Safely extract hyperparameters
        cfg = getattr(experiment, "config", {})
        algo_cfg = getattr(experiment.algorithm, "config", {})

        self.data_registry["metadata"] = {
            "experiment_id": experiment.name,
            "algorithm": algo_name,
            "environment": task_name,
            "seed": seed,
            "hyperparameters": {
                "beta1": getattr(algo_cfg, "beta1", None),
                "beta2": getattr(algo_cfg, "beta2", None),
                "intrinsic_module_type": getattr(algo_cfg, "intrinsic_module_type", None),
                "off_policy_memory_size": getattr(cfg, "buffer_size", None),
                "lr": getattr(algo_cfg, "lr", None),
            },
        }

        # Save file using the same name as the W&B run
        filename = f"{experiment.name}.json"
        self.file_path = self.log_dir / filename

        print(f"[THESIS_LOGGER] Initialized: saving to {self.file_path}")
        self.save_to_disk()

    def on_evaluation_end(self, rollouts) -> None:
        """
        Called after each evaluation loop.
        Collects training and evaluation data into JSON.

        Args:
            rollouts: List[TensorDictBase] containing evaluation data
        """
        print(
            f"[THESIS_LOGGER] on_evaluation_end called! Number of rollouts: {len(rollouts)}"
        )

        experiment = self.experiment
        step = experiment.n_iters_performed

        # Compute evaluation statistics from rollouts
        eval_stats = self._compute_eval_stats(rollouts)

        # Extract recent training metrics from the experiment logger
        train_metrics = self._extract_recent_train_metrics()

        # Clean evaluation statistics
        eval_clean = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in eval_stats.items()
        }

        # Create unified metrics dictionary for one row
        current_metrics = {
            "frame": experiment.total_frames,
            "step": step,
            # Training metrics (last logged)
            "train_return_mean": train_metrics.get("train_return_mean", None),
            "train_intrinsic_reward_mean": train_metrics.get("train_intrinsic_reward_mean", None),
            "q_loss": train_metrics.get("q_loss", None),
            "intrinsic_module_loss": train_metrics.get("intrinsic_module_loss", None),
            # Evaluation metrics (basic)
            "eval_return_mean": eval_clean.get("eval/episode_reward", None),
            "eval_return_std": eval_clean.get("eval/episode_reward_std", None),
            # DEMIR-specific metrics
            "loss_idm": train_metrics.get("loss_idm", None),
            "loss_barlow_twins": train_metrics.get("loss_barlow_twins", None),
            "intrinsic_reward_Q_mean": train_metrics.get("intrinsic_reward_Q_mean", None),
            "intrinsic_reward_N_mean": train_metrics.get("intrinsic_reward_N_mean", None),
            "current_beta1": train_metrics.get("current_beta1", None),
            "current_beta2": train_metrics.get("current_beta2", None),
        }

        # Environment-specific metrics
        env_name = self.data_registry["metadata"].get("environment", "").lower()

        if "smac" in env_name:
            # SMACv2 metrics
            current_metrics["eval_win_rate"] = eval_clean.get("eval/win_rate", None)

        elif any(x in env_name for x in ["factory", "logic", "synchronized"]):
            # Custom Reward Machine metrics for logic environments
            current_metrics["eval_success_rate"] = eval_clean.get("eval/success_rate", None)
            current_metrics["eval_bottleneck_reached_rate"] = eval_clean.get(
                "eval/bottleneck_reached_rate", None
            )
            current_metrics["eval_rm_state_0_ratio"] = eval_clean.get("eval/rm_state_0_ratio", None)
            current_metrics["eval_rm_state_1_ratio"] = eval_clean.get("eval/rm_state_1_ratio", None)
            current_metrics["eval_rm_state_2_ratio"] = eval_clean.get("eval/rm_state_2_ratio", None)
            current_metrics["eval_avg_steps_to_RM1"] = eval_clean.get("eval/avg_steps_to_RM1", None)

        self.data_registry["metrics"].append(current_metrics)
        self.save_to_disk()

    def _extract_recent_train_metrics(self) -> Dict[str, Any]:
        """
        Extracts the most recent training metrics.
        Currently returns empty dict - can be extended in the future.
        """
        return {}

    def _compute_eval_stats(self, rollouts) -> Dict[str, Any]:
        """
        Computes evaluation metrics from rollouts.

        Args:
            rollouts: List[TensorDictBase] from evaluation

        Returns:
            Dict containing evaluation statistics
        """
        eval_stats = {}

        try:
            rewards = []
            for td in rollouts:
                reward = None
                # Try different possible reward locations
                if ("next", "agents", "reward") in td.keys(True, True):
                    reward = td.get(("next", "agents", "reward"))
                elif ("next", "reward") in td.keys(True, True):
                    reward = td.get(("next", "reward"))

                if reward is not None and reward.numel() > 0:
                    rewards.append(reward.sum(0).mean().item())

            if rewards:
                eval_stats["eval/episode_reward"] = float(np.mean(rewards))
                eval_stats["eval/episode_reward_std"] = float(np.std(rewards))
        except Exception as e:
            print(f"[THESIS_LOGGER] Error calculating reward: {e}")

        # Extract battle_won for SMACv2
        try:
            battle_won = []
            for td in rollouts:
                if ("next", "info", "battle_won") in td.keys(True, True):
                    bw = td.get(("next", "info", "battle_won"))
                    if bw.numel() > 0:
                        battle_won.append(bw.mean().item())
            if battle_won:
                eval_stats["eval/win_rate"] = float(np.mean(battle_won))
        except Exception:
            pass

        # Extract success_rate for logic_env
        try:
            if "logic_env" in self.data_registry["metadata"].get("environment", "").lower():
                success = []
                for td in rollouts:
                    if ("next", "info", "success_rate") in td.keys(True, True):
                        sr = td.get(("next", "info", "success_rate"))
                        if sr.numel() > 0:
                            success.append(sr.mean().item())
                if success:
                    eval_stats["eval/success_rate"] = float(np.mean(success))
        except Exception:
            pass

        return eval_stats

    def save_to_disk(self):
        """Safely save data to disk using a temporary file to prevent corruption."""
        if self.file_path is None:
            return

        temp_path = self.file_path.with_suffix(".json.tmp")

        try:
            with open(temp_path, "w") as f:
                json.dump(self.data_registry, f, indent=4)
            # Atomic replace
            temp_path.replace(self.file_path)
        except Exception as e:
            print(f"Error saving JSON: {e}")