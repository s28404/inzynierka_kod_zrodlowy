#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# ==============================================================================
# Author: Kajetan Frąckowiak
# Date: 2026
# Modification in this file
#   [1] # Calculate win_rate from the battle_won flag and saves to W&B and JSON.
#   [2] # Add full method for logging DEMIR metrics: EDM/EFM size, beta, scale.
#   [3] # Add edm_size and efm_size as measures of exploration (increasing with frame count).
#   [4] # Add beta1, beta2, demir_scale logged continuously — for filtering in W&B.
#   [5] # Add shaping_reward_mean, novelty_mean, quality_mean (if provided).
#   [6] # Add list of 5 W&B keys ready for plot export for publication.
#   [7] # Use `experiment_name` instead of a date for file naming. Catching environmental logs.
#   [8] # Debug: write when `log_collection` is called for tracing.
#   [9] # CSV logging hook: a small CSV row for thesis analysis if the experiment provided `csv_path` via `_experiment_ref`.
#   [10] # Append a flattened JSONL line to a central `logs_thesis directory for easy aggregation of thesis metrics.
#   [11] # Custom Reward Machine metrics for logic_env / synchronized tasks: success, bottleneck rates and RM state ratios.
# ==============================================================================
import json
import os
import warnings
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchrl

from tensordict import TensorDictBase
from torch import Tensor

from torchrl.record import TensorboardLogger
from torchrl.record.loggers import get_logger
from torchrl.record.loggers.wandb import WandbLogger

from benchmarl.environments import Task


class Logger:
    def __init__(
        self,
        experiment_name: str,
        folder_name: str,
        experiment_config,
        algorithm_name: str,
        environment_name: str,
        task_name: str,
        model_name: str,
        group_map: Dict[str, List[str]],
        seed: int,
        project_name: str,
        wandb_extra_kwargs: Dict[str, Any],
    ):
        self.experiment_config = experiment_config
        self.algorithm_name = algorithm_name
        self.environment_name = environment_name
        self.task_name = task_name
        self.model_name = model_name
        self.group_map = group_map
        self.seed = seed
        #############################
        # [7] Start. Use `experiment_name` instead of a date for file naming.
        #############################
        self.experiment_name = experiment_name
        #############################
        # [7] End.
        #############################

        if experiment_config.create_json:
            self.json_writer = JsonWriter(
                folder=folder_name,
                name=experiment_name + ".json",
                algorithm_name=algorithm_name,
                task_name=task_name,
                environment_name=environment_name,
                seed=seed,
            )
        else:
            self.json_writer = None

        self.loggers: List[torchrl.record.loggers.Logger] = []
        for logger_name in experiment_config.loggers:
            wandb_project = wandb_extra_kwargs.get("project", project_name)
            if wandb_project != project_name:
                raise ValueError(
                    f"wandb_extra_kwargs.project ({wandb_project}) is different from the project_name ({project_name})"
                )
            self.loggers.append(
                get_logger(
                    logger_type=logger_name,
                    logger_name=folder_name,
                    experiment_name=experiment_name,
                    wandb_kwargs={
                        "group": task_name,
                        "id": experiment_name,
                        "project": project_name,
                        **wandb_extra_kwargs,
                    },
                )
            )

    def log_hparams(self, **kwargs):
        for logger in self.loggers:
            if isinstance(logger, TensorboardLogger):
                # Tensorboard does not like nested dictionaries -> flatten them
                def flatten(dictionary, parent_key="", separator="_"):
                    items = []
                    for key, value in dictionary.items():
                        new_key = parent_key + separator + key if parent_key else key
                        if isinstance(value, MutableMapping):
                            items.extend(
                                flatten(value, new_key, separator=separator).items()
                            )
                        elif isinstance(value, Sequence):
                            for i, v in enumerate(value):
                                items.append((new_key + separator + str(i), v))
                        else:
                            items.append((new_key, value))
                    return dict(items)

                # Convert any non-supported values
                for key, value in kwargs.items():
                    if not isinstance(value, (int, float, str, Tensor)):
                        kwargs[key] = str(value)

                logger.log_hparams(flatten(kwargs))
            else:
                logger.log_hparams(kwargs)

    def log_collection(
        self,
        batch: TensorDictBase,
        task: Task,
        total_frames: int,
        step: int,
    ) -> float:
        #############################
        # [8] Start. Debug: write when `log_collection` is called for tracing.
        #############################
        with open("/tmp/log_collection_called.txt", "a") as f:
            f.write(
                f"[log_collection] CALLED with total_frames={total_frames}, step={step}\n"
            )
            f.flush()
        #############################
        # [8] End.
        #############################

        to_log = {}
        groups_episode_rewards = []
        gobal_done = self._get_global_done(batch)  # Does not have agent dim
        any_episode_ended = gobal_done.nonzero().numel() > 0
        if not any_episode_ended:
            warnings.warn(
                "No episode terminated this iteration and thus the episode rewards will be NaN, "
                "this is normal if your horizon is longer then one iteration. Learning is proceeding fine."
                "The episodes will probably terminate in a future iteration."
            )
        for group in self.group_map.keys():
            group_episode_rewards = self._log_individual_and_group_rewards(
                group,
                batch,
                gobal_done,
                any_episode_ended,
                to_log,
                log_individual_agents=False,  # Turn on if you want single agent granularity
            )
            # group_episode_rewards has shape (n_episodes) as we took the mean over agents in the group
            groups_episode_rewards.append(group_episode_rewards)

            if "info" in batch.get(("next", group)).keys():
                to_log.update(
                    {
                        f"collection/{group}/info/{key}": value.to(torch.float)
                        .mean()
                        .item()
                        for key, value in batch.get(("next", group, "info")).items()
                    }
                )
        if "info" in batch.keys():
            to_log.update(
                {
                    f"collection/info/{key}": value.to(torch.float).mean().item()
                    for key, value in batch.get(("next", "info")).items()
                }
            )
        to_log.update(task.log_info(batch))
        # global_episode_rewards has shape (n_episodes) as we took the mean over groups
        global_episode_rewards = self._log_global_episode_reward(
            groups_episode_rewards, to_log, prefix="collection"
        )

        self.log(to_log, step=step)

        #############################
        # [9] Start. CSV logging hook: a small CSV row for thesis analysis if the experiment provided `csv_path` via `_experiment_ref`.
        #############################
        try:
            if (
                hasattr(self, "_experiment_ref")
                and self._experiment_ref
                and hasattr(self._experiment_ref, "csv_path")
                and self._experiment_ref.csv_path
            ):
                import csv
                import datetime
                from pathlib import Path

                csv_path = (
                    Path(self._experiment_ref.csv_path)
                    if not isinstance(self._experiment_ref.csv_path, Path)
                    else self._experiment_ref.csv_path
                )

                row = {
                    "frame": total_frames,
                    "step": step,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "eval_return_mean": global_episode_rewards.mean().item(),
                    "algorithm": getattr(
                        self._experiment_ref, "algorithm_name", "unknown"
                    ),
                    "seed": getattr(self._experiment_ref, "seed", "unknown"),
                }

                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writerow(row)
        except Exception:
            pass  # Silent fail
        #############################
        # [9] End.
        #############################

        return global_episode_rewards.mean().item()

    def log_training(self, group: str, training_td: TensorDictBase, step: int):
        if not len(self.loggers):
            return
        to_log = {
            f"train/{group}/{key}": value.mean().item()
            for key, value in training_td.items()
        }
        self.log(to_log, step=step)

    def log_evaluation(
        self,
        rollouts: List[TensorDictBase],
        total_frames: int,
        step: int,
        video_frames: Optional[List] = None,
    ):
        if (
            not len(self.loggers) and not self.experiment_config.create_json
        ) or not len(rollouts):
            return

        # Cut rollouts at first done
        max_length_rollout_0 = 0
        for i in range(len(rollouts)):
            r = rollouts[i]
            next_done = self._get_global_done(r).squeeze(-1)

            # First done index for this traj
            done_index = next_done.nonzero(as_tuple=True)[0]
            if done_index.numel() > 0:
                done_index = done_index[0]
                r = r[: done_index + 1]
            if i == 0:
                max_length_rollout_0 = max(r.batch_size[0], max_length_rollout_0)
            rollouts[i] = r

        to_log = {}
        json_metrics = {}
        for group in self.group_map.keys():
            # returns has shape (n_episodes)
            returns = torch.stack(
                [self._get_reward(group, td).sum(0).mean() for td in rollouts],
                dim=0,
            )
            self._log_min_mean_max(
                to_log, f"eval/{group}/reward/episode_reward", returns
            )
            json_metrics[group + "_return"] = returns

        mean_group_return = self._log_global_episode_reward(
            list(json_metrics.values()), to_log, prefix="eval"
        )
        # mean_group_return has shape (n_episodes) as we take the mean groups
        json_metrics["return"] = mean_group_return

        to_log["eval/reward/episode_len_mean"] = sum(
            td.batch_size[0] for td in rollouts
        ) / len(rollouts)

        #############################
        # [1] Start. Calculate win_rate from the battle_won flag and saves to W&B and JSON.
        #############################
        try:
            battle_won_per_ep = []
            for td in rollouts:
                done = self._get_global_done(td).squeeze(-1)
                bw = td.get(("next", "info", "battle_won"), None)
                if bw is not None and done.any():
                    battle_won_per_ep.append(bw[done].to(torch.float).mean().item())

            if battle_won_per_ep:
                win_rate = float(np.mean(battle_won_per_ep))
                to_log["eval/info/win_rate"] = win_rate
                json_metrics["win_rate"] = torch.tensor(
                    [float(w) for w in battle_won_per_ep]
                )
        except Exception:
            pass  # Non-SMACv2 task
        #############################
        # [1] End.
        #############################

        #############################
        # [11] Start. Custom Reward Machine metrics for logic_env / synchronized tasks: success, bottleneck rates and RM state ratios.
        #############################
        try:
            if "logic_env" in self.task_name or "synchronized" in self.task_name:
                success_rates = []
                bottleneck_rates = []
                rm_0_ratios = []
                rm_1_ratios = []
                rm_2_ratios = []

                for td in rollouts:
                    obs = td.get(("next", "agents", "observation"))
                    rm_state_seq = obs[:, 0, 6]

                    max_rm = rm_state_seq.max().item()
                    success_rates.append(1.0 if max_rm >= 2 else 0.0)
                    bottleneck_rates.append(1.0 if max_rm >= 1 else 0.0)

                    total_steps = rm_state_seq.numel()
                    rm_0_ratios.append((rm_state_seq == 0).sum().item() / total_steps)
                    rm_1_ratios.append((rm_state_seq == 1).sum().item() / total_steps)
                    rm_2_ratios.append((rm_state_seq == 2).sum().item() / total_steps)

                if success_rates:
                    to_log["eval/info/eval_success_rate"] = float(
                        np.mean(success_rates)
                    )
                    to_log["eval/info/eval_bottleneck_reached_rate"] = float(
                        np.mean(bottleneck_rates)
                    )
                    to_log["eval/info/eval_rm_state_0_ratio"] = float(
                        np.mean(rm_0_ratios)
                    )
                    to_log["eval/info/eval_rm_state_1_ratio"] = float(
                        np.mean(rm_1_ratios)
                    )
                    to_log["eval/info/eval_rm_state_2_ratio"] = float(
                        np.mean(rm_2_ratios)
                    )

                    json_metrics["eval_success_rate"] = torch.tensor(success_rates)
                    json_metrics["eval_bottleneck_reached_rate"] = torch.tensor(
                        bottleneck_rates
                    )
                    json_metrics["eval_rm_state_0_ratio"] = torch.tensor(rm_0_ratios)
                    json_metrics["eval_rm_state_1_ratio"] = torch.tensor(rm_1_ratios)
                    json_metrics["eval_rm_state_2_ratio"] = torch.tensor(rm_2_ratios)
        except Exception as e:
            return e
        #############################
        # [11] End.
        #############################

        if self.json_writer is not None:
            self.json_writer.write(
                metrics=json_metrics,
                total_frames=total_frames,
                evaluation_step=total_frames
                // self.experiment_config.evaluation_interval,
            )
            json_file = str(self.json_writer.path)
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.save(
                        json_file, base_path=os.path.dirname(json_file)
                    )

        try:
            import datetime
            import csv
            from pathlib import Path

            def _mean_to_float(value):
                if isinstance(value, torch.Tensor):
                    return float(value.detach().float().mean().cpu().item())
                return float(np.asarray(value, dtype=np.float32).mean())

            returns = json_metrics.get("return", None)
            if returns is not None:
                if isinstance(returns, torch.Tensor):
                    returns_np = returns.detach().float().cpu().numpy()
                else:
                    returns_np = np.asarray(returns, dtype=np.float32)
                eval_return_mean = float(returns_np.mean())
                eval_return_std = float(returns_np.std())
                eval_return_min = float(returns_np.min())
                eval_return_max = float(returns_np.max())
            else:
                eval_return_mean = eval_return_std = eval_return_min = (
                    eval_return_max
                ) = None

            csv_dir = Path("logs_thesis")
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"{self.experiment_name}.csv"

            row = {
                "frame": total_frames,
                "step": step,
                "timestamp": datetime.datetime.now().isoformat(),
                "eval_return_mean": eval_return_mean,
                "eval_return_std": eval_return_std,
                "eval_return_min": eval_return_min,
                "eval_return_max": eval_return_max,
                "algorithm": getattr(self, "algorithm_name", "unknown"),
                "environment": getattr(self, "environment_name", "unknown"),
                "task": getattr(self, "task_name", "unknown"),
                "seed": getattr(self, "seed", "unknown"),
            }

            if "win_rate" in json_metrics:
                row["eval_win_rate"] = _mean_to_float(json_metrics["win_rate"])
            if "eval_success_rate" in json_metrics:
                row["eval_success_rate"] = _mean_to_float(
                    json_metrics["eval_success_rate"]
                )
            if "eval_bottleneck_reached_rate" in json_metrics:
                row["eval_bottleneck_reached_rate"] = _mean_to_float(
                    json_metrics["eval_bottleneck_reached_rate"]
                )
            if "eval_rm_state_0_ratio" in json_metrics:
                row["eval_rm_state_0_ratio"] = _mean_to_float(
                    json_metrics["eval_rm_state_0_ratio"]
                )
            if "eval_rm_state_1_ratio" in json_metrics:
                row["eval_rm_state_1_ratio"] = _mean_to_float(
                    json_metrics["eval_rm_state_1_ratio"]
                )
            if "eval_rm_state_2_ratio" in json_metrics:
                row["eval_rm_state_2_ratio"] = _mean_to_float(
                    json_metrics["eval_rm_state_2_ratio"]
                )

            file_exists = csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists or f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[CSV LOGGER ERROR] Failed to write CSV: {e}")
        # -------------------------------------------------------------

        self.log(to_log, step=step)
        if video_frames is not None and max_length_rollout_0 > 1:
            video_frames = np.stack(video_frames[: max_length_rollout_0 - 1], axis=0)
            if len(video_frames.shape) == 4:
                vid = torch.tensor(
                    np.transpose(video_frames, (0, 3, 1, 2)),
                    dtype=torch.uint8,
                ).unsqueeze(0)
                for logger in self.loggers:
                    if isinstance(logger, WandbLogger):
                        logger.log_video("eval/video", vid, fps=20, commit=False)
                    else:
                        # Other loggers cannot deal with odd video sizes so we check if the video dimensions are odd and make them even
                        for index in (-1, -2):
                            if vid.shape[index] % 2 != 0:
                                vid = vid.index_select(
                                    index, torch.arange(1, vid.shape[index])
                                )
                        # End of check

                        logger.log_video("eval_video", vid, step=step)

    #############################
    # [2] Start. Add full method for logging DEMIR metrics: EDM/EFM size, beta, scale.
    #############################
    def log_demir_stats(
        self,
        demir_module,
        group: str,
        step: int,
        shaping_reward_mean: Optional[float] = None,
        novelty_mean: Optional[float] = None,
        quality_mean: Optional[float] = None,
    ) -> None:
        """Log DEMIR episodic memory statistics to WandB / TensorBoard."""
        if not len(self.loggers):
            return

        cfg = demir_module.config

        def _cfg(name, default):
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(name, default)
            return getattr(cfg, name, default)

        to_log: Dict[str, Any] = {
            #############################
            # [3] Start. Add edm_size and efm_size as measures of exploration
            #############################
            f"demir/{group}/edm_size": demir_module.edm_index.ntotal,
            f"demir/{group}/efm_size": demir_module.efm_index.ntotal,
            #############################
            # [3] End.
            #############################
            #############################
            # [4] Start. Add beta1, beta2, demir_scale logged continuously — for filtering in W&B.
            #############################
            f"demir/{group}/beta1": _cfg("beta1", 1.0),
            f"demir/{group}/beta2": _cfg("beta2", 0.5),
            f"demir/{group}/demir_scale": _cfg("demir_scale", 0.05),
            #############################
            # [4] End.
            #############################
        }

        #############################
        # [5] Start. Add shaping_reward_mean, novelty_mean, quality_mean (if provided).
        #############################
        if shaping_reward_mean is not None:
            to_log[f"demir/{group}/shaping_reward_mean"] = shaping_reward_mean
        if novelty_mean is not None:
            to_log[f"demir/{group}/novelty_mean"] = novelty_mean
        if quality_mean is not None:
            to_log[f"demir/{group}/quality_mean"] = quality_mean
        #############################
        # [5] End.
        #############################

        self.log(to_log, step=step)

    #############################
    # [2] End.
    #############################

    #############################
    # [6] Start. Add list of 5 W&B keys ready for plot export for publication.
    #############################
    ELSEVIER_METRIC_KEYS: List[str] = [
        "eval/info/win_rate",  # (1) Win Rate – główna metryka SMACv2
        "eval/reward/episode_reward_mean",  # (2) Episode Return
        "demir/{group}/edm_size",  # (3) EDM Coverage
        "demir/{group}/novelty_mean",  # (4) Novelty (β₂)
        "demir/{group}/quality_mean",  # (5) Quality (β₁)
    ]
    #############################
    # [6] End.
    #############################

    def commit(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({}, commit=True)

    #############################
    # [7] Start. Use `experiment_name` instead of a date for file naming. Catching environmental logs.
    #############################
    def log(self, dict_to_log: Dict, step: int = None):
        thesis_dir = Path("logs_thesis")
        thesis_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{thesis_dir}/{self.experiment_name}.jsonl"

        with open(fname, "a") as f:
            clean_dict = {}
            for k, v in dict_to_log.items():
                if isinstance(v, torch.Tensor):
                    clean_dict[k] = v.item() if v.numel() == 1 else v.tolist()
                elif hasattr(v, "item"):
                    clean_dict[k] = v.item()
                else:
                    clean_dict[k] = v
            clean_dict["_step"] = step
            f.write(json.dumps(clean_dict) + "\n")

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log(dict_to_log, commit=False)
            else:
                for key, value in dict_to_log.items():
                    logger.log_scalar(key.replace("/", "_"), value, step=step)

    #############################
    # [7] End.
    #############################

    def finish(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                wandb.finish()

    def _get_reward(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        reward = td.get(("next", group, "reward"), None)
        if reward is None:
            reward = (
                td.get(("next", "reward")).expand(td.get(group).shape).unsqueeze(-1)
            )
        return reward.mean(-2) if remove_agent_dim else reward

    def _get_agents_done(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        done = td.get(("next", group, "done"), None)
        if done is None:
            done = td.get(("next", "done")).expand(td.get(group).shape).unsqueeze(-1)

        return done.any(-2) if remove_agent_dim else done

    def _get_global_done(
        self,
        td: TensorDictBase,
    ):
        done = td.get(("next", "done"))
        return done

    def _get_episode_reward(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        episode_reward = td.get(("next", group, "episode_reward"), None)
        if episode_reward is None:
            episode_reward = (
                td.get(("next", "episode_reward"))
                .expand(td.get(group).shape)
                .unsqueeze(-1)
            )
        return episode_reward.mean(-2) if remove_agent_dim else episode_reward

    def _log_individual_and_group_rewards(
        self,
        group: str,
        batch: TensorDictBase,
        global_done: Tensor,
        any_episode_ended: bool,
        to_log: Dict[str, Tensor],
        prefix: str = "collection",
        log_individual_agents: bool = True,
    ):
        reward = self._get_reward(group, batch)  # Has agent dim
        episode_reward = self._get_episode_reward(group, batch)  # Has agent dim
        n_agents_in_group = episode_reward.shape[-2]

        # Add multiagent dim
        unsqueeze_global_done = global_done.unsqueeze(-1).expand(
            (*batch.get_item_shape(group), 1)
        )
        #######
        # All trajectories are considered done at the global done
        #######

        # 1. Here we log rewards from individual agent data
        if log_individual_agents:
            for i in range(n_agents_in_group):
                self._log_min_mean_max(
                    to_log,
                    f"{prefix}/{group}/reward/agent_{i}/reward",
                    reward[..., i, :],
                )
                if any_episode_ended:
                    agent_global_done = unsqueeze_global_done[..., i, :]
                    self._log_min_mean_max(
                        to_log,
                        f"{prefix}/{group}/reward/agent_{i}/episode_reward",
                        episode_reward[..., i, :][agent_global_done],
                    )

        # 2. Here we log rewards from group data taking the mean over agents
        group_episode_reward = episode_reward.mean(-2)[global_done]
        if any_episode_ended:
            self._log_min_mean_max(
                to_log, f"{prefix}/{group}/reward/episode_reward", group_episode_reward
            )
        self._log_min_mean_max(to_log, f"{prefix}/reward/reward", reward)

        return group_episode_reward

    def _log_global_episode_reward(
        self, episode_rewards: List[Tensor], to_log: Dict[str, Tensor], prefix: str
    ):
        # Each element in the list is the episode reward (with shape n_episodes) for the group at the global done,
        # so they will have same shape as done is shared
        episode_rewards = torch.stack(episode_rewards, dim=0).mean(
            0
        )  # Mean over groups
        if episode_rewards.numel() > 0:
            self._log_min_mean_max(
                to_log, f"{prefix}/reward/episode_reward", episode_rewards
            )

        return episode_rewards

    def _log_min_mean_max(self, to_log: Dict[str, Tensor], key: str, value: Tensor):
        to_log.update(
            {
                key + "_min": value.min().item(),
                key + "_mean": value.mean().item(),
                key + "_max": value.max().item(),
            }
        )


class JsonWriter:
    """
    Writer to create json files for reporting according to marl-eval

    Follows conventions from https://github.com/instadeepai/marl-eval/tree/main#usage-

    Args:
        folder (str): folder where to write the file
        name (str): file name
        algorithm_name (str): algorithm name
        task_name (str): task name
        environment_name (str): environment name
        seed (int): seed of the experiment

    """

    def __init__(
        self,
        folder: str,
        name: str,
        algorithm_name: str,
        task_name: str,
        environment_name: str,
        seed: int,
    ):
        self.path = Path(folder) / Path(name)
        self.experiment_name = name.replace(".json", "")
        self.algorithm_name = algorithm_name
        self.task_name = task_name
        self.environment_name = environment_name
        self.seed = seed
        self.run_data = {"absolute_metrics": {}}
        self.data = {
            environment_name: {
                task_name: {algorithm_name: {f"seed_{seed}": self.run_data}}
            }
        }

    def write(
        self, total_frames: int, metrics: Dict[str, List[Tensor]], evaluation_step: int
    ):
        """
        Writes a step into the json reporting file

        Args:
            total_frames (int): total frames collected so far in the experiment
            metrics (dictionary mapping str to tensor): each value is a 1-dim tensor for the metric in key
                of len equal to the number of evaluation episodes for this step.
            evaluation_step (int): the evaluation step

        """
        metrics = {k: val.tolist() for k, val in metrics.items()}
        step_metrics = {"step_count": total_frames}
        step_metrics.update(metrics)
        step_str = f"step_{evaluation_step}"
        if step_str in self.run_data:
            self.run_data[step_str].update(step_metrics)
        else:
            self.run_data[step_str] = step_metrics

        # Store the maximum of each metric
        for metric_name in metrics.keys():
            if len(metrics[metric_name]):
                max_metric = max(metrics[metric_name])
                if metric_name in self.run_data["absolute_metrics"]:
                    prev_max_metric = self.run_data["absolute_metrics"][metric_name][0]
                    max_metric = max(max_metric, prev_max_metric)
                self.run_data["absolute_metrics"][metric_name] = [max_metric]

        import json
        import os
        from datetime import datetime

        with open(self.path, "w+") as f:
            json.dump(self.data, f, indent=4)

        #############################
        # [10] Start. Append a flattened JSONL line to a central `logs_thesis`
        # directory for easy aggregation of thesis metrics.
        #############################
        base_dir = "logs_thesis"
        os.makedirs(base_dir, exist_ok=True)
        flat_metrics = {
            k: (sum(vals) / len(vals) if len(vals) > 0 else 0.0)
            for k, vals in metrics.items()
        }
        flat_metrics["step_count"] = total_frames
        flat_metrics["evaluation_step"] = evaluation_step
        flat_metrics["timestamp_str"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Add experiment identifiers
        flat_metrics["experiment_name"] = self.experiment_name
        flat_metrics["algorithm"] = self.algorithm_name
        flat_metrics["environment"] = self.environment_name
        flat_metrics["task"] = self.task_name
        flat_metrics["seed"] = self.seed

        file_name = "thesis_metrics_raw.jsonl"
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "a") as fw:
            fw.write(json.dumps(flat_metrics) + "\n")
        #############################
        # [10] End.
        #############################
