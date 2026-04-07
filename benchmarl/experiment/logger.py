#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor zmian: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Dodano logowanie win_rate (SMACv2 battle_won) do JSON i WandB;
#        dodano metodę log_demir_stats do śledzenia metryk DEMIR.
# Oryginał: BenchMARL (Meta Platforms), https://github.com/facebookresearch/BenchMARL
#
# Ramki komentarzowe dodane w tej sesji (inline block markers):
#
#   [1] # ── Win Rate (SMACv2 battle_won flag) ──────────────────────────
#       Lokalizacja: log_evaluation()
#       Oblicza win_rate z flagi battle_won i zapisuje do W&B oraz JSON.
#
#   [2] # ── DEMIR publication metrics ───────────────────────────────────
#       Lokalizacja: log_demir_stats() — nagłówek metody
#       Pełna metoda logowania metryk DEMIR: EDM/EFM size, beta, scale.
#
#   [3] # ── Coverage / Diversity ─────────────────────────────────────────
#       Lokalizacja: log_demir_stats() — sekcja to_log
#       edm_size i efm_size jako miary eksploracji (rosnące z liczbą klatek).
#
#   [4] # ── Ablation markers ──────────────────────────────────────────────
#       Lokalizacja: log_demir_stats() — sekcja to_log
#       beta1, beta2, demir_scale logowane stale — filtrowanie w W&B.
#
#   [5] # ── Intrinsic reward decomposition (opcjonalne) ──────────────────
#       Lokalizacja: log_demir_stats() — opcjonalne pola to_log
#       shaping_reward_mean, novelty_mean, quality_mean (jeśli podane).
#
#   [6] # ── Convenience: 5 Metric Keys do eksportu PDF (Elsevier) ────────
#       Lokalizacja: ELSEVIER_METRIC_KEYS (atrybut klasy Logger)
#       Lista 5 kluczy W&B gotowych do eksportu wykresu do publikacji.
#
#   [7] # ── Zmiana nazewnictwa plików .jsonl pod W&B ──────────────────────
#       Lokalizacja: log() oraz __init__()
#       Użycie `experiment_name` zamiast daty do nazywania plików. Wyłapywanie logów środowiskowych.
#
# ---

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
        self.experiment_name = experiment_name

        # --- KAJETAN MOD: Unikalna data dla osobnych plików dla każdego uruchomienia ---
        from datetime import datetime

        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
        # DEBUG: Write to file na samym poczatku
        with open("/tmp/log_collection_called.txt", "a") as f:
            f.write(
                f"[log_collection] CALLED with total_frames={total_frames}, step={step}\n"
            )
            f.flush()

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

        # --- CSV LOGGING HOOK ---
        # Write metrics to CSV for thesis analysis
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
        # --- END CSV LOGGING ---

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

        # ── Win Rate (SMACv2 battle_won flag) ───────────────────────────────
        # Kluczowa metryka dla publikacji Elsevier: oś X = total_frames,
        # oś Y = % wygranych bitew. Wymagana do Sample-Efficiency comparison.
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
                # Przechowujemy też w JSON (do marl-eval / Seaborn)
                json_metrics["win_rate"] = torch.tensor(
                    [float(w) for w in battle_won_per_ep]
                )
        except Exception:
            pass  # Non-SMACv2 task – brak battle_won, pomijamy
        # ────────────────────────────────────────────────────────────────────

        # ── Autorskie metryki RM (logic_env_factory) ─────────────────────────
        try:
            if "logic_env" in self.task_name or "synchronized" in self.task_name:
                success_rates = []
                bottleneck_rates = []
                rm_0_ratios = []
                rm_1_ratios = []
                rm_2_ratios = []

                for td in rollouts:
                    # Wyciągamy rm_state z obserwacji dowolnego agenta (index 6).
                    # Kształt ob: [time, n_agents, obs_dim]
                    obs = td.get(("next", "agents", "observation"))
                    # Bierzemy z pierwszego agenta, bo RM state jest globalne/współdzielone
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
            pass  # Puste gdy nie pasuje do środowiska
        # ────────────────────────────────────────────────────────────────────

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

        # --- ZAPIS DO CSV (MODYFIKACJA DLA PRACY INŻYNIERSKIEJ) ---
        try:
            import datetime
            import csv
            from pathlib import Path

            returns = json_metrics.get("return", None)
            if returns is not None:
                returns = returns.numpy()
                eval_return_mean = float(returns.mean())
                eval_return_std = float(returns.std())
                eval_return_min = float(returns.min())
                eval_return_max = float(returns.max())
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
                row["eval_win_rate"] = float(json_metrics["win_rate"].float().mean())
            if "eval_success_rate" in json_metrics:
                row["eval_success_rate"] = float(
                    json_metrics["eval_success_rate"].float().mean()
                )
            if "eval_bottleneck_reached_rate" in json_metrics:
                row["eval_bottleneck_reached_rate"] = float(
                    json_metrics["eval_bottleneck_reached_rate"].float().mean()
                )
            if "eval_rm_state_0_ratio" in json_metrics:
                row["eval_rm_state_0_ratio"] = float(
                    json_metrics["eval_rm_state_0_ratio"].float().mean()
                )
            if "eval_rm_state_1_ratio" in json_metrics:
                row["eval_rm_state_1_ratio"] = float(
                    json_metrics["eval_rm_state_1_ratio"].float().mean()
                )
            if "eval_rm_state_2_ratio" in json_metrics:
                row["eval_rm_state_2_ratio"] = float(
                    json_metrics["eval_rm_state_2_ratio"].float().mean()
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

    # ── DEMIR publication metrics ─────────────────────────────────────────
    def log_demir_stats(
        self,
        demir_module,
        group: str,
        step: int,
        shaping_reward_mean: Optional[float] = None,
        novelty_mean: Optional[float] = None,
        quality_mean: Optional[float] = None,
    ) -> None:
        """Log DEMIR episodic memory statistics to WandB / TensorBoard.

        Designed for three plot types required by Elsevier reviewers:

        1. **Diversity / Coverage** – ``demir/{group}/edm_size`` grows as the
           agent explores new states. Compare DEMIR vs QMIX baseline on a
           heat-map or coverage-vs-frames plot.

        2. **Ablation Study** – ``demir/{group}/beta1`` and ``beta2`` are
           logged every step, so each WandB run can be filtered by beta
           combination::

               Full DEMIR  : beta1 > 0, beta2 > 0
               No Quality  : beta1 = 0, beta2 > 0
               No Novelty  : beta1 > 0, beta2 = 0
               Baseline    : beta1 = 0, beta2 = 0  (= QMIX)

        3. **Intrinsic reward decomposition** – optional
           ``shaping_reward_mean``, ``novelty_mean``, ``quality_mean`` allow
           plotting the two reward components separately.

        Call this method from the algorithm's ``_loss_fn`` or via a
        :class:`~benchmarl.experiment.callback.Callback`.

        Args:
            demir_module: the :class:`DecentralizedEpisodicReward` instance.
            group (str): agent group name.
            step (int): current training step.
            shaping_reward_mean (float, optional): pre-computed mean phi.
            novelty_mean (float, optional): mean novelty (EDM component).
            quality_mean (float, optional): mean quality (EFM component).
        """
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
            # ── Coverage / Diversity ──────────────────────────────────────
            f"demir/{group}/edm_size": demir_module.edm_index.ntotal,
            f"demir/{group}/efm_size": demir_module.efm_index.ntotal,
            # ── Ablation markers ─────────────────────────────────────────
            # Stałe per-run — umożliwiają filtrowanie w WandB po beta.
            f"demir/{group}/beta1": _cfg("beta1", 1.0),
            f"demir/{group}/beta2": _cfg("beta2", 0.5),
            f"demir/{group}/demir_scale": _cfg("demir_scale", 0.05),
        }

        # ── Intrinsic reward decomposition (opcjonalne) ───────────────────
        if shaping_reward_mean is not None:
            to_log[f"demir/{group}/shaping_reward_mean"] = shaping_reward_mean
        if novelty_mean is not None:
            to_log[f"demir/{group}/novelty_mean"] = novelty_mean
        if quality_mean is not None:
            to_log[f"demir/{group}/quality_mean"] = quality_mean

        self.log(to_log, step=step)

    # ── Convenience: 5 Metric Keys do eksportu PDF (Elsevier) ─────────────
    ELSEVIER_METRIC_KEYS: List[str] = [
        "eval/info/win_rate",  # (1) Win Rate  – główna metryka SMACv2
        "eval/reward/episode_reward_mean",  # (2) Episode Return  – backup jeśli brak win_rate
        "demir/{group}/edm_size",  # (3) EDM Coverage  – dowód eksploracji
        "demir/{group}/novelty_mean",  # (4) Novelty (β₂)  – składnik ablacji
        "demir/{group}/quality_mean",  # (5) Quality (β₁)  – składnik ablacji
        # Uwaga: zastąp {group} rzeczywistą nazwą grupy (np. 'agents').
        # Do generowania PDF użyj: wandb.Api().run(...).history(keys=KEYS)
        # i biblioteki Seaborn lub marl-eval. Zapisz jako .PDF/.EPS.
    ]
    # ─────────────────────────────────────────────────────────────────────

    def commit(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({}, commit=True)

    def log(self, dict_to_log: Dict, step: int = None):
        # --- DODANE PRZEZ KAJETANA: Niestandardowy zrzut ewaluacji z każdego log() ---
        # Zapisz my log-file do katalogu logs_thesis (bezwzględna ścieżka by ominąć Hydrę)
        thesis_dir = Path("/home/kajetan/Documents/inzynierka_kod_zrodlowy/logs_thesis")
        thesis_dir.mkdir(parents=True, exist_ok=True)
        # Nazywamy w oparciu o obiekt głównej konfiguracji + timestamp dla UNIKALNOŚCI KAŻDEGO URUCHOMIENIA
        safe_task_name = self.task_name.replace("/", "_")

        # Zapis pod dokładnie taką nazwą jak w wandb
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
        # ---------------------------------------------------------------------

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log(dict_to_log, commit=False)
            else:
                for key, value in dict_to_log.items():
                    logger.log_scalar(key.replace("/", "_"), value, step=step)

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

        base_dir = "/home/kajetan/Documents/inzynierka_kod_zrodlowy/logs_thesis"
        os.makedirs(base_dir, exist_ok=True)
        # Bierzemy średnie wartości z list dla danego kroku ewaluacji, chyba że lista jest pusta
        flat_metrics = {
            k: (sum(vals) / len(vals) if len(vals) > 0 else 0.0)
            for k, vals in metrics.items()
        }
        flat_metrics["step_count"] = total_frames
        flat_metrics["evaluation_step"] = evaluation_step
        flat_metrics["timestamp_str"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        file_name = f"thesis_metrics_raw.jsonl"
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "a") as fw:
            fw.write(json.dumps(flat_metrics) + "\n")
