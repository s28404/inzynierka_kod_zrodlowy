#  Copyright (c) 2026 Kajetan Frąckowiak, s28404
#
#  Projekt: Algorytm DEMIR dla SMACv2 i Custom Logic Environment
#  Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
#  Praca Inżynierska (2026)
#
#  Opis: Niestandardowy logger JSON dla BenchMARL zbierający logi do pracy inżynierskiej.
#  Zbiera dane treningowe i ewaluacyjne per `eval_interval` oraz bezpiecznie nadpisuje plik JSON.
#  Odporny na przerwania programu, zawiera struktury kluczowych metadanych do eksperymentu
#  oraz wspiera ablacje DEMIR, NGU, QMIX, RND.
#  Dodane przez: Kajetan Frąckowiak
#
# Wprowadzone zmiany:
# [1] Zmiana logiki ekstrakcji `algo_name` z wiersza poleceń / Hydry dla właściwego rozpoznawania wariantów QMIX
# [2] Zmiana sposobu nazywania plików wynikowych z formatu daty na nazwę wygenerowaną przez W&B (experiment.name)

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from benchmarl.experiment.callback import Callback


class ThesisJSONLoggerCallback(Callback):
    """
    Niestandardowy logger JSON dla BenchMARL zbierający logi do pracy inżynierskiej.
    Zbiera dane treningowe i ewaluacyjne per `eval_interval` oraz bezpiecznie nadpisuje plik JSON.
    """

    def __init__(self, log_dir: str = "logs"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.file_path: Optional[Path] = None

        self.data_registry = {"metadata": {}, "metrics": []}

    def on_train_step(self, batch, group: str):
        """Standardowy callback z BenchMARL - nie używany w tej implementacji."""
        pass

    def on_setup(self):
        """Wywoławane na starcie - gwarancja że callback jest załadowany."""
        print("[THESIS_LOGGER] ✅ Callback załadowany!")

    def on_train_start(self, experiment) -> None:
        """
        Uruchamiane raz na starcie eksperymentu. Zbieramy tu metadane.
        """
        print("[THESIS_LOGGER] on_train_start wywoławana!")
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ekstrakcja kluczowych danych z konfiguracji eksperymentu BenchMARL
        algo_name = experiment.algorithm.name.lower()

        # Jeśli algorytmy DEMIR/NGU dziedziczą po QMIX, algo_name będzie "qmix".
        # Spróbujmy wyciągnąć prawdziwą nazwę podaną w wierszu poleceń.
        import sys

        for arg in sys.argv:
            if arg.startswith("algorithm="):
                algo_name = arg.split("=")[1]
                break

        # Dodatkowo, jeśli z jakiegoś powodu z sys.argv nie poszło, próbujemy z hydry:
        if algo_name == "qmix" or algo_name == "":
            try:
                from hydra.core.hydra_config import HydraConfig

                choices = HydraConfig.get().runtime.choices
                if "algorithm" in choices:
                    algo_name = choices["algorithm"]
            except Exception:
                pass

        task_name = experiment.task.name
        seed = experiment.seed

        # Bezpieczne wyciąganie hiperparametrów - BenchMARL zapisuje je w configs
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
                "intrinsic_module_type": getattr(
                    algo_cfg, "intrinsic_module_type", None
                ),
                "off_policy_memory_size": getattr(cfg, "buffer_size", None),
                "lr": getattr(algo_cfg, "lr", None),
            },
        }

        # Zapisz plik pod identyczną nazwą jak run w W&B
        filename = f"{experiment.name}.json"
        self.file_path = self.log_dir / filename

        print(f"[THESIS_LOGGER] Inicjalizacja: zapisywanie do {self.file_path}")
        self.save_to_disk()

    def on_evaluation_end(self, rollouts) -> None:
        """
        Uruchamiane raz po zakończeniu danej pętli ewaluacyjnej.
        Zbiera dane treningowe i ewaluacyjne do JSON.

        Args:
            rollouts: List[TensorDictBase] zawierające dane z ewaluacji
        """
        print(
            f"[THESIS_LOGGER] on_evaluation_end wywoławana! Liczba rollout'ów: {len(rollouts)}"
        )

        experiment = self.experiment
        step = experiment.n_iters_performed

        # Oblicz eval_stats z rollout'ów
        eval_stats = self._compute_eval_stats(rollouts)

        # Zbierz ostatnie zalogowane metryki treningowe z experiment.logger
        # (jeśli dostępne w jego systemie logowania)
        train_metrics = self._extract_recent_train_metrics()

        # Oczyszczenie statystyk ewaluacyjnych
        eval_clean = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in eval_stats.items()
        }

        # 2. Tworzenie ujednoliconego słownika dla jednego wiersza metryk
        current_metrics = {
            "frame": experiment.total_frames,
            "step": step,
            # Trening (Ostatnie zalogowane metryki)
            "train_return_mean": train_metrics.get("train_return_mean", None),
            "train_intrinsic_reward_mean": train_metrics.get(
                "train_intrinsic_reward_mean", None
            ),
            "q_loss": train_metrics.get("q_loss", None),
            "intrinsic_module_loss": train_metrics.get("intrinsic_module_loss", None),
            # Ewaluacja (Bazowe)
            "eval_return_mean": eval_clean.get("eval/episode_reward", None),
            "eval_return_std": eval_clean.get("eval/episode_reward_std", None),
            # DEMIR - specyficzne metryki
            "loss_idm": train_metrics.get("loss_idm", None),
            "loss_barlow_twins": train_metrics.get("loss_barlow_twins", None),
            "intrinsic_reward_Q_mean": train_metrics.get(
                "intrinsic_reward_Q_mean", None
            ),
            "intrinsic_reward_N_mean": train_metrics.get(
                "intrinsic_reward_N_mean", None
            ),
            "current_beta1": train_metrics.get("current_beta1", None),
            "current_beta2": train_metrics.get("current_beta2", None),
        }

        # 3. Specyficzne dla środowisk
        env_name = self.data_registry["metadata"]["environment"]

        if "smac" in env_name.lower():
            # SMAC metryki
            current_metrics["eval_win_rate"] = eval_clean.get("eval/win_rate", None)

        elif (
            "factory" in env_name.lower()
            or "logic" in env_name.lower()
            or "synchronized" in env_name.lower()
        ):
            # Custom Reward Machine metryki dla SynchronizedFactory w sekcji logic_env
            current_metrics["eval_success_rate"] = eval_clean.get(
                "eval/success_rate", None
            )
            current_metrics["eval_bottleneck_reached_rate"] = eval_clean.get(
                "eval/bottleneck_reached_rate", None
            )
            current_metrics["eval_rm_state_0_ratio"] = eval_clean.get(
                "eval/rm_state_0_ratio", None
            )
            current_metrics["eval_rm_state_1_ratio"] = eval_clean.get(
                "eval/rm_state_1_ratio", None
            )
            current_metrics["eval_rm_state_2_ratio"] = eval_clean.get(
                "eval/rm_state_2_ratio", None
            )
            current_metrics["eval_avg_steps_to_RM1"] = eval_clean.get(
                "eval/avg_steps_to_RM1", None
            )

        self.data_registry["metrics"].append(current_metrics)
        self.save_to_disk()

    def _extract_recent_train_metrics(self) -> Dict[str, Any]:
        """
        Wyciąga ostatnie zalogowane metryki treningowe.
        Szuka ich w dostępnych źródłach z experiment.logger.
        """
        metrics = {}

        try:
            # Jeśli logger ma jakiś sposób dostępu do ostatnich logów
            # to tutaj byśmy je wyciągnęli. Na razie zwrócimy puste dict.
            # W przyszłości można to rozszerzyć.
            pass
        except Exception:
            pass

        return metrics

    def _compute_eval_stats(self, rollouts) -> Dict[str, Any]:
        """
        Oblicza metryki ewaluacji z rollout'ów.

        Args:
            rollouts: List[TensorDictBase] zawierające dane z ewaluacji

        Returns:
            Dict zawierający eval_stats, głównie episode_reward
        """
        import torch
        import numpy as np

        eval_stats = {}

        try:
            # Obliczenie average episode reward
            rewards = []
            for td in rollouts:
                # Spróbuj wyciągnąć reward dla wszystkich agentów
                if ("next", "agents", "reward") in td.keys(True, True):
                    reward = td.get(("next", "agents", "reward"))
                elif ("next", "reward") in td.keys(True, True):
                    reward = td.get(("next", "reward"))
                else:
                    # Szukaj najczęściej w BenchMARL (group-based)
                    # Przeszukaj klucze aby znaleźć reward
                    found = False
                    for key in td.keys(True, True):
                        if isinstance(key, tuple) and "reward" in key[-1]:
                            reward = td.get(key)
                            rewards.append(
                                reward.sum(0).mean().item()
                                if reward.numel() > 0
                                else 0.0
                            )
                            found = True
                            break
                    if found:
                        continue

                if reward is not None:
                    rewards.append(
                        reward.sum(0).mean().item() if reward.numel() > 0 else 0.0
                    )

            if rewards:
                eval_stats["eval/episode_reward"] = float(np.mean(rewards))
                eval_stats["eval/episode_reward_std"] = float(np.std(rewards))
        except Exception as e:
            print(f"[THESIS_LOGGER] Błąd obliczania reward: {e}")

        # Próba wyciągnięcia battle_won dla SMACv2
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

        # Próba wyciągnięcia success_rate dla logic_env
        try:
            if (
                "logic_env"
                in self.data_registry["metadata"].get("environment", "").lower()
            ):
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
        """Bezpieczny zrzut do pliku (z plikiem tymczasowym zabezpieczającym przed przerwaniem SIGINT)."""
        if self.file_path is None:
            return

        temp_path = self.file_path.with_suffix(".json.tmp")

        try:
            with open(temp_path, "w") as f:
                json.dump(self.data_registry, f, indent=4)
            # Atomowe zastąpienie pliku
            temp_path.replace(self.file_path)
        except Exception as e:
            print(f"Błąd zapisu JSON: {e}")
