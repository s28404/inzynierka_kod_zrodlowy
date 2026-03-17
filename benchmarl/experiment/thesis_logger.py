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

        # Bufor na metryki treningowe między ewaluacjami (do wyliczenia średniej z przedziału)
        self.train_buffer = []

    def on_train_start(self, experiment) -> None:
        """
        Uruchamiane raz na starcie eksperymentu. Zbieramy tu metadane.
        """
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

        self.save_to_disk()

    def on_train_step_end(
        self, step: int, log_dict: Dict[str, Any], experiment
    ) -> None:
        """
        Zapisuje bieżącą paczkę strat (losses) i metryk eksploracyjnych do bufora.
        """
        # Konwersja tensorów na wartości skalarne
        clean_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in log_dict.items()
        }
        self.train_buffer.append(clean_dict)

    def on_evaluation_end(
        self, step: int, eval_stats: Dict[str, Any], experiment
    ) -> None:
        """
        Uruchamiane raz po zakończeniu danej pętli ewaluacyjnej.
        Łączy straty treningowe z wynikami ewaluacji.
        """
        # 1. Agregacja danych treningowych od ostatniej ewaluacji
        train_aggr = {}
        if self.train_buffer:
            keys = set(k for d in self.train_buffer for k in d.keys())
            for k in keys:
                values = [
                    d[k] for d in self.train_buffer if k in d and d[k] is not None
                ]
                if values:
                    train_aggr[k] = sum(values) / len(values)
            self.train_buffer = []  # Czyszczenie bufora na kolejną epokę

        # Oczyszczenie statystyk ewaluacyjnych
        eval_clean = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in eval_stats.items()
        }

        # 2. Tworzenie ujednoliconego słownika dla jednego wiersza metryk
        current_metrics = {
            "frame": experiment.total_frames,
            "step": step,
            # Trening (Złapane metryki z log_dict TorchRL / BenchMARL)
            "train_return_mean": train_aggr.get("collection/episode_reward", None),
            "train_intrinsic_reward_mean": train_aggr.get(
                "collection/intrinsic_reward", None
            ),
            "q_loss": train_aggr.get("loss/q_loss", None),
            "intrinsic_module_loss": train_aggr.get("loss/intrinsic_loss", None),
            # Ewaluacja (Bazowe)
            "eval_return_mean": eval_clean.get("eval/episode_reward", None),
            "eval_return_std": eval_clean.get("eval/episode_reward_std", None),
            # DEMIR - specyficzne metryki
            "loss_idm": train_aggr.get("loss/loss_idm", None),
            "loss_barlow_twins": train_aggr.get("loss/loss_barlow_twins", None),
            "intrinsic_reward_Q_mean": train_aggr.get(
                "collection/intrinsic_reward_Q", None
            ),
            "intrinsic_reward_N_mean": train_aggr.get(
                "collection/intrinsic_reward_N", None
            ),
            "current_beta1": train_aggr.get("collection/current_beta1", None),
            "current_beta2": train_aggr.get("collection/current_beta2", None),
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
