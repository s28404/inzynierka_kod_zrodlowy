#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor zmian: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Skrypt treningowy dla Logic Environment. Skopiowany z smacv2_run.py
#        do uruchamiania QMIX i innych algorytmów na SynchronizedFactory.
# Originał: BenchMARL (Meta Platforms), https://github.com/facebookresearch/BenchMARL
# ---

import hydra
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import load_experiment_from_hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import csv
from pathlib import Path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment: Experiment = load_experiment_from_hydra(cfg, task_name=task_name)

    # Przeniesiono zapis CSV bezpośrednio do benchmarl/experiment/logger.py

    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
