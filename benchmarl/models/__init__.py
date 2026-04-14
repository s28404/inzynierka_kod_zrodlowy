#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor zmian: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Ograniczono rejestr modeli do MLP i GRU (usunięto CNN, GNN,
#        LSTM, Deepsets — nieużywane w eksperymentach DEMIR/QMIX).
# Oryginał: BenchMARL (Meta Platforms), https://github.com/facebookresearch/BenchMARL
# ---

from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)
from .mlp import Mlp, MlpConfig
from .gru import Gru, GruConfig

classes = [
    "Mlp",
    "MlpConfig",
    "Gru",
    "GruConfig",
]

model_config_registry = {
    "mlp": MlpConfig,
    "gru": GruConfig,
}
