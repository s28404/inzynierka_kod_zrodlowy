#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor zmian: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Usunięto MAPPO i MASAC; dodano warianty demir/rnd/ngu do rejestru.
# Oryginał: BenchMARL (Meta Platforms), https://github.com/facebookresearch/BenchMARL
# ---

from .common import Algorithm, AlgorithmConfig
from .qmix import Qmix, QmixConfig

classes = [
    "Qmix",
    "QmixConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "qmix": QmixConfig,
    "demir": QmixConfig,  # DEMIR = QMIX z demir_scale > 0
    "rnd":   QmixConfig,  # RND   = QMIX z rnd_scale > 0
    "ngu":   QmixConfig,  # NGU   = QMIX z ngu_scale > 0
}
