#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# ==============================================================================
# Author: Kajetan Frąckowiak
# Date: 2026
# Modifications in this file:
# [1] # Added own algorithm config registries
# ==============================================================================


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
    #############################
    # [1]. Start. Added own algorithm config registries.
    #############################
    "demir": QmixConfig,
    "rnd":   QmixConfig, 
    "ngu":   QmixConfig,
    #############################
    # [1] End.
    #############################
}