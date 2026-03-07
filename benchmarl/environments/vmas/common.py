#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor zmian: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Ograniczono enum VmasTask do trzech map eksperymentalnych:
#        SIMPLE_SPREAD, SIMPLE_TAG, SIMPLE_CRYPTO.
# Oryginał: BenchMARL (Meta Platforms), https://github.com/facebookresearch/BenchMARL
# ---
import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING


class VmasClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: VmasEnv(
            scenario=self.name.lower(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        # Pobieramy pełną specyfikację i usuwamy wymiar batcha (np. [10, 3, 4] -> [3, 4])
        observation_spec = env.full_observation_spec.to("cpu")
        if len(env.batch_size) > 0:
            observation_spec = observation_spec[0]
        
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        info_spec = env.full_observation_spec.to("cpu")
        if len(env.batch_size) > 0:
            info_spec = info_spec[0]
            
        for group in self.group_map(env):
            del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        else:
            return None

    def action_spec(self, env: EnvBase) -> Composite:
        # Tutaj leżał główny powód błędu ValueError
        action_spec = env.full_action_spec.to("cpu")
        if len(env.batch_size) > 0:
            action_spec = action_spec[0]
        return action_spec
    
    @staticmethod
    def env_name() -> str:
        return "vmas"


class VmasTask(Task):
    """Enum for VMAS tasks."""

    BALANCE = None
    SAMPLING = None
    NAVIGATION = None
    TRANSPORT = None
    REVERSE_TRANSPORT = None
    WHEEL = None
    DISPERSION = None
    MULTI_GIVE_WAY = None
    DROPOUT = None
    GIVE_WAY = None
    WIND_FLOCKING = None
    PASSAGE = None
    JOINT_PASSAGE = None
    JOINT_PASSAGE_SIZE = None
    BALL_PASSAGE = None
    BALL_TRAJECTORY = None
    BUZZ_WIRE = None
    FLOCKING = None
    DISCOVERY = None
    FOOTBALL = None
    SIMPLE_CRYPTO = None
    SIMPLE_SPREAD = None
    SIMPLE_TAG = None

    @staticmethod
    def associated_class():
        return VmasClass
