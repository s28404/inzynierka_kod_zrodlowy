"""
Module implementing configuration to SynchronizedFactory environment

Author: Kajetan Frąckowiak
Date: 2026

Description: This file contains the whole configuration classes to SynchronizedFactory 
"""

import copy
from typing import Callable, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

import torch
import numpy as np
from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

# Import custom environment
from benchmarl.environments.logic_env.factory import SynchronizedFactory

@dataclass
class LogicEnvConfig:
    grid_size: int = 10
    n_agents: int = 3


class LogicEnvClass(TaskClass):
    """TaskClass wrapper dla custom logic-based environment."""
    
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        
        def _make_env():
            pz_env = SynchronizedFactory(render_mode=None)
            env = PettingZooWrapper(pz_env, device=device)
            return env
        
        return _make_env

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return 200

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec.clone()
    
    @staticmethod
    def env_name() -> str:
        return "logic_env"


class LogicEnvTask(Task, Enum):
    SYNCHRONIZED = "synchronized" 
    
    @staticmethod
    def associated_class():
        return LogicEnvClass
