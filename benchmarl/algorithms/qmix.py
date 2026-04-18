#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# ==============================================================================
# SEKCJA MODYFIKACJI AKADEMICKIEJ - PRACA INŻYNIERSKA (PJATK 2026)
# ==============================================================================
# Autor: Kajetan Frąckowiak, s28404
# Data: 2026
# Praca inżynierska: Polsko-Japońska Akademia Technik Komputerowych
# Modyfikacje w tym pliku :
#   [1] # Przypisanie parametrów nagród wewnętrznych (DEMIR, RND, NGU) w konstruktorze klasy Qmix.
#   [2] # Inicjalizacja modułów nagrody wewnętrznej (DEMIR, RND, NGU)
#   [3] # Zastowanie nagród wewnętrznych (DEMIR, RND, NGU) i dodanie ich do nagrody zewnętrznej w batchu.
#   [4] # Dodanie pól konfiguracyjnych dla nagród wewnętrznych (DEMIR, RND, NGU) w dataclass QmixConfig.
# ==============================================================================

import torch
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Composite, Unbounded
from torchrl.modules import EGreedyModule, QMixer, QValueModule
from torchrl.objectives import LossModule, QMixerLoss, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.algorithms.demir_module import DecentralizedEpisodicReward
from benchmarl.algorithms.rnd_module import RNDModule
from benchmarl.algorithms.ngu_module import NGUModule


class Qmix(Algorithm):
    """QMIX (from `https://arxiv.org/abs/1803.11485 <https://arxiv.org/abs/1803.11485>`__).

    Args:
        mixing_embed_dim (int): hidden dimension of the mixing network
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_value (bool): whether to separate the target value networks from the value networks used for
            data collection.

    """

    def __init__(
        self, mixing_embed_dim: int, delay_value: bool, loss_function: str, **kwargs
    ):  
        #############################
        # [1] Początek. Przypisanie parametrów nagród wewnętrznych (DEMIR, RND, NGU) w konstruktorze klasy Qmix.
        #############################
        self.int_rew_config = {
            "emb_dim_state": kwargs.pop("emb_dim_state", 64),
            "emb_dim_action": kwargs.pop("emb_dim_action", 16),
            "emb_dim_reward": kwargs.pop("emb_dim_reward", 8),
            "alpha": kwargs.pop("alpha", 0.5),
            "beta1": kwargs.pop("beta1", 1.0),
            "beta2": kwargs.pop("beta2", 0.5),
            "k": kwargs.pop("k", 10),
            "sigma": kwargs.pop("sigma", 0.1),
            "n_efm": kwargs.pop("n_efm", 10000),
            "n_edm": kwargs.pop("n_edm", 5000),
            "demir_scale": kwargs.pop("demir_scale", 0.0),
            "encoder_type": kwargs.pop("encoder_type", "idm"),
            # RND params
            "rnd_scale": kwargs.pop("rnd_scale", 0.0),
            "rnd_embed_dim": kwargs.pop("rnd_embed_dim", 64),
            "rnd_hidden_dim": kwargs.pop("rnd_hidden_dim", 256),
            "rnd_lr": kwargs.pop("rnd_lr", 1e-4),
            # NGU params
            "ngu_scale": kwargs.pop("ngu_scale", 0.0),
            "ngu_embed_dim": kwargs.pop("ngu_embed_dim", 64),
            "ngu_hidden_dim": kwargs.pop("ngu_hidden_dim", 256),
            "ngu_k": kwargs.pop("ngu_k", 10),
            "ngu_L": kwargs.pop("ngu_L", 5.0),
            "ngu_epsilon": kwargs.pop("ngu_epsilon", 0.001),
            "ngu_n_episodic": kwargs.pop("ngu_n_episodic", 10000),
            "ngu_lr": kwargs.pop("ngu_lr", 1e-4),
        }
        #############################
        # [1] Koniec.
        #############################

        super().__init__(**kwargs)

        self.delay_value = delay_value
        self.loss_function = loss_function
        self.mixing_embed_dim = mixing_embed_dim

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("QMIX is not compatible with continuous actions.")
        else:
            # Loss
            loss_module = QMixerLoss(
                policy_for_loss,
                self.get_mixer(group),
                delay_value=self.delay_value,
                loss_function=self.loss_function,
                action_space=self.action_spec[group, "action"],
            )
            loss_module.set_keys(
                reward="reward",
                action=(group, "action"),
                done="done",
                terminated="terminated",
                action_value=(group, "action_value"),
                local_value=(group, "chosen_action_value"),
                global_value="chosen_action_value",
                priority="td_error",
            )
            loss_module.make_value_estimator(
                ValueEstimators.TD0, gamma=self.experiment_config.gamma
            )

            return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return {"loss": loss.parameters()}

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        logits_shape = [
            *self.action_spec[group, "action"].shape,
            self.action_spec[group, "action"].space.n,
        ]

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"action_value": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )

        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        value_module = QValueModule(
            action_value_key=(group, "action_value"),
            action_mask_key=action_mask_key,
            out_keys=[
                (group, "action"),
                (group, "action_value"),
                (group, "chosen_action_value"),
            ],
            spec=self.action_spec[group, "action"],
            action_space=None,
        )
        #############################
        # [2] Początek. Inicjalizacja modułów nagrody wewnętrznej (DEMIR, RND, NGU)
        #############################
        if not hasattr(self, "demir_modules"):
            self.demir_modules = {}

        if group not in self.demir_modules:
            obs_key = list(self.observation_spec[group].keys())[0]
            obs_dim = self.observation_spec[group, obs_key].shape[-1]
            action_dim = self.action_spec[group, "action"].space.n

            self.demir_modules[group] = DecentralizedEpisodicReward(
                obs_dim=obs_dim, action_dim=action_dim, config=self.int_rew_config
            ).to(self.device)

        if not hasattr(self, "rnd_modules"):
            self.rnd_modules = {}
        if group not in self.rnd_modules:
            obs_key = list(self.observation_spec[group].keys())[0]
            obs_dim = self.observation_spec[group, obs_key].shape[-1]
            self.rnd_modules[group] = RNDModule(
                obs_dim=obs_dim,
                config=self.int_rew_config,
            ).to(self.device)

        if not hasattr(self, "ngu_modules"):
            self.ngu_modules = {}
        if group not in self.ngu_modules:
            obs_key = list(self.observation_spec[group].keys())[0]
            obs_dim = self.observation_spec[group, obs_key].shape[-1]
            action_dim = self.action_spec[group, "action"].space.n
            self.ngu_modules[group] = NGUModule(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=self.int_rew_config,
            ).to(self.device)
        #############################
        # [2] Koniec.
        #############################

        return TensorDictSequential(actor_module, value_module)

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        greedy = EGreedyModule(
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
            action_key=(group, "action"),
            spec=self.action_spec[(group, "action")],
            action_mask_key=action_mask_key,
            eps_init=self.experiment_config.exploration_eps_init,
            eps_end=self.experiment_config.exploration_eps_end,
        ).to(self.device)
        return TensorDictSequential(*policy_for_loss, greedy)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))

        done_key = ("next", "done")
        terminated_key = ("next", "terminated")
        reward_key = ("next", "reward")

        if done_key not in keys:
            batch.set(
                done_key,
                batch.get(("next", group, "done")).any(-2),
            )
        if terminated_key not in keys:
            batch.set(
                terminated_key,
                batch.get(("next", group, "terminated")).any(-2),
            )

        if reward_key not in keys:
            batch.set(
                reward_key,
                batch.get(("next", group, "reward")).mean(-2),
            )

        def _reduce_intrinsic_to_reward_shape(r_int: torch.Tensor) -> torch.Tensor:
            # Intrinsic modules return per-agent rewards; QMIX expects one scalar reward per env/time step.
            if r_int.shape[-1] == 1 and r_int.ndim > batch.get(reward_key).ndim:
                r_int = r_int.squeeze(-1)
            while r_int.ndim > batch.get(reward_key).ndim:
                r_int = r_int.mean(dim=-1)
            return r_int
        #############################
        # [3] Początek. Zastowanie nagród wewnętrznych (DEMIR, RND, NGU) i dodanie ich do nagrody zewnętrznej w batchu.
        #############################
        # reward_key: [batch_size, time_steps] (średnia nagroda dla wszystkich agentów w każdej chwili czasowej)
        # r_int_to_add: [batch_size, time_steps] (akumulator nagród wewnętrznych)
        r_int_to_add = torch.zeros_like(batch.get(reward_key))
        
        # --- LOGIKA DEMIR ---
        scale = self.int_rew_config.get("demir_scale", 0.0)
        if scale > 0 and hasattr(self, "demir_modules") and group in self.demir_modules:
            demir = self.demir_modules[group]
            # r_int z DEMIR: [batch_size, time_steps] (nagroda już zagregowana po agentach)
            r_int = demir.get_shaping_reward(batch, group=group, gamma=self.experiment_config.gamma)
            # r_int_to_add: [batch_size, time_steps] + r_int: [batch_size, time_steps] -> [batch_size, time_steps]
            r_int_to_add = r_int_to_add + scale * r_int

        # --- LOGIKA RND ---
        rnd_scale = self.int_rew_config.get("rnd_scale", 0.0)
        if rnd_scale > 0 and hasattr(self, "rnd_modules") and group in self.rnd_modules:
            rnd = self.rnd_modules[group]
            # RND novelty is measured on post-transition observations.
            obs_key = ("next", group, "observation")
            obs_for_rnd = batch.get(obs_key) if obs_key in keys else batch.get((group, "observation"))
            # obs z batchu: [batch_size, time_steps, n_agents, obs_dim]
            # r_int z RND: [batch_size, time_steps, n_agents, 1]
            r_int = rnd.compute_intrinsic_reward(obs=obs_for_rnd, group=group, train=True)
            r_int = _reduce_intrinsic_to_reward_shape(r_int)
            # r_int_to_add: [batch_size, time_steps] + r_int: [batch_size, time_steps] -> [batch_size, time_steps]
            r_int_to_add = r_int_to_add + rnd_scale * r_int

        # --- LOGIKA NGU ---
        ngu_scale = self.int_rew_config.get("ngu_scale", 0.0)
        if ngu_scale > 0 and hasattr(self, "ngu_modules") and group in self.ngu_modules:
            ngu = self.ngu_modules[group]
            
            if batch.get(done_key).any():
                ngu.reset_episodic_memory()
            
            next_obs = batch.get(("next", group, "observation"))
            if next_obs is not None:
                # obs, next_obs: [batch_size, time_steps, n_agents, obs_dim]
                # action: [batch_size, time_steps, n_agents]
                # r_int z NGU: [batch_size, time_steps, n_agents, 1]
                r_int = ngu.compute_intrinsic_reward(
                    obs=batch[group, "observation"],
                    next_obs=next_obs,
                    action=batch[group, "action"],
                    group=group
                )
                r_int = _reduce_intrinsic_to_reward_shape(r_int)
                # r_int_to_add: [batch_size, time_steps] + r_int: [batch_size, time_steps] -> [batch_size, time_steps]
                r_int_to_add = r_int_to_add + ngu_scale * r_int

        # Add intrinsic reward to total reward
        if torch.abs(r_int_to_add).max() > 0:
            batch.set(reward_key, batch.get(reward_key) + r_int_to_add)
        #############################
        # [3] Koniec.
        #############################

        return batch

    #####################
    # Custom new methods
    #####################

    def get_mixer(self, group: str) -> TensorDictModule:
        n_agents = len(
            self.group_map[group]
        )

        if self.state_spec is not None:
            global_state_key = list(self.state_spec.keys(True, True))[
                0
            ]
            state_shape = self.state_spec[global_state_key].shape
            in_keys = [
                (group, "chosen_action_value"),
                global_state_key,
            ]
        else:
            group_observation_keys = list(self.observation_spec[group].keys(True, True))
            if len(group_observation_keys) > 1:
                raise ValueError(
                    "QMIX called without a global state and multiple observation keys, currently the mixer"
                    "takes only one observation key, please raise an issue if you need this fauture."
                )
            group_observation_key = group_observation_keys[
                0
            ]  # [0] bo zakładamy, że jest tylko jeden klucz dla obserwacji grupy (np. "agents"), który będzie używany jako stan dla mixera
            state_shape = self.observation_spec[group, group_observation_key].shape
            in_keys = [(group, "chosen_action_value"), (group, group_observation_key)]

        mixer = TensorDictModule(
            module=QMixer(
                state_shape=state_shape,
                mixing_embed_dim=self.mixing_embed_dim,
                n_agents=n_agents,
                device=self.device,
            ),
            in_keys=in_keys,
            out_keys=["chosen_action_value"],
        )

        return mixer


@dataclass
class QmixConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Qmix`."""

    mixing_embed_dim: int = MISSING
    delay_value: bool = MISSING
    loss_function: str = MISSING

    #############################
    # [4] Początek. Dodanie pól konfiguracyjnych dla nagród wewnętrznych (DEMIR, RND, NGU) w dataclass QmixConfig.
    #############################
    # Parametry DEMIR
    emb_dim_state: int = 64
    emb_dim_action: int = 16
    emb_dim_reward: int = 8
    alpha: float = 0.5
    beta1: float = 1.0
    beta2: float = 0.5
    k: int = 10
    sigma: float = 0.1
    n_efm: int = 10000
    n_edm: int = 5000
    demir_scale: float = 0.0
    encoder_type: str = (
        "idm"
    )
    # Parametry RND
    rnd_scale: float = 0.0
    rnd_embed_dim: int = 64
    rnd_hidden_dim: int = 256
    rnd_lr: float = 1e-4
    # Parametry NGU
    ngu_scale: float = 0.0
    ngu_embed_dim: int = 64
    ngu_hidden_dim: int = 256
    ngu_k: int = 10
    ngu_L: float = 5.0
    ngu_epsilon: float = 0.001
    ngu_n_episodic: int = 10000
    ngu_lr: float = 1e-4
    #############################
    # [4] Koniec.
    #############################

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Qmix

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
