#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor zmian: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Integracja modułów DEMIR, RND i NGU z algorytmem QMIX;
#        rozszerzenie QmixConfig o parametry intrinsic reward;
#        routing DEMIR/RND/NGU w process_batch; usunięcie MAPPO/MASAC.
# Oryginał: BenchMARL (Meta Platforms), https://github.com/facebookresearch/BenchMARL
#
# Ramki komentarzowe dodane w tej sesji (inline block markers):
#
#   [1] # INTEGRATION WITH DEMIR
#       # END OF DEMIR INTEGRATION
#       Lokalizacja: _get_policy_for_loss()
#       Inicjalizacja DecentralizedEpisodicReward per group.
#
#   [2] # INTEGRATION WITH RND
#       # END OF RND INTEGRATION
#       Lokalizacja: _get_policy_for_loss()
#       Inicjalizacja RNDModule per group.
#
#   [3] # INTEGRATION WITH NGU
#       # END OF NGU INTEGRATION
#       Lokalizacja: _get_policy_for_loss()
#       Inicjalizacja NGUModule per group.
#
#   [4] # INTEGRATION WITH INTRINSIC REWARD (DEMIR / RND / NGU)
#       # END OF INTRINSIC REWARD INTEGRATION
#       Lokalizacja: process_batch()
#       Routing nagrody wewnętrznej: obliczenie r_int, skalowanie,
#       dodanie do nagrody zewnętrznej, aktualizacja pamięci (DEMIR).
#
#   [5] # INTEGRATION WITH INTRINSIC REWARD (DEMIR / RND / NGU)
#       # END OF INTRINSIC REWARD INTEGRATION
#       Lokalizacja: QmixConfig (dataclass)
#       Pola konfiguracyjne: demir_scale, encoder_type, beta1, beta2,
#       rnd_scale/embed/hidden/lr, ngu_scale/embed/hidden/k/L/eps/n/lr.
#
# ---

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
        self, 
        mixing_embed_dim: int, 
        delay_value: bool, 
        loss_function: str, 
        **kwargs
    ):
        # 1. Wyciągamy parametry DEMIRA z kwargs (tak jak w Twoim MAPPO)
        self.demir_config = {
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
            "rnd_scale":      kwargs.pop("rnd_scale", 0.0),
            "rnd_embed_dim":  kwargs.pop("rnd_embed_dim", 64),
            "rnd_hidden_dim": kwargs.pop("rnd_hidden_dim", 256),
            "rnd_lr":         kwargs.pop("rnd_lr", 1e-4),
            # NGU params
            "ngu_scale":      kwargs.pop("ngu_scale", 0.0),
            "ngu_embed_dim":  kwargs.pop("ngu_embed_dim", 64),
            "ngu_hidden_dim": kwargs.pop("ngu_hidden_dim", 256),
            "ngu_k":          kwargs.pop("ngu_k", 10),
            "ngu_L":          kwargs.pop("ngu_L", 5.0),
            "ngu_epsilon":    kwargs.pop("ngu_epsilon", 0.001),
            "ngu_n_episodic": kwargs.pop("ngu_n_episodic", 10000),
            "ngu_lr":         kwargs.pop("ngu_lr", 1e-4),
        }

        # 2. Wywołujemy super() z pozostałymi kwargs
        super().__init__(**kwargs)

        # 3. Przypisujemy standardowe parametry QMIX
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
            {group: self.observation_spec[group].clone().to(self.device)},
            device=self.device,
        )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"action_value": Unbounded(shape=logits_shape).to(self.device)},
                    shape=(n_agents,),
                    device=self.device,
                )
            },
            device=self.device,
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
        # INTEGRATION WITH DEMIR
        if not hasattr(self, "demir_modules"):
            self.demir_modules = {}

        if group not in self.demir_modules:
            obs_key = list(self.observation_spec[group].keys())[0]
            obs_dim = self.observation_spec[group, obs_key].shape[-1]
            action_dim = self.action_spec[group, "action"].space.n

            self.demir_modules[group] = DecentralizedEpisodicReward(
                obs_dim=obs_dim, 
                action_dim=action_dim, 
                config=self.demir_config
            ).to(self.device)
        # END OF DEMIR INTEGRATION

        # INTEGRATION WITH RND
        if not hasattr(self, "rnd_modules"):
            self.rnd_modules = {}
        if group not in self.rnd_modules:
            obs_key = list(self.observation_spec[group].keys())[0]
            obs_dim = self.observation_spec[group, obs_key].shape[-1]
            self.rnd_modules[group] = RNDModule(
                obs_dim=obs_dim,
                config=self.demir_config,
            ).to(self.device)
        # END OF RND INTEGRATION

        # INTEGRATION WITH NGU
        if not hasattr(self, "ngu_modules"):
            self.ngu_modules = {}
        if group not in self.ngu_modules:
            obs_key = list(self.observation_spec[group].keys())[0]
            obs_dim = self.observation_spec[group, obs_key].shape[-1]
            action_dim = self.action_spec[group, "action"].space.n
            self.ngu_modules[group] = NGUModule(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=self.demir_config,
            ).to(self.device)
        # END OF NGU INTEGRATION

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
        )
        return TensorDictSequential(*policy_for_loss, greedy).to(self.device)

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

        # INTEGRATION WITH INTRINSIC REWARD (DEMIR / RND / NGU)
        # Kajetan Frąckowiak, s28404 — routing nagrody wewnętrznej w process_batch
        scale = self.demir_config.get("demir_scale", 0.0)
        rnd_scale = self.demir_config.get("rnd_scale", 0.0)
        ngu_scale = self.demir_config.get("ngu_scale", 0.0)

        if hasattr(self, "demir_modules") and group in self.demir_modules and scale > 0:
            demir = self.demir_modules[group]
            # 1. Oblicz r_int
            r_int = demir.get_shaping_reward(
                batch, 
                group=group,
                gamma=self.experiment_config.gamma
            )
            
            # 2. Skalowanie i dodawanie
            if r_int.shape != batch[reward_key].shape:
                r_int_reduced = r_int.mean(dim=-2)
            else:
                r_int_reduced = r_int

            current_reward = batch.get(reward_key)
            batch.set(reward_key, current_reward + (scale * r_int_reduced))
            
            # 3. AKTUALIZACJA PAMIĘCI
            if "td_error" not in batch.keys():
                demir.update_memory(
                    obs=batch[group, "observation"],
                    action=batch[group, "action"],
                    reward_ext=batch[reward_key],
                    td_error=r_int.detach(),
                    next_obs=batch.get(("next", group, "observation"))
                )

        elif hasattr(self, "rnd_modules") and group in self.rnd_modules and rnd_scale > 0:
            rnd = self.rnd_modules[group]
            r_int = rnd.compute_intrinsic_reward(
                obs=batch[group, "observation"],
                group=group,
                train=True,
            )
            if r_int.shape != batch[reward_key].shape:
                r_int_reduced = r_int.mean(dim=-2)
            else:
                r_int_reduced = r_int
            current_reward = batch.get(reward_key)
            batch.set(reward_key, current_reward + (rnd_scale * r_int_reduced))

        elif hasattr(self, "ngu_modules") and group in self.ngu_modules and ngu_scale > 0:
            ngu = self.ngu_modules[group]
            next_obs = batch.get(("next", group, "observation"))
            if next_obs is not None:
                r_int = ngu.compute_intrinsic_reward(
                    obs=batch[group, "observation"],
                    next_obs=next_obs,
                    action=batch[group, "action"],
                    group=group,
                )
                if r_int.shape != batch[reward_key].shape:
                    r_int_reduced = r_int.mean(dim=-2)
                else:
                    r_int_reduced = r_int
                current_reward = batch.get(reward_key)
                batch.set(reward_key, current_reward + (ngu_scale * r_int_reduced))
        # END OF INTRINSIC REWARD INTEGRATION

        return batch

    #####################
    # Custom new methods
    #####################

    def get_mixer(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])

        if self.state_spec is not None:
            global_state_key = list(self.state_spec.keys(True, True))[0]
            state_shape = self.state_spec[global_state_key].shape
            in_keys = [(group, "chosen_action_value"), global_state_key]
        else:
            group_observation_keys = list(self.observation_spec[group].keys(True, True))
            if len(group_observation_keys) > 1:
                raise ValueError(
                    "QMIX called without a global state and multiple observation keys, currently the mixer"
                    "takes only one observation key, please raise an issue if you need this fauture."
                )
            group_observation_key = group_observation_keys[0]
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

    # INTEGRATION WITH INTRINSIC REWARD (DEMIR / RND / NGU)
    # Kajetan Frąckowiak, s28404 — pola konfiguracyjne dodane do QmixConfig
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
    encoder_type: str = "idm"  # "idm" = IDM+decorrelation | "mlp" = tylko decorrelation (ablacja)
    # RND params
    rnd_scale: float = 0.0
    rnd_embed_dim: int = 64
    rnd_hidden_dim: int = 256
    rnd_lr: float = 1e-4
    # NGU params
    ngu_scale: float = 0.0
    ngu_embed_dim: int = 64
    ngu_hidden_dim: int = 256
    ngu_k: int = 10
    ngu_L: float = 5.0
    ngu_epsilon: float = 0.001
    ngu_n_episodic: int = 10000
    ngu_lr: float = 1e-4
    # END OF INTRINSIC REWARD INTEGRATION

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
