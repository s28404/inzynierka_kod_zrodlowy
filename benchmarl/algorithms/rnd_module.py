"""
Module implementing the RND (Random Network Distillation) mechanism for MARL based on:
https://arxiv.org/abs/1810.12894 and https://arxiv.org/abs/2503.13077.

Author: Kajetan Frąckowiak
Date: 2026

Description: This file contains a complete implementation of the RND mechanism.
"""

import numpy as np
import torch
from torch import nn
from benchmarl.algorithms.common import RunningMeanStd

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


class RNDModule(nn.Module):
    """
    Random Network Distillation - decentralized per-agent intrinsic reward.

    Params (config):
        rnd_embed_dim  : embedding dimension (default 64)
        rnd_hidden_dim : hidden layer size (default 256)
        rnd_lr         : learning rate for predictor (default 1e-4)
    """

    def __init__(self, obs_dim: int, config=None):
        super().__init__()
        self.config = config

        # We set default values for the hyperparemeters, they are used if are not found in the config.
        embed_dim = self._param("rnd_embed_dim", 64)
        hidden_dim = self._param("rnd_hidden_dim", 256)
        lr = self._param("rnd_lr", 1e-4)

        # Fixed random target - never trained
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Trainable predictor
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.rms = RunningMeanStd()

    def _param(self, name, default):
        if self.config is None:
            return default
        return getattr(self.config, name, default)

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, group: str, train: bool = True
    ) -> torch.Tensor:
        """
        Compute the intrinsic reward $r_{int} = \|\text{predictor}(obs) - \text{target}(obs)\|^2$.

        Args:
            obs   : [batch_size, n_agents, obs_dim] or [batch_size*n_agents, obs_dim]
            group : name of the agent group (for wandb logging)
            train : whether to update the predictor weights

        Returns:
            r_int : [batch_size, n_agents, 1] or [batch_size, 1]
        """
        # obs_dim is the dimension of the vector what the one concrete agent sees like position [x, y], health [hp] would have obs_dim=3
        original_shape = obs.shape                        # [batch_size, n_agents, obs_dim]
        obs_flat = obs.reshape(-1, obs.shape[-1]).float() # [batch_size*n_agents, obs_dim]

        # Extract features from the target network (no updates)
        with torch.no_grad():
            target_feat = self.target(obs_flat)           # [batch_size*n_agents, obs_dim] -> [batch_size*n_agents, embed_dim]

        if train:
            self.optimizer.zero_grad()
            # Extract features from the predictor (trained network)
            pred_feat = self.predictor(obs_flat)          # [batch_size*n_agents, obs_dim] -> [batch_size*n_agents, embed_dim]
            loss = nn.functional.mse_loss(pred_feat, target_feat)
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                pred_feat = self.predictor(obs_flat)

        # Calculate RND reward (squared feature difference)
        with torch.no_grad():
            # ((pred_feat.detach() - target_feat) ** 2).shape = [batch_size*n_agents, embed_dim]
            # we want the mean across the embed_dim .mean(dim=-1) -> [batch_size*n_agents]
            r_int_flat = ((pred_feat.detach() - target_feat) ** 2).mean(dim=-1) 

        # Normalize using the std and clip, not the full z-score from rms to not get negative rewards
        # Rms wants numpy arrays
        r_np = r_int_flat.cpu().numpy()  # [batch_size*n_agents]
        self.rms.update(r_np)            # update running mean and std with the new batch of rewards
        r_std = np.sqrt(self.rms.var) + 1e-8
        r_norm = np.clip(r_np / r_std, 0.0, self._param("rnd_reward_clip", 10.0))
        
        # Convert back to tensor: [batch_size*n_agents]
        r_norm_t = torch.from_numpy(r_norm.astype(np.float32)).to(obs.device)

        if _wandb is not None and _wandb.run is not None:
            _wandb.log(
                {
                    f"rnd/{group}/intrinsic_reward_raw": float(r_np.mean()),
                    f"rnd/{group}/intrinsic_reward_norm": float(r_norm.mean()),
                },
                commit=False,
            )

        # Reshape back to the original shape
        # original_shape = [batch_size, n_agents, obs_dim]
        # We want obs_dim to be 1, because the reward is a scalar
        # [batch_size*n_agents] -> [batch_size, n_agents, 1]
        result = r_norm_t.reshape(*original_shape[:-1], 1)
        return result