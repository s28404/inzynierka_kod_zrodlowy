"""
Moduł implementujący mechanizm RND (Random Network Distillation) do MARL bazująć na:
https://arxiv.org/abs/1810.12894 oraz https://arxiv.org/abs/2503.13077.

Autor: Kajetan Frąckowiak (s28404)
Data: 2026
Praca inżynierska: Polsko-Japońska Akademia Technik Komputerowych

Opis: Plik zawiera pełną implementację mechanizmu RND.
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

        embed_dim = self._param("rnd_embed_dim", 64)
        hidden_dim = self._param("rnd_hidden_dim", 256)
        lr = self._param("rnd_lr", 1e-4)

        # Fixed random target - NEVER trained
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

    def compute_intrinsic_reward(self, obs: torch.Tensor, group: str,
                                  train: bool = True) -> torch.Tensor:
        """
        Compute r_int = ||predictor(obs) - target(obs)||^2.
        Optionally trains the predictor.

        Args:
            obs : (B, n_agents, obs_dim) or (B*n, obs_dim)
            group: agent group name (for wandb logging)
            train: whether to update predictor weights

        Returns:
            r_int : same leading shape as obs, last dim squeezed to 1
        """
        original_shape = obs.shape
        obs_flat = obs.reshape(-1, obs.shape[-1])

        with torch.no_grad():
            target_feat = self.target(obs_flat)

        if train:
            self.optimizer.zero_grad()
            pred_feat = self.predictor(obs_flat)
            loss = nn.functional.mse_loss(pred_feat, target_feat)
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                pred_feat = self.predictor(obs_flat)

        # r_int per sample  (B*n,)
        with torch.no_grad():
            r_int_flat = ((pred_feat.detach() - target_feat) ** 2).mean(dim=-1)

        # Normalize with running stats
        r_np = r_int_flat.cpu().numpy()
        self.rms.update(r_np)
        r_norm = self.rms.normalize(r_np)
        r_norm_t = torch.from_numpy(r_norm.astype(np.float32)).to(obs.device)

        # Wandb logging
        if _wandb is not None and _wandb.run is not None:
            _wandb.log({
                f"rnd/{group}/intrinsic_reward": float(r_np.mean()),
            }, commit=False)

        # Reshape back to (B, n_agents, 1) or (B, 1)
        result = r_norm_t.reshape(*original_shape[:-1], 1)
        return result
