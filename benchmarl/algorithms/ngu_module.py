"""
Module implementing the NGU (Never Give Up) mechanism for MARL based on:
https://arxiv.org/abs/2002.06038 and https://arxiv.org/abs/2512.01321.

Author: Kajetan Frąckowiak 
Date: 2026

Description: This file contains a full implementation of the NGU mechanism.
"""

import numpy as np
import torch
from torch import nn
from benchmarl.algorithms.common import RunningMeanStd

try:
    import faiss
except ImportError:
    faiss = None

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


class NGUModule(nn.Module):
    """
    Never Give Up - decentralized per-agent intrinsic reward.

    Params (config):
        ngu_embed_dim   : state embedding dimension (default 64)
        ngu_hidden_dim  : IDM hidden size (default 256)
        ngu_k           : k nearest neighbours for episodic reward (default 10)
        ngu_L           : upper bound for alpha (default 5.0)
        ngu_epsilon     : denominator stabilizer for r_episodic (default 0.001)
        ngu_n_episodic  : rolling episodic buffer size (default 10000)
        ngu_lr          : learning rate (default 1e-4)
        ngu_rebuild_interval : FAISS rebuild frequency (default 50)
    """

    def __init__(self, obs_dim: int, action_dim: int, config=None):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        d = self._p("ngu_embed_dim", 64)
        hidden = self._p("ngu_hidden_dim", 256)
        self.k = self._p("ngu_k", 10)
        self.L = self._p("ngu_L", 5.0)
        self.eps = self._p("ngu_epsilon", 0.001)
        self.n_episodic = self._p("ngu_n_episodic", 10000)
        lr = self._p("ngu_lr", 1e-4)
        self.rebuild_interval = self._p("ngu_rebuild_interval", 50)

        # --- Embedding network (trained via IDM) ---
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )
        # Inverse Dynamics Model: (e_s, e_s') -> predicted action
        self.idm = nn.Sequential(
            nn.Linear(d * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

        # --- RND for lifelong curiosity ---
        self.rnd_target = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, d),
        )
        for p in self.rnd_target.parameters():
            p.requires_grad = False
            
        self.rnd_predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, d),
        )

        self.optimizer = torch.optim.Adam(
            list(self.phi.parameters())
            + list(self.idm.parameters())
            + list(self.rnd_predictor.parameters()),
            lr=lr,
        )

        # --- Episodic buffer (rolling, FAISS-backed) ---
        self.epi_buffer = np.zeros((self.n_episodic, d), dtype=np.float32)
        self.epi_ptr = 0
        self.epi_count = 0
        self.faiss_index = faiss.IndexFlatL2(d) if faiss is not None else None
        self._update_counter = 0

        # --- Normalization ---
        self.rnd_rms = RunningMeanStd()
        self.ep_rms = RunningMeanStd()

    def reset_episodic_memory(self):
        """
        Resets the episodic buffer and FAISS index at the start of a new episode.
        This is CRITICAL for NGU to function as intended.
        """
        d = self._p("ngu_embed_dim", 64)
        self.epi_buffer = np.zeros((self.n_episodic, d), dtype=np.float32)
        self.epi_ptr = 0
        self.epi_count = 0
        self.faiss_index = faiss.IndexFlatL2(d) if faiss is not None else None
        self._update_counter = 0

    def _p(self, name, default):
        if self.config is None:
            return default
        return getattr(self.config, name, default)

    def _vec_write(self, buf, ptr, count, data, capacity):
        n = data.shape[0]
        if n >= capacity:
            buf[:] = data[-capacity:]
            return 0, capacity
        end = ptr + n
        if end <= capacity:
            buf[ptr:end] = data
        else:
            split = capacity - ptr
            buf[ptr:] = data[:split]
            buf[: n - split] = data[split:]
        return end % capacity, min(count + n, capacity)

    def _rebuild_faiss_index(self):
        if faiss is None:
            return
        dim = self.epi_buffer.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        filled = min(self.epi_count, self.n_episodic)
        if filled > 0:
            self.faiss_index.add(self.epi_buffer[:filled])

    def _episodic_reward_from_knn(self, query_embeddings: np.ndarray) -> np.ndarray:
        filled = min(self.epi_count, self.n_episodic)
        if filled < self.k:
            return np.ones(query_embeddings.shape[0], dtype=np.float32)

        if (
            faiss is not None
            and self.faiss_index is not None
            and self.faiss_index.ntotal >= self.k
        ):
            distances, _ = self.faiss_index.search(query_embeddings, self.k)
        else:
            ref = self.epi_buffer[:filled]
            sq_dist = np.sum(
                (query_embeddings[:, None, :] - ref[None, :, :]) ** 2,
                axis=-1,
            )
            distances = np.partition(sq_dist, kth=self.k - 1, axis=1)[:, : self.k]

        return 1.0 / (np.sqrt(np.mean(distances, axis=1) + 1e-8) + self.eps)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        group: str,
    ) -> torch.Tensor:
        """
        Compute the NGU (Never Give Up) intrinsic reward.

        Args:
            obs      : [batch_size, n_agents, obs_dim]
            next_obs : [batch_size, n_agents, obs_dim]
            action   : [batch_size, n_agents] (discrete actions) or [batch_size, n_agents, action_dim] (continuous)
            group    : name of the agent group

        Returns:
            r_int : [batch_size, n_agents, 1] (intrinsic reward)
        """
        # Store the original shape for reconstruction
        original_shape = obs.shape 
        
        # Flatten: [batch_size, n_agents, obs_dim] -> [batch_size*n_agents, obs_dim]
        obs_flat = obs.reshape(-1, obs.shape[-1]).float()
        next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1]).float()
        n = obs_flat.shape[0]  # total items (batch_size * n_agents)

        # Flatten actions: [batch_size, n_agents] -> [batch_size*n_agents]
        if not action.is_floating_point():
            action_idx = action.reshape(-1).long()
        else:
            action_idx = action.reshape(-1, action.shape[-1]).argmax(dim=-1)

        # --- Train IDM + RND predictor ---
        self.optimizer.zero_grad()

        # Embedding extraction
        # obs_flat, next_obs_flat: [batch_size*n_agents, obs_dim] -> e_s, e_s_next: [batch_size*n_agents, embed_dim]
        e_s = self.phi(obs_flat)
        e_s_next = self.phi(next_obs_flat)

        # IDM loss: predict discrete action using cross-entropy.
        pred_action_logits = self.idm(torch.cat([e_s, e_s_next], dim=-1))
        idm_loss = nn.functional.cross_entropy(pred_action_logits, action_idx)

        # RND loss: train predictor on post-transition states.
        with torch.no_grad():
            rnd_target_feat = self.rnd_target(next_obs_flat)
        rnd_pred_feat = self.rnd_predictor(next_obs_flat)
        rnd_loss = nn.functional.mse_loss(rnd_pred_feat, rnd_target_feat)

        (idm_loss + rnd_loss).backward()
        self.optimizer.step()

        with torch.no_grad():
            # Use next-state embeddings for episodic novelty
            e_query_np = e_s_next.detach().cpu().numpy().astype(np.float32)
            rnd_r_np = (
                ((rnd_pred_feat.detach() - rnd_target_feat) ** 2)
                .mean(dim=-1)
                .cpu()
                .numpy()
            )

        # --- Episodic reward (k-NN in embedding space) ---
        r_ep = self._episodic_reward_from_knn(e_query_np)

        # --- Lifelong curiosity (RND-based alpha) ---
        self.rnd_rms.update(rnd_r_np)
        rnd_std = np.sqrt(self.rnd_rms.var) + 1e-8
        rnd_scaled = np.clip(rnd_r_np / rnd_std, 0.0, self.L - 1.0)
        alpha = np.clip(1.0 + rnd_scaled, 1.0, self.L)

        # --- Combined intrinsic reward ---
        r_int = r_ep * alpha

        # Normalize and clip the intrinsic reward
        self.ep_rms.update(r_int)
        ep_std = np.sqrt(self.ep_rms.var) + 1e-8
        r_int_norm = np.clip(r_int / ep_std, 0.0, self._p("ngu_reward_clip", 10.0))

        # --- Update episodic memory after reward computation ---
        old_ptr = self.epi_ptr
        old_count = self.epi_count
        self.epi_ptr, self.epi_count = self._vec_write(
            self.epi_buffer,
            self.epi_ptr,
            self.epi_count,
            e_query_np,
            self.n_episodic,
        )

        if faiss is not None:
            can_append_incrementally = (
                self.faiss_index is not None
                and old_count < self.n_episodic
                and self.faiss_index.ntotal == old_count
                and old_ptr + e_query_np.shape[0] <= self.n_episodic
            )
            if can_append_incrementally:
                self.faiss_index.add(e_query_np)
            else:
                self._rebuild_faiss_index()

        if _wandb is not None and _wandb.run is not None:
            _wandb.log(
                {
                    f"ngu/{group}/r_episodic": float(r_ep.mean()),
                    f"ngu/{group}/alpha": float(alpha.mean()),
                    f"ngu/{group}/r_int_raw": float(r_int.mean()),
                    f"ngu/{group}/r_int_norm": float(r_int_norm.mean()),
                },
                commit=False,
            )

        # Reshape back to original dimensions
        # r_int_norm: [batch_size*n_agents] -> r_t: [batch_size*n_agents, 1] -> result: [batch_size, n_agents, 1]
        r_t = torch.from_numpy(r_int_norm.astype(np.float32)).to(obs.device)
        return r_t.reshape(*original_shape[:-1], 1)