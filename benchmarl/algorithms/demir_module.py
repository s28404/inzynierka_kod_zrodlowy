"""
Module implementing the DEMIR (Decentralized Episodic Memory for Intrinsic Reward) mechanism.

Author: Kajetan Frąckowiak
Date: 2026

Description: This file contains the full implementation of the DEMIR mechanism.
"""

import torch
import numpy as np
import faiss
from torch import nn
from benchmarl.algorithms.common import RunningMeanStd
from benchmarl.utils import get_td_value

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


class DemirEncoders(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        if isinstance(config, dict):
            self.d_s = config.get("emb_dim_state", 64)
            self.d_a = config.get("emb_dim_action", 16)
            self.d_r = config.get("emb_dim_reward", 8)
        else:
            self.d_s = getattr(config, "emb_dim_state", 64)
            self.d_a = getattr(config, "emb_dim_action", 16)
            self.d_r = getattr(config, "emb_dim_reward", 8)

        self.phi_s = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, self.d_s)
        )
        self.phi_a = nn.Sequential(
            nn.Linear(action_dim, 32), nn.ReLU(), nn.Linear(32, self.d_a)
        )
        self.phi_r = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, self.d_r))
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.d_s * 2, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

    def forward_kinematics(self, e_s, e_s_next):
        # e_s.shape = e_s_next.shape = [batch_size, d_s]
        return self.inverse_dynamics(torch.cat([e_s, e_s_next], dim=-1))

    def encoder_state(self, obs):
        return self.phi_s(obs)

    def encode_full_experience(self, obs, action, reward_ext):
        e_s = self.phi_s(obs)
        e_a = self.phi_a(action)

        # Expected reward.shape = [batch_size, 1] <- dim() = 2
        # dim() = 0 means reward_ext is a scalar, we need to reshape it to [1, 1] to feed into phi_r
        # dim() = 1 means reward_ext is a vector of shape [batch_size], we need to reshape it to [batch_size, 1] to feed into phi_r
        if reward_ext.dim() == 0 or reward_ext.dim() == 1:
            reward_ext = reward_ext.view(-1, 1)

        e_r = self.phi_r(reward_ext)
        # e_s.shape = [batch_size, d_s]
        # e_a.shape = [batch_size, d_a]
        # e_r.shape = [batch_size, d_r]
        e_F = torch.cat([e_s, e_a, e_r], dim=-1)
        return e_s, e_F


class DecentralizedEpisodicReward(nn.Module):
    def __init__(self, obs_dim, action_dim, config=None, **kwargs):
        super().__init__()
        self.config = config

        self.k = self._get_config_param("k", 10)
        self.n_efm = self._get_config_param("n_efm", 10000)
        self.n_edm = self._get_config_param("n_edm", 5000)
        self.emb_dim_state = self._get_config_param("emb_dim_state", 64)
        self.emb_dim_action = self._get_config_param("emb_dim_action", 16)
        self.emb_dim_reward = self._get_config_param("emb_dim_reward", 8)

        self.encoders = DemirEncoders(obs_dim, action_dim, self.config)

        d_total = self.emb_dim_state + self.emb_dim_action + self.emb_dim_reward
        self.efm_index = faiss.IndexFlatL2(d_total)
        self.edm_index = faiss.IndexFlatL2(self.emb_dim_state)

        self.efm_buffer = np.zeros((self.n_efm, d_total), dtype=np.float32)
        self.edm_buffer = np.zeros((self.n_edm, self.emb_dim_state), dtype=np.float32)
        self.efm_rhos = np.zeros(self.n_efm, dtype=np.float32)
        self.efm_ptr = 0
        self.edm_ptr = 0
        self.efm_count = 0
        self.edm_count = 0

        self.encoder_type = self._get_config_param("encoder_type", "idm")

        self.encoder_opt = torch.optim.Adam(self.encoders.parameters(), lr=1e-4)

        self.rebuild_interval = self._get_config_param("rebuild_interval", 100)
        self._update_counter = 0

        self.quality_rms = RunningMeanStd()
        self.novelty_rms = RunningMeanStd()

    def _get_config_param(self, name, default):
        if self.config is None:
            return default
        return getattr(
            self.config,
            name,
            getattr(self.config, "get", lambda x, y: y)(name, default),
        )

    def _get_k_neighbors_batch(self, index, query_embeddings, k):
        if index.ntotal < k:
            return None, None
        # query_embeddings.shape = [batch_size, dim]
        x = query_embeddings.detach().cpu().numpy().astype("float32")

        if x.ndim == 1:
            x = x.reshape(1, -1)  # from [dim] to [1, dim] for single query
        distances, indices = index.search(x, k)
        return distances, indices

    def _compute_potential(self, e_s, e_F):
        dist_q, idx_q = self._get_k_neighbors_batch(self.efm_index, e_F, self.k)
        dist_n, _ = self._get_k_neighbors_batch(self.edm_index, e_s, self.k)

        sigma = getattr(self.config, "sigma", 0.1)
        if idx_q is not None:
            mask = idx_q >= 0
            # dist_q is already squared by faiss.IndexFlatL2
            weights = np.exp(-(dist_q) / (2 * sigma**2)) * mask
            rhos = self.efm_rhos[idx_q % self.n_efm]
            # weights.shape = rhos.shape = [batch_size, k]
            quality = np.sum(weights * rhos, axis=1) / (np.sum(weights, axis=1) + 1e-6)
        else:
            quality = np.zeros(e_s.shape[0])

        novelty = (
            # dist_n is already squared by faiss.IndexFlatL2
            np.mean(np.sqrt(dist_n), axis=1) if dist_n is not None else np.zeros(e_s.shape[0])
        )

        return quality, novelty

    def get_shaping_reward(self, batch, group, gamma=0.99):
        # shape: obs, next_obs: [batch_size, n_agents, obs_dim]
        obs = batch.get((group, "observation"))
        next_obs = batch.get(("next", group, "observation"))

        # shape: action: [batch_size, n_agents] (discrete) or [batch_size, n_agents, action_dim] (continuous)
        action = (
            batch.get((group, "action"))
            if (group, "action") in list(batch.keys(True, True))
            else None
        )
        # shape: reward_ext: [batch_size, n_agents] (local) or [batch_size, 1] (global/QMIX)
        reward_ext = get_td_value(
            batch,
            [
                ("next", group, "reward"),
                (group, "reward"),
                ("next", "reward"),
                "reward",
            ],
        )
        if reward_ext is None:
            if obs is not None:
                # shape: obs.shape: [batch_size, n_agents, obs_dim] -> reward_ext: [batch_size, n_agents]
                reward_ext = torch.zeros(obs.shape[:-1], device=obs.device)
            else:
                reward_ext = torch.zeros((), device="cpu")

        warmup = getattr(self.config, "warmup", 1000)
        if self.efm_index.ntotal < self.k or self._update_counter < warmup:
            return torch.zeros_like(reward_ext)

        # shape: obs, next_obs: [batch_size, n_agents, obs_dim]
        # shape: obs_flat, next_obs_flat: [batch_size * n_agents, obs_dim]
        obs_flat = obs.reshape(-1, obs.shape[-1])
        next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1])

        n_target = obs_flat.shape[0]

        # shape: reward_flat: [batch_size * n_agents, 1] for per-agent reward, or [batch_size, 1] for global reward (QMIX)
        reward_flat = reward_ext.reshape(-1, 1)
        if reward_flat.shape[0] != n_target:
            repeats = n_target // reward_flat.shape[0]
            # shape: reward_flat: [batch_size, 1] -> [batch_size * n_agents, 1]
            reward_flat = reward_flat.repeat_interleave(repeats, dim=0)

        # shape: action -> action_flat: [batch_size * n_agents, action_dim]
        if action.dtype in [torch.long, torch.int]:
            # encoders.phi_a.shape[0] is the first layer of phi_a, which is a Linear layer, and in_features gives us the input dimension, which is the action_dim for discrete actions
            a_dim = self.encoders.phi_a[0].in_features
            # shape: [batch_size, n_agents] -> [batch_size, n_agents, action_dim] -> [batch_size * n_agents, action_dim]
            action_flat = (
                # action.shape = [batch_size, n_agents]
                torch.nn.functional.one_hot(action, num_classes=a_dim)
                .float()
                # reshape to [batch_size * n_agents, action_dim]
                .reshape(-1, a_dim)
            )
        else:
            # shape: [batch_size, n_agents, action_dim] -> [batch_size * n_agents, action_dim]
            action_flat = action.reshape(-1, action.shape[-1])

        if action_flat.shape[0] != n_target:
            # shape: [batch_size, action_dim] -> [batch_size * n_agents, action_dim]
            repeats = n_target // action_flat.shape[0]
            action_flat = action_flat.repeat_interleave(repeats, dim=0)

        # inputs shape: obs_flat: [batch_size * n_agents, obs_dim], action_flat: [batch_size * n_agents, action_dim], reward_flat: [batch_size * n_agents, 1]
        # outputs shape: e_s_t, e_F_t: [batch_size * n_agents, embedding_dim] (state and full experience embeddings at time t)
        e_s_t, e_F_t = self.encoders.encode_full_experience(
            obs_flat, action_flat, reward_flat
        )
        e_s_tp1, e_F_tp1 = self.encoders.encode_full_experience(
            next_obs_flat, action_flat, reward_flat
        )

        q_t, n_t = self._compute_potential(e_s_t, e_F_t)
        q_tp1, n_tp1 = self._compute_potential(e_s_tp1, e_F_tp1)

        self.quality_rms.update(q_tp1)
        self.novelty_rms.update(n_tp1)

        def _norm_full(q, n):
            # shape: q, n: [batch_size * n_agents] -> normalized: [batch_size * n_agents]
            qn = self.quality_rms.normalize(q)
            nn = self.novelty_rms.normalize(n)
            qn = np.clip(qn, -1.0, 1.0)
            nn = np.clip(nn, -1.0, 1.0)
            return qn, nn

        # shape: q_t_n, n_t_n, q_tp1_n, n_tp1_n: each is [batch_size * n_agents]
        q_t_n, n_t_n = _norm_full(q_t, n_t)
        q_tp1_n, n_tp1_n = _norm_full(q_tp1, n_tp1)

        beta1 = getattr(self.config, "beta1", 1.0)
        beta2 = getattr(self.config, "beta2", 0.5)

        # shape: phi_t, phi_tp1: [batch_size * n_agents]
        phi_t = beta1 * q_t_n + beta2 * n_t_n
        phi_tp1 = beta1 * q_tp1_n + beta2 * n_tp1_n

        # shape: phi_t, phi_tp1: [batch_size * n_agents] (numpy) -> [batch_size * n_agents] (torch)
        phi_t = torch.from_numpy(phi_t.astype(np.float32)).to(obs.device)
        phi_tp1 = torch.from_numpy(phi_tp1.astype(np.float32)).to(obs.device)

        # shape: phi_t, phi_tp1: [batch_size * n_agents] -> r_int: [batch_size * n_agents]
        r_int = (gamma * phi_tp1) - phi_t

        if r_int.numel() > reward_ext.numel():
            # shape: [batch_size * n_agents] -> average out over agents
            # shape: r_int: [batch_size * n_agents] -> [batch_size, n_agents] -> [batch_size] (mean across agents)
            n_agents_dim = r_int.numel() // reward_ext.numel()
            r_int = r_int.view(-1, n_agents_dim).mean(dim=-1)

        if _wandb is not None and _wandb.run is not None:
            _wandb.log(
                {
                    f"demir/{group}/phi_t": phi_t.mean().item(),
                    f"demir/{group}/r_int": r_int.mean().item(),
                },
                commit=False,
            )

        return r_int.view_as(reward_ext)

    def update_memory(self, obs, action, reward_ext, td_error, next_obs=None):
        with torch.enable_grad():
            obs_flat = obs.reshape(
                -1, obs.shape[-1]
            )  #  [batch_size, n_agents, obs_dim] -> [batch_size*n_agents, obs_dim]
            n_flat = obs_flat.shape[0]

            reward_flat = reward_ext.reshape(
                -1, 1
            )  # [batch_size, n_agents] -> [batch_size*n_agents, 1]
            if reward_flat.shape[0] != n_flat:
                repeats = n_flat // reward_flat.shape[0]
                # shape: reward_flat: [batch_size, 1] -> [batch_size*n_agents, 1]
                reward_flat = reward_flat.repeat_interleave(repeats, dim=0)

            # td_error.shape = [batch_size, n_agents], td_error_flat: [batch_size*n_agents, 1]
            td_error_flat = td_error.reshape(-1, 1)
            if td_error_flat.shape[0] != n_flat:
                repeats = n_flat // td_error_flat.shape[0]
                # shape: td_error_flat: [batch_size, 1] -> [batch_size*n_agents, 1]
                td_error_flat = td_error_flat.repeat_interleave(repeats, dim=0)

            if action.dtype in [torch.long, torch.int]:
                # encoders.phi_a.shape[0] is the first layer of phi_a, which is a Linear layer, and in_features gives us the input dimension, which is the action_dim for discrete actions
                a_dim = self.encoders.phi_a[0].in_features
                action_flat = (
                    # action.shape = [batch_size, n_agents]
                    torch.nn.functional.one_hot(action, num_classes=a_dim)
                    .float()
                    .reshape(-1, a_dim)
                )
            else:
                action_flat = action.reshape(-1, action.shape[-1])

            self.encoder_opt.zero_grad()

            if not obs_flat.requires_grad:
                obs_flat.requires_grad_(True)

            # obs_flat.shape = [batch_size*n_agents, obs_dim]
            # self.encoders.phi_s(obs_flat).shape = [batch_size*n_agents, emb_dim_state]
            e_s = self.encoders.phi_s(obs_flat)
            # e_s.mean(dim=0).shape = [emb_dim_state], e_s.std(dim=0).shape = [emb_dim_state]
            e_s_norm = (e_s - e_s.mean(dim=0)) / (e_s.std(dim=0, unbiased=False) + 1e-6)

            if self.encoder_type in ["idm", "idm_no_barlow"] and next_obs is not None:
                # next_obs.shape = [batch_size, n_agents, obs_dim] -> next_obs_flat: [batch_size*n_agents, obs_dim]
                next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1])

                if not next_obs_flat.requires_grad:
                    next_obs_flat.requires_grad_(True)

                # encoders.phi_s(next_obs_flat).shape = [batch_size*n_agents, emb_dim_state]
                e_s_next = self.encoders.phi_s(next_obs_flat)
                # pred_action.shape = [batch_size*n_agents, action_dim]
                pred_action = self.encoders.forward_kinematics(e_s, e_s_next)
                inv_loss = nn.functional.mse_loss(pred_action, action_flat)

                if self.encoder_type == "idm_no_barlow":
                    encoder_loss = inv_loss
                else:
                    # e_s_norm.shape = [batch_size*n_agents, emb_dim_state]
                    # n_flat.shape = scalar = total number of samples in the batch
                    # c.shape = [emb_dim_state, emb_dim_state]
                    c = e_s_norm.T @ e_s_norm / n_flat
                    diag_vals = torch.diag(c)
                    red_loss = (c - torch.diag(diag_vals)).pow(2).mean()
                    encoder_loss = inv_loss + 0.01 * red_loss

                encoder_loss.backward()
                self.encoder_opt.step()
            elif self.encoder_type == "mlp":
                c = e_s_norm.T @ e_s_norm / n_flat
                diag_vals = torch.diag(c)
                red_loss = (c - torch.diag(diag_vals)).pow(2).mean()
                (0.01 * red_loss).backward()
                self.encoder_opt.step()

        with torch.no_grad():
            # obs_flat.shape = [batch_size*n_agents, obs_dim], action_flat.shape = [batch_size*n_agents, action_dim], reward_flat.shape = [batch_size*n_agents, 1]
            # e_s.shape = [batch_size*n_agents, emb_dim_state]
            # e_F.shape = [batch_size*n_agents, emb_dim_state + emb_dim_action + emb_dim_reward]
            e_s, e_F = self.encoders.encode_full_experience(
                obs_flat, action_flat, reward_flat
            )

        alpha = getattr(self.config, "alpha", 0.5)
        rho = alpha * torch.abs(reward_flat) + (1 - alpha) * torch.abs(td_error_flat)

        rho_np = rho.detach().cpu().numpy().flatten()
        efm_np = e_F.detach().cpu().numpy().astype("float32")
        edm_np = e_s.detach().cpu().numpy().astype("float32")

        def _vec_write(buf, ptr, count, data, capacity):
            # buf.shape = [capacity, dim],
            # data.shape = [n, dim], where n is the number of new entries to write, and dim is the same as buf.shape[1]
            # capacity is the maximum number of entries the buffer can hold (buf.shape[0])
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

        efm_start_ptr = self.efm_ptr
        self.efm_ptr, self.efm_count = _vec_write(
            self.efm_buffer, self.efm_ptr, self.efm_count, efm_np, self.n_efm
        )

        # efm_rhos.shape = [n_efm], rho_np.shape = [n], where n is the number of new entries we want to write
        # efm_rhos.reshape(-1, 1) -> [n_efm, 1], rho_np.reshape(-1, 1) -> [n, 1]
        # We need [n_efm, 1] and [n, 1] shapes to perform the vectorized write operation,
        _vec_write(
            self.efm_rhos.reshape(-1, 1),
            efm_start_ptr,
            0,
            rho_np.reshape(-1, 1),
            self.n_efm,
        )

        self.edm_ptr, self.edm_count = _vec_write(
            self.edm_buffer, self.edm_ptr, self.edm_count, edm_np, self.n_edm
        )

        self._update_counter += 1

        if (
            self.efm_index.ntotal == 0
            or self._update_counter % self.rebuild_interval == 0
        ):
            filled_efm = min(self.efm_count, self.n_efm)
            filled_edm = min(self.edm_count, self.n_edm)

            if filled_efm > 0:
                # efm_buffer.shape = [n_efm, d_total]
                new_efm = faiss.IndexFlatL2(self.efm_buffer.shape[1])
                new_efm.add(self.efm_buffer[:filled_efm])
                self.efm_index = new_efm

            if filled_edm > 0:
                # edm_buffer.shape = [n_edm, emb_dim_state]
                new_edm = faiss.IndexFlatL2(self.edm_buffer.shape[1])
                new_edm.add(self.edm_buffer[:filled_edm])
                self.edm_index = new_edm
