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

        d = self._param("ngu_embed_dim", 64)
        hidden = self._param("ngu_hidden_dim", 256)
        self.k = self._param("ngu_k", 10)
        self.L = self._param("ngu_L", 5.0)
        self.eps = self._param("ngu_epsilon", 0.001)
        self.n_episodic = self._param("ngu_n_episodic", 10000)
        lr = self._param("ngu_lr", 1e-4)
        self.rebuild_interval = self._param("ngu_rebuild_interval", 50)

        #  NGU kernel parameters from Badia et al., 2020
        self.kernel_epsilon = self._param("ngu_kernel_epsilon", 0.0001)  # eps in K(x,y) = eps / (d^2/d_m^2 + eps)
        self.pseudo_counts = self._param("ngu_pseudo_counts", 0.001)      # c in denominator

        #  Running average of squared distances of k-th nearest neighbor (d_m^2)
        # This normalizes the kernel across different density regions.
        self._running_d_m_sq = 1.0  # Initialize to 1.0 to avoid division by zero
        self._d_m_update_rate = 0.01  # Exponential moving average rate

        # Embedding network (trained via IDM)
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )
        # Inverse Dynamics Model: (e_s, e_s') -> predicted action
        self.idm = nn.Sequential(
             # to predict action that caused stransition we need two states not one so we multiple d by 2
            nn.Linear(d * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

        # RND for lifelong curiosity
        #  Use ReLU instead of LeakyReLU (Badia et al., 2020 uses ReLU)
        self.rnd_target = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )
        for p in self.rnd_target.parameters():
            p.requires_grad = False
            
        #  Use ReLU instead of LeakyReLU (Badia et al., 2020 uses ReLU)
        self.rnd_predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

        self.optimizer = torch.optim.Adam(
            list(self.phi.parameters())
            + list(self.idm.parameters())
            + list(self.rnd_predictor.parameters()),
            lr=lr,
        )

        # Episodic buffer (rolling, FAISS-backed) 
        self.epi_buffer = np.zeros((self.n_episodic, d), dtype=np.float32)
        self.epi_ptr = 0
        self.epi_count = 0
        self.faiss_index = faiss.IndexFlatL2(d) if faiss is not None else None
        self._update_counter = 0

        # Normalization 
        self.rnd_rms = RunningMeanStd()
        self.ep_rms = RunningMeanStd()

        #  UVFA with multiple beta values (Badia et al., 2020, Sec. 2.3)
        # NGU trains N policies in parallel, each with a different beta (mixing coefficient)
        # that balances intrinsic vs extrinsic reward. The betas are logarithmically spaced
        # from beta_min to beta_max so that some policies explore heavily while others exploit.
        self.n_policies = self._param("ngu_n_policies", 32)
        self.beta_min = self._param("ngu_beta_min", 0.0)  # pure extrinsic
        self.beta_max = self._param("ngu_beta_max", 1.0)  # max intrinsic scaling
        # Logarithmically spaced betas: more policies near 0 (exploit), fewer near 1 (explore)
        if self.n_policies > 1:
            log_betas = np.linspace(0, 1, self.n_policies)
            self.betas = torch.tensor(
                self.beta_min + (self.beta_max - self.beta_min) * log_betas,
                dtype=torch.float32,
            )
        else:
            self.betas = torch.tensor([self.beta_max], dtype=torch.float32)

        # Per-policy episodic memories (each policy has its own episodic buffer)
        self.policy_epi_buffers = [np.zeros((self.n_episodic, d), dtype=np.float32) for _ in range(self.n_policies)]
        self.policy_epi_ptrs = [0] * self.n_policies
        self.policy_epi_counts = [0] * self.n_policies
        self.policy_faiss_indices = [faiss.IndexFlatL2(d) if faiss is not None else None for _ in range(self.n_policies)]

    def reset_episodic_memory(self):
        """
        Resets the episodic buffer and FAISS index at the start of a new episode.
        Also resets per-policy episodic memories for UVFA.
        """
        d = self._param("ngu_embed_dim", 64)
        self.epi_buffer = np.zeros((self.n_episodic, d), dtype=np.float32)
        self.epi_ptr = 0
        self.epi_count = 0
        self.faiss_index = faiss.IndexFlatL2(d) if faiss is not None else None
        self._update_counter = 0
        #  Reset per-policy episodic memories for UVFA
        for i in range(self.n_policies):
            self.policy_epi_buffers[i] = np.zeros((self.n_episodic, d), dtype=np.float32)
            self.policy_epi_ptrs[i] = 0
            self.policy_epi_counts[i] = 0
            self.policy_faiss_indices[i] = faiss.IndexFlatL2(d) if faiss is not None else None

    def _param(self, name, default):
        if self.config is None:
            return default
        return getattr(self.config, name, default)

    def _vec_write(self, buf, ptr, count, data, n_episodic):
        # data.shape = [n, d]
        n = data.shape[0]               # number of new items to write
        if n >= n_episodic:
            # If the number of new items exceeds n_episodic
            # take the whole buffer from start to the end buf[:]
            # and fill with the last n_episodic items from data
            buf[:] = data[-n_episodic:]
            return 0, n_episodic
        # end = where we ended up in last write + how many new items we want to write
        end = ptr + n
        if end <= n_episodic:
            # When ptr=9000, n=500, n_episodic=10000
            # We can just write from 9000 to 9500
            buf[ptr:end] = data
        else:
            # When ptr=9000, n=1500, n_episodic=10000
            # Split=1000
            split = n_episodic - ptr
            # buf[9000:10000] = data[:1000]
            buf[ptr:] = data[:split]
            # buf[0:500] = data[1000:1500]
            buf[: n - split] = data[split:]
        epi_ptr = end % n_episodic
        epi_count = min(count + n, n_episodic)
        return epi_ptr, epi_count

    def _rebuild_faiss_index(self):
        if faiss is None:
            return
        # Since faiss does not know about our overwriting buf[ptr:end] = data
        # we need to rebuild the index from the buffer every self.rebuild_interval updates.
        # with filling the index with the data that are in the buffer
        # epi_buffer.shape = [n_episodic, d]
        dim = self.epi_buffer.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        filled = min(self.epi_count, self.n_episodic)
        if filled > 0:
            self.faiss_index.add(self.epi_buffer[:filled])

    def _episodic_reward_from_knn(self, query_embeddings: np.ndarray) -> np.ndarray:
        #  Implements the full NGU kernel density estimation from Badia et al., 2020:
        #   r_episodic = 1 / (sum_{f_i in N_k} K(f(x_t), f_i) + c)
        #   K(x, y) = eps / (d^2(x, y) / d_m^2 + eps)
        # where d_m^2 is the running average of squared distances of k-th nearest neighbors.
        filled = min(self.epi_count, self.n_episodic)
        if filled < self.k:
            # Not enough neighbors yet — return max reward to encourage exploration
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

        # distances.shape = [batch_size*n_agents, k]  — these are L2 squared distances from FAISS
        #  Update running d_m^2: exponential moving average of the k-th NN squared distance
        # The k-th nearest neighbor is the last column (distances are sorted by FAISS)
        kth_sq_dist = distances[:, -1]  # shape: [batch_size*n_agents]
        batch_d_m_sq = float(np.mean(kth_sq_dist))
        self._running_d_m_sq = (
            (1.0 - self._d_m_update_rate) * self._running_d_m_sq
            + self._d_m_update_rate * max(batch_d_m_sq, 1e-8)
        )

        d_m_sq = self._running_d_m_sq
        kernel_vals = self.kernel_epsilon / (distances / (d_m_sq + 1e-8) + self.kernel_epsilon)
        # kernel_vals.shape = [batch_size*n_agents, k]

        r_episodic = 1.0 / (np.sum(kernel_vals, axis=1) + self.pseudo_counts)

        return r_episodic

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
        original_shape = obs.shape # [batch_size, n_agents, obs_dim]
        
        obs_flat = obs.reshape(-1, obs.shape[-1]).float()                # [batch_size, n_agents, obs_dim] -> [batch_size*n_agents, obs_dim]
        next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1]).float() # [batch_size, n_agents, obs_dim] -> [batch_size*n_agents, obs_dim]
        n = obs_flat.shape[0]

        # Batch of actions' indices that we want to predict with IDM. If actions are discrete, we can just flatten them.
        if not action.is_floating_point():
            action_idx = action.reshape(-1).long()                           # [batch_size, n_agents] -> [batch_size*n_agents]
        else:
            action_idx = action.reshape(-1, action.shape[-1]).argmax(dim=-1) # [batch_size, n_agents, action_dim] -> [batch_size*n_agents, action_dim] -> [batch_size*n_agents]

        self.optimizer.zero_grad()

        e_s = self.phi(obs_flat)           # [batch_size*n_agents, obs_dim] -> [batch_size*n_agents, embed_dim]
        e_s_next = self.phi(next_obs_flat) # [batch_size*n_agents, obs_dim] -> [batch_size*n_agents, embed_dim]

        pred_action_logits = self.idm(torch.cat([e_s, e_s_next], dim=-1))       # [batch_size*n_agents, embed_dim*2] -> [batch_size*n_agents, action_dim]
        idm_loss = nn.functional.cross_entropy(pred_action_logits, action_idx)
 
        with torch.no_grad():
            rnd_target_feat = self.rnd_target(next_obs_flat) # [batch_size*n_agents, obs_dim] -> [batch_size*n_agents, embed_dim]
        rnd_pred_feat = self.rnd_predictor(next_obs_flat)    # [batch_size*n_agents, obs_dim] -> [batch_size*n_agents, embed_dim]
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
            # rnd_r_np.shape = [batch_size*n_agents]

        # Episodic reward from k-nearest neighbors in the embedding space
        r_ep = self._episodic_reward_from_knn(e_query_np)

        # Lifelong curiosity (RND-based alpha) 
        self.rnd_rms.update(rnd_r_np)
        rnd_std = np.sqrt(self.rnd_rms.var) + 1e-8
        rnd_scaled = np.clip(rnd_r_np / rnd_std, 0.0, self.L - 1.0)
        alpha = np.clip(1.0 + rnd_scaled, 1.0, self.L)

        # Combined intrinsic reward 
        r_int = r_ep * alpha

        # Normalize and clip the intrinsic reward
        self.ep_rms.update(r_int)
        ep_std = np.sqrt(self.ep_rms.var) + 1e-8
        r_int_norm = np.clip(r_int / ep_std, 0.0, self._param("ngu_reward_clip", 10.0))

        # Update episodic memory after reward computation 
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

    def compute_intrinsic_reward_uvfa(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        group: str,
    ) -> torch.Tensor:
        """
         UVFA variant: compute intrinsic rewards for all N beta policies.
        
        Returns intrinsic rewards shaped as [batch_size, n_agents, n_policies]
        where each slice along the last dim corresponds to a different beta policy.
        This implements the NGU paper's UVFA with N=32 beta policies.
        
        Each policy i gets: r_int_i = r_episodic * alpha_i, where alpha_i is scaled
        by beta_i. Higher beta = more intrinsic exploration reward.
        """
        original_shape = obs.shape

        obs_flat = obs.reshape(-1, obs.shape[-1]).float()
        next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1]).float()
        n = obs_flat.shape[0]

        if not action.is_floating_point():
            action_idx = action.reshape(-1).long()
        else:
            action_idx = action.reshape(-1, action.shape[-1]).argmax(dim=-1)

        self.optimizer.zero_grad()

        e_s = self.phi(obs_flat)
        e_s_next = self.phi(next_obs_flat)

        pred_action_logits = self.idm(torch.cat([e_s, e_s_next], dim=-1))
        idm_loss = nn.functional.cross_entropy(pred_action_logits, action_idx)

        with torch.no_grad():
            rnd_target_feat = self.rnd_target(next_obs_flat)
        rnd_pred_feat = self.rnd_predictor(next_obs_flat)
        rnd_loss = nn.functional.mse_loss(rnd_pred_feat, rnd_target_feat)

        (idm_loss + rnd_loss).backward()
        self.optimizer.step()

        with torch.no_grad():
            e_query_np = e_s_next.detach().cpu().numpy().astype(np.float32)
            rnd_r_np = (
                ((rnd_pred_feat.detach() - rnd_target_feat) ** 2)
                .mean(dim=-1)
                .cpu()
                .numpy()
            )

        # Compute base episodic reward (shared across all policies)
        r_ep = self._episodic_reward_from_knn(e_query_np)

        # Lifelong curiosity (RND-based alpha) - shared base
        self.rnd_rms.update(rnd_r_np)
        rnd_std = np.sqrt(self.rnd_rms.var) + 1e-8
        rnd_scaled = np.clip(rnd_r_np / rnd_std, 0.0, self.L - 1.0)
        alpha_base = np.clip(1.0 + rnd_scaled, 1.0, self.L)

        #  Compute per-policy intrinsic rewards using UVFA betas
        # r_int_i = r_episodic * alpha * beta_i
        # Each beta_i controls how much intrinsic reward policy i receives
        n_flat = r_ep.shape[0]  # batch_size * n_agents
        n_pol = self.n_policies
        betas_np = self.betas.cpu().numpy()  # [n_policies]

        # r_ep: [n_flat], alpha_base: [n_flat], betas: [n_policies]
        # Result: [n_flat, n_policies] where col i = r_ep * alpha_base * beta_i
        r_int_all = np.outer(r_ep * alpha_base, betas_np)  # [n_flat, n_policies]

        # Normalize per policy
        self.ep_rms.update(r_int_all.ravel())
        ep_std = np.sqrt(self.ep_rms.var) + 1e-8
        r_int_norm_all = np.clip(r_int_all / ep_std, 0.0, self._param("ngu_reward_clip", 10.0))

        # Update shared episodic memory
        old_ptr = self.epi_ptr
        old_count = self.epi_count
        self.epi_ptr, self.epi_count = self._vec_write(
            self.epi_buffer, self.epi_ptr, self.epi_count,
            e_query_np, self.n_episodic,
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
                    f"ngu/{group}/alpha": float(alpha_base.mean()),
                    f"ngu/{group}/r_int_raw": float(r_int_all.mean()),
                    f"ngu/{group}/r_int_norm": float(r_int_norm_all.mean()),
                },
                commit=False,
            )

        # Reshape: [n_flat, n_policies] -> [batch_size, n_agents, n_policies]
        r_t = torch.from_numpy(r_int_norm_all.astype(np.float32)).to(obs.device)
        n_agents = original_shape[1] if len(original_shape) > 2 else 1
        return r_t.reshape(original_shape[0], n_agents, n_pol)