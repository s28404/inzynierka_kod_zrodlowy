
"""
Moduł implementujący mechanizm NGU (Never Give Up) do MARL bazująć na:
https://arxiv.org/abs/2002.06038 oraz https://arxiv.org/abs/2512.01321.

Autor: Kajetan Frąckowiak (s28404)
Data: 2026
Praca inżynierska: Polsko-Japońska Akademia Technik Komputerowych

Opis: Plik zawiera pełną implementację mechanizmu NGU.
"""
import numpy as np
import torch
import faiss
from torch import nn
from benchmarl.algorithms.common import RunningMeanStd

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
        # Inverse Dynamics Model: (e_s, e_s') -> action
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
            list(self.phi.parameters()) +
            list(self.idm.parameters()) +
            list(self.rnd_predictor.parameters()),
            lr=lr,
        )

        # --- Episodic buffer (rolling, FAISS-backed) ---
        self.epi_buffer = np.zeros((self.n_episodic, d), dtype=np.float32)
        self.epi_ptr = 0
        self.epi_count = 0
        self.faiss_index = faiss.IndexFlatL2(d)
        self._update_counter = 0

        # --- Normalization ---
        self.rnd_rms = RunningMeanStd()
        self.ep_rms  = RunningMeanStd()

    def reset_episodic_memory(self):
        """
        Resets the episodic buffer and FAISS index at the start of a new episode.
        This is CRITICAL for NGU to function as intended.
        """
        d = self._p("ngu_embed_dim", 64)
        self.epi_buffer = np.zeros((self.n_episodic, d), dtype=np.float32)
        self.epi_ptr = 0
        self.epi_count = 0
        self.faiss_index = faiss.IndexFlatL2(d)
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
            buf[:n - split] = data[split:]
        return end % capacity, min(count + n, capacity)

    def compute_intrinsic_reward(self, obs: torch.Tensor, next_obs: torch.Tensor,
                                  action: torch.Tensor, group: str) -> torch.Tensor:
        """
        Compute NGU intrinsic reward.

        Args:
            obs      : (B, n_agents, obs_dim)
            next_obs : (B, n_agents, obs_dim)
            action   : (B, n_agents, ...) - discrete or continuous
            group    : agent group name

        Returns:
            r_int : (B, n_agents, 1)
        """
        original_shape = obs.shape
        obs_flat      = obs.reshape(-1, obs.shape[-1])
        next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1])
        n = obs_flat.shape[0]

        # Action flat
        if action.dtype in [torch.long, torch.int]:
            a_flat = nn.functional.one_hot(
                action.reshape(-1), num_classes=self.action_dim
            ).float()
        else:
            a_flat = action.reshape(-1, action.shape[-1])

        # --- Train IDM + RND predictor ---
        self.optimizer.zero_grad()

        e_s      = self.phi(obs_flat)
        e_s_next = self.phi(next_obs_flat)

        # IDM loss
        pred_action = self.idm(torch.cat([e_s, e_s_next], dim=-1))
        idm_loss = nn.functional.mse_loss(pred_action, a_flat)

        # RND loss
        with torch.no_grad():
            rnd_target_feat = self.rnd_target(obs_flat)
        rnd_pred_feat = self.rnd_predictor(obs_flat)
        rnd_loss = nn.functional.mse_loss(rnd_pred_feat, rnd_target_feat)

        (idm_loss + rnd_loss).backward()
        self.optimizer.step()

        with torch.no_grad():
            e_s_np = e_s.detach().cpu().numpy().astype(np.float32)
            rnd_r_np = ((rnd_pred_feat.detach() - rnd_target_feat) ** 2).mean(dim=-1).cpu().numpy()

        # --- Update episodic buffer & FAISS ---
        old_ptr = self.epi_ptr
        self.epi_ptr, self.epi_count = self._vec_write(
            self.epi_buffer, self.epi_ptr, self.epi_count, e_s_np, self.n_episodic)

        self._update_counter += 1
        if self._update_counter % self.rebuild_interval == 0:
            filled = min(self.epi_count, self.n_episodic)
            self.faiss_index = faiss.IndexFlatL2(self.epi_buffer.shape[1])
            self.faiss_index.add(self.epi_buffer[:filled])

        # --- Episodic reward (k-NN distances in embedding space) ---
        if self.faiss_index.ntotal >= self.k:
            q = e_s_np
            distances, _ = self.faiss_index.search(q, self.k)
            # r_ep = 1 / (sqrt(mean_dist) + epsilon)   [NGU eq. 1]
            r_ep = 1.0 / (np.sqrt(np.mean(distances, axis=1)) + self.eps)
        else:
            r_ep = np.ones(n, dtype=np.float32)

        # --- Lifelong curiosity (RND-based alpha) ---
        self.rnd_rms.update(rnd_r_np)
        rnd_norm = self.rnd_rms.normalize(rnd_r_np)
        alpha = np.clip(1.0 + rnd_norm / self.L, 1.0, self.L)

        # --- Combined intrinsic reward ---
        r_int = r_ep * alpha

        # Normalize episodic before combining (optional stability)
        self.ep_rms.update(r_int)
        r_int_norm = self.ep_rms.normalize(r_int)

        if _wandb is not None and _wandb.run is not None:
            _wandb.log({
                f"ngu/{group}/r_episodic": float(r_ep.mean()),
                f"ngu/{group}/alpha":      float(alpha.mean()),
                f"ngu/{group}/r_int":      float(r_int.mean()),
            }, commit=False)

        r_t = torch.from_numpy(r_int_norm.astype(np.float32)).to(obs.device)
        return r_t.reshape(*original_shape[:-1], 1)
