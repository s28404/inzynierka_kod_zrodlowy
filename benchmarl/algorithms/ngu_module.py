# Autor: Kajetan Frąckowiak, s28404 (2026)
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Moduł NGU (Never Give Up) — epizodyczna i długoterminowa ciekawość.
#        Embedding trenowany przez IDM; lifelong curiosity przez RND;
#        bufor epizodyczny FAISS; kombinowana nagroda wewnętrzna r_int = r_e * alpha.
#
# Sekcje wewnętrzne (inline section markers):
#
#   [1] # --- Embedding network (trained via IDM) ---
#       Lokalizacja: __init__()  —  phi_s: MLP, idm: sieć (e_s, e_s') → action.
#
#   [2] # --- RND for lifelong curiosity ---
#       Lokalizacja: __init__()  —  rnd_target (fixed) + rnd_pred (trainable).
#
#   [3] # --- Episodic buffer (rolling, FAISS-backed) ---
#       Lokalizacja: __init__()  —  episodyczny bufor k-NN z FAISS.
#
#   [4] # --- Normalization ---
#       Lokalizacja: __init__()  —  RunningMeanStd do normalizacji nagród.
#
#   [5] # --- Train IDM + RND predictor ---
#       Lokalizacja: compute_intrinsic_reward()  —  backward pass IDM + RND.
#
#   [6] # --- Update episodic buffer & FAISS ---
#       Lokalizacja: compute_intrinsic_reward()  —  zapis embeddingów do bufora.
#
#   [7] # --- Episodic reward (k-NN distances in embedding space) ---
#       Lokalizacja: compute_intrinsic_reward()  —  r_e = f(dist k-NN).
#
#   [8] # --- Lifelong curiosity (RND-based alpha) ---
#       Lokalizacja: compute_intrinsic_reward()  —  alpha = 1 + błąd RND.
#
#   [9] # --- Combined intrinsic reward ---
#       Lokalizacja: compute_intrinsic_reward()  —  r_int = r_e * alpha.

"""
Never Give Up (NGU) - per-agent decentralized intrinsic reward for MARL.
Based on: Badia et al. (2020) "Never Give Up: Learning Directed Exploration Strategies"
https://arxiv.org/abs/2002.06038

Intrinsic reward:
    r_int = r_episodic * min(max(alpha, 1), L)
    alpha  = 1 + r_RND_normalized / L
    r_episodic = 1 / (sqrt(mean k-NN distances in embedding space) + epsilon)

Embedding trained via Inverse Dynamics Model (IDM): (e_s, e_s') -> action.
Applied decentralized: each agent group gets its own NGU module.
"""
import numpy as np
import torch
import faiss
from torch import nn

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


class RunningMeanStd:
    """Welford's online normalization."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        b_mean = float(np.mean(x))
        b_var  = float(np.var(x))
        b_count = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        self.var = (self.var * self.count + b_var * b_count +
                    delta ** 2 * self.count * b_count / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


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
        self.faiss_index = faiss.IndexHNSWFlat(d, 32)
        self._update_counter = 0

        # --- Normalization ---
        self.rnd_rms = RunningMeanStd()
        self.ep_rms  = RunningMeanStd()

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
            self.faiss_index = faiss.IndexHNSWFlat(self.epi_buffer.shape[1], 32)
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
        # Intra-batch std normalization
        r_int_norm = r_int_norm / (float(np.std(r_int_norm)) + 1e-8)

        if _wandb is not None and _wandb.run is not None:
            _wandb.log({
                f"ngu/{group}/r_episodic": float(r_ep.mean()),
                f"ngu/{group}/alpha":      float(alpha.mean()),
                f"ngu/{group}/r_int":      float(r_int.mean()),
            }, commit=False)

        r_t = torch.from_numpy(r_int_norm.astype(np.float32)).to(obs.device)
        return r_t.reshape(*original_shape[:-1], 1)
