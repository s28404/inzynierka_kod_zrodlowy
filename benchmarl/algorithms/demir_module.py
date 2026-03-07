# Autor: Kajetan Frąckowiak, s28404 (2026)
# Plik napisany od podstaw w ramach pracy inżynierskiej
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Moduł Decentralized Episodic Memory with Intrinsic Reward (DEMIR).
#        Dwukierunkowa pamięć epizodyczna (EFM + EDM) z enkoderami IDM lub MLP,
#        indeksowaniem FAISS HNSW i normalizacją Welforda (RunningMeanStd).

import torch
import numpy as np
import faiss
from torch import nn
from types import SimpleNamespace
try:
    import wandb as _wandb
except ImportError:
    _wandb = None


class RunningMeanStd:
    """Welford's online algorithm - normalizacja jakości i nowości w czasie treningu."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        b_mean = float(np.mean(x))
        b_var = float(np.var(x))
        b_count = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        self.var = (self.var * self.count + b_var * b_count +
                    delta ** 2 * self.count * b_count / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


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
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.d_s)
        )
        self.phi_a = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.d_a)
        )
        self.phi_r = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, self.d_r)
        )
        # Auxiliary: Inverse Dynamics Model (e_s_t, e_s_{t+1}) -> akcja
        # Gwarantuje "actionable" embeddingi - ignoruje szum nie wpływający na agenta
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.d_s * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward_kinematics(self, e_s, e_s_next):
        """Przewiduje akcję z pary stanów (Inverse Dynamics)."""
        return self.inverse_dynamics(torch.cat([e_s, e_s_next], dim=-1))

    def encoder_state(self, obs):
        return self.phi_s(obs)
    
    def encode_full_experience(self, obs, action, reward_ext):
        e_s = self.phi_s(obs)
        e_a = self.phi_a(action)
        
        # Reward musi być (batch, 1)
        if reward_ext.dim() == 0 or reward_ext.dim() == 1:
            reward_ext = reward_ext.view(-1, 1)
            
        e_r = self.phi_r(reward_ext)
        e_F = torch.cat([e_s, e_a, e_r], dim=-1)
        return e_s, e_F

class DecentralizedEpisodicReward(nn.Module):
    def __init__(self, obs_dim, action_dim, config=None, **kwargs):
        super().__init__()
        self.config = config 

        # 1. Parametry z bezpiecznikami
        self.k = self._get_config_param("k", 10)
        self.n_efm = self._get_config_param("n_efm", 10000)
        self.n_edm = self._get_config_param("n_edm", 5000)
        self.emb_dim_state = self._get_config_param("emb_dim_state", 64)
        self.emb_dim_action = self._get_config_param("emb_dim_action", 16)
        self.emb_dim_reward = self._get_config_param("emb_dim_reward", 8)

        # 2. Inicjalizacja Enkoderów
        self.encoders = DemirEncoders(obs_dim, action_dim, self.config)

        # 3. Inicjalizacja FAISS
        d_total = self.emb_dim_state + self.emb_dim_action + self.emb_dim_reward
        self.efm_index = faiss.IndexHNSWFlat(d_total, 32)
        self.edm_index = faiss.IndexHNSWFlat(self.emb_dim_state, 32)

        # 4. Rolling buffers (ograniczony rozmiar - zapobiega novelty=0 w on-policy)
        self.efm_buffer = np.zeros((self.n_efm, d_total), dtype=np.float32)
        self.edm_buffer = np.zeros((self.n_edm, self.emb_dim_state), dtype=np.float32)
        self.efm_rhos = np.zeros(self.n_efm, dtype=np.float32)
        self.efm_ptr = 0
        self.edm_ptr = 0
        self.efm_count = 0
        self.edm_count = 0

        # 5. Typ enkodera: "idm" (Inverse Dynamics) lub "mlp" (tylko dekorelacja)
        self.encoder_type = self._get_config_param("encoder_type", "idm")

        # 6. Optymalizator enkoderów (auxiliary reward prediction loss)
        self.encoder_opt = torch.optim.Adam(self.encoders.parameters(), lr=1e-4)

        # 7. Lazy FAISS rebuild - przebudowuj co N kroków, nie co krok
        self.rebuild_interval = self._get_config_param("rebuild_interval", 100)
        self._update_counter = 0

        # 7. Running normalization jakości i nowości (stabilizuje beta1/beta2)
        self.quality_rms = RunningMeanStd()
        self.novelty_rms = RunningMeanStd()

    def _get_config_param(self, name, default):
        if self.config is None: return default
        return getattr(self.config, name, getattr(self.config, "get", lambda x, y: y)(name, default))

    def _get_k_neighbors_batch(self, index, query_embeddings, k):
        if index.ntotal < k:
            return None, None
        x = query_embeddings.detach().cpu().numpy().astype('float32')
        if x.ndim == 1: x = x.reshape(1, -1)
        distances, indices = index.search(x, k)
        return distances, indices

    def _compute_potential(self, e_s, e_F):
        """Pomocnicza funkcja do obliczania surowego potencjału Phi z embeddingów."""
        # 1. Szukanie w pamięci
        dist_q, idx_q = self._get_k_neighbors_batch(self.efm_index, e_F, self.k)
        dist_n, _ = self._get_k_neighbors_batch(self.edm_index, e_s, self.k)

        # 2. QUALITY (EFM)
        sigma = getattr(self.config, "sigma", 0.1)
        if idx_q is not None:
            mask = (idx_q >= 0)
            weights = np.exp(-dist_q**2 / (2 * sigma**2)) * mask
            rhos = self.efm_rhos[idx_q % self.n_efm]
            quality = np.sum(weights * rhos, axis=1) / (np.sum(weights, axis=1) + 1e-6)
        else:
            quality = np.zeros(e_s.shape[0])

        # 3. NOVELTY (EDM)
        novelty = np.mean(dist_n, axis=1) if dist_n is not None else np.zeros(e_s.shape[0])
        
        return quality, novelty

    def get_shaping_reward(self, batch, group, gamma=0.99):
        # 1. Pobieranie danych
        obs = batch.get((group, "observation"))
        next_obs = batch.get(("next", group, "observation"))
        action = batch.get((group, "action"))
        reward_ext = batch.get(("next", group, "reward"))

        if self.efm_index.ntotal < self.k:
            return torch.zeros_like(reward_ext)

        # 2. Przygotowanie danych (spłaszczanie i akcje)
        obs_flat = obs.reshape(-1, obs.shape[-1])
        next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1])
        reward_flat = reward_ext.reshape(-1, 1)
        
        if action.dtype in [torch.long, torch.int]:
            a_dim = self.encoders.phi_a[0].in_features
            action_flat = torch.nn.functional.one_hot(action, num_classes=a_dim).float().reshape(-1, a_dim)
        else:
            action_flat = action.reshape(-1, action.shape[-1])

        # 3. Obliczanie embeddingów dla OBU stanów
        # Dla stanu obecnego (t)
        e_s_t, e_F_t = self.encoders.encode_full_experience(obs_flat, action_flat, reward_flat)
        # Dla stanu następnego (t+1) - uwaga: tutaj akcja/reward są estymowane lub uproszczone
        e_s_tp1, e_F_tp1 = self.encoders.encode_full_experience(next_obs_flat, action_flat, reward_flat)

        # 4. Obliczanie surowych składowych potencjału
        q_t, n_t = self._compute_potential(e_s_t, e_F_t)
        q_tp1, n_tp1 = self._compute_potential(e_s_tp1, e_F_tp1)

        # 5. Normalizacja Welforda (tylko na podstawie stanów t+1, by nie dublować statystyk)
        self.quality_rms.update(q_tp1)
        self.novelty_rms.update(n_tp1)
        
        def _norm_full(q, n):
            qn = self.quality_rms.normalize(q)
            nn = self.novelty_rms.normalize(n)
            # Intra-batch std correction
            qn = qn / (float(np.std(qn)) + 1e-8)
            nn = nn / (float(np.std(nn)) + 1e-8)
            return qn, nn

        q_t_n, n_t_n = _norm_full(q_t, n_t)
        q_tp1_n, n_tp1_n = _norm_full(q_tp1, n_tp1)

        # 6. Finalny Potencjał Phi (Ng 1999)
        beta1 = getattr(self.config, "beta1", 1.0)
        beta2 = getattr(self.config, "beta2", 0.5)

        phi_t = beta1 * q_t_n + beta2 * n_t_n
        phi_tp1 = beta1 * q_tp1_n + beta2 * n_tp1_n

        # Konwersja na tensory
        phi_t = torch.from_numpy(phi_t.astype(np.float32)).to(obs.device)
        phi_tp1 = torch.from_numpy(phi_tp1.astype(np.float32)).to(obs.device)

        # 7. SHAPING REWARD: r_int = gamma * Phi(s_t+1) - Phi(s_t)
        r_int = (gamma * phi_tp1) - phi_t

        # Logowanie (opcjonalne)
        if _wandb is not None and _wandb.run is not None:
            _wandb.log({f"demir/{group}/phi_t": phi_t.mean().item(), 
                        f"demir/{group}/r_int": r_int.mean().item()}, commit=False)

        return r_int.view_as(reward_ext)

    def update_memory(self, obs, action, reward_ext, td_error, next_obs=None):
        obs_flat = obs.reshape(-1, obs.shape[-1])
        n_flat = obs_flat.shape[0]

        reward_flat = reward_ext.reshape(-1, 1)
        if reward_flat.shape[0] != n_flat:
            # Nagroda globalna (QMIX) - rozszerzamy na liczbę agentów
            repeats = n_flat // reward_flat.shape[0]
            reward_flat = reward_flat.repeat_interleave(repeats, dim=0)

        td_error_flat = td_error.reshape(-1, 1)
        if td_error_flat.shape[0] != n_flat:
            repeats = n_flat // td_error_flat.shape[0]
            td_error_flat = td_error_flat.repeat_interleave(repeats, dim=0)

        # Obsługa akcji
        if action.dtype in [torch.long, torch.int]:
            a_dim = self.encoders.phi_a[0].in_features
            action_flat = torch.nn.functional.one_hot(action, num_classes=a_dim).float().reshape(-1, a_dim)
        else:
            action_flat = action.reshape(-1, action.shape[-1])

        # Forward pass + auxiliary encoder loss
        # encoder_type="idm": IDM + Decorrelation ("actionable" embeddingi)
        # encoder_type="mlp": tylko Decorrelation (ablacja - brak IDM)
        self.encoder_opt.zero_grad()
        e_s = self.encoders.phi_s(obs_flat)
        e_s_norm = (e_s - e_s.mean(dim=0)) / (e_s.std(dim=0) + 1e-6)


        if self.encoder_type == "idm" and next_obs is not None:
            next_obs_flat = next_obs.reshape(-1, next_obs.shape[-1])
            e_s_next = self.encoders.phi_s(next_obs_flat)
            # A. Inverse Dynamics Loss: (e_s, e_s_next) -> akcja
            pred_action = self.encoders.forward_kinematics(e_s, e_s_next)
            inv_loss = nn.functional.mse_loss(pred_action, action_flat)
            # B. Decorrelation (Redundancy Reduction) - zapobiega kolapsowi wymiarów
            # Normalizacja wzdłuż batcha
            c = e_s_norm.T @ e_s_norm / n_flat
            diag_vals = torch.diag(c)
            red_loss = (c - torch.diag(diag_vals)).pow(2).mean()
            encoder_loss = inv_loss + 0.01 * red_loss
            encoder_loss.backward()
            self.encoder_opt.step()
        elif self.encoder_type == "mlp":
            # Ablacja IDM: tylko dekorelacja, bez constraintu "actionable"
            c = e_s_norm.T @ e_s_norm / n_flat
            diag_vals = torch.diag(c)
            red_loss = (c - torch.diag(diag_vals)).pow(2).mean()
            (0.01 * red_loss).backward()
            self.encoder_opt.step()

        # Re-encode bez gradienta do FAISS
        with torch.no_grad():
            e_s, e_F = self.encoders.encode_full_experience(obs_flat, action_flat, reward_flat)

        # Obliczanie Rho
        alpha = getattr(self.config, "alpha", 0.5)
        rho = alpha * torch.abs(reward_flat) + (1 - alpha) * torch.abs(td_error_flat)
        
        # Konwersja do numpy dla FAISS
        rho_np = rho.detach().cpu().numpy().flatten()
        efm_np = e_F.detach().cpu().numpy().astype('float32')
        edm_np = e_s.detach().cpu().numpy().astype('float32')

        # Rolling buffer - wektoryzowane wpisanie (szybkie dla dużych batchy on-policy)
        def _vec_write(buf, ptr, count, data, capacity):
            """Wpisuje data do buffera cyklicznego. Zwraca (nowy_ptr, nowy_count)."""
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

        # 1. Zapis do EFM (Full Experience Memory) - tego brakowało w Twoim fragmencie
        efm_start_ptr = self.efm_ptr
        self.efm_ptr, self.efm_count = _vec_write(
            self.efm_buffer, self.efm_ptr, self.efm_count, efm_np, self.n_efm)
        
        # 2. Synchronizacja priorytetów rho z buforem EFM
        # Używamy efm_start_ptr, aby rho trafiło dokładnie tam, gdzie wektor e_F
        _vec_write(self.efm_rhos.reshape(-1, 1), efm_start_ptr, 0,
                   rho_np.reshape(-1, 1), self.n_efm)

        self.edm_ptr, self.edm_count = _vec_write(
            self.edm_buffer, self.edm_ptr, self.edm_count, edm_np, self.n_edm)

        # 2. POPRAWKA: Dynamiczny i inteligentny Rebuild FAISS
        self._update_counter += 1
        
        # WARUNKI AKTUALIZACJI:
        # A. Indeks jest pusty (ntotal == 0) -> musimy dodać dane natychmiast
        # B. Licznik przekroczył rebuild_interval -> optymalizujemy strukturę HNSW
        if self.efm_index.ntotal == 0 or self._update_counter % self.rebuild_interval == 0:
            
            # Pobieramy faktycznie zapisaną liczbę próbek (nie więcej niż pojemność)
            filled_efm = min(self.efm_count, self.n_efm)
            filled_edm = min(self.edm_count, self.n_edm)

            if filled_efm > 0:
                # Tworzymy nowy, czysty indeks HNSW
                # 32 to parametr M (liczba połączeń w grafie) - standard dla HNSW
                new_efm = faiss.IndexHNSWFlat(self.efm_buffer.shape[1], 32)
                new_efm.add(self.efm_buffer[:filled_efm])
                self.efm_index = new_efm

            if filled_edm > 0:
                new_edm = faiss.IndexHNSWFlat(self.edm_buffer.shape[1], 32)
                new_edm.add(self.edm_buffer[:filled_edm])
                self.edm_index = new_edm