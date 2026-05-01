"""
Module implementing R2D2 https://openreview.net/pdf?id=r1lyTjAqYX

Author: Kajetan Frąckowiak
Date: 2026

Variants:
- r2d2       : baseline recurrent DQN
- r2d2_rnd   : R2D2 + RND intrinsic reward
- r2d2_ngu   : R2D2 + NGU intrinsic reward
- r2d2_demir : R2D2 + DEMIR intrinsic reward

Description: This file contains the full implementation of the R2D2 algorithm
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper
from tensordict import TensorDict
from torch import nn

from benchmarl.algorithms.demir_module import DecentralizedEpisodicReward
from benchmarl.algorithms.ngu_module import NGUModule
from benchmarl.algorithms.rnd_module import RNDModule

VARIANT_TO_INTRINSIC = {
    "r2d2": "none",
    "r2d2_rnd": "rnd",
    "r2d2_ngu": "ngu",
    "r2d2_demir": "demir",
}

VARIANT_DEFAULT_SCALE = {
    "r2d2": 0.0,
    "r2d2_rnd": 0.1,
    "r2d2_ngu": 0.1,
    "r2d2_demir": 0.05,
}


@dataclass
class R2D2Config:
    variant: str
    env_id: str
    seed: int
    total_steps: int
    warmup_steps: int
    train_every: int
    grad_steps: int
    batch_size: int
    replay_capacity_sequences: int
    burn_in: int
    unroll_len: int
    n_step: int
    gamma: float
    lr: float
    adam_eps: float
    hidden_dim: int
    target_update_interval: int
    max_grad_norm: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    prio_alpha: float
    prio_beta_start: float
    prio_beta_end: float
    prio_eta: float
    intrinsic_scale: float
    log_interval: int
    eval_interval: int
    eval_episodes: int
    checkpoint_interval: int
    num_threads: int
    save_dir: str
    demir_beta1: float
    demir_beta2: float
    demir_encoder_type: str
    save_dir: str


class RecurrentDuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.val_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # obs_seq shape: [batch, time, obs_dim]
        z = self.encoder(obs_seq)
        out, next_state = self.lstm(z, state)
        adv = self.adv_head(out)
        val = self.val_head(out)
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        return q, next_state


class PrioritizedSequenceReplay:
    def __init__(
        self,
        capacity_sequences: int,
        burn_in: int,
        unroll_len: int,
        n_step: int,
        alpha: float,
    ):
        self.capacity_sequences = capacity_sequences
        self.burn_in = burn_in
        self.unroll_len = unroll_len
        self.n_step = n_step
        self.sequence_horizon = burn_in + unroll_len + n_step
        self.alpha = alpha

        self.episodes: Dict[int, Dict[str, np.ndarray]] = {}
        self.episode_order: deque[int] = deque()
        self.sequence_refs: List[Tuple[int, int]] = []
        self.priorities: List[float] = []
        self.next_episode_id = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return len(self.sequence_refs)

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        episode_len = actions.shape[0]
        if episode_len < self.sequence_horizon:
            return

        episode_id = self.next_episode_id
        self.next_episode_id += 1

        self.episodes[episode_id] = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }
        self.episode_order.append(episode_id)

        max_start = episode_len - self.sequence_horizon
        for start in range(max_start + 1):
            self.sequence_refs.append((episode_id, start))
            self.priorities.append(self.max_priority)

        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        while len(self.sequence_refs) > self.capacity_sequences and self.episode_order:
            oldest_episode = self.episode_order.popleft()
            if oldest_episode in self.episodes:
                del self.episodes[oldest_episode]

            keep_mask = [ref[0] != oldest_episode for ref in self.sequence_refs]
            self.sequence_refs = [
                ref for ref, keep in zip(self.sequence_refs, keep_mask) if keep
            ]
            self.priorities = [p for p, keep in zip(self.priorities, keep_mask) if keep]

        if self.priorities:
            self.max_priority = max(self.priorities)
        else:
            self.max_priority = 1.0

    def sample(self, batch_size: int, beta: float) -> Dict[str, np.ndarray]:
        if len(self.sequence_refs) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        priorities = np.asarray(self.priorities, dtype=np.float64)
        probs = np.power(priorities, self.alpha)
        probs /= probs.sum()

        n = len(self.sequence_refs)
        replace = n < batch_size
        sample_indices = np.random.choice(
            n, size=batch_size, replace=replace, p=probs
        ).astype(np.int64)

        weights = np.power(n * probs[sample_indices], -beta)
        weights /= weights.max() + 1e-8

        obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        for i in sample_indices:
            episode_id, start = self.sequence_refs[int(i)]
            episode = self.episodes[episode_id]
            end = start + self.sequence_horizon

            obs_batch.append(episode["obs"][start : end + 1])
            action_batch.append(episode["actions"][start:end])
            reward_batch.append(episode["rewards"][start:end])
            done_batch.append(episode["dones"][start:end])

        return {
            "obs": np.stack(obs_batch).astype(np.float32),
            "actions": np.stack(action_batch).astype(np.int64),
            "rewards": np.stack(reward_batch).astype(np.float32),
            "dones": np.stack(done_batch).astype(np.float32),
            "weights": weights.astype(np.float32),
            "sample_indices": sample_indices,
        }

    def update_priorities(
        self, sample_indices: np.ndarray, new_priorities: np.ndarray
    ) -> None:
        for idx, prio in zip(sample_indices.tolist(), new_priorities.tolist()):
            p = max(float(prio), 1e-6)
            self.priorities[int(idx)] = p
            if p > self.max_priority:
                self.max_priority = p


class IntrinsicRewardAdapter:
    def __init__(
        self,
        cfg: R2D2Config,
        intrinsic_kind: str,
        obs_dim: int,
        action_dim: int,
        gamma: float,
    ):
        self.kind = intrinsic_kind
        self.gamma = gamma

        if self.kind == "rnd":
            self.module = RNDModule(
                obs_dim=obs_dim,
                config={
                    "rnd_embed_dim": 64,
                    "rnd_hidden_dim": 256,
                    "rnd_lr": 1e-4,
                },
            )
        elif self.kind == "ngu":
            self.module = NGUModule(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config={
                    "ngu_embed_dim": 64,
                    "ngu_hidden_dim": 256,
                    "ngu_k": 10,
                    "ngu_L": 5.0,
                    "ngu_epsilon": 0.001,
                    "ngu_n_episodic": 10000,
                    "ngu_lr": 1e-4,
                },
            )
        elif self.kind == "demir":
            self.module = DecentralizedEpisodicReward(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config={
                    "emb_dim_state": 64,
                    "emb_dim_action": 16,
                    "emb_dim_reward": 8,
                    "alpha": 0.5,
                    "beta1": cfg.demir_beta1,
                    "beta2": cfg.demir_beta2,
                    "k": 10,
                    "sigma": 0.5,
                    "n_efm": 10000,
                    "n_edm": 5000,
                    "warmup": 100,
                    "encoder_type": cfg.demir_encoder_type,
                },
            )
        else:
            self.module = None

    def reset_episode(self) -> None:
        if self.kind == "ngu" and self.module is not None:
            self.module.reset_episodic_memory()

    def compute(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward_ext: float,
    ) -> float:
        if self.kind == "none" or self.module is None:
            return 0.0

        obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32).view(1, 1, -1)
        action_t = torch.tensor([[action]], dtype=torch.long)

        if self.kind == "rnd":
            r_int = self.module.compute_intrinsic_reward(
                obs=next_obs_t,
                group="agents",
                train=True,
            )
            return float(r_int.squeeze().item())

        if self.kind == "ngu":
            r_int = self.module.compute_intrinsic_reward(
                obs=obs_t,
                next_obs=next_obs_t,
                action=action_t,
                group="agents",
            )
            return float(r_int.squeeze().item())

        # DEMIR
        reward_t = torch.tensor([reward_ext], dtype=torch.float32)
        td_error_t = torch.zeros_like(reward_t)

        td = TensorDict(
            {
                ("agents", "observation"): obs_t,
                ("agents", "action"): action_t,
                ("next", "agents", "observation"): next_obs_t,
                ("next", "reward"): reward_t,
            },
            batch_size=[1],
        )

        with torch.no_grad():
            r_int = self.module.get_shaping_reward(
                td,
                group="agents",
                gamma=self.gamma,
            )
            self.module.update_memory(
                obs=obs_t,
                action=action_t,
                reward_ext=reward_t,
                td_error=td_error_t,
                next_obs=next_obs_t,
            )

        return float(r_int.squeeze().item())


def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id)
    env = ImgObsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return arr / 10.0


def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    frac = min(max(step / float(total_steps), 0.0), 1.0)
    return start + frac * (end - start)


def select_action(
    model: RecurrentDuelingQNetwork,
    obs: np.ndarray,
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
    epsilon: float,
    action_dim: int,
) -> Tuple[int, tuple[torch.Tensor, torch.Tensor]]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        q_values, next_hidden = model(obs_t, hidden)

    if random.random() < epsilon:
        action = random.randrange(action_dim)
    else:
        action = int(q_values[0, 0].argmax().item())
    return action, next_hidden


def train_step(
    cfg: R2D2Config,
    online_net: RecurrentDuelingQNetwork,
    target_net: RecurrentDuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedSequenceReplay,
    step: int,
) -> Tuple[float, np.ndarray]:
    beta = linear_schedule(
        cfg.prio_beta_start,
        cfg.prio_beta_end,
        step,
        cfg.total_steps,
    )
    batch = replay.sample(cfg.batch_size, beta)

    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32)
    weights = torch.as_tensor(batch["weights"], dtype=torch.float32).view(-1, 1)

    burn = cfg.burn_in
    unroll = cfg.unroll_len
    n_step = cfg.n_step

    if burn > 0:
        with torch.no_grad():
            _, h_online = online_net(obs[:, :burn, :], None)
            _, h_target = target_net(obs[:, :burn, :], None)
    else:
        h_online = None
        h_target = None

    q_online_long, _ = online_net(obs[:, burn : burn + n_step + unroll, :], h_online)
    q_roll = q_online_long[:, :unroll, :]
    q_future_online = q_online_long[:, n_step : n_step + unroll, :]

    with torch.no_grad():
        q_target_long, _ = target_net(
            obs[:, burn : burn + n_step + unroll, :], h_target
        )
        q_future_target = q_target_long[:, n_step : n_step + unroll, :]

    action_roll = actions[:, burn : burn + unroll]
    chosen_q = q_roll.gather(dim=-1, index=action_roll.unsqueeze(-1)).squeeze(-1)

    next_actions = q_future_online.argmax(dim=-1, keepdim=True)
    next_q = q_future_target.gather(dim=-1, index=next_actions).squeeze(-1)

    returns = torch.zeros_like(next_q)
    discounts = torch.ones_like(next_q)
    for k in range(n_step):
        r_k = rewards[:, burn + k : burn + k + unroll]
        d_k = dones[:, burn + k : burn + k + unroll]
        returns = returns + discounts * r_k
        discounts = discounts * cfg.gamma * (1.0 - d_k)

    targets = returns + discounts * next_q

    td_error = targets.detach() - chosen_q
    loss_elem = F.smooth_l1_loss(chosen_q, targets.detach(), reduction="none")
    loss = (weights * loss_elem).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), cfg.max_grad_norm)
    optimizer.step()

    td_abs = td_error.detach().abs()
    seq_priorities = cfg.prio_eta * td_abs.max(dim=1).values + (
        1.0 - cfg.prio_eta
    ) * td_abs.mean(dim=1)
    seq_priorities = (seq_priorities + 1e-6).cpu().numpy()
    replay.update_priorities(batch["sample_indices"], seq_priorities)

    return float(loss.item()), seq_priorities


def evaluate_policy(
    env_id: str,
    model: RecurrentDuelingQNetwork,
    seed: int,
    episodes: int,
) -> float:
    returns = []

    for ep in range(episodes):
        env = make_env(env_id, seed=seed + 10000 + ep)
        obs_raw, _ = env.reset()
        obs = preprocess_obs(obs_raw)
        done = False
        hidden = None
        ep_return = 0.0

        while not done:
            action, hidden = select_action(
                model=model,
                obs=obs,
                hidden=hidden,
                epsilon=0.0,
                action_dim=env.action_space.n,
            )
            next_obs_raw, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            obs = preprocess_obs(next_obs_raw)
            ep_return += float(reward)

        returns.append(ep_return)
        env.close()

    return float(np.mean(returns))


def build_run_name(cfg: R2D2Config) -> str:
    timestamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    env_name = cfg.env_id.replace("-", "_")
    return f"{cfg.variant}_{env_name}_seed{cfg.seed}_{timestamp}"


def parse_args() -> R2D2Config:
    parser = argparse.ArgumentParser(description="R2D2 MiniGrid trainer")
    parser.add_argument(
        "--variant",
        type=str,
        default="r2d2",
        choices=sorted(VARIANT_TO_INTRINSIC.keys()),
    )
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--grad-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity-sequences", type=int, default=40_000)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--unroll-len", type=int, default=40)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    parser.add_argument("--prio-alpha", type=float, default=0.6)
    parser.add_argument("--prio-beta-start", type=float, default=0.4)
    parser.add_argument("--prio-beta-end", type=float, default=1.0)
    parser.add_argument("--prio-eta", type=float, default=0.9)
    parser.add_argument("--intrinsic-scale", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="logs_thesis/minigrid")
    parser.add_argument("--demir-beta1", type=float, default=0.7)
    parser.add_argument("--demir-beta2", type=float, default=0.3)
    parser.add_argument("--demir-encoder-type", type=str, default="idm")

    args = parser.parse_args()

    intrinsic_scale = (
        VARIANT_DEFAULT_SCALE[args.variant]
        if args.intrinsic_scale is None
        else float(args.intrinsic_scale)
    )

    return R2D2Config(
        variant=args.variant,
        env_id=args.env_id,
        seed=args.seed,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        train_every=args.train_every,
        grad_steps=args.grad_steps,
        batch_size=args.batch_size,
        replay_capacity_sequences=args.replay_capacity_sequences,
        burn_in=args.burn_in,
        unroll_len=args.unroll_len,
        n_step=args.n_step,
        gamma=args.gamma,
        lr=args.lr,
        adam_eps=args.adam_eps,
        hidden_dim=args.hidden_dim,
        target_update_interval=args.target_update_interval,
        max_grad_norm=args.max_grad_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        prio_alpha=args.prio_alpha,
        prio_beta_start=args.prio_beta_start,
        prio_beta_end=args.prio_beta_end,
        prio_eta=args.prio_eta,
        intrinsic_scale=intrinsic_scale,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        num_threads=args.num_threads,
        save_dir=args.save_dir,
        demir_beta1=args.demir_beta1,
        demir_beta2=args.demir_beta2,
        demir_encoder_type=args.demir_encoder_type,
    )


def main() -> None:
    cfg = parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)

    run_name = build_run_name(cfg)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / f"{run_name}.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "step",
            "episodes",
            "mean_ext_return_100",
            "mean_int_return_100",
            "mean_total_return_100",
            "epsilon",
            "replay_sequences",
            "loss",
            "eval_ext_return",
        ],
    )
    csv_writer.writeheader()

    env = make_env(cfg.env_id, seed=cfg.seed)
    obs_raw, _ = env.reset(seed=cfg.seed)
    obs = preprocess_obs(obs_raw)
    obs_dim = obs.shape[0]
    action_dim = int(env.action_space.n)

    online_net = RecurrentDuelingQNetwork(obs_dim, action_dim, cfg.hidden_dim)
    target_net = RecurrentDuelingQNetwork(obs_dim, action_dim, cfg.hidden_dim)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    replay = PrioritizedSequenceReplay(
        capacity_sequences=cfg.replay_capacity_sequences,
        burn_in=cfg.burn_in,
        unroll_len=cfg.unroll_len,
        n_step=cfg.n_step,
        alpha=cfg.prio_alpha,
    )

    intrinsic_kind = VARIANT_TO_INTRINSIC[cfg.variant]
    intrinsic = IntrinsicRewardAdapter(
        cfg=cfg,
        intrinsic_kind=intrinsic_kind,
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=cfg.gamma,
    )

    episode_obs: List[np.ndarray] = [obs.copy()]
    episode_actions: List[int] = []
    episode_rewards: List[float] = []
    episode_dones: List[float] = []

    episode_ext_return = 0.0
    episode_int_return = 0.0
    episode_total_return = 0.0
    episodes_finished = 0

    ext_return_window: deque[float] = deque(maxlen=100)
    int_return_window: deque[float] = deque(maxlen=100)
    total_return_window: deque[float] = deque(maxlen=100)

    actor_hidden = None
    train_updates = 0
    last_loss = float("nan")
    last_eval_return = float("nan")

    print("=" * 72)
    print(f"Run: {run_name}")
    print(
        f"Variant: {cfg.variant} | Intrinsic kind: {intrinsic_kind} | Scale: {cfg.intrinsic_scale}"
    )
    print(f"Env: {cfg.env_id} | Device: CPU | Threads: {torch.get_num_threads()}")
    print("=" * 72)

    for step in range(1, cfg.total_steps + 1):
        epsilon = linear_schedule(
            cfg.epsilon_start,
            cfg.epsilon_end,
            step,
            cfg.epsilon_decay_steps,
        )

        action, next_hidden = select_action(
            model=online_net,
            obs=obs,
            hidden=actor_hidden,
            epsilon=epsilon,
            action_dim=action_dim,
        )

        next_obs_raw, reward_ext, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        next_obs = preprocess_obs(next_obs_raw)

        reward_int = intrinsic.compute(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward_ext=float(reward_ext),
        )
        reward_total = float(reward_ext) + cfg.intrinsic_scale * reward_int

        episode_actions.append(int(action))
        episode_rewards.append(reward_total)
        episode_dones.append(float(done))
        episode_obs.append(next_obs.copy())

        episode_ext_return += float(reward_ext)
        episode_int_return += float(reward_int)
        episode_total_return += float(reward_total)

        obs = next_obs
        actor_hidden = next_hidden

        if done:
            replay.add_episode(
                obs=np.stack(episode_obs).astype(np.float32),
                actions=np.asarray(episode_actions, dtype=np.int64),
                rewards=np.asarray(episode_rewards, dtype=np.float32),
                dones=np.asarray(episode_dones, dtype=np.float32),
            )

            ext_return_window.append(episode_ext_return)
            int_return_window.append(episode_int_return)
            total_return_window.append(episode_total_return)
            episodes_finished += 1

            intrinsic.reset_episode()

            obs_raw, _ = env.reset()
            obs = preprocess_obs(obs_raw)
            actor_hidden = None

            episode_obs = [obs.copy()]
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_ext_return = 0.0
            episode_int_return = 0.0
            episode_total_return = 0.0

        if (
            step >= cfg.warmup_steps
            and len(replay) >= cfg.batch_size
            and step % cfg.train_every == 0
        ):
            for _ in range(cfg.grad_steps):
                last_loss, _ = train_step(
                    cfg=cfg,
                    online_net=online_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    replay=replay,
                    step=step,
                )
                train_updates += 1

                if train_updates % cfg.target_update_interval == 0:
                    target_net.load_state_dict(online_net.state_dict())

        if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
            last_eval_return = evaluate_policy(
                env_id=cfg.env_id,
                model=online_net,
                seed=cfg.seed,
                episodes=cfg.eval_episodes,
            )
            print(
                f"[eval] step={step} episodes={episodes_finished} "
                f"mean_ext_return={last_eval_return:.3f}"
            )

        if cfg.log_interval > 0 and step % cfg.log_interval == 0:
            mean_ext = (
                float(np.mean(ext_return_window)) if ext_return_window else float("nan")
            )
            mean_int = (
                float(np.mean(int_return_window)) if int_return_window else float("nan")
            )
            mean_total = (
                float(np.mean(total_return_window))
                if total_return_window
                else float("nan")
            )

            print(
                f"[train] step={step:>8} episodes={episodes_finished:>5} "
                f"eps={epsilon:.3f} replay={len(replay):>6} "
                f"loss={last_loss:.5f} ext100={mean_ext:.3f} total100={mean_total:.3f}"
            )

            csv_writer.writerow(
                {
                    "step": step,
                    "episodes": episodes_finished,
                    "mean_ext_return_100": mean_ext,
                    "mean_int_return_100": mean_int,
                    "mean_total_return_100": mean_total,
                    "epsilon": epsilon,
                    "replay_sequences": len(replay),
                    "loss": last_loss,
                    "eval_ext_return": last_eval_return,
                }
            )
            csv_file.flush()

        if cfg.checkpoint_interval > 0 and step % cfg.checkpoint_interval == 0:
            checkpoint_path = ckpt_dir / f"{run_name}_step{step}.pt"
            torch.save(
                {
                    "step": step,
                    "config": cfg,
                    "online_net": online_net.state_dict(),
                    "target_net": target_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    final_ckpt = ckpt_dir / f"{run_name}_final.pt"
    torch.save(
        {
            "step": cfg.total_steps,
            "config": cfg,
            "online_net": online_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        final_ckpt,
    )

    env.close()
    csv_file.close()

    print("=" * 72)
    print(f"Training finished: {run_name}")
    print(f"CSV log: {csv_path}")
    print(f"Final checkpoint: {final_ckpt}")
    print("=" * 72)


if __name__ == "__main__":
    main()
