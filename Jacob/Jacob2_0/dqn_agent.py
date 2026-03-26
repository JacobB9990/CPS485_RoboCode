"""DQN agent: GRU encoder + MLP Q-head, Double DQN, experience replay.

Designed to sit inside the Tank Royale bot process — no sidecar needed.
The bot holds one DQNAgent instance for its entire lifetime and calls:
  agent.on_episode_start()       — resets GRU hidden state
  agent.select_action(state)     — epsilon-greedy inference
  agent.push_and_train(...)      — stores transition, runs mini-batch
  agent.on_episode_end(won)      — decays epsilon, saves weights
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _DQNNet(nn.Module):
    def __init__(
        self, state_dim: int, gru_hidden: int, mlp_hidden: int, n_actions: int
    ):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(state_dim, gru_hidden, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, n_actions),
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, state_dim)
        out, h_new = self.gru(x, h)  # out: (B, 1, gru_hidden)
        q = self.mlp(out[:, -1, :])  # (B, n_actions)
        return q, h_new

    def init_hidden(
        self, batch: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        return torch.zeros(1, batch, self.gru_hidden, device=device)


@dataclass
class _Transition:
    state: np.ndarray  # (state_dim,)
    action: int
    reward: float
    next_state: np.ndarray  # (state_dim,)
    done: bool
    hidden: np.ndarray  # (1, 1, gru_hidden) — snapshot at decision time


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[_Transition] = []
        self.pos = 0

    def push(self, *args) -> None:
        t = _Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int) -> List[_Transition]:
        return random.sample(self.buffer, n)

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """Full DQN agent — call from your Bot subclass."""

    def __init__(
        self,
        state_dim: int = 16,
        n_actions: int = 7,
        gru_hidden: int = 64,
        mlp_hidden: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        min_buffer: int = 1_000,
        target_update_freq: int = 500,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        weights_path: str = "weights_dqn.pt",
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.weights_path = weights_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = _DQNNet(state_dim, gru_hidden, mlp_hidden, n_actions).to(
            self.device
        )
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = _ReplayBuffer(buffer_size)

        # Per-episode inference state
        self.hidden: torch.Tensor = self.online.init_hidden(device=self.device)

        self.steps = 0
        self.battles = 0
        self.wins = 0
        self.q_update_abs_sum = 0.0
        self.q_update_count = 0

        self._load()

    def on_episode_start(self) -> None:
        self.hidden = self.online.init_hidden(device=self.device)
        self.q_update_abs_sum = 0.0
        self.q_update_count = 0

    def select_action(self, state: list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        self.online.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals, self.hidden = self.online(s, self.hidden)
        return int(q_vals.argmax().item())

    def push_and_train(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        h_np = self.hidden.cpu().numpy()
        self.buffer.push(
            np.array(state, dtype=np.float32),
            action,
            float(reward),
            np.array(next_state, dtype=np.float32),
            done,
            h_np,
        )
        self.steps += 1

        if len(self.buffer) >= self.min_buffer:
            self._train_step()

    def on_episode_end(self, won: bool) -> dict:
        self.battles += 1
        if won:
            self.wins += 1

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        avg_abs_td = (
            self.q_update_abs_sum / self.q_update_count
            if self.q_update_count > 0
            else 0.0
        )
        stats = {
            "battles": self.battles,
            "wins": self.wins,
            "win_rate": round(self.wins / self.battles, 3),
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.buffer),
            "steps": self.steps,
            "avg_abs_td": round(avg_abs_td, 5),
        }
        self._save()
        return stats

    def _train_step(self) -> None:
        batch = self.buffer.sample(self.batch_size)

        states = (
            torch.from_numpy(np.stack([t.state for t in batch])).float().to(self.device)
        )
        actions = (
            torch.from_numpy(np.array([t.action for t in batch])).long().to(self.device)
        )
        rewards = torch.from_numpy(
            np.array([t.reward for t in batch], dtype=np.float32)
        ).to(self.device)
        next_states = (
            torch.from_numpy(np.stack([t.next_state for t in batch]))
            .float()
            .to(self.device)
        )
        dones = torch.from_numpy(
            np.array([t.done for t in batch], dtype=np.float32)
        ).to(self.device)
        hiddens = (
            torch.from_numpy(np.concatenate([t.hidden for t in batch], axis=1))
            .float()
            .to(self.device)
        )

        self.online.train()
        q_vals, _ = self.online(states, hiddens)
        q_vals = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online picks action, target evaluates it
            next_q_online, _ = self.online(next_states, hiddens)
            best_a = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target, _ = self.target(next_states, hiddens)
            next_q = next_q_target.gather(1, best_a).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(q_vals, targets)

        td_error = (targets - q_vals).abs().mean().item()
        self.q_update_abs_sum += td_error
        self.q_update_count += 1

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------
    # Persistence — same slot as the SARSA q_table but .pt format
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            torch.save(
                {
                    "online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "steps": self.steps,
                    "battles": self.battles,
                    "wins": self.wins,
                },
                self.weights_path,
            )
        except Exception as exc:
            print(f"[DQNAgent] Save failed: {exc}")

    def _load(self) -> None:
        if not os.path.exists(self.weights_path):
            print("[DQNAgent] No checkpoint found — starting fresh.")
            return
        try:
            ckpt = torch.load(self.weights_path, map_location=self.device)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            self.optimizer.load_state_dict(ckpt["optim"])
            self.epsilon = ckpt["epsilon"]
            self.steps = ckpt["steps"]
            self.battles = ckpt["battles"]
            self.wins = ckpt.get("wins", 0)
            print(
                f"[DQNAgent] Loaded checkpoint — "
                f"battle {self.battles}, ε={self.epsilon:.4f}"
            )
        except Exception as exc:
            print(f"[DQNAgent] Load failed ({exc}) — starting fresh.")
