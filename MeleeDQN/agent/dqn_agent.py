"""DQN agent for the MeleeDQN bot family."""

from __future__ import annotations

import os
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, hidden_sizes: Iterable[int] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        input_size = n_observations
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool) -> None:
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.985,
        eps_start: float = 0.95,
        eps_end: float = 0.05,
        eps_decay_steps: int = 15000,
        tau: float = 0.005,
        batch_size: int = 128,
        memory_capacity: int = 50000,
        weights_path: str = "melee_dqn_weights.pt",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.weights_path = weights_path
        self._eval_mode = False
        self._eval_epsilon = 0.0

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_capacity)

        self.steps_done = 0
        self.episode = 0
        self.wins = 0
        self.last_loss: float | None = None
        self._load()

    def current_epsilon(self) -> float:
        if self._eval_mode:
            return self._eval_epsilon
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * self.steps_done / self.eps_decay_steps)

    def select_action(self, state: np.ndarray) -> int:
        eps = self.current_epsilon()
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(state_t).argmax(dim=1).item())

    def push_transition(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool) -> None:
        if self._eval_mode:
            return
        self.memory.push(state, action, next_state, reward, done)
        self.steps_done += 1
        if len(self.memory) >= self.batch_size:
            self.train_step()

    def train_step(self) -> None:
        batch = self.memory.sample(self.batch_size)
        transitions = Transition(*zip(*batch))

        states = torch.tensor(np.stack(transitions.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(transitions.action, dtype=torch.long, device=self.device)
        next_states = torch.tensor(np.stack(transitions.next_state), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(transitions.done, dtype=torch.bool, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q_values[dones] = 0.0
            targets = rewards + (self.gamma * next_q_values)

        loss = nn.SmoothL1Loss()(q_values, targets)
        self.last_loss = float(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.soft_update()

    def soft_update(self) -> None:
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def on_episode_start(self) -> None:
        self.episode += 1

    def on_episode_end(self, won: bool) -> None:
        if won:
            self.wins += 1
        self.save()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self.policy_net.train()
        self.target_net.eval()

    def set_eval_mode(self, epsilon: float = 0.0) -> None:
        self._eval_mode = True
        self._eval_epsilon = float(epsilon)
        self.policy_net.eval()
        self.target_net.eval()

    def save(self) -> None:
        path = Path(self.weights_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "episode": self.episode,
                "wins": self.wins,
            },
            path,
        )

    def load(self) -> None:
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.weights_path):
            return
        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps_done = int(checkpoint.get("steps_done", 0))
            self.episode = int(checkpoint.get("episode", 0))
            self.wins = int(checkpoint.get("wins", 0))
        except Exception as exc:
            print(f"[DQNAgent] Load failed: {exc}")
