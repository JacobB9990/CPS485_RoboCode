"""DQN agent for RoboCode inspired by PyTorch tutorial.

Network: 3-layer feedforward with hidden layers of 128 units
Experience Replay: Cyclic buffer of transitions
Target Network: Soft update with TAU=0.005
Training: SmoothL1Loss (Huber loss), Adam optimizer
"""

from __future__ import annotations

import os
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define transition structure
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class DQN(nn.Module):
    """Feedforward DQN network: state -> 128 -> 128 -> action values."""

    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayMemory:
    """Cyclic buffer for experience replay."""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(
        self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool
    ) -> None:
        """Store a transition."""
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """DQN Agent with experience replay, target network, and soft updates."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay_steps: int = 2500,
        tau: float = 0.005,
        batch_size: int = 128,
        memory_capacity: int = 10000,
        device: torch.device | None = None,
        weights_path: str = "dqn_weights.pt",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.weights_path = weights_path
        self.training_enabled = True
        self.fixed_epsilon: float | None = None

        # Policy and target networks
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_capacity)

        self.steps_done = 0
        self.episode = 0
        self.wins = 0

        self._load()

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        # Compute epsilon: decay from eps_start to eps_end over eps_decay_steps
        eps = self.current_epsilon()

        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def push_transition(
        self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool
    ) -> None:
        """Store transition and train if buffer is full enough."""
        if not self.training_enabled:
            return

        self.memory.push(state, action, next_state, reward, done)
        self.steps_done += 1

        if len(self.memory) >= self.batch_size:
            self._train_step()

    def _train_step(self) -> None:
        """Mini-batch training step with Double DQN."""
        batch = self.memory.sample(self.batch_size)
        transitions = Transition(*zip(*batch))

        # Convert to tensors
        states = torch.FloatTensor(np.stack(transitions.state)).to(self.device)
        actions = torch.LongTensor(transitions.action).to(self.device)
        next_states = torch.FloatTensor(np.stack(transitions.next_state)).to(self.device)
        rewards = torch.FloatTensor(transitions.reward).to(self.device)
        dones = torch.BoolTensor(transitions.done).to(self.device)

        # Compute Q(s,a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute max Q(s', a') for next states
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            # Zero out Q for terminal states
            next_q_values[dones] = 0.0
            targets = rewards + self.gamma * next_q_values

        # Compute loss (Huber/SmoothL1)
        loss = nn.SmoothL1Loss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

    def _soft_update(self) -> None:
        """Soft update of target network: θ' ← τθ + (1-τ)θ'."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def on_episode_start(self) -> None:
        """Called at episode start."""
        self.episode += 1

    def on_episode_end(self, won: bool) -> None:
        """Called at episode end."""
        if won:
            self.wins += 1
        self._save()

    def current_epsilon(self) -> float:
        """Return epsilon used by policy (supports fixed epsilon in eval mode)."""
        if self.fixed_epsilon is not None:
            return float(self.fixed_epsilon)

        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.steps_done / self.eps_decay_steps
        )

    def set_eval_mode(self, epsilon: float = 0.0) -> None:
        """Disable online learning and use fixed epsilon for deterministic evaluation."""
        self.training_enabled = False
        self.fixed_epsilon = max(0.0, min(1.0, float(epsilon)))

    def set_train_mode(self) -> None:
        """Enable online learning and decay-based epsilon schedule."""
        self.training_enabled = True
        self.fixed_epsilon = None

    def _save(self) -> None:
        """Save weights and metadata."""
        try:
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "steps_done": self.steps_done,
                    "episode": self.episode,
                    "wins": self.wins,
                },
                self.weights_path,
            )
        except Exception as e:
            print(f"[DQNAgent] Save failed: {e}")

    def _load(self) -> None:
        """Load weights if they exist."""
        if not os.path.exists(self.weights_path):
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps_done = checkpoint["steps_done"]
            self.episode = checkpoint["episode"]
            self.wins = checkpoint["wins"]
            print(
                f"[DQNAgent] Loaded checkpoint: episode={self.episode}, "
                f"wins={self.wins}, steps={self.steps_done}"
            )
        except Exception as e:
            print(f"[DQNAgent] Load failed: {e}")
