# dqn_agent.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
GAMMA           = 0.99      # discount factor
LR              = 1e-3      # Adam learning rate
EPSILON_START   = 1.0       # start fully random
EPSILON_END     = 0.05      # minimum exploration floor
EPSILON_DECAY   = 0.995     # multiply epsilon by this each episode
TARGET_SYNC     = 200       # copy policy → target every N training steps
BUFFER_CAPACITY = 10_000


# ── Network definition ─────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Simple 3-layer MLP.  Input = state_dim floats, output = one Q-value per action.
    ReLU activations, no output activation (Q-values are unbounded).
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Agent ──────────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    One DQNAgent per enemy category.

    act(state_vec)                  → int action (epsilon-greedy)
    remember(s, a, r, s', done)     → push to replay buffer
    train()                         → one gradient step if buffer is ready
    decay_epsilon()                 → call once per episode end
    save(path) / load(path)         → persist weights between sessions
    """

    def __init__(self, state_dim: int, action_dim: int, category: str = "UNKNOWN"):
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.category    = category

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()   # target net never trains directly

        # Optimizer & loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn   = nn.MSELoss()

        # Replay buffer (one per agent = one per category)
        self.buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

        # Exploration
        self.epsilon = EPSILON_START

        # Step counter for target sync
        self._train_steps = 0

    # ── Action selection ───────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        During early training epsilon is high so we explore randomly.
        As epsilon decays we increasingly trust the policy net.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Memory ────────────────────────────────────────────────────────────────

    def remember(self,
                 state:      np.ndarray,
                 action:     int,
                 reward:     float,
                 next_state: np.ndarray,
                 done:       bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> float | None:
        """
        One gradient step using a random mini-batch from the replay buffer.
        Returns the loss value for logging, or None if buffer isn't ready yet.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── Current Q-values from policy net ──
        # Gather the Q-value for the action that was actually taken
        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── Target Q-values from target net (no gradient) ──
        with torch.no_grad():
            q_next   = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + GAMMA * q_next * (1.0 - dones_t)

        # ── Loss and backprop ──
        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping keeps training stable — prevents exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── Sync target net periodically ──
        self._train_steps += 1
        if self._train_steps % TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"[{self.category}] Target net synced at step {self._train_steps}")

        return loss.item()

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Call once at the end of each episode."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str = "weights") -> None:
        """Save policy net weights to disk. Call periodically during training."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.category}.pt")
        torch.save({
            "policy_net":   self.policy_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "train_steps":  self._train_steps,
        }, path)
        print(f"[{self.category}] Saved to {path}")

    def load(self, directory: str = "weights") -> None:
        """Load weights if they exist, otherwise start fresh."""
        path = os.path.join(directory, f"{self.category}.pt")
        if not os.path.exists(path):
            print(f"[{self.category}] No weights found at {path}, starting fresh.")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon      = checkpoint["epsilon"]
        self._train_steps = checkpoint["train_steps"]
        print(f"[{self.category}] Loaded from {path} (ε={self.epsilon:.3f}, steps={self._train_steps})")