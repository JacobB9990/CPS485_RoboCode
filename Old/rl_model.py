"""Deep Q-Network (DQN) reinforcement learning model for Robocode.

Uses a 3-layer MLP to approximate Q-values for a 13-action discrete
action space.  During gameplay the model runs inference only; training
is done offline with collected experience data (see train.py).

PyTorch is an optional dependency -- the heuristic model works without it.
"""

import math
import os
import random

from game_state import GameState
from actions import Action, ActionType
from model_interface import ModelInterface

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False



# Network architecture (only defined when PyTorch is installed)
if TORCH_AVAILABLE:

    class DQNetwork(nn.Module):
        """Feed-forward Q-network.

        Input:  state feature vector  (size = GameState.feature_size())
        Output: Q-value per action    (size = ActionType.count())
        """

        def __init__(
            self,
            state_size: int,
            action_size: int,
            hidden_size: int = 128,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.out = nn.Linear(hidden_size // 2, action_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.out(x)



# RL Model

class RLModel(ModelInterface):
    """DQN-based reinforcement learning controller.

    Workflow:
      1. Run games with --model rl --collect to gather experience.
      2. Train offline with train.py using the saved experience.
      3. Run games with --model rl --model-path models/dqn_model.pt
         (low epsilon) to use the trained policy.
    """

    def __init__(
        self,
        model_path: str | None = None,
        epsilon: float = 0.1,
        device: str = "auto",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the RL model. "
                "Install with: pip install torch"
            )

        self._state_size = GameState.feature_size()
        self._action_size = ActionType.count()

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # Build network
        self._network = DQNetwork(
            self._state_size, self._action_size
        ).to(self._device)

        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self._network.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            print(f"[RLModel] Loaded weights from {model_path}")

        self._network.eval()
        self._epsilon = epsilon
        self._last_enemy_bearing: float = 0.0

        # Experience collection -- store snapshots, not references
        self._prev_features: list[float] | None = None
        self._prev_reward_info: dict | None = None
        self._prev_action: ActionType | None = None
        self._experiences: list[dict] = []


    # ModelInterface
    def decide(self, state: GameState) -> Action:
        # Snapshot current state NOW (before the GameState object gets mutated)
        current_features = state.to_feature_vector()
        current_reward_info = self._snapshot_reward_info(state)

        # Store transition from previous tick
        if self._prev_features is not None and self._prev_action is not None:
            reward = self._calculate_reward(self._prev_reward_info, state)
            self._experiences.append(
                {
                    "state": self._prev_features,
                    "action": int(self._prev_action),
                    "reward": reward,
                    "next_state": current_features,
                    "done": False,
                }
            )

        # Track gun bearing to enemy for fire/track actions
        enemy = state.nearest_enemy
        if enemy is not None:
            target_bearing = math.degrees(
                math.atan2(enemy.x - state.x, enemy.y - state.y)
            )
            self._last_enemy_bearing = target_bearing - state.gun_direction

        # Epsilon-greedy action selection
        if random.random() < self._epsilon:
            action_type = ActionType(random.randint(0, self._action_size - 1))
        else:
            state_tensor = (
                torch.FloatTensor(current_features).unsqueeze(0).to(self._device)
            )
            with torch.no_grad():
                q_values = self._network(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            action_type = ActionType(action_idx)

        # Save snapshots (not references) for next tick's transition
        self._prev_features = current_features
        self._prev_reward_info = current_reward_info
        self._prev_action = action_type

        return Action.from_action_type(action_type, self._last_enemy_bearing)

    def on_round_start(self, round_number: int) -> None:
        self._prev_features = None
        self._prev_reward_info = None
        self._prev_action = None

    def on_round_end(self, round_number: int, won: bool) -> None:
        # Mark last experience as terminal with win/loss bonus
        if self._experiences:
            self._experiences[-1]["done"] = True
            self._experiences[-1]["reward"] += 10.0 if won else -5.0
        self._prev_features = None
        self._prev_reward_info = None
        self._prev_action = None

    @property
    def name(self) -> str:
        return "RLModel (DQN)"

    
    # Reward shaping
    @staticmethod
    def _snapshot_reward_info(state: GameState) -> dict:
        """Capture reward-relevant values as a plain dict (not a reference)."""
        enemy = state.nearest_enemy
        return {
            "energy": state.energy,
            "nearest_enemy_energy": enemy.energy if enemy else None,
            "nearest_wall_distance": state.nearest_wall_distance,
        }

    @staticmethod
    def _calculate_reward(prev_info: dict, curr: GameState) -> float:
        """Reward from prev snapshot vs current live state."""
        reward = 0.0

        # Small survival bonus each tick
        reward += 0.1

        # Energy delta
        reward += (curr.energy - prev_info["energy"]) * 0.5

        # Hit by bullet penalty
        if curr.hit_by_bullet:
            reward -= 2.0

        # Hit wall penalty
        if curr.hit_wall:
            reward -= 1.0

        # Our bullet hit an enemy
        if curr.bullet_hit_enemy:
            reward += 3.0

        # Enemy energy decrease (proxy for our bullet hitting them)
        prev_enemy_e = prev_info["nearest_enemy_energy"]
        if prev_enemy_e is not None and curr.nearest_enemy:
            enemy_delta = curr.nearest_enemy.energy - prev_enemy_e
            if enemy_delta < 0:
                reward += abs(enemy_delta) * 0.5

        # Wall proximity penalty
        if curr.nearest_wall_distance < 50:
            reward -= 0.3

        return reward

    
    # Experience access
    
    def get_experiences(self) -> list[dict]:
        """Return collected experiences and clear the internal buffer."""
        exps = self._experiences.copy()
        self._experiences.clear()
        return exps

    def save_model(self, path: str) -> None:
        """Save current network weights to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._network.state_dict(), path)
        print(f"[RLModel] Saved weights to {path}")
