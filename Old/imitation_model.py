"""Behavioral cloning (imitation learning) model for Robocode.

Learns to replicate the heuristic bot's actions by training a neural
network on recorded (state, action) pairs using supervised learning.

The trained imitation model can then be used as a warm-start for the
RL model, giving it a strong baseline to improve upon.

Architecture matches DQNetwork so weights can be transferred directly
to the RL model for fine-tuning.
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


def action_to_discrete(action: Action, state: GameState) -> ActionType:
    """Map a continuous Action back to the nearest discrete ActionType.

    Used to create labeled training data from the heuristic bot's
    continuous decisions.
    """
    # If firing, classify by fire power
    if action.fire_power >= 2.5:
        return ActionType.FIRE_HIGH
    if action.fire_power >= 1.5:
        return ActionType.FIRE_MEDIUM
    if action.fire_power > 0:
        return ActionType.FIRE_LOW

    # Pure dodge: large turn + large forward move
    if abs(action.body_turn) > 60 and action.body_move > 50:
        return ActionType.DODGE

    # Track enemy: significant gun turn, no body action
    if (
        abs(action.gun_turn) > 5
        and abs(action.body_move) < 10
        and abs(action.body_turn) < 10
    ):
        return ActionType.TRACK_ENEMY

    # Movement classification
    moving_forward = action.body_move > 10
    moving_backward = action.body_move < -10
    turning_left = action.body_turn < -10
    turning_right = action.body_turn > 10

    if moving_forward and turning_left:
        return ActionType.FORWARD_LEFT
    if moving_forward and turning_right:
        return ActionType.FORWARD_RIGHT
    if moving_forward:
        return ActionType.FORWARD
    if moving_backward and turning_left:
        return ActionType.BACKWARD_LEFT
    if moving_backward and turning_right:
        return ActionType.BACKWARD_RIGHT
    if moving_backward:
        return ActionType.BACKWARD
    if turning_left:
        return ActionType.TURN_LEFT
    if turning_right:
        return ActionType.TURN_RIGHT

    # Default: forward
    return ActionType.FORWARD


class ImitationModel(ModelInterface):
    """Behavioral cloning model that imitates the heuristic bot.

    Uses the same DQNetwork architecture as the RL model so weights
    can be transferred. During inference, picks the action with the
    highest predicted probability (softmax over network outputs).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the imitation model. "
                "Install with: pip install torch"
            )

        from rl_model import DQNetwork

        self._state_size = GameState.feature_size()
        self._action_size = ActionType.count()

        if device == "auto":
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        self._network = DQNetwork(
            self._state_size, self._action_size
        ).to(self._device)

        if model_path and os.path.exists(model_path):
            self._network.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            print(f"[ImitationModel] Loaded weights from {model_path}")

        self._network.eval()
        self._last_enemy_bearing: float = 0.0

    def decide(self, state: GameState) -> Action:
        # Track gun bearing for fire/track actions
        enemy = state.nearest_enemy
        if enemy is not None:
            target_bearing = math.degrees(
                math.atan2(enemy.x - state.x, enemy.y - state.y)
            )
            self._last_enemy_bearing = target_bearing - state.gun_direction

        features = state.to_feature_vector()
        state_tensor = (
            torch.FloatTensor(features).unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            logits = self._network(state_tensor)
            action_idx = logits.argmax(dim=1).item()

        action_type = ActionType(action_idx)
        return Action.from_action_type(action_type, self._last_enemy_bearing)

    def save_model(self, path: str) -> None:
        """Save network weights to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._network.state_dict(), path)
        print(f"[ImitationModel] Saved weights to {path}")

    @property
    def name(self) -> str:
        return "ImitationModel (Behavioral Cloning)"
