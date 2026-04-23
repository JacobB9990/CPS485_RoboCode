"""Action definitions for the AI agent.

Defines both a discrete action space (for RL) and a continuous
Action dataclass (for direct execution by the bot).
"""

from dataclasses import dataclass
from enum import IntEnum


class ActionType(IntEnum):
    """Discrete action types for the RL model.

    Each maps to a compound movement/fire behavior.
    """

    # Movement actions
    FORWARD = 0
    FORWARD_LEFT = 1
    FORWARD_RIGHT = 2
    BACKWARD = 3
    BACKWARD_LEFT = 4
    BACKWARD_RIGHT = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7

    # Fire actions (aim at tracked enemy, then fire)
    FIRE_LOW = 8      # power 1.0
    FIRE_MEDIUM = 9   # power 2.0
    FIRE_HIGH = 10    # power 3.0

    # Targeting
    TRACK_ENEMY = 11  # turn gun toward enemy without firing

    # Evasion
    DODGE = 12        # sharp perpendicular move

    @staticmethod
    def count() -> int:
        """Total number of discrete actions."""
        return 13


@dataclass
class Action:
    """Continuous action that the bot executes.

    Positive body_turn = turn right; negative = turn left.
    Positive body_move = forward; negative = backward.
    Positive gun_turn = turn gun right; negative = turn gun left.
    fire_power 0 = don't fire; 0.1-3.0 = fire.
    """

    body_turn: float = 0.0
    body_move: float = 0.0
    gun_turn: float = 0.0
    fire_power: float = 0.0

    @staticmethod
    def from_action_type(
        action_type: ActionType,
        gun_bearing_to_enemy: float = 0.0,
    ) -> "Action":
        """Convert a discrete ActionType to a continuous Action.

        Args:
            action_type: The discrete action to convert.
            gun_bearing_to_enemy: Current gun-relative bearing to the
                nearest enemy (degrees). Used for fire/track actions.
        """
        if action_type == ActionType.FORWARD:
            return Action(body_move=80)
        elif action_type == ActionType.FORWARD_LEFT:
            return Action(body_turn=-30, body_move=80)
        elif action_type == ActionType.FORWARD_RIGHT:
            return Action(body_turn=30, body_move=80)
        elif action_type == ActionType.BACKWARD:
            return Action(body_move=-80)
        elif action_type == ActionType.BACKWARD_LEFT:
            return Action(body_turn=-30, body_move=-80)
        elif action_type == ActionType.BACKWARD_RIGHT:
            return Action(body_turn=30, body_move=-80)
        elif action_type == ActionType.TURN_LEFT:
            return Action(body_turn=-45)
        elif action_type == ActionType.TURN_RIGHT:
            return Action(body_turn=45)
        elif action_type == ActionType.FIRE_LOW:
            return Action(gun_turn=gun_bearing_to_enemy, fire_power=1.0)
        elif action_type == ActionType.FIRE_MEDIUM:
            return Action(gun_turn=gun_bearing_to_enemy, fire_power=2.0)
        elif action_type == ActionType.FIRE_HIGH:
            return Action(gun_turn=gun_bearing_to_enemy, fire_power=3.0)
        elif action_type == ActionType.TRACK_ENEMY:
            return Action(gun_turn=gun_bearing_to_enemy)
        elif action_type == ActionType.DODGE:
            return Action(body_turn=90, body_move=100)
        return Action()
