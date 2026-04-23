"""Rule-based heuristic AI model for Robocode.

A competitive baseline strategy that uses hand-coded rules:
  - Strafes perpendicular to the nearest enemy (hard to hit)
  - Leads targets with predictive aiming
  - Adjusts fire power based on distance and energy
  - Avoids walls by steering toward the arena center
  - Dodges after being hit by reversing strafe direction

Supports difficulty levels (0.0-1.0) for progressive training:
  - 0.0: Basic movement, no lead aiming, slow reactions
  - 0.5: Default competitive behavior
  - 1.0: Aggressive dodging, tight aiming, randomized patterns
"""

import math
import random

from game_state import GameState
from actions import Action
from model_interface import ModelInterface


class HeuristicModel(ModelInterface):
    """Deterministic rule-based bot controller with adjustable difficulty.

    Args:
        difficulty: Float from 0.0 to 1.0 controlling bot sophistication.
            0.0 = simple patterns, no lead aiming, slow dodge
            0.5 = default competitive behavior
            1.0 = aggressive dodge, tight aiming, pattern variation
    """

    WALL_MARGIN = 75.0
    CLOSE_RANGE = 150.0
    MEDIUM_RANGE = 350.0
    STRAFE_DISTANCE = 100.0

    def __init__(self, difficulty: float = 0.5) -> None:
        self._difficulty = max(0.0, min(1.0, difficulty))
        self._strafe_dir: int = 1
        self._ticks_since_hit: int = 100
        self._dodge_active: bool = False
        self._pattern_timer: int = 0
        self._pattern_variation: float = 0.0

    
    # ModelInterface
    
    def decide(self, state: GameState) -> Action:
        action = Action()
        enemy = state.nearest_enemy

        # --- Pattern variation (high difficulty) ---
        self._pattern_timer += 1
        if self._difficulty > 0.6 and self._pattern_timer % 30 == 0:
            self._pattern_variation = random.uniform(-20, 20)

        # --- Dodge: reverse strafe when hit ---
        if state.hit_by_bullet:
            self._dodge_active = True
            self._ticks_since_hit = 0
            self._strafe_dir *= -1
            # High difficulty: random dodge direction
            if self._difficulty > 0.7 and random.random() < 0.3:
                self._strafe_dir *= -1  # fake-out

        self._ticks_since_hit += 1
        # Dodge duration scales with difficulty (5-15 ticks)
        dodge_duration = int(5 + self._difficulty * 10)
        if self._ticks_since_hit > dodge_duration:
            self._dodge_active = False

        # --- Body movement ---
        wall_action = self._wall_avoidance(state)
        if wall_action is not None:
            action.body_turn = wall_action.body_turn
            action.body_move = wall_action.body_move
        elif enemy is not None:
            bearing_to_enemy = math.degrees(
                math.atan2(enemy.x - state.x, enemy.y - state.y)
            )
            strafe_angle = bearing_to_enemy + (90 * self._strafe_dir)
            strafe_angle += self._pattern_variation * self._difficulty
            turn_needed = _normalize_angle(strafe_angle - state.direction)

            if self._dodge_active:
                dodge_speed = 1.2 + self._difficulty * 0.8  # 1.2x to 2.0x
                action.body_move = self.STRAFE_DISTANCE * dodge_speed
                action.body_turn = turn_needed
            else:
                action.body_move = self.STRAFE_DISTANCE * self._strafe_dir
                # Lower difficulty = less responsive turning
                turn_responsiveness = 0.3 + self._difficulty * 0.4  # 0.3 to 0.7
                action.body_turn = turn_needed * turn_responsiveness

            # Low difficulty: occasionally pause movement
            if self._difficulty < 0.3 and random.random() < 0.1:
                action.body_move *= 0.3
        else:
            # No enemy visible -- advance and scan
            action.body_move = 50
            action.body_turn = 10

        # --- Gun targeting ---
        if enemy is not None:
            dx = enemy.x - state.x
            dy = enemy.y - state.y
            distance = math.hypot(dx, dy)

            # Lead prediction only at difficulty > 0.3
            if distance > 0 and enemy.speed != 0 and self._difficulty > 0.3:
                power = self._fire_power(distance, state.energy)
                bullet_speed = 20.0 - 3.0 * power
                travel_time = distance / bullet_speed
                # Lead accuracy scales with difficulty
                lead_factor = min(1.0, (self._difficulty - 0.3) / 0.5)
                enemy_dir_rad = math.radians(enemy.direction)
                predicted_x = enemy.x + enemy.speed * math.sin(enemy_dir_rad) * travel_time * lead_factor
                predicted_y = enemy.y + enemy.speed * math.cos(enemy_dir_rad) * travel_time * lead_factor
                predicted_x = max(0, min(state.arena_width, predicted_x))
                predicted_y = max(0, min(state.arena_height, predicted_y))
                target_bearing = math.degrees(
                    math.atan2(predicted_x - state.x, predicted_y - state.y)
                )
            else:
                target_bearing = math.degrees(math.atan2(dx, dy))

            # Aim accuracy noise (lower difficulty = more noise)
            if self._difficulty < 0.8:
                noise = random.gauss(0, (1.0 - self._difficulty) * 8)
                target_bearing += noise

            gun_turn = _normalize_angle(target_bearing - state.gun_direction)
            action.gun_turn = gun_turn

            # Fire threshold widens at lower difficulty
            fire_threshold = 15 - self._difficulty * 10  # 15 to 5 degrees
            if abs(gun_turn) < fire_threshold and state.gun_heat == 0:
                action.fire_power = self._fire_power(distance, state.energy)
        else:
            # Spin gun to scan for enemies
            action.gun_turn = 15

        return action

    def on_round_start(self, round_number: int) -> None:
        self._strafe_dir = 1
        self._ticks_since_hit = 100
        self._dodge_active = False
        self._pattern_timer = 0
        self._pattern_variation = 0.0

    @property
    def difficulty(self) -> float:
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        self._difficulty = max(0.0, min(1.0, value))

    @property
    def name(self) -> str:
        return f"HeuristicModel (difficulty={self._difficulty:.1f})"

    
    # Helpers
    
    def _fire_power(self, distance: float, energy: float) -> float:
        if energy < 5:
            return 0.5
        if distance < self.CLOSE_RANGE:
            return 3.0
        if distance < self.MEDIUM_RANGE:
            return 2.0
        return 1.0

    def _wall_avoidance(self, state: GameState) -> Action | None:
        walls = state.distance_to_walls
        min_wall = min(walls, key=walls.get)
        min_dist = walls[min_wall]

        if min_dist > self.WALL_MARGIN:
            return None

        center_x = state.arena_width / 2
        center_y = state.arena_height / 2
        bearing_to_center = math.degrees(
            math.atan2(center_x - state.x, center_y - state.y)
        )
        turn_needed = _normalize_angle(bearing_to_center - state.direction)
        return Action(body_turn=turn_needed, body_move=100)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle
