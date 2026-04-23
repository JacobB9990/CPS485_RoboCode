"""Game state extraction and representation for the AI agent.

Captures all relevant game data into a structured format that can be
consumed by any AI model (heuristic, RL, or external).
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnemyState:
    """Last known state of an enemy robot."""

    x: float
    y: float
    energy: float
    direction: float
    speed: float
    scan_turn: int  # game turn when this was last scanned

    def age(self, current_turn: int) -> int:
        """How many turns ago this enemy was scanned."""
        return current_turn - self.scan_turn


@dataclass
class GameState:
    """Complete snapshot of the game state from the bot's perspective.

    This is the single source of truth that gets passed to AI models.
    All values are raw game units; normalization happens in to_feature_vector().
    """

    # Bot's own state
    x: float = 0.0
    y: float = 0.0
    energy: float = 100.0
    direction: float = 0.0
    gun_direction: float = 0.0
    radar_direction: float = 0.0
    velocity: float = 0.0
    gun_heat: float = 0.0

    # Arena info
    arena_width: float = 800.0
    arena_height: float = 600.0

    # Enemy data (keyed by scanned_bot_id)
    enemies: dict[int, EnemyState] = field(default_factory=dict)

    # Per-tick event flags (reset each tick)
    hit_by_bullet: bool = False
    bullet_bearing: float = 0.0
    hit_wall: bool = False
    hit_bot: bool = False
    bullet_hit_enemy: bool = False

    # Game meta
    turn_number: int = 0
    enemy_count: int = 0
    round_number: int = 0

    @property
    def nearest_enemy(self) -> Optional[EnemyState]:
        """Return the nearest enemy based on last scan data."""
        if not self.enemies:
            return None
        return min(
            self.enemies.values(),
            key=lambda e: math.hypot(e.x - self.x, e.y - self.y),
        )

    @property
    def distance_to_walls(self) -> dict[str, float]:
        """Distance to each wall from current position."""
        return {
            "north": self.arena_height - self.y,
            "south": self.y,
            "east": self.arena_width - self.x,
            "west": self.x,
        }

    @property
    def nearest_wall_distance(self) -> float:
        """Distance to the nearest wall."""
        return min(self.distance_to_walls.values())

    def to_feature_vector(self) -> list[float]:
        """Convert game state to a normalized feature vector for ML models.

        Returns a fixed-size vector (25 floats) regardless of number of enemies.
        Features are normalized to roughly [0, 1] or [-1, 1] range.
        """
        features = []

        # Bot position (normalized by arena size) [2]
        features.append(self.x / max(self.arena_width, 1))
        features.append(self.y / max(self.arena_height, 1))

        # Bot energy (normalized by 100) [1]
        features.append(self.energy / 100.0)

        # Body heading as sin/cos for circular continuity [2]
        dir_rad = math.radians(self.direction)
        features.append(math.sin(dir_rad))
        features.append(math.cos(dir_rad))

        # Gun heading as sin/cos [2]
        gun_rad = math.radians(self.gun_direction)
        features.append(math.sin(gun_rad))
        features.append(math.cos(gun_rad))

        # Velocity (normalized by max speed 8) [1]
        features.append(self.velocity / 8.0)

        # Gun heat (normalized, typically 0-3) [1]
        features.append(self.gun_heat / 3.0)

        # Wall distances (normalized by arena diagonal) [4]
        diag = math.hypot(self.arena_width, self.arena_height)
        walls = self.distance_to_walls
        features.append(walls["north"] / diag)
        features.append(walls["south"] / diag)
        features.append(walls["east"] / diag)
        features.append(walls["west"] / diag)

        # Nearest enemy features (or zeros if no enemy known) [9]
        enemy = self.nearest_enemy
        if enemy is not None:
            dx = (enemy.x - self.x) / max(self.arena_width, 1)
            dy = (enemy.y - self.y) / max(self.arena_height, 1)
            features.append(dx)
            features.append(dy)
            features.append(enemy.energy / 100.0)
            enemy_dir_rad = math.radians(enemy.direction)
            features.append(math.sin(enemy_dir_rad))
            features.append(math.cos(enemy_dir_rad))
            features.append(enemy.speed / 8.0)
            dist = math.hypot(enemy.x - self.x, enemy.y - self.y)
            features.append(dist / diag)
            bearing = math.atan2(enemy.x - self.x, enemy.y - self.y)
            features.append(math.sin(bearing))
            features.append(math.cos(bearing))
        else:
            features.extend([0.0] * 9)

        # Event flags [3]
        features.append(1.0 if self.hit_by_bullet else 0.0)
        features.append(1.0 if self.hit_wall else 0.0)
        features.append(1.0 if self.hit_bot else 0.0)

        return features

    @staticmethod
    def feature_size() -> int:
        """Return the size of the feature vector."""
        return 25

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for serialization / logging."""
        result = {
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "direction": self.direction,
            "gun_direction": self.gun_direction,
            "radar_direction": self.radar_direction,
            "velocity": self.velocity,
            "gun_heat": self.gun_heat,
            "arena_width": self.arena_width,
            "arena_height": self.arena_height,
            "turn_number": self.turn_number,
            "enemy_count": self.enemy_count,
            "hit_by_bullet": self.hit_by_bullet,
            "hit_wall": self.hit_wall,
            "hit_bot": self.hit_bot,
            "bullet_hit_enemy": self.bullet_hit_enemy,
        }
        if self.enemies:
            result["enemies"] = {
                str(k): {
                    "x": e.x,
                    "y": e.y,
                    "energy": e.energy,
                    "direction": e.direction,
                    "speed": e.speed,
                }
                for k, e in self.enemies.items()
            }
        return result
