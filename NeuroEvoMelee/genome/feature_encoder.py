"""Feature encoder for NeuroEvoMelee."""

from __future__ import annotations

import math
from collections.abc import Collection

from robocode_tank_royale.bot_api.bot import Bot

from .enemy_state import EnemyState


class FeatureEncoder:
    INPUT_SIZE = 20

    @staticmethod
    def encode(bot: Bot, enemies: Collection[EnemyState], nearest: EnemyState | None, weakest: EnemyState | None, preferred: EnemyState | None) -> list[float]:
        features = [0.0] * FeatureEncoder.INPUT_SIZE

        arena_width = max(1.0, float(getattr(bot, "arena_width", 1.0)))
        arena_height = max(1.0, float(getattr(bot, "arena_height", 1.0)))
        max_dist = math.hypot(arena_width, arena_height)

        x = float(bot.x)
        y = float(bot.y)
        min_wall = min(x, y, arena_width - x, arena_height - y)
        wall_proximity = 1.0 - FeatureEncoder._clamp01(min_wall / (min(arena_width, arena_height) * 0.5))

        features[0] = FeatureEncoder._clamp01(float(bot.energy) / 100.0)
        features[1] = FeatureEncoder._clamp_signed(float(getattr(bot, "speed", 0.0)) / 8.0)
        heading_rad = math.radians(float(getattr(bot, "direction", 0.0)))
        features[2] = math.sin(heading_rad)
        features[3] = math.cos(heading_rad)
        features[4] = wall_proximity

        FeatureEncoder._write_enemy_features(features, 5, bot, nearest, max_dist)
        FeatureEncoder._write_enemy_features(features, 9, bot, weakest, max_dist)

        threat_sum = 0.0
        threat_max = 0.0
        close_count = 0
        for enemy in enemies:
            if not enemy.alive:
                continue
            dist = math.dist((x, y), (enemy.x, enemy.y))
            inv = 1.0 / max(1.0, dist)
            threat = inv * (0.5 + (enemy.energy / 100.0))
            threat_sum += threat
            threat_max = max(threat_max, threat)
            if dist < 220.0:
                close_count += 1

        alive = max(0, int(getattr(bot, "others", 0)))
        features[13] = FeatureEncoder._clamp01(threat_sum * 60.0)
        features[14] = FeatureEncoder._clamp01(threat_max * 160.0)
        features[15] = FeatureEncoder._clamp01(close_count / 8.0)
        features[16] = FeatureEncoder._clamp01(alive / 10.0)
        features[17] = FeatureEncoder._clamp01(float(getattr(bot, "gun_heat", 0.0)) / 1.6)

        if preferred is None:
            features[18] = 0.0
            features[19] = 0.0
        else:
            gun_bearing = math.radians(float(bot.gun_bearing_to(preferred.x, preferred.y)))
            features[18] = FeatureEncoder._clamp01((math.cos(gun_bearing) + 1.0) * 0.5)

            nearest_dist = max_dist if nearest is None else math.dist((x, y), (nearest.x, nearest.y))
            weakest_dist = max_dist if weakest is None else math.dist((x, y), (weakest.x, weakest.y))
            nearest_pressure = 1.0 / max(1.0, nearest_dist)
            weakest_vulnerability = 0.0 if weakest is None else (1.0 - FeatureEncoder._clamp01(weakest.energy / 100.0)) * (1.0 / max(1.0, weakest_dist))
            features[19] = FeatureEncoder._clamp_signed((weakest_vulnerability - nearest_pressure) * 240.0)

        return features

    @staticmethod
    def _write_enemy_features(features: list[float], offset: int, bot: Bot, enemy: EnemyState | None, max_dist: float) -> None:
        if enemy is None or not enemy.alive:
            features[offset : offset + 4] = [1.0, 0.0, 0.0, 0.0]
            return

        dist = math.dist((bot.x, bot.y), (enemy.x, enemy.y))
        bearing = math.radians(float(bot.bearing_to(enemy.x, enemy.y)))
        features[offset] = FeatureEncoder._clamp01(dist / max_dist)
        features[offset + 1] = math.sin(bearing)
        features[offset + 2] = math.cos(bearing)
        features[offset + 3] = FeatureEncoder._clamp01(enemy.energy / 100.0)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _clamp_signed(value: float) -> float:
        return max(-1.0, min(1.0, value))
