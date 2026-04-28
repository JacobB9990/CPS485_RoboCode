from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


MAX_ENEMIES = 12
TARGET_STICKINESS_BONUS = 0.18
TARGET_SWITCH_MARGIN = 0.08
OBSERVATION_DIM = 42
ACTION_BRANCH_SIZES = (5, 5, 4, 5)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm(value: float, scale: float, low: float = -1.0, high: float = 1.0) -> float:
    if scale <= 0:
        return 0.0
    return _clip(value / scale, low, high)


def _angle_normalize(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass(slots=True)
class SelfState:
    energy: float
    x: float
    y: float
    velocity: float
    heading: float
    gun_heading: float
    gun_heat: float


@dataclass(slots=True)
class EnemyState:
    name: str
    x: float
    y: float
    distance: float
    abs_bearing: float
    relative_bearing: float
    velocity: float
    heading: float
    energy: float
    last_seen_tick: int
    alive: bool = True


@dataclass(slots=True)
class BattleSnapshot:
    tick: int
    arena_width: float
    arena_height: float
    self_state: SelfState
    enemies: dict[str, EnemyState]
    alive_enemy_count: int
    current_placement: int
    bullet_damage_dealt: float = 0.0
    bullet_damage_taken: float = 0.0
    kills_gained: int = 0
    hit_wall: bool = False
    fired_power: float = 0.0
    bullet_hit: bool = False
    won: bool = False
    done: bool = False


@dataclass(slots=True)
class TargetSelection:
    target_name: str | None
    score: float
    switched: bool


@dataclass(slots=True)
class RewardBreakdown:
    total: float
    damage_dealt: float
    kills: float
    survival: float
    placement: float
    win_bonus: float
    damage_taken: float
    wall_penalty: float
    danger_penalty: float
    wasted_fire_penalty: float


@dataclass(slots=True)
class DecodedAction:
    movement_mode: str
    move_distance: float
    body_turn_radians: float
    gun_turn_radians: float
    fire_power: float
    radar_turn_radians: float


class StickyTargetSelector:
    """Hand-coded target selector with switch hysteresis for PPO stability."""

    def select(
        self, snapshot: BattleSnapshot, previous_target: str | None
    ) -> TargetSelection:
        if not snapshot.enemies:
            return TargetSelection(target_name=None, score=0.0, switched=False)

        current_score = -1e9
        best_name = None
        best_score = -1e9

        for enemy in snapshot.enemies.values():
            if not enemy.alive:
                continue
            score = self._score_enemy(snapshot, enemy)
            if enemy.name == previous_target:
                current_score = score
                score += TARGET_STICKINESS_BONUS
            if score > best_score:
                best_score = score
                best_name = enemy.name

        if previous_target and best_name != previous_target:
            if best_score < current_score + TARGET_SWITCH_MARGIN:
                return TargetSelection(previous_target, current_score, switched=False)

        return TargetSelection(
            target_name=best_name,
            score=best_score,
            switched=best_name != previous_target,
        )

    def _score_enemy(self, snapshot: BattleSnapshot, enemy: EnemyState) -> float:
        max_dist = math.hypot(snapshot.arena_width, snapshot.arena_height)
        age_ticks = max(snapshot.tick - enemy.last_seen_tick, 0)
        freshness = 1.0 - _clip(age_ticks / 40.0, 0.0, 1.0)
        distance_term = 1.0 - _clip(enemy.distance / max_dist, 0.0, 1.0)
        weak_term = 1.0 - _clip(enemy.energy / 100.0, 0.0, 1.0)
        aim_term = 1.0 - abs(enemy.relative_bearing) / math.pi
        threat_term = _clip(
            (enemy.energy / 100.0) * (1.0 - _clip(enemy.distance / max_dist, 0.0, 1.0)),
            0.0,
            1.0,
        )
        return 0.40 * distance_term + 0.25 * weak_term + 0.20 * freshness + 0.10 * aim_term + 0.05 * threat_term


class MeleeObservationBuilder:
    """Fixed-size melee encoder built from enemy aggregates and target summaries."""

    def build(
        self, snapshot: BattleSnapshot, target_name: str | None
    ) -> tuple[np.ndarray, dict[str, float]]:
        s = snapshot.self_state
        width = snapshot.arena_width
        height = snapshot.arena_height
        max_dist = math.hypot(width, height)
        enemies = [enemy for enemy in snapshot.enemies.values() if enemy.alive]

        nearest = min(enemies, key=lambda e: e.distance, default=None)
        weakest = min(enemies, key=lambda e: e.energy, default=None)
        strongest_threat = max(
            enemies,
            key=lambda e: self._threat_score(snapshot, e),
            default=None,
        )
        current_target = snapshot.enemies.get(target_name) if target_name else None

        wall_distances = [
            _clip(s.x / width, 0.0, 1.0),
            _clip((width - s.x) / width, 0.0, 1.0),
            _clip(s.y / height, 0.0, 1.0),
            _clip((height - s.y) / height, 0.0, 1.0),
        ]

        crowd_features = self._crowd_density_features(snapshot, enemies)
        danger_score = self.compute_local_danger(snapshot)

        obs = np.array(
            [
                _clip(s.energy / 100.0, 0.0, 1.0),
                _clip(s.x / width, 0.0, 1.0),
                _clip(s.y / height, 0.0, 1.0),
                _norm(s.velocity, 8.0),
                _norm(_angle_normalize(s.heading), math.pi),
                _norm(_angle_normalize(s.gun_heading), math.pi),
                _clip(s.gun_heat / 2.0, 0.0, 1.0),
                *wall_distances,
                _clip(snapshot.alive_enemy_count / MAX_ENEMIES, 0.0, 1.0),
                *self._enemy_block(snapshot, nearest),
                *self._enemy_block(snapshot, weakest),
                *self._enemy_block(snapshot, strongest_threat),
                *crowd_features,
                _clip(danger_score, 0.0, 1.0),
                *self._enemy_block(snapshot, current_target),
            ],
            dtype=np.float32,
        )
        diagnostics = {
            "danger_score": float(danger_score),
            "enemy_count": float(snapshot.alive_enemy_count),
            "nearest_distance": float(nearest.distance if nearest else max_dist),
        }
        return obs, diagnostics

    def _enemy_block(self, snapshot: BattleSnapshot, enemy: EnemyState | None) -> list[float]:
        if enemy is None:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        max_dist = math.hypot(snapshot.arena_width, snapshot.arena_height)
        age_ticks = max(snapshot.tick - enemy.last_seen_tick, 0)
        return [
            _clip(enemy.distance / max_dist, 0.0, 1.0),
            _norm(enemy.relative_bearing, math.pi),
            _norm(enemy.velocity, 8.0),
            _norm(_angle_normalize(enemy.heading), math.pi),
            _clip(enemy.energy / 100.0, 0.0, 1.0),
            _clip(age_ticks / 80.0, 0.0, 1.0),
        ]

    def _crowd_density_features(
        self, snapshot: BattleSnapshot, enemies: Iterable[EnemyState]
    ) -> list[float]:
        enemies = list(enemies)
        if not enemies:
            return [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        max_dist = math.hypot(snapshot.arena_width, snapshot.arena_height)
        distances = np.array([enemy.distance for enemy in enemies], dtype=np.float32)
        bearings = np.array([enemy.abs_bearing for enemy in enemies], dtype=np.float32)
        weights = np.array(
            [1.0 / max(enemy.distance, 36.0) for enemy in enemies], dtype=np.float32
        )
        weights /= max(weights.sum(), 1e-6)

        mean_dist = float(distances.mean())
        std_dist = float(distances.std()) if len(distances) > 1 else 0.0
        min_dist = float(distances.min())
        center_x = float(np.sum(np.sin(bearings) * weights))
        center_y = float(np.sum(np.cos(bearings) * weights))
        center_bearing = math.atan2(center_x, center_y)
        low_density_bearing = _angle_normalize(center_bearing + math.pi)
        enemy_energy_sum = float(sum(enemy.energy for enemy in enemies))

        return [
            _clip(mean_dist / max_dist, 0.0, 1.0),
            _clip(std_dist / (0.5 * max_dist), 0.0, 1.0),
            _clip(min_dist / max_dist, 0.0, 1.0),
            _norm(center_bearing, math.pi),
            _norm(low_density_bearing, math.pi),
            _clip(enemy_energy_sum / (100.0 * MAX_ENEMIES), 0.0, 1.0),
        ]

    def compute_local_danger(self, snapshot: BattleSnapshot) -> float:
        s = snapshot.self_state
        width = snapshot.arena_width
        height = snapshot.arena_height
        wall_margin = min(s.x, width - s.x, s.y, height - s.y)
        wall_danger = 1.0 - _clip(wall_margin / 120.0, 0.0, 1.0)

        enemy_danger = 0.0
        for enemy in snapshot.enemies.values():
            if not enemy.alive:
                continue
            age_ticks = max(snapshot.tick - enemy.last_seen_tick, 0)
            freshness = 1.0 - _clip(age_ticks / 50.0, 0.0, 1.0)
            dist_term = 1.0 - _clip(enemy.distance / 600.0, 0.0, 1.0)
            energy_term = _clip(enemy.energy / 100.0, 0.0, 1.0)
            angle_term = 1.0 - abs(enemy.relative_bearing) / math.pi
            enemy_danger += freshness * dist_term * (0.65 * energy_term + 0.35 * angle_term)

        return _clip(0.55 * wall_danger + 0.45 * enemy_danger, 0.0, 1.0)

    def _threat_score(self, snapshot: BattleSnapshot, enemy: EnemyState) -> float:
        age_ticks = max(snapshot.tick - enemy.last_seen_tick, 0)
        freshness = 1.0 - _clip(age_ticks / 50.0, 0.0, 1.0)
        distance_term = 1.0 - _clip(enemy.distance / 700.0, 0.0, 1.0)
        bearing_term = 1.0 - abs(enemy.relative_bearing) / math.pi
        return freshness * (0.45 * _clip(enemy.energy / 100.0, 0.0, 1.0) + 0.35 * distance_term + 0.20 * bearing_term)


class MeleeActionDecoder:
    """Decodes PPO multi-discrete actions into Robocode-friendly commands."""

    MOVEMENT_MODES = {
        0: "hold",
        1: "toward_low_density",
        2: "perpendicular_left",
        3: "perpendicular_right",
        4: "escape_crowd",
    }
    BODY_TURNS = {
        0: -math.radians(30),
        1: -math.radians(12),
        2: 0.0,
        3: math.radians(12),
        4: math.radians(30),
    }
    FIRE_POWERS = {
        0: 0.0,
        1: 0.8,
        2: 1.6,
        3: 2.4,
    }
    RADAR_TURNS = {
        0: -math.radians(60),
        1: -math.radians(20),
        2: 0.0,
        3: math.radians(20),
        4: math.radians(60),
    }

    def __init__(self, observation_builder: MeleeObservationBuilder | None = None) -> None:
        self.observation_builder = observation_builder or MeleeObservationBuilder()

    def decode(
        self, action: tuple[int, int, int, int], snapshot: BattleSnapshot, target_name: str | None
    ) -> DecodedAction:
        move_idx, turn_idx, fire_idx, radar_idx = action
        target = snapshot.enemies.get(target_name) if target_name else None

        movement_mode = self.MOVEMENT_MODES[int(move_idx)]
        base_heading = self._movement_heading(snapshot, target, movement_mode)
        smoothed_heading = self._wall_smoothed_heading(snapshot, base_heading)
        body_turn = _angle_normalize(smoothed_heading - snapshot.self_state.heading)
        body_turn += self.BODY_TURNS[int(turn_idx)]
        body_turn = _angle_normalize(body_turn)

        move_distance = 0.0 if movement_mode == "hold" else 120.0
        gun_turn = self._gun_turn(snapshot, target)
        fire_power = self._legal_fire_power(snapshot, target, self.FIRE_POWERS[int(fire_idx)])
        radar_turn = self._radar_turn(snapshot, target) + self.RADAR_TURNS[int(radar_idx)]

        return DecodedAction(
            movement_mode=movement_mode,
            move_distance=move_distance,
            body_turn_radians=body_turn,
            gun_turn_radians=gun_turn,
            fire_power=fire_power,
            radar_turn_radians=_angle_normalize(radar_turn),
        )

    def _movement_heading(
        self,
        snapshot: BattleSnapshot,
        target: EnemyState | None,
        movement_mode: str,
    ) -> float:
        crowd_features = self.observation_builder._crowd_density_features(
            snapshot, [enemy for enemy in snapshot.enemies.values() if enemy.alive]
        )
        low_density_bearing = crowd_features[4] * math.pi
        low_density_heading = _angle_normalize(snapshot.self_state.heading + low_density_bearing)

        if movement_mode == "toward_low_density":
            return low_density_heading
        if movement_mode == "escape_crowd":
            return low_density_heading
        if target is None:
            return low_density_heading
        if movement_mode == "perpendicular_left":
            return _angle_normalize(target.abs_bearing + math.pi / 2.0)
        if movement_mode == "perpendicular_right":
            return _angle_normalize(target.abs_bearing - math.pi / 2.0)
        return snapshot.self_state.heading

    def _wall_smoothed_heading(self, snapshot: BattleSnapshot, desired_heading: float) -> float:
        width = snapshot.arena_width
        height = snapshot.arena_height
        x = snapshot.self_state.x
        y = snapshot.self_state.y
        test_heading = desired_heading
        for _ in range(12):
            future_x = x + math.sin(test_heading) * 140.0
            future_y = y + math.cos(test_heading) * 140.0
            if 48.0 <= future_x <= width - 48.0 and 48.0 <= future_y <= height - 48.0:
                return test_heading
            test_heading = _angle_normalize(test_heading + math.radians(8))
        return desired_heading

    def _gun_turn(self, snapshot: BattleSnapshot, target: EnemyState | None) -> float:
        if target is None:
            return 0.0
        return _angle_normalize(target.abs_bearing - snapshot.self_state.gun_heading)

    def _radar_turn(self, snapshot: BattleSnapshot, target: EnemyState | None) -> float:
        if target is None:
            return math.radians(45)
        return _angle_normalize(target.abs_bearing - snapshot.self_state.heading)

    def _legal_fire_power(
        self, snapshot: BattleSnapshot, target: EnemyState | None, requested_power: float
    ) -> float:
        if requested_power <= 0.0 or target is None:
            return 0.0
        if snapshot.self_state.gun_heat > 0.15:
            return 0.0
        if snapshot.self_state.energy <= 0.5:
            return 0.0

        distance_scale = 1.0 - _clip(target.distance / 900.0, 0.0, 0.65)
        return _clip(requested_power * distance_scale, 0.0, min(3.0, snapshot.self_state.energy - 0.1))


class MeleeRewardShaper:
    """Dense reward shaping with bounded components for multi-enemy PPO."""

    def __init__(self, observation_builder: MeleeObservationBuilder | None = None) -> None:
        self.observation_builder = observation_builder or MeleeObservationBuilder()

    def compute(
        self,
        previous_snapshot: BattleSnapshot,
        current_snapshot: BattleSnapshot,
        action: DecodedAction,
    ) -> RewardBreakdown:
        danger = self.observation_builder.compute_local_danger(current_snapshot)
        max_rank = current_snapshot.alive_enemy_count + 1

        damage_dealt = 0.030 * current_snapshot.bullet_damage_dealt
        kills = 0.600 * current_snapshot.kills_gained
        survival = 0.0025
        placement = 0.0
        if previous_snapshot.current_placement > current_snapshot.current_placement:
            placement = 0.20 * (previous_snapshot.current_placement - current_snapshot.current_placement)
        if current_snapshot.done:
            normalized_finish = 1.0 - ((current_snapshot.current_placement - 1) / max(max_rank - 1, 1))
            placement += 0.75 * normalized_finish

        win_bonus = 1.25 if current_snapshot.won else 0.0
        damage_taken = -0.035 * current_snapshot.bullet_damage_taken
        wall_penalty = -0.12 if current_snapshot.hit_wall else 0.0
        danger_penalty = -0.020 * danger
        wasted_fire_penalty = self._wasted_fire_penalty(current_snapshot, action)

        total = (
            damage_dealt
            + kills
            + survival
            + placement
            + win_bonus
            + damage_taken
            + wall_penalty
            + danger_penalty
            + wasted_fire_penalty
        )
        total = float(_clip(total, -2.0, 2.0))
        return RewardBreakdown(
            total=total,
            damage_dealt=damage_dealt,
            kills=kills,
            survival=survival,
            placement=placement,
            win_bonus=win_bonus,
            damage_taken=damage_taken,
            wall_penalty=wall_penalty,
            danger_penalty=danger_penalty,
            wasted_fire_penalty=wasted_fire_penalty,
        )

    def _wasted_fire_penalty(self, snapshot: BattleSnapshot, action: DecodedAction) -> float:
        if action.fire_power <= 0.0:
            return 0.0
        if snapshot.bullet_hit:
            return 0.0

        target_penalty = -0.02
        if snapshot.self_state.gun_heat > 0.0:
            target_penalty -= 0.01
        return target_penalty
