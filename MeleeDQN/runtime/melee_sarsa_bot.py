"""Python conversion of the melee SARSA bot."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import DeathEvent, HitByBulletEvent, HitWallEvent, ScannedBotEvent, WonRoundEvent

from MeleeDQN.agent.sarsa_table import SarsaTable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_Q_TABLE_PATH = ROOT / "data" / "q_table_sarsa.json"
DEFAULT_LOG_PATH = ROOT / "logs" / "melee_sarsa_training_log.jsonl"


def _normalize_angle(degrees: float) -> float:
    while degrees > 180:
        degrees -= 360
    while degrees < -180:
        degrees += 360
    return degrees


def _bullet_damage(power: float) -> float:
    return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


@dataclass
class LastScan:
    x: float
    y: float
    turn: int


class MeleeSarsaBot(Bot):
    ORBIT_CLOCKWISE = 0
    ORBIT_COUNTERCLOCKWISE = 1
    RETREAT_CLUSTER = 2
    ADVANCE_OPEN_SPACE = 3
    RADAR_SWEEP_LEFT = 4
    RADAR_SWEEP_RIGHT = 5
    FIRE_LOW = 6
    FIRE_MEDIUM = 7
    FIRE_HIGH = 8
    EVADE = 9
    ACTION_COUNT = 10

    ALPHA = 0.12
    GAMMA = 0.95
    EPSILON_MIN = 0.08
    EPSILON_DECAY = 0.9975
    STALE_SCAN_TICKS = 24.0
    WALL_MARGIN = 72.0
    SHARED_Q_TABLE = SarsaTable(ACTION_COUNT)

    q_table_loaded = False
    shared_epsilon = 1.0
    shared_episode_counter = 0

    def __init__(self, alpha: float, gamma: float, epsilon: float, epsilon_decay: float, epsilon_min: float, q_table_path: str, log_path: str) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table_path = q_table_path
        self.log_path = log_path

        self.random = random.Random()
        self.enemies: dict[str, dict[str, float | bool | int]] = {}
        self.current_target_name: str | None = None
        self.previous_state: str | None = None
        self.previous_action: int | None = None
        self.step_reward_accumulator = 0.0
        self.episode_reward = 0.0
        self.td_abs_sum = 0.0
        self.td_count = 0
        self.crowded_ticks = 0
        self.wall_hits = 0
        self.fire_actions = 0
        self.episode_number = 0
        self.starting_opponent_count = 0
        self.last_scan: LastScan | None = None

    def run(self) -> None:
        if not self.__class__.q_table_loaded:
            self.__class__.SHARED_Q_TABLE.load(self.q_table_path)
            self.__class__.q_table_loaded = True
        self.epsilon = self.__class__.shared_epsilon

        self.set_adjust_gun_for_robot_turn(True)
        self.set_adjust_radar_for_gun_turn(True)
        self.set_adjust_radar_for_robot_turn(True)
        self.set_max_velocity(8.0)

        self._reset_round_state()

        while self.running:
            self.starting_opponent_count = max(self.starting_opponent_count, self.others)
            self._remove_stale_enemies()

            state_view = self._build_state_view()
            self.step_reward_accumulator += 0.01
            self._update_crowding_penalty(state_view)

            action = self._select_action(state_view)
            if self.previous_state is not None and self.previous_action is not None:
                self._sarsa_update(self.previous_state, self.previous_action, self.step_reward_accumulator, state_view[0], action, False)
                self.episode_reward += self.step_reward_accumulator

            self.step_reward_accumulator = 0.0
            self._execute_action(action, state_view)
            self.previous_state = state_view[0]
            self.previous_action = action

            self.execute()

    def _reset_round_state(self) -> None:
        self.episode_number = self.__class__.shared_episode_counter + 1
        self.__class__.shared_episode_counter = self.episode_number
        self.enemies.clear()
        self.current_target_name = None
        self.previous_state = None
        self.previous_action = None
        self.step_reward_accumulator = 0.0
        self.episode_reward = 0.0
        self.td_abs_sum = 0.0
        self.td_count = 0
        self.crowded_ticks = 0
        self.wall_hits = 0
        self.fire_actions = 0
        self.starting_opponent_count = self.others

    def _build_state_view(self) -> tuple[str, int, int, bool, dict[str, float] | None, dict[str, float] | None]:
        self._choose_target(self.current_target_name, True)
        nearest = self._freshest_nearest_enemy()
        weakest = self._freshest_weakest_enemy()
        target = self._current_target()

        key = (
            f"me{self._bucket_my_energy(self.energy)}|nd{self._bucket_distance(nearest['distance'] if nearest else 900.0)}|"
            f"nb{self._bucket_bearing(nearest['bearing'] if nearest else 0.0)}|wd{self._bucket_compact_distance(weakest['distance'] if weakest else 900.0)}|"
            f"nn{self._bucket_nearby_enemies(self._count_nearby_enemies(260.0))}|dg{self._compute_danger_bucket()}|"
            f"wp{self._bucket_wall(self._min_wall_distance())}|gr{1 if self.gun_heat <= 0.0001 else 0}|"
            f"te{self._bucket_target_energy(target['energy'] if target else 100.0)}|td{self._bucket_distance(target['distance'] if target else 900.0)}"
        )
        return key, self._compute_danger_bucket(), self._bucket_wall(self._min_wall_distance()), self.gun_heat <= 0.0001, target, nearest

    def _choose_target(self, preferred_name: str | None, allow_sticky_hold: bool) -> None:
        incumbent = self.enemies.get(preferred_name) if preferred_name is not None else None
        if incumbent is not None and ((not incumbent.get("alive", False)) or (self.time - int(incumbent.get("lastSeen", self.time)) > self.STALE_SCAN_TICKS)):
            incumbent = None

        best = None
        best_score = float("-inf")
        for enemy in self.enemies.values():
            if not enemy.get("alive", False) or (self.time - int(enemy.get("lastSeen", self.time)) > self.STALE_SCAN_TICKS):
                continue
            score = self._score_enemy(enemy)
            if score > best_score:
                best_score = score
                best = enemy

        if best is None:
            self.current_target_name = None
            return
        if allow_sticky_hold and incumbent is not None and self._score_enemy(incumbent) >= best_score - 0.35:
            self.current_target_name = str(incumbent["name"])
            return
        self.current_target_name = str(best["name"])

    def _score_enemy(self, enemy: dict[str, float | bool | int]) -> float:
        distance = float(enemy["distance"])
        energy = float(enemy["energy"])
        last_seen = int(enemy["lastSeen"])
        bearing = float(enemy["relBearing"])
        lateral_velocity = float(enemy.get("lateralVelocity", 0.0))
        distance_score = 1.35 * (1.0 - max(0.0, min(1.0, distance / 800.0)))
        weakness_score = 1.10 * (1.0 - max(0.0, min(1.0, energy / 100.0)))
        freshness_score = 1.00 * (1.0 - max(0.0, min(1.0, (self.time - last_seen) / self.STALE_SCAN_TICKS)))
        aim_ease_score = 0.90 * (1.0 - max(0.0, min(1.0, abs(lateral_velocity) / 8.0)))
        bearing_score = 0.45 * (1.0 - max(0.0, min(1.0, abs(bearing) / math.pi)))
        stickiness = 0.70 if enemy.get("name") == self.current_target_name else 0.0
        return distance_score + weakness_score + freshness_score + aim_ease_score + bearing_score + stickiness

    def _select_action(self, state_view: tuple[str, int, int, bool, dict[str, float] | None, dict[str, float] | None]) -> int:
        if self.random.random() < self.epsilon:
            return self._explore_action(state_view)
        values = self.__class__.SHARED_Q_TABLE.get(state_view[0])
        best = float("-inf")
        best_action = 0
        ties = 0
        for idx, value in enumerate(values):
            adjusted = value + self._action_bias(idx, state_view)
            if adjusted > best + 1e-9:
                best = adjusted
                best_action = idx
                ties = 1
            elif abs(adjusted - best) <= 1e-9 and self.random.randint(0, ties) == 0:
                best_action = idx
            if abs(adjusted - best) <= 1e-9:
                ties += 1
        return best_action

    def _explore_action(self, state_view: tuple[str, int, int, bool, dict[str, float] | None, dict[str, float] | None]) -> int:
        weights = []
        for idx in range(self.ACTION_COUNT):
            weight = max(0.15, 1.0 + self._action_bias(idx, state_view))
            weights.append(weight)
        total = sum(weights)
        draw = self.random.random() * total
        for idx, weight in enumerate(weights):
            draw -= weight
            if draw <= 0.0:
                return idx
        return self.ACTION_COUNT - 1

    def _action_bias(self, action: int, state_view: tuple[str, int, int, bool, dict[str, float] | None, dict[str, float] | None]) -> float:
        danger_bucket = state_view[1]
        wall_bucket = state_view[2]
        gun_ready = state_view[3]
        target = state_view[4]
        bias = 0.0
        target_missing = target is None or (self.time - int(target.get("lastSeen", self.time)) > 6)
        gun_aligned = target is not None and abs(_normalize_angle(math.degrees(math.atan2(float(target["x"]) - self.x, float(target["y"]) - self.y) - math.radians(self.gun_direction)))) < 11.0

        if target_missing and action in {self.RADAR_SWEEP_LEFT, self.RADAR_SWEEP_RIGHT}:
            bias += 2.7
        if danger_bucket >= 2:
            if action == self.EVADE:
                bias += 3.0
            if action == self.RETREAT_CLUSTER:
                bias += 2.4
            if action in {self.ORBIT_CLOCKWISE, self.ORBIT_COUNTERCLOCKWISE}:
                bias += 1.5
            if action == self.FIRE_HIGH:
                bias -= 0.7
        elif target is not None:
            if action in {self.ORBIT_CLOCKWISE, self.ORBIT_COUNTERCLOCKWISE}:
                bias += 1.2
            if action == self.ADVANCE_OPEN_SPACE:
                bias += 0.8
        if not gun_ready or not gun_aligned:
            if action in {self.FIRE_LOW, self.FIRE_MEDIUM, self.FIRE_HIGH}:
                bias -= 0.9
        else:
            if action in {self.FIRE_LOW, self.FIRE_MEDIUM}:
                bias += 0.9
            if action == self.FIRE_HIGH and danger_bucket <= 1:
                bias += 0.5
        if wall_bucket >= 2:
            if action in {self.ADVANCE_OPEN_SPACE, self.EVADE, self.RETREAT_CLUSTER}:
                bias += 1.8
            if action in {self.ORBIT_CLOCKWISE, self.ORBIT_COUNTERCLOCKWISE}:
                bias += 0.5
        return bias

    def _execute_action(self, action: int, state_view: tuple[str, int, int, bool, dict[str, float] | None, dict[str, float] | None]) -> None:
        target = state_view[4]
        if action == self.ORBIT_CLOCKWISE:
            self._orbit_target(target, -1)
        elif action == self.ORBIT_COUNTERCLOCKWISE:
            self._orbit_target(target, 1)
        elif action == self.RETREAT_CLUSTER:
            self._retreat_from_cluster()
        elif action == self.ADVANCE_OPEN_SPACE:
            self._advance_into_open_space()
        elif action == self.RADAR_SWEEP_LEFT:
            self.turn_radar_right(-85)
            self._track_gun(target)
        elif action == self.RADAR_SWEEP_RIGHT:
            self.turn_radar_right(85)
            self._track_gun(target)
        elif action == self.FIRE_LOW:
            self._fire_if_aligned(target, 1.0)
        elif action == self.FIRE_MEDIUM:
            self._fire_if_aligned(target, 1.8)
        elif action == self.FIRE_HIGH:
            self._fire_if_aligned(target, 2.6)
        elif action == self.EVADE:
            self._evade()
        else:
            self._advance_into_open_space()
        self._maintain_radar_lock(target)

    def _orbit_target(self, target: dict[str, float] | None, orbit_direction: int) -> None:
        if target is None:
            self._advance_into_open_space()
            return
        abs_bearing = self._absolute_bearing(self.x, self.y, float(target["x"]), float(target["y"]))
        desired = abs_bearing + orbit_direction * math.pi / 2.0
        if float(target["distance"]) < 150.0:
            desired += orbit_direction * math.pi / 10.0
        desired = self._wall_smoothed_heading(desired, 110.0, orbit_direction)
        self._set_back_as_front(desired, 120.0)
        self._track_gun(target)

    def _retreat_from_cluster(self) -> None:
        center = self._cluster_center(320.0)
        if center is not None:
            heading = self._absolute_bearing(center[0], center[1], self.x, self.y)
        else:
            nearest = self._freshest_nearest_enemy()
            heading = self._absolute_bearing(nearest['x'], nearest['y'], self.x, self.y) if nearest is not None else self._best_open_space_heading()
        heading = self._wall_smoothed_heading(heading, 150.0, 1)
        self._set_back_as_front(heading, 160.0)
        self._track_gun(self._current_target())

    def _advance_into_open_space(self) -> None:
        heading = self._wall_smoothed_heading(self._best_open_space_heading(), 130.0, 1)
        self._set_back_as_front(heading, 130.0)
        self._track_gun(self._current_target())

    def _evade(self) -> None:
        heading = self._wall_smoothed_heading(self._best_open_space_heading(), 180.0, 1 if self.random.choice([True, False]) else -1)
        self.set_max_velocity(8.0)
        self._set_back_as_front(heading, 180.0)
        target = self._current_target()
        if target is not None:
            self.turn_gun_right(math.degrees(_normalize_angle(math.atan2(float(target['x']) - self.x, float(target['y']) - self.y) - math.radians(self.gun_direction))))

    def _fire_if_aligned(self, target: dict[str, float] | None, requested_power: float) -> None:
        self.fire_actions += 1
        if target is None:
            self.turn_radar_right(90)
            return
        power = requested_power
        if self.energy < 22.0:
            power = min(power, 1.4)
        if self._compute_danger_bucket() >= 2:
            power = min(power, 1.8)
        gun_turn = self._predictive_gun_turn(target, power)
        self.turn_gun_right(math.degrees(gun_turn))
        if self.gun_heat <= 0.0001 and self.energy > power + 0.15 and abs(math.degrees(gun_turn)) < 8.0:
            self.fire(power)

    def _maintain_radar_lock(self, target: dict[str, float] | None) -> None:
        if target is not None and (self.time - int(target.get('lastSeen', self.time))) <= 4:
            radar_turn = _normalize_angle(math.degrees(target['absBearing'] - math.radians(self.radar_direction)))
            self.turn_radar_right(radar_turn * 2.0)
        else:
            self.turn_radar_right(60)

    def _track_gun(self, target: dict[str, float] | None) -> None:
        if target is None:
            return
        self.turn_gun_right(math.degrees(self._predictive_gun_turn(target, 1.8)))

    def _predictive_gun_turn(self, target: dict[str, float], power: float) -> float:
        bullet_speed = 20.0 - 3.0 * power
        fire_time = float(target['distance']) / max(11.0, bullet_speed)
        predicted_x = self._clamp(float(target['x']) + math.sin(float(target['heading'])) * float(target['velocity']) * fire_time, self.WALL_MARGIN, self.arena_width - self.WALL_MARGIN)
        predicted_y = self._clamp(float(target['y']) + math.cos(float(target['heading'])) * float(target['velocity']) * fire_time, self.WALL_MARGIN, self.arena_height - self.WALL_MARGIN)
        aim_bearing = self._absolute_bearing(self.x, self.y, predicted_x, predicted_y)
        return _normalize_angle(math.degrees(aim_bearing - math.radians(self.gun_direction)))

    def _set_back_as_front(self, go_angle: float, distance: float) -> None:
        angle = _normalize_angle(math.degrees(go_angle - math.radians(self.direction)))
        if abs(angle) > 90.0:
            if angle < 0.0:
                self.turn_right(180.0 + angle)
            else:
                self.turn_left(180.0 - angle)
            self.back(distance)
        else:
            if angle < 0.0:
                self.turn_left(-angle)
            else:
                self.turn_right(angle)
            self.forward(distance)

    def _wall_smoothed_heading(self, angle: float, distance: float, turn_direction: int) -> float:
        smoothed = angle
        for _ in range(24):
            test_x = self.x + math.sin(smoothed) * distance
            test_y = self.y + math.cos(smoothed) * distance
            if self.WALL_MARGIN < test_x < self.arena_width - self.WALL_MARGIN and self.WALL_MARGIN < test_y < self.arena_height - self.WALL_MARGIN:
                break
            smoothed += turn_direction * 0.18
        return smoothed

    def _best_open_space_heading(self) -> float:
        best_score = float("-inf")
        best_heading = math.radians(self.direction)
        for idx in range(16):
            heading = idx * (math.pi / 8.0)
            test_x = self._clamp(self.x + math.sin(heading) * 170.0, 18.0, self.arena_width - 18.0)
            test_y = self._clamp(self.y + math.cos(heading) * 170.0, 18.0, self.arena_height - 18.0)
            wall_score = min(self._min_wall_distance(test_x, test_y), 180.0) / 180.0 * 2.4
            enemy_score = 0.0
            for enemy in self.enemies.values():
                if not enemy.get('alive', False) or (self.time - int(enemy.get('lastSeen', self.time)) > self.STALE_SCAN_TICKS):
                    continue
                distance = math.dist((test_x, test_y), (float(enemy['x']), float(enemy['y'])))
                enemy_score += min(distance, 500.0) / 500.0
                if distance < 180.0:
                    enemy_score -= 1.2
            score = wall_score + enemy_score
            if score > best_score:
                best_score = score
                best_heading = heading
        return best_heading

    def _cluster_center(self, max_distance: float) -> tuple[float, float] | None:
        points = [enemy for enemy in self.enemies.values() if enemy.get('alive', False) and (self.time - int(enemy.get('lastSeen', self.time)) <= self.STALE_SCAN_TICKS) and float(enemy.get('distance', 0.0)) <= max_distance]
        if not points:
            return None
        return sum(float(enemy['x']) for enemy in points) / len(points), sum(float(enemy['y']) for enemy in points) / len(points)

    def _freshest_nearest_enemy(self) -> dict[str, float] | None:
        best = None
        best_distance = float('inf')
        for enemy in self.enemies.values():
            if not enemy.get('alive', False) or (self.time - int(enemy.get('lastSeen', self.time)) > self.STALE_SCAN_TICKS):
                continue
            if float(enemy['distance']) < best_distance:
                best_distance = float(enemy['distance'])
                best = enemy
        return best

    def _freshest_weakest_enemy(self) -> dict[str, float] | None:
        best = None
        best_energy = float('inf')
        best_distance = float('inf')
        for enemy in self.enemies.values():
            if not enemy.get('alive', False) or (self.time - int(enemy.get('lastSeen', self.time)) > self.STALE_SCAN_TICKS):
                continue
            energy = float(enemy['energy'])
            distance = float(enemy['distance'])
            if energy < best_energy - 1e-9 or (abs(energy - best_energy) <= 1e-9 and distance < best_distance):
                best_energy = energy
                best_distance = distance
                best = enemy
        return best

    def _current_target(self) -> dict[str, float] | None:
        if self.current_target_name is None:
            return None
        target = self.enemies.get(self.current_target_name)
        if target is None or not target.get('alive', False) or (self.time - int(target.get('lastSeen', self.time)) > self.STALE_SCAN_TICKS):
            self.current_target_name = None
            return None
        return target

    def _compute_danger_bucket(self) -> int:
        danger = 0.0
        for enemy in self.enemies.values():
            if not enemy.get('alive', False) or (self.time - int(enemy.get('lastSeen', self.time)) > self.STALE_SCAN_TICKS):
                continue
            distance_factor = max(0.0, 320.0 - float(enemy['distance'])) / 320.0
            energy_factor = 0.5 + max(0.0, min(1.0, float(enemy['energy']) / 100.0))
            danger += distance_factor * energy_factor
        if self._min_wall_distance() < 85.0:
            danger += 0.7
        if self.energy < 25.0:
            danger += 0.6
        if danger < 0.7:
            return 0
        if danger < 1.5:
            return 1
        if danger < 2.5:
            return 2
        return 3

    def _update_crowding_penalty(self, state_view: tuple[str, int, int, bool, dict[str, float] | None, dict[str, float] | None]) -> None:
        if state_view[1] >= 2 and self._count_nearby_enemies(220.0) >= 2:
            self.crowded_ticks += 1
            if self.crowded_ticks > 5:
                self.step_reward_accumulator -= 0.040
        else:
            self.crowded_ticks = 0

    def _count_nearby_enemies(self, radius: float) -> int:
        count = 0
        for enemy in self.enemies.values():
            if enemy.get('alive', False) and (self.time - int(enemy.get('lastSeen', self.time)) <= self.STALE_SCAN_TICKS) and float(enemy['distance']) <= radius:
                count += 1
        return count

    def _remove_stale_enemies(self) -> None:
        for enemy in self.enemies.values():
            if self.time - int(enemy.get('lastSeen', self.time)) > 80:
                enemy['alive'] = False

    def _sarsa_update(self, state: str, action: int, reward: float, next_state: str, next_action: int, done: bool) -> None:
        current_values = self.__class__.SHARED_Q_TABLE.get(state)
        next_values = self.__class__.SHARED_Q_TABLE.get(next_state)
        target = reward if done else reward + self.GAMMA * next_values[next_action]
        td_error = target - current_values[action]
        current_values[action] += self.ALPHA * td_error
        self.td_abs_sum += abs(td_error)
        self.td_count += 1

    def _finish_episode(self, won: bool) -> None:
        placement = 1 if won else self.others + 1
        total_bots = max(2, self.starting_opponent_count + 1)
        placement_bonus = 1.8 * (total_bots - placement) / (total_bots - 1)
        terminal_reward = 2.4 if won else -1.1
        self.step_reward_accumulator += terminal_reward + placement_bonus

        if self.previous_state is not None and self.previous_action is not None:
            values = self.__class__.SHARED_Q_TABLE.get(self.previous_state)
            td_error = self.step_reward_accumulator - values[self.previous_action]
            values[self.previous_action] += self.ALPHA * td_error
            self.td_abs_sum += abs(td_error)
            self.td_count += 1
            self.episode_reward += self.step_reward_accumulator

        avg_abs_td = 0.0 if self.td_count == 0 else self.td_abs_sum / self.td_count
        self._append_log(won, placement, avg_abs_td)
        self.__class__.SHARED_Q_TABLE.save(self.q_table_path)
        self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
        self.__class__.shared_epsilon = self.epsilon

    def _append_log(self, won: bool, placement: int, avg_abs_td: float) -> None:
        row = json.dumps({
            "episode": self.episode_number,
            "won": won,
            "placement": placement,
            "epsilon": round(self.epsilon, 5),
            "total_reward": round(self.episode_reward, 4),
            "avg_abs_td_error": round(avg_abs_td, 6),
            "wall_hits": self.wall_hits,
            "fire_actions": self.fire_actions,
            "turns": self.time,
        }) + "\n"
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(row)

    def on_scanned_bot(self, scanned_bot_event: ScannedBotEvent) -> None:
        abs_bearing = math.radians(self.direction) + scanned_bot_event.bearing
        self.enemies[scanned_bot_event.name] = {
            "name": scanned_bot_event.name,
            "x": self.x + math.sin(abs_bearing) * scanned_bot_event.distance,
            "y": self.y + math.cos(abs_bearing) * scanned_bot_event.distance,
            "distance": scanned_bot_event.distance,
            "absBearing": abs_bearing,
            "relBearing": scanned_bot_event.bearing,
            "energy": scanned_bot_event.energy,
            "velocity": scanned_bot_event.velocity,
            "heading": scanned_bot_event.direction,
            "lateralVelocity": scanned_bot_event.velocity * math.sin(scanned_bot_event.direction - abs_bearing),
            "lastSeen": self.time,
            "alive": True,
        }
        self._choose_target(scanned_bot_event.name, True)

    def on_robot_death(self, event) -> None:
        snapshot = self.enemies.get(event.name)
        if snapshot is not None:
            snapshot["alive"] = False
            if self.time - int(snapshot.get("lastSeen", self.time)) <= 2:
                self.step_reward_accumulator += 1.35
        if event.name == self.current_target_name:
            self.current_target_name = None

    def on_bullet_hit(self, event) -> None:
        power = float(getattr(getattr(event, "bullet", None), "power", 1.0))
        self.step_reward_accumulator += 0.070 * _bullet_damage(power)

    def on_hit_by_bullet(self, event) -> None:
        power = float(getattr(getattr(event, "bullet", None), "power", 1.0))
        self.step_reward_accumulator -= 0.095 * _bullet_damage(power)

    def on_hit_wall(self, event: HitWallEvent) -> None:
        del event
        self.wall_hits += 1
        self.step_reward_accumulator -= 0.60
        self.back(90)
        self.turn_right(_normalize_angle(120 - getattr(event, "bearing", 0.0)))

    def on_hit_robot(self, event) -> None:
        penalty = 0.25 + (0.35 if self._compute_danger_bucket() >= 2 else 0.0)
        self.step_reward_accumulator -= penalty
        if getattr(event, "is_my_fault", False):
            self.back(60)

    def on_won_round(self, event: WonRoundEvent) -> None:
        del event
        self._finish_episode(True)

    def on_death(self, death_event: DeathEvent) -> None:
        del death_event
        self._finish_episode(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SARSA Robocode bot")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--q-table-path", default=str(DEFAULT_Q_TABLE_PATH))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    args = parser.parse_args()

    bot = MeleeSarsaBot(args.alpha, args.gamma, args.epsilon, args.epsilon_decay, args.epsilon_min, args.q_table_path, args.log_path)
    bot.start()


if __name__ == "__main__":
    main()
