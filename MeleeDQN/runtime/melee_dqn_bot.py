"""Python conversion of the melee DQN Robocode Tank Royale bot."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import (
    DeathEvent,
    HitByBulletEvent,
    HitRobotEvent,
    HitWallEvent,
    ScannedBotEvent,
    WonRoundEvent,
)

from MeleeDQN.agent.dqn_agent import DQNAgent

ROOT = Path(__file__).resolve().parents[1]
STATE_SIZE = 48
ACTION_COUNT = 15
DEFAULT_WEIGHTS_PATH = ROOT / "checkpoints" / "melee_dqn_weights.pt"
DEFAULT_LOG_PATH = ROOT / "logs" / "melee_dqn_training_log.jsonl"
DEFAULT_STATE_LOG_PATH = ROOT / "logs" / "game_states.jsonl"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _wrap_radians(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _bullet_damage(power: float) -> float:
    return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


class ActionType(IntEnum):
    AHEAD_SHORT = 0
    AHEAD_MEDIUM = 1
    BACK_SHORT = 2
    BACK_MEDIUM = 3
    TURN_LEFT_SMALL = 4
    TURN_RIGHT_SMALL = 5
    TURN_LEFT_MEDIUM = 6
    TURN_RIGHT_MEDIUM = 7
    STRAFE_LEFT = 8
    STRAFE_RIGHT = 9
    HEAD_TO_OPEN_SPACE = 10
    FLEE_CLUSTER = 11
    FIRE_1 = 12
    FIRE_2 = 13
    FIRE_3 = 14

    @classmethod
    def from_id(cls, value: int) -> "ActionType":
        try:
            return cls(value)
        except ValueError:
            return cls.HEAD_TO_OPEN_SPACE

    def is_fire_action(self) -> bool:
        return self in {ActionType.FIRE_1, ActionType.FIRE_2, ActionType.FIRE_3}


@dataclass
class BattleStats:
    episode: int = 0
    tick: int = 0
    livingEnemies: int = 0
    placement: int = 0
    survivalTicks: int = 0
    damageDealt: float = 0.0
    damageTaken: float = 0.0
    kills: int = 0
    targetSwitches: int = 0


@dataclass
class EnemySnapshot:
    name: str
    x: float = 0.0
    y: float = 0.0
    distance: float = 0.0
    bearing_radians: float = 0.0
    heading_radians: float = 0.0
    velocity: float = 0.0
    energy: float = 100.0
    last_seen_tick: int = 0
    last_energy_drop: float = 0.0

    def update(self, robot: Bot, event: ScannedBotEvent, tick: int) -> None:
        previous_energy = self.energy
        absolute_bearing = math.radians(robot.direction) + event.bearing
        self.distance = event.distance
        self.bearing_radians = event.bearing
        self.heading_radians = event.direction
        self.velocity = event.velocity
        self.energy = event.energy
        self.x = robot.x + math.sin(absolute_bearing) * self.distance
        self.y = robot.y + math.cos(absolute_bearing) * self.distance
        self.last_seen_tick = tick
        self.last_energy_drop = max(0.0, previous_energy - self.energy) if previous_energy > 0.0 else 0.0

    def age(self, current_tick: int) -> int:
        return max(0, current_tick - self.last_seen_tick)

    def absolute_bearing_from(self, robot: Bot) -> float:
        return math.atan2(self.x - robot.x, self.y - robot.y)

    def gun_turn_from(self, robot: Bot) -> float:
        return _wrap_radians(self.absolute_bearing_from(robot) - math.radians(robot.gun_direction))

    def heading_toward_robot_error(self, robot: Bot) -> float:
        to_robot = math.atan2(robot.x - self.x, robot.y - self.y)
        return abs(_wrap_radians(to_robot - self.heading_radians))


class EnemyManager:
    def __init__(self) -> None:
        self.enemies: dict[str, EnemySnapshot] = {}

    def update(self, robot: Bot, event: ScannedBotEvent, tick: int) -> None:
        snapshot = self.enemies.get(event.name)
        if snapshot is None:
            snapshot = EnemySnapshot(event.name)
            self.enemies[event.name] = snapshot
        snapshot.update(robot, event, tick)

    def on_robot_death(self, event) -> None:
        self.enemies.pop(event.name, None)

    def reset_round(self) -> None:
        self.enemies.clear()

    def all(self) -> list[EnemySnapshot]:
        return list(self.enemies.values())

    def count(self) -> int:
        return len(self.enemies)

    def get(self, name: str | None) -> EnemySnapshot | None:
        if name is None:
            return None
        return self.enemies.get(name)

    def nearest(self) -> EnemySnapshot | None:
        return min(self.enemies.values(), key=lambda enemy: enemy.distance, default=None)

    def weakest(self) -> EnemySnapshot | None:
        return min(self.enemies.values(), key=lambda enemy: enemy.energy, default=None)

    def stalest(self, current_tick: int) -> EnemySnapshot | None:
        return max(self.enemies.values(), key=lambda enemy: enemy.age(current_tick), default=None)

    def closest_distance(self) -> float:
        nearest = self.nearest()
        return 0.0 if nearest is None else nearest.distance

    def average_distance(self) -> float:
        if not self.enemies:
            return 0.0
        return sum(enemy.distance for enemy in self.enemies.values()) / len(self.enemies)


class DangerMap:
    WALL_MARGIN = 80.0

    def crowding_score(self, robot: Bot, enemies: list[EnemySnapshot]) -> float:
        if not enemies:
            return 0.0
        danger = 0.0
        for enemy in enemies:
            distance = max(36.0, enemy.distance)
            danger += 1.0 / (distance * distance)
        return _clamp01(danger * 25000.0)

    def count_within(self, enemies: list[EnemySnapshot], range_: float) -> int:
        return sum(1 for enemy in enemies if enemy.distance <= range_)

    def safest_heading(self, robot: Bot, enemies: list[EnemySnapshot]) -> float:
        best_heading = math.radians(robot.direction)
        best_score = float("-inf")
        for i in range(32):
            heading = (2.0 * math.pi * i) / 32.0
            score = self._heading_score(robot, enemies, heading)
            if score > best_score:
                best_score = score
                best_heading = heading
        return best_heading

    def escape_heading(self, robot: Bot, enemies: list[EnemySnapshot]) -> float:
        if not enemies:
            return self.safest_heading(robot, enemies)

        vx = 0.0
        vy = 0.0
        for enemy in enemies:
            dx = robot.x - enemy.x
            dy = robot.y - enemy.y
            dist_sq = max(1600.0, dx * dx + dy * dy)
            vx += dx / dist_sq
            vy += dy / dist_sq

        vx += self._wall_repulsion(robot.x, robot.arena_width)
        vy += self._wall_repulsion(robot.y, robot.arena_height)
        return math.atan2(vx, vy)

    def forward_unsafe(self, robot: Bot) -> bool:
        heading = math.radians(robot.direction)
        next_x = robot.x + math.sin(heading) * 120.0
        next_y = robot.y + math.cos(heading) * 120.0
        return self._near_wall(next_x, next_y, robot)

    def _heading_score(self, robot: Bot, enemies: list[EnemySnapshot], heading: float) -> float:
        probe_x = robot.x + math.sin(heading) * 140.0
        probe_y = robot.y + math.cos(heading) * 140.0
        if self._near_wall(probe_x, probe_y, robot):
            return -10.0

        score = 0.0
        for enemy in enemies:
            dx = probe_x - enemy.x
            dy = probe_y - enemy.y
            dist_sq = max(1600.0, dx * dx + dy * dy)
            score += math.log(dist_sq)
        turn_penalty = abs(_wrap_radians(heading - math.radians(robot.direction)))
        return score - 0.75 * turn_penalty

    def _near_wall(self, x: float, y: float, robot: Bot) -> bool:
        return (
            x < self.WALL_MARGIN
            or y < self.WALL_MARGIN
            or x > robot.arena_width - self.WALL_MARGIN
            or y > robot.arena_height - self.WALL_MARGIN
        )

    def _wall_repulsion(self, value: float, max_value: float) -> float:
        low = max(1.0, value - self.WALL_MARGIN)
        high = max(1.0, max_value - self.WALL_MARGIN - value)
        return (1.0 / low) - (1.0 / high)


class TargetSelector:
    SWITCH_MARGIN = 0.18
    MIN_TARGET_HOLD_TICKS = 10

    def __init__(self) -> None:
        self.current_target_name: str | None = None
        self.last_switch_tick = 0
        self.switch_count = 0

    def select(self, robot: Bot, enemies: list[EnemySnapshot], tick: int) -> EnemySnapshot | None:
        best = None
        best_score = float("-inf")
        for enemy in enemies:
            score = self.score(robot, enemy, tick)
            if score > best_score:
                best_score = score
                best = enemy

        current = next((enemy for enemy in enemies if enemy.name == self.current_target_name), None)
        if current is None and best is not None:
            self.current_target_name = best.name
            self.last_switch_tick = tick
            return best
        if current is None:
            return None
        if best is None:
            self.current_target_name = None
            return None

        current_score = self.score(robot, current, tick)
        should_switch = (
            best.name != current.name
            and best_score > current_score + self.SWITCH_MARGIN
            and tick - self.last_switch_tick >= self.MIN_TARGET_HOLD_TICKS
        )
        if should_switch:
            self.current_target_name = best.name
            self.last_switch_tick = tick
            self.switch_count += 1
            return best
        return current

    def score(self, robot: Bot, enemy: EnemySnapshot, tick: int) -> float:
        distance_score = 1.0 - _clamp01(enemy.distance / 900.0)
        low_energy_score = 1.0 - _clamp01(enemy.energy / 100.0)
        freshness_score = 1.0 - _clamp01(enemy.age(tick) / 40.0)
        gun_ease_score = 1.0 - _clamp01(abs(enemy.gun_turn_from(robot)) / math.pi)

        fire_intent_score = 0.0
        if 0.1 <= enemy.last_energy_drop <= 3.0:
            fire_intent_score += 0.55
        if enemy.heading_toward_robot_error(robot) < math.radians(22.0):
            fire_intent_score += 0.45

        score = (
            0.34 * distance_score
            + 0.22 * low_energy_score
            + 0.15 * freshness_score
            + 0.15 * fire_intent_score
            + 0.14 * gun_ease_score
        )

        if enemy.name == self.current_target_name:
            score += 0.08
        return score

    def most_threatening(self, robot: Bot, enemies: list[EnemySnapshot], tick: int) -> EnemySnapshot | None:
        best = None
        best_threat = float("-inf")
        for enemy in enemies:
            threat = 0.55 * (1.0 - _clamp01(enemy.distance / 750.0)) + 0.25 * _clamp01(enemy.energy / 100.0) + 0.20 * _clamp01(self.score(robot, enemy, tick))
            if threat > best_threat:
                best_threat = threat
                best = enemy
        return best

    def reset_round(self) -> None:
        self.current_target_name = None
        self.last_switch_tick = 0
        self.switch_count = 0


class RewardTracker:
    DAMAGE_DEALT_COEF = 0.045
    DAMAGE_TAKEN_COEF = -0.06
    KILL_COEF = 1.2
    SURVIVAL_TICK_COEF = 0.003
    WALL_HIT_COEF = -0.18
    ROBOT_COLLISION_COEF = -0.14
    DENSE_ZONE_COEF = -0.08
    INACTIVITY_COEF = -0.02
    TARGET_SWITCH_COEF = -0.03
    FIRE_COST_COEF = -0.005
    ALIVE_FINISH_BONUS = 1.0

    def __init__(self) -> None:
        self.reset_round()

    def reset_round(self) -> None:
        self.pending_reward = 0.0
        self.total_reward = 0.0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.kills = 0
        self.survival_ticks = 0
        self.recently_hit_ticks = 0
        self.repeated_still_ticks = 0
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_heading = 0.0
        self.initialized = False

    def on_tick(self, robot: Bot, crowding_score: float, target_switches: int) -> None:
        self.survival_ticks += 1
        self.pending_reward += self.SURVIVAL_TICK_COEF
        self.pending_reward += self.DENSE_ZONE_COEF * crowding_score
        self.pending_reward += self.TARGET_SWITCH_COEF * target_switches

        if self.recently_hit_ticks > 0:
            self.recently_hit_ticks -= 1

        if not self.initialized:
            self.initialized = True
            self.last_x = robot.x
            self.last_y = robot.y
            self.last_heading = math.radians(robot.direction)
            return

        moved = math.hypot(robot.x - self.last_x, robot.y - self.last_y)
        turn = abs(_wrap_radians(math.radians(robot.direction) - self.last_heading))
        if moved < 4.0 and turn > 0.6:
            self.repeated_still_ticks += 1
        elif moved > 8.0:
            self.repeated_still_ticks = max(0, self.repeated_still_ticks - 2)
        else:
            self.repeated_still_ticks = max(0, self.repeated_still_ticks - 1)

        if self.repeated_still_ticks >= 6:
            self.pending_reward += self.INACTIVITY_COEF

        self.last_x = robot.x
        self.last_y = robot.y
        self.last_heading = math.radians(robot.direction)

    def on_bullet_damage_dealt(self, damage: float) -> None:
        self.damage_dealt += damage
        self.pending_reward += self.DAMAGE_DEALT_COEF * damage

    def on_bullet_damage_taken(self, damage: float) -> None:
        self.damage_taken += damage
        self.pending_reward += self.DAMAGE_TAKEN_COEF * damage
        self.recently_hit_ticks = 8

    def on_kill(self) -> None:
        self.kills += 1
        self.pending_reward += self.KILL_COEF

    def on_hit_wall(self) -> None:
        self.pending_reward += self.WALL_HIT_COEF

    def on_robot_collision(self) -> None:
        self.pending_reward += self.ROBOT_COLLISION_COEF

    def on_fire_command(self) -> None:
        self.pending_reward += self.FIRE_COST_COEF

    def consume_step_reward(self) -> float:
        clipped = max(-2.0, min(2.0, self.pending_reward))
        self.total_reward += clipped
        self.pending_reward = 0.0
        return clipped

    def finish_round(self, placement: int, total_bots: int, alive_at_end: bool) -> float:
        placement_ratio = 1.0 if total_bots <= 1 else 1.0 - ((placement - 1.0) / (total_bots - 1.0))
        final_reward = (1.4 * placement_ratio) + (self.ALIVE_FINISH_BONUS if alive_at_end else 0.0)
        self.pending_reward += final_reward
        return self.consume_step_reward()

    def was_recently_hit(self) -> bool:
        return self.recently_hit_ticks > 0


class StateEncoder:
    ENEMY_BLOCK_SIZE = 8
    GLOBAL_FEATURE_COUNT = 16
    STATE_SIZE = GLOBAL_FEATURE_COUNT + (ENEMY_BLOCK_SIZE * 4)

    def encode(self, robot: Bot, enemy_manager: EnemyManager, selector: TargetSelector, danger_map: DangerMap, reward_tracker: RewardTracker, tick: int) -> np.ndarray:
        state = np.zeros(self.STATE_SIZE, dtype=np.float32)
        field_width = max(1.0, robot.arena_width)
        field_height = max(1.0, robot.arena_height)
        max_distance = math.hypot(field_width, field_height)

        close_count = danger_map.count_within(enemy_manager.all(), 200.0)
        medium_count = danger_map.count_within(enemy_manager.all(), 400.0) - close_count
        far_count = max(0, enemy_manager.count() - close_count - medium_count)

        state[0] = _clamp01(robot.energy / 100.0)
        state[1] = _clamp_signed(robot.x / field_width)
        state[2] = _clamp_signed(robot.y / field_height)
        state[3] = _clamp01(robot.x / field_width)
        state[4] = _clamp01((field_width - robot.x) / field_width)
        state[5] = _clamp01(robot.y / field_height)
        state[6] = _clamp01((field_height - robot.y) / field_height)
        state[7] = _clamp01(float(getattr(robot, "enemy_count", 0)) / 12.0)
        state[8] = _clamp01(enemy_manager.closest_distance() / max_distance)
        state[9] = _clamp01(enemy_manager.average_distance() / max_distance)
        state[10] = 1.0 if reward_tracker.was_recently_hit() else 0.0
        state[11] = _clamp01(robot.gun_heat / 1.6)
        state[12] = danger_map.crowding_score(robot, enemy_manager.all())
        state[13] = _clamp01(close_count / 10.0)
        state[14] = _clamp01(medium_count / 10.0)
        state[15] = _clamp01(far_count / 10.0)

        nearest = enemy_manager.nearest()
        weakest = enemy_manager.weakest()
        threatening = selector.most_threatening(robot, enemy_manager.all(), tick)
        current_target = enemy_manager.get(selector.current_target_name)

        self._write_enemy_block(state, self.GLOBAL_FEATURE_COUNT, robot, nearest, tick, max_distance)
        self._write_enemy_block(state, self.GLOBAL_FEATURE_COUNT + self.ENEMY_BLOCK_SIZE, robot, weakest, tick, max_distance)
        self._write_enemy_block(state, self.GLOBAL_FEATURE_COUNT + (self.ENEMY_BLOCK_SIZE * 2), robot, threatening, tick, max_distance)
        self._write_enemy_block(state, self.GLOBAL_FEATURE_COUNT + (self.ENEMY_BLOCK_SIZE * 3), robot, current_target, tick, max_distance)
        return state

    def _write_enemy_block(self, state: np.ndarray, offset: int, robot: Bot, enemy: EnemySnapshot | None, tick: int, max_distance: float) -> None:
        state[offset : offset + self.ENEMY_BLOCK_SIZE] = 0.0
        if enemy is None:
            return
        absolute_bearing = enemy.absolute_bearing_from(robot)
        relative_bearing = _wrap_radians(absolute_bearing - math.radians(robot.direction))
        state[offset] = math.sin(relative_bearing)
        state[offset + 1] = math.cos(relative_bearing)
        state[offset + 2] = _clamp01(enemy.distance / max_distance)
        state[offset + 3] = math.sin(enemy.heading_radians)
        state[offset + 4] = math.cos(enemy.heading_radians)
        state[offset + 5] = _clamp_signed(enemy.velocity / 8.0)
        state[offset + 6] = _clamp01(enemy.energy / 100.0)
        state[offset + 7] = _clamp01(min(enemy.age(tick), 40) / 40.0)


class SocketDqnClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.socket: socket.socket | None = None
        self.out = None
        self.infile = None

    def connect_if_needed(self) -> None:
        if self.socket is not None:
            try:
                self.socket.getpeername()
                return
            except OSError:
                self.close()

        self.socket = socket.create_connection((self.host, self.port), timeout=2.0)
        self.out = self.socket.makefile("w", encoding="utf-8", newline="\n")
        self.infile = self.socket.makefile("r", encoding="utf-8", newline="\n")

    def close(self) -> None:
        try:
            if self.out is not None:
                self.out.close()
        finally:
            self.out = None
        try:
            if self.infile is not None:
                self.infile.close()
        finally:
            self.infile = None
        try:
            if self.socket is not None:
                self.socket.close()
        finally:
            self.socket = None

    def request_action(self, state: np.ndarray, reward: float, done: bool, stats: BattleStats) -> int:
        if self.socket is None or self.out is None or self.infile is None:
            raise OSError("socket not connected")

        payload = [
            "STEP",
            f"{reward:.6f}",
            "1" if done else "0",
            str(stats.episode),
            str(stats.tick),
            str(stats.livingEnemies),
            str(stats.placement),
            str(stats.survivalTicks),
            f"{stats.damageDealt:.6f}",
            f"{stats.damageTaken:.6f}",
            str(stats.kills),
            str(stats.targetSwitches),
            ",".join(f"{value:.6f}" for value in state),
        ]
        self.out.write("|".join(payload) + "\n")
        self.out.flush()
        response = self.infile.readline()
        if not response or not response.startswith("ACTION|"):
            return int(ActionType.HEAD_TO_OPEN_SPACE)
        try:
            return int(response.split("|", 1)[1].strip())
        except ValueError:
            return int(ActionType.HEAD_TO_OPEN_SPACE)


class MeleeDqnBot(Bot):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        gamma: float = 0.985,
        eps_start: float = 0.95,
        eps_end: float = 0.05,
        eps_decay_steps: int = 15000,
        tau: float = 0.005,
        batch_size: int = 128,
        memory_capacity: int = 50000,
        weights_path: str = str(DEFAULT_WEIGHTS_PATH),
        log_path: str = str(DEFAULT_LOG_PATH),
        state_log_path: str | None = None,
        socket_host: str = "127.0.0.1",
        socket_port: int = 5000,
        eval_mode: bool = False,
        eval_epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.state_log_path = state_log_path
        self.eval_mode = eval_mode
        self.episode_number = 0
        self.local_tick = 0
        self.initial_bot_count = 0
        self.last_damage_tick = -999
        self.last_applied_switch_count = 0
        self.last_action = ActionType.HEAD_TO_OPEN_SPACE
        self.last_damaged_enemy_name: str | None = None
        self.prev_state: np.ndarray | None = None
        self.prev_action: int | None = None
        self.step_reward = 0.0
        self.episode_reward = 0.0

        self.enemy_manager = EnemyManager()
        self.target_selector = TargetSelector()
        self.danger_map = DangerMap()
        self.state_encoder = StateEncoder()
        self.reward_tracker = RewardTracker()
        self.socket_client = SocketDqnClient(socket_host, socket_port)
        self.local_agent = DQNAgent(
            n_observations=STATE_SIZE,
            n_actions=ACTION_COUNT,
            learning_rate=learning_rate,
            gamma=gamma,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            tau=tau,
            batch_size=batch_size,
            memory_capacity=memory_capacity,
            weights_path=weights_path,
        )
        if self.eval_mode:
            self.local_agent.set_eval_mode(epsilon=eval_epsilon)
        else:
            self.local_agent.set_train_mode()
        self._socket_ready = False

    def run(self) -> None:
        self.episode_number += 1
        self.initial_bot_count = int(getattr(self, "enemy_count", 0)) + 1
        self.local_tick = 0
        self.last_damage_tick = -999
        self.last_applied_switch_count = 0
        self.last_action = ActionType.HEAD_TO_OPEN_SPACE
        self.last_damaged_enemy_name = None
        self.prev_state = None
        self.prev_action = None
        self.step_reward = 0.0
        self.episode_reward = 0.0

        self.enemy_manager.reset_round()
        self.target_selector.reset_round()
        self.reward_tracker.reset_round()

        self.set_adjust_gun_for_robot_turn(True)
        self.set_adjust_radar_for_gun_turn(True)
        self.set_adjust_radar_for_robot_turn(True)
        self.set_max_velocity(8.0)

        self._socket_ready = self._try_connect_socket()
        print(
            f"[MeleeDqnBot] Episode {self.episode_number} mode={'eval' if self.eval_mode else 'train'} "
            f"socket={'on' if self._socket_ready else 'off'}"
        )

        while self.running:
            self.local_tick += 1
            target = self.target_selector.select(self, self.enemy_manager.all(), self.local_tick)
            switch_delta = max(0, self.target_selector.switch_count - self.last_applied_switch_count)
            self.last_applied_switch_count = self.target_selector.switch_count
            self.reward_tracker.on_tick(self, self.danger_map.crowding_score(self, self.enemy_manager.all()), switch_delta)

            state = self.state_encoder.encode(
                self,
                self.enemy_manager,
                self.target_selector,
                self.danger_map,
                self.reward_tracker,
                self.local_tick,
            )

            action_id = self._request_action(state)
            action = self._sanitize_action(ActionType.from_id(action_id), target)
            self._aim_gun_at(target)
            self._execute_action(action, target)
            self._update_radar(target)
            self.execute()
            self.last_action = action
            if not self._socket_ready:
                self.prev_state = state
                self.prev_action = int(action)
            self._append_state_snapshot("tick")

    def on_scanned_bot(self, event: ScannedBotEvent) -> None:
        self.enemy_manager.update(self, event, self.local_tick)

    def on_bot_death(self, event) -> None:
        self.enemy_manager.on_robot_death(event)
        if event.name == self.last_damaged_enemy_name and (self.local_tick - self.last_damage_tick) <= 2:
            self.reward_tracker.on_kill()

    def on_bullet_hit(self, event) -> None:
        self.last_damaged_enemy_name = event.name
        self.last_damage_tick = self.local_tick
        self.reward_tracker.on_bullet_damage_dealt(_bullet_damage(event.bullet.power))

    def on_hit_by_bullet(self, event: HitByBulletEvent) -> None:
        self.reward_tracker.on_bullet_damage_taken(_bullet_damage(event.bullet.power))

    def on_hit_wall(self, event: HitWallEvent) -> None:
        self.reward_tracker.on_hit_wall()

    def on_hit_robot(self, event: HitRobotEvent) -> None:
        self.reward_tracker.on_robot_collision()

    def on_death(self, event: DeathEvent) -> None:
        self._send_terminal(False)

    def on_won_round(self, event: WonRoundEvent) -> None:
        self._send_terminal(True)

    def _try_connect_socket(self) -> bool:
        try:
            self.socket_client.connect_if_needed()
            return True
        except OSError:
            return False

    def _request_action(self, state: np.ndarray) -> int:
        stats = self._build_stats(placement=0)
        reward = self.reward_tracker.consume_step_reward()
        self.step_reward += reward
        self.episode_reward += reward
        if self._socket_ready:
            try:
                return self.socket_client.request_action(state, reward, False, stats)
            except OSError:
                self._socket_ready = False

        if self.prev_state is not None and self.prev_action is not None:
            self.local_agent.push_transition(self.prev_state, self.prev_action, state, reward, False)
        return self.local_agent.select_action(state)

    def _build_stats(self, placement: int) -> BattleStats:
        return BattleStats(
            episode=self.episode_number,
            tick=self.local_tick,
            livingEnemies=int(getattr(self, "enemy_count", 0)),
            placement=placement,
            survivalTicks=self.reward_tracker.survival_ticks,
            damageDealt=self.reward_tracker.damage_dealt,
            damageTaken=self.reward_tracker.damage_taken,
            kills=self.reward_tracker.kills,
            targetSwitches=self.target_selector.switch_count,
        )

    def _send_terminal(self, won: bool) -> None:
        enemy_count = int(getattr(self, "enemy_count", 0))
        total_bots = max(self.initial_bot_count, enemy_count + 1)
        placement = 1 if won else enemy_count + 1
        terminal_reward = self.reward_tracker.finish_round(placement, total_bots, won)
        terminal_state = np.zeros(STATE_SIZE, dtype=np.float32)
        stats = self._build_stats(placement)
        self.step_reward += terminal_reward
        self.episode_reward += terminal_reward

        if self._socket_ready:
            try:
                self.socket_client.request_action(terminal_state, terminal_reward, True, stats)
            except OSError:
                pass
        elif self.prev_state is not None and self.prev_action is not None:
            self.local_agent.push_transition(self.prev_state, self.prev_action, terminal_state, terminal_reward, True)

        if not self.eval_mode:
            self.local_agent.on_episode_end(won)
        self._append_log(
            {
                "episode": self.episode_number,
                "won": won,
                "placement": placement,
                "total_bots": total_bots,
                "steps": self.local_tick,
                "survival_ticks": self.reward_tracker.survival_ticks,
                "kills": self.reward_tracker.kills,
                "damage_dealt": round(self.reward_tracker.damage_dealt, 3),
                "damage_taken": round(self.reward_tracker.damage_taken, 3),
                "target_switches": self.target_selector.switch_count,
                "total_reward": round(self.episode_reward, 3),
                "epsilon": round(self.local_agent.current_epsilon(), 4),
                "loss": self.local_agent.last_loss,
                "socket_enabled": self._socket_ready,
                "mode": "eval" if self.eval_mode else "train",
            }
        )
        self._append_state_snapshot("terminal", done=True, placement=placement, won=won)

    def _append_log(self, row: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except Exception as exc:
            print(f"[MeleeDqnBot] Log failed: {exc}")

    def _append_state_snapshot(
        self,
        trigger: str,
        done: bool = False,
        placement: int | None = None,
        won: bool | None = None,
    ) -> None:
        if not self.state_log_path:
            return
        enemies = []
        for enemy in self.enemy_manager.all():
            enemies.append(
                {
                    "name": enemy.name,
                    "x": round(enemy.x, 3),
                    "y": round(enemy.y, 3),
                    "distance": round(enemy.distance, 3),
                    "bearing_radians": round(enemy.bearing_radians, 6),
                    "heading_radians": round(enemy.heading_radians, 6),
                    "velocity": round(enemy.velocity, 3),
                    "energy": round(enemy.energy, 3),
                    "age_ticks": enemy.age(self.local_tick),
                }
            )
        row = {
            "episode": self.episode_number,
            "tick": self.local_tick,
            "trigger": trigger,
            "done": done,
            "won": won,
            "placement": placement,
            "mode": "eval" if self.eval_mode else "train",
            "bot": "MeleeDQN",
            "position": {"x": round(float(self.x), 3), "y": round(float(self.y), 3)},
            "heading": round(float(self.direction), 3),
            "gun_heading": round(float(self.gun_direction), 3),
            "radar_heading": round(float(self.radar_direction), 3),
            "velocity": round(float(getattr(self, "speed", 0.0)), 3),
            "energy": round(float(self.energy), 3),
            "others": int(getattr(self, "enemy_count", 0)),
            "enemies": enemies,
            "damage_dealt": round(self.reward_tracker.damage_dealt, 3),
            "damage_taken": round(self.reward_tracker.damage_taken, 3),
            "kills": self.reward_tracker.kills,
            "target_switches": self.target_selector.switch_count,
            "wall_distances": {
                "left": round(float(self.x), 3),
                "right": round(float(max(0.0, self.arena_width - self.x)), 3),
                "bottom": round(float(self.y), 3),
                "top": round(float(max(0.0, self.arena_height - self.y)), 3),
            },
        }
        try:
            os.makedirs(os.path.dirname(self.state_log_path) or ".", exist_ok=True)
            with open(self.state_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except Exception as exc:
            print(f"[MeleeDqnBot] State log failed: {exc}")

    def _execute_action(self, action: ActionType, target: EnemySnapshot | None) -> None:
        if action == ActionType.AHEAD_SHORT:
            self.forward(80)
        elif action == ActionType.AHEAD_MEDIUM:
            self.forward(160)
        elif action == ActionType.BACK_SHORT:
            self.back(80)
        elif action == ActionType.BACK_MEDIUM:
            self.back(160)
        elif action == ActionType.TURN_LEFT_SMALL:
            self.turn_left(15)
            self.forward(40)
        elif action == ActionType.TURN_RIGHT_SMALL:
            self.turn_right(15)
            self.forward(40)
        elif action == ActionType.TURN_LEFT_MEDIUM:
            self.turn_left(35)
            self.forward(50)
        elif action == ActionType.TURN_RIGHT_MEDIUM:
            self.turn_right(35)
            self.forward(50)
        elif action == ActionType.STRAFE_LEFT:
            self._strafe_target(target, -1.0)
        elif action == ActionType.STRAFE_RIGHT:
            self._strafe_target(target, 1.0)
        elif action == ActionType.HEAD_TO_OPEN_SPACE:
            self._move_toward_heading(self.danger_map.safest_heading(self, self.enemy_manager.all()), 140.0)
        elif action == ActionType.FLEE_CLUSTER:
            self._move_toward_heading(self.danger_map.escape_heading(self, self.enemy_manager.all()), 150.0)
        elif action == ActionType.FIRE_1:
            self._fire_if_aligned(target, 1.0)
        elif action == ActionType.FIRE_2:
            self._fire_if_aligned(target, 2.0)
        elif action == ActionType.FIRE_3:
            self._fire_if_aligned(target, 3.0)

    def _sanitize_action(self, action: ActionType, target: EnemySnapshot | None) -> ActionType:
        if action.is_fire_action():
            if target is None or self.gun_heat > 0.0 or abs(target.gun_turn_from(self)) > math.radians(14.0):
                return ActionType.FLEE_CLUSTER if self.danger_map.crowding_score(self, self.enemy_manager.all()) > 0.45 else ActionType.HEAD_TO_OPEN_SPACE
            return action

        if self.danger_map.forward_unsafe(self) and action in {ActionType.AHEAD_SHORT, ActionType.AHEAD_MEDIUM}:
            return ActionType.FLEE_CLUSTER

        if action in {ActionType.STRAFE_LEFT, ActionType.STRAFE_RIGHT} and target is None:
            return ActionType.HEAD_TO_OPEN_SPACE

        return action

    def _aim_gun_at(self, target: EnemySnapshot | None) -> None:
        if target is None:
            return
        bullet_power = 1.8
        bullet_speed = 20.0 - (3.0 * bullet_power)
        time_to_target = target.distance / bullet_speed if bullet_speed > 0.0 else 0.0
        future_x = target.x + math.sin(target.heading_radians) * target.velocity * time_to_target
        future_y = target.y + math.cos(target.heading_radians) * target.velocity * time_to_target
        future_x = max(18.0, min(self.arena_width - 18.0, future_x))
        future_y = max(18.0, min(self.arena_height - 18.0, future_y))
        aim_bearing = math.atan2(future_x - self.x, future_y - self.y)
        self.turn_gun_left(math.degrees(_wrap_radians(aim_bearing - math.radians(self.gun_direction))))

    def _strafe_target(self, target: EnemySnapshot | None, direction_sign: float) -> None:
        if target is None:
            self._move_toward_heading(self.danger_map.safest_heading(self, self.enemy_manager.all()), 100.0)
            return
        abs_bearing = target.absolute_bearing_from(self)
        desired_heading = abs_bearing + (direction_sign * (math.pi / 2.0))
        desired_heading += (0.35 if target.distance < 180.0 else 0.0) * direction_sign
        self._move_toward_heading(desired_heading, 120.0 if target.distance < 180.0 else 80.0)

    def _move_toward_heading(self, desired_heading: float, distance: float) -> None:
        angle = _wrap_radians(desired_heading - math.radians(self.direction))
        if abs(angle) > math.pi / 2.0:
            reverse_angle = _wrap_radians(angle + math.pi)
            self.turn_right(math.degrees(reverse_angle))
            self.back(distance)
        else:
            self.turn_right(math.degrees(angle))
            self.forward(distance)

    def _fire_if_aligned(self, target: EnemySnapshot | None, power: float) -> None:
        if target is None:
            self._move_toward_heading(self.danger_map.safest_heading(self, self.enemy_manager.all()), 100.0)
            return

        self.reward_tracker.on_fire_command()
        self._aim_gun_at(target)
        if self.gun_heat == 0.0 and self.energy > power + 0.2 and abs(target.gun_turn_from(self)) < math.radians(8.0):
            self.fire(power)
        else:
            self._strafe_target(target, -1.0 if self.last_action == ActionType.STRAFE_LEFT else 1.0)

    def _update_radar(self, target: EnemySnapshot | None) -> None:
        stale_enemy = self.enemy_manager.stalest(self.local_tick)
        radar_focus = stale_enemy if stale_enemy is not None and stale_enemy.age(self.local_tick) > 6 else target
        if radar_focus is None:
            self.turn_radar_right(float("inf"))
            return
        absolute_bearing = radar_focus.absolute_bearing_from(self)
        radar_turn = _wrap_radians(absolute_bearing - math.radians(self.radar_direction))
        overshoot = math.copysign(math.radians(22.0), radar_turn if radar_turn != 0.0 else 1.0)
        self.turn_radar_right(math.degrees(radar_turn + overshoot))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MeleeDQN bot")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--state-log-path", default="")
    parser.add_argument("--socket-host", default="127.0.0.1")
    parser.add_argument("--socket-port", type=int, default=5000)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    args = parser.parse_args()

    MeleeDqnBot(
        weights_path=args.weights_path,
        log_path=args.log_path,
        state_log_path=args.state_log_path or None,
        socket_host=args.socket_host,
        socket_port=args.socket_port,
        eval_mode=args.eval,
        eval_epsilon=args.eval_epsilon,
    ).start()


if __name__ == "__main__":
    main()
