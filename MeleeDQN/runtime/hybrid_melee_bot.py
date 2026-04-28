"""Python conversion of the hybrid heuristic melee bot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import BulletHitEvent, HitByBulletEvent, HitRobotEvent, HitWallEvent, ScannedBotEvent, WonRoundEvent, DeathEvent


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class EnemySnapshot:
    name: str
    x: float
    y: float
    energy: float
    heading_radians: float
    velocity: float
    distance: float
    absolute_bearing_radians: float
    last_seen_tick: int
    alive: bool = True

    def age(self, current_tick: int) -> int:
        return max(0, current_tick - self.last_seen_tick)

    def mark_dead(self) -> "EnemySnapshot":
        return EnemySnapshot(self.name, self.x, self.y, self.energy, self.heading_radians, self.velocity, self.distance, self.absolute_bearing_radians, self.last_seen_tick, False)


class TacticalMode(Enum):
    SURVIVE = 1
    ENGAGE = 2
    REPOSITION = 3
    FINISH_WEAK_TARGET = 4
    ESCAPE_CROWD = 5


@dataclass
class DangerMap:
    rows: int
    cols: int
    battlefield_width: float
    battlefield_height: float

    def __post_init__(self) -> None:
        self.cell_width = self.battlefield_width / self.cols
        self.cell_height = self.battlefield_height / self.rows
        self.danger = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]

    def add_danger(self, row: int, col: int, value: float) -> None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.danger[row][col] += value

    def get_danger_at(self, x: float, y: float) -> float:
        col = min(self.cols - 1, max(0, int(x / self.cell_width)))
        row = min(self.rows - 1, max(0, int(y / self.cell_height)))
        return self.danger[row][col]

    def pick_safest_cell_center(self, robot_x: float, robot_y: float) -> tuple[float, float]:
        best_danger = float("inf")
        best_point = (robot_x, robot_y)
        for row in range(self.rows):
            for col in range(self.cols):
                x = (col + 0.5) * self.cell_width
                y = (row + 0.5) * self.cell_height
                distance_penalty = math.dist((robot_x, robot_y), (x, y)) * 0.0025
                score = self.danger[row][col] + distance_penalty
                if score < best_danger:
                    best_danger = score
                    best_point = (x, y)
        return best_point


@dataclass
class EnemyTracker:
    enemies: dict[str, EnemySnapshot] = None

    def __post_init__(self) -> None:
        if self.enemies is None:
            self.enemies = {}

    def on_scanned_robot(self, robot: Bot, event: ScannedBotEvent) -> None:
        absolute_bearing = math.radians(robot.direction) + event.bearing
        self.enemies[event.name] = EnemySnapshot(event.name, robot.x + math.sin(absolute_bearing) * event.distance, robot.y + math.cos(absolute_bearing) * event.distance, event.energy, event.direction, event.velocity, event.distance, absolute_bearing, robot.time, True)

    def on_robot_death(self, event) -> None:
        snapshot = self.enemies.get(event.name)
        if snapshot is not None:
            self.enemies[event.name] = snapshot.mark_dead()

    def get_alive_enemies(self, current_tick: int) -> list[EnemySnapshot]:
        alive = [enemy for enemy in self.enemies.values() if enemy.alive and enemy.age(current_tick) <= 40]
        return sorted(alive, key=lambda enemy: enemy.distance)

    def count_nearby(self, x: float, y: float, radius: float, current_tick: int) -> int:
        radius_sq = radius * radius
        return sum(1 for enemy in self.get_alive_enemies(current_tick) if ((enemy.x - x) ** 2 + (enemy.y - y) ** 2) <= radius_sq)


@dataclass
class BotContext:
    time: int
    x: float
    y: float
    energy: float
    velocity: float
    heading_radians: float
    gun_heading_radians: float
    radar_heading_radians: float
    battlefield_width: float
    battlefield_height: float
    others: int
    enemies: list[EnemySnapshot]
    enemy_tracker: EnemyTracker
    danger_map: DangerMap | None

    def is_low_energy(self) -> bool:
        return self.energy < 25.0

    def is_crowded(self) -> bool:
        return self.others >= 4 or self.enemy_tracker.count_nearby(self.x, self.y, 225.0, self.time) >= 3

    def is_near_wall(self) -> bool:
        return self.x < 70.0 or self.y < 70.0 or self.x > self.battlefield_width - 70.0 or self.y > self.battlefield_height - 70.0


class DangerMapBuilder:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols

    def build(self, context: BotContext) -> DangerMap:
        danger_map = DangerMap(self.rows, self.cols, context.battlefield_width, context.battlefield_height)
        for row in range(self.rows):
            for col in range(self.cols):
                x = (col + 0.5) * (context.battlefield_width / self.cols)
                y = (row + 0.5) * (context.battlefield_height / self.rows)
                risk = self._wall_risk(x, y, context)
                for enemy in context.enemies:
                    distance = math.dist((x, y), (enemy.x, enemy.y))
                    risk += (enemy.energy + 20.0) / max(75.0, distance)
                risk += context.enemy_tracker.count_nearby(x, y, 200.0, context.time) * 1.25
                danger_map.add_danger(row, col, risk)
        return danger_map

    def _wall_risk(self, x: float, y: float, context: BotContext) -> float:
        margin = 80.0
        return margin / max(1.0, x) + margin / max(1.0, context.battlefield_width - x) + margin / max(1.0, y) + margin / max(1.0, context.battlefield_height - y)


class WeightedTargetSelector:
    def select_target(self, context: BotContext) -> EnemySnapshot | None:
        best = None
        best_score = float("-inf")
        for enemy in context.enemies:
            score = self.score_enemy(enemy, context)
            if score > best_score:
                best_score = score
                best = enemy
        return best

    def score_enemy(self, enemy: EnemySnapshot, context: BotContext) -> float:
        distance_score = 350.0 / max(100.0, enemy.distance)
        low_energy_bonus = max(0.0, 40.0 - enemy.energy) * 0.08
        freshness_bonus = max(0.0, 30.0 - enemy.age(context.time)) * 0.06
        crowd_penalty = context.enemy_tracker.count_nearby(enemy.x, enemy.y, 175.0, context.time) * 0.45
        wall_bonus = 0.55 if enemy.x < 90.0 or enemy.y < 90.0 or enemy.x > context.battlefield_width - 90.0 or enemy.y > context.battlefield_height - 90.0 else 0.0
        return distance_score + low_energy_bonus + freshness_bonus + wall_bonus - crowd_penalty


class RuleBasedTacticalManager:
    def choose_mode(self, context: BotContext, target: EnemySnapshot | None) -> TacticalMode:
        if context.is_crowded():
            return TacticalMode.ESCAPE_CROWD
        if context.is_low_energy() and context.others > 1:
            return TacticalMode.SURVIVE
        if target is not None and target.energy < 18.0 and target.distance < 300.0:
            return TacticalMode.FINISH_WEAK_TARGET
        if context.is_near_wall():
            return TacticalMode.REPOSITION
        return TacticalMode.ENGAGE


class ModeAwareMovementController:
    def __init__(self) -> None:
        self.strafe_direction = 1

    def reverse_direction(self) -> None:
        self.strafe_direction *= -1

    def apply(self, robot: Bot, context: BotContext, mode: TacticalMode, target: EnemySnapshot | None) -> None:
        if mode in {TacticalMode.ESCAPE_CROWD, TacticalMode.REPOSITION}:
            self._move_to_safest_cell(robot, context)
            return
        if target is None:
            robot.ahead(100.0)
            robot.turn_right(math.degrees(0.4))
            return
        if mode == TacticalMode.SURVIVE:
            self._orbit(robot, context, target, 225.0, 0.9)
        elif mode == TacticalMode.FINISH_WEAK_TARGET:
            self._orbit(robot, context, target, 160.0, 0.5)
        else:
            self._orbit(robot, context, target, 250.0, 0.7)

    def _move_to_safest_cell(self, robot: Bot, context: BotContext) -> None:
        safest = context.danger_map.pick_safest_cell_center(context.x, context.y) if context.danger_map is not None else (context.x, context.y)
        angle = _normalize(math.atan2(safest[0] - context.x, safest[1] - context.y) - context.heading_radians)
        distance = math.dist((context.x, context.y), safest)
        self._set_back_as_front(robot, angle, min(160.0, distance))

    def _orbit(self, robot: Bot, context: BotContext, target: EnemySnapshot, preferred_range: float, aggressiveness: float) -> None:
        absolute_bearing = target.absolute_bearing_radians
        desired_heading = absolute_bearing + (math.pi / 2.0 * self.strafe_direction)
        distance_error = target.distance - preferred_range
        robot.turn_right(math.degrees(_normalize(desired_heading - context.heading_radians)))
        robot.ahead(120.0 + distance_error * aggressiveness)

    def _set_back_as_front(self, robot: Bot, angle: float, distance: float) -> None:
        normalized = _normalize(angle)
        if abs(normalized) > math.pi / 2.0:
            robot.turn_right(math.degrees(_normalize(normalized + math.pi)))
            robot.back(distance)
        else:
            robot.turn_right(math.degrees(normalized))
            robot.ahead(distance)


class SweepRadarController:
    def apply(self, robot: Bot, context: BotContext, target: EnemySnapshot | None) -> None:
        if target is None:
            robot.turn_radar_right(float("inf"))
            return
        radar_turn = _normalize(target.absolute_bearing_radians - context.radar_heading_radians)
        extra_turn = math.atan(36.0 / max(1.0, target.distance))
        robot.turn_radar_right(math.degrees(radar_turn + (extra_turn if radar_turn >= 0 else -extra_turn)))


class GuessFactorGunController:
    def choose_fire_power(self, context: BotContext, target: EnemySnapshot, mode: TacticalMode) -> float:
        distance_factor = _clamp(450.0 / max(120.0, target.distance), 0.45, 2.2)
        finishing_bonus = 0.45 if target.energy < 16.0 else 0.0
        survival_penalty = 0.5 if mode == TacticalMode.SURVIVE else 0.0
        crowd_penalty = 0.25 if context.is_crowded() else 0.0
        energy_budget = _clamp(context.energy / 35.0, 0.5, 1.5)
        hit_chance_proxy = _clamp(1.4 - abs(target.velocity) / 8.0, 0.55, 1.15)
        power = (1.1 * distance_factor * hit_chance_proxy * energy_budget) + finishing_bonus - survival_penalty - crowd_penalty
        power = min(power, target.energy / 4.0 + 0.5)
        power = min(power, context.energy - 0.2)
        return _clamp(power, 0.1, 3.0)

    def apply(self, robot: Bot, context: BotContext, target: EnemySnapshot | None, mode: TacticalMode) -> None:
        if target is None:
            return
        power = self.choose_fire_power(context, target, mode)
        bullet_speed = 20.0 - 3.0 * power
        time_to_target = target.distance / bullet_speed
        predicted_x = _clamp(target.x + math.sin(target.heading_radians) * target.velocity * time_to_target, 18.0, context.battlefield_width - 18.0)
        predicted_y = _clamp(target.y + math.cos(target.heading_radians) * target.velocity * time_to_target, 18.0, context.battlefield_height - 18.0)
        aim_bearing = _absolute_bearing(context.x, context.y, predicted_x, predicted_y)
        robot.turn_gun_right(math.degrees(_normalize(aim_bearing - context.gun_heading_radians)))
        if robot.gun_heat == 0.0 and abs(robot.gun_turn_remaining_radians) < math.radians(8.0):
            robot.fire(power)


class HybridMeleeBot(Bot):
    def __init__(self) -> None:
        super().__init__()
        self.enemy_tracker = EnemyTracker()
        self.danger_map_builder = DangerMapBuilder(10, 10)
        self.tactical_manager = RuleBasedTacticalManager()
        self.target_selector = WeightedTargetSelector()
        self.movement_controller = ModeAwareMovementController()
        self.radar_controller = SweepRadarController()
        self.gun_controller = GuessFactorGunController()
        self.current_mode = TacticalMode.ENGAGE

    def run(self) -> None:
        self.set_adjust_gun_for_robot_turn(True)
        self.set_adjust_radar_for_gun_turn(True)
        while self.running:
            context = self._build_context()
            target = self.target_selector.select_target(context)
            self.current_mode = self.tactical_manager.choose_mode(context, target)
            self.movement_controller.apply(self, context, self.current_mode, target)
            self.gun_controller.apply(self, context, target, self.current_mode)
            self.radar_controller.apply(self, context, target)
            self.execute()

    def on_scanned_bot(self, event: ScannedBotEvent) -> None:
        self.enemy_tracker.on_scanned_robot(self, event)

    def on_robot_death(self, event) -> None:
        self.enemy_tracker.on_robot_death(event)

    def on_hit_by_bullet(self, event: HitByBulletEvent) -> None:
        self.movement_controller.reverse_direction()

    def on_hit_wall(self, event: HitWallEvent) -> None:
        self.movement_controller.reverse_direction()

    def on_hit_robot(self, event: HitRobotEvent) -> None:
        self.movement_controller.reverse_direction()

    def on_bullet_hit(self, event: BulletHitEvent) -> None:
        if event.energy <= 0.0:
            self.movement_controller.reverse_direction()

    def _build_context(self) -> BotContext:
        enemies = self.enemy_tracker.get_alive_enemies(self.time)
        partial = BotContext(self.time, self.x, self.y, self.energy, self.velocity, math.radians(self.direction), math.radians(self.gun_direction), math.radians(self.radar_direction), self.arena_width, self.arena_height, self.others, enemies, self.enemy_tracker, None)
        danger_map = self.danger_map_builder.build(partial)
        return BotContext(partial.time, partial.x, partial.y, partial.energy, partial.velocity, partial.heading_radians, partial.gun_heading_radians, partial.radar_heading_radians, partial.battlefield_width, partial.battlefield_height, partial.others, partial.enemies, partial.enemy_tracker, danger_map)


def main() -> None:
    HybridMeleeBot().start()


if __name__ == "__main__":
    main()
