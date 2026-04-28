"""Python conversion of the melee opponent-model bot."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import BulletHitEvent, DeathEvent, HitByBulletEvent, HitRobotEvent, HitWallEvent, ScannedBotEvent, WonRoundEvent


WALL_MARGIN = 80.0
MAX_TRACK_AGE = 45.0
DISENGAGE_DISTANCE = 180.0
RADAR_OVERSCAN = math.radians(18.0)
MOVEMENT_STEP = 140.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_bearing(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _absolute_bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.atan2(x2 - x1, y2 - y1)


@dataclass
class EnemyModel:
    name: str
    alive: bool = True
    energy: float = 100.0
    x: float = 0.0
    y: float = 0.0
    distance: float = 1000.0
    absolute_bearing: float = 0.0
    heading_radians: float = 0.0
    velocity: float = 0.0
    lateral_velocity: float = 0.0
    closing_velocity: float = 0.0
    last_seen_time: int = -1
    nearby_enemy_count: int = 0
    aggression_level: float = 0.0
    average_distance: float = 450.0
    firing_frequency: float = 0.0
    target_me_likelihood: float = 0.0
    estimated_accuracy: float = 0.0
    threat_score: float = 0.0
    heading_change_ema: float = 0.0
    velocity_change_ema: float = 0.0
    stationary_ratio: float = 0.0
    strafe_ratio: float = 0.0
    last_heading_radians: float = 0.0
    last_velocity: float = 0.0
    last_energy: float = 100.0
    scans: int = 0
    estimated_shots: int = 0
    close_contact_count: int = 0
    bullets_hit_me: int = 0
    bullets_i_hit: int = 0
    bullets_i_fired: int = 0
    movement_style: str = "mixed"
    category: str = "balanced"

    def update_from_scan(self, robot: Bot, event: ScannedBotEvent) -> None:
        self.alive = True
        self.scans += 1
        self.last_seen_time = robot.time
        self.absolute_bearing = math.radians(robot.direction) + event.bearing
        self.heading_radians = event.direction
        self.velocity = event.velocity
        self.distance = event.distance
        self.energy = event.energy
        self.x = robot.x + math.sin(self.absolute_bearing) * self.distance
        self.y = robot.y + math.cos(self.absolute_bearing) * self.distance
        self.lateral_velocity = event.velocity * math.sin(event.direction - self.absolute_bearing)
        self.closing_velocity = -event.velocity * math.cos(event.direction - self.absolute_bearing)
        self.average_distance = self._blend(self.average_distance, self.distance, 0.08)
        self.stationary_ratio = self._blend(self.stationary_ratio, 1.0 if abs(self.velocity) < 1.0 else 0.0, 0.10)
        self.strafe_ratio = self._blend(self.strafe_ratio, 1.0 if abs(self.lateral_velocity) > 4.0 else 0.0, 0.10)
        self.heading_change_ema = self._blend(self.heading_change_ema, abs(_normalize_bearing(self.heading_radians - self.last_heading_radians)), 0.18)
        self.velocity_change_ema = self._blend(self.velocity_change_ema, abs(self.velocity - self.last_velocity), 0.18)

        energy_drop = self.last_energy - self.energy
        if 0.1 <= energy_drop <= 3.0:
            self.estimated_shots += 1
            self.firing_frequency = self._blend(self.firing_frequency, 1.0, 0.18)
            if self.distance < 280.0 or abs(event.bearing) < math.radians(20.0):
                self.target_me_likelihood = self._blend(self.target_me_likelihood, 1.0, 0.14)
        else:
            self.firing_frequency = self._blend(self.firing_frequency, 0.0, 0.04)

        close_pressure = _clamp(1.0 - self.distance / 250.0, 0.0, 1.0)
        chase_pressure = _clamp(self.closing_velocity / 8.0, 0.0, 1.0)
        ram_pressure = _clamp(self.close_contact_count / 3.0, 0.0, 1.0)
        self.aggression_level = _clamp(0.45 * self._blend(self.aggression_level, close_pressure, 0.10) + 0.35 * chase_pressure + 0.20 * ram_pressure, 0.0, 1.0)
        self.update_movement_style()
        self.refresh_derived_metrics()
        self.last_heading_radians = self.heading_radians
        self.last_velocity = self.velocity
        self.last_energy = self.energy

    def refresh_derived_metrics(self) -> None:
        if self.estimated_shots > 0:
            self.estimated_accuracy = _clamp(self.bullets_hit_me / self.estimated_shots, 0.0, 1.0)
        self.category = self.classify()
        proximity = _clamp(1.0 - self.distance / 600.0, 0.0, 1.0)
        energy_factor = _clamp(self.energy / 100.0, 0.0, 1.0)
        category_bonus = 0.0
        if self.category == "close_range_aggressor":
            category_bonus += 0.12
        elif self.category == "high_accuracy_threat":
            category_bonus += 0.15
        elif self.category == "spinner_weak_bot":
            category_bonus -= 0.10
        elif self.category == "passive_survivor":
            category_bonus -= 0.04
        self.threat_score = _clamp(0.27 * self.aggression_level + 0.18 * proximity + 0.18 * self.firing_frequency + 0.17 * self.target_me_likelihood + 0.12 * self.estimated_accuracy + 0.08 * energy_factor + category_bonus, 0.0, 1.0)

    def register_hit_on_me(self) -> None:
        self.bullets_hit_me += 1
        self.target_me_likelihood = self._blend(self.target_me_likelihood, 1.0, 0.22)

    def register_my_shot(self) -> None:
        self.bullets_i_fired += 1

    def register_my_hit(self) -> None:
        self.bullets_i_hit += 1

    def update_movement_style(self) -> None:
        if self.heading_change_ema > 0.45 and self.velocity_change_ema < 1.2:
            self.movement_style = "spinner"
        elif self.strafe_ratio > 0.55:
            self.movement_style = "strafer"
        elif self.aggression_level > 0.72 and self.average_distance < 180.0:
            self.movement_style = "rammer"
        elif self.stationary_ratio > 0.45:
            self.movement_style = "camper"
        else:
            self.movement_style = "mixed"

    def classify(self) -> str:
        if self.movement_style == "spinner" or self.stationary_ratio > 0.58:
            return "spinner_weak_bot"
        if self.estimated_accuracy > 0.18 and self.firing_frequency > 0.12:
            return "high_accuracy_threat"
        if self.aggression_level > 0.60 and self.average_distance < 260.0:
            return "close_range_aggressor"
        if self.aggression_level < 0.28 and self.firing_frequency < 0.06 and self.average_distance > 320.0:
            return "passive_survivor"
        return "balanced"

    @staticmethod
    def _blend(current: float, observation: float, weight: float) -> float:
        return current + (observation - current) * weight


class AdaptivePolicy:
    def __init__(self) -> None:
        self.center_aversion = 0.35
        self.disengage_bias = 0.50
        self.anti_ram_bias = 0.40

    def update(self, live_enemies: list[EnemyModel]) -> None:
        aggressors = sum(1 for enemy in live_enemies if enemy.category == "close_range_aggressor")
        passive = sum(1 for enemy in live_enemies if enemy.category == "passive_survivor")
        spinners = sum(1 for enemy in live_enemies if enemy.category == "spinner_weak_bot")
        snipers = sum(1 for enemy in live_enemies if enemy.category == "high_accuracy_threat")
        self.center_aversion = 0.30 + aggressors * 0.08 + snipers * 0.05
        self.disengage_bias = 0.45 + aggressors * 0.10
        self.anti_ram_bias = 0.35 + spinners * 0.06 + aggressors * 0.05
        if passive > aggressors + snipers:
            self.center_aversion -= 0.08


class MeleeOpponentModelBot(Bot):
    def __init__(self) -> None:
        super().__init__()
        self.enemies: dict[str, EnemyModel] = {}
        self.adaptive_policy = AdaptivePolicy()
        self.move_direction = 1
        self.last_hit_by_bullet_time = -1
        self.current_target: EnemyModel | None = None

    def run(self) -> None:
        self.set_adjust_radar_for_gun_turn(True)
        self.set_adjust_gun_for_robot_turn(True)
        self.set_adjust_radar_for_robot_turn(True)
        self.turn_radar_right(float("inf"))
        while self.running:
            self._refresh_threats()
            live = self._live_enemies()
            self.adaptive_policy.update(live)
            self.current_target = self._choose_target()
            self._do_movement(live)
            self._do_gun_and_fire()
            self._do_radar()
            self.execute()

    def on_scanned_bot(self, event: ScannedBotEvent) -> None:
        model = self.enemies.get(event.name)
        if model is None:
            model = EnemyModel(event.name)
            self.enemies[event.name] = model
        model.update_from_scan(self, event)

    def on_robot_death(self, event) -> None:
        model = self.enemies.get(event.name)
        if model is not None:
            model.alive = False

    def on_hit_by_bullet(self, event: HitByBulletEvent) -> None:
        self.last_hit_by_bullet_time = self.time
        shooter = self._guess_shooter(event.bullet.bearing)
        if shooter is not None:
            shooter.register_hit_on_me()
        self.move_direction *= -1

    def on_bullet_hit(self, event: BulletHitEvent) -> None:
        model = self.enemies.get(event.name)
        if model is not None:
            model.register_my_hit()

    def on_hit_robot(self, event: HitRobotEvent) -> None:
        model = self.enemies.get(event.name)
        if model is not None:
            model.close_contact_count += 1
        self.move_direction *= -1
        self.back(80)

    def on_hit_wall(self, event: HitWallEvent) -> None:
        self.move_direction *= -1
        self.turn_right(math.degrees(_normalize_bearing(math.pi / 2 - getattr(event, 'bearing_radians', 0.0))))
        self.ahead(120)

    def on_won_round(self, event: WonRoundEvent) -> None:
        del event

    def on_death(self, event: DeathEvent) -> None:
        del event

    def _refresh_threats(self) -> None:
        for model in self.enemies.values():
            if model.alive:
                model.refresh_derived_metrics()

    def _choose_target(self) -> EnemyModel | None:
        alive = self._live_enemies()
        if not alive:
            return None
        best = None
        best_score = float("-inf")
        for enemy in alive:
            freshness = _clamp(1.0 - ((self.time - enemy.last_seen_time) / MAX_TRACK_AGE), 0.0, 1.0)
            if freshness <= 0.0:
                continue
            distance_factor = 1.0 - _clamp(enemy.distance / 900.0, 0.0, 1.0)
            accuracy_window = 1.0 - _clamp(abs(enemy.lateral_velocity) / 8.0, 0.0, 1.0)
            low_energy_finish = 1.0 - _clamp(enemy.energy / 100.0, 0.0, 1.0)
            score = enemy.threat_score * 0.45 + distance_factor * 0.22 + accuracy_window * 0.18 + low_energy_finish * 0.10 + freshness * 0.05
            if enemy.category == "spinner_weak_bot":
                score += 0.08
            if score > best_score:
                best_score = score
                best = enemy
        return best

    def _do_gun_and_fire(self) -> None:
        if self.current_target is None or self.others == 0:
            return
        fire_power = self._choose_fire_power(self.current_target)
        bullet_speed = 20.0 - 3.0 * fire_power
        predicted = self._predict_linear_position(self.current_target, bullet_speed)
        gun_angle = _absolute_bearing(self.x, self.y, predicted[0], predicted[1])
        self.turn_gun_right(math.degrees(_normalize_bearing(gun_angle - math.radians(self.gun_direction))))
        if abs(self.gun_turn_remaining_radians) < math.radians(6.0) and self.gun_heat == 0.0 and self.energy > fire_power:
            self.fire(fire_power)
            self.current_target.register_my_shot()

    def _choose_fire_power(self, target: EnemyModel) -> float:
        power = 1.4
        if target.distance < 200:
            power += 0.8
        elif target.distance < 350:
            power += 0.4
        if target.threat_score > 0.75:
            power += 0.4
        if self.others > 3:
            power -= 0.3
        if self.energy < 25:
            power -= 0.4
        return _clamp(power, 0.8, 2.8)

    def _do_movement(self, alive: list[EnemyModel]) -> None:
        if not alive:
            self.ahead(120 * self.move_direction)
            return
        top_threat = max(alive, key=lambda enemy: enemy.threat_score)
        should_disengage = top_threat.distance < DISENGAGE_DISTANCE + self.adaptive_policy.disengage_bias * 70.0 and (top_threat.threat_score > (0.82 - self.adaptive_policy.disengage_bias * 0.2) or self.energy < top_threat.energy)
        candidates = [
            top_threat.absolute_bearing + (math.pi / 2.0 * self.move_direction),
            top_threat.absolute_bearing + (math.pi / 2.0 * self.move_direction) + math.radians(30),
            top_threat.absolute_bearing + (math.pi / 2.0 * self.move_direction) - math.radians(30),
            top_threat.absolute_bearing + (math.pi / 2.0 * self.move_direction) + math.radians(60),
            top_threat.absolute_bearing + (math.pi / 2.0 * self.move_direction) - math.radians(60),
        ]
        if should_disengage:
            candidates.append(top_threat.absolute_bearing + math.pi)
        best_angle = candidates[0]
        best_danger = float("inf")
        for angle in candidates:
            smoothed = self._wall_smooth(angle, self.move_direction)
            destination = self._project(self.x, self.y, smoothed, MOVEMENT_STEP)
            danger = self._danger_at(destination, alive)
            if danger < best_danger:
                best_danger = danger
                best_angle = smoothed
        self._go_to(self._project(self.x, self.y, best_angle, MOVEMENT_STEP))
        if should_disengage or (self.last_hit_by_bullet_time >= 0 and self.time - self.last_hit_by_bullet_time < 8):
            self.move_direction *= -1

    def _do_radar(self) -> None:
        if self.current_target is None:
            self.turn_radar_right(float("inf"))
            return
        radar_turn = _normalize_bearing(self.current_target.absolute_bearing - math.radians(self.radar_direction))
        extra = RADAR_OVERSCAN if radar_turn >= 0 else -RADAR_OVERSCAN
        self.turn_radar_right(math.degrees(radar_turn + extra))

    def _danger_at(self, destination: tuple[float, float], alive: list[EnemyModel]) -> float:
        danger = 0.0
        for enemy in alive:
            distance = max(36.0, math.dist(destination, (enemy.x, enemy.y)))
            cluster_factor = 1.0 + (enemy.nearby_enemy_count * 0.12)
            directness = 1.0 - abs(math.cos(_absolute_bearing(enemy.x, enemy.y, destination[0], destination[1]) - enemy.absolute_bearing))
            ram_penalty = self.adaptive_policy.anti_ram_bias * _clamp(1.0 - distance / 220.0, 0.0, 1.0) if enemy.category == "close_range_aggressor" else 0.0
            danger += enemy.threat_score * cluster_factor * (65000.0 / (distance * distance)) * (0.65 + 0.35 * directness + ram_penalty)
        edge_penalty = max(0.0, WALL_MARGIN - destination[0]) * 0.025 + max(0.0, WALL_MARGIN - destination[1]) * 0.025 + max(0.0, destination[0] - (self.arena_width - WALL_MARGIN)) * 0.025 + max(0.0, destination[1] - (self.arena_height - WALL_MARGIN)) * 0.025
        center = (self.arena_width / 2.0, self.arena_height / 2.0)
        center_penalty = self.adaptive_policy.center_aversion * _clamp(1.0 - math.dist(destination, center) / 260.0, 0.0, 1.0)
        return danger + edge_penalty + center_penalty

    def _guess_shooter(self, bullet_bearing_radians: float) -> EnemyModel | None:
        absolute_bullet_bearing = math.radians(self.direction) + bullet_bearing_radians + math.pi
        best = None
        best_diff = float("inf")
        for enemy in self._live_enemies():
            diff = abs(_normalize_bearing(enemy.absolute_bearing - absolute_bullet_bearing))
            if diff < best_diff:
                best_diff = diff
                best = enemy
        return best if best_diff < math.radians(20.0) else None

    def _live_enemies(self) -> list[EnemyModel]:
        alive = [enemy for enemy in self.enemies.values() if enemy.alive]
        for enemy in alive:
            enemy.nearby_enemy_count = sum(1 for other in alive if other is not enemy and math.dist((enemy.x, enemy.y), (other.x, other.y)) < 220.0)
        return alive

    def _predict_linear_position(self, enemy: EnemyModel, bullet_speed: float) -> tuple[float, float]:
        predicted = (enemy.x, enemy.y)
        heading = enemy.heading_radians
        velocity = enemy.velocity
        ticks = 0
        while (ticks := ticks + 1) * bullet_speed < math.dist((self.x, self.y), predicted):
            predicted = self._project(predicted[0], predicted[1], heading, velocity)
            predicted = (_clamp(predicted[0], WALL_MARGIN, self.arena_width - WALL_MARGIN), _clamp(predicted[1], WALL_MARGIN, self.arena_height - WALL_MARGIN))
        return predicted

    def _go_to(self, destination: tuple[float, float]) -> None:
        angle = _normalize_bearing(_absolute_bearing(self.x, self.y, destination[0], destination[1]) - math.radians(self.direction))
        turn = math.atan(math.tan(angle))
        self.turn_right(math.degrees(turn))
        if angle == turn:
            self.ahead(math.dist((self.x, self.y), destination))
        else:
            self.back(math.dist((self.x, self.y), destination))

    def _wall_smooth(self, source_angle: float, direction: int) -> float:
        angle = source_angle
        test = self._project(self.x, self.y, angle, WALL_MARGIN)
        guard = 0
        while not self._in_safe_field(test) and guard < 25:
            angle += direction * 0.12
            test = self._project(self.x, self.y, angle, WALL_MARGIN)
            guard += 1
        return angle

    def _in_safe_field(self, point: tuple[float, float]) -> bool:
        return WALL_MARGIN < point[0] < self.arena_width - WALL_MARGIN and WALL_MARGIN < point[1] < self.arena_height - WALL_MARGIN

    @staticmethod
    def _project(x: float, y: float, angle: float, length: float) -> tuple[float, float]:
        return x + math.sin(angle) * length, y + math.cos(angle) * length


def main() -> None:
    MeleeOpponentModelBot().start()


if __name__ == "__main__":
    main()
