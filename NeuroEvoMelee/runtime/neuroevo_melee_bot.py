"""Python runtime for the NeuroEvoMelee bot."""

from __future__ import annotations

import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import BotDeathEvent, DeathEvent, HitByBulletEvent, RoundStartedEvent, ScannedBotEvent, WonRoundEvent

from NeuroEvoMelee.genome import EnemyState, FeatureEncoder, GenomeLoader, GenomeNetwork

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENOME_PATH = ROOT / "data" / "current_genome.json"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class NeuroEvoMeleeBot(Bot):
    OUTPUT_SIZE = 4

    def __init__(self, genome_path: str | None = None, telemetry_path: str | None = None) -> None:
        super().__init__()
        resolved_genome_path = Path(genome_path or os.environ.get("NEURO_GENOME_PATH", str(DEFAULT_GENOME_PATH)))
        self.policy = GenomeLoader.load(resolved_genome_path, FeatureEncoder.INPUT_SIZE, self.OUTPUT_SIZE)
        resolved_telemetry = telemetry_path if telemetry_path is not None else os.environ.get("NEURO_TELEMETRY_PATH")
        self.telemetry_path = Path(resolved_telemetry) if resolved_telemetry and str(resolved_telemetry).strip() else None
        self.enemies: dict[int, EnemyState] = {}
        self.target_preference_bias = 0.0
        self.last_energy = 100.0
        self.damage_taken = 0.0
        self.damage_dealt = 0.0
        self.kills = 0
        self.local_tick = 0
        self.round_start_turn = 0
        self.last_damaged_enemy_name: str | None = None
        self.last_damage_tick = -999

    def run(self) -> None:
        self.set_adjust_gun_for_robot_turn(True)
        self.set_adjust_radar_for_gun_turn(True)
        self.set_adjust_radar_for_robot_turn(True)
        self.set_max_velocity(8.0)
        self.turn_radar_right(float("inf"))

        while self.running:
            self.local_tick += 1
            self._refresh_damage_taken()
            nearest = self._find_nearest()
            weakest = self._find_weakest()
            preferred = self._choose_preferred_target(nearest, weakest)
            state = FeatureEncoder.encode(self, self.enemies.values(), nearest, weakest, preferred)
            action = self.policy.forward(state)
            self._apply_action(action, preferred)
            self.execute()

    def on_round_started(self, event: RoundStartedEvent) -> None:
        self.enemies.clear()
        self.target_preference_bias = 0.0
        self.damage_taken = 0.0
        self.damage_dealt = 0.0
        self.kills = 0
        self.local_tick = 0
        self.last_energy = float(self.energy)
        self.round_start_turn = int(getattr(event, "turn_number", getattr(event, "turnNumber", 0)))
        self.last_damaged_enemy_name = None
        self.last_damage_tick = -999

    def on_scanned_bot(self, event: ScannedBotEvent) -> None:
        enemy_id = self._event_int(event, ("scanned_bot_id", "scannedBotId", "bot_id", "botId", "id"), default=-1)
        if enemy_id < 0:
            return
        enemy = self.enemies.get(enemy_id)
        if enemy is None:
            enemy = EnemyState(enemy_id)
            self.enemies[enemy_id] = enemy
        enemy.x = float(event.x)
        enemy.y = float(event.y)
        enemy.energy = float(event.energy)
        enemy.direction = float(getattr(event, "direction", getattr(event, "heading", 0.0)))
        enemy.speed = float(getattr(event, "speed", 0.0))
        enemy.last_seen_turn = self._event_int(event, ("turn_number", "turnNumber", "turn", "time"), default=self.local_tick)
        enemy.alive = True

    def on_bot_death(self, event: BotDeathEvent) -> None:
        victim_id = self._event_int(event, ("victim_id", "victimId", "bot_id", "botId", "id"), default=-1)
        if victim_id < 0:
            return
        enemy = self.enemies.get(victim_id)
        if enemy is not None:
            enemy.alive = False
            if getattr(event, "name", None) == self.last_damaged_enemy_name or getattr(event, "victim_name", None) == self.last_damaged_enemy_name:
                if (self.local_tick - self.last_damage_tick) <= 2:
                    self.kills += 1

    def on_hit_by_bullet(self, event: HitByBulletEvent) -> None:
        self._refresh_damage_taken()

    def on_bullet_hit(self, event) -> None:  # noqa: ANN001
        bullet = getattr(event, "bullet", None)
        power = float(getattr(bullet, "power", 1.0))
        self.damage_dealt += (4.0 * power) + max(0.0, 2.0 * (power - 1.0))
        self.last_damaged_enemy_name = getattr(event, "name", None)
        self.last_damage_tick = self.local_tick

    def on_death(self, event: DeathEvent) -> None:
        del event
        self._write_telemetry(False, int(getattr(self, "enemy_count", 0)) + 1)

    def on_won_round(self, event: WonRoundEvent) -> None:
        del event
        self._write_telemetry(True, 1)

    def _refresh_damage_taken(self) -> None:
        energy = float(self.energy)
        delta = self.last_energy - energy
        if delta > 0.0:
            self.damage_taken += delta
        self.last_energy = energy

    def _choose_preferred_target(self, nearest: EnemyState | None, weakest: EnemyState | None) -> EnemyState | None:
        if nearest is None:
            return weakest
        if weakest is None:
            return nearest
        return weakest if self.target_preference_bias >= 0.0 else nearest

    def _find_nearest(self) -> EnemyState | None:
        alive = [enemy for enemy in self.enemies.values() if enemy.alive]
        return min(alive, key=lambda enemy: math.dist((self.x, self.y), (enemy.x, enemy.y)), default=None)

    def _find_weakest(self) -> EnemyState | None:
        alive = [enemy for enemy in self.enemies.values() if enemy.alive]
        return min(alive, key=lambda enemy: enemy.energy, default=None)

    def _apply_action(self, action: list[float], target: EnemyState | None) -> None:
        turn = action[0] * 35.0
        move = action[1] * 140.0
        fire_raw = (action[2] + 1.0) * 0.5
        preference_adjust = action[3]
        self.target_preference_bias = _clamp(self.target_preference_bias * 0.85 + (preference_adjust * 0.15), -1.0, 1.0)

        self.turn_right(turn)
        self.ahead(move)

        if target is not None:
            gun_turn = self.gun_bearing_to(target.x, target.y)
            self.turn_gun_right(gun_turn)
        else:
            self.turn_gun_right(22.0)

        if target is not None and self.gun_heat <= 0.0 and fire_raw > 0.18:
            power = _clamp(fire_raw * 3.0, 0.2, 3.0)
            if self.energy > power + 0.1:
                self.fire(power)

        if target is None:
            self.turn_radar_right(float("inf"))
        else:
            radar_turn = self.bearing_to(target.x, target.y)
            self.turn_radar_right(radar_turn)

    def _write_telemetry(self, won: bool, placement: int) -> None:
        if self.telemetry_path is None:
            return
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "won": won,
            "placement": placement,
            "energy": float(self.energy),
            "damage_taken": self.damage_taken,
            "damage_dealt": self.damage_dealt,
            "kills": self.kills,
            "turn": self.local_tick,
            "round_start_turn": self.round_start_turn,
        }
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    @staticmethod
    def _event_int(event: object, names: tuple[str, ...], default: int = 0) -> int:
        for name in names:
            if hasattr(event, name):
                try:
                    return int(getattr(event, name))
                except (TypeError, ValueError):
                    continue
        return default


def main() -> None:
    parser = ArgumentParser(description="Run the NeuroEvoMelee bot")
    parser.add_argument("--genome", default=None, help="Path to the genome JSON file")
    parser.add_argument("--telemetry", default=None, help="Optional JSONL telemetry path")
    args = parser.parse_args()

    NeuroEvoMeleeBot(genome_path=args.genome, telemetry_path=args.telemetry).start()


if __name__ == "__main__":
    main()
