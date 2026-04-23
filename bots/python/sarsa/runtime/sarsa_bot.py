"""Minimal tabular SARSA bot for Robocode Tank Royale.

This file intentionally keeps the baseline compact and debuggable:
- Small discretized state space (tabular friendly)
- 6 discrete macro-actions
- Event-shaped rewards + terminal win/loss reward
- On-policy SARSA updates

Run this bot repeatedly to train. It persists Q-values and logs per episode.
"""

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
from robocode_tank_royale.bot_api.events import (
	DeathEvent,
	HitByBulletEvent,
	HitWallEvent,
	ScannedBotEvent,
	WonRoundEvent,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_Q_TABLE_PATH = REPO_ROOT / "data" / "q_tables" / "sarsa" / "q_table_sarsa.json"
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "sarsa" / "training_log.jsonl"


def _normalize_angle(degrees: float) -> float:
	"""Normalize angle to [-180, 180]."""
	angle = degrees
	while angle > 180:
		angle -= 360
	while angle < -180:
		angle += 360
	return angle


def _bullet_damage(power: float) -> float:
	"""Robocode bullet damage approximation."""
	return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


@dataclass
class LastScan:
	x: float
	y: float
	turn: int


class SarsaBot(Bot):
	# Action IDs
	STRAFE_LEFT = 0
	STRAFE_RIGHT = 1
	FORWARD = 2
	BACKWARD = 3
	FIRE_LOW = 4
	FIRE_MEDIUM = 5
	ACTION_COUNT = 6

	def __init__(
		self,
		alpha: float,
		gamma: float,
		epsilon: float,
		epsilon_decay: float,
		epsilon_min: float,
		q_table_path: str,
		log_path: str,
	) -> None:
		super().__init__()

		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min

		self.q_table_path = q_table_path
		self.log_path = log_path
		self.q: dict[str, list[float]] = defaultdict(
			lambda: [0.0] * self.ACTION_COUNT
		)
		self._load_q_table()

		self.local_tick = 0
		self.episode_number = 0

		self.last_scan: LastScan | None = None

		self.prev_state: str | None = None
		self.prev_action: int | None = None
		self.step_reward_accumulator = 0.0

		self.episode_reward = 0.0
		self.wall_hits = 0
		self.fire_actions = 0
		self.q_update_abs_sum = 0.0
		self.q_update_count = 0

	def run(self) -> None:
		self.episode_number += 1
		self.local_tick = 0
		self.prev_state = None
		self.prev_action = None
		self.step_reward_accumulator = 0.0

		self.episode_reward = 0.0
		self.wall_hits = 0
		self.fire_actions = 0
		self.q_update_abs_sum = 0.0
		self.q_update_count = 0

		print(
			"[SarsaBot] Episode",
			self.episode_number,
			"epsilon=",
			f"{self.epsilon:.4f}",
		)

		while self.running:
			self.local_tick += 1

			state = self._encode_state()
			action = self._select_action(state)

			if self.prev_state is not None and self.prev_action is not None:
				reward = self.step_reward_accumulator
				self._sarsa_update(
					state=self.prev_state,
					action=self.prev_action,
					reward=reward,
					next_state=state,
					next_action=action,
					done=False,
				)
				self.episode_reward += reward

			self.step_reward_accumulator = 0.0
			self._execute_action(action)

			self.prev_state = state
			self.prev_action = action

			# Keep radar sweeping so scan freshness stays meaningful.
			self.turn_radar_right(45)

	# ------------------------- State & policy -------------------------
	def _encode_state(self) -> str:
		# 1) Enemy distance bin: near, mid, far
		distance_bin = 2
		bearing_bin = 0
		scan_fresh = 0

		if self.last_scan is not None:
			dx = self.last_scan.x - self.x
			dy = self.last_scan.y - self.y
			distance = math.hypot(dx, dy)
			if distance < 150:
				distance_bin = 0
			elif distance < 400:
				distance_bin = 1

			abs_bearing = math.degrees(math.atan2(dx, dy))
			rel_bearing = _normalize_angle(abs_bearing - self.direction)
			if rel_bearing >= 0:
				bearing_bin = 0 if rel_bearing < 90 else 2
			else:
				bearing_bin = 1 if rel_bearing > -90 else 3

			# 2) Scan freshness: fresh if seen recently, else stale.
			scan_fresh = 1 if (self.local_tick - self.last_scan.turn) <= 10 else 0

		# 3) Wall proximity: safe vs near wall.
		nearest_wall = min(
			self.x,
			self.y,
			max(0.0, self.arena_width - self.x),
			max(0.0, self.arena_height - self.y),
		)
		wall_bin = 1 if nearest_wall < 80 else 0

		# 4) Gun readiness.
		gun_ready_bin = 1 if self.gun_heat <= 0.0001 else 0

		# 5) Own energy: low / medium / high.
		if self.energy < 25:
			energy_bin = 0
		elif self.energy < 60:
			energy_bin = 1
		else:
			energy_bin = 2

		return (
			f"d{distance_bin}|b{bearing_bin}|w{wall_bin}|"
			f"g{gun_ready_bin}|e{energy_bin}|s{scan_fresh}"
		)

	def _select_action(self, state: str) -> int:
		if random.random() < self.epsilon:
			return random.randint(0, self.ACTION_COUNT - 1)

		values = self.q[state]
		max_q = max(values)
		best_actions = [idx for idx, val in enumerate(values) if val == max_q]
		return random.choice(best_actions)

	# ------------------------- Control -------------------------
	def _execute_action(self, action: int) -> None:
		if action == self.STRAFE_LEFT:
			self.turn_left(30)
			self.forward(80)
			return

		if action == self.STRAFE_RIGHT:
			self.turn_right(30)
			self.forward(80)
			return

		if action == self.FORWARD:
			self.forward(100)
			return

		if action == self.BACKWARD:
			self.back(100)
			return

		if action == self.FIRE_LOW:
			self._aim_and_fire(1.0)
			return

		if action == self.FIRE_MEDIUM:
			self._aim_and_fire(2.0)

	def _aim_and_fire(self, power: float) -> None:
		self.fire_actions += 1

		if self.last_scan is not None:
			bearing = self.gun_bearing_to(self.last_scan.x, self.last_scan.y)
			self.turn_gun_left(bearing)

		if self.gun_heat <= 0.0001 and self.energy > (power + 0.1):
			self.fire(power)
			# Small anti-spam cost helps discourage blind firing.
			self.step_reward_accumulator -= 0.01

	# ------------------------- Learning -------------------------
	def _sarsa_update(
		self,
		state: str,
		action: int,
		reward: float,
		next_state: str,
		next_action: int,
		done: bool,
	) -> None:
		current_q = self.q[state][action]
		target = reward
		if not done:
			target += self.gamma * self.q[next_state][next_action]

		td_error = target - current_q
		self.q[state][action] = current_q + self.alpha * td_error
		self.q_update_abs_sum += abs(td_error)
		self.q_update_count += 1

	def _finalize_episode(self, won: bool) -> None:
		terminal_reward = 1.0 if won else -1.0
		self.step_reward_accumulator += terminal_reward

		if self.prev_state is not None and self.prev_action is not None:
			# Terminal update has no bootstrap term.
			reward = self.step_reward_accumulator
			current_q = self.q[self.prev_state][self.prev_action]
			td_error = reward - current_q
			self.q[self.prev_state][self.prev_action] = (
				current_q + self.alpha * td_error
			)
			self.q_update_abs_sum += abs(td_error)
			self.q_update_count += 1
			self.episode_reward += reward

		avg_abs_td = (
			self.q_update_abs_sum / self.q_update_count
			if self.q_update_count > 0
			else 0.0
		)

		log_row = {
			"episode": self.episode_number,
			"won": won,
			"epsilon": self.epsilon,
			"total_reward": self.episode_reward,
			"wall_hits": self.wall_hits,
			"fire_actions": self.fire_actions,
			"avg_abs_td_error": avg_abs_td,
			"turns": self.local_tick,
		}
		self._append_log(log_row)
		self._save_q_table()

		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
		print(
			f"[SarsaBot] End episode={self.episode_number} won={won} "
			f"reward={self.episode_reward:.3f} avg|td|={avg_abs_td:.4f}"
		)

	def _load_q_table(self) -> None:
		if not os.path.exists(self.q_table_path):
			return

		try:
			with open(self.q_table_path, "r", encoding="utf-8") as f:
				data = json.load(f)
			for state, values in data.items():
				if isinstance(values, list) and len(values) == self.ACTION_COUNT:
					self.q[state] = [float(v) for v in values]
			print(f"[SarsaBot] Loaded Q-table from {self.q_table_path}")
		except Exception as exc:
			print(f"[SarsaBot] Failed to load Q-table: {exc}")

	def _save_q_table(self) -> None:
		try:
			os.makedirs(os.path.dirname(self.q_table_path) or ".", exist_ok=True)
			with open(self.q_table_path, "w", encoding="utf-8") as f:
				json.dump(dict(self.q), f, indent=2)
		except Exception as exc:
			print(f"[SarsaBot] Failed to save Q-table: {exc}")

	def _append_log(self, row: dict) -> None:
		try:
			os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
			with open(self.log_path, "a", encoding="utf-8") as f:
				f.write(json.dumps(row) + "\n")
		except Exception as exc:
			print(f"[SarsaBot] Failed to append log: {exc}")

	# ------------------------- Events -------------------------
	def on_scanned_bot(self, scanned_bot_event: ScannedBotEvent) -> None:
		self.last_scan = LastScan(
			x=float(scanned_bot_event.x),
			y=float(scanned_bot_event.y),
			turn=self.local_tick,
		)

	def on_hit_by_bullet(self, hit_by_bullet_event: HitByBulletEvent) -> None:
		power = float(
			getattr(getattr(hit_by_bullet_event, "bullet", None), "power", 1.0)
		)
		damage = _bullet_damage(power)
		self.step_reward_accumulator += -0.025 * damage

	# Keep this name for compatibility with the current project's existing bot code.
	def on_bullet_hit(self, bullet_hit_bot_event) -> None:  # noqa: ANN001
		power = float(
			getattr(getattr(bullet_hit_bot_event, "bullet", None), "power", 1.0)
		)
		damage = _bullet_damage(power)
		self.step_reward_accumulator += 0.02 * damage

	def on_hit_wall(self, bot_hit_wall_event: HitWallEvent) -> None:
		del bot_hit_wall_event
		self.wall_hits += 1
		self.step_reward_accumulator -= 0.05

	def on_won_round(self, won_round_event: WonRoundEvent) -> None:
		del won_round_event
		self._finalize_episode(won=True)
		self.turn_right(360)

	def on_death(self, death_event: DeathEvent) -> None:
		del death_event
		self._finalize_episode(won=False)


def main() -> None:
	parser = argparse.ArgumentParser(description="Minimal SARSA Robocode bot")
	parser.add_argument("--alpha", type=float, default=0.10)
	parser.add_argument("--gamma", type=float, default=0.95)
	parser.add_argument("--epsilon", type=float, default=1.0)
	parser.add_argument("--epsilon-decay", type=float, default=0.995)
	parser.add_argument("--epsilon-min", type=float, default=0.05)
	parser.add_argument(
		"--q-table-path",
		default=str(DEFAULT_Q_TABLE_PATH),
	)
	parser.add_argument(
		"--log-path",
		default=str(DEFAULT_LOG_PATH),
	)

	args = parser.parse_args()

	bot = SarsaBot(
		alpha=args.alpha,
		gamma=args.gamma,
		epsilon=args.epsilon,
		epsilon_decay=args.epsilon_decay,
		epsilon_min=args.epsilon_min,
		q_table_path=args.q_table_path,
		log_path=args.log_path,
	)
	bot.start()


if __name__ == "__main__":
	main()
