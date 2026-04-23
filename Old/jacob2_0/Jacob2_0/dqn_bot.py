"""DQN bot for Robocode Tank Royale.

Structural mirror of sarsa_bot.py — same event hooks, same logging format,
same argparse interface. The Q-table is replaced by a GRU+MLP network that
lives in-process (no sidecar, no HTTP).

State vector (16 continuous features):
  [0]  self energy / 100
  [1]  self speed / 8
  [2]  sin(self direction)
  [3]  cos(self direction)
  [4]  self x / arena_width          — wall proximity
  [5]  self y / arena_height
  [6]  (arena_width  - x) / width
  [7]  (arena_height - y) / height
  [8]  sin(bearing to enemy)         — relative direction
  [9]  cos(bearing to enemy)
  [10] enemy distance / max_dist
  [11] enemy energy / 100
  [12] Δbearing / π                  — enemy velocity signal (no abs!)
  [13] Δdistance / max_dist
  [14] gun heat / 4                  — readiness (0 = ready)
  [15] scan freshness                — 1 if seen within 10 ticks, else 0

Actions (7):
  0  STRAFE_LEFT    1  STRAFE_RIGHT   2  FORWARD   3  BACKWARD
  4  FIRE_LOW       5  FIRE_MEDIUM    6  FIRE_HIGH
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import (
    DeathEvent,
    HitByBulletEvent,
    HitWallEvent,
    ScannedBotEvent,
    WonRoundEvent,
)

from dqn_agent import DQNAgent

STATE_DIM = 16
N_ACTIONS = 7


def _bullet_damage(power: float) -> float:
    return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


@dataclass
class LastScan:
    x: float
    y: float
    energy: float
    direction: float  # degrees
    turn: int


class DQNBot(Bot):
    STRAFE_LEFT = 0
    STRAFE_RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    FIRE_LOW = 4
    FIRE_MEDIUM = 5
    FIRE_HIGH = 6

    def __init__(self, agent_kwargs: dict) -> None:
        super().__init__()
        agent_kwargs = dict(agent_kwargs)  # don't mutate caller's dict
        self.log_path = agent_kwargs.pop("log_path", "training_log_dqn.jsonl")
        self.agent = DQNAgent(**agent_kwargs)

        self.local_tick = 0
        self.episode_number = 0

        self.last_scan: LastScan | None = None
        self.prev_bearing = 0.0  # for Δbearing
        self.prev_dist = 0.0  # for Δdistance

        self.prev_state: list[float] | None = None
        self.prev_action: int | None = None
        self.step_reward_accumulator = 0.0

        self.episode_reward = 0.0
        self.wall_hits = 0
        self.fire_actions = 0

        self.log_path: str = agent_kwargs.get("log_path", "training_log_dqn.jsonl")

    # Main loop — identical flow to SARSA bot
    def run(self) -> None:
        self.episode_number += 1
        self.local_tick = 0
        self.prev_state = None
        self.prev_action = None
        self.step_reward_accumulator = 0.0
        self.episode_reward = 0.0
        self.wall_hits = 0
        self.fire_actions = 0

        self.agent.on_episode_start()

        print(
            f"[DQNBot] Episode {self.episode_number} "
            f"ε={self.agent.epsilon:.4f} "
            f"buffer={len(self.agent.buffer)}"
        )

        while self.running:
            self.local_tick += 1

            state = self._encode_state()
            action = self.agent.select_action(state)

            # Push the PREVIOUS transition (now that we have next_state = state)
            if self.prev_state is not None and self.prev_action is not None:
                self.agent.push_and_train(
                    self.prev_state,
                    self.prev_action,
                    self.step_reward_accumulator,
                    state,
                    done=False,
                )
                self.episode_reward += self.step_reward_accumulator

            self.step_reward_accumulator = 0.0
            self._execute_action(action)

            self.prev_state = state
            self.prev_action = action

            self.turn_radar_right(45)

    # State encoding — continuous instead of discretized
    def _encode_state(self) -> list[float]:
        w = float(self.arena_width)
        h = float(self.arena_height)
        max_dist = math.hypot(w, h)

        bearing = 0.0
        dist = max_dist * 0.5
        e_energy = 100.0
        fresh = 0.0

        if self.last_scan is not None:
            dx = self.last_scan.x - self.x
            dy = self.last_scan.y - self.y
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dx, dy) - math.radians(self.direction)
            # Normalize to [-π, π]
            while bearing > math.pi:
                bearing -= 2 * math.pi
            while bearing < -math.pi:
                bearing += 2 * math.pi
            e_energy = self.last_scan.energy
            fresh = 1.0 if (self.local_tick - self.last_scan.turn) <= 10 else 0.0

        d_bearing = bearing - self.prev_bearing
        d_dist = dist - self.prev_dist
        while d_bearing > math.pi:
            d_bearing -= 2 * math.pi
        while d_bearing < -math.pi:
            d_bearing += 2 * math.pi

        self.prev_bearing = bearing
        self.prev_dist = dist

        spd = getattr(self, "speed", 0.0)

        return [
            self.energy / 100.0,
            spd / 8.0,
            math.sin(math.radians(self.direction)),
            math.cos(math.radians(self.direction)),
            self.x / w,
            self.y / h,
            (w - self.x) / w,
            (h - self.y) / h,
            math.sin(bearing),
            math.cos(bearing),
            dist / max_dist,
            e_energy / 100.0,
            d_bearing / math.pi,
            d_dist / max_dist,
            min(self.gun_heat / 4.0, 1.0),
            fresh,
        ]

    # Action execution — identical to SARSA bot
    def _execute_action(self, action: int) -> None:
        if action == self.STRAFE_LEFT:
            self.turn_left(30)
            self.forward(80)
        elif action == self.STRAFE_RIGHT:
            self.turn_right(30)
            self.forward(80)
        elif action == self.FORWARD:
            self.forward(100)
        elif action == self.BACKWARD:
            self.back(100)
        elif action == self.FIRE_LOW:
            self._aim_and_fire(1.0)
        elif action == self.FIRE_MEDIUM:
            self._aim_and_fire(2.0)
        elif action == self.FIRE_HIGH:
            self._aim_and_fire(3.0)

    def _aim_and_fire(self, power: float) -> None:
        self.fire_actions += 1
        if self.last_scan is not None:
            bearing = self.gun_bearing_to(self.last_scan.x, self.last_scan.y)
            self.turn_gun_left(bearing)
        if self.gun_heat <= 0.0001 and self.energy > (power + 0.1):
            self.fire(power)
            self.step_reward_accumulator -= 0.01  # small anti-spam cost

    # Terminal transition + logging — mirrors _finalize_episode
    def _finalize_episode(self, won: bool) -> None:
        terminal_reward = 1.0 if won else -1.0
        self.step_reward_accumulator += terminal_reward

        if self.prev_state is not None and self.prev_action is not None:
            self.agent.push_and_train(
                self.prev_state,
                self.prev_action,
                self.step_reward_accumulator,
                [0.0] * STATE_DIM,  # absorbing state
                done=True,
            )
            self.episode_reward += self.step_reward_accumulator

        stats = self.agent.on_episode_end(won)

        log_row = {
            "episode": self.episode_number,
            "won": won,
            "epsilon": stats["epsilon"],
            "total_reward": round(self.episode_reward, 3),
            "wall_hits": self.wall_hits,
            "fire_actions": self.fire_actions,
            "avg_abs_td_error": stats["avg_abs_td"],
            "turns": self.local_tick,
            "buffer_size": stats["buffer_size"],
            "win_rate": stats["win_rate"],
        }
        self._append_log(log_row)

        print(
            f"[DQNBot] End episode={self.episode_number} won={won} "
            f"reward={self.episode_reward:.3f} ε={stats['epsilon']:.4f} "
            f"win_rate={stats['win_rate']:.3f}"
        )

    def on_scanned_bot(self, e: ScannedBotEvent) -> None:
        self.last_scan = LastScan(
            x=float(e.x),
            y=float(e.y),
            energy=float(e.energy),
            direction=float(getattr(e, "direction", 0.0)),
            turn=self.local_tick,
        )

    def on_hit_by_bullet(self, e: HitByBulletEvent) -> None:
        power = float(getattr(getattr(e, "bullet", None), "power", 1.0))
        damage = _bullet_damage(power)
        self.step_reward_accumulator += -0.025 * damage

    def on_bullet_hit(self, e) -> None:  # noqa: ANN001
        power = float(getattr(getattr(e, "bullet", None), "power", 1.0))
        damage = _bullet_damage(power)
        self.step_reward_accumulator += 0.02 * damage

    def on_hit_wall(self, e: HitWallEvent) -> None:
        del e
        self.wall_hits += 1
        self.step_reward_accumulator -= 0.05

    def on_won_round(self, e: WonRoundEvent) -> None:
        del e
        self._finalize_episode(won=True)
        self.turn_right(360)

    def on_death(self, e: DeathEvent) -> None:
        del e
        self._finalize_episode(won=False)

    # ─────────────────────────────────────────────────────────────────
    # Logging — identical to SARSA bot
    # ─────────────────────────────────────────────────────────────────

    def _append_log(self, row: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as exc:
            print(f"[DQNBot] Log failed: {exc}")


# Entry point — same argparse pattern as SARSA bot
def main() -> None:
    parser = argparse.ArgumentParser(description="DQN Robocode Tank Royale bot")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--min-buffer", type=int, default=1_000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument(
        "--weights-path",
        default=os.path.join(os.path.dirname(__file__), "weights_dqn.pt"),
    )
    parser.add_argument(
        "--log-path",
        default=os.path.join(os.path.dirname(__file__), "training_log_dqn.jsonl"),
    )
    args = parser.parse_args()

    agent_kwargs = {
        "state_dim": STATE_DIM,
        "n_actions": N_ACTIONS,
        "gru_hidden": args.gru_hidden,
        "mlp_hidden": args.mlp_hidden,
        "gamma": args.gamma,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "min_buffer": args.min_buffer,
        "target_update_freq": args.target_update_freq,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "weights_path": args.weights_path,
        "log_path": args.log_path,
    }

    bot = DQNBot(agent_kwargs)
    bot.start()


if __name__ == "__main__":
    main()
