"""DQN-based RoboCode Tank Royale Bot.

Follows PyTorch tutorial architecture with:
- Continuous state encoding (16 features)
- 7 discrete actions (movement & firing)
- Experience replay & target network updates
- JSON logging of training progress
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

import numpy as np

STATE_DIM = 16
N_ACTIONS = 7


def _bullet_damage(power: float) -> float:
    """Calculate damage from bullet power."""
    return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


@dataclass
class LastScan:
    """Store last enemy scan."""

    x: float
    y: float
    energy: float
    turn: int


class DQNBot(Bot):
    """RoboCode Tank Royale bot using DQN agent."""

    # Actions
    STRAFE_LEFT = 0
    STRAFE_RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    FIRE_LOW = 4
    FIRE_MEDIUM = 5
    FIRE_HIGH = 6

    def __init__(
        self,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay_steps: int = 2500,
        tau: float = 0.005,
        batch_size: int = 128,
        memory_capacity: int = 10000,
        weights_path: str = "dqn_weights.pt",
        log_path: str = "dqn_training_log.jsonl",
        eval_mode: bool = False,
        eval_epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.eval_mode = eval_mode
        self.eval_episodes = 0
        self.eval_wins = 0
        self.agent = DQNAgent(
            n_observations=STATE_DIM,
            n_actions=N_ACTIONS,
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
            self.agent.set_eval_mode(epsilon=eval_epsilon)
        else:
            self.agent.set_train_mode()

        self.local_tick = 0
        self.last_scan: LastScan | None = None
        self.prev_bearing = 0.0
        self.prev_dist = 0.0

        self.prev_state: np.ndarray | None = None
        self.prev_action: int | None = None
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.episode_number = 0

    def run(self) -> None:
        """Main episode loop."""
        self.episode_number += 1
        if self.eval_mode:
            self.eval_episodes += 1
        self.local_tick = 0
        self.prev_state = None
        self.prev_action = None
        self.step_reward = 0.0
        self.episode_reward = 0.0

        if not self.eval_mode:
            self.agent.on_episode_start()

        print(
            f"[DQNBot] Episode {self.episode_number} | "
            f"mode={'eval' if self.eval_mode else 'train'} | "
            f"epsilon={self.agent.current_epsilon():.3f} | "
            f"buffer={len(self.agent.memory)}"
        )

        while self.running:
            self.local_tick += 1

            # Encode current state
            state = self._encode_state()
            action = self.agent.select_action(state)

            # Store previous transition
            if self.prev_state is not None and self.prev_action is not None:
                self.agent.push_transition(
                    self.prev_state, self.prev_action, state, self.step_reward, done=False
                )
                self.episode_reward += self.step_reward

            self.step_reward = 0.0
            self._execute_action(action)

            self.prev_state = state.copy()
            self.prev_action = action

            self.turn_radar_right(45)

    def _encode_state(self) -> np.ndarray:
        """Encode continuous state (16 features)."""
        w = float(self.arena_width)
        h = float(self.arena_height)
        max_dist = math.hypot(w, h)

        # Enemy info
        bearing = 0.0
        dist = max_dist * 0.5
        e_energy = 100.0
        fresh = 0.0

        if self.last_scan is not None:
            dx = self.last_scan.x - self.x
            dy = self.last_scan.y - self.y
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dx, dy) - math.radians(self.direction)
            while bearing > math.pi:
                bearing -= 2 * math.pi
            while bearing < -math.pi:
                bearing += 2 * math.pi
            e_energy = self.last_scan.energy
            fresh = 1.0 if (self.local_tick - self.last_scan.turn) <= 10 else 0.0

        # Delta features
        d_bearing = bearing - self.prev_bearing
        d_dist = dist - self.prev_dist
        while d_bearing > math.pi:
            d_bearing -= 2 * math.pi
        while d_bearing < -math.pi:
            d_bearing += 2 * math.pi

        self.prev_bearing = bearing
        self.prev_dist = dist

        spd = getattr(self, "speed", 0.0)

        state = np.array(
            [
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
            ],
            dtype=np.float32,
        )

        return state

    def _execute_action(self, action: int) -> None:
        """Execute discrete action."""
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
        """Aim and fire at enemy."""
        if self.last_scan is not None:
            bearing = self.gun_bearing_to(self.last_scan.x, self.last_scan.y)
            self.turn_gun_left(bearing)
        if self.gun_heat <= 0.0001 and self.energy > (power + 0.1):
            self.fire(power)
            self.step_reward -= 0.01

    def _finalize_episode(self, won: bool) -> None:
        """End episode and log results."""
        terminal_reward = 1.0 if won else -1.0
        self.step_reward += terminal_reward

        if self.eval_mode and won:
            self.eval_wins += 1

        if self.prev_state is not None and self.prev_action is not None:
            if not self.eval_mode:
                self.agent.push_transition(
                    self.prev_state,
                    self.prev_action,
                    np.zeros(STATE_DIM, dtype=np.float32),
                    self.step_reward,
                    done=True,
                )
            self.episode_reward += self.step_reward

        if not self.eval_mode:
            self.agent.on_episode_end(won)

        if self.eval_mode:
            win_rate = self.eval_wins / self.eval_episodes if self.eval_episodes > 0 else 0.0
        else:
            win_rate = self.agent.wins / self.agent.episode if self.agent.episode > 0 else 0.0

        log_row = {
            "episode": self.episode_number,
            "won": won,
            "total_reward": round(self.episode_reward, 3),
            "steps": self.local_tick,
            "mode": "eval" if self.eval_mode else "train",
            "epsilon": round(self.agent.current_epsilon(), 4),
            "buffer_size": len(self.agent.memory),
            "win_rate": round(win_rate, 3),
            "training_steps": self.agent.steps_done,
        }
        self._append_log(log_row)

        print(
            f"[DQNBot] Episode {self.episode_number} finished | "
            f"won={won} | reward={self.episode_reward:.2f} | "
            f"win_rate={win_rate:.3f}"
        )

    # ─── Events ───

    def on_scanned_bot(self, scanned_bot_event: ScannedBotEvent) -> None:
        self.last_scan = LastScan(
            x=float(scanned_bot_event.x),
            y=float(scanned_bot_event.y),
            energy=float(scanned_bot_event.energy),
            turn=self.local_tick,
        )

    def on_hit_by_bullet(self, hit_by_bullet_event: HitByBulletEvent) -> None:
        power = float(getattr(getattr(hit_by_bullet_event, "bullet", None), "power", 1.0))
        damage = _bullet_damage(power)
        self.step_reward -= 0.025 * damage

    def on_bullet_hit(self, bullet_hit_bot_event) -> None:
        power = float(getattr(getattr(bullet_hit_bot_event, "bullet", None), "power", 1.0))
        damage = _bullet_damage(power)
        self.step_reward += 0.02 * damage

    def on_hit_wall(self, bot_hit_wall_event: HitWallEvent) -> None:
        self.step_reward -= 0.05

    def on_won_round(self, won_round_event: WonRoundEvent) -> None:
        self._finalize_episode(won=True)
        self.turn_right(360)

    def on_death(self, death_event: DeathEvent) -> None:
        self._finalize_episode(won=False)

    # ─── Logging ───

    def _append_log(self, row: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            print(f"[DQNBot] Log failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN RoboCode Tank Royale Bot")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-start", type=float, default=0.9)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=2500)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-capacity", type=int, default=10000)
    parser.add_argument(
        "--weights-path",
        default=os.path.join(os.path.dirname(__file__), "dqn_weights.pt"),
    )
    parser.add_argument(
        "--log-path",
        default=os.path.join(os.path.dirname(__file__), "dqn_training_log.jsonl"),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run in evaluation mode (no online learning/checkpoint updates).",
    )
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.0,
        help="Fixed epsilon used during --eval (default: 0.0).",
    )
    args = parser.parse_args()

    bot = DQNBot(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        tau=args.tau,
        batch_size=args.batch_size,
        memory_capacity=args.memory_capacity,
        weights_path=args.weights_path,
        log_path=args.log_path,
        eval_mode=args.eval,
        eval_epsilon=args.eval_epsilon,
    )
    bot.start()


if __name__ == "__main__":
    main()
