from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import BotDeathEvent, DeathEvent, HitByBulletEvent, HitWallEvent, ScannedBotEvent, WonRoundEvent

STATE_DIM = 16
N_ACTIONS = 7
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = ROOT / "checkpoints" / "ppo_weights.pt"
DEFAULT_LOG_PATH = ROOT / "logs" / "ppo_training_log.jsonl"


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(x)
        return self.policy_head(hidden), self.value_head(hidden)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        weights_path: str = str(DEFAULT_WEIGHTS_PATH),
        eval_mode: bool = False,
    ) -> None:
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.weights_path = weights_path
        self.eval_mode = eval_mode

        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []

        self.last_policy_loss: float | None = None
        self.last_value_loss: float | None = None
        self.last_entropy: float | None = None

        self._load()
        if self.eval_mode:
            self.model.eval()
        else:
            self.model.train()

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(state_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if self.eval_mode:
            action = int(torch.argmax(probs, dim=-1).item())
            log_prob = float(dist.log_prob(torch.tensor(action)).item())
            entropy = float(dist.entropy().item())
        else:
            sampled = dist.sample()
            action = int(sampled.item())
            log_prob = float(dist.log_prob(sampled).item())
            entropy = float(dist.entropy().item())
            self.states.append(state.copy())
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(float(value.item()))
        self.last_entropy = entropy
        return action

    def store_reward(self, reward: float, done: bool) -> None:
        if self.eval_mode:
            return
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        returns = []
        cumulative = 0.0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                cumulative = 0.0
            cumulative = reward + (self.gamma * cumulative)
            returns.insert(0, cumulative)

        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        values_t = torch.as_tensor(self.values, dtype=torch.float32)
        return returns_t, (returns_t - values_t)

    def update(self) -> None:
        if self.eval_mode or not self.states:
            return

        states = torch.as_tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.as_tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32)
        returns, advantages = self.compute_returns_advantages()

        for _ in range(4):
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + (0.5 * value_loss) - (0.01 * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.last_policy_loss = float(policy_loss.item())
            self.last_value_loss = float(value_loss.item())
            self.last_entropy = float(entropy.item())

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self._save()

    def _save(self) -> None:
        path = Path(self.weights_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "last_policy_loss": self.last_policy_loss,
                "last_value_loss": self.last_value_loss,
                "last_entropy": self.last_entropy,
            },
            path,
        )

    def _load(self) -> None:
        if not os.path.exists(self.weights_path):
            return
        try:
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state and not self.eval_mode:
                self.optimizer.load_state_dict(optimizer_state)
            self.last_policy_loss = checkpoint.get("last_policy_loss")
            self.last_value_loss = checkpoint.get("last_value_loss")
            self.last_entropy = checkpoint.get("last_entropy")
        except Exception as exc:
            print(f"[PPOBot] Load failed: {exc}")


def _bullet_damage(power: float) -> float:
    return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


@dataclass
class LastScan:
    x: float
    y: float
    energy: float
    turn: int


class PPOBot(Bot):
    STRAFE_LEFT = 0
    STRAFE_RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    FIRE_LOW = 4
    FIRE_MEDIUM = 5
    FIRE_HIGH = 6

    def __init__(self, weights_path: str, log_path: str, eval_mode: bool = False) -> None:
        super().__init__()
        self.log_path = log_path
        self.eval_mode = eval_mode
        self.agent = PPOAgent(STATE_DIM, N_ACTIONS, weights_path=weights_path, eval_mode=eval_mode)

        self.local_tick = 0
        self.last_scan: LastScan | None = None
        self.prev_bearing = 0.0
        self.prev_dist = 0.0
        self.prev_state: np.ndarray | None = None
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.episode_number = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.kills = 0
        self.wall_hits = 0
        self.fire_actions = 0
        self.initial_bot_count = 0
        self.last_damaged_enemy_name: str | None = None
        self.last_damage_tick = -999

    def run(self) -> None:
        self.episode_number += 1
        self.local_tick = 0
        self.prev_state = None
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.kills = 0
        self.wall_hits = 0
        self.fire_actions = 0
        self.initial_bot_count = int(getattr(self, "enemy_count", 0)) + 1
        self.last_damaged_enemy_name = None
        self.last_damage_tick = -999

        while self.running:
            self.local_tick += 1
            state = self._encode_state()
            action = self.agent.select_action(state)

            if self.prev_state is not None:
                self.agent.store_reward(self.step_reward, done=False)
                self.episode_reward += self.step_reward

            self.step_reward = 0.0
            self._execute_action(action)
            self.prev_state = state
            self.turn_radar_right(45)

    def _encode_state(self) -> np.ndarray:
        w = float(self.arena_width)
        h = float(self.arena_height)
        max_dist = math.hypot(w, h)

        bearing = 0.0
        dist = max_dist * 0.5
        enemy_energy = 100.0
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
            enemy_energy = self.last_scan.energy
            fresh = 1.0 if (self.local_tick - self.last_scan.turn) <= 10 else 0.0

        d_bearing = bearing - self.prev_bearing
        d_dist = dist - self.prev_dist
        self.prev_bearing = bearing
        self.prev_dist = dist
        speed = getattr(self, "speed", 0.0)

        return np.array(
            [
                self.energy / 100.0,
                speed / 8.0,
                math.sin(math.radians(self.direction)),
                math.cos(math.radians(self.direction)),
                self.x / w,
                self.y / h,
                (w - self.x) / w,
                (h - self.y) / h,
                math.sin(bearing),
                math.cos(bearing),
                dist / max_dist,
                enemy_energy / 100.0,
                d_bearing / math.pi,
                d_dist / max_dist,
                min(self.gun_heat / 4.0, 1.0),
                fresh,
            ],
            dtype=np.float32,
        )

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
        if self.last_scan is not None:
            bearing = self.gun_bearing_to(self.last_scan.x, self.last_scan.y)
            self.turn_gun_left(bearing)
        if self.gun_heat <= 0.0001 and self.energy > (power + 0.1):
            self.fire(power)
            self.fire_actions += 1
            self.step_reward -= 0.01

    def on_scanned_bot(self, event: ScannedBotEvent) -> None:
        self.last_scan = LastScan(float(event.x), float(event.y), float(event.energy), self.local_tick)

    def on_hit_by_bullet(self, event: HitByBulletEvent) -> None:
        power = float(getattr(getattr(event, "bullet", None), "power", 1.0))
        self.damage_taken += _bullet_damage(power)
        self.step_reward -= 0.05

    def on_bullet_hit(self, event) -> None:  # noqa: ANN001
        power = float(getattr(getattr(event, "bullet", None), "power", 1.0))
        self.damage_dealt += _bullet_damage(power)
        self.last_damaged_enemy_name = getattr(event, "name", None)
        self.last_damage_tick = self.local_tick
        self.step_reward += 0.05

    def on_bot_death(self, event: BotDeathEvent) -> None:
        name = getattr(event, "name", None)
        victim_name = getattr(event, "victim_name", None)
        if name == self.last_damaged_enemy_name or victim_name == self.last_damaged_enemy_name:
            if (self.local_tick - self.last_damage_tick) <= 2:
                self.kills += 1

    def on_hit_wall(self, event: HitWallEvent) -> None:
        del event
        self.wall_hits += 1
        self.step_reward -= 0.05

    def on_won_round(self, event: WonRoundEvent) -> None:
        del event
        self._end_episode(True)

    def on_death(self, event: DeathEvent) -> None:
        del event
        self._end_episode(False)

    def _end_episode(self, won: bool) -> None:
        final_reward = 1.0 if won else -1.0
        self.step_reward += final_reward
        self.episode_reward += self.step_reward
        self.agent.store_reward(self.step_reward, done=True)
        self.agent.update()

        row = {
            "episode": self.episode_number,
            "won": won,
            "placement": 1 if won else int(getattr(self, "enemy_count", 0)) + 1,
            "total_bots": max(self.initial_bot_count, int(getattr(self, "enemy_count", 0)) + 1),
            "steps": self.local_tick,
            "total_reward": round(self.episode_reward, 3),
            "policy_loss": self.agent.last_policy_loss,
            "value_loss": self.agent.last_value_loss,
            "entropy": self.agent.last_entropy,
            "damage_dealt": round(self.damage_dealt, 3),
            "damage_taken": round(self.damage_taken, 3),
            "kills": self.kills,
            "fire_actions": self.fire_actions,
            "wall_hits": self.wall_hits,
            "mode": "eval" if self.eval_mode else "train",
        }
        self._append_log(row)

    def _append_log(self, row: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except Exception as exc:
            print(f"[PPOBot] Log failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PPOBot runtime")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    PPOBot(weights_path=args.weights_path, log_path=args.log_path, eval_mode=args.eval).start()


if __name__ == "__main__":
    main()
