from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import (
    DeathEvent,
    HitByBulletEvent,
    HitWallEvent,
    ScannedBotEvent,
    WonRoundEvent,
)

STATE_DIM = 16
N_ACTIONS = 7


# =========================
# Actor-Critic Network
# =========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)


# =========================
# PPO Agent
# =========================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.gamma = gamma
        self.clip_eps = clip_eps

        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Trajectory storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)

        logits, value = self.model(state_t)
        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())

        return action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_advantages(self):
        returns = []
        G = 0

        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(self.values)

        advantages = returns - values
        return returns, advantages

    def update(self):
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)

        returns, advantages = self.compute_returns_advantages()

        for _ in range(4):
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            policy_loss = -torch.min(ratio * advantages,
                                     clipped * advantages).mean()

            value_loss = (returns - values.squeeze()).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


# =========================
# Utility
# =========================
def _bullet_damage(power: float) -> float:
    return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


@dataclass
class LastScan:
    x: float
    y: float
    energy: float
    turn: int


# =========================
# PPO Bot
# =========================
class PPOBot(Bot):
    STRAFE_LEFT = 0
    STRAFE_RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    FIRE_LOW = 4
    FIRE_MEDIUM = 5
    FIRE_HIGH = 6

    def __init__(self):
        super().__init__()

        self.agent = PPOAgent(STATE_DIM, N_ACTIONS)

        self.local_tick = 0
        self.last_scan: LastScan | None = None
        self.prev_bearing = 0.0
        self.prev_dist = 0.0

        self.prev_state = None
        self.prev_action = None
        self.step_reward = 0.0

    def run(self):
        self.local_tick = 0
        self.prev_state = None
        self.prev_action = None
        self.step_reward = 0.0

        while self.running:
            self.local_tick += 1

            state = self._encode_state()
            action = self.agent.select_action(state)

            if self.prev_state is not None:
                self.agent.store_reward(self.step_reward, done=False)

            self.step_reward = 0.0
            self._execute_action(action)

            self.prev_state = state
            self.prev_action = action

            self.turn_radar_right(45)

    # =========================
    # STATE ENCODING (UNCHANGED)
    # =========================
    def _encode_state(self):
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

            while bearing > math.pi:
                bearing -= 2 * math.pi
            while bearing < -math.pi:
                bearing += 2 * math.pi

            e_energy = self.last_scan.energy
            fresh = 1.0 if (self.local_tick -
                            self.last_scan.turn) <= 10 else 0.0

        d_bearing = bearing - self.prev_bearing
        d_dist = dist - self.prev_dist

        self.prev_bearing = bearing
        self.prev_dist = dist

        spd = getattr(self, "speed", 0.0)

        return np.array([
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
        ], dtype=np.float32)

    # =========================
    # ACTIONS (UNCHANGED)
    # =========================
    def _execute_action(self, action):
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

    def _aim_and_fire(self, power):
        if self.last_scan is not None:
            bearing = self.gun_bearing_to(self.last_scan.x, self.last_scan.y)
            self.turn_gun_left(bearing)

        if self.gun_heat <= 0.0001 and self.energy > (power + 0.1):
            self.fire(power)
            self.step_reward -= 0.01

    # =========================
    # EVENTS
    # =========================
    def on_scanned_bot(self, e):
        self.last_scan = LastScan(e.x, e.y, e.energy, self.local_tick)

    def on_hit_by_bullet(self, e):
        self.step_reward -= 0.05

    def on_bullet_hit(self, e):
        self.step_reward += 0.05

    def on_hit_wall(self, e):
        self.step_reward -= 0.05

    def on_won_round(self, e):
        self._end_episode(True)

    def on_death(self, e):
        self._end_episode(False)

    def _end_episode(self, won):
        final_reward = 1.0 if won else -1.0
        self.step_reward += final_reward

        self.agent.store_reward(self.step_reward, done=True)
        self.agent.update()


# =========================
# MAIN
# =========================
def main():
    bot = PPOBot()
    bot.start()


if __name__ == "__main__":
    main()
