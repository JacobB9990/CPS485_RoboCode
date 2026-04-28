from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch

import robocode_compat  # noqa: F401
from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import (
    BotDeathEvent,
    DeathEvent,
    HitByBulletEvent,
    HitWallEvent,
    ScannedBotEvent,
    WonRoundEvent,
)

from PPOBotAdvanced.agent.melee_ppo_agent import PPOBatchStats, PPOTrainer
from PPOBotAdvanced.runtime.melee_env import (
    ACTION_BRANCH_SIZES,
    BattleSnapshot,
    DecodedAction,
    EnemyState,
    MeleeActionDecoder,
    MeleeObservationBuilder,
    MeleeRewardShaper,
    SelfState,
    StickyTargetSelector,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = ROOT / "checkpoints" / "ppo_weights.pt"
DEFAULT_LOG_PATH = ROOT / "logs" / "ppo_training_log.jsonl"
CHECKPOINT_BACKUP_INTERVAL = 200


def _bullet_damage(power: float) -> float:
    return 4.0 * power + max(0.0, 2.0 * (power - 1.0))


class PPOBot(Bot):
    def __init__(
        self,
        weights_path: str,
        log_path: str,
        eval_mode: bool = False,
        read_only_weights: bool = False,
    ) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.log_path = log_path
        self.eval_mode = eval_mode
        self.read_only_weights = read_only_weights

        self.trainer = PPOTrainer(branch_sizes=ACTION_BRANCH_SIZES)
        self._obs_builder = MeleeObservationBuilder()
        self._reward_shaper = MeleeRewardShaper(self._obs_builder)
        self._target_selector = StickyTargetSelector()
        self._decoder = MeleeActionDecoder(self._obs_builder)

        self._load_weights()
        if self.eval_mode:
            self.trainer.model.eval()

        # Enemy tracking — keyed by string name (from scanned_bot_id via compat)
        self._enemies: dict[str, EnemyState] = {}
        self._current_target: str | None = None

        # Previous step state for reward computation
        self._prev_snapshot: BattleSnapshot | None = None
        self._prev_decoded: DecodedAction | None = None

        # Per-step event accumulators — reset before each decision
        self._step_damage_dealt: float = 0.0
        self._step_damage_taken: float = 0.0
        self._step_kills: int = 0
        self._step_hit_wall: bool = False
        self._step_fired_power: float = 0.0
        self._step_bullet_hit: bool = False

        # Kill attribution: only credit kills where we dealt the last bullet hit
        self._last_damaged_enemy_name: str | None = None
        self._last_damage_tick: int = 0

        # Rollout buffers — accumulated per episode, cleared on episode start
        self._obs_buf: list[np.ndarray] = []
        self._action_buf: list[np.ndarray] = []
        self._log_prob_buf: list[float] = []
        self._value_buf: list[float] = []
        self._reward_buf: list[float] = []
        self._done_buf: list[float] = []

        # Episode metrics
        self.episode_number: int = 0
        self.episode_reward: float = 0.0
        self.total_damage_dealt: float = 0.0
        self.total_damage_taken: float = 0.0
        self.total_kills: int = 0
        self.total_wall_hits: int = 0
        self.total_fire_actions: int = 0
        self.local_tick: int = 0
        self.initial_bot_count: int = 0
        self._last_stats: PPOBatchStats | None = None

        self._opponent = os.environ.get("PPO_OPPONENT", "unknown")
        self._scenario = os.environ.get("PPO_SCENARIO", "unknown")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.episode_number += 1
        self.local_tick = 0
        self._enemies.clear()
        self._current_target = None
        self._prev_snapshot = None
        self._prev_decoded = None
        self._obs_buf.clear()
        self._action_buf.clear()
        self._log_prob_buf.clear()
        self._value_buf.clear()
        self._reward_buf.clear()
        self._done_buf.clear()
        self.episode_reward = 0.0
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.total_kills = 0
        self.total_wall_hits = 0
        self.total_fire_actions = 0
        self.initial_bot_count = int(getattr(self, "enemy_count", 0)) + 1
        self._last_damaged_enemy_name = None
        self._last_damage_tick = 0
        self._reset_step_events()

        while self.running:
            self.local_tick += 1
            snapshot = self._build_snapshot(done=False, won=False)

            # Compute reward for the transition from the previous decision
            if self._prev_snapshot is not None and self._prev_decoded is not None:
                reward = self._reward_shaper.compute(
                    self._prev_snapshot, snapshot, self._prev_decoded
                )
                self._reward_buf.append(reward.total)
                self._done_buf.append(0.0)
                self.episode_reward += reward.total

            self._reset_step_events()

            target_sel = self._target_selector.select(snapshot, self._current_target)
            self._current_target = target_sel.target_name

            obs, _ = self._obs_builder.build(snapshot, self._current_target)
            policy_out = self.trainer.act(obs)
            decoded = self._decoder.decode(
                tuple(int(x) for x in policy_out["action"]),
                snapshot,
                self._current_target,
            )

            self._obs_buf.append(obs)
            self._action_buf.append(policy_out["action"])
            self._log_prob_buf.append(policy_out["log_prob"])
            self._value_buf.append(policy_out["value"])

            self._prev_snapshot = snapshot
            self._prev_decoded = decoded
            self._execute_decoded(decoded)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def _build_snapshot(self, *, done: bool, won: bool) -> BattleSnapshot:
        self_state = SelfState(
            energy=float(self.energy),
            x=float(self.x),
            y=float(self.y),
            velocity=float(getattr(self, "speed", 0.0)),
            heading=math.radians(float(self.direction)),
            gun_heading=math.radians(float(getattr(self, "gun_direction", self.direction))),
            gun_heat=float(self.gun_heat),
        )
        alive_count = sum(1 for e in self._enemies.values() if e.alive)
        # alive_count + 1 approximates our current placement rank:
        # decreases as enemies die, triggering the placement-improvement reward.
        current_placement = alive_count + 1

        return BattleSnapshot(
            tick=self.local_tick,
            arena_width=float(self.arena_width),
            arena_height=float(self.arena_height),
            self_state=self_state,
            enemies=dict(self._enemies),
            alive_enemy_count=alive_count,
            current_placement=current_placement,
            bullet_damage_dealt=self._step_damage_dealt,
            bullet_damage_taken=self._step_damage_taken,
            kills_gained=self._step_kills,
            hit_wall=self._step_hit_wall,
            fired_power=self._step_fired_power,
            bullet_hit=self._step_bullet_hit,
            won=won,
            done=done,
        )

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute_decoded(self, action: DecodedAction) -> None:
        # Aim gun first (blocking) so firing happens with correct aim
        gun_deg = math.degrees(action.gun_turn_radians)
        if gun_deg > 0.5:
            self.turn_gun_right(gun_deg)
        elif gun_deg < -0.5:
            self.turn_gun_left(-gun_deg)

        if (
            action.fire_power > 0.0
            and self.gun_heat <= 0.001
            and float(self.energy) > action.fire_power + 0.1
        ):
            self.fire(action.fire_power)
            self.total_fire_actions += 1
            self._step_fired_power = action.fire_power

        # Radar (non-blocking set — executes during next body/move operation)
        radar_deg = math.degrees(action.radar_turn_radians)
        if radar_deg > 0.5:
            self.set_turn_radar_right(radar_deg)
        elif radar_deg < -0.5:
            self.set_turn_radar_left(-radar_deg)

        # Body turn then move (both blocking)
        turn_deg = math.degrees(action.body_turn_radians)
        if turn_deg > 0.5:
            self.turn_right(turn_deg)
        elif turn_deg < -0.5:
            self.turn_left(-turn_deg)

        if action.move_distance > 0.0 and action.movement_mode != "hold":
            self.forward(action.move_distance)

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_scanned_bot(self, event: ScannedBotEvent) -> None:
        name = str(getattr(event, "name", getattr(event, "scanned_bot_id", "unknown")))
        ex, ey = float(event.x), float(event.y)
        dx, dy = ex - float(self.x), ey - float(self.y)
        dist = math.hypot(dx, dy)
        abs_bearing = math.atan2(dx, dy)
        # robocode_compat already computes relative bearing and puts it in event.bearing
        rel_bearing = float(getattr(event, "bearing", abs_bearing - math.radians(float(self.direction))))
        heading_deg = float(getattr(event, "direction_degrees", getattr(event, "direction", 0.0)))
        self._enemies[name] = EnemyState(
            name=name,
            x=ex,
            y=ey,
            distance=dist,
            abs_bearing=abs_bearing,
            relative_bearing=rel_bearing,
            velocity=float(getattr(event, "velocity", getattr(event, "speed", 0.0))),
            heading=math.radians(heading_deg),
            energy=float(event.energy),
            last_seen_tick=self.local_tick,
            alive=True,
        )

    def on_hit_by_bullet(self, event: HitByBulletEvent) -> None:
        power = float(getattr(getattr(event, "bullet", None), "power", 1.0))
        dmg = _bullet_damage(power)
        self._step_damage_taken += dmg
        self.total_damage_taken += dmg

    def on_bullet_hit_bot(self, event) -> None:  # noqa: ANN001
        self._handle_bullet_hit(event)

    def on_bullet_hit(self, event) -> None:  # noqa: ANN001  legacy alias
        self._handle_bullet_hit(event)

    def _handle_bullet_hit(self, event) -> None:  # noqa: ANN001
        power = float(getattr(getattr(event, "bullet", None), "power", 1.0))
        dmg = _bullet_damage(power)
        self._step_damage_dealt += dmg
        self._step_bullet_hit = True
        self.total_damage_dealt += dmg
        victim = str(getattr(event, "name", getattr(event, "victim_id", None)) or "")
        if victim:
            self._last_damaged_enemy_name = victim
            self._last_damage_tick = self.local_tick

    def on_bot_death(self, event: BotDeathEvent) -> None:
        name = str(getattr(event, "name", getattr(event, "victim_id", "unknown")))
        if name in self._enemies:
            self._enemies[name].alive = False
        if name == self._last_damaged_enemy_name and (self.local_tick - self._last_damage_tick) <= 3:
            self._step_kills += 1
            self.total_kills += 1

    def on_hit_wall(self, event: HitWallEvent) -> None:
        del event
        self._step_hit_wall = True
        self.total_wall_hits += 1

    def on_won_round(self, event: WonRoundEvent) -> None:
        del event
        self._end_episode(won=True)

    def on_death(self, event: DeathEvent) -> None:
        del event
        self._end_episode(won=False)

    # ── Episode end ───────────────────────────────────────────────────────────

    def _end_episode(self, *, won: bool) -> None:
        # Compute terminal reward for the last action
        if self._prev_snapshot is not None and self._prev_decoded is not None:
            terminal = self._build_snapshot(done=True, won=won)
            reward = self._reward_shaper.compute(
                self._prev_snapshot, terminal, self._prev_decoded
            )
            self._reward_buf.append(reward.total)
            self._done_buf.append(1.0)
            self.episode_reward += reward.total

        stats: PPOBatchStats | None = None
        if not self.eval_mode and len(self._obs_buf) >= 4:
            print(f"[PPOBotAdvanced] ep={self.episode_number} rollout_size={len(self._obs_buf)}")
            rewards_arr = np.array(self._reward_buf, dtype=np.float32)
            values_arr = np.array(self._value_buf, dtype=np.float32)
            dones_arr = np.array(self._done_buf, dtype=np.float32)
            # bootstrap=0.0 is correct: this path is only reached via on_won_round or on_death
            returns_arr, advantages_arr = PPOTrainer.finish_rollout(
                rewards=rewards_arr,
                values=values_arr,
                dones=dones_arr,
                bootstrap_value=0.0,
                gamma=self.trainer.gamma,
                gae_lambda=self.trainer.gae_lambda,
            )
            rollout = {
                "obs": np.array(self._obs_buf, dtype=np.float32),
                "actions": np.array(self._action_buf, dtype=np.int64),
                "log_probs": np.array(self._log_prob_buf, dtype=np.float32),
                "returns": returns_arr,
                "advantages": advantages_arr,
            }
            stats = self.trainer.update(rollout)
            self._last_stats = stats

            if not self.read_only_weights:
                self._save_weights()
                if self.episode_number % CHECKPOINT_BACKUP_INTERVAL == 0:
                    self._backup_weights(f"ep{self.episode_number}")

        placement = 1 if won else (int(getattr(self, "enemy_count", 0)) + 1)
        row = {
            "episode": self.episode_number,
            "won": won,
            "placement": placement,
            "total_bots": max(
                self.initial_bot_count,
                int(getattr(self, "enemy_count", 0)) + 1,
            ),
            "steps": self.local_tick,
            "total_reward": round(self.episode_reward, 4),
            "policy_loss": round(stats.policy_loss, 6) if stats else None,
            "value_loss": round(stats.value_loss, 6) if stats else None,
            "entropy": round(stats.entropy, 6) if stats else None,
            "damage_dealt": round(self.total_damage_dealt, 3),
            "damage_taken": round(self.total_damage_taken, 3),
            "kills": self.total_kills,
            "fire_actions": self.total_fire_actions,
            "wall_hits": self.total_wall_hits,
            "mode": "eval" if self.eval_mode else "train",
            "opponent": self._opponent,
            "scenario": self._scenario,
        }
        self._append_log(row)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _reset_step_events(self) -> None:
        self._step_damage_dealt = 0.0
        self._step_damage_taken = 0.0
        self._step_kills = 0
        self._step_hit_wall = False
        self._step_fired_power = 0.0
        self._step_bullet_hit = False

    def _save_weights(self) -> None:
        path = Path(self.weights_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": self.trainer.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
            },
            tmp,
        )
        os.replace(tmp, path)

    def _backup_weights(self, tag: str) -> None:
        src = Path(self.weights_path)
        if src.exists():
            shutil.copy2(src, src.parent / f"ppo_weights_{tag}.pt")

    def _load_weights(self) -> None:
        path = Path(self.weights_path)
        if not path.exists():
            return
        try:
            ckpt = torch.load(path, map_location="cpu")
            self.trainer.model.load_state_dict(ckpt["model"])
            if not self.eval_mode and "optimizer" in ckpt:
                self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[PPOBotAdvanced] Loaded weights from {path}")
        except Exception as exc:
            print(f"[PPOBotAdvanced] Fresh start — could not load weights: {exc}")

    def _append_log(self, row: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except Exception as exc:
            print(f"[PPOBotAdvanced] Log failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PPOBotAdvanced runtime")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--read-only-weights", action="store_true")
    args = parser.parse_args()

    PPOBot(
        weights_path=args.weights_path,
        log_path=args.log_path,
        eval_mode=args.eval,
        read_only_weights=args.read_only_weights,
    ).start()


if __name__ == "__main__":
    main()
