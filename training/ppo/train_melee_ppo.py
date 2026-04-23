from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from bots.python.ppo.runtime.melee_env import (
    ACTION_BRANCH_SIZES,
    BattleSnapshot,
    DecodedAction,
    MeleeActionDecoder,
    MeleeObservationBuilder,
    MeleeRewardShaper,
    StickyTargetSelector,
)
from bots.python.ppo.agent.melee_ppo_agent import PPOTrainer


@dataclass(slots=True)
class EpisodeMetrics:
    episode_return: float
    survival_time: int
    final_placement: int
    damage_dealt: float
    damage_taken: float
    kills: int


class MeleeBattleEnv:
    """
    Exact Python-side environment API expected from the Java Robocode bot bridge.

    Java bot -> Python:
      reset(config: dict) -> BattleSnapshot
      step(decoded_action: DecodedAction) -> BattleSnapshot

    Python trainer responsibilities:
      1. select sticky target
      2. build normalized fixed-size observation
      3. sample PPO multi-discrete action
      4. decode action into Robocode controls
      5. compute shaped reward from battle deltas
    """

    def reset(self, config: dict) -> BattleSnapshot:
        raise NotImplementedError

    def step(self, action: DecodedAction) -> BattleSnapshot:
        raise NotImplementedError


CURRICULUM = [
    {"label": "2-bot", "enemy_count": 2, "rounds": 30, "opponents": ["SampleBots/Target"]},
    {"label": "4-bot", "enemy_count": 4, "rounds": 30, "opponents": ["SampleBots/SpinBot", "SampleBots/Walls"]},
    {"label": "6-bot", "enemy_count": 6, "rounds": 30, "opponents": ["SampleBots/Crazy", "SampleBots/TrackFire"]},
    {
        "label": "mixed",
        "enemy_count": 8,
        "rounds": 40,
        "opponents": [
            "SampleBots/SpinBot",
            "SampleBots/Walls",
            "SampleBots/Crazy",
            "SampleBots/TrackFire",
            "SampleBots/RamFire",
            "SampleBots/VelocityBot",
        ],
    },
]


def collect_rollout(env: MeleeBattleEnv, trainer: PPOTrainer, config: dict, horizon: int = 2048) -> tuple[dict[str, np.ndarray], EpisodeMetrics]:
    obs_builder = MeleeObservationBuilder()
    reward_shaper = MeleeRewardShaper(obs_builder)
    target_selector = StickyTargetSelector()
    decoder = MeleeActionDecoder(obs_builder)

    snapshot = env.reset(config)
    target_name = None

    obs_buffer = []
    action_buffer = []
    log_prob_buffer = []
    reward_buffer = []
    value_buffer = []
    done_buffer = []

    total_return = 0.0
    total_damage_dealt = 0.0
    total_damage_taken = 0.0
    total_kills = 0

    for _ in range(horizon):
        target_name = target_selector.select(snapshot, target_name).target_name
        obs, _ = obs_builder.build(snapshot, target_name)
        policy_out = trainer.act(obs)
        decoded_action = decoder.decode(tuple(int(x) for x in policy_out["action"]), snapshot, target_name)
        next_snapshot = env.step(decoded_action)
        reward = reward_shaper.compute(snapshot, next_snapshot, decoded_action)

        obs_buffer.append(obs)
        action_buffer.append(policy_out["action"])
        log_prob_buffer.append(policy_out["log_prob"])
        reward_buffer.append(reward.total)
        value_buffer.append(policy_out["value"])
        done_buffer.append(float(next_snapshot.done))

        total_return += reward.total
        total_damage_dealt += next_snapshot.bullet_damage_dealt
        total_damage_taken += next_snapshot.bullet_damage_taken
        total_kills += next_snapshot.kills_gained

        snapshot = next_snapshot
        if snapshot.done:
            break

    if snapshot.done:
        bootstrap_value = 0.0
    else:
        target_name = target_selector.select(snapshot, target_name).target_name
        bootstrap_obs, _ = obs_builder.build(snapshot, target_name)
        bootstrap_value = float(trainer.act(bootstrap_obs)["value"])

    rewards = np.asarray(reward_buffer, dtype=np.float32)
    values = np.asarray(value_buffer, dtype=np.float32)
    dones = np.asarray(done_buffer, dtype=np.float32)
    returns, advantages = trainer.finish_rollout(
        rewards=rewards,
        values=values,
        dones=dones,
        bootstrap_value=bootstrap_value,
        gamma=trainer.gamma,
        gae_lambda=trainer.gae_lambda,
    )

    rollout = {
        "obs": np.asarray(obs_buffer, dtype=np.float32),
        "actions": np.asarray(action_buffer, dtype=np.int64),
        "log_probs": np.asarray(log_prob_buffer, dtype=np.float32),
        "returns": returns,
        "advantages": advantages,
    }
    metrics = EpisodeMetrics(
        episode_return=float(total_return),
        survival_time=int(snapshot.tick),
        final_placement=int(snapshot.current_placement),
        damage_dealt=float(total_damage_dealt),
        damage_taken=float(total_damage_taken),
        kills=int(total_kills),
    )
    return rollout, metrics


def training_loop(env_factory, total_updates: int = 200, num_envs: int = 4) -> None:
    """
    Outline for stable melee PPO training.

    - Use multi-discrete PPO because Robocode decisions are naturally branched and bounded.
    - Keep target selection, wall smoothing, and aim geometry hard-coded for stability.
    - Train movement, turn timing, fire power, and radar bias with PPO.
    - Parallelize episode generation by running multiple Java battle workers in parallel.
    """

    trainer = PPOTrainer(branch_sizes=ACTION_BRANCH_SIZES)
    curriculum_idx = 0

    for update in range(1, total_updates + 1):
        if update in (50, 100, 150):
            curriculum_idx = min(curriculum_idx + 1, len(CURRICULUM) - 1)
        curriculum_config = CURRICULUM[curriculum_idx]

        rollouts = []
        metrics_batch = []
        envs = [env_factory(worker_id=i) for i in range(num_envs)]

        for env in envs:
            config = dict(curriculum_config)
            config["opponents"] = random.sample(
                curriculum_config["opponents"],
                k=min(len(curriculum_config["opponents"]), max(1, curriculum_config["enemy_count"] - 1)),
            )
            rollout, metrics = collect_rollout(env, trainer, config=config)
            rollouts.append(rollout)
            metrics_batch.append(metrics)

        merged = {
            "obs": np.concatenate([r["obs"] for r in rollouts], axis=0),
            "actions": np.concatenate([r["actions"] for r in rollouts], axis=0),
            "log_probs": np.concatenate([r["log_probs"] for r in rollouts], axis=0),
            "returns": np.concatenate([r["returns"] for r in rollouts], axis=0),
            "advantages": np.concatenate([r["advantages"] for r in rollouts], axis=0),
        }
        train_stats = trainer.update(merged)

        avg_return = float(np.mean([m.episode_return for m in metrics_batch]))
        avg_survival = float(np.mean([m.survival_time for m in metrics_batch]))
        avg_placement = float(np.mean([m.final_placement for m in metrics_batch]))
        avg_damage_dealt = float(np.mean([m.damage_dealt for m in metrics_batch]))
        avg_damage_taken = float(np.mean([m.damage_taken for m in metrics_batch]))
        avg_kills = float(np.mean([m.kills for m in metrics_batch]))

        print(
            {
                "update": update,
                "curriculum": curriculum_config["label"],
                "episode_return": round(avg_return, 3),
                "survival_time": round(avg_survival, 1),
                "final_placement": round(avg_placement, 2),
                "damage_dealt": round(avg_damage_dealt, 2),
                "damage_taken": round(avg_damage_taken, 2),
                "kills": round(avg_kills, 2),
                "entropy": round(train_stats.entropy, 4),
                "value_loss": round(train_stats.value_loss, 4),
                "policy_loss": round(train_stats.policy_loss, 4),
            }
        )
