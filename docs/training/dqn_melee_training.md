# Melee Training Setup

This folder adds a melee-oriented training scaffold on top of the existing headless runner.
It is designed for a main bot implemented in Java, with Python handling battle scheduling,
self-play pool management, evaluation, and checkpoint selection.

## Directory Structure

```text
Jacob/Jacob3_0/
├── headless_runner/
│   ├── run_battle.sh
│   ├── run_melee_battle.sh
│   └── src/
│       ├── RunBattle.java
│       └── RunMeleeBattle.java
├── melee_training/
│   ├── README.md
│   ├── __init__.py
│   ├── config.py
│   ├── evaluation.py
│   ├── orchestrator.py
│   └── pool.py
├── checkpoints/
│   └── checkpoint_index.jsonl
├── logs/
│   └── melee/
│       ├── train_history.jsonl
│       ├── eval_history.jsonl
│       ├── evaluation_summary.jsonl
│       └── telemetry/
└── run_melee_training.py
```

## Training Script Design

`run_melee_training.py` supports four modes:

- `plan-train`: build a randomized curriculum schedule without executing battles.
- `run-train`: execute the full curriculum and write JSONL battle logs.
- `plan-eval`: show the fixed evaluation suite.
- `run-eval`: execute the evaluation suite and compute checkpoint metrics.

The scheduler always includes the current bot and samples 2 to 10 opponents from:

- fixed benchmark bots
- older checkpoint bots from `checkpoint_index.jsonl`
- random baseline bots

The curriculum intentionally widens over time:

- `stage_1_foundation`: 2 to 4 opponents, weak and medium bots only
- `stage_2_mixed_melee`: 4 to 7 opponents, more checkpoints and chaotic mixes
- `stage_3_hard_chaos`: 6 to 10 opponents, strong and chaotic mixes dominate

## Self-Play Pool

The self-play pool is split into three buckets:

- `current_bot_dir`: the actively trained bot
- checkpoint pool: older snapshots registered in `checkpoints/checkpoint_index.jsonl`
- baseline pool: simple scripted bots that keep the population from collapsing into mirror play

Each checkpoint row is expected to look like:

```json
{"name":"MyBot_ckpt_120000","bot_dir":"/abs/path/to/materialized/bot","step":120000,"tier":"checkpoint"}
```

For a Java bot, the usual pattern is to materialize each checkpoint into its own runnable bot
directory with a unique bot name/version and a launch script that points at the saved model.

## Evaluation And Checkpoint Selection

The evaluation suite is fixed and melee-specific. Each scenario has explicit participants and
round counts so checkpoint comparisons stay apples-to-apples.

Primary selection metric:

- `average_placement`

Secondary metrics recorded:

- `win_rate`
- `average_survival_score`
- `average_damage_score`
- `average_kill_bonus`

Optional bot telemetry can add true melee metrics:

- `survival_time`
- `damage`
- `kills`

To enable that, have your Java bot write JSONL rows to `logs/melee/telemetry/` with fields like:

```json
{"battle_id":"eval_8way_chaos","survival_time":412,"damage":86.5,"kills":2}
```

Without telemetry, the runner still provides placement and Robocode score components, which are
enough for checkpoint ranking by placement.

## Battle Scheduling Plan

Use training in short slices, then evaluate before promoting a checkpoint:

1. Run one curriculum pass.
2. Save a checkpoint and materialize a bot directory for it.
3. Register that checkpoint in `checkpoint_index.jsonl`.
4. Run the fixed evaluation suite.
5. Keep the best checkpoints by lowest `average_placement`.

Recommended cadence:

- train with 40 to 60 rounds per sampled melee
- evaluate with 100 to 150 rounds per fixed scenario
- add newly promoted checkpoints back into the self-play pool
- keep only a spaced subset of checkpoints using `min_checkpoint_gap`

## Anti-Overfitting Recommendations

- Keep benchmark bots in every stage so the policy does not specialize only to self-play.
- Mix weak, medium, strong, and chaotic movers instead of using one dominant population.
- Freeze the evaluation suite; never tune directly against random eval opponents.
- Space checkpoint snapshots apart in training steps so the pool contains behavioral diversity.
- Measure placement, not just wins, because melee progress often shows up first as more second and third places.
- Rotate random seeds for training schedules while keeping evaluation seeds and scenarios fixed.

## Quick Start

Dry-run the schedule:

```bash
python3 run_melee_training.py --mode plan-train
python3 run_melee_training.py --mode plan-eval
```

Run the trainer:

```bash
python3 run_melee_training.py --mode run-train
python3 run_melee_training.py --mode run-eval
```

Replace `current_bot_dir` in `melee_config.example.json` with your Java bot directory or wrapper
once the Java bot is ready to participate in headless battles.
