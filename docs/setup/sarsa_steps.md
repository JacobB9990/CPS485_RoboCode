# Robocode RL (SARSA) Full Step-by-Step Plan

This is the exact workflow to get your baseline training running, debug it, and decide when to move on.

## 1. One-time setup

1. Install dependencies used by the Python sample bots.
2. Make sure your bot folder has the 3 required files:
	- `bots/python/sarsa/runtime/sarsa_bot.py` (your SARSA bot logic)
	- `<YourBotName>.json` (bot metadata so Robocode can load it)
	- `<YourBotName>.sh` (launcher script)
3. Use any sample bot folder as template for `.json` + `.sh`, then update:
	- Bot name
	- Description
	- Python module to launch (`bots.python.sarsa.runtime.sarsa_bot`)
4. Place your bot folder in the bots directory Robocode reads from.

## 2. Baseline hyperparameters (start here)

Use these defaults for the first training block:

1. `alpha = 0.10`
2. `gamma = 0.95`
3. `epsilon = 1.0`
4. `epsilon_decay = 0.995`
5. `epsilon_min = 0.05`

Do not tune yet. First prove the pipeline works.

## 3. Which sample bot to use first

Use this opponent curriculum in order:

1. `SpinBot` (first training opponent)
2. `TrackFire`
3. `Walls`
4. `RamFire`
5. `Crazy`

Why this order:

1. `SpinBot` gives stable, learnable early signal.
2. `TrackFire` tests aim pressure and movement timing.
3. `Walls` punishes poor positioning near boundaries.
4. `RamFire` adds aggressive close-range pressure.
5. `Crazy` tests robustness to erratic motion.

If you want a 5-minute smoke test before real training, run against `Target` once. Do not use it for meaningful training.

## 4. Episode definition and manual run protocol

1. One battle round = one episode.
2. Start with 1v1 only.
3. Keep arena and round settings fixed during a training block.
4. Run in blocks of 100 episodes per opponent.
5. Do not change reward/state/action definitions mid-block.

Suggested first block:

1. 200 episodes vs `SpinBot`
2. 200 episodes vs `TrackFire`

## 5. Files produced during training

Your bot writes:

1. `bots/python/sarsa/data/q_table_sarsa.json`: learned Q-values
2. `bots/python/sarsa/logs/training_log.jsonl`: one JSON line per episode

Back up these files before major experiments.

## 6. What to monitor every 50 episodes

Track these metrics:

1. Win rate (last 50 episodes)
2. Average episode reward
3. Damage dealt vs damage taken
4. Wall-hit count
5. Average absolute TD error (`avg_abs_td_error`)

Healthy signs:

1. Win rate trend rises (even slowly)
2. Wall hits trend down
3. TD error starts high, then generally settles

## 7. Early failure diagnosis (quick rules)

If no learning after 200 to 300 episodes:

1. Verify rewards are non-zero and both positive/negative events appear.
2. Verify state changes across time (not stuck in a few bins only).
3. Verify actions are diverse at high epsilon.
4. Check if gun can actually fire (gun heat + energy constraints).

If behavior is too passive:

1. Slightly increase hit reward scale.
2. Or reduce damage-taken penalty scale a little.

If behavior is reckless:

1. Increase damage-taken or wall penalty slightly.
2. Keep terminal loss penalty as-is until tested.

Change only one reward weight per experiment block.

## 8. Reproducible experiment structure

For each experiment, save:

1. Hyperparameters
2. Opponent bot
3. Episode range
4. Arena config
5. Result summary (win rate, reward trend, notes)

Use a simple naming convention, for example:

1. `exp01_spinbot_baseline`
2. `exp02_trackfire_baseline`

## 9. Promotion criteria before Phase 3

Move on only when all are true:

1. Logging is stable and complete.
2. Win rate is above random baseline on `SpinBot` and `TrackFire`.
3. No obvious degenerate policy (wall rubbing, never firing, blind spam).
4. Q-table is growing in visited states and not exploding numerically.

## 10. Practical daily routine

1. Run one 100-episode block.
2. Check metrics.
3. Make at most one controlled change.
4. Run another 100-episode block.
5. Write 3 to 5 bullet notes about what changed and why.

This keeps your RL work scientific instead of guess-and-check.
