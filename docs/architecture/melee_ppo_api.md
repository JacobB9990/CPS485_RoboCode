# Melee PPO Environment API

The PPO melee upgrade assumes a Java Robocode bot handles simulation and a Python trainer handles policy learning.

## Python environment contract

`reset(config: dict) -> BattleSnapshot`

`step(action: DecodedAction) -> BattleSnapshot`

## `BattleSnapshot` fields

- `tick`: current turn within the round
- `arena_width`, `arena_height`
- `self_state`: `energy`, `x`, `y`, `velocity`, `heading`, `gun_heading`, `gun_heat`
- `enemies`: map of enemy name to tracked enemy state
- `alive_enemy_count`
- `current_placement`
- `bullet_damage_dealt`
- `bullet_damage_taken`
- `kills_gained`
- `hit_wall`
- `fired_power`
- `bullet_hit`
- `won`
- `done`

## `EnemyState` fields

- `name`
- `x`, `y`
- `distance`
- `abs_bearing`
- `relative_bearing`
- `velocity`
- `heading`
- `energy`
- `last_seen_tick`
- `alive`

## Why multi-discrete PPO

The action is split into four categorical branches:

- `movement action`
- `turn action`
- `fire power action`
- `radar action`

This is simpler and usually more stable than a continuous Gaussian policy in Robocode because:

- movement and fire choices are naturally mode-based
- the decoder can enforce legal fire and wall-safe motion
- PPO avoids continuous clipping and scale-tuning issues on each branch
- entropy stays interpretable per decision head

## What stays hard-coded

- target selection with stickiness
- wall smoothing
- gun/radar geometry
- legality checks for firing

## What PPO learns

- when to move aggressively or evasively
- when to favor low-density space versus target orbiting
- turn timing
- fire power choice
- radar bias around the current tactical focus
