# Hybrid Melee Robocode Bot

This folder contains a classic Java Robocode bot scaffold for melee battles built as a hybrid hierarchy instead of a single controller.

## Architecture

Level 1 tactical manager:
- `RuleBasedTacticalManager`
- Chooses one mode each tick:
  - `SURVIVE`
  - `ENGAGE`
  - `REPOSITION`
  - `FINISH_WEAK_TARGET`
  - `ESCAPE_CROWD`

Level 2 specialized controllers:
- `ModeAwareMovementController`
- `WeightedTargetSelector`
- `GuessFactorGunController` skeleton with simple linear lead bootstrap
- `SweepRadarController`

Shared services:
- `EnemyTracker` tracks all opponents
- `DangerMapBuilder` estimates battlefield risk
- `BotContext` bundles current state for all controllers

## Suggested class plan

- `src/com/cps485/robocode/hybrid/HybridMeleeBot.java`
  Main `AdvancedRobot` orchestration loop and event wiring.
- `src/com/cps485/robocode/hybrid/TacticalMode.java`
  Tactical mode enum.
- `src/com/cps485/robocode/hybrid/BotContext.java`
  Immutable snapshot passed into decision layers.
- `src/com/cps485/robocode/hybrid/EnemySnapshot.java`
  Tracked enemy state.
- `src/com/cps485/robocode/hybrid/EnemyTracker.java`
  Multi-enemy memory and freshness handling.
- `src/com/cps485/robocode/hybrid/DangerMap.java`
  Coarse grid storing danger by cell.
- `src/com/cps485/robocode/hybrid/DangerMapBuilder.java`
  Converts enemy positions, walls, and crowding into danger estimates.
- `src/com/cps485/robocode/hybrid/RuleBasedTacticalManager.java`
  Tactical mode switching logic.
- `src/com/cps485/robocode/hybrid/WeightedTargetSelector.java`
  Enemy scoring for melee targeting.
- `src/com/cps485/robocode/hybrid/ModeAwareMovementController.java`
  Movement strategy per tactical mode.
- `src/com/cps485/robocode/hybrid/SweepRadarController.java`
  Keeps lock quality on the chosen target while refreshing the field.
- `src/com/cps485/robocode/hybrid/GuessFactorGunController.java`
  Bootstrapped gun controller with fire power policy.
- `src/com/cps485/robocode/hybrid/Geometry.java`
  Shared math helpers.

## What should be hard-coded first

Start with these deterministic parts first:

1. `EnemyTracker`
   Reliable world state is required before learning or advanced heuristics help.
2. `WeightedTargetSelector`
   Hand-coded target scoring is stable, interpretable, and easy to tune for melee.
3. `SweepRadarController`
   Good data quality is worth more than fancy policy early on.
4. `DangerMapBuilder`
   Start coarse and readable before trying finer learned movement.
5. `RuleBasedTacticalManager`
   Gives you debuggable switching and good fallback behavior.
6. `ModeAwareMovementController`
   Begin with safe orbiting, anti-cornering, and crowd escape rules.
7. `GuessFactorGunController`
   Start with linear targeting plus smart fire power, then upgrade to wave surfing or guess factors.

## Optional ML placement

Two good upgrade paths:

- Learned movement only:
  - Replace or augment `ModeAwareMovementController`
  - Keep `WeightedTargetSelector` and `SweepRadarController` hand-coded
  - Use the danger map, wall proximity, and enemy bearings as PPO or DQN inputs
- Learned tactical manager:
  - Replace `RuleBasedTacticalManager`
  - Keep movement, radar, gun, and target selection deterministic at first
  - Use mode rewards based on survival time, damage dealt, and avoiding crowded losses

## Staged implementation path

Stage 1:
- Make the bot compile and move
- Track all enemies
- Select one target
- Spin radar continuously
- Use simple linear targeting and conservative fire power

Stage 2:
- Add danger map and per-mode movement
- Introduce crowd escape and weak-target finishing
- Add stale-enemy penalties to target scoring

Stage 3:
- Upgrade gun from linear lead to wave or guess-factor aiming
- Add visit-count or segmentation data to improve hit chance estimates
- Tune movement randomness to avoid predictability

Stage 4:
- Add learning to movement only or to the tactical manager only
- Train in melee scenarios with 4 to 10 opponents
- Keep deterministic fallbacks for unseen states
