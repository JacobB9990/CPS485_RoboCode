# Melee Opponent Modeling Bot

This folder now includes `MeleeOpponentModelBot.java`, a classic Robocode `AdvancedRobot` built for melee battles instead of 1v1 dueling.

## Class design

- `MeleeOpponentModelBot`
  - Main robot loop
  - Maintains `Map<String, EnemyModel>`
  - Chooses target, movement destination, radar focus, and fire power
- `EnemyModel`
  - Per-enemy online state
  - Tracks:
    - `aggressionLevel`
    - `averageDistance`
    - `firingFrequency`
    - `movementStyle`
    - `targetMeLikelihood`
    - `estimatedAccuracy`
    - `threatScore`
- `AdaptivePolicy`
  - Adjusts global melee posture from the current enemy mix
  - In version 1 it controls:
    - center avoidance
    - disengage bias
    - anti-ram bias

## Threat score formula

Current implementation:

```text
threatScore =
    0.27 * aggressionLevel +
    0.18 * proximity +
    0.18 * firingFrequency +
    0.17 * targetMeLikelihood +
    0.12 * estimatedAccuracy +
    0.08 * energyFactor +
    categoryBonus
```

Where:

- `proximity = clamp(1 - distance / 600)`
- `energyFactor = clamp(enemyEnergy / 100)`
- `categoryBonus`
  - `+0.15` high-accuracy threat
  - `+0.12` close-range aggressor
  - `-0.10` spinner / weak bot
  - `-0.04` passive survivor

## Category rules

- `CLOSE_RANGE_AGGRESSOR`
  - `aggressionLevel > 0.60`
  - `averageDistance < 260`
- `PASSIVE_SURVIVOR`
  - low aggression
  - low firing frequency
  - long average distance
- `SPINNER_WEAK_BOT`
  - spin-heavy heading changes or very stationary behavior
- `HIGH_ACCURACY_THREAT`
  - higher measured hit rate on us plus meaningful firing frequency

## Target selection logic

The bot does not shoot the weakest enemy blindly. It ranks enemies by a combined score:

```text
0.45 * threatScore
+ 0.22 * distanceFactor
+ 0.18 * accuracyWindow
+ 0.10 * lowEnergyFinish
+ 0.05 * freshness
```

This makes it prefer:

- dangerous bots that are also hittable
- nearby threats
- weak spinners that are easy cleanup
- low-energy targets when a kill is realistic

## Movement adaptation

Movement samples several candidate escape angles and scores each destination with a danger function based on:

- enemy threat
- inverse-square distance to each enemy
- local enemy clustering
- wall proximity
- center exposure

When a top threat is too close or stronger than us, the bot adds a disengage option by sampling a direction directly away from that threat.

## ML insertion points

Keep version 1 rule-based. If you want ML later, the cleanest hooks are:

1. Replace `EnemyModel.classify()` with a learned classifier.
2. Replace the hard-coded threat formula with a small regression model.
3. Replace `dangerAt()` with a learned danger estimator over candidate destinations.

Recommended first ML feature vector per enemy:

- distance
- relative bearing
- enemy energy
- velocity
- lateral velocity
- heading change EMA
- firing frequency
- target-me likelihood
- recent bullet-hit counts
- nearby enemy count

That keeps the robot usable immediately while leaving a clear path for a lightweight model later.
