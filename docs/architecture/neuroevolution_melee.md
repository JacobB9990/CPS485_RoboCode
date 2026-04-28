# Neuroevolution Melee Bot Architecture

## 1) Goal

Evolve a Java melee policy for Robocode Tank Royale without gradient-based RL.
The evolved policy should prioritize survival and robust placement in multi-enemy battles.

## 2) High-level architecture

- Runtime bot: Java (`bots/java/neuroevo_melee`)
- Trainer: Python GA (`bots/java/neuroevo_melee/training/train_neuroevo_melee.py`)
- Evaluator: existing headless runner (`scripts/run/run_melee_battle.sh`)
- Genome transport: JSON file (`bots/java/neuroevo_melee/data/current_genome.json`)

### Control loop

1. Bot receives world events and updates enemy memory.
2. Bot encodes fixed-size state from multi-enemy context.
3. Bot runs feedforward network inference.
4. Bot applies movement, aiming, and firing actions.
5. Trainer repeatedly swaps genomes and runs melee batches.
6. Trainer computes weighted fitness and evolves next generation.

## 3) File and class layout

### Java bot

- `bots/java/neuroevo_melee/NeuroEvoMeleeBot.json`
- `bots/java/neuroevo_melee/NeuroEvoMeleeBot.sh`
- `bots/java/neuroevo_melee/src/neuroevo/NeuroEvoMeleeBot.java`
- `bots/java/neuroevo_melee/src/neuroevo/FeatureEncoder.java`
- `bots/java/neuroevo_melee/src/neuroevo/EnemyState.java`
- `bots/java/neuroevo_melee/src/neuroevo/GenomeLoader.java`
- `bots/java/neuroevo_melee/src/neuroevo/GenomeNetwork.java`

### External training

- `bots/java/neuroevo_melee/training/train_neuroevo_melee.py`
- `scripts/train/evolve_neuroevo_melee.sh`
- `bots/java/neuroevo_melee/data/current_genome.json`
- `bots/java/neuroevo_melee/data/best_genome.json`
- `bots/java/neuroevo_melee/logs/evolution_log.jsonl`

## 4) State representation (fixed size, multi-enemy)

The encoder aggregates variable enemy count into stable features:

- Self state: energy, velocity, heading, wall proximity
- Nearest enemy block: distance, bearing, energy
- Weakest enemy block: distance, bearing, energy
- Threat block: threat sum and max threat
- Crowd block: local crowd density and alive count
- Weapon/aim block: gun heat and target alignment
- Preference block: nearest-vs-weakest pressure differential

This satisfies melee support while keeping neural I/O dimension constant.

## 5) Action outputs

Network outputs:

1. movement turn
2. movement distance/sign
3. fire intensity (continuous power thresholded to no-fire/fire)
4. target preference adjustment

Target preference output continuously nudges target selection between nearest and weakest opponents.

## 6) Fitness function

Per-evaluation fitness uses a weighted sum:

- survival score: 0.20
- placement score: 0.20
- damage dealt score: 0.20
- kill bonus score: 0.12
- win rate: 0.14
- reduced damage taken score: 0.14

Where possible, runner metrics are normalized against peers in each battle, and optional bot telemetry contributes damage-taken shaping.

## 7) Genome format and evolution

Genome is a fixed-topology MLP:

- input size: 20
- hidden size: 24 (default)
- output size: 4
- parameters: `w1`, `b1`, `w2`, `b2`

JSON format fields:

- `inputSize`
- `hiddenSize`
- `outputSize`
- `w1`
- `b1`
- `w2`
- `b2`

### Mutation

- Gaussian perturbation with configurable rate/std
- Rare random reset mutation for exploration
- Hard clipping of parameter values to bounded range

### Crossover

- Uniform per-parameter inheritance from parent A or B

### Selection

- Tournament parent selection
- Elitism keeps top genomes unchanged each generation

## 8) Genome evaluation loop

For each genome:

1. Write genome to `current_genome.json`.
2. Sample multiple random melee opponent sets.
3. Run `scripts/run/run_melee_battle.sh` with bot + sampled opponents.
4. Parse runner JSON output.
5. Optionally consume recent telemetry rows.
6. Compute fitness and average across sampled scenarios.

Using multiple seeds and opponent mixes reduces overfitting to a single lobby composition.

## 9) Compute-feasibility recommendations

1. Start with small populations (16 to 32) and short rounds (20 to 40).
2. Use low evaluation sample count early (2 to 3), increase only near convergence.
3. Cache and reuse a fixed opponent pool with stratified difficulty.
4. Keep elitism modest (10 to 20 percent) to avoid premature convergence.
5. Parallelize genome evaluation in future by sharding battles across ports/machines.
6. Periodically run a fixed benchmark suite to track true progress, not just train fitness.

## 10) Suggested next upgrades

1. Add novelty score or behavior diversity objective.
2. Add structural mutations (NEAT-style) once baseline GA is stable.
3. Add richer telemetry (per-round survival turns, collision rate, wall-hit rate).
4. Introduce two-head outputs for movement and firing specialization.
