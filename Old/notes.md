# AIBot - AI Agent for Robocode Tank Royale

## Architecture Overview

```
AIBot.py  (entry point -- Robocode bot)
    |
    |-- game_state.py       GameState / EnemyState dataclasses
    |-- actions.py          Action dataclass + ActionType enum
    |-- model_interface.py  Abstract ModelInterface base class
    |       |
    |       |-- heuristic_model.py   Rule-based strategy (no deps)
    |       |-- imitation_model.py   Behavioral cloning (PyTorch)
    |       |-- rl_model.py          Policy (PyTorch)
    |
    |-- experience_buffer.py   Replay buffer (JSONL persistence)
    |-- train.py               Offline training script
    |-- train_imitation.py     Imitation learning training
    |-- train_progressive.py   Full pipeline: imitation -> RL -> progressive
    |-- evaluate.py            Model evaluation and comparison
```

### Data flow each tick

1. **AIBot** pulls bot properties (position, energy, heading, etc.) into a `GameState`.
2. Event handlers (`on_scanned_bot`, `on_hit_by_bullet`, ...) update the `GameState` with enemy data and event flags.
3. The configured `ModelInterface.decide(state)` returns an `Action`.
4. **AIBot** executes the `Action` via Robocode API calls (gun turn, fire, body turn, move).
5. If `--collect` is enabled, the RL model logs `(state, action, reward, next_state)` transitions to disk.

## How the AI Learns

The system uses a three-phase learning pipeline:

### Phase 1: Non-AI Baseline (Heuristic Bot)

The `HeuristicModel` is a hand-coded strategy that serves as the expert teacher. It uses:
- Perpendicular strafing to avoid enemy fire
- Predictive lead aiming based on enemy speed and heading
- Distance-scaled fire power (3.0 close, 2.0 medium, 1.0 far)
- Wall avoidance by steering toward arena center
- Reactive dodge when hit by bullets

The heuristic model supports adjustable **difficulty** (0.0 to 1.0):
- **0.0**: Basic movement, no lead aiming, slow reactions, occasional pauses
- **0.5**: Default competitive behavior (original strategy)
- **1.0**: Aggressive dodging, tight aiming, randomized movement patterns

### Phase 2: Imitation Learning (Behavioral Cloning)

The AI learns by imitating the heuristic bot's decisions:

1. **Data generation**: `train_imitation.py` creates randomized game states and queries the heuristic model for its action in each state. This generates (state, action) training pairs without needing to run actual matches.

2. **Action discretization**: The heuristic model outputs continuous actions (body_turn, body_move, gun_turn, fire_power). These are mapped to one of 13 discrete ActionTypes using `action_to_discrete()`.

3. **Supervised training**: A neural network is trained with cross-entropy loss to predict the correct discrete action given a state feature vector.

4. **Scenario diversity**: Training data is generated across 6 biased scenarios (normal play, near-wall, low energy, close enemy, far enemy, under fire) to ensure the model learns robust behavior.

**Why imitation learning first?** Starting RL from scratch requires enormous amounts of random exploration. By first training the network to imitate a competent baseline, the RL model starts with a strong policy and only needs to fine-tune from there.

### Phase 3: Reinforcement Learning (RL) Fine-tuning

After imitation learning provides a warm start, the RL model improves through experience:

1. **Weight transfer**: The imitation model's weights are loaded directly into the NN.

2. **Experience collection**: The bot plays matches with `--model rl --collect`, recording (state, action, reward, next_state, done) transitions.

3. **Reward shaping**: The reward function encourages:
   - Survival (+0.1/tick)
   - Hitting enemies (+3.0)
   - Avoiding damage (-2.0 for bullet hit, -1.0 for wall hit)
   - Winning rounds (+10.0) vs dying (-5.0)

4. **Offline training**: `train.py` uses NN with a target network to stabilize learning. The policy network selects actions while the target network evaluates them.

### Phase 4: Progressive Difficulty

To keep the training environment challenging as the AI improves:

1. The heuristic bot's difficulty parameter is gradually increased from 0.3 to 1.0.
2. At each difficulty level, new imitation training data is generated.
3. The model is re-trained on the combined multi-difficulty dataset.
4. This exposes the AI to increasingly sophisticated opponent behavior.

`train_progressive.py` orchestrates the full pipeline automatically.

## Running the Bot

### Prerequisites

- Python 3.10+
- Robocode Tank Royale server running
- `robocode_tank_royale` package (installed automatically by `AIBot.sh` via shared deps)

### Heuristic model (default, no extra deps)

```bash
./AIBot.sh
./AIBot.sh --difficulty 0.8      # harder heuristic
```

### Imitation model (requires PyTorch)

```bash
pip install torch          # one-time setup

# 1. Train imitation model from heuristic behavior
python train_imitation.py --samples 50000 --epochs 100

# 2. Run the trained imitation model
./AIBot.sh --model imitation --model-path models/imitation_model.pt
```

### RL model (requires PyTorch)

```bash
# 1. Collect experience data (high exploration)
./AIBot.sh --model rl --epsilon 1.0 --collect

# 2. Train offline
python train.py --experiences experiences/replay.jsonl \
                --model-out models/{model}.pt \
                --episodes 2000

# 3. Run trained policy (low exploration)
./AIBot.sh --model rl --model-path models/{model}.pt --epsilon 0.05
```

### Full progressive pipeline

```bash
# Run all phases automatically (imitation -> RL -> progressive difficulty)
python train_progressive.py

# Or with custom parameters
python train_progressive.py --imitation-samples 100000 \
                            --imitation-epochs 200 \
                            --rl-episodes 5000 \
                            --difficulty-steps 5
```

### Evaluate and compare models

```bash
python evaluate.py --models heuristic imitation rl --rounds 500
```

### CLI flags

| Flag             | Default      | Description                              |
|------------------|-------------|------------------------------------------|
| `--model`        | `heuristic` | Model type: `heuristic`, `imitation`, or `rl` |
| `--model-path`   | None        | Path to `.pt` weights for ML models      |
| `--epsilon`      | `0.1`       | Exploration rate (1.0 = fully random)    |
| `--difficulty`   | `0.5`       | Heuristic bot difficulty (0.0 to 1.0)   |
| `--collect`      | off         | Save experience data for training        |

## Module Details

### game_state.py

`GameState` captures every piece of information available to the bot:

- **Own state**: x, y, energy, direction, gun_direction, radar_direction, velocity, gun_heat
- **Arena**: width, height
- **Enemies**: dict of `EnemyState` keyed by scanned_bot_id (position, energy, direction, speed, last-scan turn)
- **Events**: hit_by_bullet, hit_wall, hit_bot, bullet_hit_enemy (boolean flags, reset each tick)

`to_feature_vector()` produces a **25-float normalized vector** for ML models:
- Bot position (2), energy (1), heading sin/cos (2), gun heading sin/cos (2), velocity (1), gun heat (1), wall distances (4), nearest enemy relative pos/energy/heading/speed/distance/bearing (9), event flags (3).

### actions.py

**ActionType** (13 discrete actions for RL):
- 8 movement types (forward/back x straight/left/right + pure turns)
- 3 fire powers (1, 2, 3)
- 1 track-enemy (aim gun without firing)
- 1 dodge (sharp perpendicular escape)

**Action** dataclass (continuous, executed by the bot):
- `body_turn`, `body_move`, `gun_turn`, `fire_power`

`Action.from_action_type()` converts discrete RL actions to continuous actions.
### model_interface.py

Abstract base with one required method:
- `decide(state: GameState) -> Action`

Optional lifecycle hooks:
- `on_round_start(round_number)`
- `on_round_end(round_number, won)`
- `on_reward(reward, state, action, next_state)`

To add a new strategy, subclass `ModelInterface`, implement `decide()`, and add a case to `load_model()` in `AIBot.py`.

### heuristic_model.py

A competitive hand-coded strategy with adjustable difficulty:

1. **Strafe**: Move perpendicular to the nearest enemy to make targeting difficult.
2. **Lead aiming**: Predict where the enemy will be when the bullet arrives (accuracy scales with difficulty).
3. **Distance-scaled fire power**: 3.0 within 150 units, 2.0 within 350, 1.0 beyond.
4. **Wall avoidance**: When within 75 units of a wall, steer toward the arena center.
5. **Hit dodge**: Reverse strafe direction when hit by a bullet (duration scales with difficulty).
6. **Pattern variation**: At high difficulty, introduces randomized movement offsets.
7. **Aim noise**: At low difficulty, adds gaussian noise to aiming.

No external dependencies -- works out of the box.

### imitation_model.py

Behavioral cloning model that uses the same Network architecture as the RL model:
- Picks the action with highest predicted score (argmax over logits)
- Weights can be transferred directly to/from the RL model
- `action_to_discrete()` maps continuous heuristic actions to discrete ActionTypes for training labels

### rl_model.py

**Network**: 3-layer MLP (128 -> 128 -> 64 -> 13 actions).

**RLModel** wraps the network with:
- Epsilon-greedy action selection
- Per-tick reward calculation (survival, energy changes, hits, wall proximity)
- Terminal rewards (+10 for round win, -5 for death)
- Experience collection for offline training

**Reward shaping**:
| Signal              | Reward  |
|---------------------|---------|
| Survival per tick   | +0.1    |
| Energy gain/loss    | x0.5    |
| Bullet hit enemy    | +3.0    |
| Enemy energy drop   | +0.5/pt |
| Hit by bullet       | -2.0    |
| Hit wall            | -1.0    |
| Near wall (<50)     | -0.3    |
| Won round           | +10.0   |
| Died                | -5.0    |

### experience_buffer.py

Fixed-capacity circular buffer with JSONL persistence.  Each entry is:
```json
{"state": [...], "action": 5, "reward": 0.3, "next_state": [...], "done": false}
```

### train.py

Offline trainer:
- Policy network selects actions; target network evaluates them
- Target network syncs every 50 episodes
- Huber loss with gradient clipping
- Supports resuming from existing weights (`--model-in`)

### train_imitation.py

Imitation learning (behavioral cloning) trainer:
- Generates synthetic training data by querying the heuristic model on random game states
- Creates biased scenarios (near-wall, under-fire, etc.) for diverse training
- Trains with cross-entropy loss and validation split
- Saves best model based on validation accuracy
- Can save/load training data as JSONL for reuse

### train_progressive.py

Complete 4-phase training pipeline:
1. **Imitation learning**: Generate data from heuristic bot, train behavioral cloning model
2. **Weight transfer**: Copy imitation model weights to RL model
3. **RL fine-tuning**: Train on collected experience data using imitation weights as warm-start
4. **Progressive difficulty**: Generate new imitation data from heuristic bot at increasing difficulty levels (0.3 -> 1.0), retrain on combined dataset

### evaluate.py

Model comparison across 6 scenario types:
- **Action diversity**: Does the model use a variety of actions?
- **Fires when close**: Does it fire at nearby enemies?
- **Wall avoidance**: Does it steer away from walls?
- **Dodge response**: Does it react to being hit?
- **Heuristic agreement**: How closely does it match the expert baseline?

## Adding a New Model

1. Create `my_model.py` that subclasses `ModelInterface`
2. Implement `decide(self, state: GameState) -> Action`
3. Add a case in `load_model()` inside `AIBot.py`:
   ```python
   elif model_type == "my_model":
       from my_model import MyModel
       return MyModel()
   ```
4. Run: `./AIBot.sh --model my_model`

This works for external LLM-based models too -- wrap the API call in `decide()` and parse the response into an `Action`.

## File Listing

| File                    | Purpose                                       |
|-------------------------|-----------------------------------------------|
| `AIBot.py`              | Main bot: events, state, action execution     |
| `AIBot.json`            | Robocode bot metadata                         |
| `AIBot.sh`              | Unix launch script                            |
| `AIBot.cmd`             | Windows launch script                         |
| `game_state.py`         | GameState / EnemyState dataclasses            |
| `actions.py`            | Action + ActionType definitions               |
| `model_interface.py`    | Abstract model base class                     |
| `heuristic_model.py`    | Rule-based baseline strategy (with difficulty)|
| `imitation_model.py`    | Behavioral cloning model                      |
| `rl_model.py`           | Reinforcement learning model                  |
| `experience_buffer.py`  | Replay buffer with JSONL persistence          |
| `train.py`              | Offline training script                       |
| `train_imitation.py`    | Imitation learning training script            |
| `train_progressive.py`  | Full progressive training pipeline            |
| `evaluate.py`           | Model evaluation and comparison               |
| `experiences/`          | Directory for collected experience/training data |
| `models/`               | Directory for trained model weights           |
