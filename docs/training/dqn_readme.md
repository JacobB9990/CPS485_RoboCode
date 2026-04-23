# DQN Bot for RoboCode Tank Royale

PyTorch tutorial-based Deep Q-Network implementation for autonomous tank combat.

## Architecture

### DQN Network
- **Input:** 16 continuous state features
- **Hidden layers:** 128 → 128 units with ReLU
- **Output:** 7 action Q-values
- **Activation:** ReLU for hidden layers

### Training Components
- **Experience Replay:** Cyclic buffer (default 10,000)
- **Target Network:** Soft update with τ=0.005
- **Loss:** SmoothL1Loss (Huber loss)
- **Optimizer:** Adam (default lr=1e-4)
- **Discount:** γ=0.99

### Action Space (7 actions)
```
0: STRAFE_LEFT     → turn_left(30) + forward(80)
1: STRAFE_RIGHT    → turn_right(30) + forward(80)
2: FORWARD         → forward(100)
3: BACKWARD        → back(100)
4: FIRE_LOW        → fire(1.0)
5: FIRE_MEDIUM     → fire(2.0)
6: FIRE_HIGH       → fire(3.0)
```

### State Features (16 total)
```
[0]  self.energy / 100
[1]  self.speed / 8
[2]  sin(self.direction)
[3]  cos(self.direction)
[4-7]  Wall proximity (normalized x, y, inverted x, inverted y)
[8-9]  sin/cos of bearing to enemy
[10] Enemy distance / max_distance
[11] Enemy energy / 100
[12] Δbearing / π (change in enemy bearing)
[13] Δdistance / max_distance (change in enemy distance)
[14] Gun heat / 4 (readiness indicator)
[15] Scan freshness (1 if seen within 10 ticks, else 0)
```

## Training

```bash
# Default settings
python dqn_bot.py

# Custom hyperparameters
python dqn_bot.py \
  --learning-rate 1e-4 \
  --gamma 0.99 \
  --eps-start 0.9 \
  --eps-end 0.05 \
  --eps-decay-steps 2500 \
  --tau 0.005 \
  --batch-size 128 \
  --memory-capacity 10000

# Evaluation (recommended for benchmark matches)
python dqn_bot.py --eval --eval-epsilon 0.0
```

Use training mode to improve the policy, then use evaluation mode to measure performance. In eval mode, the bot disables replay writes, optimizer updates, and checkpoint mutation.

## Outputs

- `dqn_weights.pt` - PyTorch model checkpoint (policy net, target net, optimizer, stats)
- `dqn_training_log.jsonl` - JSON lines log with per-episode metrics

### Log Format
```json
{
  "episode": 1,
  "won": true,
  "total_reward": 2.15,
  "steps": 152,
  "mode": "train",
  "epsilon": 0.0831,
  "buffer_size": 1024,
  "win_rate": 1.0,
  "training_steps": 8374
}
```

## Key Features

✓ Experience replay for stability
✓ Target network with soft updates (τ=0.005)
✓ Epsilon-greedy exploration (decays from 0.9 to 0.05)
✓ Gradient clipping (max_norm=1.0)
✓ Checkpoint save/load
✓ Continuous state representation
✓ Reward shaping for combat feedback

## Reward Structure

- **Win:** +1.0
- **Loss:** -1.0
- **Hit by bullet:** -0.025 × damage
- **Bullet hits enemy:** +0.02 × damage
- **Wall collision:** -0.05
- **Fire action:** -0.01 (anti-spam penalty)

## Train vs Eval Guidance

- Train mode: exploration enabled and online updates enabled.
- Eval mode: fixed epsilon (default 0.0) and no learning updates.
- Avoid comparing opponents while training online, because policy drift can make win rates unstable.

## References

- PyTorch Tutorial: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Paper: Mnih et al., "Human-level control through deep reinforcement learning" (DQN)
