# DQN Bot Setup & Execution Guide

## Prerequisites

Ensure you have the required packages installed:
```bash
pip install torch numpy robocode-tank-royale
```

## File Structure

```
Jacob3_0/
├── bots/python/dqn/
│   ├── config/DQNBot.json   # Bot metadata
│   ├── agent/dqn_agent.py   # DQN agent implementation
│   ├── runtime/dqn_bot.py   # RoboCode tank bot using DQN
│   ├── runtime/run_bot.py   # Python launcher
│   ├── checkpoints/
│   └── logs/
├── scripts/run/
└── docs/setup/dqn_setup.md
```

## Running the Bot

### Option 1: Using Python Launcher (Recommended)
```bash
python3 -m bots.python.dqn.runtime.run_bot
```

This automatically sets environment variables needed by RoboCode Tank Royale.

### Option 2: Using Shell Launcher
```bash
bash run.sh
```

### Option 3: Direct with Environment Variables
```bash
export BOT_NAME="DQNBot"
export BOT_VERSION="1.0.0"
python3 dqn_bot.py
```

## Configuration

All hyperparameters can be customized via command-line arguments:

```bash
python3 -m bots.python.dqn.runtime.run_bot \
  --learning-rate 1e-4 \
  --gamma 0.99 \
  --eps-start 0.9 \
  --eps-end 0.05 \
  --eps-decay-steps 2500 \
  --tau 0.005 \
  --batch-size 128 \
  --memory-capacity 10000 \
  --weights-path /path/to/weights.pt \
  --log-path /path/to/log.jsonl

# Evaluation run (frozen policy, no online learning)
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0
```

Use eval mode for fair bot-vs-bot benchmarking. It disables replay storage, gradient updates, and checkpoint saves during matches.

## Output Files

### bots/python/dqn/checkpoints/dqn_weights.pt
PyTorch checkpoint containing:
- Policy network state_dict
- Target network state_dict
- Optimizer state_dict
- Training counters (steps, episodes, wins)

Automatically loaded on startup if it exists.

### bots/python/dqn/logs/dqn_training_log.jsonl
JSON Lines format log, one line per episode:
```json
{"episode": 1, "won": true, "total_reward": 2.15, "steps": 152, "mode": "train", "epsilon": 0.0831, "buffer_size": 1024, "win_rate": 1.0, "training_steps": 8374}
```

## Troubleshooting

### "Missing environment variable: BOT_NAME"
Use the provided launchers (`python3 -m bots.python.dqn.runtime.run_bot` or `scripts/run/run_dqn_bot.sh`) which set environment variables automatically.

### "Failed to read bot info json file: DQNBot.json"
Make sure RoboCode is pointed at `bots/python/dqn/config/DQNBot.json` for this bot.

### Out of Memory
Reduce `--batch-size` or `--memory-capacity`:
```bash
python3 -m bots.python.dqn.runtime.run_bot --batch-size 64 --memory-capacity 5000
```

### GPU Issues
The code automatically falls back to CPU. To force CPU:
```bash
# There's no direct flag; modify dqn_agent.py line ~74:
# self.device = torch.device("cpu")
```

## Training Tips

1. **First runs**: Expect random behavior (ε starts at 0.9)
2. **Monitor win_rate**: Should gradually improve with training
3. **Memory grows**: Buffer fills to capacity over time
4. **Save regularly**: Use `--weights-path` to save in separate files per run
5. **Checkpoints**: Automatically saved after each episode

Example: Collect training runs separately:
```bash
python3 -m bots.python.dqn.runtime.run_bot --log-path bots/python/dqn/logs/run_1.jsonl --weights-path bots/python/dqn/checkpoints/run_1.pt
python3 -m bots.python.dqn.runtime.run_bot --log-path bots/python/dqn/logs/run_2.jsonl --weights-path bots/python/dqn/checkpoints/run_2.pt
```

## Architecture Overview

**Network**: 3-layer feedforward (16 input → 128 → 128 → 7 output)
**Training**: Double DQN with soft target updates
**Exploration**: Epsilon-greedy (0.9 → 0.05)
**State**: 16 continuous features (normalized energy, position, enemy range, etc.)
**Actions**: 7 discrete (strafe, move, fire)

See README.md for full details.
