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