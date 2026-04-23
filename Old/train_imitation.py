"""Imitation learning training script.

Generates synthetic training data by running the heuristic model on
randomized game states, then trains a neural network to predict the
heuristic's discrete action choices via supervised learning.

This avoids the need to run actual Robocode matches for data collection --
the heuristic model is deterministic so we can query it directly.

Usage:
    python train_imitation.py --samples 50000 --epochs 100 \
                              --model-out models/imitation_model.pt

    # Resume training from existing weights:
    python train_imitation.py --model-in models/imitation_model.pt \
                              --model-out models/imitation_model.pt \
                              --epochs 50
"""

import argparse
import json
import math
import os
import random
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch is required for training. Install with: pip install torch")
    sys.exit(1)

from game_state import GameState, EnemyState
from actions import Action, ActionType
from heuristic_model import HeuristicModel
from imitation_model import action_to_discrete
from rl_model import DQNetwork


def generate_random_state(
    arena_w: float = 800.0,
    arena_h: float = 600.0,
    num_enemies: int = 1,
) -> GameState:
    """Create a plausible random game state for training data generation."""
    state = GameState()
    state.arena_width = arena_w
    state.arena_height = arena_h

    # Randomize own bot
    state.x = random.uniform(30, arena_w - 30)
    state.y = random.uniform(30, arena_h - 30)
    state.energy = random.uniform(5, 100)
    state.direction = random.uniform(0, 360)
    state.gun_direction = random.uniform(0, 360)
    state.radar_direction = random.uniform(0, 360)
    state.velocity = random.uniform(-8, 8)
    state.gun_heat = random.choice([0.0] * 7 + [random.uniform(0.1, 3.0)] * 3)
    state.turn_number = random.randint(1, 1000)
    state.enemy_count = num_enemies

    # Randomize event flags (mostly off, occasionally on)
    state.hit_by_bullet = random.random() < 0.05
    state.hit_wall = random.random() < 0.03
    state.hit_bot = random.random() < 0.02
    state.bullet_hit_enemy = random.random() < 0.05
    if state.hit_by_bullet:
        state.bullet_bearing = random.uniform(-180, 180)

    # Add enemies
    for i in range(num_enemies):
        enemy = EnemyState(
            x=random.uniform(30, arena_w - 30),
            y=random.uniform(30, arena_h - 30),
            energy=random.uniform(5, 100),
            direction=random.uniform(0, 360),
            speed=random.uniform(-8, 8),
            scan_turn=state.turn_number - random.randint(0, 5),
        )
        state.enemies[100 + i] = enemy

    return state


def generate_training_data(
    num_samples: int,
    heuristic: HeuristicModel,
    arena_w: float = 800.0,
    arena_h: float = 600.0,
) -> tuple[list[list[float]], list[int]]:
    """Generate (state, action) pairs by querying the heuristic model.

    Returns:
        features: List of feature vectors (each 25 floats).
        labels: List of discrete action indices.
    """
    features = []
    labels = []

    # Generate diverse situations
    scenarios = [
        {"name": "normal", "weight": 0.50},
        {"name": "near_wall", "weight": 0.15},
        {"name": "low_energy", "weight": 0.10},
        {"name": "close_enemy", "weight": 0.10},
        {"name": "far_enemy", "weight": 0.10},
        {"name": "under_fire", "weight": 0.05},
    ]

    for _ in range(num_samples):
        # Pick a scenario weighted by probability
        r = random.random()
        cumulative = 0.0
        scenario = "normal"
        for s in scenarios:
            cumulative += s["weight"]
            if r <= cumulative:
                scenario = s["name"]
                break

        state = generate_random_state(arena_w, arena_h)

        # Bias the state for the chosen scenario
        if scenario == "near_wall":
            wall = random.choice(["n", "s", "e", "w"])
            if wall == "n":
                state.y = arena_h - random.uniform(10, 60)
            elif wall == "s":
                state.y = random.uniform(10, 60)
            elif wall == "e":
                state.x = arena_w - random.uniform(10, 60)
            elif wall == "w":
                state.x = random.uniform(10, 60)

        elif scenario == "low_energy":
            state.energy = random.uniform(1, 15)

        elif scenario == "close_enemy":
            if state.enemies:
                e = next(iter(state.enemies.values()))
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(30, 120)
                e.x = state.x + dist * math.cos(angle)
                e.y = state.y + dist * math.sin(angle)
                e.x = max(10, min(arena_w - 10, e.x))
                e.y = max(10, min(arena_h - 10, e.y))

        elif scenario == "far_enemy":
            if state.enemies:
                e = next(iter(state.enemies.values()))
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(400, 600)
                e.x = state.x + dist * math.cos(angle)
                e.y = state.y + dist * math.sin(angle)
                e.x = max(10, min(arena_w - 10, e.x))
                e.y = max(10, min(arena_h - 10, e.y))

        elif scenario == "under_fire":
            state.hit_by_bullet = True
            state.bullet_bearing = random.uniform(-180, 180)

        # Query the heuristic
        heuristic.on_round_start(1)
        action = heuristic.decide(state)

        # Convert continuous action to discrete label
        label = action_to_discrete(action, state)

        features.append(state.to_feature_vector())
        labels.append(int(label))

    return features, labels


def save_training_data(
    features: list[list[float]],
    labels: list[int],
    path: str,
) -> None:
    """Save training data to JSONL for inspection or reuse."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for feat, label in zip(features, labels):
            f.write(json.dumps({"state": feat, "action": label}) + "\n")
    print(f"[TrainImitation] Saved {len(features)} samples to {path}")


def load_training_data(path: str) -> tuple[list[list[float]], list[int]]:
    """Load training data from JSONL."""
    features = []
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                features.append(entry["state"])
                labels.append(entry["action"])
    print(f"[TrainImitation] Loaded {len(features)} samples from {path}")
    return features, labels


def train_imitation(
    features: list[list[float]],
    labels: list[int],
    model_save_path: str,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    model_load_path: str | None = None,
    validation_split: float = 0.1,
) -> dict:
    """Train behavioral cloning model via supervised learning.

    Returns:
        dict with training metrics (final loss, accuracy, etc.)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    state_size = GameState.feature_size()
    action_size = ActionType.count()

    # Build network (same architecture as DQN)
    network = DQNetwork(state_size, action_size).to(device)

    if model_load_path and os.path.exists(model_load_path):
        network.load_state_dict(
            torch.load(model_load_path, map_location=device, weights_only=True)
        )
        print(f"Resumed from {model_load_path}")

    # Split into train/validation
    n = len(features)
    n_val = max(1, int(n * validation_split))
    n_train = n - n_val

    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = torch.FloatTensor([features[i] for i in train_idx])
    y_train = torch.LongTensor([labels[i] for i in train_idx])
    X_val = torch.FloatTensor([features[i] for i in val_idx])
    y_val = torch.LongTensor([labels[i] for i in val_idx])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {n_train} samples, validating on {n_val} samples")
    print(f"Action distribution: {_action_distribution(labels)}")

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        # Training
        network.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = network(batch_X)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_X.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        network.eval()
        with torch.no_grad():
            val_logits = network(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_acc = (val_logits.argmax(dim=1) == y_val.to(device)).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in network.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
            )

    # Save best model
    if best_state_dict is not None:
        network.load_state_dict(best_state_dict)
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save(network.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path} (best val acc: {best_val_acc:.3f})")

    return {
        "best_val_accuracy": best_val_acc,
        "final_train_loss": train_loss,
        "final_train_accuracy": train_acc,
        "final_val_loss": val_loss,
        "final_val_accuracy": val_acc,
        "num_train_samples": n_train,
        "num_val_samples": n_val,
    }


def _action_distribution(labels: list[int]) -> dict[str, int]:
    """Count occurrences of each action type."""
    dist = {}
    for label in labels:
        name = ActionType(label).name
        dist[name] = dist.get(name, 0) + 1
    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train imitation model")
    parser.add_argument(
        "--samples", type=int, default=50000,
        help="Number of synthetic training samples to generate",
    )
    parser.add_argument(
        "--data-out",
        default="experiences/imitation_data.jsonl",
        help="Path to save generated training data",
    )
    parser.add_argument(
        "--data-in",
        default=None,
        help="Load pre-generated training data instead of generating new",
    )
    parser.add_argument(
        "--model-out",
        default="models/imitation_model.pt",
        help="Path to save trained model weights",
    )
    parser.add_argument(
        "--model-in",
        default=None,
        help="Path to existing weights to resume training from",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    # Generate or load training data
    if args.data_in and os.path.exists(args.data_in):
        features, labels = load_training_data(args.data_in)
    else:
        print(f"Generating {args.samples} synthetic training samples...")
        heuristic = HeuristicModel()
        features, labels = generate_training_data(args.samples, heuristic)
        save_training_data(features, labels, args.data_out)

    # Train
    metrics = train_imitation(
        features=features,
        labels=labels,
        model_save_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_load_path=args.model_in,
    )

    print("\nTraining complete:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
