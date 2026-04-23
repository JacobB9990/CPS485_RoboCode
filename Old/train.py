"""Offline training script for the DQN model.

Reads experience data collected during gameplay, trains a DQN using
Double-DQN with target network, and saves the resulting weights.

Usage:
    python train.py --experiences experiences/replay.jsonl \
                    --model-out models/dqn_model.pt \
                    --episodes 2000
"""

import argparse
import os
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch is required for training. Install with: pip install torch")
    sys.exit(1)

from rl_model import DQNetwork
from experience_buffer import ExperienceBuffer
from game_state import GameState
from actions import ActionType


def train(
    experience_path: str,
    model_save_path: str,
    episodes: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    target_update_freq: int = 50,
    model_load_path: str | None = None,
) -> None:
    """Train a DQN from collected experience data.

    Uses Double-DQN: the policy network selects actions while the
    target network evaluates them, reducing overestimation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    state_size = GameState.feature_size()
    action_size = ActionType.count()

    # Networks
    policy_net = DQNetwork(state_size, action_size).to(device)
    target_net = DQNetwork(state_size, action_size).to(device)

    if model_load_path and os.path.exists(model_load_path):
        policy_net.load_state_dict(
            torch.load(model_load_path, map_location=device, weights_only=True)
        )
        print(f"Resumed from {model_load_path}")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Load experience data
    buffer = ExperienceBuffer()
    buffer.load(experience_path)

    if len(buffer) < batch_size:
        print(
            f"Not enough data ({len(buffer)} transitions). "
            f"Need at least {batch_size}. Run more games with --collect."
        )
        return

    print(f"Training on {len(buffer)} transitions for {episodes} episodes...")

    total_loss = 0.0
    for episode in range(1, episodes + 1):
        batch = buffer.sample(batch_size)

        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        next_states = batch["next_states"].to(device)
        dones = batch["dones"].to(device)

        # Q(s, a) from policy network
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: r + gamma * Q_target(s', argmax_a Q_policy(s', a))
        with torch.no_grad():
            next_actions = policy_net(next_states).argmax(dim=1)
            next_q = target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + gamma * next_q * (~dones)

        loss = nn.SmoothL1Loss()(q_values, target_q)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            avg_loss = total_loss / target_update_freq
            print(f"  Episode {episode:>5}/{episodes} | Avg Loss: {avg_loss:.4f}")
            total_loss = 0.0

    # Save
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save(policy_net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN for Robocode")
    parser.add_argument(
        "--experiences",
        default="experiences/replay.jsonl",
        help="Path to JSONL experience file",
    )
    parser.add_argument(
        "--model-out",
        default="models/dqn_model.pt",
        help="Path to save trained model weights",
    )
    parser.add_argument(
        "--model-in",
        default=None,
        help="Path to existing weights to resume training from",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    train(
        experience_path=args.experiences,
        model_save_path=args.model_out,
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        model_load_path=args.model_in,
    )
