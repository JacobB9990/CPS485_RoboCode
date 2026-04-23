"""Progressive training pipeline: Imitation Learning -> RL Fine-tuning.

Implements the full training workflow:
  1. Generate training data from the heuristic bot at various difficulty levels
  2. Train an imitation model via behavioral cloning (supervised learning)
  3. Transfer imitation weights to the RL model as a warm-start
  4. Fine-tune with reinforcement learning using experience replay
  5. Progressively increase heuristic bot difficulty to keep training challenging

This script runs offline -- it does not need a running Robocode server.
The RL fine-tuning step uses previously collected experience data.

Usage:
    # Full pipeline (fresh start)
    python train_progressive.py

    # Skip imitation, RL only (from existing imitation model)
    python train_progressive.py --skip-imitation \
        --imitation-model models/imitation_model.pt

    # Custom parameters
    python train_progressive.py --imitation-samples 100000 \
        --imitation-epochs 200 --rl-episodes 5000 --difficulty-steps 5
"""

import argparse
import json
import os
import sys

try:
    import torch
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from game_state import GameState
from actions import ActionType
from heuristic_model import HeuristicModel
from rl_model import DQNetwork
from experience_buffer import ExperienceBuffer
from train_imitation import generate_training_data, train_imitation, save_training_data
from train import train as train_rl


def progressive_pipeline(
    # Imitation learning params
    imitation_samples: int = 50000,
    imitation_epochs: int = 100,
    imitation_lr: float = 1e-3,
    skip_imitation: bool = False,
    imitation_model_path: str = "models/imitation_model.pt",
    # RL fine-tuning params
    rl_episodes: int = 2000,
    rl_lr: float = 5e-4,
    rl_gamma: float = 0.99,
    rl_batch_size: int = 64,
    experience_path: str = "experiences/replay.jsonl",
    # Progressive difficulty params
    difficulty_steps: int = 3,
    difficulty_start: float = 0.3,
    difficulty_end: float = 1.0,
    samples_per_difficulty: int = 15000,
    # Output
    output_dir: str = "models",
    data_dir: str = "experiences",
) -> dict:
    """Run the full progressive training pipeline.

    Returns:
        dict with metrics from each training phase.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    results = {"phases": []}

    # ==================================================================
    # Phase 1: Imitation Learning (Behavioral Cloning)
    # ==================================================================
    if not skip_imitation:
        print("=" * 60)
        print("PHASE 1: Imitation Learning (Behavioral Cloning)")
        print("=" * 60)

        # Generate training data from heuristic bot at default difficulty
        heuristic = HeuristicModel(difficulty=0.5)
        print(f"\nGenerating {imitation_samples} training samples from {heuristic.name}...")
        features, labels = generate_training_data(
            imitation_samples, heuristic
        )

        data_path = os.path.join(data_dir, "imitation_data.jsonl")
        save_training_data(features, labels, data_path)

        # Train imitation model
        print(f"\nTraining imitation model for {imitation_epochs} epochs...")
        imitation_metrics = train_imitation(
            features=features,
            labels=labels,
            model_save_path=imitation_model_path,
            epochs=imitation_epochs,
            learning_rate=imitation_lr,
        )

        results["phases"].append({
            "name": "imitation_learning",
            "metrics": imitation_metrics,
        })
        print(f"\nImitation model saved to {imitation_model_path}")
        print(f"  Best validation accuracy: {imitation_metrics['best_val_accuracy']:.3f}")
    else:
        print("Skipping imitation learning (using existing model)")
        if not os.path.exists(imitation_model_path):
            print(f"WARNING: {imitation_model_path} not found. RL will start from scratch.")

    # ==================================================================
    # Phase 2: Transfer to RL Model
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Transfer Imitation Weights to RL Model")
    print("=" * 60)

    rl_model_path = os.path.join(output_dir, "rl_from_imitation.pt")

    if os.path.exists(imitation_model_path):
        # Copy imitation weights as the RL starting point
        state_dict = torch.load(imitation_model_path, weights_only=True)
        torch.save(state_dict, rl_model_path)
        print(f"Transferred imitation weights to {rl_model_path}")
    else:
        rl_model_path = None
        print("No imitation model found. RL will train from scratch.")

    # ==================================================================
    # Phase 3: RL Fine-tuning
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: RL Fine-tuning with Experience Replay")
    print("=" * 60)

    if not os.path.exists(experience_path):
        print(
            f"\nNo experience data found at {experience_path}."
            f"\nTo collect data, run the bot with: ./AIBot.sh --model rl --collect --epsilon 0.3"
            f"\nSkipping RL fine-tuning."
        )
        results["phases"].append({
            "name": "rl_finetuning",
            "skipped": True,
            "reason": "no experience data",
        })
    else:
        # Check if we have enough data
        buf = ExperienceBuffer()
        buf.load(experience_path)

        if len(buf) < rl_batch_size:
            print(
                f"\nNot enough experience data ({len(buf)} transitions)."
                f"\nNeed at least {rl_batch_size}. Skipping RL fine-tuning."
            )
            results["phases"].append({
                "name": "rl_finetuning",
                "skipped": True,
                "reason": f"insufficient data ({len(buf)} transitions)",
            })
        else:
            final_rl_path = os.path.join(output_dir, "dqn_model.pt")
            print(f"\nFine-tuning with {len(buf)} transitions for {rl_episodes} episodes...")

            train_rl(
                experience_path=experience_path,
                model_save_path=final_rl_path,
                episodes=rl_episodes,
                batch_size=rl_batch_size,
                learning_rate=rl_lr,
                gamma=rl_gamma,
                model_load_path=rl_model_path,
            )

            results["phases"].append({
                "name": "rl_finetuning",
                "model_path": final_rl_path,
                "episodes": rl_episodes,
                "experience_size": len(buf),
            })

    # ==================================================================
    # Phase 4: Progressive Difficulty Training
    # ==================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Progressive Difficulty Imitation Data")
    print("=" * 60)

    if difficulty_steps > 0:
        difficulties = []
        step_size = (difficulty_end - difficulty_start) / max(difficulty_steps - 1, 1)
        for i in range(difficulty_steps):
            difficulties.append(difficulty_start + i * step_size)

        all_features = []
        all_labels = []

        for diff in difficulties:
            heuristic = HeuristicModel(difficulty=diff)
            print(f"\nGenerating {samples_per_difficulty} samples at difficulty={diff:.2f}...")
            features, labels = generate_training_data(
                samples_per_difficulty, heuristic
            )
            all_features.extend(features)
            all_labels.extend(labels)

        # Save progressive training data
        progressive_data_path = os.path.join(data_dir, "progressive_data.jsonl")
        save_training_data(all_features, all_labels, progressive_data_path)

        # Train on the combined progressive data
        progressive_model_path = os.path.join(output_dir, "progressive_imitation.pt")
        print(f"\nTraining on {len(all_features)} samples across {len(difficulties)} difficulty levels...")

        # Start from the imitation model as a warm-start
        warm_start = imitation_model_path if os.path.exists(imitation_model_path) else None

        progressive_metrics = train_imitation(
            features=all_features,
            labels=all_labels,
            model_save_path=progressive_model_path,
            epochs=imitation_epochs,
            learning_rate=imitation_lr * 0.5,  # Lower LR for fine-tuning
            model_load_path=warm_start,
        )

        results["phases"].append({
            "name": "progressive_difficulty",
            "difficulties": difficulties,
            "total_samples": len(all_features),
            "metrics": progressive_metrics,
        })

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print("\nGenerated models:")

    model_files = [
        ("Imitation model", imitation_model_path),
        ("RL warm-start", os.path.join(output_dir, "rl_from_imitation.pt")),
        ("RL fine-tuned", os.path.join(output_dir, "dqn_model.pt")),
        ("Progressive model", os.path.join(output_dir, "progressive_imitation.pt")),
    ]
    for label, path in model_files:
        exists = "OK" if os.path.exists(path) else "SKIPPED"
        print(f"  [{exists}] {label}: {path}")

    print("\nTo run the trained bot:")
    print("  ./AIBot.sh --model imitation --model-path models/imitation_model.pt")
    print("  ./AIBot.sh --model rl --model-path models/dqn_model.pt --epsilon 0.05")

    # Save results
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Progressive training pipeline: Imitation -> RL -> Progressive difficulty"
    )

    # Imitation learning
    parser.add_argument("--imitation-samples", type=int, default=50000)
    parser.add_argument("--imitation-epochs", type=int, default=100)
    parser.add_argument("--imitation-lr", type=float, default=1e-3)
    parser.add_argument("--skip-imitation", action="store_true")
    parser.add_argument(
        "--imitation-model", default="models/imitation_model.pt",
        help="Path for imitation model weights",
    )

    # RL fine-tuning
    parser.add_argument("--rl-episodes", type=int, default=2000)
    parser.add_argument("--rl-lr", type=float, default=5e-4)
    parser.add_argument("--rl-gamma", type=float, default=0.99)
    parser.add_argument("--rl-batch-size", type=int, default=64)
    parser.add_argument(
        "--experience-path", default="experiences/replay.jsonl",
        help="Path to collected RL experience data",
    )

    # Progressive difficulty
    parser.add_argument("--difficulty-steps", type=int, default=3)
    parser.add_argument("--difficulty-start", type=float, default=0.3)
    parser.add_argument("--difficulty-end", type=float, default=1.0)
    parser.add_argument("--samples-per-difficulty", type=int, default=15000)

    # Output
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--data-dir", default="experiences")

    args = parser.parse_args()

    progressive_pipeline(
        imitation_samples=args.imitation_samples,
        imitation_epochs=args.imitation_epochs,
        imitation_lr=args.imitation_lr,
        skip_imitation=args.skip_imitation,
        imitation_model_path=args.imitation_model,
        rl_episodes=args.rl_episodes,
        rl_lr=args.rl_lr,
        rl_gamma=args.rl_gamma,
        rl_batch_size=args.rl_batch_size,
        experience_path=args.experience_path,
        difficulty_steps=args.difficulty_steps,
        difficulty_start=args.difficulty_start,
        difficulty_end=args.difficulty_end,
        samples_per_difficulty=args.samples_per_difficulty,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
