#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.dqn.config import MeleeTrainingConfig
from training.dqn.orchestrator import MeleeOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Melee training orchestrator for Robocode.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "dqn" / "melee_config.example.json"),
        help="Path to training config JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["plan-train", "run-train", "plan-eval", "run-eval"],
        default="plan-train",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = MeleeTrainingConfig.load(args.config)
    orchestrator = MeleeOrchestrator(config, seed=args.seed)

    if args.mode == "plan-train":
        rows = [battle_to_dict(row) for row in orchestrator.build_training_plan()]
        print(json.dumps(rows, indent=2))
        return

    if args.mode == "plan-eval":
        rows = [battle_to_dict(row) for row in orchestrator.build_evaluation_plan()]
        print(json.dumps(rows, indent=2))
        return

    if args.mode == "run-train":
        rows = orchestrator.run(orchestrator.build_training_plan(), dry_run=args.dry_run)
        print(json.dumps(rows, indent=2))
        return

    result = orchestrator.evaluate_checkpoint_set(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


def battle_to_dict(battle) -> dict:
    return {
        "battle_id": battle.battle_id,
        "mode": battle.mode,
        "label": battle.label,
        "rounds": battle.rounds,
        "participants": [str(bot.path) for bot in battle.participants],
    }


if __name__ == "__main__":
    main()
