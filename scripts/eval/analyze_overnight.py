#!/usr/bin/env python3
"""Analyze overnight DQN logs and print a compact morning report.

Usage:
  python3 analyze_overnight.py
  python3 analyze_overnight.py --run-dir logs/overnight_20260331_010000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Stats:
    episodes: int = 0
    wins: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    max_training_steps: int = 0

    def add(self, row: dict) -> None:
        self.episodes += 1
        if bool(row.get("won", False)):
            self.wins += 1
        self.total_reward += float(row.get("total_reward", 0.0))
        self.total_steps += int(row.get("steps", 0))
        self.max_training_steps = max(self.max_training_steps, int(row.get("training_steps", 0)))

    @property
    def win_rate(self) -> float:
        return (self.wins / self.episodes) if self.episodes else 0.0

    @property
    def avg_reward(self) -> float:
        return (self.total_reward / self.episodes) if self.episodes else 0.0

    @property
    def avg_steps(self) -> float:
        return (self.total_steps / self.episodes) if self.episodes else 0.0


def find_latest_run(logs_root: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(logs_root, "overnight_*")))
    return candidates[-1] if candidates else None


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_stats(run_dir: str) -> Tuple[Dict[str, Dict[str, Stats]], Stats]:
    grouped: Dict[str, Dict[str, Stats]] = {"train": {}, "eval": {}}
    overall = Stats()

    for mode in ("train", "eval"):
        pattern = os.path.join(run_dir, f"dqn_{mode}_*.jsonl")
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)
            matchup = name.replace(f"dqn_{mode}_", "").replace(".jsonl", "")
            stats = grouped[mode].setdefault(matchup, Stats())
            for row in read_jsonl(path):
                stats.add(row)
                overall.add(row)

    return grouped, overall


def print_table(title: str, entries: Dict[str, Stats]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not entries:
        print("No data")
        return

    header = f"{'matchup':<14} {'episodes':>8} {'wins':>6} {'win_rate':>10} {'avg_reward':>12} {'avg_steps':>10}"
    print(header)
    print("-" * len(header))

    for matchup, s in sorted(entries.items()):
        print(
            f"{matchup:<14} {s.episodes:>8} {s.wins:>6} {s.win_rate*100:>9.1f}% "
            f"{s.avg_reward:>12.3f} {s.avg_steps:>10.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze overnight DQN logs")
    parser.add_argument(
        "--logs-root",
        default=os.path.join(os.path.dirname(__file__), "logs"),
        help="Root logs directory (default: ./logs)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Specific overnight run dir (default: latest overnight_* under logs root)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(args.logs_root)
    if not run_dir:
        print("No overnight run directory found.")
        print("Expected something like: logs/overnight_YYYYMMDD_HHMMSS")
        return 1

    if not os.path.isdir(run_dir):
        print(f"Run directory does not exist: {run_dir}")
        print("Tip: omit --run-dir to use latest, or provide an existing overnight_* directory.")
        return 1

    grouped, overall = load_stats(run_dir)

    print(f"Run directory: {run_dir}")
    print(
        f"Overall: episodes={overall.episodes}, wins={overall.wins}, "
        f"win_rate={overall.win_rate*100:.1f}%, avg_reward={overall.avg_reward:.3f}, "
        f"avg_steps={overall.avg_steps:.1f}, training_steps={overall.max_training_steps}"
    )

    print_table("Training Summary", grouped["train"])
    print_table("Evaluation Summary", grouped["eval"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
