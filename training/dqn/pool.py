from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import BotEntryConfig, StageConfig


@dataclass(frozen=True)
class PoolBot:
    name: str
    path: Path
    source: str
    tier: str
    checkpoint_step: int | None = None


def load_checkpoint_pool(index_path: Path) -> list[PoolBot]:
    if not index_path.exists():
        return []

    bots: list[PoolBot] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        bots.append(
            PoolBot(
                name=row["name"],
                path=Path(row["bot_dir"]),
                source="checkpoint",
                tier=row.get("tier", "checkpoint"),
                checkpoint_step=row.get("step"),
            )
        )
    return bots


def materialize_static_pool(rows: Iterable[BotEntryConfig], source: str) -> list[PoolBot]:
    return [
        PoolBot(name=row.name, path=Path(row.path), source=source, tier=row.tier)
        for row in rows
    ]


def sample_opponents(
    *,
    rng: random.Random,
    stage: StageConfig,
    benchmark_bots: list[PoolBot],
    checkpoint_bots: list[PoolBot],
    baseline_bots: list[PoolBot],
) -> list[PoolBot]:
    allowed_benchmarks = [bot for bot in benchmark_bots if bot.tier in stage.benchmark_tiers]
    available_unique = len({bot.path for bot in [*allowed_benchmarks, *checkpoint_bots, *baseline_bots]})
    if available_unique == 0:
        raise ValueError("No opponents available for melee sampling.")

    count = min(rng.randint(stage.min_opponents, stage.max_opponents), available_unique)
    sampled: list[PoolBot] = []
    used_paths: set[Path] = set()

    weighted_groups = []
    if allowed_benchmarks:
        weighted_groups.append(("benchmark", stage.benchmark_weight))
    if checkpoint_bots:
        weighted_groups.append(("checkpoint", stage.checkpoint_weight))
    if baseline_bots:
        weighted_groups.append(("baseline", stage.baseline_weight))

    while len(sampled) < count:
        group = _weighted_choice(rng, weighted_groups)
        if group == "benchmark":
            bot = rng.choice(allowed_benchmarks)
        elif group == "checkpoint":
            bot = rng.choice(checkpoint_bots)
        else:
            bot = rng.choice(baseline_bots)

        if bot.path in used_paths:
            continue
        used_paths.add(bot.path)
        sampled.append(bot)

    return sampled


def _weighted_choice(rng: random.Random, rows: list[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in rows)
    pick = rng.random() * total
    cursor = 0.0
    for value, weight in rows:
        cursor += weight
        if pick <= cursor:
            return value
    return rows[-1][0]
