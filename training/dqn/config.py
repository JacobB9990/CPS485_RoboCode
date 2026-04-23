from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BotEntryConfig:
    name: str
    path: str
    tier: str
    weight: float = 1.0


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    rounds: int
    participants: list[str]
    tags: list[str]


@dataclass(frozen=True)
class StageConfig:
    name: str
    battles: int
    rounds: int
    min_opponents: int
    max_opponents: int
    benchmark_weight: float
    checkpoint_weight: float
    baseline_weight: float
    benchmark_tiers: list[str]
    allow_mirror_current_bot: bool = False


@dataclass(frozen=True)
class SelectionConfig:
    keep_top_k: int
    min_checkpoint_gap: int


@dataclass(frozen=True)
class PathsConfig:
    project_root: str
    runner_script: str
    current_bot_dir: str
    checkpoints_dir: str
    logs_dir: str
    checkpoint_index_file: str
    telemetry_glob: str | None = None


@dataclass(frozen=True)
class MeleeTrainingConfig:
    paths: PathsConfig
    benchmark_bots: list[BotEntryConfig]
    baseline_bots: list[BotEntryConfig]
    curriculum: list[StageConfig]
    evaluation_suite: list[ScenarioConfig]
    selection: SelectionConfig

    @staticmethod
    def load(config_path: str | Path) -> "MeleeTrainingConfig":
        path = Path(config_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        config_dir = path.resolve().parent

        def resolve_path(value: str) -> str:
            raw = Path(value)
            if raw.is_absolute():
                return str(raw)
            return str((config_dir / raw).resolve())

        paths_data = dict(data["paths"])
        for key in (
            "project_root",
            "runner_script",
            "current_bot_dir",
            "checkpoints_dir",
            "logs_dir",
            "checkpoint_index_file",
        ):
            paths_data[key] = resolve_path(paths_data[key])
        if paths_data.get("telemetry_glob"):
            paths_data["telemetry_glob"] = resolve_path(paths_data["telemetry_glob"])

        benchmark_rows = []
        for row in data["benchmark_bots"]:
            row = dict(row)
            row["path"] = resolve_path(row["path"])
            benchmark_rows.append(BotEntryConfig(**row))

        baseline_rows = []
        for row in data["baseline_bots"]:
            row = dict(row)
            row["path"] = resolve_path(row["path"])
            baseline_rows.append(BotEntryConfig(**row))

        return MeleeTrainingConfig(
            paths=PathsConfig(**paths_data),
            benchmark_bots=benchmark_rows,
            baseline_bots=baseline_rows,
            curriculum=[StageConfig(**row) for row in data["curriculum"]],
            evaluation_suite=[ScenarioConfig(**row) for row in data["evaluation_suite"]],
            selection=SelectionConfig(**data["selection"]),
        )
