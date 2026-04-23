from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BattleBotResult:
    name: str
    version: str
    rank: int
    total_score: int
    survival_score: int
    last_survivor_bonus: int
    bullet_damage_score: int
    bullet_kill_bonus: int
    ram_damage_score: int
    ram_kill_bonus: int
    first_places: int
    second_places: int
    third_places: int


@dataclass(frozen=True)
class BattleSummary:
    rounds: int
    results: list[BattleBotResult]


@dataclass(frozen=True)
class AggregateMetrics:
    win_rate: float
    average_placement: float
    average_damage_score: float
    average_survival_score: float
    average_kill_bonus: float
    average_survival_time: float | None
    average_damage: float | None
    average_kills: float | None


def parse_battle_result(stdout: str) -> BattleSummary:
    line = None
    for raw in stdout.splitlines():
        text = raw.strip()
        if text.startswith("{") and '"results"' in text:
            line = text
    if line is None:
        raise ValueError("Could not find JSON battle result in runner output.")

    data = json.loads(line)
    results = [
        BattleBotResult(
            name=row["name"],
            version=row["version"],
            rank=row["rank"],
            total_score=row["totalScore"],
            survival_score=row["survivalScore"],
            last_survivor_bonus=row["lastSurvivorBonus"],
            bullet_damage_score=row["bulletDamageScore"],
            bullet_kill_bonus=row["bulletKillBonus"],
            ram_damage_score=row["ramDamageScore"],
            ram_kill_bonus=row["ramKillBonus"],
            first_places=row["firstPlaces"],
            second_places=row["secondPlaces"],
            third_places=row["thirdPlaces"],
        )
        for row in data["results"]
    ]
    return BattleSummary(rounds=data["rounds"], results=results)


def aggregate_metrics(
    *,
    our_bot_name: str,
    battle_summaries: list[BattleSummary],
    telemetry_rows: list[dict],
) -> AggregateMetrics:
    our_rows = []
    for summary in battle_summaries:
        for row in summary.results:
            if row.name == our_bot_name:
                our_rows.append(row)
                break

    if not our_rows:
        raise ValueError(f"No results found for bot {our_bot_name!r}")

    wins = sum(1 for row in our_rows if row.rank == 1)
    placement = sum(row.rank for row in our_rows) / len(our_rows)
    damage_score = sum(row.bullet_damage_score + row.ram_damage_score for row in our_rows) / len(
        our_rows
    )
    survival_score = sum(row.survival_score for row in our_rows) / len(our_rows)
    kill_bonus = sum(row.bullet_kill_bonus + row.ram_kill_bonus for row in our_rows) / len(
        our_rows
    )

    if telemetry_rows:
        average_survival_time = sum(row["survival_time"] for row in telemetry_rows) / len(
            telemetry_rows
        )
        average_damage = sum(row["damage"] for row in telemetry_rows) / len(telemetry_rows)
        average_kills = sum(row["kills"] for row in telemetry_rows) / len(telemetry_rows)
    else:
        average_survival_time = None
        average_damage = None
        average_kills = None

    return AggregateMetrics(
        win_rate=wins / len(our_rows),
        average_placement=placement,
        average_damage_score=damage_score,
        average_survival_score=survival_score,
        average_kill_bonus=kill_bonus,
        average_survival_time=average_survival_time,
        average_damage=average_damage,
        average_kills=average_kills,
    )


def load_telemetry_rows(log_glob: str | None) -> list[dict]:
    if not log_glob:
        return []

    rows: list[dict] = []
    for path in sorted(Path().glob(log_glob)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
