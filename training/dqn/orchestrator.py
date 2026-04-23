from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import MeleeTrainingConfig, ScenarioConfig, StageConfig
from .evaluation import AggregateMetrics, aggregate_metrics, load_telemetry_rows, parse_battle_result
from .pool import PoolBot, load_checkpoint_pool, materialize_static_pool, sample_opponents


@dataclass(frozen=True)
class ScheduledBattle:
    battle_id: str
    mode: str
    rounds: int
    label: str
    participants: list[PoolBot]


class MeleeOrchestrator:
    def __init__(self, config: MeleeTrainingConfig, seed: int = 7):
        self.config = config
        self.rng = random.Random(seed)
        self.project_root = Path(config.paths.project_root)
        self.logs_dir = Path(config.paths.logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def build_training_plan(self) -> list[ScheduledBattle]:
        benchmark_bots = materialize_static_pool(self.config.benchmark_bots, "benchmark")
        baseline_bots = materialize_static_pool(self.config.baseline_bots, "baseline")
        checkpoint_bots = self._filtered_checkpoint_pool()

        current = PoolBot(
            name="current_bot",
            path=Path(self.config.paths.current_bot_dir),
            source="current",
            tier="current",
        )

        battles: list[ScheduledBattle] = []
        for stage in self.config.curriculum:
            for battle_index in range(stage.battles):
                opponents = sample_opponents(
                    rng=self.rng,
                    stage=stage,
                    benchmark_bots=benchmark_bots,
                    checkpoint_bots=checkpoint_bots,
                    baseline_bots=baseline_bots,
                )
                participants = [current, *opponents]
                battle_id = f"{stage.name}_{battle_index + 1:03d}"
                battles.append(
                    ScheduledBattle(
                        battle_id=battle_id,
                        mode="train",
                        rounds=stage.rounds,
                        label=stage.name,
                        participants=participants,
                    )
                )
        return battles

    def build_evaluation_plan(self) -> list[ScheduledBattle]:
        current = PoolBot(
            name="current_bot",
            path=Path(self.config.paths.current_bot_dir),
            source="current",
            tier="current",
        )
        name_to_bot = {
            bot.name: bot for bot in materialize_static_pool(self.config.benchmark_bots, "benchmark")
        }
        name_to_bot.update(
            {bot.name: bot for bot in materialize_static_pool(self.config.baseline_bots, "baseline")}
        )
        for bot in self._filtered_checkpoint_pool():
            name_to_bot[bot.name] = bot

        plan: list[ScheduledBattle] = []
        for scenario in self.config.evaluation_suite:
            participants = [current]
            participants.extend(name_to_bot[name] for name in scenario.participants)
            plan.append(
                ScheduledBattle(
                    battle_id=f"eval_{scenario.name}",
                    mode="eval",
                    rounds=scenario.rounds,
                    label=scenario.name,
                    participants=participants,
                )
            )
        return plan

    def run(self, plan: list[ScheduledBattle], dry_run: bool = False) -> list[dict]:
        history: list[dict] = []
        for battle in plan:
            row = self._execute_battle(battle, dry_run=dry_run)
            history.append(row)
            self._append_jsonl(self.logs_dir / f"{battle.mode}_history.jsonl", row)
        return history

    def evaluate_checkpoint_set(self, dry_run: bool = False) -> dict:
        plan = self.build_evaluation_plan()
        run_rows = self.run(plan, dry_run=dry_run)
        if dry_run:
            return {"mode": "dry_run", "battles": len(run_rows)}

        summaries = [parse_battle_result(row["runner_stdout"]) for row in run_rows]
        telemetry = load_telemetry_rows(self.config.paths.telemetry_glob)
        metrics = aggregate_metrics(
            our_bot_name=self._resolve_our_bot_name(summaries),
            battle_summaries=summaries,
            telemetry_rows=telemetry,
        )
        selection_row = self._selection_row(metrics)
        self._append_jsonl(self.logs_dir / "evaluation_summary.jsonl", selection_row)
        leaderboard = self._build_leaderboard()
        (self.logs_dir / "checkpoint_leaderboard.json").write_text(
            json.dumps(leaderboard, indent=2),
            encoding="utf-8",
        )
        return selection_row

    def _execute_battle(self, battle: ScheduledBattle, dry_run: bool) -> dict:
        cmd = [
            self.config.paths.runner_script,
            str(battle.rounds),
            "0",
            *[str(bot.path) for bot in battle.participants],
        ]
        if dry_run:
            stdout = ""
            return {
                "battle_id": battle.battle_id,
                "mode": battle.mode,
                "label": battle.label,
                "rounds": battle.rounds,
                "participants": [self._serialize_bot(bot) for bot in battle.participants],
                "command": cmd,
                "runner_stdout": stdout,
            }

        completed = subprocess.run(
            cmd,
            cwd=self.project_root,
            text=True,
            capture_output=True,
            check=True,
        )
        return {
            "battle_id": battle.battle_id,
            "mode": battle.mode,
            "label": battle.label,
            "rounds": battle.rounds,
            "participants": [self._serialize_bot(bot) for bot in battle.participants],
            "command": cmd,
            "runner_stdout": completed.stdout,
            "runner_stderr": completed.stderr,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

    def _filtered_checkpoint_pool(self) -> list[PoolBot]:
        checkpoints = load_checkpoint_pool(Path(self.config.paths.checkpoint_index_file))
        min_gap = self.config.selection.min_checkpoint_gap
        if min_gap <= 0:
            return checkpoints

        filtered: list[PoolBot] = []
        last_step = None
        for bot in sorted(checkpoints, key=lambda row: row.checkpoint_step or -1):
            if last_step is None or bot.checkpoint_step is None or bot.checkpoint_step - last_step >= min_gap:
                filtered.append(bot)
                last_step = bot.checkpoint_step
        return filtered

    def _resolve_our_bot_name(self, summaries) -> str:
        first = summaries[0]
        known_names = {bot.name for bot in materialize_static_pool(self.config.benchmark_bots, "benchmark")}
        known_names.update(bot.name for bot in materialize_static_pool(self.config.baseline_bots, "baseline"))
        known_names.update(bot.name for bot in self._filtered_checkpoint_pool())
        for row in first.results:
            if row.name not in known_names:
                return row.name
        return first.results[0].name

    def _selection_row(self, metrics: AggregateMetrics) -> dict:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate": Path(self.config.paths.current_bot_dir).name,
            "selection_metric": "average_placement",
            "average_placement": round(metrics.average_placement, 4),
            "win_rate": round(metrics.win_rate, 4),
            "average_damage_score": round(metrics.average_damage_score, 4),
            "average_survival_score": round(metrics.average_survival_score, 4),
            "average_kill_bonus": round(metrics.average_kill_bonus, 4),
            "average_survival_time": metrics.average_survival_time,
            "average_damage": metrics.average_damage,
            "average_kills": metrics.average_kills,
        }

    def _append_jsonl(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _build_leaderboard(self) -> list[dict]:
        summary_path = self.logs_dir / "evaluation_summary.jsonl"
        if not summary_path.exists():
            return []

        rows = []
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

        rows.sort(key=lambda row: (row["average_placement"], -row["win_rate"], -row["average_damage_score"]))
        top_k = self.config.selection.keep_top_k
        leaderboard = []
        for index, row in enumerate(rows, start=1):
            enriched = dict(row)
            enriched["leaderboard_rank"] = index
            enriched["keep_checkpoint"] = index <= top_k
            leaderboard.append(enriched)
        return leaderboard

    def _serialize_bot(self, bot: PoolBot) -> dict:
        row = asdict(bot)
        row["path"] = str(bot.path)
        return row
