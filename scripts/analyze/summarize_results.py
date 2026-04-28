#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs"


def load_records() -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    paths: list[Path] = []
    for path in sorted(LOG_ROOT.rglob("*.jsonl")):
        paths.append(path)
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "meta" in payload:
                    continue
                payload["_source_file"] = str(path.relative_to(ROOT))
                rows.append(payload)
    return rows, paths


def ensure_dirs(timestamp: str) -> tuple[Path, Path]:
    analysis_dir = LOG_ROOT / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir / f"summary_{timestamp}.json", figures_dir


def main() -> int:
    rows, paths = load_records()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path, figures_dir = ensure_dirs(timestamp)

    if not rows:
        summary_path.write_text(json.dumps({"message": "No JSONL records found", "files_scanned": []}, indent=2), encoding="utf-8")
        return 0

    df = pd.DataFrame(rows)
    episodes = df[df["record_type"] == "episode"].copy() if "record_type" in df.columns else pd.DataFrame()
    summaries = df[df["record_type"] == "summary"].copy() if "record_type" in df.columns else pd.DataFrame()
    states = df[df["record_type"] == "state"].copy() if "record_type" in df.columns else pd.DataFrame()

    win_rate_rows = []
    if not summaries.empty:
        exploded = summaries.copy()
        exploded["opponent"] = exploded["opponents"].apply(lambda value: value[0] if isinstance(value, list) and len(value) == 1 else ",".join(value) if isinstance(value, list) else value)
        win_rate_rows = exploded[["bot", "opponent", "win_rate", "avg_reward", "avg_survival_time"]].to_dict(orient="records")

    reward_curves = {}
    if not episodes.empty:
        for bot, group in episodes.groupby("bot"):
            reward_curves[bot] = group.sort_values("episode")[["episode", "total_reward"] + [col for col in ["loss", "policy_loss", "value_loss", "entropy", "epsilon", "avg_abs_td_error"] if col in group.columns]].fillna("").to_dict(orient="records")

    melee_summary = {}
    if not summaries.empty:
        melee = summaries[summaries["opponents"].apply(lambda value: isinstance(value, list) and len(value) > 1)]
        for bot, group in melee.groupby("bot"):
            melee_summary[bot] = group[["scenario", "placement_distribution", "avg_kills_per_round", "avg_damage_dealt"]].to_dict(orient="records")

    map_size = {}
    if not summaries.empty and "arena_width" in summaries.columns:
        maps = summaries[summaries["script"] == "train_jacob3_0_mapsize"]
        if not maps.empty:
            base = maps.set_index("scenario")["avg_reward"].to_dict()
            map_size = {
                "raw": maps[["scenario", "arena_width", "arena_height", "win_rate", "avg_reward"]].to_dict(orient="records"),
                "reward_delta_large_vs_small": float(base.get("jacob3_0_spinbot_large", 0.0) - base.get("jacob3_0_spinbot_small", 0.0)),
                "reward_delta_medium_vs_small": float(base.get("jacob3_0_spinbot_medium", 0.0) - base.get("jacob3_0_spinbot_small", 0.0)),
            }

    neuroevo = {}
    neuro_path = LOG_ROOT / f"evolve_neuroevo_suite_{os.environ.get('PIPELINE_TIMESTAMP', '')}.jsonl"
    if not neuro_path.exists():
        candidates = sorted(LOG_ROOT.glob("evolve_neuroevo_suite_*.jsonl"))
        neuro_path = candidates[-1] if candidates else neuro_path
    if neuro_path.exists():
        neuro_rows = []
        with neuro_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                payload = json.loads(raw_line)
                if "meta" in payload:
                    continue
                if "generation" in payload:
                    neuro_rows.append(payload)
        neuroevo["fitness_curves"] = neuro_rows

    summary = {
        "generated_at": timestamp,
        "files_scanned": [str(path.relative_to(ROOT)) for path in paths],
        "record_count": len(rows),
        "state_record_count": int(len(states)),
        "per_bot_win_rates_by_opponent": win_rate_rows,
        "reward_curves": reward_curves,
        "melee_placement_distributions": melee_summary,
        "map_size_performance_deltas": map_size,
        "neuroevo_fitness_curves": neuroevo,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not summaries.empty:
        chart_df = pd.DataFrame(win_rate_rows)
        if not chart_df.empty:
            pivot = chart_df.pivot_table(index="bot", columns="opponent", values="win_rate", fill_value=0.0)
            pivot.plot(kind="bar", figsize=(12, 6))
            plt.ylabel("Win Rate")
            plt.tight_layout()
            plt.savefig(figures_dir / "win_rate_bar_chart.png")
            plt.close()

    if not episodes.empty:
        plt.figure(figsize=(12, 6))
        for bot, group in episodes.groupby("bot"):
            plt.plot(group["episode"], group["total_reward"], label=bot)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "reward_curves.png")
        plt.close()

    if neuroevo.get("fitness_curves"):
        neuro_df = pd.DataFrame(neuroevo["fitness_curves"])
        plt.figure(figsize=(12, 6))
        plt.plot(neuro_df["generation"], neuro_df["best_fitness"], label="best_fitness")
        plt.plot(neuro_df["generation"], neuro_df["mean_fitness"], label="mean_fitness")
        plt.fill_between(neuro_df["generation"], neuro_df["mean_fitness"] - neuro_df["std_fitness"], neuro_df["mean_fitness"] + neuro_df["std_fitness"], alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "fitness_curves.png")
        plt.close()

    if not summaries.empty:
        placement_rows = []
        for _, row in summaries.iterrows():
            placement_dist = row.get("placement_distribution", {})
            if isinstance(placement_dist, dict):
                for placement, count in placement_dist.items():
                    placement_rows.append({"bot": row["bot"], "placement": placement, "count": count})
        if placement_rows:
            placement_df = pd.DataFrame(placement_rows).pivot_table(index="bot", columns="placement", values="count", fill_value=0.0)
            plt.figure(figsize=(10, 6))
            plt.imshow(placement_df.values, aspect="auto")
            plt.xticks(range(len(placement_df.columns)), placement_df.columns)
            plt.yticks(range(len(placement_df.index)), placement_df.index)
            plt.colorbar(label="Count")
            plt.tight_layout()
            plt.savefig(figures_dir / "placement_heatmap.png")
            plt.close()

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
