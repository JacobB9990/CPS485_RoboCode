#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
RUNNER_JAR = ROOT / "scripts" / "run" / "tools" / "robocode-tankroyale-runner-0.38.2.jar"
SERVER_JAR = ROOT / "scripts" / "run" / "tools" / "robocode-tankroyale-server-0.38.2.jar"
JAVA_HELPER = ROOT / "scripts" / "run" / "RunBattleConfigured.java"
JAVA_BUILD_DIR = ROOT / "scripts" / "run" / ".build"
CONFIG_DIR = ROOT / "configs" / "robocode"
RUNTIME_ROOT = ROOT / "logs" / "_runtime_bots"

SAMPLE_BOT_PATHS = {
    "SpinBot": ROOT / "SampleBots" / "SpinBot",
    "TrackFire": ROOT / "SampleBots" / "TrackFire",
    "RamFire": ROOT / "SampleBots" / "RamFire",
    "Corners": ROOT / "SampleBots" / "Corners",
    "Crazy": ROOT / "SampleBots" / "Crazy",
    "VelocityBot": ROOT / "SampleBots" / "VelocityBot",
    "Walls": ROOT / "SampleBots" / "Walls",
    "Target": ROOT / "SampleBots" / "Target",
    "Fire": ROOT / "SampleBots" / "Fire",
}

BOT_SPECS = {
    "Jacob3_0": {
        "name": "Jacob3_0",
        "version": "1.0.0",
        "module": "Jacob3_0.runtime.dqn_bot",
        "weights": ROOT / "Jacob3_0" / "checkpoints" / "dqn_weights.pt",
        "log_flag": "--log-path",
        "state_flag": "--state-log-path",
        "supports_eval": True,
        "env": {"BOT_NAME": "Jacob3_0", "BOT_VERSION": "1.0.0"},
    },
    "MeleeDQN": {
        "name": "MeleeDQN",
        "version": "1.0.0",
        "module": "MeleeDQN.runtime.melee_dqn_bot",
        "weights": ROOT / "MeleeDQN" / "checkpoints" / "melee_dqn_weights.pt",
        "log_flag": "--log-path",
        "state_flag": "--state-log-path",
        "supports_eval": True,
        "extra_args": ["--socket-port", "5999"],
    },
    "PPOBot": {
        "name": "PPOBot",
        "version": "1.0",
        "module": "PPOBot.runtime.PPO_Bot",
        "weights": ROOT / "PPOBot" / "checkpoints" / "ppo_weights.pt",
        "log_flag": "--log-path",
        "supports_eval": True,
    },
    "SarsaBot": {
        "name": "SarsaBot",
        "version": "1.0",
        "module": "SarsaBot.runtime.sarsa_bot",
        "weights": ROOT / "SarsaBot" / "data" / "q_table_sarsa.json",
        "log_flag": "--log-path",
        "supports_eval": True,
    },
    "NeuroEvoMelee": {
        "name": "NeuroEvoMelee",
        "version": "0.1.0",
        "module": "NeuroEvoMelee.runtime.neuroevo_melee_bot",
        "weights": ROOT / "NeuroEvoMelee" / "data" / "current_genome.json",
        "log_flag": "--telemetry",
        "supports_eval": False,
    },
}


def compile_helper() -> None:
    JAVA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    target_class = JAVA_BUILD_DIR / "RunBattleConfigured.class"
    if target_class.exists() and target_class.stat().st_mtime >= JAVA_HELPER.stat().st_mtime:
        return
    subprocess.run(
        ["javac", "-cp", str(RUNNER_JAR), "-d", str(JAVA_BUILD_DIR), str(JAVA_HELPER)],
        cwd=ROOT,
        check=True,
    )


def write_runtime_package(
    bot_key: str,
    mode: str,
    bot_log_path: Path,
    state_log_path: Path | None,
    runtime_dir: Path,
    *,
    package_name: str | None = None,
    version_override: str | None = None,
) -> Path:
    spec = BOT_SPECS[bot_key]
    resolved_name = package_name or spec["name"]
    resolved_version = version_override or spec["version"]
    pkg_dir = runtime_dir / resolved_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / f"{resolved_name}.json").write_text(
        json.dumps(
            {
                "name": resolved_name,
                "version": resolved_version,
                "authors": ["CPS485"],
                "description": f"Generated runtime package for {bot_key}",
                "platform": "Python 3",
                "programmingLang": "Python",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    env_exports = [f'export PYTHONPATH="{ROOT}${{PYTHONPATH:+:$PYTHONPATH}}"']
    env_values = dict(spec.get("env", {}))
    if "BOT_NAME" in env_values:
        env_values["BOT_NAME"] = resolved_name
    if "BOT_VERSION" in env_values:
        env_values["BOT_VERSION"] = resolved_version
    for key, value in env_values.items():
        env_exports.append(f'export {key}="{value}"')

    args = [sys.executable, "-m", spec["module"]]
    if "weights" in spec:
        if bot_key == "NeuroEvoMelee":
            args += ["--genome", str(spec["weights"])]
        else:
            args += ["--weights-path" if bot_key != "SarsaBot" else "--q-table-path", str(spec["weights"])]
    args += [spec["log_flag"], str(bot_log_path)]
    state_flag = spec.get("state_flag")
    if state_flag and state_log_path is not None:
        args += [state_flag, str(state_log_path)]
    args += spec.get("extra_args", [])
    if mode == "eval" and spec.get("supports_eval"):
        args += ["--eval"]
        if bot_key in {"Jacob3_0", "MeleeDQN", "SarsaBot"}:
            args += ["--eval-epsilon", "0.0"]

    launcher = pkg_dir / f"{resolved_name}.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        + "\n".join(env_exports)
        + "\nexec "
        + " ".join(shlex_quote(part) for part in args)
        + "\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return pkg_dir


def parse_opponent_token(token: str) -> tuple[str, str | None]:
    if ":" in token:
        base, alias = token.split(":", 1)
        return base, alias
    return token, None


def resolve_opponent_package(token: str, runtime_dir: Path) -> tuple[str, Path]:
    base_name, alias = parse_opponent_token(token)
    if base_name in SAMPLE_BOT_PATHS:
        return base_name, SAMPLE_BOT_PATHS[base_name]
    if base_name not in BOT_SPECS:
        raise ValueError(f"Unknown opponent: {token}")

    display_name = alias or base_name
    log_path = runtime_dir / f"{display_name}_raw.jsonl"
    pkg = write_runtime_package(
        base_name,
        "eval",
        log_path,
        None,
        runtime_dir,
        package_name=display_name,
    )
    return display_name, pkg


def shlex_quote(value: str) -> str:
    return shlex.quote(value)


def write_battle_config(config_name: str, bot_names: list[str], rounds: int, width: int, height: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{config_name}.json"
    payload = {
        "gameType": "classic",
        "arenaWidth": width,
        "arenaHeight": height,
        "numberOfRounds": rounds,
        "gunCoolingRate": 0.1,
        "maxInactivityTurns": 600 if len(bot_names) > 2 else 450,
        "turnTimeout": 30000,
        "bots": [{"name": name, "version": "generated"} for name in bot_names],
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_rows(output_path: Path, rows: list[dict]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def summarize_records(records: list[dict], bot_key: str, opponents: list[str], rounds: int, width: int, height: int) -> dict:
    if not records:
        return {
            "record_type": "summary",
            "bot": bot_key,
            "opponents": opponents,
            "rounds": rounds,
            "arena_width": width,
            "arena_height": height,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "avg_survival_time": 0.0,
            "avg_damage_dealt": 0.0,
            "placement_distribution": {},
            "avg_kills_per_round": 0.0,
        }

    rewards = [float(row.get("total_reward", 0.0)) for row in records]
    steps = [float(row.get("steps", row.get("turns", row.get("survival_ticks", row.get("turn", 0))))) for row in records]
    wins = [1.0 if row.get("won") else 0.0 for row in records]
    damage = [float(row.get("damage_dealt", 0.0)) for row in records]
    kills = [float(row.get("kills", 0.0)) for row in records]
    placements: dict[str, int] = {}
    for row in records:
        placement = str(row.get("placement", "unknown"))
        placements[placement] = placements.get(placement, 0) + 1

    return {
        "record_type": "summary",
        "bot": bot_key,
        "opponents": opponents,
        "rounds": rounds,
        "arena_width": width,
        "arena_height": height,
        "win_rate": round(mean(wins), 4),
        "avg_reward": round(mean(rewards), 4),
        "avg_survival_time": round(mean(steps), 4),
        "avg_damage_dealt": round(mean(damage), 4),
        "placement_distribution": placements,
        "avg_kills_per_round": round(mean(kills), 4),
    }


def run_scenario(args: argparse.Namespace) -> int:
    compile_helper()
    if not RUNNER_JAR.exists() or not SERVER_JAR.exists():
        raise FileNotFoundError("Missing Tank Royale runner/server jars in scripts/run/tools")

    runtime_dir = RUNTIME_ROOT / args.timestamp / args.script_name / args.scenario_name
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    bot_log_path = runtime_dir / f"{args.bot}_raw.jsonl"
    state_log_path = runtime_dir / f"{args.bot}_states.jsonl" if args.collect_states else None
    bot_pkg = write_runtime_package(args.bot, args.mode, bot_log_path, state_log_path, runtime_dir)
    opponent_tokens = [opponent.strip() for opponent in args.opponents if opponent.strip()]
    opponent_names = []
    opponent_paths = []
    for opponent in opponent_tokens:
        display_name, path = resolve_opponent_package(opponent, runtime_dir)
        opponent_names.append(display_name)
        opponent_paths.append(path)

    battle_config = write_battle_config(
        args.scenario_name,
        [args.bot, *opponent_names],
        args.rounds,
        args.arena_width,
        args.arena_height,
    )
    print(f"[royale_harness] scenario={args.scenario_name} config={battle_config}")

    cmd = [
        "java",
        "-cp",
        f"{JAVA_BUILD_DIR}:{RUNNER_JAR}",
        "RunBattleConfigured",
        str(args.rounds),
        str(args.arena_width),
        str(args.arena_height),
        str(args.port),
        str(bot_pkg),
        *[str(path) for path in opponent_paths],
    ]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Battle process failed for {args.scenario_name}")

    raw_records = load_jsonl_records(bot_log_path)
    normalized_records = []
    for index, record in enumerate(raw_records, start=1):
        row = dict(record)
        row.update(
            {
                "record_type": "episode",
                "script": args.script_name,
                "scenario": args.scenario_name,
                "bot": args.bot,
                "mode": args.mode,
                "opponents": opponent_names,
                "rounds": args.rounds,
                "arena_width": args.arena_width,
                "arena_height": args.arena_height,
            }
        )
        row.setdefault("episode", index)
        normalized_records.append(row)

    summary = summarize_records(normalized_records, args.bot, opponent_names, args.rounds, args.arena_width, args.arena_height)
    summary.update(
        {
            "script": args.script_name,
            "scenario": args.scenario_name,
            "mode": args.mode,
            "battle_config": str(battle_config.relative_to(ROOT)),
        }
    )
    append_rows(Path(args.output_jsonl), normalized_records + [summary])

    if args.collect_states and state_log_path is not None:
        state_rows = load_jsonl_records(state_log_path)
        normalized_states = []
        for row in state_rows:
            enriched = dict(row)
            enriched.update(
                {
                    "record_type": "state",
                    "script": args.script_name,
                    "scenario": args.scenario_name,
                    "opponents": opponent_names,
                    "rounds": args.rounds,
                    "arena_width": args.arena_width,
                    "arena_height": args.arena_height,
                }
            )
            normalized_states.append(enriched)
        append_rows(Path(args.output_jsonl), normalized_states)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Robocode Tank Royale experiment harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-scenario")
    run_parser.add_argument("--script-name", required=True)
    run_parser.add_argument("--scenario-name", required=True)
    run_parser.add_argument("--timestamp", required=True)
    run_parser.add_argument("--output-jsonl", required=True)
    run_parser.add_argument("--bot", required=True, choices=sorted(BOT_SPECS))
    run_parser.add_argument("--mode", default="train", choices=["train", "eval"])
    run_parser.add_argument("--rounds", type=int, required=True)
    run_parser.add_argument("--arena-width", type=int, default=800)
    run_parser.add_argument("--arena-height", type=int, default=600)
    run_parser.add_argument("--port", type=int, default=0)
    run_parser.add_argument("--collect-states", action="store_true")
    run_parser.add_argument("--opponents", nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "run-scenario":
        return run_scenario(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
