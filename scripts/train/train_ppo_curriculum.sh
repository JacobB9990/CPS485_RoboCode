#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="train_ppo_curriculum"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/common.sh"

# ── Argument parsing ──────────────────────────────────────────────────────────
ROUNDS_OVERRIDE=""
PARALLEL=1
SMOKE_TEST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)    ROUNDS_OVERRIDE="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2";        shift 2 ;;
        --smoke-test) SMOKE_TEST=1;         shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 2  ;;
    esac
done

prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "PPOBot"

CHECKPOINT_DIR="$ROOT_DIR/PPOBot/checkpoints"

# ── Helpers ───────────────────────────────────────────────────────────────────

check_win_rate() {
    local scenario_base="$1"
    local jsonl="$2"

    python3 - "$scenario_base" "$jsonl" <<'PYEOF'
import sys

scenario_base, jsonl_path = sys.argv[1], sys.argv[2]

print(f"[failsafe disabled] {scenario_base} — no checks performed")
PYEOF
}

# Run one stage with PARALLEL workers.
# Worker 1 = writer (PPOBot), workers 2..N = readers (PPOBotReader, stale weights).
# Each worker runs rounds/PARALLEL rounds. Results are merged into the main JSONL.
run_stage() {
    local stage_num="$1"
    local scenario_base="$2"
    local rounds="$3"
    shift 3
    local opponents=("$@")

    local rounds_per_worker=$(( (rounds + PARALLEL - 1) / PARALLEL ))
    local base_port=7650
    local pids=()
    local tmp_jsons=()

    echo "=== Stage ${stage_num}: ${scenario_base} (${rounds} rounds, ${PARALLEL}x parallel) ==="

    for i in $(seq 1 "$PARALLEL"); do
        local port=$(( base_port + i ))
        # Unique scenario name per worker avoids runtime-dir collisions in the harness
        local scenario_name="${scenario_base}_w${i}"
        local tmp_json="${JSONL_FILE}.stage${stage_num}_w${i}.tmp"
        tmp_jsons+=("$tmp_json")
        local bot_key="PPOBotReader"
        [[ "$i" -eq 1 ]] && bot_key="PPOBot"

        PPO_OPPONENT="${opponents[0]}" PPO_SCENARIO="$scenario_base" \
        python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
            --script-name "$SCRIPT_NAME" \
            --scenario-name "$scenario_name" \
            --timestamp "$TIMESTAMP" \
            --output-jsonl "$tmp_json" \
            --bot "$bot_key" \
            --mode train \
            --rounds "$rounds_per_worker" \
            --port "$port" \
            --opponents "${opponents[@]}" &
        pids+=($!)
    done

    local exit_code=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "Worker PID $pid failed"; exit_code=1; }
    done
    [[ "$exit_code" -eq 0 ]] || { echo "ERROR: Stage ${stage_num} had failures."; exit 1; }

    # Merge temp JSONLs → main JSONL
    for tmp in "${tmp_jsons[@]}"; do
        [[ -f "$tmp" ]] && { cat "$tmp" >> "$JSONL_FILE"; rm -f "$tmp"; }
    done

    # Stage checkpoint backup
    local src="$CHECKPOINT_DIR/ppo_weights.pt"
    if [[ -f "$src" ]]; then
        local dst="$CHECKPOINT_DIR/ppo_weights_stage${stage_num}_${scenario_base}.pt"
        cp "$src" "$dst"
        echo "[checkpoint] Saved $dst"
    fi

    check_win_rate "$scenario_base" "$JSONL_FILE"
    echo "[curriculum] Stage ${stage_num} (${scenario_base}) done."
    echo ""
}

# ── Smoke test ────────────────────────────────────────────────────────────────

if [[ "$SMOKE_TEST" -eq 1 ]]; then
    echo "=== Smoke test ==="

    # 1. Import check
    cd "$ROOT_DIR"
    python3 -c "from PPOBot.runtime.PPO_Bot import PPOBot; print('Import OK')"

    # 2. Run 4 rounds vs Target (writer only)
    SMOKE_JSON="${JSONL_FILE}.smoke.tmp"
    PPO_OPPONENT="Target" PPO_SCENARIO="smoke_test" \
    python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
        --script-name "smoke_test" \
        --scenario-name "smoke_test_w1" \
        --timestamp "$TIMESTAMP" \
        --output-jsonl "$SMOKE_JSON" \
        --bot PPOBot \
        --mode train \
        --rounds 4 \
        --opponents Target

    # 3. Verify outputs
    python3 - "$SMOKE_JSON" "$CHECKPOINT_DIR/ppo_weights.pt" <<'PYEOF'
import json, sys
from pathlib import Path

smoke_json, weights_path = sys.argv[1], sys.argv[2]
failures = []

# weights file
wp = Path(weights_path)
if not wp.exists() or wp.stat().st_size == 0:
    failures.append(f"weights file missing or empty: {weights_path}")
else:
    print(f"OK: weights {wp.stat().st_size} bytes")

# JSONL rows
if not Path(smoke_json).exists():
    failures.append(f"JSONL not found: {smoke_json}")
else:
    rows = []
    with open(smoke_json) as f:
        for line in f:
            line = line.strip()
            if not line or '"meta"' in line:
                continue
            try:
                r = json.loads(line)
                if r.get("record_type") != "summary":
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        failures.append("no episode rows in JSONL")
    else:
        print(f"OK: {len(rows)} episode rows")
        required = [
            "policy_loss", "value_loss", "entropy",
            "damage_dealt", "damage_taken", "kills",
            "fire_actions", "wall_hits", "total_reward",
            "episode", "won", "placement", "total_bots",
            "steps", "mode", "opponent", "scenario",
        ]
        last = rows[-1]
        missing = [f for f in required if f not in last]
        if missing:
            failures.append(f"missing JSONL fields: {missing}")
        else:
            print(f"OK: all required fields present")
        if last.get("policy_loss") is None:
            failures.append("policy_loss is null — update() may not have run (rollout too small?)")
        else:
            print(f"OK: policy_loss={last['policy_loss']}, entropy={last['entropy']}")

if failures:
    print("\nSMOKE FAIL:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\n=== Smoke test PASSED ===")
PYEOF

    rm -f "$SMOKE_JSON"
    echo "JSONL_PATH=$JSONL_FILE"
    exit 0
fi

# ── Curriculum ────────────────────────────────────────────────────────────────
# Stages ordered easy → hard. Refresher stages (R1, R2) combat catastrophic forgetting.
# With PARALLEL=4, ~2700 effective rounds run in ~2.5–3 hours wall time.
# Use --rounds N to override all stage counts (useful for smoke/quick tests).

stage_r() { echo "${ROUNDS_OVERRIDE:-$1}"; }

run_stage  1  "ppo_vs_target"      "$(stage_r 300)"  Target
run_stage  2  "ppo_vs_fire"        "$(stage_r 200)"  Fire
run_stage  3  "ppo_vs_velocitybot" "$(stage_r 200)"  VelocityBot
run_stage  4  "ppo_vs_walls"       "$(stage_r 200)"  Walls
run_stage  5  "ppo_vs_spinbot"     "$(stage_r 200)"  SpinBot
run_stage  6  "ppo_vs_corners"     "$(stage_r 200)"  Corners
run_stage  7  "ppo_vs_crazy"       "$(stage_r 200)"  Crazy
run_stage  8  "ppo_vs_trackfire"   "$(stage_r 200)"  TrackFire
run_stage  9  "ppo_vs_ramfire"     "$(stage_r 200)"  RamFire
run_stage  R1 "ppo_refresher_1"   "$(stage_r  50)"  Target Fire
run_stage 10  "ppo_3bot_easy"      "$(stage_r 200)"  Target Fire
run_stage 11  "ppo_3bot_medium"    "$(stage_r 200)"  SpinBot Walls
run_stage 12  "ppo_4bot_hard"      "$(stage_r 150)"  Crazy TrackFire RamFire
run_stage  R2 "ppo_refresher_2"   "$(stage_r  50)"  Target Fire
run_stage 13  "ppo_5bot_mix"       "$(stage_r 150)"  SpinBot Crazy TrackFire RamFire
run_stage 14  "ppo_vs_jacob3_0"    "$(stage_r 200)"  Jacob3_0

echo "=== Curriculum complete! ==="
echo "Run: python3 scripts/analyze/summarize_results.py"
echo "JSONL_PATH=$JSONL_FILE"
