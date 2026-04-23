import argparse
import math
import os
import random
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from game_state import GameState, EnemyState
from actions import Action, ActionType
from heuristic_model import HeuristicModel
from model_interface import ModelInterface
from imitation_model import action_to_discrete


def generate_scenario(
    scenario_type: str,
    arena_w: float = 800.0,
    arena_h: float = 600.0,
) -> GameState:
    """Generate a specific game scenario for evaluation."""
    state = GameState()
    state.arena_width = arena_w
    state.arena_height = arena_h
    state.x = random.uniform(50, arena_w - 50)
    state.y = random.uniform(50, arena_h - 50)
    state.energy = random.uniform(20, 100)
    state.direction = random.uniform(0, 360)
    state.gun_direction = random.uniform(0, 360)
    state.radar_direction = random.uniform(0, 360)
    state.velocity = random.uniform(-8, 8)
    state.gun_heat = 0.0
    state.turn_number = random.randint(10, 500)
    state.enemy_count = 1

    if scenario_type == "close_enemy":
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(40, 120)
        enemy = EnemyState(
            x=state.x + dist * math.cos(angle),
            y=state.y + dist * math.sin(angle),
            energy=random.uniform(10, 80),
            direction=random.uniform(0, 360),
            speed=random.uniform(-4, 4),
            scan_turn=state.turn_number,
        )
        enemy.x = max(10, min(arena_w - 10, enemy.x))
        enemy.y = max(10, min(arena_h - 10, enemy.y))
        state.enemies[100] = enemy

    elif scenario_type == "far_enemy":
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(400, 600)
        enemy = EnemyState(
            x=state.x + dist * math.cos(angle),
            y=state.y + dist * math.sin(angle),
            energy=random.uniform(10, 80),
            direction=random.uniform(0, 360),
            speed=random.uniform(-8, 8),
            scan_turn=state.turn_number,
        )
        enemy.x = max(10, min(arena_w - 10, enemy.x))
        enemy.y = max(10, min(arena_h - 10, enemy.y))
        state.enemies[100] = enemy

    elif scenario_type == "near_wall":
        wall = random.choice(["n", "s", "e", "w"])
        if wall == "n":
            state.y = arena_h - random.uniform(10, 40)
        elif wall == "s":
            state.y = random.uniform(10, 40)
        elif wall == "e":
            state.x = arena_w - random.uniform(10, 40)
        elif wall == "w":
            state.x = random.uniform(10, 40)
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(100, 300)
        enemy = EnemyState(
            x=state.x + dist * math.cos(angle),
            y=state.y + dist * math.sin(angle),
            energy=50,
            direction=random.uniform(0, 360),
            speed=random.uniform(-4, 4),
            scan_turn=state.turn_number,
        )
        enemy.x = max(10, min(arena_w - 10, enemy.x))
        enemy.y = max(10, min(arena_h - 10, enemy.y))
        state.enemies[100] = enemy

    elif scenario_type == "under_fire":
        state.hit_by_bullet = True
        state.bullet_bearing = random.uniform(-180, 180)
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(100, 300)
        enemy = EnemyState(
            x=state.x + dist * math.cos(angle),
            y=state.y + dist * math.sin(angle),
            energy=50,
            direction=random.uniform(0, 360),
            speed=random.uniform(-4, 4),
            scan_turn=state.turn_number,
        )
        enemy.x = max(10, min(arena_w - 10, enemy.x))
        enemy.y = max(10, min(arena_h - 10, enemy.y))
        state.enemies[100] = enemy

    elif scenario_type == "no_enemy":
        pass  # No enemies in sight

    elif scenario_type == "low_energy":
        state.energy = random.uniform(1, 10)
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(100, 300)
        enemy = EnemyState(
            x=state.x + dist * math.cos(angle),
            y=state.y + dist * math.sin(angle),
            energy=50,
            direction=random.uniform(0, 360),
            speed=0,
            scan_turn=state.turn_number,
        )
        enemy.x = max(10, min(arena_w - 10, enemy.x))
        enemy.y = max(10, min(arena_h - 10, enemy.y))
        state.enemies[100] = enemy

    return state


def evaluate_model(
    model: ModelInterface,
    rounds: int = 200,
    reference_model: ModelInterface | None = None,
) -> dict:
    """Evaluate a model across diverse scenarios.

    Returns a dict of metric scores.
    """
    scenarios = [
        "close_enemy",
        "far_enemy",
        "near_wall",
        "under_fire",
        "no_enemy",
        "low_energy",
    ]

    action_counts = {a.name: 0 for a in ActionType}
    total_actions = 0
    fires_when_close = 0
    close_enemy_rounds = 0
    wall_avoidance_correct = 0
    near_wall_rounds = 0
    dodge_on_hit = 0
    hit_rounds = 0
    agreement_with_reference = 0
    total_compared = 0

    model.on_round_start(1)
    if reference_model:
        reference_model.on_round_start(1)

    for _ in range(rounds):
        scenario = random.choice(scenarios)
        state = generate_scenario(scenario)
        action = model.decide(state)

        # Classify the continuous action as discrete for analysis
        discrete = action_to_discrete(action, state)
        action_counts[discrete.name] += 1
        total_actions += 1

        # Metric: fires when enemy is close
        if scenario == "close_enemy":
            close_enemy_rounds += 1
            if action.fire_power > 0:
                fires_when_close += 1

        # Metric: turns away from wall when near it
        if scenario == "near_wall":
            near_wall_rounds += 1
            walls = state.distance_to_walls
            min_wall = min(walls, key=walls.get)
            # Check if the body turn moves us toward center
            center_x = state.arena_width / 2
            center_y = state.arena_height / 2
            bearing_to_center = math.degrees(
                math.atan2(center_x - state.x, center_y - state.y)
            )
            angle_diff = abs(bearing_to_center - state.direction - action.body_turn)
            while angle_diff > 180:
                angle_diff -= 360
            if abs(angle_diff) < 90 or abs(action.body_move) > 30:
                wall_avoidance_correct += 1

        # Metric: changes movement when hit
        if scenario == "under_fire":
            hit_rounds += 1
            if abs(action.body_move) > 30 or abs(action.body_turn) > 20:
                dodge_on_hit += 1

        # Metric: agreement with reference model
        if reference_model:
            ref_action = reference_model.decide(state)
            ref_discrete = action_to_discrete(ref_action, state)
            total_compared += 1
            if discrete == ref_discrete:
                agreement_with_reference += 1

    # Calculate metrics
    action_types_used = sum(1 for c in action_counts.values() if c > 0)

    results = {
        "model_name": model.name,
        "total_rounds": rounds,
        "action_diversity": action_types_used / ActionType.count(),
        "action_distribution": {
            k: v / total_actions for k, v in action_counts.items() if v > 0
        },
        "fires_when_close": fires_when_close / max(close_enemy_rounds, 1),
        "wall_avoidance_rate": wall_avoidance_correct / max(near_wall_rounds, 1),
        "dodge_on_hit_rate": dodge_on_hit / max(hit_rounds, 1),
    }

    if reference_model:
        results["agreement_with_heuristic"] = (
            agreement_with_reference / max(total_compared, 1)
        )

    return results


def load_model_for_eval(
    model_type: str,
    model_path: str | None = None,
    difficulty: float = 0.5,
) -> ModelInterface:
    """Load a model for evaluation."""
    if model_type == "heuristic":
        return HeuristicModel(difficulty=difficulty)
    elif model_type == "imitation":
        from imitation_model import ImitationModel
        return ImitationModel(model_path=model_path)
    elif model_type == "rl":
        from rl_model import RLModel
        return RLModel(model_path=model_path, epsilon=0.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_evaluation_report(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    # Header
    names = [r["model_name"] for r in results]
    col_width = max(25, max(len(n) for n in names) + 2)
    header = f"{'Metric':<30}" + "".join(f"{n:>{col_width}}" for n in names)
    print(header)
    print("-" * len(header))

    # Metrics
    metrics = [
        ("Action diversity", "action_diversity", "{:.1%}"),
        ("Fires when close", "fires_when_close", "{:.1%}"),
        ("Wall avoidance", "wall_avoidance_rate", "{:.1%}"),
        ("Dodge on hit", "dodge_on_hit_rate", "{:.1%}"),
        ("Heuristic agreement", "agreement_with_heuristic", "{:.1%}"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<30}"
        for r in results:
            val = r.get(key)
            if val is not None:
                row += f"{fmt.format(val):>{col_width}}"
            else:
                row += f"{'N/A':>{col_width}}"
        print(row)

    # Action distribution
    print("\n" + "-" * 70)
    print("Action Distribution:")
    all_actions = set()
    for r in results:
        all_actions.update(r["action_distribution"].keys())

    for action_name in sorted(all_actions):
        row = f"  {action_name:<28}"
        for r in results:
            val = r["action_distribution"].get(action_name, 0)
            row += f"{val:>{col_width}.1%}"
        print(row)

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["heuristic"],
        choices=["heuristic", "imitation", "rl"],
        help="Model types to evaluate",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model weights (for imitation/rl models)",
    )
    parser.add_argument(
        "--imitation-path",
        default="models/imitation_model.pt",
        help="Path to imitation model weights",
    )
    parser.add_argument(
        "--rl-path",
        default="models/dqn_model.pt",
        help="Path to RL model weights",
    )
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--difficulty", type=float, default=0.5)

    args = parser.parse_args()

    # Always use heuristic as reference
    reference = HeuristicModel(difficulty=args.difficulty)

    all_results = []
    for model_type in args.models:
        print(f"\nEvaluating {model_type}...")

        if model_type == "heuristic":
            model = HeuristicModel(difficulty=args.difficulty)
            path_used = None
        elif model_type == "imitation":
            path_used = args.model_path or args.imitation_path
            if not os.path.exists(path_used):
                print(f"  SKIPPED: {path_used} not found")
                continue
            model = load_model_for_eval("imitation", path_used)
        elif model_type == "rl":
            path_used = args.model_path or args.rl_path
            if not os.path.exists(path_used):
                print(f"  SKIPPED: {path_used} not found")
                continue
            model = load_model_for_eval("rl", path_used)
        else:
            continue

        ref = reference if model_type != "heuristic" else None
        result = evaluate_model(model, rounds=args.rounds, reference_model=ref)
        all_results.append(result)

    if all_results:
        print_evaluation_report(all_results)
    else:
        print("\nNo models were evaluated. Check model paths.")
