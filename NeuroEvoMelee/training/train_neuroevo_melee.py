"""Genetic trainer for the NeuroEvoMelee genome format."""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from NeuroEvoMelee.genome import FeatureEncoder, GenomeLoader, GenomeNetwork

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT_GENOME = ROOT / "data" / "current_genome.json"
DEFAULT_BEST_GENOME = ROOT / "data" / "best_genome.json"
DEFAULT_LOG_PATH = ROOT / "logs" / "evolution_log.jsonl"


@dataclass
class GenomeCandidate:
    genome: GenomeNetwork
    fitness: float = float("-inf")


def _serialize_genome(genome: GenomeNetwork, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(genome.to_json_dict(), indent=2), encoding="utf-8")


def _random_genome(input_size: int, hidden_size: int, output_size: int, rng: random.Random) -> GenomeNetwork:
    scale = 0.35
    w1 = [rng.uniform(-scale, scale) for _ in range(input_size * hidden_size)]
    b1 = [rng.uniform(-scale, scale) for _ in range(hidden_size)]
    w2 = [rng.uniform(-scale, scale) for _ in range(hidden_size * output_size)]
    b2 = [rng.uniform(-scale, scale) for _ in range(output_size)]
    return GenomeNetwork(input_size, hidden_size, output_size, w1, b1, w2, b2)


def _mutate(genome: GenomeNetwork, rng: random.Random, mutation_rate: float, mutation_std: float) -> GenomeNetwork:
    def perturb(values: list[float]) -> list[float]:
        mutated = []
        for value in values:
            if rng.random() < mutation_rate:
                if rng.random() < 0.03:
                    mutated.append(rng.uniform(-0.5, 0.5))
                else:
                    mutated.append(max(-2.0, min(2.0, value + rng.gauss(0.0, mutation_std))))
            else:
                mutated.append(value)
        return mutated

    return GenomeNetwork(genome.input_size, genome.hidden_size, genome.output_size, perturb(genome.w1), perturb(genome.b1), perturb(genome.w2), perturb(genome.b2))


def _crossover(parent_a: GenomeNetwork, parent_b: GenomeNetwork, rng: random.Random) -> GenomeNetwork:
    def mix(values_a: list[float], values_b: list[float]) -> list[float]:
        return [value_a if rng.random() < 0.5 else value_b for value_a, value_b in zip(values_a, values_b)]

    return GenomeNetwork(parent_a.input_size, parent_a.hidden_size, parent_a.output_size, mix(parent_a.w1, parent_b.w1), mix(parent_a.b1, parent_b.b1), mix(parent_a.w2, parent_b.w2), mix(parent_a.b2, parent_b.b2))


def _evaluate_locally(genome: GenomeNetwork, rng: random.Random) -> float:
    total = 0.0
    for _ in range(24):
        inputs = [rng.uniform(-1.0, 1.0) for _ in range(genome.input_size)]
        outputs = genome.forward(inputs)
        movement = 1.0 - abs(outputs[0])
        commitment = abs(outputs[1])
        fire_drive = max(0.0, outputs[2])
        preference = abs(outputs[3])
        total += 0.35 * movement + 0.25 * commitment + 0.20 * fire_drive + 0.20 * preference
    return total / 24.0


def _evaluate_with_command(genome_path: Path, command_template: str) -> float:
    command = command_template.format(genome=str(genome_path))
    completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return float("-inf")
    output = completed.stdout.strip().splitlines()
    if not output:
        return float("-inf")
    last_line = output[-1]
    try:
        payload = json.loads(last_line)
        for key in ("fitness", "score", "reward"):
            if key in payload:
                return float(payload[key])
    except json.JSONDecodeError:
        pass
    try:
        return float(last_line)
    except ValueError:
        return float(len(output))


def _append_log(log_path: Path, generation: int, best_fitness: float, mean_fitness: float, std_fitness: float, genome_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "generation": generation,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "current_genome": str(genome_path),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train NeuroEvoMelee genomes")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--mutation-rate", type=float, default=0.12)
    parser.add_argument("--mutation-std", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--current-genome", default=str(DEFAULT_CURRENT_GENOME))
    parser.add_argument("--best-genome", default=str(DEFAULT_BEST_GENOME))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--evaluate-command", default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    current_path = Path(args.current_genome)
    best_path = Path(args.best_genome)
    log_path = Path(args.log_path)

    base_genome = GenomeLoader.load(current_path, FeatureEncoder.INPUT_SIZE, 4)
    population: list[GenomeCandidate] = [GenomeCandidate(base_genome)]
    while len(population) < args.population:
        population.append(GenomeCandidate(_random_genome(base_genome.input_size, base_genome.hidden_size, base_genome.output_size, rng)))

    elite_count = max(1, int(round(args.population * args.elite_fraction)))

    for generation in range(1, args.generations + 1):
        for candidate in population:
            if args.evaluate_command.strip():
                _serialize_genome(candidate.genome, current_path)
                candidate.fitness = _evaluate_with_command(current_path, args.evaluate_command)
            else:
                candidate.fitness = _evaluate_locally(candidate.genome, rng)

        population.sort(key=lambda candidate: candidate.fitness, reverse=True)
        best = population[0]
        best_fitness = best.fitness
        mean_fitness = sum(candidate.fitness for candidate in population) / len(population)
        variance = sum((candidate.fitness - mean_fitness) ** 2 for candidate in population) / len(population)
        std_fitness = math.sqrt(max(variance, 0.0))
        _serialize_genome(best.genome, best_path)
        _serialize_genome(best.genome, current_path)
        _append_log(log_path, generation, best_fitness, mean_fitness, std_fitness, current_path)
        print(f"[NeuroEvoMelee] generation={generation} best={best_fitness:.4f} mean={mean_fitness:.4f} std={std_fitness:.4f}")

        elites = [GenomeCandidate(candidate.genome, candidate.fitness) for candidate in population[:elite_count]]
        next_population: list[GenomeCandidate] = elites[:]
        tournament_size = min(4, len(population))
        while len(next_population) < args.population:
            def pick_parent() -> GenomeNetwork:
                competitors = rng.sample(population[: max(elite_count * 2, len(population))], tournament_size)
                return max(competitors, key=lambda candidate: candidate.fitness).genome

            parent_a = pick_parent()
            parent_b = pick_parent()
            child = _crossover(parent_a, parent_b, rng)
            child = _mutate(child, rng, args.mutation_rate, args.mutation_std)
            next_population.append(GenomeCandidate(child))

        population = next_population

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
