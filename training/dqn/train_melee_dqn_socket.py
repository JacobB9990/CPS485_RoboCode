"""Socket trainer for the melee AdvancedRobot bot.

This keeps the same DQN shape as the existing Jacob3_0 agent:
- fixed-size state vector
- replay buffer
- target network
- epsilon-greedy exploration

The Java bot sends one line per decision step:
STEP|reward|done|episode|tick|living|placement|survival|damageDealt|damageTaken|kills|switches|state_csv
"""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bots.python.dqn.agent.melee_dqn_agent import DQNAgent

STATE_DIM = 48
N_ACTIONS = 15
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "dqn" / "melee_training_log.jsonl"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "data" / "checkpoints" / "dqn" / "melee_dqn_weights.pt"


@dataclass
class StepPacket:
    reward: float
    done: bool
    episode: int
    tick: int
    living_enemies: int
    placement: int
    survival_ticks: int
    damage_dealt: float
    damage_taken: float
    kills: int
    target_switches: int
    state: np.ndarray


class MeleeTrainer:
    def __init__(self, host: str, port: int, log_path: Path, weights_path: Path) -> None:
        self.host = host
        self.port = port
        self.log_path = log_path
        self.agent = DQNAgent(
            n_observations=STATE_DIM,
            n_actions=N_ACTIONS,
            weights_path=str(weights_path),
        )
        self.prev_state: np.ndarray | None = None
        self.prev_action: int | None = None
        self.episode_rewards: list[float] = []

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            print(f"[trainer] listening on {self.host}:{self.port}")

            while True:
                conn, addr = server.accept()
                print(f"[trainer] connected from {addr}")
                with conn:
                    self.handle_connection(conn)

    def handle_connection(self, conn: socket.socket) -> None:
        reader = conn.makefile("r", encoding="utf-8", newline="\n")
        writer = conn.makefile("w", encoding="utf-8", newline="\n")

        for raw_line in reader:
            line = raw_line.strip()
            if not line:
                continue

            packet = self.parse_packet(line)
            if packet is None:
                writer.write(f"ACTION|{self.safe_fallback_action()}\n")
                writer.flush()
                continue

            self.episode_rewards.append(packet.reward)

            if self.prev_state is not None and self.prev_action is not None:
                next_state = np.zeros(STATE_DIM, dtype=np.float32) if packet.done else packet.state
                self.agent.push_transition(
                    self.prev_state,
                    self.prev_action,
                    next_state,
                    packet.reward,
                    packet.done,
                )

            if packet.done:
                self.log_episode(packet)
                self.prev_state = None
                self.prev_action = None
                self.episode_rewards.clear()
                self.agent.episodes += 1
                self.agent.save()
                writer.write(f"ACTION|{self.safe_fallback_action()}\n")
                writer.flush()
                continue

            action = self.agent.select_action(packet.state, explore_fire_bias=packet.living_enemies <= 2)
            self.prev_state = packet.state
            self.prev_action = action
            writer.write(f"ACTION|{action}\n")
            writer.flush()

        self.prev_state = None
        self.prev_action = None

    def parse_packet(self, line: str) -> StepPacket | None:
        if not line.startswith("STEP|"):
            return None

        try:
            parts = line.split("|", maxsplit=12)
            state = np.array([float(value) for value in parts[12].split(",")], dtype=np.float32)
            if state.shape[0] != STATE_DIM:
                return None

            return StepPacket(
                reward=float(parts[1]),
                done=bool(int(parts[2])),
                episode=int(parts[3]),
                tick=int(parts[4]),
                living_enemies=int(parts[5]),
                placement=int(parts[6]),
                survival_ticks=int(parts[7]),
                damage_dealt=float(parts[8]),
                damage_taken=float(parts[9]),
                kills=int(parts[10]),
                target_switches=int(parts[11]),
                state=state,
            )
        except (ValueError, IndexError):
            return None

    def safe_fallback_action(self) -> int:
        return 10

    def curriculum_enemy_count(self, episode: int) -> tuple[int, int]:
        if episode < 200:
            return (2, 3)
        if episode < 700:
            return (4, 6)
        return (7, 10)

    def log_episode(self, packet: StepPacket) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        enemy_min, enemy_max = self.curriculum_enemy_count(packet.episode)
        payload = {
            "episode": packet.episode,
            "curriculum_enemy_range": [enemy_min, enemy_max],
            "survival_time": packet.survival_ticks,
            "placement": packet.placement,
            "damage_dealt": round(packet.damage_dealt, 3),
            "damage_taken": round(packet.damage_taken, 3),
            "kills": packet.kills,
            "target_switch_count": packet.target_switches,
            "average_reward": round(sum(self.episode_rewards) / max(1, len(self.episode_rewards)), 4),
            "epsilon": round(self.agent.current_epsilon(), 4),
            "buffer_size": len(self.agent.memory),
            "training_steps": self.agent.steps_done,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Melee DQN socket trainer for classic Robocode")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    trainer = MeleeTrainer(
        host=args.host,
        port=args.port,
        log_path=args.log_path,
        weights_path=args.weights_path,
    )
    trainer.serve_forever()


if __name__ == "__main__":
    main()
