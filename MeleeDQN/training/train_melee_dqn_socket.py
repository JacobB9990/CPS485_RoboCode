"""Socket-based trainer for the MeleeDQN bot."""

from __future__ import annotations

import argparse
import socket
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from MeleeDQN.agent.dqn_agent import DQNAgent
from MeleeDQN.runtime.melee_dqn_bot import ACTION_COUNT, STATE_SIZE, ActionType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = ROOT / "checkpoints" / "melee_dqn_weights.pt"


@dataclass
class SessionState:
    previous_state: np.ndarray | None = None
    previous_action: int | None = None


class MeleeDqnServer:
    def __init__(self, host: str, port: int, weights_path: str, eval_epsilon: float = 0.0) -> None:
        self.host = host
        self.port = port
        self.agent = DQNAgent(STATE_SIZE, ACTION_COUNT, weights_path=weights_path)
        self.agent.set_eval_mode(epsilon=eval_epsilon)

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"[MeleeDqnServer] listening on {self.host}:{self.port}")
            while True:
                connection, address = server_socket.accept()
                print(f"[MeleeDqnServer] client connected: {address}")
                threading.Thread(target=self._handle_client, args=(connection,), daemon=True).start()

    def _handle_client(self, connection: socket.socket) -> None:
        session = SessionState()
        with connection:
            reader = connection.makefile("r", encoding="utf-8", newline="\n")
            writer = connection.makefile("w", encoding="utf-8", newline="\n")
            with reader, writer:
                for raw_line in reader:
                    line = raw_line.strip()
                    if not line:
                        continue
                    response = self._process_line(line, session)
                    writer.write(f"ACTION|{response}\n")
                    writer.flush()

    def _process_line(self, line: str, session: SessionState) -> int:
        parts = line.split("|", 12)
        if len(parts) != 13 or parts[0] != "STEP":
            return int(ActionType.HEAD_TO_OPEN_SPACE)

        reward = float(parts[1])
        done = parts[2] == "1"
        state = np.fromstring(parts[12], sep=",", dtype=np.float32)
        if state.size != STATE_SIZE:
            return int(ActionType.HEAD_TO_OPEN_SPACE)

        if session.previous_state is not None and session.previous_action is not None:
            self.agent.push_transition(session.previous_state, session.previous_action, state, reward, done)

        if done:
            session.previous_state = None
            session.previous_action = None
            return int(ActionType.HEAD_TO_OPEN_SPACE)

        session.previous_state = state
        session.previous_action = self.agent.select_action(state)
        return session.previous_action


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MeleeDQN socket AI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    args = parser.parse_args()

    server = MeleeDqnServer(args.host, args.port, args.weights, args.eval_epsilon)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
