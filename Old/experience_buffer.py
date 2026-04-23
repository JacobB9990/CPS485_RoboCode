import json
import os
import random

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExperienceBuffer:

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._buffer: list[dict] = []
        self._position: int = 0

    
    # Adding data
    def add(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition
        self._position = (self._position + 1) % self._capacity

    def add_batch(self, experiences: list[dict]) -> None:
        for exp in experiences:
            self.add(
                exp["state"],
                exp["action"],
                exp["reward"],
                exp["next_state"],
                exp["done"],
            )

    
    # Sampling 
    def sample(self, batch_size: int) -> dict:
        """Sample a random mini-batch.

        Returns tensors when PyTorch is available, plain lists otherwise.
        """
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))

        states = [t["state"] for t in batch]
        actions = [t["action"] for t in batch]
        rewards = [t["reward"] for t in batch]
        next_states = [t["next_state"] for t in batch]
        dones = [t["done"] for t in batch]

        if TORCH_AVAILABLE:
            return {
                "states": torch.FloatTensor(states),
                "actions": torch.LongTensor(actions),
                "rewards": torch.FloatTensor(rewards),
                "next_states": torch.FloatTensor(next_states),
                "dones": torch.BoolTensor(dones),
            }

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    
    # Persistence
    def save(self, path: str) -> None:
        """Write buffer contents to a JSONL file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for transition in self._buffer:
                f.write(json.dumps(transition) + "\n")
        print(f"[ExperienceBuffer] Saved {len(self._buffer)} experiences to {path}")

    def load(self, path: str) -> None:
        """Append transitions from a JSONL file into the buffer."""
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    self.add(
                        t["state"], t["action"], t["reward"],
                        t["next_state"], t["done"],
                    )
        print(f"[ExperienceBuffer] Loaded {len(self._buffer)} experiences from {path}")

    def __len__(self) -> int:
        return len(self._buffer)
