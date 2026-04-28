# replay_buffer.py
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Circular buffer storing (state, action, reward, next_state, done) tuples.

    One instance per DQN category — experiences never bleed across policies.
    Sampling returns ready-to-train numpy arrays so the agent never
    touches raw Python lists.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self,
             state:      np.ndarray,
             action:     int,
             reward:     float,
             next_state: np.ndarray,
             done:       bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        Returns a tuple of five numpy arrays:
            (states, actions, rewards, next_states, dones)
        All shaped for direct use in a PyTorch training step.
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self, batch_size: int = 64) -> bool:
        """True once the buffer has enough samples to train on."""
        return len(self.buffer) >= batch_size