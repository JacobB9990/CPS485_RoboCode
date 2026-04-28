# state_vector.py
import numpy as np

BATTLEFIELD_W = 800.0
BATTLEFIELD_H  = 600.0
MAX_DISTANCE   = 1000.0
MAX_ENERGY     = 100.0
MAX_VELOCITY   = 8.0

class StateBuilder:
    """
    Converts the raw state dict from server.py into a normalized
    numpy float32 vector suitable for direct input to a DQN.

    Output vector (8 values, all in [-1, 1] or [0, 1]):
        0  distance       0–1      (0 = touching, 1 = far corner)
        1  bearing       -1–1      (normalized from -π to π)
        2  enemy_energy   0–1
        3  my_energy      0–1
        4  enemy_velocity -1–1     (negative = reversing)
        5  enemy_heading -1–1      (normalized from 0 to 2π)
        6  my_x           0–1      (position on battlefield)
        7  my_y           0–1
    """

    def build(self, state: dict) -> np.ndarray:
        vec = np.array([
            self._norm(state["distance"],     0, MAX_DISTANCE),
            self._norm_angle(state["bearing"]),
            self._norm(state["enemy_energy"], 0, MAX_ENERGY),
            self._norm(state["my_energy"],    0, MAX_ENERGY),
            self._norm(state["velocity"],     -MAX_VELOCITY, MAX_VELOCITY),
            self._norm_angle(state["heading"]),
            self._norm(state["my_x"],         0, BATTLEFIELD_W),
            self._norm(state["my_y"],         0, BATTLEFIELD_H),
        ], dtype=np.float32)

        return np.clip(vec, -1.0, 1.0)

    def _norm(self, value: float, min_val: float, max_val: float) -> float:
        """Scale value to [0, 1]."""
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def _norm_angle(self, radians: float) -> float:
        """Normalize any radian angle to [-1, 1]."""
        import math
        normalized = math.atan2(math.sin(radians), math.cos(radians))
        return normalized / math.pi

    @property
    def state_dim(self) -> int:
        return 8