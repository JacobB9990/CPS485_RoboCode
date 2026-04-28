"""Tracked enemy state for NeuroEvoMelee."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnemyState:
    id: int
    x: float = 0.0
    y: float = 0.0
    energy: float = 100.0
    direction: float = 0.0
    speed: float = 0.0
    last_seen_turn: int = 0
    alive: bool = True

    def age(self, current_turn: int) -> int:
        return max(0, current_turn - self.last_seen_turn)
