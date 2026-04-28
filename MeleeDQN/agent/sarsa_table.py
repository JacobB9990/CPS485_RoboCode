"""Persistent Q-table used by the MeleeSarsaBot conversion."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class SarsaTable:
    def __init__(self, action_count: int):
        self.action_count = action_count
        self.table: dict[str, list[float]] = defaultdict(lambda: [0.0] * self.action_count)

    def get(self, state: str) -> list[float]:
        return self.table[state]

    def load(self, path: str | Path) -> None:
        file_path = Path(path)
        if not file_path.exists():
            return
        with file_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for key, values in raw.items():
            if isinstance(values, list) and len(values) == self.action_count:
                self.table[key] = [float(value) for value in values]

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.table, handle, indent=2, sort_keys=True)
