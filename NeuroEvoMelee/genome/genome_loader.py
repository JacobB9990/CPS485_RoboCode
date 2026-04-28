"""Genome JSON loader for NeuroEvoMelee."""

from __future__ import annotations

import json
from pathlib import Path

from .genome_network import GenomeNetwork


class GenomeLoader:
    @staticmethod
    def load(genome_path: str | Path, expected_input_size: int, expected_output_size: int) -> GenomeNetwork:
        path = Path(genome_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            input_size = int(raw.get("inputSize", expected_input_size))
            hidden_size = int(raw.get("hiddenSize", 24))
            output_size = int(raw.get("outputSize", expected_output_size))
            if input_size != expected_input_size or output_size != expected_output_size:
                raise ValueError("Genome shape mismatch")
            w1 = GenomeLoader._extract_array(raw, "w1", input_size * hidden_size)
            b1 = GenomeLoader._extract_array(raw, "b1", hidden_size)
            w2 = GenomeLoader._extract_array(raw, "w2", hidden_size * output_size)
            b2 = GenomeLoader._extract_array(raw, "b2", output_size)
            return GenomeNetwork(input_size, hidden_size, output_size, w1, b1, w2, b2)
        except Exception:
            return GenomeNetwork.create_fallback(expected_input_size, 24, expected_output_size)

    @staticmethod
    def _extract_array(raw: dict[str, object], key: str, expected_length: int) -> list[float]:
        value = raw.get(key)
        if not isinstance(value, list) or len(value) != expected_length:
            raise ValueError(f"Invalid array for {key}")
        return [float(item) for item in value]
