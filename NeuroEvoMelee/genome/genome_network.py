"""Fixed-topology genome network used by NeuroEvoMelee."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class GenomeNetwork:
    input_size: int
    hidden_size: int
    output_size: int
    w1: list[float]
    b1: list[float]
    w2: list[float]
    b2: list[float]

    @classmethod
    def create_fallback(cls, input_size: int, hidden_size: int, output_size: int) -> "GenomeNetwork":
        w1 = [0.0] * (input_size * hidden_size)
        b1 = [0.0] * hidden_size
        w2 = [0.0] * (hidden_size * output_size)
        b2 = [0.0] * output_size
        if output_size > 1:
            b2[1] = 0.35
        if output_size > 2:
            b2[2] = -0.6
        return cls(input_size, hidden_size, output_size, w1, b1, w2, b2)

    def forward(self, inputs: Iterable[float]) -> list[float]:
        values = list(inputs)
        if len(values) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size} but got {len(values)}")

        hidden = [0.0] * self.hidden_size
        output = [0.0] * self.output_size

        for hidden_index in range(self.hidden_size):
            total = self.b1[hidden_index]
            base = hidden_index * self.input_size
            for input_index in range(self.input_size):
                total += self.w1[base + input_index] * values[input_index]
            hidden[hidden_index] = math.tanh(total)

        for output_index in range(self.output_size):
            total = self.b2[output_index]
            base = output_index * self.hidden_size
            for hidden_index in range(self.hidden_size):
                total += self.w2[base + hidden_index] * hidden[hidden_index]
            output[output_index] = math.tanh(total)

        return output

    def to_json_dict(self) -> dict[str, object]:
        return {
            "inputSize": self.input_size,
            "hiddenSize": self.hidden_size,
            "outputSize": self.output_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
