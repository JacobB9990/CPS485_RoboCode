from abc import ABC, abstractmethod

from game_state import GameState
from actions import Action


class ModelInterface(ABC):
    """Base class for all AI models that control the bot.

    Implement decide() at minimum. Override the lifecycle hooks
    for models that learn or need per-round setup/teardown.
    """

    @abstractmethod
    def decide(self, state: GameState) -> Action:
        """Given the current game state, decide what action to take.

        This is called once per bot tick. Implementations must return
        quickly enough to stay within the game tick rate.

        Args:
            state: Current game state snapshot.

        Returns:
            Action to execute this tick.
        """

    def on_round_start(self, round_number: int) -> None:
        """Called at the start of each round. Override for setup."""

    def on_round_end(self, round_number: int, won: bool) -> None:
        """Called at the end of each round. Override for learning/logging."""

    def on_reward(
        self,
        reward: float,
        state: GameState,
        action: Action,
        next_state: GameState,
    ) -> None:
        """Called when a reward signal is available. Override for RL training."""

    @property
    def name(self) -> str:
        """Human-readable name for this model."""
        return self.__class__.__name__
