"""
Usage:
    python AIBot.py                                # heuristic (default)
    python AIBot.py --model rl --collect            # RL with random exploration, save data
    python AIBot.py --model rl --model-path models/dqn_model.pt  # trained RL policy
"""

import argparse
import math
import os

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.events import (
    ScannedBotEvent,
    HitByBulletEvent,
    HitBotEvent,
    HitWallEvent,
    DeathEvent,
    WonRoundEvent,
    BulletHitBotEvent,
)

from game_state import GameState, EnemyState
from actions import Action
from model_interface import ModelInterface
from heuristic_model import HeuristicModel


def load_model(
    model_type: str,
    model_path: str | None = None,
    epsilon: float = 0.1,
    difficulty: float = 0.5,
) -> ModelInterface:
    """Factory: instantiate the requested AI model."""
    if model_type == "heuristic":
        return HeuristicModel(difficulty=difficulty)
    elif model_type == "imitation":
        from imitation_model import ImitationModel
        return ImitationModel(model_path=model_path)
    elif model_type == "rl":
        from rl_model import RLModel
        return RLModel(model_path=model_path, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class AIBot(Bot):
    """
    The bot owns the game-state extraction and action-execution logic.
    All strategic decisions are delegated to self._model.decide().
    """

    def __init__(
        self,
        model: ModelInterface,
        collect_experience: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self._collect = collect_experience
        self._state = GameState()
        self._round_number = 0
        self._turn_count = 0
        self._ticks_since_save = 0
        self._SAVE_INTERVAL = 200  # flush experiences every N ticks
        print(f"[AIBot] Model: {self._model.name}")

    
    # Main loop
    def run(self) -> None:
        self._round_number += 1
        self._model.on_round_start(self._round_number)
        self._update_own_state()

        while self.running:
            self._turn_count += 1
            self._ticks_since_save += 1

            # Reset per-tick event flags
            self._state.hit_by_bullet = False
            self._state.hit_wall = False
            self._state.hit_bot = False
            self._state.bullet_hit_enemy = False

            # Pull latest bot properties
            self._update_own_state()

            # Decide and act
            action = self._model.decide(self._state)
            self._execute_action(action)

            # Periodically flush experiences so data isn't lost on Ctrl+C
            if self._collect and self._ticks_since_save >= self._SAVE_INTERVAL:
                self._save_experiences()
                self._ticks_since_save = 0

            # Default: spin radar to keep scanning
            self.turn_radar_right(360)

    
    # State extraction
    def _update_own_state(self) -> None:
        self._state.x = self.x
        self._state.y = self.y
        self._state.energy = self.energy
        self._state.direction = self.direction
        self._state.gun_direction = self.gun_direction
        self._state.radar_direction = self.radar_direction
        self._state.velocity = self.speed
        self._state.gun_heat = self.gun_heat
        self._state.arena_width = self.arena_width
        self._state.arena_height = self.arena_height
        self._state.turn_number = self._turn_count
        self._state.enemy_count = self.enemy_count
        self._state.round_number = self._round_number

    
    # Action execution
    def _execute_action(self, action: Action) -> None:
        """Translate an Action into sequential Robocode API calls."""

        # 1. Gun turn (prioritized -- aiming matters most)
        if abs(action.gun_turn) > 0.5:
            if action.gun_turn > 0:
                self.turn_gun_right(min(action.gun_turn, 20))
            else:
                self.turn_gun_left(min(abs(action.gun_turn), 20))

        # 2. Fire (if gun is cool and we have enough energy)
        if (
            action.fire_power > 0
            and self.gun_heat == 0
            and self.energy > action.fire_power
        ):
            self.fire(max(0.1, min(3.0, action.fire_power)))

        # 3. Body turn
        if abs(action.body_turn) > 0.5:
            if action.body_turn > 0:
                self.turn_right(min(abs(action.body_turn), 45))
            else:
                self.turn_left(min(abs(action.body_turn), 45))

        # 4. Body movement
        if abs(action.body_move) > 1:
            if action.body_move > 0:
                self.forward(min(action.body_move, 150))
            else:
                self.back(min(abs(action.body_move), 150))

    
    # Event handlers
    def on_scanned_bot(self, e: ScannedBotEvent) -> None:
        enemy = EnemyState(
            x=float(e.x),
            y=float(e.y),
            energy=float(e.energy),
            direction=float(e.direction),
            speed=float(e.speed),
            scan_turn=self._turn_count,
        )
        self._state.enemies[e.scanned_bot_id] = enemy

    def on_hit_by_bullet(self, e: HitByBulletEvent) -> None:
        self._state.hit_by_bullet = True
        self._state.bullet_bearing = self.calc_bearing(e.bullet.direction)

    def on_hit_wall(self, e: HitWallEvent) -> None:
        self._state.hit_wall = True

    def on_hit_bot(self, e: HitBotEvent) -> None:
        self._state.hit_bot = True
        enemy = EnemyState(
            x=float(e.x),
            y=float(e.y),
            energy=float(e.energy),
            direction=0.0,
            speed=0.0,
            scan_turn=self._turn_count,
        )
        self._state.enemies[e.victim_id] = enemy

    def on_bullet_hit(self, e: BulletHitBotEvent) -> None:
        self._state.bullet_hit_enemy = True

    def on_death(self, e: DeathEvent) -> None:
        self._model.on_round_end(self._round_number, won=False)
        self._save_experiences()

    def on_won_round(self, e: WonRoundEvent) -> None:
        self._model.on_round_end(self._round_number, won=True)
        self._save_experiences()
        self.turn_right(360)

    
    # Experience persistence   
    def _save_experiences(self) -> None:
        if not self._collect:
            return
        try:
            from rl_model import RLModel
            if isinstance(self._model, RLModel):
                from experience_buffer import ExperienceBuffer

                experiences = self._model.get_experiences()
                if not experiences:
                    return

                exp_dir = os.path.join(os.path.dirname(__file__), "experiences")
                exp_path = os.path.join(exp_dir, "replay.jsonl")

                buf = ExperienceBuffer()
                buf.load(exp_path)
                buf.add_batch(experiences)
                buf.save(exp_path)
        except Exception as e:
            print(f"[AIBot] Error saving experiences: {e}")


# Entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Robocode Bot")
    parser.add_argument(
        "--model",
        default="heuristic",
        choices=["heuristic", "imitation", "rl"],
        help="AI model to use (default: heuristic)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for RL model (default: 0.1)",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.5,
        help="Difficulty level for heuristic model 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect experience data for RL training",
    )

    args = parser.parse_args()

    model = load_model(args.model, args.model_path, args.epsilon, args.difficulty)
    bot = AIBot(model=model, collect_experience=args.collect)
    bot.start()


if __name__ == "__main__":
    main()
