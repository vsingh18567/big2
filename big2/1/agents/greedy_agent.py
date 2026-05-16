from __future__ import annotations

from big2.play.types import PlayAction, PlayTurn
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.cards import Combo


class GreedyAgent:
    """Play-mode adapter for the simulator greedy strategy."""

    def choose_action(self, turn: PlayTurn) -> PlayAction:
        action = greedy_strategy(list(turn.legal_actions))
        return PlayAction(action=action)

    def observe_action(self, actor_id: int, action: Combo) -> None:
        return None

    def on_game_end(self, winner: int | None) -> None:
        return None
