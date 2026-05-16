from __future__ import annotations

from big2.play.types import PlayAction, PlayTurn
from big2.simulator.cards import Combo
from big2.simulator.smart_strategy import smart_strategy


class SmartAgent:
    """Play-mode adapter for the simulator smart strategy."""

    def choose_action(self, turn: PlayTurn) -> PlayAction:
        action = smart_strategy(
            list(turn.legal_actions),
            list(turn.hand),
            turn.current_trick,
        )
        return PlayAction(action=action)

    def observe_action(self, actor_id: int, action: Combo) -> None:
        return None

    def on_game_end(self, winner: int | None) -> None:
        return None
