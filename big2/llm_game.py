#!/usr/bin/env python3
"""
LLM-based Big 2 game system with support for LLM, neural network, and human players.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import litellm
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import nn
import os
from big2.nn import combo_to_action_vector
from big2.simulator.cards import (
    FLUSH,
    FOUR_KIND,
    FULLHOUSE,
    PAIR,
    PASS,
    SINGLE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPLE,
    Combo,
    card_name,
)
from big2.simulator.env import Big2Env

# ==============================================================================
# Pydantic Models
# ==============================================================================


class MoveResponse(BaseModel):
    """Structured response from LLM for move selection."""

    move_number: int = Field(..., description="The number of the selected move (1-indexed)")
    reasoning: str | None = Field(None, description="Optional reasoning for the move")


class LLMConfig(BaseModel):
    """Configuration for an LLM player."""

    model: str = Field(default="gpt-4o-mini", description="LiteLLM model identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=150, ge=1, le=4000)


class NNConfig(BaseModel):
    """Configuration for a neural network player."""

    model_path: str = Field(default="big2_model.pt", description="Path to trained model")
    device: str = Field(default="cpu", description="Device to run model on")


class PlayerConfig(BaseModel):
    """Configuration for a single player."""

    type: Literal["llm", "nn", "human"]
    llm_config: LLMConfig | None = None
    nn_config: NNConfig | None = None

    @model_validator(mode="after")
    def validate_config(self) -> PlayerConfig:
        """Ensure type-specific config is provided."""
        if self.type == "llm" and self.llm_config is None:
            self.llm_config = LLMConfig()
        elif self.type == "nn" and self.nn_config is None:
            self.nn_config = NNConfig()
        return self


class GameConfig(BaseModel):
    """Configuration for a game with multiple players."""

    players: list[PlayerConfig] = Field(..., min_length=4, max_length=4)

    @field_validator("players")
    @classmethod
    def validate_players(cls, v: list[PlayerConfig]) -> list[PlayerConfig]:
        """Ensure exactly 4 players with at most 1 human."""
        if len(v) != 4:
            raise ValueError("Must have exactly 4 players")
        human_count = sum(1 for p in v if p.type == "human")
        if human_count > 1:
            raise ValueError("Can have at most 1 human player")
        return v


# ==============================================================================
# Player Base Class and Implementations
# ==============================================================================


class Player(ABC):
    """Base class for all player types."""

    def __init__(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def select_move(self, hand: list[int], candidates: list[Combo], game_context: dict[str, Any]) -> Combo:
        """Select a move given the current game state."""
        pass

    @abstractmethod
    def observe_action(self, player_id: int, action: Combo, game_context: dict[str, Any]) -> None:
        """Observe an action taken by any player (including self)."""
        pass


class LLMPlayer(Player):
    """Player that uses an LLM to make decisions via litellm."""

    def __init__(self, player_id: int, config: LLMConfig):
        super().__init__(player_id)
        self.config = config
        self.messages: list[dict[str, str]] = []
        self._initialize_system_prompt()
        self.total_cost = 0.0

    def _initialize_system_prompt(self) -> None:
        """Set up the system prompt explaining Big 2 rules."""
        system_prompt = """You are playing Big 2 (Big Two), a card game. Here are the rules:

CARD RANKING:
- Ranks: 3 (lowest) < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2 (highest)
- Suits: ♦ (diamonds) < ♣ (clubs) < ♥ (hearts) < ♠ (spades)

COMBO TYPES (in order of precedence):
1. Single: One card
2. Pair: Two cards of the same rank
3. Triple: Three cards of the same rank
4. Straight: Five consecutive ranks (2 cannot be in a straight)
5. Flush: Five cards of the same suit
6. Full House: Three of a kind + a pair
7. Four of a Kind: Four cards of the same rank + one kicker
8. Straight Flush: Five consecutive cards of the same suit

GAMEPLAY:
- You must play a combo of the same type and size that beats the current trick
- Higher combos in 5-card types can beat lower types (e.g., Flush beats Straight)
- You can PASS if you don't want to play or can't beat the current trick
- The goal is to get rid of all your cards first

When given your hand and legal moves, respond with ONLY the move number (e.g., "1", "2", etc.).
"""
        self.messages.append({"role": "system", "content": system_prompt})

    def _format_hand(self, hand: list[int]) -> str:
        """Format hand for LLM display."""
        return " ".join(card_name(c) for c in sorted(hand))

    def _format_combo(self, combo: Combo) -> str:
        """Format a combo for display."""
        type_names = {
            PASS: "PASS",
            SINGLE: "Single",
            PAIR: "Pair",
            TRIPLE: "Triple",
            STRAIGHT: "Straight",
            FLUSH: "Flush",
            FULLHOUSE: "Full House",
            FOUR_KIND: "Four of a Kind",
            STRAIGHT_FLUSH: "Straight Flush",
        }
        if combo.type == PASS:
            return "PASS"
        cards_str = " ".join(card_name(c) for c in combo.cards)
        return f"{type_names.get(combo.type, f'Type{combo.type}')}: {cards_str}"

    def _format_legal_moves(self, candidates: list[Combo]) -> str:
        """Format legal moves as a numbered list."""
        lines = []
        for i, combo in enumerate(candidates, 1):
            lines.append(f"{i}. {self._format_combo(combo)}")
        return "\n".join(lines)

    def select_move(self, hand: list[int], candidates: list[Combo], game_context: dict[str, Any]) -> Combo:
        """Use LLM to select a move."""
        if not candidates:
            return Combo(PASS, [], ())

        # Build user message with current game state
        trick_pile = game_context.get("trick_pile")
        trick_player = game_context.get("trick_player")
        opponent_counts = game_context.get("opponent_counts", {})

        message_parts = [f"Your hand: {self._format_hand(hand)}", ""]

        if trick_pile and trick_pile.type != PASS:
            trick_str = self._format_combo(trick_pile)
            player_str = f" (played by Player {trick_player})" if trick_player is not None else ""
            message_parts.append(f"Current trick: {trick_str}{player_str}")
        else:
            message_parts.append("Current trick: None (you lead)")

        if opponent_counts:
            counts_str = ", ".join(f"P{pid}={count}" for pid, count in opponent_counts.items())
            message_parts.append(f"Opponent card counts: {counts_str}")

        message_parts.extend(["", "Legal moves:", self._format_legal_moves(candidates), ""])
        message_parts.append("Reply with ONLY the move number.")

        user_message = "\n".join(message_parts)
        self.messages.append({"role": "user", "content": user_message})

        # Call LLM via litellm
        try:
            response = litellm.completion(
                model=self.config.model,
                messages=self.messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            cost = litellm.completion_cost(completion_response=response)
            self.total_cost += float(cost) 

            assistant_message = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": assistant_message})

            # Parse move number
            # Try to extract the first integer from the response
            import re

            match = re.search(r"\d+", assistant_message)
            if match:
                move_idx = int(match.group()) - 1  # Convert to 0-indexed
                if 0 <= move_idx < len(candidates):
                    return candidates[move_idx]

            # Fallback: return first non-PASS if available, else PASS
            for combo in candidates:
                if combo.type != PASS:
                    return combo
            return candidates[0]

        except Exception as e:
            print(f"LLM error for player {self.player_id}: {e}")
            # Fallback to first move
            return candidates[0]

    def observe_action(self, player_id: int, action: Combo, game_context: dict[str, Any]) -> None:
        """Record an action in the message history for context."""
        # Add a system-like message to track game state
        if player_id != self.player_id:
            # action_str = self._format_combo(action)
            # observation = f"Player {player_id} played: {action_str}"
            # We add this as context, but not as a full user/assistant exchange
            # This helps the LLM understand what's happening
            # For now, we'll just track it implicitly through the next turn's context
            pass


class NNPlayer(Player):
    """Player that uses a trained neural network policy."""

    def __init__(self, player_id: int, config: NNConfig, n_players: int = 4):
        super().__init__(player_id)
        self.config = config
        self.policy = self._load_policy(n_players)

    def _load_policy(self, n_players: int) -> nn.Module:
        """Load the trained policy model."""
        from pathlib import Path

        from big2.nn import MLPPolicy, SetPoolPolicy

        model_path = self.config.model_path
        if not Path(model_path).is_absolute():
            # Try relative to big2 directory
            print(f"Loading model from {model_path}")
            base_dir = Path(__file__).parent
            model_path = os.path.join(base_dir, model_path)
            print(f"Loading model from {model_path}")

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try loading with both architectures
        try:
            policy = MLPPolicy(n_players=n_players, device=device).to(device)
            policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except Exception:
            policy = SetPoolPolicy(n_players=n_players, device=device).to(device)
            policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        policy.eval()
        return policy

    def select_move(self, hand: list[int], candidates: list[Combo], game_context: dict[str, Any]) -> Combo:
        """Use neural network to select a move."""
        if not candidates:
            return Combo(PASS, [], ())

        state = game_context.get("state")
        if state is None:
            # Fallback to first move
            return candidates[0]

        with torch.no_grad():
            st = torch.from_numpy(state[np.newaxis, :]).long().to(self.policy.device)
            action_feats = [[combo_to_action_vector(c) for c in candidates]]
            logits_list, values = self.policy(st, action_feats)
            logits = logits_list[0]
            idx = int(torch.argmax(logits).item())
            return candidates[idx]

    def observe_action(self, player_id: int, action: Combo, game_context: dict[str, Any]) -> None:
        """NN player doesn't need to track history explicitly."""
        pass


class HumanPlayer(Player):
    """Player controlled via API (waits for human input)."""

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.pending_move: Combo | None = None

    def select_move(self, hand: list[int], candidates: list[Combo], game_context: dict[str, Any]) -> Combo:
        """This should not be called directly; moves come via API."""
        raise RuntimeError("HumanPlayer.select_move should not be called directly")

    def set_move(self, move: Combo) -> None:
        """Set the move from API."""
        self.pending_move = move

    def observe_action(self, player_id: int, action: Combo, game_context: dict[str, Any]) -> None:
        """Human observes via UI/API, no need to track here."""
        pass


# ==============================================================================
# Game Runner
# ==============================================================================


class GameRunner:
    """Orchestrates a game with multiple player types."""

    def __init__(self, game_id: str, config: GameConfig):
        self.game_id = game_id
        self.config = config
        self.env = Big2Env(n_players=4)
        self.players: list[Player] = []
        self.human_player_id: int | None = None
        self.state = None
        self.last_trick_player: int | None = None
        self.history: list[dict[str, Any]] = []

        self._initialize_players()

    def _initialize_players(self) -> None:
        """Create player instances based on config."""
        for i, player_config in enumerate(self.config.players):
            if player_config.type == "llm":
                assert player_config.llm_config is not None
                player = LLMPlayer(i, player_config.llm_config)
            elif player_config.type == "nn":
                assert player_config.nn_config is not None
                player = NNPlayer(i, player_config.nn_config, n_players=4)
            elif player_config.type == "human":
                player = HumanPlayer(i)
                self.human_player_id = i
            else:
                raise ValueError(f"Unknown player type: {player_config.type}")
            self.players.append(player)

    def start_game(self) -> dict[str, Any]:
        """Start the game and return initial state."""
        self.state = self.env.reset()
        # Play until human's turn or game ends
        self._play_until_human_or_done()
        print(f"Total cost: {sum(player.total_cost if hasattr(player, 'total_cost') else 0.0 for player in self.players)}")
        return self.get_state()

    def _get_game_context(self, player_id: int) -> dict[str, Any]:
        """Build game context for a player."""
        # Get opponent counts
        opponent_counts = {}
        for i in range(4):
            if i != player_id:
                opponent_counts[i] = len(self.env.hands[i])

        return {
            "trick_pile": self.env.trick_pile,
            "trick_player": self.last_trick_player,
            "opponent_counts": opponent_counts,
            "state": self.env._obs(player_id),
            "passes_in_row": self.env.passes_in_row,
        }

    def _play_until_human_or_done(self) -> None:
        """Play AI/LLM turns until it's the human's turn or game ends."""
        while not self.env.done:
            current = self.env.current_player
            if current == self.human_player_id:
                # Human's turn, stop and wait for API input
                break

            # AI or LLM turn
            player = self.players[current]
            candidates = self.env.legal_candidates(current)
            if not candidates:
                candidates = [Combo(PASS, [], ())]

            game_context = self._get_game_context(current)
            action = player.select_move(self.env.hands[current], candidates, game_context)

            # Record action in history
            self.history.append(
                {
                    "player_id": current,
                    "action": self._format_combo_for_history(action),
                    "cards_remaining": len(self.env.hands[current]) - len(action.cards),
                }
            )

            # Update trick tracker
            if action.type != PASS:
                self.last_trick_player = current

            # Execute action
            self.state, done = self.env.step(action)

            # Notify all players of the action
            for p in self.players:
                p.observe_action(current, action, game_context)

            # Check if trick was cleared
            if self.env.trick_pile is None or self.env.trick_pile.type == PASS:
                self.last_trick_player = None

    def _format_combo_for_history(self, combo: Combo) -> dict[str, Any]:
        """Format combo for history storage."""
        type_names = {
            PASS: "PASS",
            SINGLE: "Single",
            PAIR: "Pair",
            TRIPLE: "Triple",
            STRAIGHT: "Straight",
            FLUSH: "Flush",
            FULLHOUSE: "Full House",
            FOUR_KIND: "Four of a Kind",
            STRAIGHT_FLUSH: "Straight Flush",
        }
        return {
            "type": combo.type,
            "type_name": type_names.get(combo.type, f"Type{combo.type}"),
            "cards": combo.cards,
            "display": " ".join(card_name(c) for c in combo.cards) if combo.cards else "PASS",
        }

    def submit_human_action(self, move_index: int) -> dict[str, Any]:
        """Submit a human player's action."""
        if self.env.done:
            raise ValueError("Game is already finished")

        if self.env.current_player != self.human_player_id:
            raise ValueError("Not human player's turn")

        candidates = self.env.legal_candidates(self.human_player_id)
        if not candidates:
            candidates = [Combo(PASS, [], ())]

        if move_index < 0 or move_index >= len(candidates):
            raise ValueError(f"Invalid move index: {move_index}")

        action = candidates[move_index]

        # Record in history
        self.history.append(
            {
                "player_id": self.human_player_id,
                "action": self._format_combo_for_history(action),
                "cards_remaining": len(self.env.hands[self.human_player_id]) - len(action.cards),
            }
        )

        # Update trick tracker
        if action.type != PASS:
            self.last_trick_player = self.human_player_id

        # Execute action
        self.state, done = self.env.step(action)

        # Check if trick was cleared
        if self.env.trick_pile is None or self.env.trick_pile.type == PASS:
            self.last_trick_player = None

        # Continue with AI/LLM players
        self._play_until_human_or_done()

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Get current game state."""
        current_player = self.env.current_player
        is_human_turn = current_player == self.human_player_id

        # Format trick
        trick = None
        if self.env.trick_pile is not None and self.env.trick_pile.type != PASS:
            trick = self._format_combo_for_history(self.env.trick_pile)
            trick["player"] = self.last_trick_player if self.last_trick_player is not None else current_player

        # Get human hand and legal moves if it's their turn
        human_hand = None
        legal_moves = None
        if self.human_player_id is not None:
            hand = sorted(self.env.hands[self.human_player_id])
            human_hand = {
                "cards": hand,
                "display": [card_name(c) for c in hand],
                "count": len(hand),
            }

            if is_human_turn:
                candidates = self.env.legal_candidates(self.human_player_id)
                if not candidates:
                    candidates = [Combo(PASS, [], ())]
                legal_moves = [self._format_combo_for_history(c) for c in candidates]

        # Player info
        players_info = []
        for i in range(4):
            player_config = self.config.players[i]
            players_info.append(
                {
                    "id": i,
                    "type": player_config.type,
                    "cards_left": len(self.env.hands[i]),
                    "is_current": i == current_player,
                }
            )

        return {
            "game_id": self.game_id,
            "current_player": current_player,
            "is_human_turn": is_human_turn,
            "human_player_id": self.human_player_id,
            "human_hand": human_hand,
            "trick": trick,
            "legal_moves": legal_moves,
            "players": players_info,
            "done": self.env.done,
            "winner": self.env.winner,
            "passes_in_row": self.env.passes_in_row,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Get full game history."""
        return self.history
