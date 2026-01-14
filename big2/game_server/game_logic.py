"""Core game logic integrating with the Big 2 simulator."""

import secrets
from datetime import UTC, datetime
from threading import Lock
from uuid import UUID, uuid4

from big2.game_server.models import (
    ActionStatus,
    Card,
    GameState,
    MoveHistoryEntry,
    Rank,
    Suit,
)
from big2.simulator.cards import (
    PASS,
    Combo,
    hand_to_combo,
)
from big2.simulator.env import Big2Env

# Mapping between API model and simulator internals
SUIT_TO_INT = {"diamonds": 0, "clubs": 1, "hearts": 2, "spades": 3}
INT_TO_SUIT = {0: Suit.DIAMONDS, 1: Suit.CLUBS, 2: Suit.HEARTS, 3: Suit.SPADES}

RANK_TO_INT = {
    "3": 0,
    "4": 1,
    "5": 2,
    "6": 3,
    "7": 4,
    "8": 5,
    "9": 6,
    "T": 7,
    "10": 7,
    "J": 8,
    "Q": 9,
    "K": 10,
    "A": 11,
    "2": 12,
}
INT_TO_RANK = {
    0: Rank.THREE,
    1: Rank.FOUR,
    2: Rank.FIVE,
    3: Rank.SIX,
    4: Rank.SEVEN,
    5: Rank.EIGHT,
    6: Rank.NINE,
    7: Rank.TEN,
    8: Rank.JACK,
    9: Rank.QUEEN,
    10: Rank.KING,
    11: Rank.ACE,
    12: Rank.TWO,
}


def card_to_int(card: Card) -> int:
    """Convert API Card model to simulator card integer."""
    rank_int = RANK_TO_INT[card.rank.value]
    suit_int = SUIT_TO_INT[card.suit.value]
    return rank_int * 4 + suit_int


def int_to_card(card_id: int) -> Card:
    """Convert simulator card integer to API Card model."""
    rank_int = card_id // 4
    suit_int = card_id % 4
    return Card(suit=INT_TO_SUIT[suit_int], rank=INT_TO_RANK[rank_int])


class Player:
    """Represents a player in the game."""

    def __init__(self, player_id: str, token: str):
        self.player_id = player_id
        self.token = token
        self.hand: list[int] = []  # Card IDs in simulator format


class Game:
    """Represents a Big 2 game with state management."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.created_at = datetime.now(UTC)
        self.state = GameState.WAITING
        self.players: list[Player] = []
        self.turn_order: list[str] = []
        self.current_turn_idx: int = 0
        self.turn_number: int = 0
        self.move_count: int = 0
        self.game_history: list[MoveHistoryEntry] = []
        self.last_move_id: UUID | None = None
        self.last_move_at: datetime | None = None
        self.winner: str | None = None
        self.env: Big2Env | None = None
        self.lock = Lock()

    def add_player(self) -> tuple[str, str]:
        """
        Add a player to the game.

        Returns:
            Tuple of (player_id, token)

        Raises:
            ValueError: If game is full or already started
        """
        with self.lock:
            if len(self.players) >= 4:
                raise ValueError("Game is full (4 players max)")
            if self.state != GameState.WAITING:
                raise ValueError("Game has already started")

            player_id = f"player_{secrets.token_hex(8)}"
            token = secrets.token_urlsafe(32)
            player = Player(player_id, token)
            self.players.append(player)

            # Start game when 4 players join
            if len(self.players) == 4:
                self._start_game()

            return player_id, token

    def _start_game(self) -> None:
        """Initialize and start the game (called when 4 players join)."""
        self.state = GameState.STARTED
        self.env = Big2Env(n_players=4)
        self.env.reset()

        # Map simulator hands to player hands
        for i, player in enumerate(self.players):
            player.hand = self.env.hands[i].copy()

        # Determine turn order based on who has 3 of diamonds
        starting_idx = self.env.current_player
        self.turn_order = [self.players[(starting_idx + i) % 4].player_id for i in range(4)]
        self.current_turn_idx = 0
        self.turn_number = 1

    def get_player_by_id(self, player_id: str) -> Player | None:
        """Get player by ID."""
        for player in self.players:
            if player.player_id == player_id:
                return player
        return None

    def get_player_by_token(self, token: str) -> Player | None:
        """Get player by token."""
        for player in self.players:
            if player.token == token:
                return player
        return None

    def validate_action(
        self, player_id: str, token: str, cards: list[Card], expected_turn_number: int | None
    ) -> tuple[ActionStatus, str | None]:
        """
        Validate a player action.

        Returns:
            Tuple of (status, reason)
        """
        with self.lock:
            # Check game state
            if self.state == GameState.WAITING:
                return ActionStatus.ERROR, "Game has not started yet"
            if self.state == GameState.FINISHED:
                return ActionStatus.GAME_OVER, "Game has already finished"

            # Verify player token
            player = self.get_player_by_token(token)
            if not player or player.player_id != player_id:
                return ActionStatus.ERROR, "Invalid player or token"

            # Check turn
            current_player_id = self.turn_order[self.current_turn_idx]
            if player_id != current_player_id:
                return ActionStatus.OUT_OF_TURN, f"Not your turn (current: {current_player_id})"

            # Check expected turn number
            if expected_turn_number is not None and expected_turn_number != self.turn_number:
                return (
                    ActionStatus.STALE_TURN,
                    f"expected turn_number {self.turn_number} but got {expected_turn_number}",
                )

            # Convert cards to simulator format
            if len(cards) == 0:
                # Pass move
                combo = Combo(PASS, [], ())
            else:
                card_ints = [card_to_int(c) for c in cards]
                # Verify player has all cards
                for card_int in card_ints:
                    if card_int not in player.hand:
                        return ActionStatus.ILLEGAL_MOVE, f"You don't have card {int_to_card(card_int)}"

                combo = hand_to_combo(card_ints)
                if combo is None:
                    return ActionStatus.ILLEGAL_MOVE, "Invalid card combination"

            # Check if move is legal according to game rules
            if self.env is None:
                return ActionStatus.ERROR, "Game environment not initialized"

            legal_combos = self.env.legal_candidates(self.env.current_player)
            if combo not in legal_combos:
                return ActionStatus.ILLEGAL_MOVE, "Move violates game rules"

            return ActionStatus.SUCCESS, None

    def apply_action(self, player_id: str, cards: list[Card]) -> UUID:
        """
        Apply a validated action to the game state.

        Args:
            player_id: Player making the move
            cards: Cards played (empty for pass)

        Returns:
            Move ID
        """
        with self.lock:
            move_id = uuid4()
            self.last_move_id = move_id
            self.last_move_at = datetime.now(UTC)
            self.move_count += 1

            # Record in history
            self.game_history.append(MoveHistoryEntry(player=player_id, cards=cards))

            # Apply to simulator
            if len(cards) == 0:
                combo = Combo(PASS, [], ())
            else:
                card_ints = [card_to_int(c) for c in cards]
                combo = hand_to_combo(card_ints)

            if self.env:
                # Update player hand
                player = self.get_player_by_id(player_id)
                if player and combo and combo.type != PASS:
                    for card_int in combo.cards:
                        if card_int in player.hand:
                            player.hand.remove(card_int)

                # Apply to environment
                _, done = self.env.step(combo)

                if done:
                    self.state = GameState.FINISHED
                    self.winner = player_id
                else:
                    # Advance turn
                    self.current_turn_idx = (self.current_turn_idx + 1) % 4
                    self.turn_number += 1

            return move_id

    def get_cards_left(self) -> dict[str, int]:
        """Get cards remaining for each player."""
        return {player.player_id: len(player.hand) for player in self.players}
