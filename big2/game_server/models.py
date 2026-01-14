"""Pydantic models for the Big 2 game server API."""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Suit(str, Enum):
    """Card suits."""

    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    HEARTS = "hearts"
    SPADES = "spades"


class Rank(str, Enum):
    """Card ranks."""

    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"
    TWO = "2"


class Card(BaseModel):
    """A playing card."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"suit": "hearts", "rank": "10"},
                {"suit": "spades", "rank": "A"},
            ]
        }
    )

    suit: Suit = Field(..., description="The suit of the card")
    rank: Rank = Field(..., description="The rank of the card")


class GameState(str, Enum):
    """Game state enum."""

    WAITING = "waiting"
    STARTED = "started"
    FINISHED = "finished"
    ERROR = "error"


class ActionType(str, Enum):
    """Action types."""

    PLAY = "play"
    PASS = "pass"


class ActionStatus(str, Enum):
    """Action response status."""

    SUCCESS = "success"
    ERROR = "error"
    ILLEGAL_MOVE = "illegal_move"
    OUT_OF_TURN = "out_of_turn"
    STALE_TURN = "stale_turn"
    GAME_OVER = "game_over"


class RegisterGameResponse(BaseModel):
    """Response for game registration."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"game_id": "1234567890", "game_created_at": "2021-01-01T00:00:00Z"}]}
    )

    game_id: str = Field(..., description="Unique game identifier")
    game_created_at: datetime = Field(..., description="ISO 8601 timestamp of game creation")


class JoinGameResponse(BaseModel):
    """Response for joining a game."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "game_id": "1234567890",
                    "game_created_at": "2021-01-01T00:00:00Z",
                    "player_id": "player_abc",
                    "num_players": 2,
                    "token": "secret_token_xyz",
                    "state": "waiting",
                }
            ]
        }
    )

    game_id: str = Field(..., description="Unique game identifier")
    game_created_at: datetime = Field(..., description="ISO 8601 timestamp of game creation")
    player_id: str = Field(..., description="Unique player identifier")
    num_players: Literal[1, 2, 3, 4] = Field(..., description="Number of players currently in the game")
    token: str = Field(..., description="Secret token for authentication")
    state: GameState = Field(..., description="Current game state")


class MoveHistoryEntry(BaseModel):
    """A single move in game history."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "player": "player_1",
                    "cards": [
                        {"suit": "hearts", "rank": "10"},
                        {"suit": "diamonds", "rank": "10"},
                    ],
                },
                {"player": "player_2", "cards": []},
            ]
        }
    )

    player: str = Field(..., description="Player ID who made the move")
    cards: list[Card] = Field(..., description="Cards played (empty array means pass)")


class GetGameStateResponse(BaseModel):
    """Response for getting game state."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "state": "started",
                    "turn_number": 12,
                    "turn": "player_1",
                    "turn_order": ["player_1", "player_2", "player_3", "player_4"],
                    "last_move_id": "bcaa9f25-a8ef-4db1-a833-7f8dba0c9f3d",
                    "last_move_at": "2021-01-01T00:05:02Z",
                    "move_count": 18,
                    "cards_left": {"player_1": 10, "player_2": 8, "player_3": 5, "player_4": 12},
                    "game_history": [
                        {
                            "player": "player_1",
                            "cards": [
                                {"suit": "hearts", "rank": "10"},
                                {"suit": "diamonds", "rank": "10"},
                            ],
                        }
                    ],
                    "winner": None,
                }
            ]
        }
    )

    state: GameState = Field(..., description="Current game state")
    turn_number: int = Field(..., description="Monotonically increasing turn sequence")
    turn: str | None = Field(None, description="Current player ID (null if game not started)")
    turn_order: list[str] = Field(default_factory=list, description="Fixed turn order (set when game starts)")
    last_move_id: UUID | None = Field(None, description="Server-generated ID of last move")
    last_move_at: datetime | None = Field(None, description="Timestamp of last move")
    move_count: int = Field(..., description="Total applied moves for this game")
    cards_left: dict[str, int] = Field(default_factory=dict, description="Number of cards remaining per player")
    game_history: list[MoveHistoryEntry] = Field(default_factory=list, description="Complete game history")
    winner: str | None = Field(None, description="Winner player ID (if game finished)")


class SubmitActionRequest(BaseModel):
    """Request to submit an action (play or pass)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "action": "play",
                    "cards": [
                        {"suit": "hearts", "rank": "10"},
                        {"suit": "diamonds", "rank": "10"},
                    ],
                    "expected_turn_number": 12,
                    "token": "secret_token_xyz",
                },
                {"action": "pass", "cards": [], "token": "secret_token_xyz"},
            ]
        }
    )

    action: ActionType = Field(..., description="Action type")
    cards: list[Card] = Field(default_factory=list, description="Cards to play (empty for pass)")
    expected_turn_number: int | None = Field(None, description="Optional guard against stale/out-of-order moves")
    token: str = Field(..., description="Secret authentication token")


class SubmitActionResponse(BaseModel):
    """Response to action submission."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "applied_move_id": "bcaa9f25-a8ef-4db1-a833-7f8dba0c9f3d",
                    "turn_number": 13,
                    "move_count": 19,
                    "reason": None,
                },
                {
                    "status": "stale_turn",
                    "applied_move_id": None,
                    "turn_number": None,
                    "move_count": None,
                    "reason": "expected turn_number 12 but got 11",
                },
            ]
        }
    )

    status: ActionStatus = Field(..., description="Result status")
    applied_move_id: UUID | None = Field(None, description="Server-generated move ID (if applied)")
    turn_number: int | None = Field(None, description="Next expected turn number (if applied)")
    move_count: int | None = Field(None, description="Total move count after this action")
    reason: str | None = Field(None, description="Error/rejection reason (present on non-success)")


class ErrorResponse(BaseModel):
    """Generic error response."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"detail": "Game not found"}]})

    detail: str = Field(..., description="Error message")
