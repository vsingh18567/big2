"""Big 2 Game Server - FastAPI application."""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware

from big2.game_server.auth import verify_api_key
from big2.game_server.models import (
    ActionStatus,
    ErrorResponse,
    GetGameStateResponse,
    JoinGameResponse,
    RegisterGameResponse,
    SubmitActionRequest,
    SubmitActionResponse,
)
from big2.game_server.store import game_store

# Create FastAPI app
app = FastAPI(
    title="Big 2 Game Server",
    description="HTTP API for Big 2 card game with JSON request/response",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Big 2 Game Server"}


@app.post(
    "/api/game/register/{game_id}",
    response_model=RegisterGameResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Game created successfully"},
        400: {"model": ErrorResponse, "description": "Game already exists"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
    },
    tags=["Game Management"],
)
async def register_game(
    game_id: Annotated[str, Path(description="Unique game identifier")],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> RegisterGameResponse:
    """
    Register a new game.

    Creates a new game with the specified ID. The game will be in 'waiting' state
    until 4 players join.
    """
    try:
        game = game_store.create_game(game_id)
        return RegisterGameResponse(
            game_id=game.game_id,
            game_created_at=game.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@app.post(
    "/api/game/join/{game_id}",
    response_model=JoinGameResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully joined game"},
        400: {"model": ErrorResponse, "description": "Cannot join game"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Game not found"},
    },
    tags=["Game Management"],
)
async def join_game(
    game_id: Annotated[str, Path(description="Game ID to join")],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> JoinGameResponse:
    """
    Join an existing game.

    Adds a player to the game. When the 4th player joins, the game automatically
    starts and cards are dealt.
    """
    game = game_store.get_game(game_id)
    if not game:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Game not found")

    try:
        player_id, token = game.add_player()
        return JoinGameResponse(
            game_id=game.game_id,
            game_created_at=game.created_at,
            player_id=player_id,
            num_players=len(game.players),
            token=token,
            state=game.state,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@app.get(
    "/api/game/state/{game_id}",
    response_model=GetGameStateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Game state retrieved"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Game not found"},
    },
    tags=["Game State"],
)
async def get_game_state(
    game_id: Annotated[str, Path(description="Game ID")],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> GetGameStateResponse:
    """
    Get current game state.

    Returns complete game state including turn information, move history,
    cards remaining per player, and winner (if game finished).
    """
    game = game_store.get_game(game_id)
    if not game:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Game not found")

    current_turn = None
    if game.state.value == "started" and game.turn_order:
        current_turn = game.turn_order[game.current_turn_idx]

    return GetGameStateResponse(
        state=game.state,
        turn_number=game.turn_number,
        turn=current_turn,
        turn_order=game.turn_order,
        last_move_id=game.last_move_id,
        last_move_at=game.last_move_at,
        move_count=game.move_count,
        cards_left=game.get_cards_left(),
        game_history=game.game_history,
        winner=game.winner,
    )


@app.post(
    "/api/game/action/{game_id}",
    response_model=SubmitActionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Action processed (check status field for result)"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Game not found"},
    },
    tags=["Game Actions"],
)
async def submit_action(
    game_id: Annotated[str, Path(description="Game ID")],
    request: SubmitActionRequest,
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> SubmitActionResponse:
    """
    Submit a game action (play cards or pass).

    Validates the move according to Big 2 rules and applies it if legal.
    Returns detailed status including success/failure reason.

    The action is validated for:
    - Correct player turn
    - Valid card combination
    - Legal move according to Big 2 rules
    - Player owns the cards being played
    """
    game = game_store.get_game(game_id)
    if not game:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Game not found")

    # Get player ID from token
    player = game.get_player_by_token(request.token)
    if not player:
        return SubmitActionResponse(
            status=ActionStatus.ERROR,
            applied_move_id=None,
            turn_number=None,
            move_count=None,
            reason="Invalid token",
        )

    player_id = player.player_id

    # Validate action
    status_result, reason = game.validate_action(player_id, request.token, request.cards, request.expected_turn_number)

    if status_result != ActionStatus.SUCCESS:
        return SubmitActionResponse(
            status=status_result,
            applied_move_id=None,
            turn_number=None,
            move_count=None,
            reason=reason,
        )

    # Apply action
    move_id = game.apply_action(player_id, request.cards)

    return SubmitActionResponse(
        status=ActionStatus.SUCCESS,
        applied_move_id=move_id,
        turn_number=game.turn_number,
        move_count=game.move_count,
        reason=None,
    )


# Optional: Admin/debug endpoints (not in RPC.md but useful for testing)


@app.get(
    "/api/admin/games",
    tags=["Admin"],
    response_model=list[str],
)
async def list_games(
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> list[str]:
    """List all game IDs (admin/debug endpoint)."""
    return game_store.list_games()


@app.delete(
    "/api/admin/game/{game_id}",
    tags=["Admin"],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_game(
    game_id: Annotated[str, Path(description="Game ID to delete")],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> None:
    """Delete a game (admin/debug endpoint)."""
    if not game_store.delete_game(game_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Game not found")
