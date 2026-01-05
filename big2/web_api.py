#!/usr/bin/env python3
"""
FastAPI backend for Big 2 web interface.
"""

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from big2.nn import MLPPolicy, combo_to_action_vector
from big2.simulator.cards import PAIR, PASS, SINGLE, TRIPLE, Combo, card_name
from big2.simulator.env import Big2Env


def format_combo(combo: Combo) -> dict[str, Any]:
    """Format a combo for JSON serialization."""
    type_names = {
        PASS: "PASS",
        SINGLE: "Single",
        PAIR: "Pair",
        TRIPLE: "Triple",
        4: "Straight",
        5: "Flush",
        6: "Full House",
        7: "Four of a Kind",
        8: "Straight Flush",
    }
    type_name = type_names.get(combo.type, f"Type {combo.type}")
    cards_str = " ".join([card_name(c) for c in combo.cards]) if combo.cards else ""
    display = type_name if combo.type == PASS else f"{type_name}: {cards_str}"
    return {
        "type": combo.type,
        "type_name": type_name,
        "cards": combo.cards,
        "display": display,
    }


def ai_select_action(policy: MLPPolicy, state: np.ndarray, candidates: list[Combo]) -> Combo:
    """Select an action for an AI player using the trained policy."""
    with torch.no_grad():
        st = torch.from_numpy(state[np.newaxis, :]).long().to(policy.device)
        action_feats = [[combo_to_action_vector(c) for c in candidates]]
        logits_list, values = policy(st, action_feats)
        logits = logits_list[0]
        idx = int(torch.argmax(logits).item())
        return candidates[idx]


class GameState:
    """Manages game state for a single game session."""

    def __init__(self, game_id: str, env: Big2Env, policy: MLPPolicy, human_player: int):
        self.game_id = game_id
        self.env = env
        self.policy = policy
        self.human_player = human_player
        self.state = env.reset()
        self.last_trick_player: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert game state to JSON-serializable dictionary."""
        current_player = self.env.current_player
        is_human_turn = current_player == self.human_player

        # Format hand
        hand = sorted(self.env.hands[self.human_player])
        hand_display = [card_name(c) for c in hand]

        # Format current trick
        trick = None
        if self.env.trick_pile is not None and self.env.trick_pile.type != PASS:
            trick = format_combo(self.env.trick_pile)
            trick["player"] = self.last_trick_player if self.last_trick_player is not None else current_player

        # Get legal moves for human player
        legal_moves = []
        bot_suggestion = None
        if is_human_turn:
            candidates = self.env.legal_candidates(self.human_player)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            legal_moves = [format_combo(c) for c in candidates]

            # Calculate what the bot would play with the human's hand
            try:
                # Get the state from the human player's perspective
                human_state = self.env._obs(self.human_player)
                bot_action = ai_select_action(self.policy, human_state, candidates)
                bot_suggestion = format_combo(bot_action)
            except Exception:
                # If there's an error calculating bot suggestion, just skip it
                bot_suggestion = None

        # Opponent info
        opponents = []
        for i in range(1, self.env.n_players):
            opp_idx = (self.human_player + i) % self.env.n_players
            opponents.append(
                {
                    "id": opp_idx,
                    "name": f"Player {opp_idx}",
                    "cards_left": len(self.env.hands[opp_idx]),
                    "is_current": opp_idx == current_player,
                }
            )

        return {
            "game_id": self.game_id,
            "human_player": self.human_player,
            "current_player": current_player,
            "is_human_turn": is_human_turn,
            "hand": hand,
            "hand_display": hand_display,
            "hand_count": len(hand),
            "trick": trick,
            "legal_moves": legal_moves,
            "bot_suggestion": bot_suggestion,
            "opponents": opponents,
            "done": self.env.done,
            "winner": self.env.winner,
            "passes_in_row": self.env.passes_in_row,
        }


# Global state: in-memory game storage
games: dict[str, GameState] = {}
policy_cache: MLPPolicy | None = None
policy_config: dict[str, Any] = {}


def load_policy(model_path: str, n_players: int, device: str) -> MLPPolicy:
    """Load and cache the policy model."""
    global policy_cache, policy_config
    # Resolve model path relative to big2 directory if not absolute
    if not Path(model_path).is_absolute():
        model_path = str(BASE_DIR / model_path)

    config_key = {"model_path": model_path, "n_players": n_players, "device": device}
    if policy_cache is None or policy_config != config_key:
        policy = MLPPolicy(n_players=n_players, device=device).to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        policy.eval()
        policy_cache = policy
        policy_config = config_key
    return policy_cache


app = FastAPI(title="Big 2 Web Interface")

# Get the directory containing this file
BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"

# Serve static files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend files not found")
    return FileResponse(str(index_path))


class StartGameRequest(BaseModel):
    model_path: str = "big2_model.pt"
    n_players: int = 4
    device: str = "cpu"


@app.post("/api/game/start")
async def start_game(request: StartGameRequest) -> dict[str, Any]:
    """Start a new game."""
    try:
        # Determine device
        if request.device == "auto":
            request.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load policy
        policy = load_policy(request.model_path, request.n_players, request.device)

        # Create environment
        env = Big2Env(request.n_players)

        # Human player is the one who starts (has 3♦)
        human_player = env.current_player

        # Create game state
        game_id = str(uuid.uuid4())
        game_state = GameState(game_id, env, policy, human_player)
        games[game_id] = game_state

        return game_state.to_dict()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Model file '{request.model_path}' not found"
        ) from FileNotFoundError
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting game: {str(e)}") from e


@app.get("/api/game/{game_id}/state")
async def get_game_state(game_id: str) -> dict[str, Any]:
    """Get current game state."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    return games[game_id].to_dict()


class ActionRequest(BaseModel):
    move_index: int | None = None
    cards: list[int] | None = None


@app.post("/api/game/{game_id}/action")
async def submit_action(game_id: str, action: ActionRequest) -> dict[str, Any]:
    """Submit an action and process AI turns."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game_state = games[game_id]

    if game_state.env.done:
        raise HTTPException(status_code=400, detail="Game is already finished")

    if not game_state.env.current_player == game_state.human_player:
        raise HTTPException(status_code=400, detail="Not your turn")

    # Get legal moves
    candidates = game_state.env.legal_candidates(game_state.human_player)
    if not candidates:
        candidates = [Combo(PASS, [], ())]

    # Determine which action to take
    if action.move_index is not None:
        if action.move_index < 0 or action.move_index >= len(candidates):
            raise HTTPException(status_code=400, detail="Invalid move index")
        selected_action = candidates[action.move_index]
    elif action.cards is not None:
        # Find matching combo from cards
        selected_action = None
        for combo in candidates:
            if set(combo.cards) == set(action.cards):
                selected_action = combo
                break
        if selected_action is None:
            raise HTTPException(status_code=400, detail="Invalid card selection - not a legal move")
    else:
        raise HTTPException(status_code=400, detail="Must provide either move_index or cards")

    # Execute human action
    if selected_action.type != PASS:
        game_state.last_trick_player = game_state.human_player
    game_state.state, done = game_state.env.step(selected_action)

    # Process AI turns until human turn or game over
    ai_actions = []
    while not game_state.env.done and game_state.env.current_player != game_state.human_player:
        current_ai_player = game_state.env.current_player
        ai_candidates = game_state.env.legal_candidates(current_ai_player)
        if not ai_candidates:
            ai_candidates = [Combo(PASS, [], ())]

        ai_action = ai_select_action(game_state.policy, game_state.state, ai_candidates)
        if ai_action.type != PASS:
            game_state.last_trick_player = current_ai_player

        ai_actions.append(
            {
                "player": current_ai_player,
                "action": format_combo(ai_action),
            }
        )

        game_state.state, done = game_state.env.step(ai_action)

        # Check if trick was cleared
        if game_state.env.trick_pile is None or game_state.env.trick_pile.type == PASS:
            game_state.last_trick_player = None

    return {
        **game_state.to_dict(),
        "human_action": format_combo(selected_action),
        "ai_actions": ai_actions,
    }


@app.get("/api/game/{game_id}/status")
async def get_game_status(game_id: str) -> dict[str, Any]:
    """Get game status (done, winner)."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game_state = games[game_id]
    return {
        "done": game_state.env.done,
        "winner": game_state.env.winner,
        "is_human_winner": game_state.env.winner == game_state.human_player if game_state.env.done else None,
    }
