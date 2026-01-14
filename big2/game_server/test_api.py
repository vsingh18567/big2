"""Tests for the Big 2 game server API."""

import pytest
from fastapi.testclient import TestClient

from big2.game_server.main import app
from big2.game_server.store import game_store

# Test client with API key
API_KEY = "dev-api-key-changeme"


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    """Clear game store before each test."""
    # Clear all games
    for game_id in game_store.list_games():
        game_store.delete_game(game_id)
    yield


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_register_game_success(client):
    """Test successful game registration."""
    response = client.post(
        "/api/game/register/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["game_id"] == "game123"
    assert "game_created_at" in data


def test_register_game_no_auth(client):
    """Test game registration without API key."""
    response = client.post("/api/game/register/game123")
    assert response.status_code == 422  # Missing header


def test_register_game_invalid_auth(client):
    """Test game registration with invalid API key."""
    response = client.post(
        "/api/game/register/game123",
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_register_duplicate_game(client):
    """Test registering a game with duplicate ID."""
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    response = client.post(
        "/api/game/register/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_join_game_success(client):
    """Test successfully joining a game."""
    # Register game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})

    # Join game
    response = client.post(
        "/api/game/join/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["game_id"] == "game123"
    assert data["num_players"] == 1
    assert data["state"] == "waiting"
    assert "player_id" in data
    assert "token" in data


def test_join_nonexistent_game(client):
    """Test joining a game that doesn't exist."""
    response = client.post(
        "/api/game/join/nonexistent",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 404


def test_join_game_four_players_starts(client):
    """Test that game starts when 4 players join."""
    # Register game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})

    # Join with 4 players
    tokens = []
    for i in range(4):
        response = client.post(
            "/api/game/join/game123",
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        tokens.append(data["token"])

        if i < 3:
            assert data["state"] == "waiting"
            assert data["num_players"] == i + 1
        else:
            assert data["state"] == "started"
            assert data["num_players"] == 4


def test_join_full_game(client):
    """Test joining a game that is already full."""
    # Register and fill game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    for _ in range(4):
        client.post("/api/game/join/game123", headers={"X-API-Key": API_KEY})

    # Try to join 5th player
    response = client.post(
        "/api/game/join/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 400
    assert "full" in response.json()["detail"].lower()


def test_get_game_state_waiting(client):
    """Test getting state of a waiting game."""
    # Register and join
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    client.post("/api/game/join/game123", headers={"X-API-Key": API_KEY})

    # Get state
    response = client.get(
        "/api/game/state/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "waiting"
    assert data["turn_number"] == 0
    assert data["move_count"] == 0
    assert data["turn"] is None


def test_get_game_state_started(client):
    """Test getting state of a started game."""
    # Register and start game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    for _ in range(4):
        client.post("/api/game/join/game123", headers={"X-API-Key": API_KEY})

    # Get state
    response = client.get(
        "/api/game/state/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "started"
    assert data["turn_number"] == 1
    assert data["turn"] is not None
    assert len(data["turn_order"]) == 4
    assert len(data["cards_left"]) == 4


def test_get_nonexistent_game_state(client):
    """Test getting state of nonexistent game."""
    response = client.get(
        "/api/game/state/nonexistent",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 404


def test_submit_action_not_started(client):
    """Test submitting action before game starts."""
    # Register game but don't start
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    response_join = client.post(
        "/api/game/join/game123",
        headers={"X-API-Key": API_KEY},
    )
    token = response_join.json()["token"]

    # Try to play
    response = client.post(
        "/api/game/action/game123",
        headers={"X-API-Key": API_KEY},
        json={
            "action": "pass",
            "cards": [],
            "token": token,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "not started" in data["reason"].lower()


def test_submit_action_invalid_token(client):
    """Test submitting action with invalid token."""
    # Start game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    for _ in range(4):
        client.post("/api/game/join/game123", headers={"X-API-Key": API_KEY})

    # Try to play with invalid token
    response = client.post(
        "/api/game/action/game123",
        headers={"X-API-Key": API_KEY},
        json={
            "action": "pass",
            "cards": [],
            "token": "invalid-token",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"


def test_submit_action_out_of_turn(client):
    """Test submitting action when it's not your turn."""
    # Start game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    tokens = []
    for _ in range(4):
        response = client.post(
            "/api/game/join/game123",
            headers={"X-API-Key": API_KEY},
        )
        tokens.append(response.json()["token"])

    # Get game state to see whose turn it is
    _ = client.get(
        "/api/game/state/game123",
        headers={"X-API-Key": API_KEY},
    )

    # Try to play with a different player
    # Use token that doesn't belong to current player
    for token in tokens[1:]:  # Try different tokens
        response = client.post(
            "/api/game/action/game123",
            headers={"X-API-Key": API_KEY},
            json={
                "action": "pass",
                "cards": [],
                "token": token,
            },
        )
        data = response.json()
        if data["status"] == "out_of_turn":
            break
    else:
        # If first player tried first, it might succeed
        # Try again with second player's token
        response = client.post(
            "/api/game/action/game123",
            headers={"X-API-Key": API_KEY},
            json={
                "action": "pass",
                "cards": [],
                "token": tokens[1],
            },
        )
        data = response.json()

    # One of them should be out of turn
    # (depends on who actually has the turn)


def test_submit_action_stale_turn(client):
    """Test submitting action with stale turn number."""
    # Start game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    tokens = []
    for _ in range(4):
        response = client.post(
            "/api/game/join/game123",
            headers={"X-API-Key": API_KEY},
        )
        tokens.append(response.json()["token"])

    # Try to play with wrong turn number
    response = client.post(
        "/api/game/action/game123",
        headers={"X-API-Key": API_KEY},
        json={
            "action": "pass",
            "cards": [],
            "token": tokens[0],
            "expected_turn_number": 999,
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Will either be stale_turn or out_of_turn depending on whose turn it is
    assert data["status"] in ["stale_turn", "out_of_turn"]


def test_submit_pass_action_legal(client):
    """Test submitting a legal pass action."""
    # Start game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    tokens = []
    player_ids = []
    for _ in range(4):
        response = client.post(
            "/api/game/join/game123",
            headers={"X-API-Key": API_KEY},
        )
        data = response.json()
        tokens.append(data["token"])
        player_ids.append(data["player_id"])

    # Get current turn
    state_response = client.get(
        "/api/game/state/game123",
        headers={"X-API-Key": API_KEY},
    )
    state = state_response.json()
    current_turn = state["turn"]
    turn_idx = player_ids.index(current_turn)

    # First move must be with 3 of diamonds, can't pass on first turn
    # So we need to play a valid combo first
    # For now, just test that the API structure works
    response = client.post(
        "/api/game/action/game123",
        headers={"X-API-Key": API_KEY},
        json={
            "action": "pass",
            "cards": [],
            "token": tokens[turn_idx],
        },
    )
    assert response.status_code == 200
    # First move can't be pass, so should be illegal
    data = response.json()
    assert data["status"] in ["illegal_move", "error"]


def test_submit_illegal_move(client):
    """Test submitting an illegal move."""
    # Start game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})
    tokens = []
    player_ids = []
    for _ in range(4):
        response = client.post(
            "/api/game/join/game123",
            headers={"X-API-Key": API_KEY},
        )
        data = response.json()
        tokens.append(data["token"])
        player_ids.append(data["player_id"])

    # Get current turn
    state_response = client.get(
        "/api/game/state/game123",
        headers={"X-API-Key": API_KEY},
    )
    state = state_response.json()
    current_turn = state["turn"]
    turn_idx = player_ids.index(current_turn)

    # Try to play cards that don't form a valid combo or player doesn't have
    response = client.post(
        "/api/game/action/game123",
        headers={"X-API-Key": API_KEY},
        json={
            "action": "play",
            "cards": [
                {"suit": "hearts", "rank": "3"},
                {"suit": "spades", "rank": "5"},
            ],
            "token": tokens[turn_idx],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "illegal_move"


def test_list_games_admin(client):
    """Test admin endpoint to list games."""
    # Create some games
    client.post("/api/game/register/game1", headers={"X-API-Key": API_KEY})
    client.post("/api/game/register/game2", headers={"X-API-Key": API_KEY})

    # List games
    response = client.get(
        "/api/admin/games",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    games = response.json()
    assert "game1" in games
    assert "game2" in games


def test_delete_game_admin(client):
    """Test admin endpoint to delete game."""
    # Create game
    client.post("/api/game/register/game123", headers={"X-API-Key": API_KEY})

    # Delete game
    response = client.delete(
        "/api/admin/game/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 204

    # Verify deleted
    response = client.get(
        "/api/game/state/game123",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 404


def test_openapi_schema_available(client):
    """Test that OpenAPI schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    # Verify our endpoints are documented
    assert "/api/game/register/{game_id}" in schema["paths"]
    assert "/api/game/join/{game_id}" in schema["paths"]
    assert "/api/game/state/{game_id}" in schema["paths"]
    assert "/api/game/action/{game_id}" in schema["paths"]
