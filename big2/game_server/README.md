# Big 2 Game Server

A FastAPI-based HTTP game server for the Big 2 card game, implementing the RPC protocol defined in `RPC.md`.

## Features

- **Complete RPC Implementation**: All endpoints from RPC.md specification
- **Type-Safe API**: Pydantic models with comprehensive OpenAPI documentation
- **Move Validation**: Integration with Big 2 simulator for legal move checking
- **In-Memory State**: Thread-safe game management without persistence
- **API Key Authentication**: Simple header-based authentication
- **CORS Support**: Ready for browser-based clients

## Installation

This server is part of the `big2` package. Ensure dependencies are installed:

```bash
uv pip install fastapi uvicorn
```

## Running the Server

### Quick Start (using the provided script)

```bash
cd /Users/vikramsingh/Desktop/coding/big2
./big2/game_server/run_server.sh
```

Or with environment variables:

```bash
export BIG2_API_KEY=your-secret-key
export BIG2_PORT=8080
./big2/game_server/run_server.sh
```

### Development Mode

```bash
cd /Users/vikramsingh/Desktop/coding/big2
uvicorn big2.game_server.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn big2.game_server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

### Interactive Documentation (Auto-Generated)

Once the server is running, access:
- **Swagger UI**: http://localhost:8000/docs (interactive, test endpoints in browser)
- **ReDoc**: http://localhost:8000/redoc (clean documentation view)
- **OpenAPI Schema**: http://localhost:8000/openapi.json (raw JSON spec)

### Export OpenAPI Schema

Export the OpenAPI specification to a file:

```bash
# Export as JSON
python big2/game_server/export_openapi.py -o openapi.json

# Export as YAML (requires PyYAML)
python big2/game_server/export_openapi.py -o openapi.yaml -f yaml
```

The exported schema can be used with:
- **Code generators**: OpenAPI Generator, Swagger Codegen
- **API testing**: Postman, Insomnia, REST Client
- **Documentation sites**: Redocly, Stoplight
- **Mock servers**: Prism, Mockoon

## Authentication

All endpoints require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: dev-api-key-changeme" http://localhost:8000/api/game/register/game123
```

The default API key is `dev-api-key-changeme`. Set a custom key via environment variable:

```bash
export BIG2_API_KEY=your-secret-key
```

## API Endpoints

### Game Management

#### Register Game
```http
POST /api/game/register/{game_id}
```

Creates a new game.

#### Join Game
```http
POST /api/game/join/{game_id}
```

Joins an existing game. Returns player ID and token. Game starts when 4 players join.

### Game State

#### Get Game State
```http
GET /api/game/state/{game_id}
```

Returns complete game state including:
- Current turn and turn order
- Move history
- Cards remaining per player
- Winner (if finished)

### Game Actions

#### Submit Action
```http
POST /api/game/action/{game_id}
```

Submit a move (play cards or pass). Requires player token in request body.

Request body:
```json
{
  "action": "play" | "pass",
  "cards": [
    { "suit": "hearts", "rank": "10" }
  ],
  "expected_turn_number": 12,
  "token": "player-token"
}
```

Response includes:
- `status`: success, error, illegal_move, out_of_turn, stale_turn, game_over
- `reason`: Detailed error message if not successful
- `applied_move_id`: UUID of applied move
- `turn_number`: Next turn number

### Admin Endpoints (Debug)

#### List Games
```http
GET /api/admin/games
```

#### Delete Game
```http
DELETE /api/admin/game/{game_id}
```

## Testing

Run tests with pytest:

```bash
pytest big2/game_server/test_api.py -v
```

## Architecture

```
main.py           - FastAPI application and routes
models.py         - Pydantic models (request/response schemas)
game_logic.py     - Game state management and simulator integration
store.py          - In-memory thread-safe game store
auth.py           - API key authentication
```

### Game Flow

1. **Register**: Create game with unique ID
2. **Join**: 4 players join, game auto-starts and deals cards
3. **Play**: Players submit moves in turn order
4. **Validate**: Each move validated against Big 2 rules via simulator
5. **End**: Game finishes when a player empties their hand

### Move Validation

Moves are validated for:
- **Turn order**: Correct player's turn
- **Card ownership**: Player has the cards being played
- **Valid combination**: Cards form a legal Big 2 combo (single, pair, triple, 5-card)
- **Game rules**: Move beats previous play or is a valid pass
- **Special rules**: First move must include 3 of diamonds

## Example Usage

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "dev-api-key-changeme"}

# Create game
response = requests.post(f"{BASE_URL}/api/game/register/mygame", headers=HEADERS)
print(response.json())

# Join as player 1
response = requests.post(f"{BASE_URL}/api/game/join/mygame", headers=HEADERS)
player1 = response.json()
token1 = player1["token"]
player_id1 = player1["player_id"]

# Join 3 more players...
for _ in range(3):
    requests.post(f"{BASE_URL}/api/game/join/mygame", headers=HEADERS)

# Get game state
response = requests.get(f"{BASE_URL}/api/game/state/mygame", headers=HEADERS)
state = response.json()
print(f"Current turn: {state['turn']}")

# Submit move (if it's your turn)
if state['turn'] == player_id1:
    response = requests.post(
        f"{BASE_URL}/api/game/action/mygame",
        headers=HEADERS,
        json={
            "action": "play",
            "cards": [{"suit": "diamonds", "rank": "3"}],
            "token": token1
        }
    )
    print(response.json())
```

### cURL Examples

```bash
# Register game
curl -X POST http://localhost:8000/api/game/register/game123 \
  -H "X-API-Key: dev-api-key-changeme"

# Join game
curl -X POST http://localhost:8000/api/game/join/game123 \
  -H "X-API-Key: dev-api-key-changeme"

# Get state
curl http://localhost:8000/api/game/state/game123 \
  -H "X-API-Key: dev-api-key-changeme"

# Play cards
curl -X POST http://localhost:8000/api/game/action/game123 \
  -H "X-API-Key: dev-api-key-changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "play",
    "cards": [{"suit": "diamonds", "rank": "3"}],
    "token": "your-player-token"
  }'
```

## Type Safety

The API leverages Pydantic for comprehensive type safety:

- **Enums**: `GameState`, `ActionType`, `ActionStatus`, `Suit`, `Rank`
- **Models**: All requests/responses fully typed
- **Validation**: Automatic request validation with detailed errors
- **OpenAPI**: Auto-generated docs perfectly reflect types

This ensures:
- Invalid requests rejected before reaching game logic
- API contract is self-documenting
- Client generation from OpenAPI schema is accurate

## Limitations

- **No Persistence**: Games are in-memory only
- **No Reconnection**: If server restarts, all games are lost
- **No Spectators**: Only players in a game can view state
- **Simple Auth**: Single API key for all operations

## Future Enhancements

- Database persistence (PostgreSQL/Redis)
- Player accounts and sessions
- Spectator mode
- Game replay
- Rate limiting
- WebSocket support for real-time updates
- AI opponent integration

