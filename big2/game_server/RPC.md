# Big 2 Game Server Protocol
- HTTP API using JSON for request and response

## API Endpoints

### Register game
Request:
```
POST /api/game/register/{game_id}
```
Response:
```
{
    "game_id": "1234567890",
    "game_created_at": "2021-01-01T00:00:00Z"
}
```

### Join game
Request:
```
POST /api/game/join/{game_id}
```
Response:
```
{
    "game_id": "1234567890",
    "game_created_at": "2021-01-01T00:00:00Z",
    "player_id": "1234567890"
    "num_players": 1 | 2 | 3 | 4
    "token": "1234567890" // secret token for auth
    "state": "waiting" | "started",
}
```

### Get game state
Request:
```
GET /api/game/state/{game_id}
```
Response:
```
{
    "state": "waiting" | "started" | "finished" | "error",
    "turn_number": 12, // monotonically increasing turn sequence
    "turn": "1234567890", // current player id
    "turn_order": [
        "1234567890",
        "4567890123",
        "7890123456",
        "9012345678"
    ], // fixed when game starts (after 4 join)
    "last_move_id": "bcaa9f25-a8ef-4db1-a833-7f8dba0c9f3d", // server-generated
    "last_move_at": "2021-01-01T00:05:02Z",
    "move_count": 18, // total applied moves for this game
    "cards_left": {
        "1234": 10,
        "5678": 10,
        "9012": 10,
        "3456": 10,
    }
    "game_history" : [
        {
            "player": "1234567890", // player id
            "cards": [
                { "suit": "hearts", "rank": 10 },
                { "suit": "diamonds", "rank": 10 },
                { "suit": "clubs", "rank": 10 },
                { "suit": "spades", "rank": 10 },
                { "suit": "spades", "rank": "J" } // 4 of a kind
            ]
        },
        {
            "player": "1234567890", // player id
            "cards": [] // empty array = pass
        },
    ],
    "winner": "1234567890"
}
```

### Submit action
Request:
```
POST /api/game/action/{game_id}
{
    "action": "play" | "pass"
    "cards": [
        { "suit": "hearts", "rank": 10 },
        { "suit": "diamonds", "rank": 10 },
        { "suit": "clubs", "rank": 10 },
        { "suit": "spades", "rank": 10 },
        { "suit": "spades", "rank": "J" } // 4 of a kind
    ],
    "expected_turn_number": 12, // optional: guard against stale/out-of-order
    "token": "1234567890" // secret token for auth
}
```
Response:
```
{
    "status": "success" | "error" | "illegal_move" | "out_of_turn" | "stale_turn" | "game_over",
    "applied_move_id": "bcaa9f25-a8ef-4db1-a833-7f8dba0c9f3d",
    "turn_number": 13, // if applied, next expected turn number
    "move_count": 19,
    "reason": "expected turn_number 12 but got 11" // present on non-success
}
```

