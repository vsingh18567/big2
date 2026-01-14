# Big 2 Game Server

HTTP API for Big 2 card game with JSON request/response

# Base URL


| URL | Description |
|-----|-------------|


# APIs

## GET /

Root

Health check endpoint.




### Responses

#### 200


Successful Response








## POST /api/game/register/{game_id}

Register Game

Register a new game.

Creates a new game with the specified ID. The game will be in 'waiting' state
until 4 players join.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| game_id | string | True | Unique game identifier |
| x-api-key | string | True |  |


### Responses

#### 201


Game created successfully


[RegisterGameResponse](#registergameresponse)







#### 400


Game already exists


[ErrorResponse](#errorresponse)







#### 401


Invalid API key


[ErrorResponse](#errorresponse)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/game/join/{game_id}

Join Game

Join an existing game.

Adds a player to the game. When the 4th player joins, the game automatically
starts and cards are dealt.


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| game_id | string | True | Game ID to join |
| x-api-key | string | True |  |


### Responses

#### 200


Successfully joined game


[JoinGameResponse](#joingameresponse)







#### 400


Cannot join game


[ErrorResponse](#errorresponse)







#### 401


Invalid API key


[ErrorResponse](#errorresponse)







#### 404


Game not found


[ErrorResponse](#errorresponse)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/game/state/{game_id}

Get Game State

Get current game state.

Returns complete game state including turn information, move history,
cards remaining per player, and winner (if game finished).


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| game_id | string | True | Game ID |
| x-api-key | string | True |  |


### Responses

#### 200


Game state retrieved


[GetGameStateResponse](#getgamestateresponse)







#### 401


Invalid API key


[ErrorResponse](#errorresponse)







#### 404


Game not found


[ErrorResponse](#errorresponse)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## POST /api/game/action/{game_id}

Submit Action

Submit a game action (play cards or pass).

Validates the move according to Big 2 rules and applies it if legal.
Returns detailed status including success/failure reason.

The action is validated for:
- Correct player turn
- Valid card combination
- Legal move according to Big 2 rules
- Player owns the cards being played


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| game_id | string | True | Game ID |
| x-api-key | string | True |  |


### Request Body

[SubmitActionRequest](#submitactionrequest)







### Responses

#### 200


Action processed (check status field for result)


[SubmitActionResponse](#submitactionresponse)







#### 401


Invalid API key


[ErrorResponse](#errorresponse)







#### 404


Game not found


[ErrorResponse](#errorresponse)







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## GET /api/admin/games

List Games

List all game IDs (admin/debug endpoint).


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| x-api-key | string | True |  |


### Responses

#### 200


Successful Response


array







#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







## DELETE /api/admin/game/{game_id}

Delete Game

Delete a game (admin/debug endpoint).


### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| game_id | string | True | Game ID to delete |
| x-api-key | string | True |  |


### Responses

#### 204


Successful Response




#### 422


Validation Error


[HTTPValidationError](#httpvalidationerror)







# Components



## ActionStatus


Action response status.




## ActionType


Action types.




## Card


A playing card.


| Field | Type | Description |
|-------|------|-------------|
| suit |  | The suit of the card |
| rank |  | The rank of the card |


## ErrorResponse


Generic error response.


| Field | Type | Description |
|-------|------|-------------|
| detail | string | Error message |


## GameState


Game state enum.




## GetGameStateResponse


Response for getting game state.


| Field | Type | Description |
|-------|------|-------------|
| state |  | Current game state |
| turn_number | integer | Monotonically increasing turn sequence |
| turn |  | Current player ID (null if game not started) |
| turn_order | array | Fixed turn order (set when game starts) |
| last_move_id |  | Server-generated ID of last move |
| last_move_at |  | Timestamp of last move |
| move_count | integer | Total applied moves for this game |
| cards_left | object | Number of cards remaining per player |
| game_history | array | Complete game history |
| winner |  | Winner player ID (if game finished) |


## HTTPValidationError



| Field | Type | Description |
|-------|------|-------------|
| detail | array |  |


## JoinGameResponse


Response for joining a game.


| Field | Type | Description |
|-------|------|-------------|
| game_id | string | Unique game identifier |
| game_created_at | string | ISO 8601 timestamp of game creation |
| player_id | string | Unique player identifier |
| num_players | integer | Number of players currently in the game |
| token | string | Secret token for authentication |
| state |  | Current game state |


## MoveHistoryEntry


A single move in game history.


| Field | Type | Description |
|-------|------|-------------|
| player | string | Player ID who made the move |
| cards | array | Cards played (empty array means pass) |


## Rank


Card ranks.




## RegisterGameResponse


Response for game registration.


| Field | Type | Description |
|-------|------|-------------|
| game_id | string | Unique game identifier |
| game_created_at | string | ISO 8601 timestamp of game creation |


## SubmitActionRequest


Request to submit an action (play or pass).


| Field | Type | Description |
|-------|------|-------------|
| action |  | Action type |
| cards | array | Cards to play (empty for pass) |
| expected_turn_number |  | Optional guard against stale/out-of-order moves |
| token | string | Secret authentication token |


## SubmitActionResponse


Response to action submission.


| Field | Type | Description |
|-------|------|-------------|
| status |  | Result status |
| applied_move_id |  | Server-generated move ID (if applied) |
| turn_number |  | Next expected turn number (if applied) |
| move_count |  | Total move count after this action |
| reason |  | Error/rejection reason (present on non-success) |


## Suit


Card suits.




## ValidationError



| Field | Type | Description |
|-------|------|-------------|
| loc | array |  |
| msg | string |  |
| type | string |  |
