# Big 2 Environment State

This document explains the core Rust environment concepts. For the current
system wiki, start at [docs/wiki/README.md](wiki/README.md).

The current Rust crate lives in `big2-rust/`.

## Card IDs

The deck has 52 cards. Each card has a stable integer ID from `0` to `51`.

Ranks are ordered from low to high:

```text
3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A, 2
```

Suits are ordered:

```text
D, C, H, S
```

So the first few card IDs are:

```text
0 = 3D
1 = 3C
2 = 3H
3 = 3S
4 = 4D
...
48 = 2D
51 = 2S
```

This matches the Python simulator's existing card ordering.

## CardMask

`CardMask` is the Rust type for a set of cards:

```rust
pub type CardMask = u64;
```

A `u64` has 64 bits. Big 2 only needs 52 cards, so each card gets one bit:

```text
bit 0  = 3D
bit 1  = 3C
...
bit 51 = 2S
```

This means a hand, a move, or the set of already-played cards can all be stored
as one integer.

Common operations are cheap:

```rust
// Does this hand contain every card in this move?
action.mask & hand == action.mask

// Remove played cards from a hand.
hand &= !action.mask

// Add played cards to the public played pile.
cards_played |= action.mask

// Count cards in a hand.
hand.count_ones()
```

## Hands

A hand is the set of cards one player currently holds.

In the Rust state:

```rust
hands: [CardMask; 4]
```

Each player gets one `CardMask`.

Example:

```text
Player has 3D, 3C, 4D
```

That hand is:

```rust
card_bit(0) | card_bit(1) | card_bit(4)
```

## Moves

A move is what a player chooses to play on their turn.

Examples:

```text
Pass
Single 3D
Pair [3D 3C]
Triple [QD QC QH]
Straight [3D 4C 5D 6H 7S]
Flush [...]
Full house [...]
Four of a kind [...]
Straight flush [...]
```

The Rust representation is `MoveMeta`:

```rust
pub struct MoveMeta {
    pub id: MoveId,
    pub mask: CardMask,
    pub kind: MoveKind,
    pub num_cards: u8,
    pub primary_rank: u8,
    pub secondary_rank: u8,
    pub high_suit: u8,
    pub ranks_desc: [u8; 5],
}
```

The `mask` says which cards the move uses. `Pass` has mask `0`.

The metadata exists so we can compare moves without re-parsing the cards every
time.

## Action Catalog

The action catalog precomputes every possible move once:

```rust
ActionCatalog
```

It stores:

```text
MoveId -> MoveMeta
CardMask -> MoveId
five-card CardMask -> MoveId
```

This gives us stable global action IDs.

At runtime, the environment does not invent new move objects. It returns legal
`MoveId`s from the catalog.

Example:

```text
candidate_ids = [0, 17, 91, 144]
```

where `0` is always `Pass`.

## GameState

`GameState` is the complete internal game truth:

```rust
pub struct GameState {
    pub hands: [CardMask; 4],
    pub current_player: u8,

    pub last_move_id: Option<MoveId>,
    pub last_actor: Option<u8>,

    pub passed: [bool; 4],
    pub cards_played: CardMask,
    pub cards_remaining: [u8; 4],
    pub finished: [bool; 4],

    pub is_first_turn: bool,
    pub done: bool,
    pub rng_state: u64,
}
```

Important distinction: `GameState` contains hidden information, including all
players' hands. The neural net must not receive this whole structure.

The policy only receives an observation built from:

```text
own hand
public played cards
public trick state
public counts/flags
```

## RulesConfig

Rule choices are explicit in `RulesConfig`:

```rust
pub struct RulesConfig {
    pub require_three_diamond_open: bool,
    pub allow_wheel_straight: bool,
    pub allow_two_in_straight: bool,
    pub game_ends_on_first_out: bool,
    pub passed_player_may_reenter: bool,
}
```

Defaults match the current Python-style behavior:

```text
opening move must include 3D
wheel straights disabled
2s in straights disabled
game ends when first player goes out
passed players cannot re-enter the active trick
```

`RulesConfig` affects both catalog construction and legal move generation.

For example, allowing `2` in straights changes which five-card masks become
valid moves in the catalog.

## Legal Move Generation

Legal move generation takes:

```text
GameState
ActionCatalog
RulesConfig
```

and returns:

```rust
Vec<MoveId>
```

The basic flow is:

1. If the game is done, return no moves.
2. If the player already passed in an active trick and re-entry is disabled,
   return only `Pass`.
3. If there is no active previous move, this is a free lead:
   - `Pass` is illegal.
   - any move contained in the player's hand is legal.
   - on first turn, require `3D` if configured.
4. If there is an active previous move:
   - `Pass` is legal.
   - non-pass moves must be contained in hand.
   - non-pass moves must beat the previous move.

The important containment check is:

```rust
action.mask & hand == action.mask
```

## Vectorized Environment

`Big2VecEnv` wraps many independent `Big2Env` games behind one batched reset and
step interface.

The batch output is intentionally rectangular:

```text
observations:   [num_envs, OBS_DIM]
candidate_ids:  [num_envs, max_candidates]
candidate_mask: [num_envs, max_candidates]
current_player: [num_envs]
done:           [num_envs]
final_rewards:  [num_envs, 4]
```

`candidate_ids` uses `-1` in padded slots, and `candidate_mask` marks those
slots as `false`. The policy should score only slots where the mask is `true`.

If an environment has more legal moves than `max_candidates`, the row is
truncated and `truncated_candidate_lists` is incremented. This should never be
ignored in training; increase `max_candidates` if it happens.

`current_player` is the absolute seat that owns each row's turn. The
observation row is already normalized to that player's perspective.

`step` accepts one global `MoveId` per environment. Done environments are left
unchanged so a rollout collector can read terminal rewards before deciding when
to reset or replace that slot. `reset_done` resets only selected vector slots and
then returns a full batch snapshot.

Current terminal rewards are:

```text
first player out: +1.0
other players:   -cards_remaining / 13.0
non-terminal:     0.0 for all players
```

## Step

`step(action_id)` validates the action against legal moves before mutating state.

For a non-pass move:

```text
remove move cards from current player's hand
add move cards to cards_played
decrease cards_remaining
set last_move_id
set last_actor
clear passed flags
mark first turn complete
possibly mark player finished/done
advance turn unless game is done
```

For `Pass`:

```text
mark current player as passed
if all other players passed, clear active trick
advance turn
```

When three players pass after a move, the turn advances back to the last actor
with a free lead.

## Observation

The neural net does not see `GameState` directly.

It sees an `Observation`:

```rust
pub struct Observation {
    pub features: Vec<f32>,
}
```

Current observation dimension:

```text
OBS_DIM = 135
```

Current feature layout:

```text
own hand:                  52 bits
cards played:              52 bits
last move kind:             9 one-hot bits
last move num cards:        1 scalar
last move primary rank:     1 scalar
last move secondary rank:   1 scalar
last move high suit:        4 one-hot bits
cards remaining:            4 scalars, perspective-normalized
passed flags:               4 bits, perspective-normalized
free lead flag:             1 bit
first turn flag:            1 bit
last actor relative:        5 one-hot bits
```

The `last actor relative` field has 5 slots:

```text
none
self
left
across
right
```

## Perspective Normalization

The same neural network should be able to play all seats.

So observations are normalized from the acting player's perspective:

```text
relative player 0 = self
relative player 1 = next player clockwise
relative player 2 = across
relative player 3 = previous player clockwise
```

For example, if absolute player `2` is acting, card counts are encoded as:

```text
[player 2, player 3, player 0, player 1]
```

This lets the model learn one policy instead of four seat-specific policies.

## Hidden Information Boundary

The environment stores all hands because it must enforce rules.

The observation only encodes the acting player's own hand:

```text
own hand: yes
opponent hands: no
cards already played: yes
cards remaining counts: yes
```

This boundary is important. If opponent hands leak into observations, training
will learn from information the agent cannot legally know.

## Current Scope

Implemented now:

```text
single-game environment
action catalog
legal move generation
rules config
observation encoding
batched/vectorized environment
padded candidate arrays
terminal reward arrays
PyO3/maturin Python bindings
tests for the above
```

Not implemented yet:

```text
recent public action history
throughput benchmarks
```
