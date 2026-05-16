# Rust Environment

This page documents the Rust game engine contract in `big2-rust/`: card
encoding, move IDs, rules, single-game stepping, and vectorized batches.

## Card Representation

The deck has 52 stable card IDs, `0..51`. Ranks are ordered low to high:

```text
3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A, 2
```

Suits are ordered:

```text
D, C, H, S
```

Card IDs are rank-major, then suit-major:

```text
0 = 3D
1 = 3C
2 = 3H
3 = 3S
4 = 4D
...
51 = 2S
```

Card sets use `CardMask = u64`. Bit `n` means card ID `n` is present, so hands,
moves, and public played cards are all one integer. Common checks are bitwise:

```rust
// Does this hand contain every card in this move?
action.mask & hand == action.mask

// Remove played cards from a hand.
hand &= !action.mask
```

## Action Catalog

`ActionCatalog` is a precomputed global move table. The environment returns
stable `MoveId` values from this table instead of constructing moves in Python.

The catalog contains:

- `Pass`, always `MoveId` `0`
- all singles
- all pairs
- all triples
- all valid five-card moves under the selected `RulesConfig`

Each catalog row is `MoveMeta`:

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

`MoveKind` values are ordered as:

```text
0 Pass
1 Single
2 Pair
3 Triple
4 Straight
5 Flush
6 FullHouse
7 FourOfKind
8 StraightFlush
```

For same-kind comparisons, singles, pairs, triples, straights, and straight
flushes compare `(primary_rank, high_suit)`. Flushes compare
`(ranks_desc, high_suit)`. Full houses and four-of-a-kind moves compare
`primary_rank`. Between five-card kinds, the `MoveKind` order above is the
strength order.

## Rules Config

`RulesConfig` makes rule choices explicit:

```rust
pub struct RulesConfig {
    pub require_three_diamond_open: bool,
    pub allow_wheel_straight: bool,
    pub allow_two_in_straight: bool,
    pub game_ends_on_first_out: bool,
    pub passed_player_may_reenter: bool,
}
```

Defaults are:

```text
require_three_diamond_open = true
allow_wheel_straight = false
allow_two_in_straight = false
game_ends_on_first_out = true
passed_player_may_reenter = false
```

Rules affect both catalog construction and legal move generation. For example,
straight rules determine which five-card masks are cataloged, and the opening
rule filters the first legal move list to moves containing `3D`.

## Single Environment

`Big2Env` owns one complete game state:

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

`reset(seed)` shuffles with the internal XorShift RNG, deals 13 cards to each
player, and sets `current_player` to the holder of `3D`.

`legal_move_ids(&catalog)` returns legal global `MoveId`s for
`state.current_player`:

- done games return no moves
- a player who has passed in an active trick gets only `Pass` unless re-entry is
  enabled
- on a free lead, pass is illegal and any contained move can be led
- on the first turn, opening moves must include `3D` when configured
- after a previous move, `Pass` is legal and non-pass moves must beat that move

`step(action_id, catalog)` validates the ID before mutation. Non-pass moves
remove cards from the acting hand, add them to `cards_played`, update
`last_move_id` and `last_actor`, clear pass flags, and may set `done`. Pass
moves mark the acting player as passed; when three players have passed, the
active trick is cleared.

## Observation

Policies do not receive `GameState` directly. `observe_current_player` returns a
flat `Observation` of length `OBS_DIM = 135`, normalized from the acting
player's perspective.

Feature layout:

```text
own hand:                  52 bits
cards played:              52 bits
last move kind:             9 one-hot bits
last move num cards:        1 scalar, / 5
last move primary rank:     1 scalar, / 12
last move secondary rank:   1 scalar, / 12
last move high suit:        4 one-hot bits
cards remaining:            4 scalars, relative seats, / 13
passed flags:               4 bits, relative seats
free lead flag:             1 bit
first turn flag:            1 bit
last actor relative:        5 one-hot bits
```

Relative seat `0` is self, then next clockwise, across, and previous clockwise.
The final `last actor relative` group has slots for none, self, left, across,
and right.

## Vectorized Environment

`Big2VecEnv` batches independent `Big2Env` games behind one action catalog.
`reset`, `reset_indices`, `step`, and `snapshot` return `VecEnvBatch`.

Batch fields:

```text
num_envs:                    scalar
obs_dim:                     scalar, currently 135
max_candidates:              scalar, default 2048
observations:                [num_envs, obs_dim] row-major f32
candidate_ids:               [num_envs, max_candidates] row-major i32
candidate_mask:              [num_envs, max_candidates] row-major bool
current_player:              [num_envs] u8
done:                        [num_envs] bool
final_rewards:               [num_envs, 4] f32
truncated_candidate_lists:   scalar count
```

Each active environment expects one selected catalog `MoveId` in `step`. The
action array length must equal `num_envs`; done slots are skipped and left
unchanged so rollout code can consume terminal rewards before resetting them.

Candidate rows are rectangular. Padded slots use candidate ID `-1` and mask
`false`. Policies should score only `candidate_mask == true` entries. If a legal
list exceeds `max_candidates`, the row is truncated and
`truncated_candidate_lists` is incremented; this is a configuration error for
training and should be fixed by increasing `max_candidates`.

`current_player` is part of the batch contract. It tells Python which absolute
seat each observation/action row belongs to, while the observation itself stays
perspective-normalized.

## Resetting Finished Slots

Rust exposes `reset_indices(&[usize])` to reset selected vector slots and return
a fresh full `VecEnvBatch` snapshot. Invalid indexes are ignored. The Python
wrapper exposes the same operation as `reset_done(env_indices)`.

## Terminal Rewards

Terminal rewards are zero until an environment is done. With the default
`game_ends_on_first_out = true` rule, the first player out is the winner:

```text
winner: +1.0
others: -cards_remaining / 13.0
```

`final_rewards` is shaped `[num_envs, 4]` in absolute player order. Rollout code
should read the acting/player ownership it needs from `current_player` and its
own trajectory bookkeeping rather than assuming the reward row is
perspective-normalized.
