# Rust Python API

The Python module is built from `big2-rust` with PyO3/maturin. Rust owns game
simulation, legal action generation, observations, terminal rewards, move
metadata, and move feature export. Python should treat returned action IDs as
opaque catalog IDs.

## Build

The Python binding is behind the `python` Cargo feature:

```sh
cd big2-rust
maturin develop --features python
```

The module name is `big2_rust`.

## Module Surface

Exports:

```python
from big2_rust import Big2VecEnv, OBS_DIM, DEFAULT_MAX_CANDIDATES
```

`OBS_DIM` is currently `135`. `DEFAULT_MAX_CANDIDATES` is currently `2048`.

`Big2VecEnv` constructor:

```python
Big2VecEnv(
    num_envs: int,
    seed: int,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    require_three_diamond_open: bool = True,
    allow_wheel_straight: bool = False,
    allow_two_in_straight: bool = False,
    game_ends_on_first_out: bool = True,
    passed_player_may_reenter: bool = False,
)
```

Properties:

```text
num_actions      number of moves in the Rust action catalog
num_envs         batch size
max_candidates   candidate row width
```

Methods:

```text
reset() -> batch
reset_done(env_indices: list[int]) -> batch
step(action_ids: list[int]) -> batch
move_metadata() -> list[tuple]
move_features() -> tuple[int, list[float]]
```

`step` requires exactly one action ID per environment. Done environments are
skipped internally, but their slots still need placeholder action IDs so the
input length equals `num_envs`. Rust raises `ValueError` for wrong action count,
unknown move IDs, or illegal moves.

## Batch Tuple

`reset`, `reset_done`, and `step` return the same tuple shape:

```text
(
    num_envs,
    obs_dim,
    max_candidates,
    observations,
    candidate_ids,
    candidate_mask,
    current_player,
    done,
    final_rewards,
    truncated_candidate_lists,
)
```

Field shapes:

```text
observations:      [num_envs, obs_dim] flattened row-major f32
candidate_ids:     [num_envs, max_candidates] flattened row-major i32
candidate_mask:    [num_envs, max_candidates] flattened row-major bool
current_player:    [num_envs] u32
done:              [num_envs] bool
final_rewards:     [num_envs, 4] nested rows of f32
```

Invalid candidate slots use ID `-1` and mask `False`. Python must use
`candidate_mask` when selecting or scoring actions. If
`truncated_candidate_lists` is nonzero during training, increase
`max_candidates`; a truncated row means legal actions were dropped.

`current_player` gives the absolute player seat for each vector slot. The
observation row is already normalized from that player's perspective. Rewards
remain absolute-player rows.

## Reset Contract

`reset()` resets every slot and returns a full fresh batch.

`reset_done(env_indices)` maps to Rust `reset_indices`. It resets only the
specified vector slots and then returns a full batch snapshot for all slots.
Indexes outside the vector are ignored by Rust. Callers normally pass the
indexes where `done` was true after consuming terminal rewards.

## Terminal Rewards

Non-terminal rows are all zeros. Terminal rows use the current first-out rule:

```text
winner: +1.0
others: -cards_remaining / 13.0
```

The returned `final_rewards` shape is `[num_envs][4]` in absolute player order,
not perspective order. A rollout collector that stores acting-seat returns
should join this with its recorded player IDs or the latest `current_player`
values as appropriate.

## Move Metadata

`move_metadata()` returns one tuple per catalog action:

```text
(
    id,
    mask,
    kind,
    num_cards,
    primary_rank,
    secondary_rank,
    high_suit,
    ranks_desc,
)
```

Types:

```text
id:              u32 catalog MoveId
mask:            u64 CardMask
kind:            u8 MoveKind enum value
num_cards:       u8
primary_rank:    u8, rank index 0..12
secondary_rank:  u8, rank index 0..12
high_suit:       u8, suit index 0..3
ranks_desc:      list[u8], length 5
```

`kind` values:

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

Ranks use `0 = 3` through `12 = 2`. Suits use `0 = D`, `1 = C`, `2 = H`,
`3 = S`. `Pass` has `mask = 0` and `num_cards = 0`.

## Move Features

`move_features()` returns:

```text
(feature_dim, features)
```

`feature_dim` is currently `97`. `features` is flattened row-major with shape
`[num_actions, feature_dim]`.

Per-move feature layout:

```text
card mask bits:         52 bits
move kind:               9 one-hot bits
num cards:               1 scalar, / 5
primary rank:           13 one-hot bits
secondary rank:         13 one-hot bits
high suit:               4 one-hot bits
ranks_desc:              5 scalars, each rank / 12, zero for empty slots
```

The feature exporter reflects raw `MoveMeta` values. `Pass` therefore has
rank-0 primary and secondary one-hot bits because its metadata ranks are `0`.
In `ranks_desc`, rank `0` (`3`) also encodes as `0.0`, so consumers should use
the mask/kind/num-card features when they need to distinguish actual low ranks
from empty padding.

These features are catalog-level constants. Python can compute them once per
environment configuration and gather rows by candidate/action ID instead of
rebuilding move encodings during rollout.
