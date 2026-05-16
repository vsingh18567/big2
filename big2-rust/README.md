# big2-rust

Rust implementation of the Big 2 rules engine and vectorized environment core.

The crate is built around a fixed global action catalog. At runtime each game
state returns only the legal `MoveId` candidates for the acting player; a policy
can score those candidates without generating card combinations in Python.

## Core Types

- `CardMask`: `u64` bitset for hands, moves, and public played cards.
- `ActionCatalog`: precomputed global catalog of pass, singles, pairs, triples,
  and valid five-card Big 2 moves.
- `Big2Env`: one complete game state with reset, legal move generation,
  observation encoding, and step validation.
- `Big2VecEnv`: a batch of independent `Big2Env` instances with rectangular
  observation and candidate buffers.

## Vectorized Batch Contract

`Big2VecEnv::reset()` and `Big2VecEnv::step(&[MoveId])` return `VecEnvBatch`:

```text
observations:   [num_envs, OBS_DIM] flattened row-major f32
candidate_ids:  [num_envs, max_candidates] flattened row-major i32
candidate_mask: [num_envs, max_candidates] flattened row-major bool
current_player: [num_envs] u8
done:           [num_envs] bool
final_rewards:  [num_envs, 4] f32
```

Invalid candidate slots use ID `-1` and mask `false`. If a legal candidate list
is longer than `max_candidates`, the row is truncated and
`truncated_candidate_lists` is incremented. That should be treated as a
configuration error during training.

Terminal rewards currently use the simple first-out rule:

```text
winner: +1.0
others: -cards_remaining / 13.0
```

## Tests

Run the Rust test suite from the repository root:

```sh
cargo test --manifest-path big2-rust/Cargo.toml
```

## Python Module

The PyO3/maturin wrapper is behind the `python` feature:

```sh
cd big2-rust
maturin develop --features python
```

The Python constructor mirrors the vectorized Rust environment:

```python
from big2_rust import Big2VecEnv

env = Big2VecEnv(num_envs=4096, seed=123)
batch = env.reset()
next_batch = env.step(action_ids)
fresh_batch = env.reset_done([0, 7, 12])
```

For now the Python batch is returned as a tuple:

```text
(num_envs, obs_dim, max_candidates, observations, candidate_ids,
 candidate_mask, current_player, done, final_rewards, truncated_candidate_lists)
```

The wrapper also exposes catalog constants for Python policy code:

```python
metadata = env.move_metadata()
feature_dim, move_features = env.move_features()
num_actions = env.num_actions
```

See `docs/wiki/README.md` for the current system wiki.
