# Python Rust PPO

This page documents the Python PPO layer in `big2/training/rust_ppo/`.
Python should stay light: it adapts Rust batches into tensors, runs the Torch
policy/value model, performs PPO updates, and writes operational artifacts.
Rust owns game compute: shuffle/reset, legal move generation, observations,
candidate lists, move metadata/features, vectorized stepping, and terminal
rewards.

## Package Map

- `env_adapter.py`: thin PyO3 batch to Torch tensor adapter.
- `metadata.py`: structured Python view of Rust move metadata and feature rows.
- `model.py`: candidate-scoring actor-critic over Rust observations and global
  move IDs.
- `rollout.py`: vectorized collection of learner-controlled PPO records.
- `update.py`: GAE and clipped PPO optimization.
- `opponents.py`: lightweight metadata-based random, greedy, and smart choices.
- `evaluate.py`: fixed-seat evaluation against random, greedy, or smart
  opponents.
- `checkpoints.py`: checkpoint save/load/latest helpers.
- `run.py`: smoke and training CLI.
- `config.py`: dataclass configuration for training, PPO, opponents, evals, and
  artifacts.

## Adapter

`RustVecEnvAdapter` wraps `big2_rust.Big2VecEnv`. It does not implement Big 2
rules in Python. Its job is to:

- construct the Rust vectorized environment with explicit rule flags;
- expose `reset()`, `reset_done(env_indices)`, and `step(action_ids)`;
- convert the Rust 10-field batch tuple into a `RustBatch`;
- move batch tensors to the configured Torch device;
- build a `MoveMetadataTable` from Rust metadata and move features.

`RustBatch` contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `num_envs` | scalar | Number of parallel Rust games. |
| `obs_dim` | scalar | Observation width emitted by Rust. |
| `max_candidates` | scalar | Fixed padded candidate width. |
| `obs` | `[num_envs, obs_dim]` | Policy observation for the current player. |
| `candidate_ids` | `[num_envs, max_candidates]` | Padded global Rust move IDs. |
| `candidate_mask` | `[num_envs, max_candidates]` | Valid candidate slots. |
| `current_player` | `[num_envs]` | Seat to act in each environment. |
| `done` | `[num_envs]` | Terminal flags after a step. |
| `final_rewards` | `[num_envs, 4]` | Terminal seat rewards from Rust. |
| `truncated_candidate_lists` | scalar | Count of Rust candidate lists clipped to `max_candidates`. |

Python sends only global move IDs back into `env.step()`. If Python starts
deriving legal cards, comparing move legality, or reconstructing observations,
that work belongs in Rust instead.

## Metadata Table

`MoveMetadataTable` validates and stores Rust's global action catalog metadata:

| Field | Meaning |
| --- | --- |
| `move_id` | Stable global Rust action ID. IDs must be sorted from `0`. |
| `mask` | Card bitmask for the move. Pass is mask `0`. |
| `kind` | Move category. `0` is pass. |
| `num_cards` | Number of cards in the move. |
| `primary_rank` | Main comparison rank from Rust metadata. |
| `secondary_rank` | Tie-break/supporting rank when relevant. |
| `high_suit` | Suit tie-break value. |
| `ranks_desc` | Descending rank summary for five-card hands. |

The table also stores `features_np`, a dense per-move feature matrix exported by
Rust. The current feature width is checked against `MOVE_FEATURE_DIM`, and
`as_tensor(device)` provides the model's immutable move feature buffer.

## Config and Entrypoint

`RustPPOConfig` groups environment size, model dimensions, PPO hyperparameters,
opponent mix, device, checkpoint cadence, metrics path, and eval cadence.
`OpponentMixConfig.normalized()` turns learner/random/greedy/smart/checkpoint
weights into sampling probabilities.

`run.py` has two paths:

- smoke mode: one rollout and one PPO update, then prints sample count/loss;
- training mode: repeated rollout/update batches, optional resume from latest
  checkpoint, JSONL metrics, periodic evals, and periodic checkpoints.

## Eval, Checkpoints, and Metrics

`evaluate_policy()` controls one fixed `policy_seat` with greedy policy actions
and controls the other seats with a fixed opponent: `random`, `greedy`, or
`smart`. It returns games, wins, win rate, and average terminal reward.

Checkpoints are Torch files named `batch_000003.pt` style and contain:

- batch number;
- model state dict;
- optimizer state dict;
- dataclass config as a dict;
- metrics attached to that batch.

Training metrics are appended as JSON Lines. Each row includes batch, samples,
policy loss, value loss, entropy, and total loss. Eval rows add per-opponent
results. Checkpoint rows add `checkpoint_path`.
