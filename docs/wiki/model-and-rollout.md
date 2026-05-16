# Model and Rollout

This page documents the Python model, rollout collection, and PPO update for
the Rust-backed PPO subsystem. Rust provides observations, legal candidates,
move IDs, move features, vectorized stepping, and terminal rewards. Python
should only choose among Rust-provided candidates and optimize the Torch model.

## Actor-Critic Model

`RustCandidateActorCritic` scores the legal candidate slots for each Rust batch
row. It combines:

- `obs`: Rust observation tensor, shape `[batch, obs_dim]`;
- `candidate_ids`: padded global move IDs, shape `[batch, max_candidates]`;
- `candidate_mask`: valid slot mask, shape `[batch, max_candidates]`;
- `move_features`: dense Rust-exported features, shape
  `[num_actions, move_feature_dim]`.

Forward outputs:

| Output | Shape | Meaning |
| --- | --- | --- |
| `logits` | `[batch, max_candidates]` | Candidate slot scores. Invalid slots are set near `-inf`. |
| `values` | `[batch]` | State value estimates for PPO. |

The model architecture is intentionally candidate-native:

- encode `obs` with an MLP and layer norms;
- embed each global move ID;
- encode Rust move features for each candidate;
- project ID and feature embeddings into action space;
- score each candidate by matching projected state and projected action;
- mask invalid candidate slots before constructing a categorical policy;
- estimate values from the encoded state.

`act()` samples or greedily selects a candidate slot, then gathers the
corresponding Rust move ID. `evaluate_actions()` recomputes log probabilities,
values, and entropy for stored slots during PPO.

## Rollout Collection

`collect_rollout()` runs a `RustVecEnvAdapter` for a fixed number of vectorized
steps. For each non-terminal environment row with any valid candidates, it
samples a controller from the opponent mix:

| Controller | Behavior |
| --- | --- |
| `learner` | Batch rows together and call the current policy once. PPO records are stored. |
| `checkpoint` | Sample from a prior checkpoint policy. No PPO record is stored. |
| `random` | Choose a random valid candidate slot. |
| `greedy` | Prefer the lowest metadata-ranked non-pass move. |
| `smart` | Use simple metadata and observation heuristics for finishing, passing on strong active hands, and conserving high cards. |

Only learner-controlled turns become `RustRolloutRecord`s. Each record stores
the environment index, acting player, observation, candidates, selected slot,
selected move ID, old log probability, value estimate, reward, and done flag.

After actions are chosen, Python sends move IDs to `env.step()`. If a learner
record's environment terminates, Python adds that player's Rust terminal reward
from `final_rewards[env_idx, player]`. Done environments are reset with
`reset_done(done_indices)`, preserving the vectorized loop.

The rollout buffer groups records by `(env_index, player)`. That gives PPO a
per-seat trajectory ordering for advantage computation without Python owning
game state.

## PPO Update

`ppo_update()` converts the rollout buffer into batched tensors, computes
advantages and returns per `(env_index, player)` trajectory, then runs clipped
PPO minibatches.

GAE uses:

- `gamma` for reward discounting;
- `lam` for generalized advantage smoothing;
- terminal `done` flags to stop bootstrapping;
- stored value estimates from collection time.

The update then normalizes advantages, shuffles samples each epoch, and
optimizes:

```text
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

where `policy_loss` is the clipped PPO surrogate, `value_loss` is MSE against
returns, and gradients are clipped by `max_grad_norm`.

The returned `PPOUpdateStats` reports average policy loss, value loss, entropy,
total loss, and sample count. Empty buffers are valid, which can happen with a
heuristic-only opponent mix; in that case all losses and samples are zero.

## Opponent Mix

`OpponentMixConfig` weights are sampled independently for each active vectorized
environment row. A checkpoint controller falls back to learner if no checkpoint
policies are supplied. The current training CLI wires learner/random/greedy/smart
weights; checkpoint policies are supported by rollout code but not yet loaded by
the CLI.

Heuristic opponents are deliberately small and metadata-based. They should stay
cheap enough for Python orchestration. Expensive card enumeration, legality, and
state transitions should remain Rust responsibilities.

## Tests

`big2/training/rust_ppo/tests/test_rust_ppo.py` covers the Python subsystem's
main contracts:

- adapter tensor shapes and metadata availability;
- invalid candidate masking in model logits;
- legal action selection by candidate slot and move ID;
- valid greedy/smart heuristic slots;
- rollout plus PPO update smoke behavior;
- batched learner policy calls during collection;
- heuristic-only rollout mixes producing no PPO samples;
- fixed-opponent eval result accounting;
- checkpoint save/load round trip.
