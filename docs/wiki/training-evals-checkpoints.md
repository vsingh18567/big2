# Training, Evals, and Checkpoints

This page covers the Rust-backed PPO CLI in
`big2/training/rust_ppo/run.py`. It assumes `big2_rust` has been installed with
`maturin develop`; see [Test Runbook](test-runbook.md).

## Starter Commands

Fast CPU smoke train with metrics and evals every batch:

```sh
uv run python -m big2.training.rust_ppo.run --train \
  --batches 2 \
  --num-envs 4 \
  --rollout-steps 8 \
  --max-candidates 256 \
  --ppo-epochs 1 \
  --mini-batch-size 32 \
  --eval-interval 1 \
  --eval-games 8 \
  --eval-num-envs 4 \
  --checkpoint-interval 1 \
  --metrics-path runs/rust_ppo/smoke_metrics.jsonl \
  --checkpoint-dir runs/rust_ppo/checkpoints
```

Small local baseline:

```sh
uv run python -m big2.training.rust_ppo.run --train \
  --batches 20 \
  --num-envs 16 \
  --rollout-steps 32 \
  --max-candidates 512 \
  --ppo-epochs 2 \
  --mini-batch-size 256 \
  --eval-interval 5 \
  --eval-games 32 \
  --eval-num-envs 8 \
  --checkpoint-interval 5 \
  --metrics-path runs/rust_ppo/local_metrics.jsonl \
  --checkpoint-dir runs/rust_ppo/checkpoints
```

Larger CPU/GPU starter:

```sh
uv run python -m big2.training.rust_ppo.run --train \
  --batches 100 \
  --num-envs 64 \
  --rollout-steps 128 \
  --max-candidates 2048 \
  --ppo-epochs 2 \
  --mini-batch-size 512 \
  --eval-interval 10 \
  --eval-games 64 \
  --eval-num-envs 16 \
  --checkpoint-interval 10 \
  --device cpu \
  --metrics-path runs/rust_ppo/train_metrics.jsonl \
  --checkpoint-dir runs/rust_ppo/checkpoints
```

Change `--device cpu` to a Torch-supported accelerator only after a smoke run
passes on that machine.

## Opponent Mix

The rollout controller samples each active environment turn from normalized
weights. Defaults are learner self-play only:

```text
learner=1.0 random=0.0 greedy=0.0 smart=0.0
```

Suggested starters:

```sh
# Learn basic legality and reward signal through self-play.
--learner-weight 1.0 --random-weight 0.0 --greedy-weight 0.0 --smart-weight 0.0

# Add weak diversity.
--learner-weight 0.8 --random-weight 0.2 --greedy-weight 0.0 --smart-weight 0.0

# Add fixed heuristic pressure once smoke metrics are stable.
--learner-weight 0.6 --random-weight 0.1 --greedy-weight 0.2 --smart-weight 0.1
```

Weights must sum to a positive value. The CLI currently exposes learner,
random, greedy, and smart weights; checkpoint opponents are present in the
config/rollout layer but are not wired as a CLI option in `run.py`.

## Outputs

Metrics are JSON Lines at `--metrics-path`:

```json
{"batch":1,"samples":32,"policy_loss":-0.01,"value_loss":0.12,"entropy":2.3,"total_loss":0.08}
```

When `--eval-interval` divides the batch number, the row includes `eval` results
for `random`, `greedy`, and `smart` opponents:

```json
"eval":{"greedy":{"opponent":"greedy","games":8,"wins":2,"win_rate":0.25,"average_reward":-0.1}}
```

When `--checkpoint-interval` divides the batch number, the row includes
`checkpoint_path`. Checkpoints are Torch files named:

```text
<checkpoint-dir>/batch_000010.pt
```

Each checkpoint stores `batch`, `model_state`, `optimizer_state`, serialized
config, and the metrics row passed to `save_checkpoint`.

Resume from the latest checkpoint in `--checkpoint-dir`:

```sh
uv run python -m big2.training.rust_ppo.run --train --resume \
  --batches 40 \
  --checkpoint-dir runs/rust_ppo/checkpoints \
  --metrics-path runs/rust_ppo/train_metrics.jsonl
```

`--batches` is the final batch index for that run. If the latest checkpoint is
`batch_000020.pt`, the resumed command above starts at batch 21 and stops at 40.

## Logging Modes

Training supports three JSONL/stdout verbosity modes:

```sh
--logging-mode minimal  # stdout prints eval rows only
--logging-mode medium   # default operational training metrics
--logging-mode max      # medium plus histograms and bucketed diagnostics
```

All modes write a config row to `--metrics-path` with the full training config,
Torch version, and git commit when available. `minimal` keeps stdout quiet except
for rows that include `eval`, which is useful for long unattended runs.

`medium` batch rows include:

- PPO health: approximate KL, clip fraction, policy ratio summary, advantage and
  return summary, value explained variance, gradient norm, and sampled action
  probability summary.
- Rollout outcomes: controller counts, learner turns, completed episodes,
  episode length summary, terminal rewards by seat, wins by seat, and pass rate.
- Candidate availability: candidate count p50/p90/p95/p99 plus truncation
  counters.
- Timing: rollout, update, eval, checkpoint, batch seconds, samples/sec, and
  env-steps/sec.

`max` additionally logs candidate-count histograms, selected move-kind counts,
and learner entropy bucketed by candidate-count range.

By default eval runs each opponent across all four policy seats. Use
`--no-eval-all-seats --eval-policy-seat 0` to restore single-seat evals when
runtime matters more than seat-balance observability.

## max_candidates and Truncation

The Rust vectorized env returns rectangular candidate buffers shaped
`[num_envs, max_candidates]`. Invalid slots are filled with move ID `-1` and
mask `false`.

If a legal move list is longer than `max_candidates`, Rust truncates that row
and increments `truncated_candidate_lists`. Treat any truncation during training
as a configuration error because the policy cannot score every legal move.

Troubleshooting steps:

- Raise `--max-candidates` first; use `2048` for serious runs unless memory
  pressure forces a smaller value.
- Reinstall `big2_rust` after Rust changes so Python and Rust agree on the batch
  contract.
- Run `cargo test --manifest-path big2-rust/Cargo.toml` and
  `uv run pytest big2/training/rust_ppo/tests/test_rust_ppo.py -v`.
- Reduce `--num-envs` before reducing `--max-candidates`; smaller candidate
  buffers change action availability, while fewer envs only reduce throughput.
