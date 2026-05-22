# Big 2 RL Wiki

This wiki documents the current Big2 v2 Big 2 reinforcement-learning stack.
It is organized by subsystem so the Rust game engine, Python PPO layer, and
operational runbooks can evolve without turning one README into the source of
all truth.

## Start Here

- [Rust Environment](rust-env.md): card masks, action catalog, rules, single-game
  state, vectorized env semantics.
- [Rust Python API](rust-python-api.md): PyO3/maturin module surface and batch
  tuple contract used by Python.
- [Big2 v2 trainer](python-big2-v2.md): Python package layout, adapter,
  config, checkpoints, eval, and training entrypoint.
- [Model and Rollout](model-and-rollout.md): actor-critic inputs/outputs,
  candidate scoring, rollout buffer, and PPO update flow.
- [Test Runbook](test-runbook.md): commands for building, installing, and
  validating the Rust/Python integration.
- [Training, Evals, and Checkpoints](training-evals-checkpoints.md): how to run
  a small training job and track performance against random, greedy, and smart
  opponents.

## Architecture Boundary

Rust owns game compute:

- shuffling/dealing/reset
- legal move generation
- action catalog and move metadata
- observation encoding
- vectorized stepping
- terminal rewards
- move feature generation

Python owns learning orchestration:

- tensor conversion at the PyO3 boundary
- Torch actor-critic model
- policy sampling and PPO update
- lightweight metadata-based heuristic opponents
- metrics, evals, and checkpoints

If a Python path starts rebuilding card legality, enumerating moves, or
performing repeated per-card/per-move translations that can be exported from
Rust once, move that work into Rust.

## Current Package Map

```text
big2-rust/
  src/                  Rust rules/env/vectorized env/PyO3 binding
  tests/                Rust tests for catalog, rules, env, observations, vec env

big2/training/big2_v2/
  env_adapter.py         PyO3 batch -> Torch tensors
  metadata.py            Python view of Rust move metadata/features
  model.py               Rust-native candidate actor-critic
  rollout.py             Vectorized rollout collection
  update.py              PPO update
  opponents.py           Metadata-based random/greedy/smart choices
  evaluate.py            Eval versus fixed opponents
  checkpoints.py         Save/load/latest checkpoint helpers
  run.py                 Smoke/train CLI
```
