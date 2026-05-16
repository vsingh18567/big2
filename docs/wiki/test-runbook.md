# Test Runbook

Operational commands for validating the Rust-backed Big 2 stack from a clean
checkout. Run commands from the repository root unless noted.

## Environment Setup

Create a virtualenv and install Python dependencies:

```sh
uv venv
source .venv/bin/activate
uv pip install -e .
```

Install the Rust PyO3 module into the active virtualenv:

```sh
uvx maturin develop --manifest-path big2-rust/Cargo.toml --features python
```

Quick import check:

```sh
python - <<'PY'
import big2_rust
env = big2_rust.Big2VecEnv(2, 123, 128)
print(env.reset()[:3])
PY
```

## Rust Validation

Build and test the Rust rules engine, vectorized env, and PyO3 feature:

```sh
cargo check --manifest-path big2-rust/Cargo.toml
cargo check --manifest-path big2-rust/Cargo.toml --features python
cargo test --manifest-path big2-rust/Cargo.toml
```

The Rust tests cover action catalog construction, legal move generation,
single-game env behavior, observation encoding, vectorized env stepping, and
candidate-list truncation accounting.

## Python Validation

Run the Rust PPO integration tests after installing `big2_rust`:

```sh
uv run pytest big2/training/rust_ppo/tests/test_rust_ppo.py -v
```

Useful targeted suites:

```sh
uv run pytest big2/simulator/tests -v
uv run pytest big2/game_server/test_api.py -v
uv run pytest big2/1/tests -v
```

Run all currently discoverable Python tests:

```sh
uv run pytest big2 -v
```

If `uv run` cannot see the local Rust extension, activate `.venv` and rerun
`uvx maturin develop --manifest-path big2-rust/Cargo.toml --features python`
before invoking pytest.

## Smoke Training Check

Run a minimal PPO smoke pass without writing checkpoints:

```sh
uv run python -m big2.training.rust_ppo.run \
  --num-envs 2 \
  --rollout-steps 4 \
  --max-candidates 128 \
  --ppo-epochs 1 \
  --mini-batch-size 8
```

Expected output is one line like:

```text
rust_ppo smoke: samples=8 loss=... entropy=...
```

## Common Failures

- `ModuleNotFoundError: big2_rust`: install the Rust module with `maturin
  develop` in the same virtualenv used by pytest or training.
- Candidate tensor shape errors: confirm the Python command and Rust module were
  built from the same checkout.
- Empty or invalid candidate rows: run `cargo test --manifest-path
  big2-rust/Cargo.toml` first, then rerun the Rust PPO tests.
- Truncation-related instability: raise `--max-candidates`; truncation means at
  least one legal candidate list did not fit the rectangular batch buffer.
