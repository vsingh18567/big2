# Autoresearch Results

Objective: improve Rust PPO Big 2 evaluation performance against `smart` and `greedy` opponents, using `runs/rust_ppo/train_metrics_256env_128cand.jsonl` as the baseline reference.

## Baseline Reference

Source metrics: `runs/rust_ppo/train_metrics_256env_128cand.jsonl`

Latest config row:

- seed: `42`
- num envs: `256`
- rollout steps: `128`
- batches target: `500`
- max candidates: `256`
- PPO epochs: `2`
- minibatch: `2048`
- learning rate: `3e-4`
- entropy coefficient: `0.01`
- opponent mix: learner `0.75`, random `0.10`, greedy `0.10`, smart `0.05`
- eval cadence: every `25` batches, `256` games per seat, all 4 seats

Observed through batch `174`, with evals through batch `150`:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.650 | 0.254 | -0.057 | 0.136 | -0.263 | 0.390 |
| 50 | 0.684 | 0.333 | 0.143 | 0.233 | -0.022 | 0.566 |
| 75 | 0.717 | 0.354 | 0.155 | 0.233 | -0.029 | 0.588 |
| 100 | 0.748 | 0.370 | 0.163 | 0.245 | -0.045 | 0.615 |
| 125 | 0.736 | 0.409 | 0.235 | 0.262 | 0.008 | 0.671 |
| 150 | 0.735 | 0.406 | 0.241 | 0.271 | 0.040 | 0.677 |

Best baseline checkpoint so far: `runs/rust_ppo/checkpoints_256env_128cand/batch_000150.pt`

Candidate truncation: `0` total observed; max candidate count `253`.

Tail diagnostics over last 25 batch rows:

- KL: `0.0096`
- clip fraction: `0.0581`
- entropy: `0.1889`
- value explained variance: `0.2687`
- pass rate: `0.4691`
- episode length mean: `50.0`

## Experiment Queue

Operational note, `2026-05-15 22:45-22:52 PDT`: the original queued first-wave
commands produced only one training batch each and no eval rows. I preserved
those files as aborted warmups and relaunched the same hypotheses with fresh
`*_v2` metrics paths as persistent sessions. A fourth code-change hypothesis,
`terminal_credit_75`, is running in the `autoresearch_terminal_boundary`
worktree after a targeted regression test passed.

### baseline_repro_75

- hypothesis: a 75-batch reproduction of the current config establishes run-to-run variance for first-wave comparisons.
- branch/worktree: `autoresearch_baseline_repro` / `/Users/vikramsingh/Desktop/coding/big2_autoresearch_baseline`
- code/config changes: none
- seed: `142`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 75 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 142 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/baseline_repro_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/baseline_repro_75_checkpoints
```

- status: queued

Relaunched as `baseline_repro_75_v2`:

- metrics: `/Users/vikramsingh/Desktop/coding/big2_autoresearch_baseline/runs/rust_ppo/autoresearch/baseline_repro_75_v2_metrics.jsonl`
- checkpoint dir: `/Users/vikramsingh/Desktop/coding/big2_autoresearch_baseline/runs/rust_ppo/autoresearch/baseline_repro_75_v2_checkpoints`
- status: completed batch `75`

Eval rows:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.716 | 0.269 | -0.012 | 0.190 | -0.127 | 0.459 |
| 50 | 0.763 | 0.376 | 0.203 | 0.205 | -0.038 | 0.581 |
| 75 | 0.750 | 0.398 | 0.236 | 0.230 | -0.028 | 0.629 |

Conclusion: reference reproduction. It improved with training, but stayed below
the existing seed-42 baseline's best batch-150 score and below the low-lr and
terminal-credit first-wave runs.

### heuristic_mix_75

- hypothesis: increasing fixed greedy/smart pressure should improve target-opponent eval sample efficiency without losing too much self-play diversity.
- branch/worktree: `autoresearch_heuristic_mix` / `/Users/vikramsingh/Desktop/coding/big2_autoresearch_mix`
- code/config changes: none
- seed: `242`
- opponent mix: learner `0.55`, random `0.05`, greedy `0.25`, smart `0.15`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 75 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 242 --learner-weight 0.55 --random-weight 0.05 --greedy-weight 0.25 --smart-weight 0.15 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/heuristic_mix_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/heuristic_mix_75_checkpoints
```

- early-stop criteria: reject if clearly below baseline reproduction on both greedy and smart after batch 50, or if throughput degrades badly.
- status: queued

Relaunched as `heuristic_mix_75_v2`:

- metrics: `/Users/vikramsingh/Desktop/coding/big2_autoresearch_mix/runs/rust_ppo/autoresearch/heuristic_mix_75_v2_metrics.jsonl`
- checkpoint dir: `/Users/vikramsingh/Desktop/coding/big2_autoresearch_mix/runs/rust_ppo/autoresearch/heuristic_mix_75_v2_checkpoints`
- status: killed after batch `60` because target eval was behind the reference
  trend on both smart and greedy after two checkpoints.

Eval rows:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.714 | 0.210 | -0.092 | 0.143 | -0.220 | 0.353 |
| 50 | 0.733 | 0.258 | -0.113 | 0.158 | -0.303 | 0.416 |

Conclusion: reject. Increasing fixed greedy/smart pressure this aggressively
reduced learner sample share and did not improve target-opponent eval.

### low_lr_75

- hypothesis: lowering learning rate to `1e-4` may slow entropy collapse and improve target-opponent stability, possibly at lower early sample efficiency.
- branch/worktree: `autoresearch_low_lr` / `/Users/vikramsingh/Desktop/coding/big2_autoresearch_low_lr`
- code/config changes: none
- seed: `342`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 75 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0001 --seed 342 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/low_lr_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/low_lr_75_checkpoints
```

- early-stop criteria: reject if it is materially behind target-opponent baseline at batch 75 and only offers slower learning.
- status: queued

Relaunched as `low_lr_75_v2`:

- metrics: `/Users/vikramsingh/Desktop/coding/big2_autoresearch_low_lr/runs/rust_ppo/autoresearch/low_lr_75_v2_metrics.jsonl`
- checkpoint dir: `/Users/vikramsingh/Desktop/coding/big2_autoresearch_low_lr/runs/rust_ppo/autoresearch/low_lr_75_v2_checkpoints`
- status: completed batch `75`

Eval rows:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.736 | 0.390 | 0.210 | 0.227 | -0.025 | 0.616 |
| 50 | 0.775 | 0.422 | 0.265 | 0.279 | 0.050 | 0.701 |
| 75 | 0.752 | 0.432 | 0.263 | 0.253 | -0.014 | 0.685 |

Tail diagnostics over last 25 batch rows: KL `0.0031`, clip fraction `0.0395`,
entropy `0.3138`, value explained variance `0.2481`, pass rate `0.4593`,
episode length mean `51.2`.

Conclusion: promote. Lower learning rate materially improved early target eval,
with the best checkpoint at batch `50`. Extend to see whether the batch-75 smart
dip is noise or slower learning saturation.

### terminal_credit_75

- hypothesis: terminal rewards should be credited to the latest learner record
  for each learner-controlled seat in an episode, not only to a learner record
  when the learner happened to take the terminal action. This should improve the
  sparse reward signal, especially loser penalties when a fixed opponent ends
  the game.
- branch/worktree: `autoresearch_terminal_boundary` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_boundary`
- code/config changes: in `big2/training/rust_ppo/rollout.py`, keep the latest
  learner record by `(env_index, player)` during a rollout; when an env
  terminates, add that player's final Rust terminal reward to the latest record
  and set `done=True`, then clear records for the reset env.
- test:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m pytest big2/training/rust_ppo/tests/test_rust_ppo.py::test_collect_rollout_credits_terminal_rewards_to_latest_learner_records big2/training/rust_ppo/tests/test_rust_ppo.py::test_collect_rollout_and_ppo_update_smoke -q
```

Result: `2 passed in 7.78s`.

- seed: `442`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 75 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 442 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_credit_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_credit_75_checkpoints
```

- first-wave result: completed batch `75`; best checkpoint is batch `75`.

Eval rows:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.754 | 0.381 | 0.217 | 0.244 | 0.003 | 0.625 |
| 50 | 0.762 | 0.402 | 0.245 | 0.260 | 0.023 | 0.662 |
| 75 | 0.767 | 0.456 | 0.309 | 0.276 | 0.053 | 0.732 |

Tail diagnostics over last 25 batch rows: KL `0.0095`, clip fraction `0.0803`,
entropy `0.3244`, value explained variance `0.4291`, pass rate `0.4834`,
episode length mean `53.1`.

Conclusion: promote. This is the strongest completed checkpoint so far, beating
the existing baseline's best observed smart+greedy score (`0.732` vs `0.677`)
with positive smart reward and no candidate truncation.

### terminal_credit_150_resume

- hypothesis: continue the strongest first-wave checkpoint to see whether the
  terminal-credit fix keeps improving through 150 batches.
- branch/worktree: `autoresearch_terminal_boundary` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_boundary`
- code/config changes: same as `terminal_credit_75`
- seed: `442`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --resume --batches 150 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 442 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_credit_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_credit_75_checkpoints
```

- result: completed through batch `150`; promoted again for continuation to
  batch `225`.

Eval rows added by the resume:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.787 | 0.447 | 0.304 | 0.265 | 0.031 | 0.712 |
| 125 | 0.785 | 0.465 | 0.320 | 0.285 | 0.062 | 0.750 |
| 150 | 0.783 | 0.458 | 0.323 | 0.315 | 0.115 | 0.773 |

Conclusion: continue. Batch `150` is the best checkpoint so far.

### low_lr_150_resume

- hypothesis: continue the low learning-rate run to determine whether the smart
  dip at batch 75 was noise or a real plateau.
- branch/worktree: `autoresearch_low_lr` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_low_lr`
- code/config changes: none
- seed: `342`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --resume --batches 150 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0001 --seed 342 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/low_lr_75_v2_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/low_lr_75_v2_checkpoints
```

- result: completed through batch `150`.

Eval rows added by the resume:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.791 | 0.431 | 0.268 | 0.234 | -0.031 | 0.665 |
| 125 | 0.790 | 0.422 | 0.256 | 0.279 | 0.031 | 0.701 |
| 150 | 0.800 | 0.470 | 0.312 | 0.252 | -0.007 | 0.722 |

Conclusion: secondary candidate. It improves greedy strongly, but smart remains
weaker and less stable than terminal-credit at `lr=3e-4`.

### terminal_low_lr_75

- hypothesis: combining the terminal-credit fix with `lr=1e-4` may keep the
  better terminal reward signal while reducing update aggressiveness.
- branch/worktree: `autoresearch_terminal_low_lr` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_low_lr`
- code/config changes: same terminal-credit rollout patch as
  `terminal_credit_75`; learning rate changed to `1e-4`.
- test:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m pytest big2/training/rust_ppo/tests/test_rust_ppo.py::test_collect_rollout_credits_terminal_rewards_to_latest_learner_records -q
```

Result: `1 passed in 3.03s`.

- seed: `1042`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 75 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0001 --seed 1042 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_low_lr_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_low_lr_75_checkpoints
```

- result: completed batch `75`.

Eval rows:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.716 | 0.363 | 0.181 | 0.236 | -0.013 | 0.599 |
| 50 | 0.732 | 0.399 | 0.256 | 0.237 | 0.006 | 0.636 |
| 75 | 0.749 | 0.435 | 0.301 | 0.258 | 0.034 | 0.692 |

Conclusion: reject as a combination. It is decent, but worse than both the
terminal-credit run and the low-lr run at comparable checkpoints.

### terminal_credit_225_resume

- hypothesis: the best run continues to improve if extended to a standard
  225-batch exploratory length.
- branch/worktree: `autoresearch_terminal_boundary` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_boundary`
- code/config changes: same as `terminal_credit_75`
- seed: `442`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --resume --batches 225 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 442 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_credit_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_credit_75_checkpoints
```

- result: completed through batch `225`; no truncation.

Eval rows added by this resume:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 175 | 0.803 | 0.501 | 0.378 | 0.303 | 0.091 | 0.804 |
| 200 | 0.775 | 0.457 | 0.324 | 0.308 | 0.106 | 0.765 |
| 225 | 0.769 | 0.514 | 0.390 | 0.310 | 0.089 | 0.824 |

Tail diagnostics over last 25 batch rows: KL `0.0091`, clip fraction `0.0599`,
entropy `0.2069`, value explained variance `0.4301`, pass rate `0.4821`,
episode length mean `53.0`.

Conclusion: best overall. Recommend this configuration and checkpoint for the
next long confirmation run.

### terminal_smart_focus_100

- hypothesis: with the terminal-credit fix, a smart-focused opponent mix may
  improve smart eval without the collapse seen in the earlier broad heuristic
  mix.
- branch/worktree: `autoresearch_terminal_low_lr` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_low_lr`
- code/config changes: same terminal-credit rollout patch; opponent mix changed
  to learner `0.60`, random `0.05`, greedy `0.05`, smart `0.30`.
- seed: `1542`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 100 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 1542 --learner-weight 0.60 --random-weight 0.05 --greedy-weight 0.05 --smart-weight 0.30 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_smart_focus_100_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_smart_focus_100_checkpoints
```

- result: completed batch `100`; no truncation.

Eval rows:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.753 | 0.345 | 0.173 | 0.229 | -0.007 | 0.574 |
| 50 | 0.734 | 0.387 | 0.229 | 0.227 | -0.015 | 0.614 |
| 75 | 0.750 | 0.413 | 0.260 | 0.293 | 0.079 | 0.706 |
| 100 | 0.754 | 0.432 | 0.282 | 0.274 | 0.058 | 0.706 |

Conclusion: reject for now. The smart-focused mix can improve smart relative to
the original no-code baseline, but it did not beat the default-mix
terminal-credit run.

### terminal_credit_seed1442_75

- hypothesis: check whether the terminal-credit improvement reproduces with a
  different seed under the original opponent mix.
- branch/worktree: `autoresearch_terminal_low_lr` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_low_lr`
- code/config changes: same terminal-credit rollout patch
- seed: `1442`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 75 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 1442 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_credit_seed1442_75_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_credit_seed1442_75_checkpoints
```

- candidate-limited: killed at batch `24` because
  `truncated_candidate_lists=1` before the first eval. Per research rules, do
  not use this run as evidence.

### terminal_credit_seed1442_75_512cand

- hypothesis: rerun the seed-1442 terminal-credit reproducibility check with a
  wider candidate buffer because the 256-candidate version truncated.
- branch/worktree: `autoresearch_terminal_low_lr` /
  `/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_low_lr`
- code/config changes: same terminal-credit rollout patch; `max_candidates=512`
- seed: `1442`
- command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 75 --num-envs 256 --rollout-steps 128 --max-candidates 512 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 1442 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_credit_seed1442_75_512cand_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_credit_seed1442_75_512cand_checkpoints
```

- result: completed batch `75`; no truncation.

Eval rows:

| batch | random wr | greedy wr | greedy reward | smart wr | smart reward | smart+greedy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.754 | 0.389 | 0.234 | 0.230 | -0.008 | 0.619 |
| 50 | 0.721 | 0.397 | 0.245 | 0.241 | 0.005 | 0.639 |
| 75 | 0.778 | 0.480 | 0.343 | 0.279 | 0.060 | 0.760 |

Tail diagnostics over last 25 batch rows: KL `0.0098`, clip fraction `0.0810`,
entropy `0.3042`, value explained variance `0.4185`, pass rate `0.4833`,
episode length mean `53.2`.

Conclusion: supports the terminal-credit hypothesis across seed and candidate
width. The 512-candidate run is slower, but it had zero truncation.

## Final Ranking

| rank | run | best batch | max candidates | greedy wr | greedy reward | smart wr | smart reward | smart+greedy | conclusion |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `terminal_credit_225_resume` | 225 | 256 | 0.514 | 0.390 | 0.310 | 0.089 | 0.824 | recommend |
| 2 | `terminal_credit_seed1442_75_512cand` | 75 | 512 | 0.480 | 0.343 | 0.279 | 0.060 | 0.760 | supports fix |
| 3 | `low_lr_150_resume` | 150 | 256 | 0.470 | 0.312 | 0.252 | -0.007 | 0.722 | secondary |
| 4 | `terminal_smart_focus_100` | 75/100 | 256 | 0.432 | 0.282 | 0.274 | 0.058 | 0.706 | reject vs terminal default |
| 5 | `terminal_low_lr_75` | 75 | 256 | 0.435 | 0.301 | 0.258 | 0.034 | 0.692 | reject combination |
| 6 | existing baseline reference | 150 | 256 | 0.406 | 0.241 | 0.271 | 0.040 | 0.677 | surpassed |
| 7 | `baseline_repro_75_v2` | 75 | 256 | 0.398 | 0.236 | 0.230 | -0.028 | 0.629 | reference only |
| 8 | `heuristic_mix_75_v2` | 50 | 256 | 0.258 | -0.113 | 0.158 | -0.303 | 0.416 | killed/reject |

## Recommended Next Configuration

Use the terminal-credit rollout fix with the original opponent mix and PPO
settings:

- `num_envs=256`
- `rollout_steps=128`
- `max_candidates=256` for this seed/config, because the winning run had
  `truncated_candidate_lists=0`; use `512` or higher for seeds that truncate.
- `ppo_epochs=2`
- `mini_batch_size=2048`
- `lr=3e-4`
- `entropy_coef=0.01`
- opponent mix: learner `0.75`, random `0.10`, greedy `0.10`, smart `0.05`
- eval/checkpoint every `25` batches, `256` eval games per seat, all 4 seats

Strongest checkpoint:
`/Users/vikramsingh/Desktop/coding/big2_autoresearch_terminal_boundary/runs/rust_ppo/autoresearch/terminal_credit_75_checkpoints/batch_000225.pt`

Recommended confirmation command:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run --train --batches 225 --num-envs 256 --rollout-steps 128 --max-candidates 256 --ppo-epochs 2 --mini-batch-size 2048 --lr 0.0003 --seed 442 --learner-weight 0.75 --random-weight 0.10 --greedy-weight 0.10 --smart-weight 0.05 --eval-interval 25 --eval-games 256 --eval-num-envs 64 --checkpoint-interval 25 --logging-mode max --device cpu --metrics-path runs/rust_ppo/autoresearch/terminal_credit_confirm_metrics.jsonl --checkpoint-dir runs/rust_ppo/autoresearch/terminal_credit_confirm_checkpoints
```
