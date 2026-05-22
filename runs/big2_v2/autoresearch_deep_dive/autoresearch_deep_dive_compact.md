# Autoresearch Deep Dive Compact

This is the compact version of `autoresearch_deep_dive.md`. It keeps the
important run history, what each intervention tried, and how each run turned
out, without the per-batch narrative detail.

## Current Best Reference

Best checkpoint:

`runs/big2_v2/dynamic_action_winloss_table_profiles_seed2442_checkpoints/batch_000550.pt`

Reference eval at batch `550`:

| random | greedy | smart | greedy+smart |
| ---: | ---: | ---: | ---: |
| 0.887 | 0.680 | 0.499 | 1.179 |

This remains the checkpoint to beat. It cleared the original target
(`greedy > 0.600`, `smart > 0.400`) and is the strongest broadly useful model
observed so far.

## Main Research Arc

### 1. Terminal-Credit Fix

Early Big2 v2 runs had a credit-assignment bug: terminal reward was only
credited when the learner happened to take the final action. The fix credits
each learner-controlled seat's final terminal reward to that seat's latest
learner record in the episode.

After this fix, the best early continuation-style run was:

| run | best batch | greedy | smart | greedy+smart | outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| `terminal_credit_500_seed442` | 400 | 0.515 | 0.354 | 0.868 | improved over baseline but plateaued |

Interpretation: PPO health looked fine, but the policy became highly decisive
and oscillated around a `0.82-0.87` combined greedy+smart band. More simple
training was unlikely to solve the plateau.

### 2. Entropy Controls and Checkpoint-League Curriculum

The next system changes added:

- CLI entropy controls and linear entropy scheduling.
- `--curriculum smart-greedy`, an eval-gated curriculum that changes entropy,
  opponent mix, and checkpoint-opponent weight based on greedy+smart eval.
- Checkpoint-opponent league play sourced from the current run or another
  checkpoint directory.

Recommended curriculum run:

| run | idea | outcome |
| --- | --- | --- |
| `terminal_credit_league_500_seed442` | raise entropy and checkpoint pressure after eval thresholds | did not materially break the plateau |

Best observed point from this family:

| batch | greedy | smart | greedy+smart |
| ---: | ---: | ---: | ---: |
| 450 | 0.518 | 0.355 | 0.873 |

Conclusion: schedule and curriculum changes alone were not enough.

### 3. Fixed-Seat Assignment and Rollout Bootstrap

Two training/eval mismatches were fixed next:

- `--controller-assignment episode-seat` and `single-learner`, so controller
  assignment can be fixed across a whole episode.
- Non-zero bootstrap value for unfinished learner trajectories at fixed rollout
  boundaries.

Run:

| run | start | idea | best useful read | outcome |
| --- | --- | --- | --- | --- |
| `single_learner_bootstrap_seed442` | warm-start from prior best | train one fixed learner seat against fixed opponent seats | batch 550: greedy 0.518, smart 0.354, combined 0.872 | mechanically better, but still plateaued |

Conclusion: this made the training setup more correct, but did not raise the
eval ceiling.

### 4. Candidate-Set Context

The model previously scored each candidate action without an explicit summary
of the whole legal candidate set. `--candidate-set-context` added masked
mean/max summaries of legal actions into policy and value paths.

Run:

| run | init | result |
| --- | --- | --- |
| `candidate_context_single_learner_seed1442` | warm-start from prior checkpoint | improved greedy matchup and combined score, but smart remained stuck |

Key evals:

| batch | greedy | smart | greedy+smart |
| ---: | ---: | ---: | ---: |
| 25 | 0.537 | 0.352 | 0.889 |
| 50 | 0.533 | 0.357 | 0.890 |
| 75 | 0.537 | 0.350 | 0.887 |
| 100 | 0.546 | 0.353 | 0.899 |

Conclusion: legal-set awareness helped, especially against greedy, but did not
solve the smart-opponent bottleneck.

### 5. Dynamic Action-Outcome Features

The next hypothesis was that the model needed direct action-outcome context:
what remains after a move, whether the move empties or nearly empties the hand,
whether it preserves important combinations, and how efficiently it responds to
the current trick.

A dense Rust-side feature tensor was rejected because it was too expensive:
`512 envs * 256 candidates * 92 floats` caused large padded transfers. The
final implementation computes dynamic candidate-outcome features inside
`Big2V2ActorCritic` from existing tensors:

- current hand mask
- candidate card mask
- remaining hand mask
- rank/suit histograms
- finish and near-finish flags
- singleton/pair/triple/quad counts
- high/low card pressure
- response deltas against the current trick

Checkpoint warm-start was adjusted so old action-projection weights are
preserved and only new feature columns initialize from zero.

### 6. Win/Loss Terminal Reward and Table Profiles

The old terminal reward gave the winner `+1` and losers
`-cards_remaining / 13`, so near-losses were only lightly punished. The new
`--terminal-reward-mode win-loss` aligns training more directly with win rate.

Controller assignment was also revised to `table-profile`:

- `learner_weight`: full self-play tables
- `greedy_weight`: one learner against three greedy opponents
- `smart_weight`: one learner against three smart opponents
- `checkpoint_opponent_weight`: one learner against three frozen checkpoint
  opponents

This kept self-play central while still applying targeted benchmark pressure.

Successful run:

| run | init | recipe |
| --- | --- | --- |
| `dynamic_action_winloss_table_profiles_seed2442` | `candidate_context_single_learner_seed1442` batch 100 | candidate context, dynamic action features, binary win/loss reward, table-profile assignment |

Important eval trajectory:

| batch | random | greedy | smart | greedy+smart | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 25 | 0.839 | 0.565 | 0.373 | 0.938 | strong early lift |
| 100 | 0.815 | 0.580 | 0.406 | 0.986 | smart first crosses 0.4 |
| 200 | 0.859 | 0.617 | 0.441 | 1.059 | original target cleared |
| 300 | 0.842 | 0.634 | 0.458 | 1.092 | still improving |
| 450 | 0.854 | 0.640 | 0.475 | 1.114 | new high before late jump |
| 550 | 0.887 | 0.680 | 0.499 | 1.179 | best checkpoint |
| 650 | 0.847 | 0.646 | 0.475 | 1.120 | plateau confirmed |

Conclusion: this was the main breakthrough. After batch `550`, batches `575`,
`600`, `625`, and `650` failed to recover toward the high, so the recipe was
stopped as plateaued.

## Post-Best Improvement Attempts

The objective after best-550 became: improve beyond greedy `0.680`, smart
`0.499`, combined `1.179`.

### 0. Cold-Start Replications

Two cold-start runs tested whether the successful
`dynamic_action_winloss_table_profiles_seed2442` recipe can reproduce without
policy warm-start. Both used the same main architecture/objective recipe:

- `--candidate-set-context`
- `--dynamic-action-features`
- `--terminal-reward-mode win-loss`
- `--controller-assignment table-profile`
- no `--init-checkpoint`
- no `--resume`

Run setup:

| run | seed | learner | greedy | smart | checkpoint | purpose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pure_dynamic_action_winloss_table_profiles_seed7442` | 7442 | 0.55 | 0.20 | 0.20 | 0.05 | closest cold-start replication of best recipe |
| `pure_dynamic_action_winloss_table_profiles_ckpt015_seed7552` | 7552 | 0.55 | 0.15 | 0.15 | 0.15 | higher frozen-checkpoint pressure variant |

Both use frozen checkpoint opponents from:

`runs/big2_v2/terminal_credit_league_500_seed442_v2_checkpoints`

Final outcome:

| run | status | best batch | random | greedy | smart | greedy+smart | final/latest read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pure_dynamic_action_winloss_table_profiles_seed7442` | completed to batch 900 | 850 | 0.838 | 0.664 | 0.507 | 1.170 | batch 900: 0.861 / 0.684 / 0.472 / 1.156 |
| `pure_dynamic_action_winloss_table_profiles_ckpt015_seed7552` | stopped at batch 804 | 650 | 0.861 | 0.659 | 0.479 | 1.138 | batch 800: 0.868 / 0.663 / 0.452 / 1.115 |

Interpretation: the 0.05 checkpoint-opponent run is a successful pure
replication of the best recipe in the important sense: it trained from scratch
to roughly the same evaluation band as the warm-start breakthrough. It did not
beat the best-550 reference on combined greedy+smart (`1.170` vs `1.179`), but
it did produce the strongest smart score observed so far at batch `850`
(`0.507` vs the reference `0.499`). The final checkpoint regressed on smart, so
`batch_000850.pt` is the keeper from this run, not `batch_000900.pt`.

The 0.15 checkpoint-pressure variant looked competitive early but faded. Its
best combined result came at batch `650` and was materially below both the
0.05 cold-start run and the best-550 reference. It was stopped rather than run
to completion.

### 1. Self-League From Best-550

Run:

`self_league_from_best550_seed3550`

Idea:

- Start from best-550.
- Reset optimizer.
- Train against the stronger dynamic-action checkpoint league instead of the
  older terminal-credit league.
- Keep self-play as the core.
- Use a mild entropy schedule to reopen exploration.

Outcome:

| best batch | random | greedy | smart | greedy+smart | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 50 | 0.873 | 0.679 | 0.463 | 1.142 | reject |

The run was mechanically stable but regressed by batch `100`. It did not
preserve best-550 smart strength.

### 2. Weakness Diagnostics

Diagnostic tooling was added:

- `big2/training/big2_v2/diagnose_policy.py`
- `big2/training/big2_v2/evaluate_checkpoint_match.py`

Diagnostics on best-550 found that the model is already strong when converting
lead/endgame positions, but weaker in response states.

Most actionable weak buckets:

| bucket | opponent | win rate | delta vs overall |
| --- | --- | ---: | ---: |
| optional-pass response | greedy | 0.557 | -0.109 |
| optional-pass response | smart | 0.404 | -0.049 |
| full-house response | smart | 0.371 | -0.082 |
| large hand, 10-13 cards | smart | 0.393 | -0.060 |

Interpretation: the next useful work should target response-state decision
quality, not generic lead conversion.

### 3. Optional-Pass Penalty

Run:

`optional_pass_penalty_from_best550_seed4550`

Idea:

- Add `--pass-penalty -0.01`.
- Penalize only optional passes, where at least one legal non-pass move exists.
- Avoid punishing forced passes.

Outcome:

| best batch | random | greedy | smart | greedy+smart | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 75 | 0.866 | 0.665 | 0.479 | 1.144 | reject |

The run stayed PPO-stable but behaviorally narrowed. Entropy and pass rate
drifted down, yet response quality did not improve enough. By batch `100`, it
was clearly below best-550.

### 4. Relative Action Features

Run:

`relative_action_features_from_best550_seed5651`

Idea:

- Improve representation rather than reward shaping.
- Expand dynamic action features with:
  - optional-vs-forced pass indicators
  - count of playable alternatives
  - selected move strength relative to weakest/strongest legal non-pass moves
  - whether a same-kind response is the cheapest available same-kind response
- Prefix-load old candidate-outcome encoder weights and zero-initialize only
  new feature columns.

Outcome:

| best batch | random | greedy | smart | greedy+smart | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 175 | 0.856 | 0.670 | 0.488 | 1.158 | reject as replacement |

This was not an optimizer failure: PPO health remained normal and candidate
truncation stayed at zero. It simply did not beat the best-550 reference.

### 5. Pairwise Residual Action Head

Run:

`pairwise_residual_from_best550_seed6751`

Idea:

- Add `--pairwise-action-head`, a nonlinear residual logit head over
  `[state_action_h, action_h, state_action_h * action_h]`.
- Initialize the final residual layer to zero so the warm-started policy begins
  with identical logits.
- Use low learning rate so PPO can learn residual corrections without
  immediately perturbing the best policy.

Reads:

| batch | random | greedy | smart | greedy+smart | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 25 | 0.880 | 0.670 | 0.476 | 1.146 | best point, still below reference |
| 50 | 0.852 | 0.655 | 0.478 | 1.132 | regressed |

Status: PPO health looked normal, but the run was already below best-550 and
moving farther away by batch `50`. Metrics stop at batch `71`, with no evidence
that the residual head was becoming a useful replacement.

## Older Run Summary

These runs were part of the pre-breakthrough search and are mostly superseded
by the later best-550 model.

| run | best batch | best greedy+smart | decision | short interpretation |
| --- | ---: | ---: | --- | --- |
| `terminal_credit_225_resume` | 225 | 0.823 | promote/continue at the time | best early terminal-credit run |
| `terminal_credit_seed1442_75_512cand` | 75 | 0.760 | promote/continue at the time | promising early but not enough evidence |
| `low_lr_150_resume` | 150 | 0.722 | secondary | stable, weaker than terminal default |
| `terminal_smart_focus_100` | 75 | 0.706 | reject vs terminal default | smart focus did not beat default recipe |
| `terminal_low_lr_75` | 75 | 0.692 | reject combination | stable but weak |
| `existing_baseline_seed42` | 150 | 0.677 | reference | old baseline |
| `baseline_repro_75_v2` | 75 | 0.629 | reference | reproducibility baseline |
| `heuristic_mix_75_v2` | 50 | 0.416 | reject | weak heuristic mix |

Excluded or weak runs:

| run | reason |
| --- | --- |
| `baseline_repro_75` | no eval; too short |
| `launch_debug` | no eval; too short |
| `low_lr_75` | no eval; too short |
| `heuristic_mix_75` | no eval; too short |
| `terminal_credit_seed1442_75` | invalid due to candidate truncation |

## Current Interpretation

The most important lesson is that the plateau was not solved by training
longer or by curriculum pressure alone. The breakthrough required the model
and objective to better match the decision problem:

- fixed episode/table assignment closer to eval
- rollout bootstrap for unfinished trajectories
- candidate-set context
- dynamic action-outcome features
- binary win/loss reward
- self-play-core table profiles with targeted opponent pressure

The remaining weakness appears concentrated in response-state decision quality,
especially optional-pass situations and higher-order responses against smart
opponents. Post-best attempts that changed opponent league pressure, reward
shaping, relative action features, or the residual action head have not beaten
the best checkpoint. The cold-start result shows the recipe is reproducible,
but also reinforces that the late-stage ceiling is real.

## Practical Recommendation

Use best-550 as the reference model unless a later run clearly exceeds it over
multiple eval checkpoints:

`runs/big2_v2/dynamic_action_winloss_table_profiles_seed2442_checkpoints/batch_000550.pt`

Keep the pure cold-start batch `850` checkpoint as the clean replication
artifact:

`runs/big2_v2/pure_dynamic_action_winloss_table_profiles_seed7442_checkpoints/batch_000850.pt`

For future experiments, compare against the full reference split, not only the
combined score:

| random | greedy | smart | greedy+smart |
| ---: | ---: | ---: | ---: |
| 0.887 | 0.680 | 0.499 | 1.179 |

Small eval differences below roughly 2-3 percentage points should be treated
as noise unless they persist across multiple checkpoints.

## Retained Checkpoint Artifacts

After cleanup, checkpoint retention is intentionally minimal:

| checkpoint | reason |
| --- | --- |
| `dynamic_action_winloss_table_profiles_seed2442_checkpoints/batch_000550.pt` | current best reference |
| `pure_dynamic_action_winloss_table_profiles_seed7442_checkpoints/batch_000850.pt` | clean cold-start replication keeper |
| `candidate_context_single_learner_seed1442_checkpoints/batch_000100.pt` | lineage checkpoint used to initialize the best run |
| `terminal_credit_league_500_seed442_v2_checkpoints/batch_000425.pt` | frozen opponent pool member |
| `terminal_credit_league_500_seed442_v2_checkpoints/batch_000450.pt` | frozen opponent pool member |
| `terminal_credit_league_500_seed442_v2_checkpoints/batch_000475.pt` | frozen opponent pool member |
| `terminal_credit_league_500_seed442_v2_checkpoints/batch_000500.pt` | frozen opponent pool member |

Metrics JSONL files are kept for rejected and discontinued runs so their
results remain auditable without retaining full checkpoint ladders.
