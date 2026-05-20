# Autoresearch Deep Dive

## 2026-05-19 Update: Plateau-Breaking System Changes

The curriculum/checkpoint-league run did not materially break the plateau. The
best observed point was batch `450` from
`runs/rust_ppo/terminal_credit_league_500_seed442_metrics_v2.jsonl`:
`greedy=0.518`, `smart=0.355`, combined `0.873`. The final batch regressed to
`greedy=0.503`, `smart=0.321`. This is not close enough to the target
(`greedy>0.600`, `smart>0.400`) to justify more schedule-only tuning.

Two structural issues were identified and patched next:

1. **Controller assignment mismatch.** Training previously sampled
   learner/random/greedy/smart/checkpoint independently every turn. Evaluation
   measures a fixed policy seat against fixed opponent seats for a whole game.
   The training system now supports fixed per-episode seat assignment via
   `--controller-assignment episode-seat` and a direct eval-like mode via
   `--controller-assignment single-learner`, where one learner seat plays a
   whole episode against fixed opponent seats.
2. **Non-terminal rollout bootstrap.** PPO GAE previously used zero bootstrap
   value at every fixed rollout boundary. The update path now accepts a
   non-zero bootstrap value for unfinished learner trajectories when the next
   learner decision state is observable at the rollout boundary.

Verification:

- `uv run pytest big2/training/rust_ppo/tests/test_rust_ppo.py -q`:
  `20 passed`
- `uvx ruff check big2/training/rust_ppo/config.py big2/training/rust_ppo/rollout.py big2/training/rust_ppo/update.py big2/training/rust_ppo/run.py big2/training/rust_ppo/tests/test_rust_ppo.py`:
  clean

Next experiment: initialize from the best prior checkpoint, but train with
fixed single-learner seats so the training distribution matches the eval task.
The run should use stronger fixed opponents and the previous checkpoint league
only as frozen opponent diversity, not as the main curriculum explanation.

Started run:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run \
  --train \
  --resume \
  --batches 900 \
  --num-envs 512 \
  --rollout-steps 128 \
  --max-candidates 256 \
  --ppo-epochs 2 \
  --mini-batch-size 2048 \
  --lr 0.0003 \
  --entropy-coef 0.015 \
  --controller-assignment single-learner \
  --learner-weight 0.0 \
  --random-weight 0.05 \
  --greedy-weight 0.45 \
  --smart-weight 0.35 \
  --checkpoint-opponent-weight 0.15 \
  --seed 442 \
  --checkpoint-opponent-dir runs/rust_ppo/terminal_credit_league_500_seed442_v2_checkpoints \
  --checkpoint-opponent-limit 6 \
  --checkpoint-opponent-stride 25 \
  --eval-interval 25 \
  --eval-games 512 \
  --eval-num-envs 128 \
  --checkpoint-interval 25 \
  --logging-mode max \
  --device cpu \
  --metrics-path runs/rust_ppo/single_learner_bootstrap_seed442_metrics.jsonl \
  --checkpoint-dir runs/rust_ppo/single_learner_bootstrap_seed442_checkpoints
```

Early sanity check after batches `451-453`: controller assignment is fixed-seat
single learner, opponent turns are mostly greedy/smart/checkpoint, KL remains
near `0.009`, entropy is higher than the plateaued run (`~0.23-0.24`), value
EV is healthy (`~0.46-0.50`), and action probability p10 has dropped to
`~0.45`, indicating the policy is less deterministically stuck.

Batch `475` eval over 2048 games/opponent: `greedy=0.524`,
`smart=0.347`, combined `0.871`. Batch `500` eval dipped to
`greedy=0.521`, `smart=0.322`, combined `0.844`. This is not enough evidence
to reject the run because it has only trained `50` fresh batches after the
warm start. Continue at least through batch `550` before making a decision.

Batch `550` provided the requested 100-fresh-batch read after the batch-450
warm start. Result: `greedy=0.518`, `smart=0.354`, combined `0.872`
over 2048 games/opponent. Training internals remained healthy (`KL=0.0064`,
`value_EV=0.506`, `entropy=0.224`, `action_p10=0.489`, no candidate
truncation), so this is not an optimizer collapse. The fixed-seat/bootstrap
intervention made the training setup more correct, but by itself did not move
the eval ceiling.

Next intervention: give the model direct context about the full legal
candidate set before scoring individual actions. The current scorer encodes
state and each action, but the state-conditioned action projection has no
pooled summary of what alternatives are available in that decision. This can
make relative choice quality hard to learn in positions where the best action
depends on the whole legal-action menu. The new `--candidate-set-context`
architecture adds masked mean/max legal-action summaries into both policy and
value paths. It can be warm-started from the batch-450 checkpoint via
`--init-checkpoint`, loading compatible old weights while initializing only the
new context layers from scratch.

Candidate-context run started:
`runs/rust_ppo/candidate_context_single_learner_seed1442_metrics.jsonl`.
Initial batch-25 eval: `greedy=0.537`, `smart=0.352`, combined `0.889`.
This is the best combined score observed so far and suggests the candidate-set
context may help against greedy, but it is not yet a smart-opponent
breakthrough. Because prior short reads were misleading, continue through at
least batch `100` before deciding whether this intervention is genuinely
working.

Batch `50` eval held the same band: `greedy=0.533`, `smart=0.357`,
combined `0.890`. This confirms the early combined lift was not a one-eval
fluke, but the missing smart-opponent improvement remains the main bottleneck.
Batch `75` remained similar: `greedy=0.537`, `smart=0.350`, combined
`0.887`. The candidate-context intervention is consistently helping the
greedy matchup relative to the old plateau, but it has not changed the smart
matchup enough.
Batch `100` improved the combined score again: `greedy=0.546`,
`smart=0.353`, combined `0.899`. This is now a real gain over the old
`~0.87` ceiling, mostly from the greedy matchup. The smart matchup is still
stuck near `0.35`, so the next bottleneck is likely not generic PPO stability
or legal-set awareness. The model probably needs stronger action-outcome
features: what remains in hand after a move, whether the move empties or
nearly empties the hand, whether it preserves/breaks important combos, and how
efficiently it beats the current trick.

Implemented and then rejected a dense Rust-side `candidate_outcome_features`
batch tensor. It was semantically right because these are action-conditioned
`f(state, action)` features, not global observation state, but operationally
wrong: with `512` envs, `256` padded candidates, and `92` floats per candidate
it moves roughly `48 MB` per env step, mostly padding. The first attempted
dynamic run at `runs/rust_ppo/dynamic_action_context_seed2442_metrics.jsonl`
only wrote its config row before being killed.

Final implementation keeps the Rust/Python batch API lean (`obs`,
`candidate_ids`, `candidate_mask`) and computes dynamic candidate-outcome
features inside `RustCandidateActorCritic` from existing tensors:

- `obs[:, 0:52]`: current player's actual hand
- `move_features[candidate_ids][:, :, 0:52]`: candidate card mask
- remaining hand mask, rank/suit histograms, finish/near-finish flags,
  singleton/pair/triple/quad counts, high/low card pressure, and response
  deltas against the current trick

This preserves the desired `f(state, action)` signal without the padded Rust
transfer. Checkpoint warm-start was also fixed: when enabling dynamic features,
the existing `action_projection.0.weight` leading columns are copied from the
source checkpoint and only the new dynamic-feature columns are zero-initialized.
This avoids throwing away the best candidate-context action scorer.

Additional plateau fix: training now supports `--terminal-reward-mode win-loss`.
The old reward gave the winner `+1` and losers `-cards_remaining/13`, so a
near-loss with one card left was only `-0.077`. That can improve average reward
while leaving smart win rate stuck. Binary terminal credit aligns the training
objective more directly with eval win rate.

Added `--controller-assignment single-learner-uniform` to close another
train/eval gap. The existing `single-learner` mode sampled each opponent seat
independently, so even a high smart weight produced many mixed tables. The new
mode samples one opponent profile per episode and assigns all three non-learner
seats to that same profile, enabling direct all-smart/all-greedy training
boards.

Correction after review: the goal is a generally strong model, not a model
overfit to fixed smart/greedy tables. Added `--controller-assignment
table-profile` so self-play is the backbone while targeted opponent tables
remain explicit. In this mode `learner_weight` means full self-play tables;
`smart_weight`, `greedy_weight`, `random_weight`, and `checkpoint_weight` mean
one learner seat against three same-profile opponents. This keeps the CLI
semantics intuitive and makes self-play central again.

Verification:

- `cargo test --manifest-path big2-rust/Cargo.toml`: clean
- `uvx maturin develop --manifest-path big2-rust/Cargo.toml`: rebuilt editable
  Python extension
- `uv run pytest big2/training/rust_ppo/tests/test_rust_ppo.py -q`:
  `28 passed`
- `uvx ruff check ...`: clean
- CLI smoke with `--candidate-set-context --dynamic-action-features
  --terminal-reward-mode win-loss`: ran a rollout/update successfully

Recommended next experiment: warm-start from the best candidate-context
checkpoint, enable dynamic action-outcome features, use binary win/loss terminal
credit, and train with table profiles: mostly self-play, plus all-smart,
all-greedy, and a small frozen-checkpoint profile. This should build general
strength while still applying direct pressure to the explicit thresholds.

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run \
  --train \
  --batches 300 \
  --num-envs 512 \
  --rollout-steps 128 \
  --max-candidates 256 \
  --ppo-epochs 2 \
  --mini-batch-size 2048 \
  --lr 0.0003 \
  --entropy-coef 0.02 \
  --candidate-set-context \
  --dynamic-action-features \
  --terminal-reward-mode win-loss \
  --init-checkpoint runs/rust_ppo/candidate_context_single_learner_seed1442_checkpoints/batch_000100.pt \
  --controller-assignment table-profile \
  --learner-weight 0.55 \
  --random-weight 0.0 \
  --greedy-weight 0.20 \
  --smart-weight 0.20 \
  --checkpoint-opponent-weight 0.05 \
  --seed 2442 \
  --checkpoint-opponent-dir runs/rust_ppo/terminal_credit_league_500_seed442_v2_checkpoints \
  --checkpoint-opponent-limit 4 \
  --checkpoint-opponent-stride 25 \
  --eval-interval 25 \
  --eval-games 512 \
  --eval-num-envs 128 \
  --checkpoint-interval 25 \
  --logging-mode max \
  --device cpu \
  --metrics-path runs/rust_ppo/dynamic_action_winloss_table_profiles_seed2442_metrics.jsonl \
  --checkpoint-dir runs/rust_ppo/dynamic_action_winloss_table_profiles_seed2442_checkpoints
```

Let this run at least `100` batches before judging it. The old
`dynamic_action_context_seed2442` path is intentionally not reused because it
contains a config-only row from the killed dense-transfer attempt.

Launched at `2026-05-19 20:56 PDT`:
`runs/rust_ppo/dynamic_action_winloss_table_profiles_seed2442_metrics.jsonl`.
Batch `1` completed in `58.5s` with `43,325` learner samples, no candidate
truncation, KL `0.018`, clip fraction `0.087`, entropy `0.178`, and value EV
`0.383`. Controller counts confirm the intended table-profile mixture:
learner/self-play turns dominate while greedy, smart, and checkpoint profile
tables are present.

Batch `25` eval: greedy `0.565`, smart `0.373`, combined `0.938`; random
`0.839`. This does not meet the final target yet (`greedy > 0.6`, `smart >
0.4`), but it is a meaningful lift over the candidate-context batch-100 eval
(`greedy 0.546`, `smart 0.353`) and is the strongest combined score observed so
far. Training health remains stable: no truncation, KL `0.0105`, clip fraction
`0.062`, entropy `0.215`, value EV `0.459`. Continue to at least batch `100`
before judging, per the earlier rule.

Batch `50` eval: greedy `0.554`, smart `0.374`, combined `0.928`; random
`0.830`. Smart is essentially flat from batch `25`, and greedy slipped below
the batch-25 value, but this is still within noise and above the previous
candidate-context plateau on smart. Health remains good: no truncation, KL
`0.0099`, clip fraction `0.073`, entropy `0.255`, value EV `0.466`. Continue to
batch `100` before changing the run.

Batch `75` eval: greedy `0.567`, smart `0.377`, combined `0.944`; random
`0.822`. This is still below the final thresholds, but it is the best combined
score for this run and smart continues to inch upward. Health remains good: no
truncation, KL `0.0088`, clip fraction `0.070`, entropy `0.272`, value EV
`0.469`. Continue to batch `100`; if smart remains under `0.39` and greedy under
`0.58` at batch `100`, the next adjustment should likely increase direct
all-smart/all-greedy profile pressure or add opponent-threat features rather
than just run longer.

Policy update from review: do not benchmark-max by training only on greedy and
smart. Self-play should remain the core because the objective is a generally
strong model. Batch `100` is a trend check, not an automatic stop. If the run is
still improving at batch `100`, let it continue until the eval trend genuinely
stagnates; only then consider changes such as more targeted profile pressure,
reward/features changes, or opponent-threat features.

Batch `100` eval: greedy `0.580`, smart `0.406`, combined `0.986`; random
`0.815`. Smart has crossed the `0.4` target for the first time in this campaign,
and greedy is still improving but below the `0.6` target. Health remains good:
no truncation, KL `0.0086`, clip fraction `0.070`, entropy `0.278`, value EV
`0.460`. This is not a stagnation point; continue the same self-play-core run
until greedy either crosses `0.6` or the trend clearly stalls.

Batch `125` eval: greedy `0.587`, smart `0.420`, combined `1.007`; random
`0.832`. Smart improved further and greedy is closer to the `0.6` target but
not there yet. Health remains acceptable: no truncation, KL `0.0092`, clip
fraction `0.068`, entropy `0.252`, value EV `0.459`. Since both target metrics
are still improving from batch `100`, keep the run going.

Policy update from review: even after the milestone is first hit, do not stop
the run immediately. Record the milestone, but keep training until the eval
trend clearly flatlines or PPO stability breaks.

Batch `150` eval: greedy `0.590`, smart `0.428`, combined `1.018`; random
`0.842`. Both target metrics improved from batch `125`, and smart is now
comfortably above the `0.4` target. Greedy remains just below `0.6`, but the
trend is still positive. Health remains good: no truncation, KL `0.0085`, clip
fraction `0.066`, entropy `0.258`, value EV `0.478`. Continue.

Batch `175` eval: greedy `0.576`, smart `0.418`, combined `0.994`; random
`0.858`. This is a regression from batch `150`, but not enough by itself to
declare flatline. Health remains good: no truncation, KL `0.0088`, clip
fraction `0.066`, entropy `0.261`, value EV `0.484`. Treat as a watch point and
continue to batch `200`.

Batch `200` eval: greedy `0.617`, smart `0.441`, combined `1.059`; random
`0.859`. This is the first checkpoint to clear both requested milestone
thresholds (`greedy > 0.6`, `smart > 0.4`). Per the latest instruction, this is
not a stopping point: keep training until the eval trend clearly flatlines.
Health remains good: no truncation, KL `0.0082`, clip fraction `0.064`, entropy
`0.284`, value EV `0.477`. The milestone is also broad enough to count as more
than narrow benchmark-maxing for now: random remains strong, smart improved, and
greedy finally moved through the threshold while self-play stays the dominant
training profile.

Batch `225` eval: greedy `0.610`, smart `0.438`, combined `1.048`; random
`0.844`. This is slightly below the batch-200 peak but still confirms the new
post-milestone band rather than reverting to the batch-175 dip. Health remains
good: no truncation, KL `0.0091`, clip fraction `0.070`, entropy `0.270`, value
EV `0.471`. Continue to batch `250`; one small pullback from the peak is not a
flatline.

Batch `250` eval: greedy `0.630`, smart `0.431`, combined `1.061`; random
`0.863`. This is not flat: greedy reached a new high, random recovered, and the
combined greedy+smart score slightly exceeded the batch-200 high despite smart
pulling back from `0.441` to `0.431`. Health remains good and actually improved
on value fit: no truncation, KL `0.0077`, clip fraction `0.067`, entropy
`0.289`, value EV `0.503`. Continue to batch `275`; current evidence supports
letting the self-play-core recipe run.

Batch `275` eval: greedy `0.623`, smart `0.440`, combined `1.063`; random
`0.861`. The run is still not flat: combined score made another small high,
smart recovered from batch `250`, and greedy stayed above `0.62`. Health
remains stable: no truncation, KL `0.0087`, clip fraction `0.066`, entropy
`0.269`, value EV `0.451`. Let the configured run reach batch `300`; if batch
`300` is still in this band or better, resume past `300` rather than stopping.

Batch `300` eval: greedy `0.634`, smart `0.458`, combined `1.092`; random
`0.842`. This is a clear new high on the target score, with both greedy and
smart improving from batch `275`. Health remains stable: no truncation, KL
`0.0084`, clip fraction `0.062`, entropy `0.268`, value EV `0.483`. The run is
not flat, so the correct action is to continue. A `--resume --batches 500`
continuation was started from the same checkpoint directory and metrics path so
optimizer state is preserved and training continues at batch `301`.

Batch `325` eval after resume: greedy `0.629`, smart `0.458`, combined `1.088`;
random `0.849`. This is a slight pullback from the batch-300 combined high but
still holds the same post-300 band, with smart essentially unchanged and greedy
still above `0.62`. Health remains good: no truncation, KL `0.0069`, clip
fraction `0.057`, entropy `0.274`, value EV `0.476`. Continue to batch `350`;
this is not enough evidence for a flatline.

Batch `350` eval: greedy `0.636`, smart `0.467`, combined `1.103`; random
`0.859`. This is a new high on both the combined target score and smart win
rate, while greedy stays at the top of its band. Health remains good: no
truncation, KL `0.0067`, clip fraction `0.057`, entropy `0.280`, value EV
`0.468`. Continue to batch `375`; the run is still improving after the resume.

Batch `375` eval: greedy `0.629`, smart `0.462`, combined `1.091`; random
`0.863`. This pulled back from the batch-350 high but remains in the post-300
high band and keeps smart above `0.46`. Health remains good: no truncation, KL
`0.0087`, clip fraction `0.064`, entropy `0.266`, value EV `0.479`. Treat this
as a watch point, not a clear flatline; continue to batch `400`.

Batch `400` eval: greedy `0.623`, smart `0.457`, combined `1.079`; random
`0.849`. This is a second pullback from the batch-350 high, so plateau risk is
now real. It is still above the batch-300 target band and PPO health remains
good: no truncation, KL `0.0085`, clip fraction `0.060`, entropy `0.270`, value
EV `0.481`. Continue to batch `425` before declaring flatline; if batch `425`
fails to recover meaningfully, the run is likely entering a high-but-flat band.

Batch `425` eval: greedy `0.635`, smart `0.469`, combined `1.104`; random
`0.844`. This recovered from the batch-400 pullback and slightly exceeded the
batch-350 combined high, with a new smart high. Health remains acceptable: no
truncation, KL `0.0079`, clip fraction `0.063`, entropy `0.270`, value EV
`0.462`. The run has not clearly flatlined; continue to batch `450`.

Batch `450` eval: greedy `0.640`, smart `0.475`, combined `1.114`; random
`0.854`. This is another clear high on greedy, smart, and the combined target
score. Health remains acceptable: no truncation, KL `0.0092`, clip fraction
`0.068`, entropy `0.271`, value EV `0.471`. The earlier batch-375/400 dip was
noise rather than flatline. Continue to batch `475`.

Batch `475` eval: greedy `0.626`, smart `0.466`, combined `1.093`; random
`0.854`. This is a pullback from the batch-450 high but still inside the
post-300 high band. Health remains good: no truncation, KL `0.0077`, clip
fraction `0.060`, entropy `0.284`, value EV `0.474`. Continue to the configured
batch `500` endpoint before deciding whether to resume again; if batch `500`
does not recover, the evidence will favor a high-band plateau.

Batch `500` eval: greedy `0.650`, smart `0.460`, combined `1.110`; random
`0.862`. This did not beat the batch-450 combined high because smart pulled
back, but greedy reached a new high and random recovered to a new high. Health
remains good: no truncation, KL `0.0086`, clip fraction `0.063`, entropy
`0.270`, value EV `0.472`. This is not a clear flatline yet, but the post-350
band is now narrow enough that plateau risk is high. Continue one more resume
segment and watch batches `525`, `550`, and `575`; if they all stay in roughly
the same `1.09-1.11` combined band with no new high, the run will have stronger
flatline evidence.

Batch `525` eval: greedy `0.654`, smart `0.466`, combined `1.120`; random
`0.861`. This is a new combined high, driven mostly by another greedy-matchup
gain while smart remains in the same high band below the batch-450 smart peak.
Health remains stable: no truncation, KL `0.0081`, clip fraction `0.061`,
entropy `0.258`, value EV `0.486`. Because the combined score made a fresh
high after batch `500`, this is not a flatline. Continue to batch `550`; the
next question is whether smart can recover above `0.475` or whether only the
greedy matchup is still improving.

Batch `550` eval: greedy `0.680`, smart `0.499`, combined `1.179`; random
`0.887`. This is a large new high, not merely noise inside the old band. Both
greedy and smart improved, and smart finally crossed the `~0.50` line over
2048 eval games. Health remains good: no truncation, KL `0.0071`, clip
fraction `0.055`, entropy `0.272`, value EV `0.473`. The right action is to
keep training unchanged. The prior plateau hypothesis is now rejected for this
run unless later evals give back the gain and stay flat for several checkpoints.

Batch `575` eval: greedy `0.645`, smart `0.467`, combined `1.112`; random
`0.861`. This gave back the batch-550 jump and returned to the previous
high-but-flat band. Treat this as a watch point, not a stop signal: batch `550`
was far enough above prior evals that one pullback could be eval variance or a
temporary oscillation. PPO health remains stable: no truncation, KL `0.0073`,
clip fraction `0.057`, entropy `0.272`, value EV `0.490`. Continue to batch
`600`; if `600` and `625` both remain in the old band without recovering toward
batch `550`, then the run likely plateaued after a transient spike.

Batch `600` eval: greedy `0.668`, smart `0.473`, combined `1.141`; random
`0.891`. This partially recovered from batch `575` and stays above the old
pre-550 band, but it did not retake the batch-550 high. Health remains
acceptable: no truncation, KL `0.0086`, clip fraction `0.061`, entropy `0.257`,
value EV `0.468`. Continue to batch `625`. If batch `625` stays around
`1.11-1.14` with smart below `0.50`, the run is likely settling into a new
high-band plateau below the batch-550 spike; if it recovers toward `1.18`,
continue unchanged.

Batch `625` eval: greedy `0.653`, smart `0.481`, combined `1.135`; random
`0.859`. This is a second eval after the batch-550 spike that failed to retake
the high, so plateau evidence is now meaningful. It is still not an optimizer
failure: no truncation, KL `0.0072`, clip fraction `0.058`, entropy `0.253`,
value EV `0.468`. Continue to the configured batch `650` endpoint before
calling it flat. If batch `650` remains in the `1.11-1.14` band, the best
interpretation is a new high-band plateau with batch `550` as the best
checkpoint so far.

Batch `650` eval: greedy `0.646`, smart `0.475`, combined `1.120`; random
`0.847`. The run has now clearly flattened after the batch-550 spike: batches
`575`, `600`, `625`, and `650` stayed in the `1.11-1.14` combined band and did
not recover toward `1.179`. Health remained stable through the endpoint:
no truncation, KL `0.0074`, clip fraction `0.058`, entropy `0.256`, value EV
`0.469`. Best checkpoint for this run is batch `550`:
`runs/rust_ppo/dynamic_action_winloss_table_profiles_seed2442_checkpoints/batch_000550.pt`.
Stop this recipe here; further progress likely needs a new intervention rather
than more unchanged training.

## 2026-05-16 Update: Current System Direction

The terminal-credit rollout fix has been promoted into the active
`vikrams-rust` checkout. The fix credits each learner-controlled seat's final
terminal reward to that seat's latest learner record in the episode, rather
than only rewarding the learner if it happened to take the terminal action. The
targeted regression test for this behavior is now part of the Rust PPO test
suite.

After promoting the fix, we ran a longer fresh seed-442 continuation-style
experiment with the same default opponent mix and PPO settings:

- metrics: `runs/rust_ppo/terminal_credit_500_seed442_metrics.jsonl`
- checkpoints: `runs/rust_ppo/terminal_credit_500_seed442_checkpoints`
- settings: `lr=3e-4`, `entropy_coef=0.01`, `num_envs=256`,
  `rollout_steps=128`, `max_candidates=256`, default mix
  learner/random/greedy/smart = `0.75/0.10/0.10/0.05`

Latest observed eval checkpoints from that run:

| batch | greedy wr | smart wr | smart+greedy | entropy | action prob mean | action prob p10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 225 | 0.500 | 0.325 | 0.825 | 0.197 | 0.888 | 0.531 |
| 250 | 0.532 | 0.332 | 0.864 | 0.196 | 0.886 | 0.527 |
| 275 | 0.511 | 0.311 | 0.821 | 0.199 | 0.888 | 0.535 |
| 300 | 0.515 | 0.321 | 0.836 | 0.196 | 0.890 | 0.544 |
| 325 | 0.494 | 0.347 | 0.841 | 0.180 | 0.896 | 0.560 |
| 350 | 0.518 | 0.335 | 0.853 | 0.183 | 0.897 | 0.568 |
| 375 | 0.510 | 0.306 | 0.815 | 0.169 | 0.903 | 0.608 |
| 400 | 0.515 | 0.354 | 0.868 | 0.180 | 0.899 | 0.581 |
| 425 | 0.503 | 0.331 | 0.834 | 0.163 | 0.906 | 0.619 |

Interpretation: the longer run has not collapsed, and batch `400` is a new
best observed checkpoint (`smart+greedy=0.868`). However, the improvement is
not monotonic and the run is now oscillating in roughly the `0.82-0.87` target
band. PPO internals remain healthy, but the policy has become highly decisive:
action probability mean is near `0.90`, and action probability p10 has risen
above `0.60` by batch `425`. This supports the plateau diagnosis: the optimizer
is stable, the critic is not obviously failing, but exploration and opponent
diversity are likely too limited after the policy becomes competent.

## What We Changed Next

To attack that plateau, the training system now has two new control mechanisms:

1. **Entropy controls.** The CLI exposes `--entropy-coef`, plus optional linear
   entropy scheduling via `--entropy-schedule linear`,
   `--entropy-start-coef`, `--entropy-end-coef`, and
   `--entropy-schedule-batches`.
2. **Eval-gated curriculum with checkpoint league play.** The
   `--curriculum smart-greedy` mode gates phases on the most recent eval
   `greedy_win_rate + smart_win_rate`. It changes entropy, fixed opponent mix,
   and checkpoint-opponent weight only after eval checkpoints. Metrics rows log
   `training_controls`, including active phase, active opponent mix, active
   entropy, checkpoint pool size, and checkpoint paths.

Current `smart-greedy` curriculum:

| phase | trigger | learner | random | greedy | smart | checkpoint | entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | no eval or `<0.70` | 0.75 | 0.10 | 0.10 | 0.05 | 0.00 | configured, normally 0.010 |
| targeted | `>=0.70` | 0.65 | 0.05 | 0.10 | 0.10 | 0.10 | 0.012 |
| challenge | `>=0.80` | 0.60 | 0.05 | 0.10 | 0.10 | 0.15 | 0.015 |
| plateau_breaker | `>=0.84` | 0.50 | 0.05 | 0.10 | 0.10 | 0.25 | 0.020 |

`--curriculum smart-greedy` now defaults to the known-good base mix above, so
the user does not need to pass learner/random/greedy/smart weights unless they
want to override the preset. `--checkpoint-dir` is where the current run writes
checkpoints. `--checkpoint-opponent-dir` is optional and only needed when the
run should train against a different checkpoint source; otherwise the checkpoint
opponent pool defaults to `--checkpoint-dir` and self-refreshes after each
checkpoint save.

Recommended next run:

```sh
/Users/vikramsingh/Desktop/coding/big2/.venv/bin/python -m big2.training.rust_ppo.run \
  --train \
  --batches 500 \
  --num-envs 256 \
  --rollout-steps 128 \
  --max-candidates 256 \
  --ppo-epochs 2 \
  --mini-batch-size 2048 \
  --lr 0.0003 \
  --entropy-coef 0.01 \
  --curriculum smart-greedy \
  --seed 442 \
  --checkpoint-opponent-limit 4 \
  --checkpoint-opponent-stride 25 \
  --eval-interval 25 \
  --eval-games 256 \
  --eval-num-envs 64 \
  --checkpoint-interval 25 \
  --logging-mode max \
  --device cpu \
  --metrics-path runs/rust_ppo/terminal_credit_league_500_seed442_metrics.jsonl \
  --checkpoint-dir runs/rust_ppo/terminal_credit_league_500_seed442_checkpoints
```

Hypothesis for this next run: once the policy crosses the `0.70`, `0.80`, and
`0.84` smart+greedy thresholds, the curriculum will increase target-opponent
pressure, add frozen checkpoint opponents, and raise entropy enough to reduce
deterministic local convergence without destabilizing PPO.

Standard run card fields:

- Evidence: last batch, eval count, truncation, candidate max.
- Outcome: best and latest smart+greedy win-rate score, greedy/smart split, reward split, trend per 25 batches.
- PPO health: last-25 KL, clip fraction, entropy, value loss, total objective, value explained variance.
- Rollout health: pass rate, rollout terminal reward, throughput.
- Decision: promote, continue, reject, or exclude.

Charts:

- `eval_progress.png`: target score, greedy win rate, smart win rate.
- `training_health.png`: KL, clipping, entropy, value fit, value loss, total PPO objective.
- `rollout_diagnostics.png`: rollout reward, pass rate, candidates, throughput.

## Ranked Summary

| rank | run | evidence | stability | best | last | best batch | trend/25 | greedy last | smart last | KL25 | clip25 | entropy25 | EV25 | value loss delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `terminal_credit_225_resume` | strong | watch | 0.823 | 0.823 | 225 | 0.0217 | 0.514 | 0.310 | 0.0091 | 0.0599 | 0.207 | 0.430 | 0.014 |
| 2 | `terminal_credit_seed1442_75_512cand` | early | watch | 0.760 | 0.760 | 75 | 0.0703 | 0.480 | 0.279 | 0.0098 | 0.0810 | 0.304 | 0.418 | 0.004 |
| 3 | `low_lr_150_resume` | medium | stable | 0.722 | 0.722 | 150 | 0.0159 | 0.470 | 0.252 | 0.0031 | 0.0349 | 0.207 | 0.267 | 0.044 |
| 4 | `terminal_smart_focus_100` | medium | watch | 0.706 | 0.706 | 75 | 0.0488 | 0.432 | 0.274 | 0.0098 | 0.0745 | 0.267 | 0.417 | -0.022 |
| 5 | `terminal_low_lr_75` | early | stable | 0.692 | 0.692 | 75 | 0.0420 | 0.435 | 0.258 | 0.0029 | 0.0374 | 0.288 | 0.431 | 0.007 |
| 6 | `existing_baseline_seed42` | medium | watch | 0.677 | 0.677 | 150 | 0.0508 | 0.406 | 0.271 | 0.0096 | 0.0581 | 0.189 | 0.269 | 0.117 |
| 7 | `baseline_repro_75_v2` | early | watch | 0.629 | 0.629 | 75 | 0.0850 | 0.398 | 0.230 | 0.0098 | 0.0792 | 0.298 | 0.255 | 0.111 |
| 8 | `heuristic_mix_75_v2` | weak | watch | 0.416 | 0.416 | 50 | 0.0635 | 0.258 | 0.158 | 0.0100 | 0.0910 | 0.415 | 0.207 | 0.041 |

## Run Cards

### `terminal_credit_225_resume`

- Decision: **promote/continue**. Evidence `strong`, stability `watch`; last batch `225`, eval rows `9`, truncation `0`.
- Outcome: best smart+greedy `0.823` at batch `225`, latest `0.823`, trend `0.0217` per 25 batches.
- Latest split: greedy WR `0.514` / reward `0.390`; smart WR `0.310` / reward `0.089`.
- PPO health last 25: KL `0.0091` with max `0.0194`, clip `0.0599`, entropy `0.207`, value EV `0.430`, value-loss change `0.014`, total-loss change `0.308`.
- Rollout health: pass rate `0.482`, terminal reward `0.028`, candidate max `145`, mean batch seconds `43.0`.

### `terminal_credit_seed1442_75_512cand`

- Decision: **promote/continue**. Evidence `early`, stability `watch`; last batch `75`, eval rows `3`, truncation `0`.
- Outcome: best smart+greedy `0.760` at batch `75`, latest `0.760`, trend `0.0703` per 25 batches.
- Latest split: greedy WR `0.480` / reward `0.343`; smart WR `0.279` / reward `0.060`.
- PPO health last 25: KL `0.0098` with max `0.0253`, clip `0.0810`, entropy `0.304`, value EV `0.418`, value-loss change `0.004`, total-loss change `0.193`.
- Rollout health: pass rate `0.483`, terminal reward `0.033`, candidate max `139`, mean batch seconds `50.7`.

### `low_lr_150_resume`

- Decision: **secondary**. Evidence `medium`, stability `stable`; last batch `150`, eval rows `6`, truncation `0`.
- Outcome: best smart+greedy `0.722` at batch `150`, latest `0.722`, trend `0.0159` per 25 batches.
- Latest split: greedy WR `0.470` / reward `0.312`; smart WR `0.252` / reward `-0.007`.
- PPO health last 25: KL `0.0031` with max `0.0066`, clip `0.0349`, entropy `0.207`, value EV `0.267`, value-loss change `0.044`, total-loss change `0.346`.
- Rollout health: pass rate `0.462`, terminal reward `0.009`, candidate max `143`, mean batch seconds `46.1`.

### `terminal_smart_focus_100`

- Decision: **reject vs terminal default**. Evidence `medium`, stability `watch`; last batch `100`, eval rows `4`, truncation `0`.
- Outcome: best smart+greedy `0.706` at batch `75`, latest `0.706`, trend `0.0488` per 25 batches.
- Latest split: greedy WR `0.432` / reward `0.282`; smart WR `0.274` / reward `0.058`.
- PPO health last 25: KL `0.0098` with max `0.0308`, clip `0.0745`, entropy `0.267`, value EV `0.417`, value-loss change `-0.022`, total-loss change `0.138`.
- Rollout health: pass rate `0.466`, terminal reward `0.030`, candidate max `252`, mean batch seconds `33.4`.

### `terminal_low_lr_75`

- Decision: **reject combination**. Evidence `early`, stability `stable`; last batch `75`, eval rows `3`, truncation `0`.
- Outcome: best smart+greedy `0.692` at batch `75`, latest `0.692`, trend `0.0420` per 25 batches.
- Latest split: greedy WR `0.435` / reward `0.301`; smart WR `0.258` / reward `0.034`.
- PPO health last 25: KL `0.0029` with max `0.0063`, clip `0.0374`, entropy `0.288`, value EV `0.431`, value-loss change `0.007`, total-loss change `0.094`.
- Rollout health: pass rate `0.447`, terminal reward `0.032`, candidate max `141`, mean batch seconds `34.8`.

### `existing_baseline_seed42`

- Decision: **reference**. Evidence `medium`, stability `watch`; last batch `174`, eval rows `6`, truncation `0`.
- Outcome: best smart+greedy `0.677` at batch `150`, latest `0.677`, trend `0.0508` per 25 batches.
- Latest split: greedy WR `0.406` / reward `0.241`; smart WR `0.271` / reward `0.040`.
- PPO health last 25: KL `0.0096` with max `0.0328`, clip `0.0581`, entropy `0.189`, value EV `0.269`, value-loss change `0.117`, total-loss change `0.751`.
- Rollout health: pass rate `0.469`, terminal reward `0.017`, candidate max `253`, mean batch seconds `23.5`.

### `baseline_repro_75_v2`

- Decision: **reference**. Evidence `early`, stability `watch`; last batch `75`, eval rows `3`, truncation `0`.
- Outcome: best smart+greedy `0.629` at batch `75`, latest `0.629`, trend `0.0850` per 25 batches.
- Latest split: greedy WR `0.398` / reward `0.236`; smart WR `0.230` / reward `-0.028`.
- PPO health last 25: KL `0.0098` with max `0.0343`, clip `0.0792`, entropy `0.298`, value EV `0.255`, value-loss change `0.111`, total-loss change `1.084`.
- Rollout health: pass rate `0.478`, terminal reward `0.016`, candidate max `139`, mean batch seconds `57.3`.

### `heuristic_mix_75_v2`

- Decision: **reject**. Evidence `weak`, stability `watch`; last batch `61`, eval rows `2`, truncation `0`.
- Outcome: best smart+greedy `0.416` at batch `50`, latest `0.416`, trend `0.0635` per 25 batches.
- Latest split: greedy WR `0.258` / reward `-0.113`; smart WR `0.158` / reward `-0.303`.
- PPO health last 25: KL `0.0100` with max `0.0245`, clip `0.0910`, entropy `0.415`, value EV `0.207`, value-loss change `0.041`, total-loss change `0.389`.
- Rollout health: pass rate `0.467`, terminal reward `0.002`, candidate max `146`, mean batch seconds `54.0`.

## Excluded Or Weak Runs

- `baseline_repro_75`: evidence `no eval`, stability `too short`, last batch `1`, eval rows `0`, truncation `0`.
- `launch_debug`: evidence `no eval`, stability `too short`, last batch `1`, eval rows `0`, truncation `0`.
- `low_lr_75`: evidence `no eval`, stability `too short`, last batch `1`, eval rows `0`, truncation `0`.
- `heuristic_mix_75`: evidence `no eval`, stability `too short`, last batch `1`, eval rows `0`, truncation `0`.
- `heuristic_mix_75_v2`: evidence `weak`, stability `watch`, last batch `61`, eval rows `2`, truncation `0`.
- `terminal_credit_seed1442_75`: evidence `exclude`, stability `invalid: truncation`, last batch `24`, eval rows `0`, truncation `1`.

## Interpretation Rules

- PPO total loss and policy loss are not expected to decrease monotonically; they are moving objectives under a changing policy/data distribution.
- Value loss is useful only with value explained variance. A run can show higher value loss because the reward signal is stronger or less sparse, while still fitting value targets better.
- Stable PPO here means KL mostly below about `0.03`, clip fraction mostly below about `0.20`, no candidate truncation, entropy declining gradually rather than collapsing, and value EV improving or staying positive.
- Eval differences below roughly 2-3 percentage points are within the noise of a 1024-game aggregate; persistent direction over multiple checkpoints matters more.
