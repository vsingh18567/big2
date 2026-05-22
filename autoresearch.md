# Autoresearch big2

## Goal
Build the strongest possible RL-powered Big 2 bot using the existing PyTorch PPO trainer and Rust simulator. The current exploratory run writes max-logging metrics to `runs/big2_v2/train_metrics_256env_128cand.jsonl`; use that file to understand the baseline behavior, available metrics, eval cadence, and runtime. Read docs/wiki to understand the whole system.

Primary objective: improve evaluation performance against the `smart` and `greedy` bots. Performance against `random` is useful as a sanity check, but it is not the optimization target.

Likely research directions include PPO hyperparameter tuning, opponent curriculum/mix changes, exploration settings, reward shaping, candidate/action representation changes, and model architecture changes. 

## Current Baseline Context
The current run configuration is roughly:

- `num_envs=256`
- `rollout_steps=128`
- `batches=500`
- `max_candidates=256`
- `ppo_epochs=2`
- `mini_batch_size=2048`
- `lr=3e-4`
- `entropy_coef=0.01`
- opponent mix: `learner=0.75`, `random=0.10`, `greedy=0.10`, `smart=0.05`
- eval every `25` batches
- eval uses `256` games per seat, `4` seats, so `1024` aggregate games per opponent

Although docs/wiki recommends `max_candidates=2048` for serious runs, the current `256`-candidate baseline showed `truncated_candidate_lists=0` through the observed batches. Keep `256` for exploratory A/B tests unless truncation appears, because it preserves the measured throughput. If any run logs nonzero truncation, treat that run as candidate-limited and rerun the hypothesis with a larger candidate width. 

Observed runtime from the current CPU run:

- normal training batch: about `20-21s`
- eval/checkpoint batch: about `100-105s`
- 25 batches: roughly `10-12m`
- 75 batches: roughly `30m`
- 100 batches: roughly `40m`
- 150 batches: roughly `60m`
- 225 batches: roughly `90m`

Use these numbers to plan experiments. A standard exploratory test should usually run up to about `90m`, which is approximately `225` batches and `9` eval checkpoints. Shorter `75-150` batch tests are still appropriate for risky ideas, slow configurations, or hypotheses that are clearly answered early.

## Metrics Available
Each JSONL metrics file includes batch-level training diagnostics and periodic eval results.

Use Python scripts to analyze JSONL metrics instead of manually eyeballing long logs. Write small repeatable scripts or one-off Python snippets to extract eval checkpoints, aggregate smart/greedy/random results, summarize training diagnostics by batch window, compare runs, and produce ranking tables. Keep any useful analysis script in the repo or paste the command into the research log so results are reproducible.

Primary eval metrics:

- `eval.smart.aggregate.win_rate`
- `eval.smart.aggregate.average_reward`
- `eval.greedy.aggregate.win_rate`
- `eval.greedy.aggregate.average_reward`

Secondary eval metrics:

- `eval.random.aggregate.win_rate`
- per-seat win rates under `eval.<opponent>.seats`
- confidence interval fields: `win_rate_ci95_low`, `win_rate_ci95_high`
- episode lengths and truncated candidate counts

Training diagnostics to watch:

- `ppo.approx_kl`
- `ppo.clip_fraction`
- `ppo.value_explained_variance`
- `ppo.return_mean`
- `entropy`
- `ppo.action_probability_mean`
- `rollout.pass_rate`
- `rollout.episode_length_mean`
- `rollout.terminal_reward_mean`
- `candidate_count_mean`, `candidate_count_max`, and `truncated_candidate_lists`
- timing fields under `timing`

Do not rely on rollout win/terminal stats alone. Treat periodic eval results as the main source of truth.

## Procedure
Use `3-5` concurrent git worktrees to test multiple hypotheses in parallel, but avoid overloading the machine. If training throughput drops badly or jobs interfere with each other, reduce concurrency.

Run up to `20` hypotheses overall. Most tests should run for `75-225` batches so they include at least `3-9` eval checkpoints. Use the longer end of that range for promising hypotheses and the shorter end for risky, slow, or clearly underperforming ideas. Do not run a full `500` batch or `12h` experiment unless a shorter run has already shown a convincing signal and the user approves the longer run.

Target stop time: `2026-05-16 08:30 America/Los_Angeles` is the latest delivery time, not a requirement to keep running experiments. If the evidence is already strong enough before then, stop early and present the best hypothesis. Otherwise, stop launching new experiments by `2026-05-16 07:45 America/Los_Angeles`, finish or terminate any remaining short runs by `2026-05-16 08:15 America/Los_Angeles`, and have the best hypothesis plus supporting evidence ready by `2026-05-16 08:30 America/Los_Angeles`. The final deliverable should include the recommended next configuration, the strongest completed checkpoint/run, and a concise ranking of the tested hypotheses.

Before comparing new ideas, establish a reference from the existing metrics file and, if practical, run one short reproduction of the current config using the same batch count as the experiments. Use the same eval cadence and logging mode so comparisons are apples-to-apples.

This research process should run autonomously. Do not ask the user for help choosing hypotheses, commands, worktree names, logging formats, or next steps. Make reasonable decisions from the repo, current metrics, and experiment results. If something fails, debug it locally, record the failure, and either fix it or move to the next hypothesis. Only stop for user input if continuing would require destructive actions, external credentials, unavailable hardware, or a genuinely ambiguous operation that could damage existing work.

Feel free to do long sleeps while waiting.

## Subagent Strategy
Use subagents deliberately to increase throughput. Treat each subagent as the owner of one hypothesis or one bounded analysis task. Use gpt 5.5 with high effort for any subagents that are doing any creative work, medium effort for executing on a hypothesis / other simpler tasks. 

Good subagent assignments:

- inspect the existing trainer/config code and propose a small set of plausible PPO or curriculum hypotheses
- implement and run one isolated experiment in one dedicated worktree
- monitor one metrics file and summarize eval checkpoints and training diagnostics
- compare finished runs and produce a ranked table of results
- investigate a failed or unstable run without blocking other experiments

When using subagents for experiments:

1. Give each subagent a specific hypothesis, worktree/branch name, metrics path, and maximum batch/time budget.
2. Tell each subagent they are not alone in the repo and must not revert unrelated changes.
3. Require each subagent to write results into `autoresearch_results.md` or a clearly named per-run note that can be merged into it.
4. Keep write ownership disjoint when multiple subagents edit code.
5. Prefer parallel read-only analysis subagents when deciding the first wave of hypotheses.
6. Prefer parallel worker subagents for experiment execution once hypotheses are selected.
7. Do not duplicate the same hypothesis across subagents unless intentionally testing seed variance.

The main agent should coordinate the run queue, maintain the research log, compare results, and decide which hypotheses to continue, stop, or refine. Subagents should not wait for user approval for ordinary research decisions.

For each hypothesis:

1. Create or reuse a dedicated worktree/branch.
2. Change one main idea at a time.
3. Record the exact command, config diff, seed, worktree, output metrics path, and checkpoint path.
4. Use max logging mode.
5. Record `max_candidates` and verify `truncated_candidate_lists` stays at `0`; if not, rerun with a larger candidate width before drawing conclusions.
6. Evaluate primarily on `smart` and `greedy` aggregate win rate/reward.
7. Prefer hypotheses that improve both `smart` and `greedy`; flag tradeoffs explicitly.
8. Stop early if a run is clearly unstable, broken, much slower than expected, or much worse than baseline after multiple evals.
9. The main agent and subagents should feel comfortable killing bad runs. Do not keep spending compute on a run just because it was scheduled. Kill it, record the reason and latest metrics, and reallocate the slot to a better hypothesis.

## Decision Rules
Because evals are noisy, do not overfit to one checkpoint. Prefer trends across eval checkpoints.

Promote a hypothesis when it shows one of these:

- clear improvement in `smart` win rate or average reward without hurting `greedy`
- clear improvement in `greedy` win rate or average reward without hurting `smart`
- similar eval performance with much better stability, speed, or sample efficiency
- a diagnostic improvement that plausibly unlocks later gains, such as healthier entropy or value explained variance

Be skeptical of:

- better rollout rewards without eval gains
- better random-bot performance only
- one-seat improvements with worse aggregate performance
- changes that only help by increasing runtime substantially
- rapid entropy collapse without smart/greedy improvement

Kill or abandon a run when:

- it is clearly below baseline on both `smart` and `greedy` after multiple eval checkpoints
- diagnostics indicate instability, such as exploding KL, extreme clip fraction, invalid metrics, or repeated failures
- runtime is much worse than expected, even if performance is not yet conclusive
- the hypothesis has already been answered and more batches are unlikely to change the decision

## Research Log
Maintain a markdown research log, preferably `autoresearch_results.md`.

For each run, record:

- hypothesis name
- branch/worktree
- code/config changes
- training command
- seed
- start/end time and batch count
- metrics file path
- checkpoint path
- eval table for random/greedy/smart at each eval checkpoint
- key diagnostics: KL, clip fraction, entropy, value explained variance, pass rate, episode length
- conclusion: promote, reject, continue longer, or follow-up hypothesis

At the end, summarize the top candidates and recommend the next long run configuration.

## Focus
Don't execute commands that aren't tied to this goal.