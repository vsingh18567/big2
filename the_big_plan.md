## Full plan: Big 2 RL agent with Rust env + PyTorch PPO

Build it as:

```text
Rust:
  Big 2 rules engine
  vectorized self-play environment
  legal action generation
  candidate action lists
  terminal scoring
  fast reset/step API

Python/PyTorch:
  neural network
  PPO rollout buffer
  PPO update loop
  checkpoint opponent pool
  evaluation harness
  logging/analysis
```

Use **PyO3 + maturin** to expose the Rust environment as a Python module. PyO3 is designed for creating native Python modules from Rust, and maturin is the standard low-friction way to build Rust-based Python packages. ([PyO3][1])

---

# 1. Core design choice

Do **not** make the policy output cards one-by-one.

Do **not** precompute full game states.

Do this instead:

```text
Enumerate all possible Big 2 moves once.
At runtime, Rust returns only the legal candidate move IDs for the current hand/state.
PyTorch scores those candidates and samples one.
```

The action space is fixed globally, but each turn only exposes a small legal subset.

Example:

```text
Global action catalog:
  0 = pass
  1 = single 3♦
  2 = single 3♣
  ...
  53 = pair 3♦3♣
  ...
  5000 = straight 3-4-5-6-7 with suits ...
  ...
```

At a specific turn, Rust returns:

```text
candidate_ids = [17, 91, 144, 2038, 2041, ...]
```

The network scores only those candidates.

This is better than producing a dense `num_actions` logit vector every turn, because Big 2’s valid 5-card combinations can make the global catalog fairly large.

---

# 2. Rust environment

## 2.1 Repository structure

Something like:

```text
big2-rust/
  Cargo.toml
  pyproject.toml

  src/
    lib.rs              # PyO3 bindings
    card.rs             # card/rank/suit constants
    action.rs           # MoveId, action catalog, metadata
    rules.rs            # move comparison, valid 5-card logic
    game.rs             # single GameState
    vec_env.rs          # batched environment
    obs.rs              # observation encoding
    rng.rs              # deterministic seeding
    tests.rs
```

Python side:

```text
trainer/
  train.py
  model.py
  ppo.py
  rollout.py
  eval.py
  bots.py
  config.py
```

---

# 3. Big 2 representation

## 3.1 Cards

Use a 52-bit integer.

```rust
type CardMask = u64;
type MoveId = u32;
```

Each card is one bit:

```text
bit 0  = 3♦
bit 1  = 3♣
bit 2  = 3♥
bit 3  = 3♠
...
bit 51 = 2♠
```

Pick one canonical ordering and never change it.

A hand is:

```rust
hand_mask: u64
```

A move is also:

```rust
move_mask: u64
```

Runtime legality starts with:

```rust
(move_mask & hand_mask) == move_mask
```

---

## 3.2 Game state

For each game:

```rust
struct GameState {
    hands: [u64; 4],
    current_player: u8,

    last_move_id: Option<MoveId>,
    last_actor: Option<u8>,

    passed: [bool; 4],
    cards_played: u64,

    cards_remaining: [u8; 4],
    finished: [bool; 4],

    is_first_turn: bool,
    done: bool,

    rng_state: u64,
}
```

Do not expose opponents’ hands in observations. Rust can store them internally, but the policy must only see legal public information plus its own hand.

---

# 4. Action catalog

At Rust startup, generate:

```rust
struct MoveMeta {
    id: MoveId,
    mask: u64,
    kind: MoveKind,
    num_cards: u8,
    primary_rank: u8,
    secondary_rank: u8,
    high_card: u8,
    high_suit: u8,
}
```

Where:

```rust
enum MoveKind {
    Pass,
    Single,
    Pair,
    Triple,
    Straight,
    Flush,
    FullHouse,
    FourOfKind,
    StraightFlush,
}
```

Precompute:

```text
ALL_MOVES: Vec<MoveMeta>
MOVE_BY_MASK: HashMap<u64, MoveId>
FIVE_CARD_MOVE_BY_MASK: HashMap<u64, MoveId>
```

You can also precompute comparison logic:

```text
beats(move_id, previous_move_id) -> bool
```

Either compute this from metadata at runtime or precompute bitsets. I would start with metadata comparison first, then optimize only if profiling says it matters.

---

## 4.1 Important rule decisions

Lock these down before training:

```text
Suit order
Rank order
Whether A-2-3-4-5 is a straight
Whether 2 can appear in straights
Whether the opening move must contain 3♦
Whether passing prevents re-entry until the trick resets
Whether the game ends when one player goes out or continues for full ranking
Scoring rules
```

These rule variants matter a lot. Put them in a `RulesConfig`.

Example:

```rust
struct RulesConfig {
    require_three_diamond_open: bool,
    allow_wheel_straight: bool,
    allow_two_in_straight: bool,
    game_ends_on_first_out: bool,
}
```

---

# 5. Legal action generation

Given:

```text
hand_mask
last_move_id
passed status
is_first_turn
current_player
```

Rust generates candidate move IDs.

## Free lead

If there is no active previous move:

```text
legal = all valid moves contained in hand
pass = illegal
```

If first move must include 3♦:

```text
legal = moves contained in hand AND move contains 3♦
```

## Responding to previous play

If there is an active previous move:

```text
legal = pass + moves contained in hand that beat previous move
```

If the player already passed in this trick, depending on your rules:

```text
legal = pass only
```

---

## 5.1 Candidate generation approach

For a 13-card hand, dynamically generate contained moves:

```text
singles: all cards in hand
pairs: rank groups with at least 2 suits
triples: rank groups with at least 3 suits
5-card hands: combinations from the 13-card hand, checked against FIVE_CARD_MOVE_BY_MASK
```

The worst case for 5-card enumeration is:

```text
C(13, 5) = 1287
```

That is fine in Rust.

Avoid Python doing any of this.

---

# 6. Rust/Python API

Expose one class:

```python
env = Big2VecEnv(
    num_envs=4096,
    seed=123,
    rules_config=...
)
```

Primary methods:

```python
obs, candidate_ids, candidate_mask = env.reset()

obs, candidate_ids, candidate_mask, done, final_rewards, info = env.step(action_ids)
```

Where:

```text
obs:            [num_envs, obs_dim]
candidate_ids: [num_envs, max_candidates]
candidate_mask:[num_envs, max_candidates]
done:           [num_envs]
final_rewards: [num_envs, 4]
```

The Python side selects one action per active game:

```python
action_ids: [num_envs]
```

Rust should validate actions in debug mode:

```rust
assert!(candidate_ids.contains(&action_id));
```

In release mode, you can still validate but avoid expensive checks if needed.

---

## 6.1 Candidate list shape

Use padded candidate lists:

```text
candidate_ids:
[
  [17, 91, 104, 2011, -1, -1, -1],
  [3,  55, 103, 7002, 8101, -1, -1],
]
```

And a boolean mask:

```text
candidate_mask:
[
  [1, 1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 1, 0, 0],
]
```

Choose a generous `max_candidates`, maybe:

```text
512 initially
1024 if needed
2048 if you want maximum safety
```

Log whenever a candidate list is truncated. It should never happen silently.

---

# 7. Observation design

Use perspective-normalized observations.

For the acting player, encode:

```text
own hand:                         52 bits
cards already played:             52 bits
last move metadata:               type/rank/suit/length
cards remaining for self:         1 number
cards remaining for left opp:     1 number
cards remaining for across opp:   1 number
cards remaining for right opp:    1 number
who passed in current trick:      4 bits, perspective-normalized
free lead/control flag:           1 bit
first turn flag:                  1 bit
relative position of last actor:  categorical
recent action history:            optional but recommended
```

I would include recent history early.

For example, last 16 public actions:

```text
for each recent action:
  relative player id
  move kind
  move rank
  move length
  pass/non-pass
```

A simple first observation vector might be around:

```text
150–300 features
```

Do not include:

```text
opponent hands
future deck information
hidden card ownership
```

---

# 8. Neural network architecture

I would use a **candidate-scoring actor-critic**, not a giant dense action head.

## 8.1 State encoder

Start simple:

```python
obs -> Linear(512) -> ReLU
    -> Linear(512) -> ReLU
    -> Linear(512) -> ReLU
    -> state_embedding
```

Use LayerNorm if training is unstable.

```text
state_embedding dim: 512
```

---

## 8.2 Action embedding

Every global move ID gets an embedding:

```python
action_embedding = nn.Embedding(num_actions, 256)
```

Also feed action metadata if useful:

```text
move kind
num cards
primary rank
secondary rank
high suit
contains 2
contains high cards
```

Then combine:

```python
action_features -> small MLP -> action_embedding
```

A good version:

```python
move_id_embedding:     128 dims
move_feature_encoder:  128 dims
combined action emb:   256 dims
```

---

## 8.3 Policy head

For each state and each legal candidate:

```text
state_embedding:  [B, 512]
candidate_embeds: [B, K, 256]
```

Project state down:

```python
query = state_proj(state_embedding)  # [B, 256]
```

Score candidates:

```python
logits = dot(query, candidate_embeds)
```

Shape:

```text
logits: [B, K]
```

Then mask padded candidates with `-inf`.

Use a categorical distribution over the remaining logits. PyTorch supports categorical distributions parameterized by logits, and TorchRL also has a `MaskedCategorical` distribution for masked action spaces. ([PyTorch Docs][2])

---

## 8.4 Value head

From the same state embedding:

```python
value = value_head(state_embedding)
```

Output:

```text
scalar value for the acting player
```

Meaning:

```text
expected final outcome for this player from this state
```

---

## 8.5 Initial model size

Start with:

```text
State encoder:        3 layers x 512
Action embedding:     128–256
Policy scoring:       dot-product candidate scorer
Value head:           2-layer MLP
Total params:         probably ~1M–5M
```

Do not start with a transformer unless the MLP clearly plateaus.

Later, if needed, upgrade to:

```text
card-set encoder
history transformer
opponent-modeling auxiliary heads
```

But the first serious model should be simple.

---

# 9. PPO training setup

## 9.1 Initial training phase

Start with pure self-play:

```text
all 4 seats controlled by current policy
shared network
store all player decisions
```

This is easiest and fully on-policy.

Each player sees observations from their own perspective. The same network controls everyone.

---

## 9.2 Rollout loop

At each vectorized step:

```python
obs, candidate_ids, candidate_mask = env.observe()

with torch.no_grad():
    logits, values = model(obs, candidate_ids, candidate_mask)
    action_pos = sample_categorical(logits)
    action_ids = gather(candidate_ids, action_pos)
    logprobs = log_prob(action_pos)

next_obs, next_candidates, done, final_rewards, info = env.step(action_ids)

buffer.add(
    obs=obs,
    candidate_ids=candidate_ids,
    candidate_mask=candidate_mask,
    action_pos=action_pos,
    action_id=action_ids,
    logprob=logprobs,
    value=values,
    env_id=...,
    player_id=...
)
```

Important: store candidate IDs from the original decision point. PPO must recompute the probability of the same action under the same candidate set.

---

## 9.3 Reward handling

This is one of the easiest places to introduce bugs.

For Big 2, I would initially use terminal-only rewards.

Example scoring:

```text
winner: +1
others: based on remaining cards, normalized negative reward
```

For example:

```text
winner reward = +1.0
loser reward  = -cards_remaining / 13.0
```

Or if you continue until full ranking:

```text
1st: +1.5
2nd: +0.5
3rd: -0.5
4th: -1.5
```

Keep this simple at first.

When an env finishes, Rust returns:

```text
final_rewards[env_id] = [r0, r1, r2, r3]
```

Python then assigns each player’s final reward to that player’s stored decisions for the episode.

The cleanest initial approach:

```text
Collect full episodes.
For every decision made by player p in that episode:
    return = final_rewards[p]
```

With `gamma = 1.0`, this is Monte Carlo PPO.

Later you can add GAE and bootstrapping for partial episodes.

---

## 9.4 PPO loss

Use standard PPO:

```text
policy ratio = exp(new_logprob - old_logprob)

policy loss =
  -min(
      ratio * advantage,
      clip(ratio, 1-eps, 1+eps) * advantage
   )

value loss =
  mse(value, return)

entropy bonus =
  entropy over legal candidates
```

Initial hyperparameters:

```text
num_envs:              2048–8192 on laptop
rollout decisions:     64k–262k per update
ppo epochs:            2–4
minibatch size:        4096–16384
learning rate:         3e-4
clip epsilon:          0.1–0.2
entropy coefficient:   0.005–0.02
value coefficient:     0.5
max grad norm:         0.5
gamma:                 1.0 initially
advantage norm:        yes
optimizer:             AdamW
```

Use `torch.compile` after the model is stable; PyTorch documents `torch.compile` as a way to optimize model or function execution through TorchDynamo-backed compilation. ([PyTorch Docs][3])

---

# 10. Opponent pool

After the agent beats random and simple scripted bots, add an opponent pool.

## Phase 1

```text
All players = current policy
```

## Phase 2

For each game:

```text
one or more seats = current policy
other seats = sampled frozen checkpoints
```

Only train PPO on actions taken by the current policy.

Do not put frozen-opponent actions into the PPO batch unless you intentionally implement off-policy correction. Simpler: exclude them.

Opponent pool:

```text
checkpoint_000
checkpoint_001
checkpoint_002
...
random bot
greedy bot
heuristic bot
```

Sampling strategy:

```text
70% recent checkpoints
20% older checkpoints
10% scripted bots
```

This reduces self-play cycling.

---

# 11. Evaluation setup

Do not evaluate only against the latest self-play opponent.

Use a fixed evaluation ladder.

## 11.1 Scripted bots

Implement several non-learning bots:

```text
Random legal
Lowest legal
Greedy dump-smallest-combo
Greedy save-2s/high cards
Aggressive control bot
Conservative pass bot
```

The model should first dominate random, then lowest-legal, then greedy bots.

---

## 11.2 Checkpoint league

Every N PPO updates:

```text
save checkpoint
run round-robin evaluation
estimate Elo/Glicko-like rating
```

Evaluate:

```text
current vs fixed bots
current vs previous checkpoints
current vs best checkpoint
current vs mixed pools
```

Rotate seats.

Big 2 has seat/deal variance, so use many games:

```text
quick eval:    1k–5k games
serious eval:  10k–100k games
```

Track confidence intervals, not just raw win rate.

---

## 11.3 Metrics to log

Log:

```text
win rate
average final rank
average terminal reward
cards remaining when losing
seat-specific win rate
pass rate
illegal action count
candidate count distribution
entropy
value loss
policy loss
approx KL
explained variance
lead conversion rate
number of times agent gains control
usage rate of singles/pairs/triples/5-card hands
```

Also log strange behavior:

```text
passes when strong move available
burns 2s too early
holds dead cards too long
overuses 5-card hands
fails opening constraints
```

---

# 12. Development milestones

## Milestone 1: Rules correctness

Before PPO, write tests.

Test:

```text
card ordering
single comparison
pair comparison
triple comparison
straight detection
flush detection
full house detection
four-of-kind detection
straight flush detection
opening 3♦ rule
pass/reset logic
game termination
scoring
```

Also use property tests:

```text
no duplicated cards after deal
all hands have 13 cards after reset
played move is always subset of hand
card count decreases correctly
cards_played never overlaps with hands
candidate list never includes illegal moves
```

This matters more than model architecture.

---

## Milestone 2: Rust random simulation

Run random legal games entirely in Rust.

Target:

```text
millions of complete random games without panic
zero invariant violations
deterministic with fixed seed
```

Benchmark:

```text
games/sec
player-turns/sec
average candidate count
max candidate count
```

---

## Milestone 3: PyO3 wrapper

Expose:

```python
Big2VecEnv.reset()
Big2VecEnv.step()
Big2VecEnv.observe()
Big2VecEnv.seed()
```

Make sure the Python/Rust crossing happens once per batch step, not once per player or once per candidate.

Good:

```python
env.step(action_ids_for_4096_games)
```

Bad:

```python
for game in games:
    env.step_one_game(...)
```

---

## Milestone 4: Random policy integration

Before neural training, run:

```text
Python samples random candidate action
Rust steps env
episodes finish
rewards are assigned correctly
```

If this is buggy, PPO will hide the bug.

---

## Milestone 5: Super-simple PPO

Train against random or greedy bots first.

Goal:

```text
agent beats random bot convincingly
agent beats lowest-legal bot
value predictions become non-random
entropy decreases slowly, not instantly
```

Do not jump directly into four-way checkpoint self-play.

---

## Milestone 6: Full self-play

Then run:

```text
shared policy controls all 4 players
terminal reward only
candidate-scoring network
PPO updates
```

Look for improvement against fixed bots.

---

## Milestone 7: Opponent pool

Add:

```text
checkpoint sampling
league evaluation
best-checkpoint tracking
```

This is where training becomes more robust.

---

# 13. Practical performance targets

On a good laptop, I would aim for:

```text
Rust env throughput:      50k–500k player-turns/sec
Initial num_envs:         2048–4096
Aggressive num_envs:      8192–16384
Rollout size:             64k–262k decisions
```

If you are below ~10k player-turns/sec, profile before changing the neural net.

Likely bottlenecks:

```text
candidate generation
Python/Rust copying
candidate tensor padding
large action scoring
rollout buffer storage
```

---

# 14. Things to avoid

Avoid these early:

```text
card-by-card action generation
dense 20k-action output head for every state
training without action masking
reward shaping before terminal reward works
using opponent hidden hands in observations
training on frozen-opponent actions as if they were current-policy actions
rewriting PPO before validating the environment
optimizing before profiling
```

---

# 15. Recommended first serious configuration

```text
Language:
  Rust env via PyO3/maturin
  PyTorch learner

Env:
  4096 parallel games
  candidate action IDs, not dense action masks
  terminal rewards
  perspective-normalized observations

Model:
  MLP state encoder, 3 x 512
  action embedding, 128–256
  dot-product candidate scorer
  scalar value head

Training:
  PPO
  rollout size 131k decisions
  3 PPO epochs
  minibatch 8192
  lr 3e-4
  clip 0.2
  entropy coef 0.01
  value coef 0.5
  gamma 1.0 initially

Eval:
  random bot
  lowest-legal bot
  greedy bot
  checkpoint league
  seat-rotated 10k-game evals
```

The most important implementation principle:

```text
Rust owns game truth.
Python owns learning.
The policy only sees legal public information.
The model scores candidate moves, not all possible moves.
Evaluation is fixed and independent of training.
```

[1]: https://pyo3.rs/?utm_source=chatgpt.com "Introduction - PyO3 user guide"
[2]: https://docs.pytorch.org/docs/stable/distributions.html?utm_source=chatgpt.com "Probability distributions - torch.distributions"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html?utm_source=chatgpt.com "torch.compile — PyTorch 2.12 documentation"
