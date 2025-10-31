# big2_selfplay_value.py
# Full self-play RL setup for Big 2 with a policy network (candidate scoring) and a value head.
# - 4-player environment
# - State per user request (13 card IDs, last-play 52 one-hot, seen 52 one-hot, opponents' counts)
# - Dynamic candidate enumeration (from current hand), scored by policy given state
# - REINFORCE with learned value baseline (actor-critic style)
# - Entropy regularization
#
# Notes:
# * This is a pragmatic, reasonably faithful Big 2 rules implementation supporting:
#   singles, pairs, triples, and 5-card hands: straight, flush, full house, four-kind(+kicker), straight flush.
# * Comparison uses Big 2 rank ordering (3 < 4 < ... < A < 2). Suits ordered ♦ < ♣ < ♥ < ♠ as tiebreakers.
# * Candidate set size stays manageable because we enumerate ONLY from the current player's 13 cards.
# * We include PASS as a legal candidate except when starting a trick that has no last-play yet.
#
# Training tips:
# * Start with fewer episodes then scale.
# * You can parallelize environments for speed.
# * Logging hooks are included; wire up tensorboard if desired.

import itertools
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------------
# Card utilities
# -------------------------------
# Card IDs: 0..51. Ranks: 0..12 map to [3,4,5,6,7,8,9,10,J,Q,K,A,2]. Suits: 0..3 are [♦,♣,♥,♠].
RANKS = list(range(13))  # 0..12 where 12 corresponds to '2'
SUITS = list(range(4))  # 0: ♦, 1: ♣, 2: ♥, 3: ♠

RANK_NAMES = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]
SUIT_NAMES = ["♦", "♣", "♥", "♠"]


def card_rank(card_id: int) -> int:
    return card_id // 4  # 0..12


def card_suit(card_id: int) -> int:
    return card_id % 4  # 0..3


def card_name(card_id: int) -> str:
    return f"{RANK_NAMES[card_rank(card_id)]}{SUIT_NAMES[card_suit(card_id)]}"


# For Big 2 ordering, higher RANK number is stronger; for ties, higher SUIT number is stronger.


def cmp_single(a: list[int], b: list[int]) -> int:
    ra, sa = card_rank(a[0]), card_suit(a[0])
    rb, sb = card_rank(b[0]), card_suit(b[0])
    if ra != rb:
        return (ra > rb) - (ra < rb)
    return (sa > sb) - (sa < sb)


# -------------------------------
# Combo detection and comparison
# -------------------------------
# Combo types: 0=PASS, 1=SINGLE, 2=PAIR, 3=TRIPLE,
# 4=STRAIGHT, 5=FLUSH, 6=FULLHOUSE, 7=FOUR_KIND, 8=STRAIGHT_FLUSH

PASS, SINGLE, PAIR, TRIPLE, STRAIGHT, FLUSH, FULLHOUSE, FOUR_KIND, STRAIGHT_FLUSH = range(9)


@dataclass
class Combo:
    type: int
    cards: list[int]  # sorted by (rank,suit)
    key: tuple  # used for comparing within same type

    def size(self) -> int:
        return len(self.cards)


# Helpers


def sort_cards(cards: list[int]) -> list[int]:
    return sorted(cards, key=lambda c: (card_rank(c), card_suit(c)))


def is_consecutive(ranks_sorted: list[int]) -> bool:
    # Handle straights in Big 2: 2 (rank=12) cannot be in a straight.
    if 12 in ranks_sorted:
        return False
    return all(ranks_sorted[i] + 1 == ranks_sorted[i + 1] for i in range(len(ranks_sorted) - 1))


def detect_combo(cards: list[int]) -> Combo | None:
    cards = sort_cards(cards)
    n = len(cards)
    if n == 0:
        return Combo(PASS, [], ())
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]

    if n == 1:
        return Combo(SINGLE, cards, (ranks[0], suits[0]))

    if n == 2:
        if ranks[0] == ranks[1]:
            # Pair key: (rank, max_suit)
            return Combo(PAIR, cards, (ranks[0], max(suits)))
        return None

    if n == 3:
        if ranks[0] == ranks[1] == ranks[2]:
            # Triple key: (rank, max_suit)
            return Combo(TRIPLE, cards, (ranks[0], max(suits)))
        return None

    if n == 5:
        # Check poker categories in Big 2 precedence: STRAIGHT < FLUSH < FULL HOUSE < FOUR_KIND < STRAIGHT_FLUSH
        counts: dict[int, int] = {}
        for r in ranks:
            counts[r] = counts.get(r, 0) + 1
        unique = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))  # sort by freq desc, rank desc
        is_flush = len(set(suits)) == 1
        is_straight = is_consecutive(sorted(set(ranks))) and len(set(ranks)) == 5

        if is_straight and is_flush:
            # Key: highest rank of straight, suit of that highest card
            high_rank = max(ranks)
            high_suit = max([card_suit(c) for c in cards if card_rank(c) == high_rank])
            return Combo(STRAIGHT_FLUSH, cards, (high_rank, high_suit))
        if unique[0][1] == 4:
            # Four-kind + kicker. Key: (rank_of_quad, kicker_rank, max_suit_in_quad)
            quad_rank = unique[0][0]
            kicker_rank = unique[1][0]
            max_suit_quad = max([card_suit(c) for c in cards if card_rank(c) == quad_rank])
            return Combo(FOUR_KIND, cards, (quad_rank, kicker_rank, max_suit_quad))
        if unique[0][1] == 3 and unique[1][1] == 2:
            # Full house. Key: (rank_of_trip, pair_rank)
            trip_rank = unique[0][0]
            pair_rank = unique[1][0]
            return Combo(FULLHOUSE, cards, (trip_rank, pair_rank))
        if is_flush:
            # Flush key: ranks sorted desc then suit high-card
            key = tuple(sorted(ranks, reverse=True) + [max(suits)])
            return Combo(FLUSH, cards, key)
        if is_straight:
            high_rank = max(ranks)
            high_suit = max([card_suit(c) for c in cards if card_rank(c) == high_rank])
            return Combo(STRAIGHT, cards, (high_rank, high_suit))
        return None

    return None


def compare_combos(a: Combo, b: Combo) -> int:
    # Only meaningful when a.type == b.type and len equal, per Big 2 trick rules.
    if a.type != b.type or a.size() != b.size():
        return 0  # undefined; caller ensures legality
    # Compare keys lexicographically
    if a.key == b.key:
        return 0
    return 1 if a.key > b.key else -1


# -------------------------------
# Candidate generation from a hand
# -------------------------------


def generate_singles(hand: list[int]) -> list[list[int]]:
    return [[c] for c in hand]


def generate_pairs(hand: list[int]) -> list[list[int]]:
    pairs = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    for _r, cards in by_rank.items():
        if len(cards) >= 2:
            for comb in itertools.combinations(sorted(cards), 2):
                pairs.append(list(comb))
    return pairs


def generate_triples(hand: list[int]) -> list[list[int]]:
    triples = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    for _r, cards in by_rank.items():
        if len(cards) >= 3:
            for comb in itertools.combinations(sorted(cards), 3):
                triples.append(list(comb))
    return triples


def generate_straights(hand: list[int]) -> list[list[int]]:
    # Build by rank sets (exclude rank=12 i.e., 2)
    by_rank_suits: dict[int, list[int]] = {}
    for c in hand:
        r = card_rank(c)
        if r == 12:
            continue
        by_rank_suits.setdefault(r, []).append(c)
    straights = []
    sorted(by_rank_suits.keys())
    # We need sequences of length 5 of consecutive ranks
    for start in range(0, 8):  # 0..7 inclusive allows start at rank 7 -> 7,8,9,10,11
        seq = [start + i for i in range(5)]
        if all(r in by_rank_suits for r in seq):
            # choose one card for each rank
            for choice in itertools.product(*[sorted(by_rank_suits[r]) for r in seq]):
                straights.append(list(choice))
    return straights


def generate_flushes(hand: list[int]) -> list[list[int]]:
    by_suit: dict[int, list[int]] = {s: [] for s in SUITS}
    for c in hand:
        by_suit[card_suit(c)].append(c)
    flushes = []
    for _s, cards in by_suit.items():
        if len(cards) >= 5:
            for comb in itertools.combinations(sorted(cards), 5):
                flushes.append(list(comb))
    return flushes


def generate_fullhouses(hand: list[int]) -> list[list[int]]:
    fulls = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    trips = []
    pairs = []
    for _r, cards in by_rank.items():
        if len(cards) >= 3:
            trips.extend([list(comb) for comb in itertools.combinations(sorted(cards), 3)])
        if len(cards) >= 2:
            pairs.extend([list(comb) for comb in itertools.combinations(sorted(cards), 2)])
    for t in trips:
        tr = card_rank(t[0])
        for p in pairs:
            pr = card_rank(p[0])
            if pr == tr:
                continue
            fulls.append(sort_cards(t + p))
    # Remove duplicates
    uniq = {tuple(sorted(x)): x for x in fulls}
    return list(uniq.values())


def generate_fourkind(hand: list[int]) -> list[list[int]]:
    quads = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    kickers = set(hand)
    for r, cards in by_rank.items():
        if len(cards) == 4:
            for k in kickers:
                if card_rank(k) != r:
                    combo = sort_cards(cards + [k])
                    quads.append(combo)
    # Remove duplicates
    uniq = {tuple(sorted(x)): x for x in quads}
    return list(uniq.values())


def generate_straightflushes(hand: list[int]) -> list[list[int]]:
    by_suit: dict[int, list[int]] = {s: [] for s in SUITS}
    for c in hand:
        by_suit[card_suit(c)].append(c)
    sflushes = []
    for _s, cards in by_suit.items():
        # map suit-specific ranks (exclude 2s)
        by_rank: dict[int, list[int]] = {}
        for c in cards:
            r = card_rank(c)
            if r == 12:
                continue
            by_rank.setdefault(r, []).append(c)
        for start in range(0, 8):
            seq = [start + i for i in range(5)]
            if all(r in by_rank for r in seq):
                for choice in itertools.product(*[sorted(by_rank[r]) for r in seq]):
                    sflushes.append(list(choice))
    return sflushes


COMBO_GENERATORS = {
    SINGLE: generate_singles,
    PAIR: generate_pairs,
    TRIPLE: generate_triples,
    STRAIGHT: generate_straights,
    FLUSH: generate_flushes,
    FULLHOUSE: generate_fullhouses,
    FOUR_KIND: generate_fourkind,
    STRAIGHT_FLUSH: generate_straightflushes,
}

# -------------------------------
# Environment
# -------------------------------


class Big2Env:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.hands = [sorted(deck[i * 13 : (i + 1) * 13], key=lambda c: (card_rank(c), card_suit(c))) for i in range(4)]
        # Start with player holding 3♦ if present; else player 0
        start_player = 0
        three_diamond = 0  # card id 0 is 3♦ by our encoding
        for p in range(4):
            if three_diamond in self.hands[p]:
                start_player = p
                break
        self.current_player = start_player
        self.last_non_pass_player = None
        self.trick_pile: Combo | None = None  # None means new trick (no constraint)
        self.passes_in_row = 0
        self.seen = [0] * 52
        # Mark own hand as seen? We'll mark as 0; seen increments when played
        self.done = False
        self.winner = None
        return self._obs(self.current_player)

    def _obs(self, player: int) -> np.ndarray:
        # 13 scalars for current hand, -1 padded; last-play 52 one-hot; seen 52; opponents counts 3
        hand = self.hands[player]
        hand_ids = hand + [-1] * (13 - len(hand))
        last_play = [0] * 52
        if self.trick_pile is not None and self.trick_pile.type != PASS:
            for c in self.trick_pile.cards:
                last_play[c] = 1
        seen = self.seen.copy()
        # Opponent counts clockwise from current player
        counts = []
        for i in range(1, 4):
            counts.append(len(self.hands[(player + i) % 4]))
        state = np.array(hand_ids + last_play + seen + counts, dtype=np.int32)
        return state

    def legal_candidates(self, player: int) -> list[Combo]:
        hand = self.hands[player]
        candidates: list[Combo] = []
        # Generate all combos from current hand
        all_sets = []
        for _t, gen in COMBO_GENERATORS.items():
            for cards in gen(hand):
                cmb = detect_combo(cards)
                if cmb is not None:
                    all_sets.append(cmb)
        # Filter by trick rules
        if self.trick_pile is None or self.trick_pile.type == PASS:
            # fresh trick: any non-PASS combo allowed
            candidates = all_sets
        else:
            # must match size and type precedence rules (for five-card: same size and category must be >= in category)
            last = self.trick_pile
            # For 1/2/3: must be same size and compare by cmp
            if last.type in (SINGLE, PAIR, TRIPLE):
                for cmb in all_sets:
                    if cmb.type == last.type and compare_combos(cmb, last) > 0:
                        candidates.append(cmb)
            else:
                # five-card: must be same size (always 5). Category must be same or higher category.
                for cmb in all_sets:
                    if cmb.size() != 5:
                        continue
                    if cmb.type == last.type:
                        if compare_combos(cmb, last) > 0:
                            candidates.append(cmb)
                    elif cmb.type > last.type:
                        candidates.append(cmb)
        # PASS allowed if we are not opening a fresh trick
        if self.trick_pile is not None and self.trick_pile.type != PASS:
            candidates.append(Combo(PASS, [], ()))
        return candidates

    def step(self, action: Combo) -> tuple[np.ndarray | None, int, bool, dict]:
        p = self.current_player
        reward = 0
        info: dict[str, Any] = {}
        if action.type == PASS:
            self.passes_in_row += 1
            if self.passes_in_row >= 3 and self.trick_pile is not None:
                # Trick ends; the last non-pass player starts new trick
                if self.last_non_pass_player is not None:
                    self.current_player = self.last_non_pass_player
                self.trick_pile = Combo(PASS, [], ())
                self.passes_in_row = 0
        else:
            # play cards
            for c in action.cards:
                self.hands[p].remove(c)
                self.seen[c] = 1
            self.trick_pile = action
            self.last_non_pass_player = p
            self.passes_in_row = 0
            if len(self.hands[p]) == 0:
                self.done = True
                self.winner = p
                # terminal rewards: +1 to winner, -1 to others
                reward = 1
        # advance turn if not ended by trick reset above
        if not self.done:
            self.current_player = (self.current_player + 1) % 4
        obs = self._obs(self.current_player) if not self.done else None
        return obs, reward, self.done, info


# -------------------------------
# Neural policy with candidate scoring + value head
# -------------------------------


class Big2PolicyValue(nn.Module):
    def __init__(
        self,
        card_vocab=53,
        card_emb_dim=32,
        state_dim=120 + 0,
        hidden=256,
        action_hidden=128,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        # State parts: 13 card IDs (-1 mapped to 52), last_play 52, seen 52, opp 3  => 13 tokens + 107 scalars
        self.pad_id = 52  # for -1
        self.card_emb = nn.Embedding(card_vocab, card_emb_dim)  # 0..51 real cards, 52 pad
        self.card_layernorm = nn.LayerNorm(card_emb_dim)

        # Linear encoders for one-hots and counts
        self.last_play_enc = nn.Linear(52, 64)
        self.seen_enc = nn.Linear(52, 64)
        self.counts_enc = nn.Linear(3, 16)

        # State trunk
        self.state_proj = nn.Linear(card_emb_dim * 2 + 64 + 64 + 16, hidden)  # using mean+max pool across 13
        self.state_ln = nn.LayerNorm(hidden)

        # Action encoder: 52 one-hot + type one-hot(9) + size scalar(1) + key features (up to 3 ints) → MLP
        self.action_enc = nn.Sequential(
            nn.Linear(52 + 9 + 1 + 3, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, action_hidden),
            nn.ReLU(),
        )
        # Policy scorer
        self.policy_head = nn.Sequential(nn.Linear(hidden + action_hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # Value head
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # state_tensor shape: (B, 120) ints
        state_tensor.shape[0]
        # Split
        hand_ids = state_tensor[:, :13].clone()
        last_play = state_tensor[:, 13 : 13 + 52].float()
        seen = state_tensor[:, 65 : 65 + 52].float()
        counts = state_tensor[:, 117:120].float()
        # Map -1 → pad_id
        hand_ids[hand_ids < 0] = self.pad_id
        emb = self.card_emb(hand_ids.long())  # (B,13,E)
        emb = self.card_layernorm(emb)
        mean_pool = emb.mean(dim=1)
        max_pool, _ = emb.max(dim=1)
        lp = F.relu(self.last_play_enc(last_play))
        sn = F.relu(self.seen_enc(seen))
        ct = F.relu(self.counts_enc(counts))
        x = torch.cat([mean_pool, max_pool, lp, sn, ct], dim=-1)
        h: torch.Tensor = F.relu(self.state_proj(x))
        h = self.state_ln(h)
        return h  # (B, hidden)

    def encode_actions(self, actions_batch: list[list[tuple[np.ndarray, tuple]]]) -> torch.Tensor:
        # actions_batch: list length B, each is a list of (action_feat_vector(52+9+1+3), keytuple padded)
        # We'll pack features already provided; here we just stack the vectors.
        # For simplicity, actions provided as plain numpy vectors length 65.
        all_feats = []
        for acts in actions_batch:
            if len(acts) == 0:
                # Edge case: no legal moves (shouldn't happen), insert PASS
                vec = np.zeros(52 + 9 + 1 + 3, dtype=np.float32)
                vec[52 + 0] = 1.0  # set PASS type? We'll handle outside; keep zeros.
                all_feats.append(torch.from_numpy(vec))
            else:
                for feat_vec, _ in acts:
                    all_feats.append(torch.from_numpy(feat_vec.astype(np.float32)))
        if len(all_feats) == 0:
            # Should not happen
            all_feats = [torch.zeros(52 + 9 + 1 + 3, dtype=torch.float32)]
        feats = torch.stack(all_feats, dim=0)
        return feats.to(self.device)

    def forward(
        self,
        state_tensor: torch.Tensor,
        actions_batch: list[list[tuple[np.ndarray, tuple]]],
    ):
        # Returns per-batch lists of policy logits and value estimates
        B = state_tensor.shape[0]
        state_h = self.forward_state(state_tensor)
        # Encode actions per example
        # Build offsets to split later
        counts = [len(a) if len(a) > 0 else 1 for a in actions_batch]
        feats = self.encode_actions(actions_batch)  # (sumA, 65)
        aenc = self.action_enc(feats)  # (sumA, action_hidden)
        # Tile state_h accordingly
        tiled = torch.cat([state_h[i].unsqueeze(0).repeat(counts[i], 1) for i in range(B)], dim=0)
        joint = torch.cat([tiled, aenc], dim=-1)
        logits = self.policy_head(joint).squeeze(-1)  # (sumA,)
        # Split logits per batch element
        split_logits = []
        idx = 0
        for c in counts:
            split_logits.append(logits[idx : idx + c])
            idx += c
        values = self.value_head(state_h).squeeze(-1)  # (B,)
        return split_logits, values


# -------------------------------
# Helper: build action features for network
# -------------------------------


def action_to_feature(cmb: Combo) -> tuple[np.ndarray, tuple]:
    # 52 one-hot for cards used
    vec = np.zeros(52 + 9 + 1 + 3, dtype=np.float32)
    for c in cmb.cards:
        vec[c] = 1.0
    # type one-hot
    vec[52 + cmb.type] = 1.0
    # size
    vec[52 + 9] = len(cmb.cards)
    # key (up to 3 numbers), place into final 3 slots (normalize-ish)
    key = list(cmb.key)[:3] if cmb.key else []
    for i, k in enumerate(key):
        vec[52 + 9 + 1 + i] = float(k)
    return vec, cmb.key


# -------------------------------
# Self-play loop and training
# -------------------------------


@dataclass
class StepRecord:
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    reward: float


def select_action(
    policy: Big2PolicyValue, state: np.ndarray, candidates: list[Combo]
) -> tuple[Combo, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Prepare batch with one element
    st = torch.from_numpy(state[np.newaxis, :]).long().to(policy.device)
    action_feats = [[action_to_feature(c) for c in candidates]]
    logits_list, values = policy(st, action_feats)
    logits = logits_list[0]
    # Softmax over candidate set
    probs = F.softmax(logits, dim=0)
    dist = torch.distributions.Categorical(probs=probs)
    idx = dist.sample()
    logprob = dist.log_prob(idx)
    entropy = dist.entropy()
    chosen = candidates[int(idx.item())]
    value = values[0]
    return chosen, logprob, entropy, value


def episode(env: Big2Env, policy: Big2PolicyValue, device="cpu"):
    # Records per player
    traj: dict[int, list[StepRecord]] = {p: [] for p in range(4)}
    state = env._obs(env.current_player)
    while True:
        p = env.current_player
        candidates = env.legal_candidates(p)
        # Safety: if no candidates somehow, force PASS
        if not candidates:
            candidates = [Combo(PASS, [], ())]
        action, logprob, entropy, value = select_action(policy, state, candidates)
        next_state, reward, done, _ = env.step(action)
        # Store step
        if next_state is None:
            raise ValueError("Next state is None")
        traj[p].append(StepRecord(logprob=logprob, entropy=entropy, value=value, reward=0.0))
        # Assign terminal rewards at the end
        if done:
            winner = env.winner
            for q in range(4):
                final_r = 1.0 if q == winner else -1.0
                # set last step reward for each player (Monte Carlo return will propagate)
                if len(traj[q]) > 0:
                    traj[q][-1] = StepRecord(
                        logprob=traj[q][-1].logprob,
                        entropy=traj[q][-1].entropy,
                        value=traj[q][-1].value,
                        reward=final_r,
                    )
            break
        else:
            state = next_state
    return traj


def compute_returns(rewards: list[float], gamma: float = 1.0) -> list[float]:
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def train_selfplay(
    episodes=2000,
    lr=3e-4,
    entropy_beta=0.01,
    value_coef=0.5,
    gamma=1.0,
    seed=42,
    device="cpu",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = Big2Env(seed=seed)
    policy = Big2PolicyValue(device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(1, episodes + 1):
        env.reset()
        traj = episode(env, policy, device=device)

        # Build losses
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy_term = torch.tensor(0.0, device=device)
        count_steps = 0
        for p in range(4):
            if len(traj[p]) == 0:
                continue
            rewards = [rec.reward for rec in traj[p]]
            returns = compute_returns(rewards, gamma)
            # Tensorize
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            values_t = torch.stack([rec.value for rec in traj[p]])
            logprobs_t = torch.stack([rec.logprob for rec in traj[p]])
            entropies_t = torch.stack([rec.entropy for rec in traj[p]])
            advantages = returns_t - values_t.detach()
            policy_loss = policy_loss - (logprobs_t * advantages).sum()
            value_loss = value_loss + F.mse_loss(values_t, returns_t)
            entropy_term = entropy_term + entropies_t.sum()
            count_steps += len(traj[p])

        loss = policy_loss + value_coef * value_loss - entropy_beta * entropy_term
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if ep % 50 == 0:
            avg_len = sum(len(traj[p]) for p in range(4)) / 4.0
            print(
                f"[Episode {ep}] loss={loss.item():.3f} pol={policy_loss.item():.3f} "
                f"val={value_loss.item():.3f} ent={entropy_term.item():.3f} steps/player~{avg_len:.1f}"
            )

    return policy


# -------------------------------
# Evaluation helpers (starting hand value and card marginal value)
# -------------------------------


def value_of_starting_hand(policy: Big2PolicyValue, hand: list[int], sims: int = 512, device="cpu") -> float:
    # Monte Carlo rollouts with frozen policy; returns expected terminal reward for seat 0 with given starting hand
    wins = 0.0
    for _s in range(sims):
        env = Big2Env()
        # Force seat 0 hand
        deck = set(range(52))
        for p in range(4):
            env.hands[p] = []
        env.hands[0] = sorted(hand, key=lambda c: (card_rank(c), card_suit(c)))
        remain = list(deck - set(hand))
        random.shuffle(remain)
        env.hands[1] = sorted(remain[:13], key=lambda c: (card_rank(c), card_suit(c)))
        env.hands[2] = sorted(remain[13:26], key=lambda c: (card_rank(c), card_suit(c)))
        env.hands[3] = sorted(remain[26:39], key=lambda c: (card_rank(c), card_suit(c)))
        # who holds 3♦ starts
        start_player = 0
        for p in range(4):
            if 0 in env.hands[p]:
                start_player = p
                break
        env.current_player = start_player
        env.trick_pile = None
        env.passes_in_row = 0
        env.seen = [0] * 52
        env.done = False
        env.winner = None

        # Rollout
        state = env._obs(env.current_player)
        while True:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            action, logprob, entropy, value = select_action(policy, state, candidates)
            next_state, reward, done, _ = env.step(action)
            if next_state is None:
                raise ValueError("Next state is None")
            if done:
                wins += 1.0 if env.winner == 0 else -1.0
                break
            else:
                state = next_state
    return wins / sims


def marginal_value_of_card(policy: Big2PolicyValue, card_id: int, sims: int = 1024, device="cpu") -> float:
    # Estimate E[V(hand∪{c}) - V(hand)] over random 12-card contexts at seat 0 via simple rollouts
    diff = 0.0
    deck = set(range(52))
    for _s in range(sims):
        # Sample 12 other cards excluding card_id
        rest = random.sample(list(deck - {card_id}), 12)
        with_c = sorted(rest + [card_id], key=lambda c: (card_rank(c), card_suit(c)))
        # Value with card via quick rollout expectation
        v_with = value_of_starting_hand(policy, with_c, sims=1, device=device)
        # Value baseline: sample a random replacement not in rest
        repl = random.choice(list(deck - set(rest)))
        baseline = value_of_starting_hand(
            policy,
            sorted(rest + [repl], key=lambda c: (card_rank(c), card_suit(c))),
            sims=1,
            device=device,
        )
        diff += v_with - baseline
    return diff / sims


# -------------------------------
# Script entry
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_beta", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    policy = train_selfplay(
        episodes=args.episodes,
        lr=args.lr,
        entropy_beta=args.entropy_beta,
        value_coef=args.value_coef,
        gamma=args.gamma,
        seed=args.seed,
        device=device,
    )

    # Example: evaluate a random starting hand
    hand = sorted(random.sample(range(52), 13), key=lambda c: (card_rank(c), card_suit(c)))
    val = value_of_starting_hand(policy, hand, sims=64, device=device)
    print("Random starting hand:", [card_name(c) for c in hand])
    print("Estimated value:", val)
