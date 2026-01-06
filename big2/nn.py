import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from big2.simulator.cards import PASS, Combo, card_suit

DROPOUT_RATE = 0.1

# Action vector dimensions:
# 52 (card flags) + 1 (pass flag) + 9 (combo type) + 1 (num_cards) + 13 (key rank) + 4 (key suit) = 80
ACTION_VECTOR_DIM = 80


def input_dim(n_players: int = 4) -> int:
    """Calculate input dimension based on number of players.

    Returns:
        cards_per_player + 52 (last_play) + 52 (seen) + (n_players - 1) (opponent counts)
        + 1 (passes_in_row) + (n_players - 1) * 52 (opponent cards)
    """
    cards_per_player = 52 // n_players
    return cards_per_player + 52 + 52 + (n_players - 1) + 1 + (n_players - 1) * 52


def combo_to_action_vector(cmb: Combo) -> np.ndarray:
    """
    Convert a Combo to a feature vector for the action encoder.

    Features:
    - 52 dims: one-hot for cards used
    - 1 dim: pass flag
    - 9 dims: combo type one-hot (PASS, SINGLE, PAIR, TRIPLE, STRAIGHT, FLUSH, FULLHOUSE, FOUR_KIND, STRAIGHT_FLUSH)
    - 1 dim: number of cards (normalized by 5)
    - 13 dims: key rank one-hot (the primary rank determining combo strength)
    - 4 dims: key suit one-hot (for tiebreakers)
    """
    vec = np.zeros(ACTION_VECTOR_DIM, dtype=np.float32)

    # Card flags (0-51)
    for c in cmb.cards:
        vec[c] = 1.0

    # Pass flag (52)
    if cmb.type == PASS:
        vec[52] = 1.0
        # Combo type one-hot for PASS (53)
        vec[53] = 1.0
        return vec

    # Combo type one-hot (53-61): 9 types
    vec[53 + cmb.type] = 1.0

    # Number of cards normalized (62)
    vec[62] = len(cmb.cards) / 5.0

    # Key rank one-hot (63-75): 13 ranks
    # Extract the primary rank from the key tuple
    if cmb.key:
        # For most combos, the first element of key contains the primary rank
        # SINGLE: (rank, suit), PAIR: (rank, max_suit), TRIPLE: (rank,)
        # STRAIGHT: (high_rank, high_suit), FLUSH: (suit, ranks...)
        # FULLHOUSE: (trip_rank, pair_rank), FOUR_KIND: (quad_rank,)
        # STRAIGHT_FLUSH: (high_suit, high_rank)
        if cmb.type in (1, 2, 3, 7):  # SINGLE, PAIR, TRIPLE, FOUR_KIND - rank is first
            key_rank = cmb.key[0]
            vec[63 + key_rank] = 1.0
        elif cmb.type == 4:  # STRAIGHT - high_rank is first
            key_rank = cmb.key[0]
            vec[63 + key_rank] = 1.0
        elif cmb.type == 5:  # FLUSH - suit is first, ranks follow
            if len(cmb.key) > 1:
                key_rank = cmb.key[1]  # highest rank is second element
                vec[63 + key_rank] = 1.0
        elif cmb.type == 6:  # FULLHOUSE - trip_rank is first
            key_rank = cmb.key[0]
            vec[63 + key_rank] = 1.0
        elif cmb.type == 8:  # STRAIGHT_FLUSH - suit is first, high_rank is second
            if len(cmb.key) > 1:
                key_rank = cmb.key[1]
                vec[63 + key_rank] = 1.0

    # Key suit one-hot (76-79): 4 suits
    # Extract suit from the highest card in the combo for consistency
    if cmb.cards:
        # Use the suit of the highest card in the combo
        max_card = max(cmb.cards)
        key_suit = card_suit(max_card)
        vec[76 + key_suit] = 1.0

    return vec


class MLPPolicy(nn.Module):
    def __init__(
        self, n_players: int = 4, card_vocab=53, card_emb_dim=32, hidden=1024, action_hidden=256, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.n_players = n_players
        self.cards_per_player = 52 // n_players
        # State parts: cards_per_player card IDs (-1 mapped to 52), last_play 52, seen 52, opp (n_players-1)
        self.pad_id = 52  # for -1
        self.card_emb = nn.Embedding(card_vocab, card_emb_dim)  # 0..51 real cards, 52 pad
        self.card_embedding_enc = nn.Linear(card_emb_dim * self.cards_per_player, hidden)

        # Linear encoders for one-hots and counts
        self.last_play_enc = nn.Linear(52, hidden)
        self.seen_enc = nn.Linear(52, hidden)
        self.counts_enc = nn.Linear(n_players - 1, hidden // 2)
        self.passes_enc = nn.Linear(1, hidden // 4)
        self.opponent_cards_enc = nn.Linear((n_players - 1) * 52, hidden)

        # State trunk
        self.state_proj = nn.Linear(hidden + hidden + hidden + hidden // 2 + hidden // 4 + hidden, hidden)
        self.state_ln = nn.LayerNorm(hidden)

        # Action encoder: 52 cards + 1 pass + 9 combo types + 1 num_cards + 13 ranks + 4 suits = 80
        self.action_enc = nn.Sequential(
            nn.Linear(ACTION_VECTOR_DIM, action_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(action_hidden, action_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        # Policy scorer
        self.policy_head = nn.Sequential(
            nn.Linear(hidden + action_hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT_RATE), nn.Linear(hidden, 1)
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # state_tensor shape: (B, input_dim) ints
        B = state_tensor.shape[0]
        # Split
        hand_ids = state_tensor[:, : self.cards_per_player].clone()
        last_play = state_tensor[:, self.cards_per_player : self.cards_per_player + 52].float()
        seen = state_tensor[:, self.cards_per_player + 52 : self.cards_per_player + 104].float()
        counts_start = self.cards_per_player + 104
        counts_end = counts_start + (self.n_players - 1)
        counts = state_tensor[:, counts_start:counts_end].float()
        passes_start = counts_end
        passes_end = passes_start + 1
        passes = state_tensor[:, passes_start:passes_end].float()
        opponent_cards = state_tensor[:, passes_end:].float()
        # Map -1 → pad_id
        hand_ids[hand_ids < 0] = self.pad_id
        emb = self.card_emb(hand_ids.long())  # (B,13,E)
        embedding_enc = F.relu(self.card_embedding_enc(emb.view(B, -1)))
        lp = F.relu(self.last_play_enc(last_play))
        sn = F.relu(self.seen_enc(seen))
        ct = F.relu(self.counts_enc(counts))
        ps = F.relu(self.passes_enc(passes))
        oc = F.relu(self.opponent_cards_enc(opponent_cards))
        x = torch.cat([embedding_enc, lp, sn, ct, ps, oc], dim=-1)
        h = F.relu(self.state_proj(x))
        h = self.state_ln(h)
        return h  # (B, hidden)

    def encode_actions(self, actions_batch: list[list[np.ndarray]]) -> torch.Tensor:
        # actions_batch: list length B, each is a list of (action_feat_vector(52+1), keytuple padded)
        # We'll pack features already provided; here we just stack the vectors.
        # For simplicity, actions provided as plain numpy vectors
        all_feats = []
        for acts in actions_batch:
            for feat_vec in acts:
                all_feats.append(torch.from_numpy(feat_vec.astype(np.float32)))
        feats = torch.stack(all_feats, dim=0)
        return feats.to(self.device)

    def forward(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]):
        # Returns per-batch lists of policy logits and value estimates
        B = state_tensor.shape[0]
        state_h = self.forward_state(state_tensor)  # (B, hidden)
        # Encode actions per example
        # Build offsets to split later
        counts = [len(a) if len(a) > 0 else 1 for a in actions_batch]
        feats = self.encode_actions(actions_batch)  # (sumA, ACTION_VECTOR_DIM)
        aenc = self.action_enc(feats)  # (sumA, action_hidden)
        # Tile state_h accordingly
        tiled = torch.cat(
            [state_h[i].unsqueeze(0).repeat(counts[i], 1) for i in range(B)], dim=0
        )  # expands (B, hidden) to (sumA, hidden) based on action count per batch
        joint = torch.cat([tiled, aenc], dim=-1)  # (sumA, hidden + action_hidden)
        logits = self.policy_head(joint).squeeze(-1)  # (sumA,)
        # Split logits per batch element
        split_logits = []
        idx = 0
        for c in counts:
            split_logits.append(logits[idx : idx + c])
            idx += c
        values = self.value_head(state_h).squeeze(-1)  # (B,)
        return split_logits, values


class MLPQNetwork(nn.Module):
    """
    Q-value approximator with the same overall architecture as `MLPPolicy`,
    but outputs a single scalar Q(s,a) for each provided (state, action).

    Notes:
    - State encoding matches `MLPPolicy.forward_state`.
    - Action encoding matches `MLPPolicy.action_enc` (via `combo_to_action_vector`).
    - Output is a per-candidate scalar (no separate value head).
    """

    def __init__(
        self, n_players: int = 4, card_vocab=53, card_emb_dim=32, hidden=1024, action_hidden=256, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.n_players = n_players
        self.cards_per_player = 52 // n_players
        # State parts: cards_per_player card IDs (-1 mapped to 52), last_play 52, seen 52, opp (n_players-1)
        self.pad_id = 52  # for -1
        self.card_emb = nn.Embedding(card_vocab, card_emb_dim)  # 0..51 real cards, 52 pad
        self.card_embedding_enc = nn.Linear(card_emb_dim * self.cards_per_player, hidden)

        # Linear encoders for one-hots and counts
        self.last_play_enc = nn.Linear(52, hidden)
        self.seen_enc = nn.Linear(52, hidden)
        self.counts_enc = nn.Linear(n_players - 1, hidden // 2)
        self.passes_enc = nn.Linear(1, hidden // 4)
        self.opponent_cards_enc = nn.Linear((n_players - 1) * 52, hidden)

        # State trunk
        self.state_proj = nn.Linear(hidden + hidden + hidden + hidden // 2 + hidden // 4 + hidden, hidden)
        self.state_ln = nn.LayerNorm(hidden)

        # Action encoder
        self.action_enc = nn.Sequential(
            nn.Linear(ACTION_VECTOR_DIM, action_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(action_hidden, action_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # Q head: scalar per (state, action)
        self.q_head = nn.Sequential(
            nn.Linear(hidden + action_hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT_RATE), nn.Linear(hidden, 1)
        )

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # state_tensor shape: (B, input_dim) ints
        B = state_tensor.shape[0]
        # Split
        hand_ids = state_tensor[:, : self.cards_per_player].clone()
        last_play = state_tensor[:, self.cards_per_player : self.cards_per_player + 52].float()
        seen = state_tensor[:, self.cards_per_player + 52 : self.cards_per_player + 104].float()
        counts_start = self.cards_per_player + 104
        counts_end = counts_start + (self.n_players - 1)
        counts = state_tensor[:, counts_start:counts_end].float()
        passes_start = counts_end
        passes_end = passes_start + 1
        passes = state_tensor[:, passes_start:passes_end].float()
        opponent_cards = state_tensor[:, passes_end:].float()
        # Map -1 → pad_id
        hand_ids[hand_ids < 0] = self.pad_id
        emb = self.card_emb(hand_ids.long())  # (B,13,E)
        embedding_enc = F.relu(self.card_embedding_enc(emb.view(B, -1)))
        lp = F.relu(self.last_play_enc(last_play))
        sn = F.relu(self.seen_enc(seen))
        ct = F.relu(self.counts_enc(counts))
        ps = F.relu(self.passes_enc(passes))
        oc = F.relu(self.opponent_cards_enc(opponent_cards))
        x = torch.cat([embedding_enc, lp, sn, ct, ps, oc], dim=-1)
        h = F.relu(self.state_proj(x))
        h = self.state_ln(h)
        return h  # (B, hidden)

    def encode_actions(self, actions_batch: list[list[np.ndarray]]) -> torch.Tensor:
        all_feats: list[torch.Tensor] = []
        for acts in actions_batch:
            for feat_vec in acts:
                all_feats.append(torch.from_numpy(feat_vec.astype(np.float32)))
        feats = torch.stack(all_feats, dim=0)
        return feats.to(self.device)

    def q_values(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]) -> list[torch.Tensor]:
        """
        Compute Q-values for each candidate action for each state in the batch.

        Returns:
            list of length B; each element is a 1D tensor of shape (num_actions_i,)
        """
        B = state_tensor.shape[0]
        state_h = self.forward_state(state_tensor)  # (B, hidden)
        counts = [len(a) if len(a) > 0 else 1 for a in actions_batch]
        feats = self.encode_actions(actions_batch)  # (sumA, ACTION_VECTOR_DIM)
        aenc = self.action_enc(feats)  # (sumA, action_hidden)
        tiled = torch.cat([state_h[i].unsqueeze(0).repeat(counts[i], 1) for i in range(B)], dim=0)
        joint = torch.cat([tiled, aenc], dim=-1)
        q_flat = self.q_head(joint).squeeze(-1)  # (sumA,)
        split_q: list[torch.Tensor] = []
        idx = 0
        for c in counts:
            split_q.append(q_flat[idx : idx + c])
            idx += c
        return split_q

    def forward(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]) -> list[torch.Tensor]:
        return self.q_values(state_tensor, actions_batch)
