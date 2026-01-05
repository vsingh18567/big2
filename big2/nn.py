import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from big2.simulator.cards import PASS, Combo

DROPOUT_RATE = 0.1


def input_dim(n_players: int = 4) -> int:
    """Calculate input dimension based on number of players.

    Returns:
        cards_per_player + 52 (last_play) + 52 (seen) + (n_players - 1) (opponent counts)
        + 1 (passes_in_row) + (n_players - 1) * 52 (opponent cards)
    """
    cards_per_player = 52 // n_players
    return cards_per_player + 52 + 52 + (n_players - 1) + 1 + (n_players - 1) * 52


def combo_to_action_vector(cmb: Combo) -> np.ndarray:
    # 52 one-hot for cards used
    vec = np.zeros(52 + 1, dtype=np.float32)
    if cmb.type == PASS:
        vec[52] = 1.0

    for c in cmb.cards:
        vec[c] = 1.0

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

        # Action encoder: 52 one-hot + type one-hot(9) + size scalar(1) + key features (up to 3 ints) → MLP
        self.action_enc = nn.Sequential(
            nn.Linear(52 + 1, action_hidden),
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
        feats = self.encode_actions(actions_batch)  # (sumA, 53)
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
