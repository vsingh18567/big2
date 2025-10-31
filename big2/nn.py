import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .simulator.cards import PASS, Combo

DROPOUT_RATE = 0.1


def input_dim() -> int:
    return 13 + 52 + 52 + 3


def output_dim() -> int:
    return 53  # 52 cards + 1 for pass


def combo_to_action_vector(cmb: Combo) -> np.ndarray:
    # 52 one-hot for cards used
    vec = np.zeros(52 + 1, dtype=np.float32)
    if cmb.type == PASS:
        vec[52] = 1.0

    for c in cmb.cards:
        vec[c] = 1.0

    return vec


class MLPPolicy(nn.Module):
    def __init__(self, card_vocab=53, card_emb_dim=32, state_dim=120, hidden=64, action_hidden=64, device="cpu"):
        super().__init__()
        self.device = device
        # State parts: 13 card IDs (-1 mapped to 52), last_play 52, seen 52, opp 3  => length 120
        self.pad_id = 52  # for -1
        self.card_emb = nn.Embedding(card_vocab, card_emb_dim)  # 0..51 real cards, 52 pad
        self.card_layernorm = nn.LayerNorm(card_emb_dim)
        self.card_embedding_enc = nn.Linear(card_emb_dim * 13, hidden)

        # Linear encoders for one-hots and counts
        self.last_play_enc = nn.Linear(52, hidden)
        self.seen_enc = nn.Linear(52, hidden)
        self.counts_enc = nn.Linear(3, hidden // 2)

        # State trunk
        self.state_proj = nn.Linear(hidden + hidden + hidden + hidden // 2, hidden)  # using mean+max pool across 13
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
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT_RATE), nn.Linear(hidden, 1)
        )

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # state_tensor shape: (B, 120) ints
        B = state_tensor.shape[0]
        # Split
        hand_ids = state_tensor[:, :13].clone()
        last_play = state_tensor[:, 13 : 13 + 52].float()
        seen = state_tensor[:, 65 : 65 + 52].float()
        counts = state_tensor[:, 117:120].float()
        # Map -1 → pad_id
        hand_ids[hand_ids < 0] = self.pad_id
        emb = self.card_emb(hand_ids.long())  # (B,13,E)
        emb = self.card_layernorm(emb)
        embedding_enc = F.dropout(F.relu(self.card_embedding_enc(emb.view(B, -1))), DROPOUT_RATE)
        lp = F.dropout(F.relu(self.last_play_enc(last_play)), DROPOUT_RATE)
        sn = F.dropout(F.relu(self.seen_enc(seen)), DROPOUT_RATE)
        ct = F.dropout(F.relu(self.counts_enc(counts)), DROPOUT_RATE)
        x = torch.cat([embedding_enc, lp, sn, ct], dim=-1)
        h = F.dropout(F.relu(self.state_proj(x)), DROPOUT_RATE)
        h = self.state_ln(h)
        return h  # (B, hidden)

    def encode_actions(self, actions_batch: list[list[np.ndarray]]) -> torch.Tensor:
        # actions_batch: list length B, each is a list of (action_feat_vector(52+1), keytuple padded)
        # We'll pack features already provided; here we just stack the vectors.
        # For simplicity, actions provided as plain numpy vectors length 65.
        all_feats = []
        for acts in actions_batch:
            if len(acts) == 0:
                # Edge case: no legal moves (shouldn't happen), insert PASS
                vec = np.zeros(52 + 1, dtype=np.float32)
                vec[52 + 0] = 1.0  # set PASS type? We'll handle outside; keep zeros.
                all_feats.append(torch.from_numpy(vec))
            else:
                for feat_vec in acts:
                    all_feats.append(torch.from_numpy(feat_vec.astype(np.float32)))
        if len(all_feats) == 0:
            # Should not happen
            all_feats = [torch.zeros(52 + 1, dtype=torch.float32)]
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
