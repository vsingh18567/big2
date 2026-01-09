from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from big2.simulator.cards import PASS, Combo, card_suit

DROPOUT_RATE = 0.1

# Action vector dimensions:
# 52 (card flags) + 1 (pass flag) + 9 (combo type) + 1 (num_cards) + 13 (key rank) + 4 (key suit) = 80
ACTION_VECTOR_DIM = 80


@dataclass
class MLPPolicyConfig:
    """Configuration for MLPPolicy architecture."""

    n_players: int = 4
    card_vocab: int = 53
    card_emb_dim: int = 32
    hidden: int = 1024
    action_hidden: int = 256
    device: str = "cpu"


@dataclass
class SetPoolPolicyConfig:
    """Configuration for SetPoolPolicy architecture."""

    n_players: int = 4
    card_vocab: int = 53
    card_emb_dim: int = 64
    hidden: int = 1536
    action_hidden: int = 384
    device: str = "cpu"


@dataclass
class MLPQNetworkConfig:
    """Configuration for MLPQNetwork architecture."""

    n_players: int = 4
    card_vocab: int = 53
    card_emb_dim: int = 32
    hidden: int = 1024
    action_hidden: int = 256
    device: str = "cpu"


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


def _pool_card_embeddings(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool card embeddings with a boolean mask.

    Args:
        emb: (B, N, E)
        mask: (B, N) True for valid tokens
    """
    mask_f = mask.to(dtype=emb.dtype).unsqueeze(-1)  # (B,N,1)
    summed = (emb * mask_f).sum(dim=1)  # (B,E)
    denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B,1)
    return summed / denom


def make_policy(
    arch: str,
    *,
    config: MLPPolicyConfig | SetPoolPolicyConfig | None = None,
    n_players: int = 4,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """
    Factory for policy networks.

    Keeping a central factory lets training/eval code swap architectures without
    sprinkling class names everywhere (important for checkpoint opponents).

    Args:
        arch: Architecture name ("mlp" or "setpool")
        config: Optional config dataclass. If None, constructs from kwargs for backward compatibility.
        n_players: Number of players (used if config is None)
        device: Device to use (used if config is None)
        **kwargs: Additional parameters for backward compatibility (card_vocab, card_emb_dim, hidden, action_hidden)
    """
    arch_norm = arch.lower().strip()

    if config is None:
        # Backward compatibility: construct config from kwargs
        if arch_norm in {"mlp", "mlppolicy"}:
            config = MLPPolicyConfig(
                n_players=n_players,
                device=device,
                card_vocab=kwargs.get("card_vocab", 53),
                card_emb_dim=kwargs.get("card_emb_dim", 32),
                hidden=kwargs.get("hidden", 1024),
                action_hidden=kwargs.get("action_hidden", 256),
            )
        elif arch_norm in {"setpool", "set_pool", "pooled"}:
            config = SetPoolPolicyConfig(
                n_players=n_players,
                device=device,
                card_vocab=kwargs.get("card_vocab", 53),
                card_emb_dim=kwargs.get("card_emb_dim", 64),
                hidden=kwargs.get("hidden", 768),
                action_hidden=kwargs.get("action_hidden", 256),
            )
        else:
            raise ValueError(f"Unknown policy arch: {arch!r}. Expected one of: mlp, setpool")

    if arch_norm in {"mlp", "mlppolicy"}:
        if not isinstance(config, MLPPolicyConfig):
            # Convert SetPoolPolicyConfig to MLPPolicyConfig if needed
            config = MLPPolicyConfig(
                n_players=config.n_players,
                card_vocab=config.card_vocab,
                card_emb_dim=config.card_emb_dim,
                hidden=config.hidden,
                action_hidden=config.action_hidden,
                device=config.device,
            )
        print("Using MLPPolicy")
        return MLPPolicy(
            n_players=config.n_players,
            card_vocab=config.card_vocab,
            card_emb_dim=config.card_emb_dim,
            hidden=config.hidden,
            action_hidden=config.action_hidden,
            device=config.device,
        )
    if arch_norm in {"setpool", "set_pool", "pooled"}:
        if not isinstance(config, SetPoolPolicyConfig):
            # Convert MLPPolicyConfig to SetPoolPolicyConfig if needed
            config = SetPoolPolicyConfig(
                n_players=config.n_players,
                card_vocab=config.card_vocab,
                card_emb_dim=config.card_emb_dim,
                hidden=config.hidden,
                action_hidden=config.action_hidden,
                device=config.device,
            )
        print("Using SetPoolPolicy")
        return SetPoolPolicy(
            n_players=config.n_players,
            card_vocab=config.card_vocab,
            card_emb_dim=config.card_emb_dim,
            hidden=config.hidden,
            action_hidden=config.action_hidden,
            device=config.device,
        )
    raise ValueError(f"Unknown policy arch: {arch!r}. Expected one of: mlp, setpool")


# -----------------------------
# Building blocks (reusable nn)
# -----------------------------


def _infer_device(device: str | torch.device | None, *, fallback: torch.device) -> torch.device:
    """Prefer explicit `device`, else fall back to a tensor's device."""
    if device is None:
        return fallback
    return torch.device(device)


def _encode_actions_to_tensor(
    actions_batch: list[list[np.ndarray]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """
    Flatten a batch of variable-length action feature lists.

    Returns:
        feats: (sumA, ACTION_VECTOR_DIM)
        counts: list length B, number of candidate actions per batch element
    """
    counts = [len(a) for a in actions_batch]
    all_feats: list[torch.Tensor] = []
    for acts in actions_batch:
        for feat_vec in acts:
            all_feats.append(torch.from_numpy(feat_vec.astype(np.float32)))
    if not all_feats:
        # This should not happen in normal training (candidates always include PASS),
        # but keep a clear error if callers accidentally pass empty candidates.
        raise ValueError("actions_batch contains no actions; cannot score candidates.")
    feats = torch.stack(all_feats, dim=0).to(device)
    return feats, counts


def _repeat_by_counts(x: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """
    Repeat each row x[i] counts[i] times, concatenated.

    Args:
        x: (B, D)
        counts: list length B
    Returns:
        (sum(counts), D)
    """
    if x.shape[0] != len(counts):
        raise ValueError(f"Batch mismatch: x has B={x.shape[0]} but counts has len={len(counts)}")
    return torch.cat([x[i].unsqueeze(0).repeat(max(1, counts[i]), 1) for i in range(x.shape[0])], dim=0)


def _split_by_counts(x: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
    """Split a flat tensor into a per-example list, using candidate counts."""
    split: list[torch.Tensor] = []
    idx = 0
    for c in counts:
        c_eff = max(1, c)
        split.append(x[idx : idx + c_eff])
        idx += c_eff
    return split


class MLP(nn.Module):
    """
    A small utility for 'professional' MLP definition: declarative layers, typed activation, and dropout.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        *,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = 0.0,
        out_dim: int | None = None,
        out_activation: Literal["none", "relu", "gelu"] = "none",
    ):
        super().__init__()
        act: nn.Module = nn.ReLU() if activation == "relu" else nn.GELU()
        out_act: nn.Module | None
        if out_activation == "none":
            out_act = None
        elif out_activation == "relu":
            out_act = nn.ReLU()
        else:
            out_act = nn.GELU()

        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(a, b))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        if out_dim is not None:
            layers.append(nn.Linear(dims[-1], out_dim))
            if out_act is not None:
                layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPStateEncoder(nn.Module):
    """
    Shared state encoder used by `MLPPolicy` and `MLPQNetwork`.

    Input state layout is identical to the original implementation, and this module is intentionally
    "dumb + explicit" to keep the contract stable.
    """

    def __init__(
        self,
        *,
        n_players: int,
        card_vocab: int,
        card_emb_dim: int,
        hidden: int,
    ):
        super().__init__()
        self.n_players = n_players
        self.cards_per_player = 52 // n_players
        self.pad_id = 52  # for -1

        self.card_emb = nn.Embedding(card_vocab, card_emb_dim)
        self.card_embedding_enc = nn.Linear(card_emb_dim * self.cards_per_player, hidden)

        self.last_play_enc = nn.Linear(52, hidden)
        self.seen_enc = nn.Linear(52, hidden)
        self.counts_enc = nn.Linear(n_players - 1, hidden // 2)
        self.passes_enc = nn.Linear(1, hidden // 4)
        self.opponent_cards_enc = nn.Linear((n_players - 1) * 52, hidden)

        self.state_proj = nn.Linear(hidden + hidden + hidden + hidden // 2 + hidden // 4 + hidden, hidden)
        self.state_ln = nn.LayerNorm(hidden)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # state_tensor shape: (B, input_dim) ints
        B = state_tensor.shape[0]

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

        hand_ids[hand_ids < 0] = self.pad_id
        emb = self.card_emb(hand_ids.long())  # (B,N,E)
        embedding_enc = F.relu(self.card_embedding_enc(emb.view(B, -1)))

        lp = F.relu(self.last_play_enc(last_play))
        sn = F.relu(self.seen_enc(seen))
        ct = F.relu(self.counts_enc(counts))
        ps = F.relu(self.passes_enc(passes))
        oc = F.relu(self.opponent_cards_enc(opponent_cards))

        x = torch.cat([embedding_enc, lp, sn, ct, ps, oc], dim=-1)
        h = F.relu(self.state_proj(x))
        h = self.state_ln(h)
        return h


class ActionEncoder(nn.Module):
    """Shared action encoder for candidate combos (from 80-d engineered feature vector)."""

    def __init__(
        self,
        *,
        in_dim: int = ACTION_VECTOR_DIM,
        hidden: int,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.mlp = MLP(in_dim, [hidden, hidden], activation=activation, dropout=dropout)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(feats)


class ConcatScalarHead(nn.Module):
    """Scalar scorer applied to concatenated (state, action) embeddings."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden: int,
        activation: Literal["relu", "gelu"] = "relu",
        dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.mlp = MLP(state_dim + action_dim, [hidden], activation=activation, dropout=dropout, out_dim=1)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.mlp(state_action).squeeze(-1)


class DotProductScorer(nn.Module):
    """Dot-product scorer between a projected state embedding and an action embedding."""

    def __init__(self, *, state_dim: int, action_dim: int, scale: bool = True):
        super().__init__()
        self.proj = nn.Linear(state_dim, action_dim)
        self.scale = scale

    def forward(self, state_h: torch.Tensor, action_h: torch.Tensor) -> torch.Tensor:
        # state_h: (B,Hs), action_h: (sumA,Ha). Caller is responsible for tiling state_h to sumA.
        if state_h.shape[0] != action_h.shape[0]:
            raise ValueError("DotProductScorer expects tiled state_h and action_h to have same first dim")
        scores = (self.proj(state_h) * action_h).sum(dim=-1)
        if self.scale:
            scores = scores / (action_h.shape[-1] ** 0.5)
        return scores


class ValueHead(nn.Module):
    """State value head V(s)."""

    def __init__(self, *, in_dim: int, hidden: int, activation: Literal["relu", "gelu"] = "relu"):
        super().__init__()
        act = "relu" if activation == "relu" else "gelu"
        self.net = MLP(in_dim, [hidden, hidden // 2], activation=act, dropout=0.0, out_dim=1)

    def forward(self, state_h: torch.Tensor) -> torch.Tensor:
        return self.net(state_h).squeeze(-1)


class CandidateScoringMixin:
    """
    Shared 'score all candidate actions' logic.

    This centralizes the tricky batching/tiling mechanics so policy/Q implementations are small.
    """

    device: str

    def _score_candidates_concat(
        self,
        *,
        state_h: torch.Tensor,
        actions_batch: list[list[np.ndarray]],
        action_encoder: nn.Module,
        scorer: nn.Module,
        device: torch.device | None = None,
    ) -> list[torch.Tensor]:
        dev = _infer_device(device, fallback=state_h.device)
        feats, counts = _encode_actions_to_tensor(actions_batch, device=dev)
        aenc = action_encoder(feats)
        tiled_state = _repeat_by_counts(state_h, counts)
        joint = torch.cat([tiled_state, aenc], dim=-1)
        scores_flat = scorer(joint)
        return _split_by_counts(scores_flat, counts)

    def _score_candidates_dot(
        self,
        *,
        state_h: torch.Tensor,
        actions_batch: list[list[np.ndarray]],
        action_encoder: nn.Module,
        scorer: DotProductScorer,
        device: torch.device | None = None,
    ) -> list[torch.Tensor]:
        dev = _infer_device(device, fallback=state_h.device)
        feats, counts = _encode_actions_to_tensor(actions_batch, device=dev)
        aenc = action_encoder(feats)
        tiled_state = _repeat_by_counts(state_h, counts)
        scores_flat = scorer(tiled_state, aenc)
        return _split_by_counts(scores_flat, counts)


class MLPPolicy(nn.Module, CandidateScoringMixin):
    def __init__(
        self, n_players: int = 4, card_vocab=53, card_emb_dim=32, hidden=1024, action_hidden=256, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.n_players = n_players  # legacy/public attribute used by training utils
        self.state_encoder = MLPStateEncoder(
            n_players=n_players, card_vocab=card_vocab, card_emb_dim=card_emb_dim, hidden=hidden
        )
        self.action_encoder = ActionEncoder(hidden=action_hidden, activation="relu", dropout=DROPOUT_RATE)
        self.policy_scorer = ConcatScalarHead(
            state_dim=hidden, action_dim=action_hidden, hidden=hidden, activation="relu", dropout=DROPOUT_RATE
        )
        self.value_head = ValueHead(in_dim=hidden, hidden=hidden, activation="relu")

        # Legacy attribute aliases (kept for backward compatibility / introspection tooling).
        # Prefer using the modular components above in new code.
        self.card_emb = self.state_encoder.card_emb
        self.card_embedding_enc = self.state_encoder.card_embedding_enc
        self.action_enc = self.action_encoder.mlp.net
        self.policy_head = self.policy_scorer.mlp.net

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state_tensor)

    def encode_actions(self, actions_batch: list[list[np.ndarray]]) -> torch.Tensor:
        feats, _counts = _encode_actions_to_tensor(actions_batch, device=torch.device(self.device))
        return feats

    def forward(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]):
        # Returns per-batch lists of policy logits and value estimates
        state_h = self.forward_state(state_tensor)  # (B, hidden)
        logits_list = self._score_candidates_concat(
            state_h=state_h,
            actions_batch=actions_batch,
            action_encoder=self.action_encoder,
            scorer=self.policy_scorer,
            device=state_tensor.device,
        )
        values = self.value_head(state_h)  # (B,)
        return logits_list, values


class SetPoolStateEncoder(nn.Module):
    """State encoder used by `SetPoolPolicy` (order-invariant pooling + shared embedding for 52-d sets)."""

    def __init__(
        self,
        *,
        n_players: int,
        card_vocab: int,
        card_emb_dim: int,
        hidden: int,
    ):
        super().__init__()
        self.n_players = n_players
        self.cards_per_player = 52 // n_players
        self.pad_id = 52

        self.card_emb = nn.Embedding(card_vocab, card_emb_dim)

        self.hand_enc = nn.Sequential(nn.Linear(card_emb_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT_RATE))
        self.set52_enc = nn.Sequential(nn.Linear(card_emb_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT_RATE))
        self.counts_enc = nn.Sequential(nn.Linear(n_players - 1, hidden), nn.GELU(), nn.Dropout(DROPOUT_RATE))
        self.passes_enc = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Dropout(DROPOUT_RATE))
        self.opp_enc = nn.Sequential(nn.Linear(card_emb_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT_RATE))

        state_in = hidden * (1 + 1 + 1 + 1 + (n_players - 1))  # hand, last_play, seen, passes, opponents
        state_in += hidden  # counts
        self.state_proj = nn.Linear(state_in, hidden)
        self.state_ln = nn.LayerNorm(hidden)
        self.state_ff = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden, hidden),
            nn.Dropout(DROPOUT_RATE),
        )

    def _encode_onehot52(self, x52: torch.Tensor) -> torch.Tensor:
        w = self.card_emb.weight[:52]  # (52,E)
        return x52.to(dtype=w.dtype) @ w

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        B = state_tensor.shape[0]

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

        hand_ids[hand_ids < 0] = self.pad_id
        hand_mask = hand_ids != self.pad_id
        hand_emb = self.card_emb(hand_ids.long())  # (B,N,E)
        hand_pool = _pool_card_embeddings(hand_emb, hand_mask)  # (B,E)
        hand_h = self.hand_enc(hand_pool)  # (B,H)

        lp_pool = self._encode_onehot52(last_play)
        sn_pool = self._encode_onehot52(seen)
        lp_h = self.set52_enc(lp_pool)
        sn_h = self.set52_enc(sn_pool)

        if opponent_cards.numel() == 0:
            opp_flat = torch.zeros((B, 0), device=state_tensor.device, dtype=hand_h.dtype)
        else:
            opp = opponent_cards.view(B, self.n_players - 1, 52)
            opp_pool = torch.einsum("bnc,ce->bne", opp, self.card_emb.weight[:52])  # (B,n-1,E)
            opp_h = self.opp_enc(opp_pool)  # (B,n-1,H)
            opp_flat = opp_h.reshape(B, -1)

        ct_h = self.counts_enc(counts)
        ps_h = self.passes_enc(passes)

        x = torch.cat([hand_h, lp_h, sn_h, ps_h, opp_flat, ct_h], dim=-1)
        h = self.state_ln(F.gelu(self.state_proj(x)))
        h = h + self.state_ff(h)
        h = self.state_ln(h)
        return h


class SetPoolPolicy(nn.Module, CandidateScoringMixin):
    """
    A stronger, more inductive-bias-friendly alternative to MLPPolicy:
    - Own hand is encoded via masked mean-pooling over card embeddings (order-invariant).
    - last_play / seen / opponent_cards are encoded via the same card embedding table, pooled from 52-d one-hot.
    - Policy head uses a dot-product scorer between state and action embeddings (cheap, expressive, stable).
    """

    def __init__(
        self,
        n_players: int = 4,
        card_vocab: int = 53,
        card_emb_dim: int = 64,
        hidden: int = 768,
        action_hidden: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.n_players = n_players
        self.state_encoder = SetPoolStateEncoder(
            n_players=n_players, card_vocab=card_vocab, card_emb_dim=card_emb_dim, hidden=hidden
        )
        self.action_encoder = ActionEncoder(hidden=action_hidden, activation="gelu", dropout=DROPOUT_RATE)
        self.policy_scorer = DotProductScorer(state_dim=hidden, action_dim=action_hidden, scale=True)
        self.value_head = ValueHead(in_dim=hidden, hidden=hidden, activation="gelu")

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state_tensor)

    def encode_actions(self, actions_batch: list[list[np.ndarray]]) -> torch.Tensor:
        feats, _counts = _encode_actions_to_tensor(actions_batch, device=torch.device(self.device))
        return feats

    def forward(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]):
        state_h = self.forward_state(state_tensor)  # (B,H)
        logits_list = self._score_candidates_dot(
            state_h=state_h,
            actions_batch=actions_batch,
            action_encoder=self.action_encoder,
            scorer=self.policy_scorer,
            device=state_tensor.device,
        )
        values = self.value_head(state_h)
        return logits_list, values


class MLPQNetwork(nn.Module, CandidateScoringMixin):
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
        self.n_players = n_players  # legacy/public attribute
        self.state_encoder = MLPStateEncoder(
            n_players=n_players, card_vocab=card_vocab, card_emb_dim=card_emb_dim, hidden=hidden
        )
        self.action_encoder = ActionEncoder(hidden=action_hidden, activation="relu", dropout=DROPOUT_RATE)
        self.q_scorer = ConcatScalarHead(
            state_dim=hidden, action_dim=action_hidden, hidden=hidden, activation="relu", dropout=DROPOUT_RATE
        )

        # Legacy attribute aliases (kept for backward compatibility / introspection tooling).
        self.card_emb = self.state_encoder.card_emb
        self.card_embedding_enc = self.state_encoder.card_embedding_enc
        self.action_enc = self.action_encoder.mlp.net
        self.q_head = self.q_scorer.mlp.net

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state_tensor)

    def encode_actions(self, actions_batch: list[list[np.ndarray]]) -> torch.Tensor:
        feats, _counts = _encode_actions_to_tensor(actions_batch, device=torch.device(self.device))
        return feats

    def q_values(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]) -> list[torch.Tensor]:
        """
        Compute Q-values for each candidate action for each state in the batch.

        Returns:
            list of length B; each element is a 1D tensor of shape (num_actions_i,)
        """
        state_h = self.forward_state(state_tensor)  # (B, hidden)
        return self._score_candidates_concat(
            state_h=state_h,
            actions_batch=actions_batch,
            action_encoder=self.action_encoder,
            scorer=self.q_scorer,
            device=state_tensor.device,
        )

    def forward(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]) -> list[torch.Tensor]:
        return self.q_values(state_tensor, actions_batch)


@dataclass
class SetPoolQNetworkConfig:
    """Configuration for SetPoolQNetwork architecture."""

    n_players: int = 4
    card_vocab: int = 53
    card_emb_dim: int = 64
    hidden: int = 768
    action_hidden: int = 256
    device: str = "cpu"


class SetPoolQNetwork(nn.Module, CandidateScoringMixin):
    """
    Q-value approximator analogous to `SetPoolPolicy`.

    - State encoding matches `SetPoolPolicy.forward_state` (order-invariant hand pooling + pooled 52-d sets).
    - Action encoding matches `SetPoolPolicy` (80-d engineered action vector).
    - Outputs a scalar Q(s,a) for each provided candidate action, returned as a per-batch list.
    """

    def __init__(
        self,
        n_players: int = 4,
        card_vocab: int = 53,
        card_emb_dim: int = 64,
        hidden: int = 768,
        action_hidden: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.n_players = n_players  # public/legacy attribute

        self.state_encoder = SetPoolStateEncoder(
            n_players=n_players, card_vocab=card_vocab, card_emb_dim=card_emb_dim, hidden=hidden
        )
        self.action_encoder = ActionEncoder(hidden=action_hidden, activation="gelu", dropout=DROPOUT_RATE)
        self.q_scorer = DotProductScorer(state_dim=hidden, action_dim=action_hidden, scale=True)

        # Legacy attribute aliases (kept for backward compatibility / introspection tooling).
        self.card_emb = self.state_encoder.card_emb
        self.hand_enc = self.state_encoder.hand_enc
        self.action_enc = self.action_encoder.mlp.net
        self.state_to_action = self.q_scorer.proj

    def forward_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state_tensor)

    def encode_actions(self, actions_batch: list[list[np.ndarray]]) -> torch.Tensor:
        feats, _counts = _encode_actions_to_tensor(actions_batch, device=torch.device(self.device))
        return feats

    def q_values(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]) -> list[torch.Tensor]:
        state_h = self.forward_state(state_tensor)  # (B, hidden)
        return self._score_candidates_dot(
            state_h=state_h,
            actions_batch=actions_batch,
            action_encoder=self.action_encoder,
            scorer=self.q_scorer,
            device=state_tensor.device,
        )

    def forward(self, state_tensor: torch.Tensor, actions_batch: list[list[np.ndarray]]) -> list[torch.Tensor]:
        return self.q_values(state_tensor, actions_batch)
