import numpy as np
import pytest
import torch

from big2.nn import SetPoolPolicy, combo_to_action_vector, input_dim
from big2.simulator.cards import SINGLE, Combo
from big2.simulator.env import Big2Env


def test_input_dim_with_small_card_universe():
    # n_players=4, cards_per_player=3 => card_count=12
    # hand(3) + last_play(12) + seen(12) + counts(3) + passes(1) + opp_cards(3*12)
    assert input_dim(n_players=4, cards_per_player=3) == 67


def test_combo_to_action_vector_respects_card_count():
    cmb = Combo(SINGLE, [0], (0, 0))
    vec = combo_to_action_vector(cmb)
    assert vec.shape[0] == 80
    vec_small = combo_to_action_vector(cmb, card_count=12)
    assert vec_small.shape[0] == 40
    assert vec_small[0] == 1.0
    with pytest.raises(ValueError):
        combo_to_action_vector(Combo(SINGLE, [12], (3, 0)), card_count=12)


def test_setpool_policy_forward_with_small_card_universe():
    env = Big2Env(n_players=4, cards_per_player=3)
    state = env.reset()
    candidates = env.legal_candidates(env.current_player)
    if not candidates:
        pytest.skip("No legal candidates available")

    policy = SetPoolPolicy(
        n_players=4,
        cards_per_player=3,
        hidden=64,
        action_hidden=32,
        device="cpu",
    ).to("cpu")
    policy.eval()

    st = torch.from_numpy(np.array([state])).long()
    action_feats = [[combo_to_action_vector(c, card_count=policy.card_universe_size) for c in candidates]]
    logits_list, values = policy(st, action_feats)
    assert len(logits_list) == 1
    assert logits_list[0].shape[0] == len(candidates)
    assert values.shape[0] == 1
