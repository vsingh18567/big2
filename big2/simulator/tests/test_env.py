import random

import numpy as np

from ..cards import (
    CARDS_PER_DECK,
    PAIR,
    PASS,
    SINGLE,
    STRAIGHT,
    Combo,
    card_rank,
    compare_combos,
    hand_to_combo,
)
from ..env import Big2Env


class TestEnv:
    """Tests for Big2Env class"""

    def test_env_initialization(self):
        """Test basic environment initialization"""
        env = Big2Env(n_players=4)
        assert env.n_players == 4
        assert len(env.hands) == 4
        assert len(env.hands[0]) == 13
        assert len(env.hands[1]) == 13
        assert len(env.hands[2]) == 13
        assert len(env.hands[3]) == 13
        assert env.done is False
        assert env.winner is None
        assert env.passes_in_row == 0

    def test_env_reset(self):
        """Test environment reset functionality"""
        env = Big2Env(n_players=4)

        # Check all cards are dealt
        all_cards = set()
        for hand in env.hands:
            all_cards.update(hand)
        assert len(all_cards) == CARDS_PER_DECK
        assert all_cards == set(range(CARDS_PER_DECK))

        # Check current player has 3 of diamonds (card 0)
        assert 0 in env.hands[env.current_player]

        # Check state tracking
        assert sum(env.seen) == 0
        assert env.trick_pile is None

    def test_observation_structure(self):
        """Test observation vector structure"""
        env = Big2Env(n_players=4)
        obs = env.reset()

        # Observation should have:
        # - cards_per_player hand slots (card ids, -1 padded)
        # - 52 last play one-hot
        # - 52 seen cards
        # - (n_players - 1) opponent counts
        # Total: cards_per_player + 52 + 52 + (n_players - 1)
        cards_per_player = CARDS_PER_DECK // env.n_players
        expected_len = cards_per_player + 52 + 52 + (env.n_players - 1)
        assert len(obs) == expected_len
        assert obs.dtype == np.int32

        # Check hand portion (first cards_per_player elements)
        hand_portion = obs[:cards_per_player]
        hand = env.hands[env.current_player]
        assert len([x for x in hand_portion if x != -1]) == len(hand)
        for card in hand:
            assert card in hand_portion

        # Check last play is all zeros initially (no plays yet)
        last_play_portion = obs[cards_per_player:cards_per_player+52]
        assert np.sum(last_play_portion) == 0

        # Check seen is all zeros initially
        seen_portion = obs[cards_per_player+52:cards_per_player+104]
        assert np.sum(seen_portion) == 0

        # Check opponent counts (should all be cards_per_player initially)
        opponent_counts = obs[cards_per_player+104:]
        assert all(c == cards_per_player for c in opponent_counts)

    def test_single_play_and_state_update(self):
        """Test playing a single card and checking state updates"""
        random.seed(42)
        env = Big2Env(n_players=4)
        env.reset()

        initial_player = env.current_player
        initial_hand = env.hands[initial_player].copy()

        # Play the 3 of diamonds (card 0)
        assert 0 in env.hands[initial_player]
        action = hand_to_combo([0])
        assert action is not None

        obs, reward, done, info = env.step(action)

        # Check state updates
        assert env.trick_pile is not None
        assert env.trick_pile.type == SINGLE
        assert env.trick_pile.cards == [0]
        assert env.seen[0] == 1
        assert 0 not in env.hands[initial_player]
        assert len(env.hands[initial_player]) == len(initial_hand) - 1
        assert env.passes_in_row == 0
        assert reward == 0  # Game not won yet
        assert done is False

        # Check current player advanced
        assert env.current_player == (initial_player + 1) % env.n_players

        # Check observation reflects the play
        cards_per_player = CARDS_PER_DECK // env.n_players
        last_play_portion = obs[cards_per_player:cards_per_player+52]
        assert last_play_portion[0] == 1  # Card 0 was played
        assert np.sum(last_play_portion) == 1  # Only one card

    def test_passing_mechanism(self):
        """Test passing and trick pile reset"""
        random.seed(123)
        env = Big2Env(n_players=4)
        env.reset()

        # Play a card to start the trick
        action = hand_to_combo([0])  # Play 3 of diamonds
        assert action is not None
        env.step(action)

        # Now have 3 players pass
        pass_action = Combo(PASS, [], ())

        # First pass
        obs, reward, done, info = env.step(pass_action)
        assert env.passes_in_row == 1
        assert env.trick_pile is not None  # Trick still active

        # Second pass
        obs, reward, done, info = env.step(pass_action)
        assert env.passes_in_row == 2
        assert env.trick_pile is not None  # Trick still active

        # Third pass - should reset the trick
        obs, reward, done, info = env.step(pass_action)
        assert env.passes_in_row == 0  # Reset
        assert env.trick_pile is None  # Trick cleared

        cards_per_player = CARDS_PER_DECK // env.n_players
        last_play_portion = obs[cards_per_player:cards_per_player+52]  # type: ignore[unreachable]
        assert np.sum(last_play_portion) == 0

    def test_legal_candidates_fresh_trick(self):
        """Test legal candidates when starting a fresh trick"""
        random.seed(456)
        env = Big2Env(n_players=4)
        env.reset()

        player = env.current_player
        candidates = env.legal_candidates(player)

        # Should have many candidates for fresh trick
        assert len(candidates) > 0

        # All should be non-PASS
        assert all(c.type != PASS for c in candidates)

        # Should include the 3 of diamonds (card 0)
        has_three_diamonds = any(c.cards == [0] for c in candidates)
        assert has_three_diamonds

    def test_legal_candidates_must_follow(self):
        """Test legal candidates when must follow previous play"""
        random.seed(789)
        env = Big2Env(n_players=4)
        env.reset()

        # Play a single card
        action = hand_to_combo([0])  # Play 3 of diamonds
        assert action is not None
        env.step(action)

        # Next player must play higher single or pass
        next_player = env.current_player
        candidates = env.legal_candidates(next_player)

        # Should have PASS as an option
        has_pass = any(c.type == PASS for c in candidates)
        assert has_pass

        # All non-PASS candidates should be singles higher than card 0
        non_pass_candidates = [c for c in candidates if c.type != PASS]
        for c in non_pass_candidates:
            assert c.type == SINGLE
            assert c.cards[0] > 0  # Higher than 3 of diamonds

    def test_pair_play(self):
        """Test playing a pair"""
        random.seed(111)
        env = Big2Env(n_players=4)
        env.reset()

        # Find a player with a pair
        player = env.current_player
        hand = env.hands[player]

        # Look for cards with same rank
        by_rank: dict[int, list[int]] = {}
        for card in hand:
            rank = card_rank(card)
            by_rank.setdefault(rank, []).append(card)

        # Find first pair
        pair_cards = None
        for _rank, cards in by_rank.items():
            if len(cards) >= 2:
                pair_cards = sorted(cards[:2])
                break

        if pair_cards:
            initial_hand_size = len(env.hands[player])
            action = hand_to_combo(pair_cards)
            assert action is not None
            assert action.type == PAIR

            obs, reward, done, info = env.step(action)

            # Check pair was played
            assert env.trick_pile is not None
            assert env.trick_pile.type == PAIR
            assert len(env.trick_pile.cards) == 2
            assert len(env.hands[player]) == initial_hand_size - 2

            # Check seen cards updated
            for card in pair_cards:
                assert env.seen[card] == 1

    def test_winning_condition(self):
        """Test winning when a player runs out of cards"""
        env = Big2Env(n_players=4)
        env.reset()

        player = env.current_player

        # Manually set up a winning scenario - player has only one card
        env.hands[player] = [0]  # Only 3 of diamonds
        env.trick_pile = None  # Fresh trick

        action = hand_to_combo([0])
        assert action is not None
        obs, reward, done, info = env.step(action)

        # Check winning conditions
        assert done is True
        assert env.winner == player
        assert reward == 1
        assert len(env.hands[player]) == 0

    def test_full_game_simulation(self):
        """Test a full game with multiple rounds"""
        random.seed(2024)
        env = Big2Env(n_players=4)
        env.reset()

        moves_played = 0
        max_moves = 200  # Safety limit

        while not env.done and moves_played < max_moves:
            player = env.current_player
            candidates = env.legal_candidates(player)

            assert len(candidates) > 0, "Player should always have at least one legal move"

            # Simple strategy: play the first legal candidate
            action = candidates[0]

            obs, reward, done, info = env.step(action)
            moves_played += 1

            # Verify observation structure remains valid
            cards_per_player = CARDS_PER_DECK // env.n_players
            expected_obs_len = cards_per_player + 52 + 52 + (env.n_players - 1)
            assert len(obs) == expected_obs_len

            # Verify hand sizes
            total_cards = sum(len(h) for h in env.hands)
            seen_cards = sum(env.seen)
            assert total_cards + seen_cards == CARDS_PER_DECK

        # Game should complete within reasonable number of moves
        assert moves_played < max_moves
        assert env.done
        assert env.winner is not None
        assert 0 <= env.winner < env.n_players
        assert len(env.hands[env.winner]) == 0

    def test_five_card_combo_play(self):
        """Test playing five-card combos (straight, flush, etc.)"""
        env = Big2Env(n_players=4)
        env.reset()

        player = env.current_player

        # Manually set up a hand with a straight (3-4-5-6-7 in different suits)
        # Card IDs: rank*4 + suit
        # 3 of diamonds=0, 4 of clubs=5, 5 of hearts=10, 6 of spades=15, 7 of diamonds=16
        straight_cards = [0, 5, 10, 15, 16]
        env.hands[player] = straight_cards.copy()
        env.trick_pile = None  # Fresh trick

        action = hand_to_combo(straight_cards)
        assert action is not None
        assert action.type == STRAIGHT

        obs, reward, done, info = env.step(action)

        # Check the straight was played
        assert env.trick_pile is not None
        assert env.trick_pile.type == STRAIGHT  # type: ignore[unreachable]
        assert len(env.trick_pile.cards) == 5
        assert sorted(env.trick_pile.cards) == sorted(straight_cards)

    def test_opponent_card_counts_in_observation(self):
        """Test that opponent card counts are correctly tracked in observation"""
        random.seed(555)
        env = Big2Env(n_players=4)
        env.reset()

        # Play several cards from different players
        for _ in range(8):
            if env.done:
                break

            player = env.current_player
            candidates = env.legal_candidates(player)
            action = candidates[0]
            obs, _, _, _ = env.step(action)

        if not env.done:
            # Check opponent counts in observation
            cards_per_player = CARDS_PER_DECK // env.n_players
            opponent_counts = obs[cards_per_player+104:]
            current = env.current_player

            # Verify counts match actual hand sizes
            expected_counts = []
            for i in range(1, env.n_players):
                opponent_idx = (current + i) % env.n_players
                expected_counts.append(len(env.hands[opponent_idx]))

            assert list(opponent_counts) == expected_counts

    def test_seen_cards_tracking(self):
        """Test that seen cards are properly tracked"""
        random.seed(666)
        env = Big2Env(n_players=4)
        env.reset()

        played_cards = set()

        # Play several moves
        for _ in range(10):
            if env.done:
                break

            candidates = env.legal_candidates(env.current_player)
            # Find a non-pass action
            action = None
            for c in candidates:
                if c.type != PASS:
                    action = c
                    break

            if action is None:
                action = candidates[0]  # Must be pass

            if action.type != PASS:
                played_cards.update(action.cards)

            obs, _, _, _ = env.step(action)

        # Verify seen cards match what was played
        for card in range(CARDS_PER_DECK):
            if card in played_cards:
                assert env.seen[card] == 1
            else:
                assert env.seen[card] == 0

    def test_combo_comparison_in_gameplay(self):
        """Test that only valid higher combos are accepted"""
        env = Big2Env(n_players=4)
        env.reset()

        player = env.current_player

        # Set up scenario: player 0 plays card 0 (3 of diamonds)
        env.hands[player] = [0, 4, 8, 12]  # 3♦, 4♦, 5♦, 6♦
        env.trick_pile = None

        # Play 3 of diamonds
        action = hand_to_combo([0])
        assert action is not None
        env.step(action)

        # Next player - check they can only play higher cards
        next_player = env.current_player
        env.hands[next_player] = [1, 2, 3]  # Other 3s (higher suits)

        candidates = env.legal_candidates(next_player)
        non_pass = [c for c in candidates if c.type != PASS]

        # All should be higher than card 0
        for c in non_pass:
            assert c.type == SINGLE
            reference_combo = hand_to_combo([0])
            assert reference_combo is not None
            result = compare_combos(c, reference_combo)
            assert result > 0

    def test_multiple_tricks(self):
        """Test that multiple tricks work correctly with resets"""
        random.seed(777)
        env = Big2Env(n_players=4)
        env.reset()

        tricks_completed = 0
        moves = 0
        max_moves = 150

        while not env.done and moves < max_moves:
            was_fresh = env.trick_pile is None

            candidates = env.legal_candidates(env.current_player)
            action = candidates[0]

            obs, _, _, _ = env.step(action)
            moves += 1

            # If trick became fresh (was not None, now is None), a trick completed
            if env.trick_pile is None and not was_fresh:
                tricks_completed += 1

        # Should have completed multiple tricks
        assert tricks_completed >= 2

    def test_state_consistency_throughout_game(self):
        """Comprehensive test checking state consistency throughout a game"""
        random.seed(999)
        env = Big2Env(n_players=4)
        initial_obs = env.reset()

        # Verify initial state
        cards_per_player = CARDS_PER_DECK // env.n_players
        expected_obs_len = cards_per_player + 52 + 52 + (env.n_players - 1)
        assert len(initial_obs) == expected_obs_len
        assert sum(len(h) for h in env.hands) == CARDS_PER_DECK
        assert sum(env.seen) == 0

        moves = 0
        max_moves = 200

        while not env.done and moves < max_moves:
            player = env.current_player
            hand_before = env.hands[player].copy()
            seen_before = sum(env.seen)

            candidates = env.legal_candidates(player)
            assert len(candidates) > 0

            # Pick non-pass if available
            action = candidates[0]
            for c in candidates:
                if c.type != PASS:
                    action = c
                    break

            obs, reward, done, info = env.step(action)
            moves += 1

            # Verify consistency
            cards_per_player = CARDS_PER_DECK // env.n_players
            expected_obs_len = cards_per_player + 52 + 52 + (env.n_players - 1)
            assert len(obs) == expected_obs_len

            # Check cards are properly removed from hand
            if action.type != PASS:
                for card in action.cards:
                    assert card not in env.hands[player]
                    assert card in hand_before
                    assert env.seen[card] == 1

                # Seen count should increase
                assert sum(env.seen) == seen_before + len(action.cards)

            # Check total cards in system
            total_in_hands = sum(len(h) for h in env.hands)
            total_seen = sum(env.seen)
            assert total_in_hands + total_seen == CARDS_PER_DECK

            # If game done, verify winner
            if done:
                assert env.winner == player  # Current player just won
                assert len(env.hands[player]) == 0
                assert reward == 1

        assert env.done
        assert moves < max_moves
