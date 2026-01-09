from big2.simulator.cards import (
    FOUR_KIND,
    PAIR,
    PASS,
    SINGLE,
    STRAIGHT,
    Combo,
    card_rank,
    card_suit,
    is_consecutive,
)


def is_power_card(card_id: int) -> bool:
    """Check if card is a 2 (rank 12, IDs 48-51)."""
    return card_rank(card_id) == 12


def count_power_cards(cards: list[int]) -> int:
    """Count number of 2s (power cards) in a combo."""
    return sum(1 for c in cards if is_power_card(c))


def game_phase(hand_size: int) -> str:
    """
    Determine game phase based on hand size.
    Returns: 'early' (>10 cards), 'mid' (6-10 cards), 'late' (≤5 cards)
    """
    if hand_size > 10:
        return "early"
    elif hand_size > 5:
        return "mid"
    else:
        return "late"


def would_break_combo(action: Combo, hand: list[int]) -> int:
    """
    Check if playing this action would break apart a valuable combo.

    Returns:
        0 = no breaking
        1 = breaking pair/triple
        2 = breaking 5-card combo (straight, flush, full house, four of a kind)

    Checks for:
    - Breaking pairs/triples
    - Breaking potential straights (5 consecutive ranks)
    - Breaking potential flushes (5+ cards of same suit)
    - Breaking potential full houses (triple + pair)
    - Breaking potential four of a kind
    """
    if action.type == PASS:
        return 0

    # Count occurrences of each rank in the hand
    rank_counts: dict[int, int] = {}
    for card in hand:
        rank = card_rank(card)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    breaks_pair_or_triple = False

    # Check if we're breaking apart pairs/triples
    action_ranks = [card_rank(c) for c in action.cards]
    for rank in action_ranks:
        count_in_hand = rank_counts.get(rank, 0)
        count_in_action = action_ranks.count(rank)

        # If we have more cards of this rank than we're playing, we might be breaking a combo
        if count_in_hand > count_in_action:
            # Check if remaining cards could form a pair/triple
            remaining = count_in_hand - count_in_action
            if remaining >= 2 and action.type == SINGLE:
                breaks_pair_or_triple = True
            if remaining >= 3 and action.type in (SINGLE, PAIR):
                breaks_pair_or_triple = True

    # Check for breaking 5-card combos
    # Only check if we're not already playing a 5-card combo
    if action.type < STRAIGHT:
        # Create a set of action cards for quick lookup
        action_card_set = set(action.cards)
        remaining_hand = [c for c in hand if c not in action_card_set]

        # Check for breaking potential four of a kind
        for rank, count in rank_counts.items():
            if count >= 4:
                # We have 4 of a kind - check if we're breaking it
                cards_of_rank_in_action = sum(1 for c in action.cards if card_rank(c) == rank)
                if cards_of_rank_in_action > 0 and cards_of_rank_in_action < 4:
                    return 2  # Breaking 5-card combo

        # Check for breaking potential full house
        # Need a triple and a pair
        trips = [r for r, c in rank_counts.items() if c >= 3]
        pairs = [r for r, c in rank_counts.items() if c >= 2]

        if len(trips) >= 1 and len(pairs) >= 2:  # Can form full house (trip + pair from different rank)
            # Check if we're breaking the components
            for rank in action_ranks:
                count_in_hand = rank_counts.get(rank, 0)
                count_in_action = action_ranks.count(rank)
                remaining = count_in_hand - count_in_action

                # Breaking a triple down below 3 or a pair down below 2
                if count_in_hand >= 3 and remaining < 3 and remaining > 0:
                    return 2  # Breaking 5-card combo
                if count_in_hand >= 2 and remaining < 2 and remaining > 0:
                    # Only penalize breaking pairs if we have multiple pairs for full house potential
                    if len(pairs) >= 2:
                        return 2  # Breaking 5-card combo

        # Check for breaking potential flush (5+ cards of same suit)
        suit_counts: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        for card in hand:
            suit_counts[card_suit(card)].append(card)

        for suit, cards_in_suit in suit_counts.items():
            if len(cards_in_suit) >= 5:
                # We have a potential flush - check if we're breaking it
                cards_of_suit_in_action = [c for c in action.cards if card_suit(c) == suit]
                if len(cards_of_suit_in_action) > 0:
                    # We're taking from this suit - would we still have 5 after?
                    if len(cards_in_suit) - len(cards_of_suit_in_action) < 5:
                        return 2  # Breaking 5-card combo

        # Check for breaking potential straight (5 consecutive ranks, no 2s)
        # Get unique ranks in hand (excluding 2s which can't be in straights)
        unique_ranks = sorted([r for r in set(rank_counts.keys()) if r != 12])

        # Find longest consecutive sequence
        if len(unique_ranks) >= 5:
            # Check all possible 5-card straights
            for i in range(len(unique_ranks) - 4):
                window = unique_ranks[i : i + 5]
                if is_consecutive(window):
                    # We have a potential straight - check if we're breaking it
                    straight_ranks = set(window)
                    action_ranks_set = set(action_ranks)

                    # If we're taking cards from this straight
                    if straight_ranks & action_ranks_set:
                        # Check if we'd still have all 5 ranks after playing action
                        remaining_ranks = {card_rank(c) for c in remaining_hand}
                        if not straight_ranks.issubset(remaining_ranks):
                            return 2  # Breaking 5-card combo

    # Return 1 if breaking pair/triple, 0 if not breaking anything
    return 1 if breaks_pair_or_triple else 0


def is_very_strong_play(trick_pile: Combo) -> bool:
    """
    Check if the current trick pile represents a very strong play.
    Strong plays include high-value combos or power cards.
    """
    if trick_pile.type == PASS:
        return False

    # High-value combo types
    if trick_pile.type >= STRAIGHT:
        return True

    # Power cards (2s)
    if count_power_cards(trick_pile.cards) > 0:
        return True

    # High singles (Ace or King)
    if trick_pile.type == SINGLE:
        rank = trick_pile.key[0] if trick_pile.key else card_rank(trick_pile.cards[0])
        if rank >= 10:  # K (10) or A (11)
            return True

    return False


def count_orphan_cards(remaining_hand: list[int]) -> int:
    """Count low singles (rank < 5, i.e., 3-6) that would be hard to play."""
    if not remaining_hand:
        return 0
    rank_counts: dict[int, int] = {}
    for card in remaining_hand:
        rank = card_rank(card)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    # Count singles with rank < 5 (cards 3, 4, 5, 6)
    return sum(1 for rank, count in rank_counts.items() if count == 1 and rank < 5)


def score_combo(combo: Combo, hand: list[int], trick_pile: Combo | None) -> float:
    """
    Score a combo based on strategic heuristics.
    Lower score is better (we want to play the lowest-scored combo).

    Hyperparameters optimized through grid search (Anti-Greedy config).
    """
    score = 0.0
    hand_size = len(hand)
    phase = game_phase(hand_size)

    # If this play wins the game, prioritize it heavily
    if combo.size() == hand_size:
        return -1000.0

    # Base: card strength (reduced weight - less greedy-like behavior)
    base_strength = sum(card_rank(c) for c in combo.cards)
    score += base_strength * 0.8  # Card quality should matter more

    # Moderate penalty for playing power cards (2s) early
    power_card_count = count_power_cards(combo.cards)
    if phase == "early":
        score += power_card_count * 10  # Moderate penalty for playing 2s early
    elif phase == "mid":
        score += power_card_count * 5  # Light penalty in mid game
    # Late game: no penalty, we need to get rid of cards

    # Penalty for breaking apart combos (varies by phase and combo type)
    break_type = would_break_combo(combo, hand)
    if break_type == 1:  # Breaking pair/triple
        if phase == "early":
            score += 8
        elif phase == "mid":
            score += 4
        # No penalty in late game
    elif break_type == 2:  # Breaking 5-card combo
        if phase == "early":
            score += 20
        elif phase == "mid":
            score += 8
        elif phase == "late":
            score += 4

    # Penalty for leaving orphan cards (low unplayable singles)
    remaining = [c for c in hand if c not in set(combo.cards)]
    orphans = count_orphan_cards(remaining)
    score += orphans * 6  # Make orphans actually matter

    # Strong bonus for getting rid of more cards
    score -= combo.size() * 4  # Reduce dominance of combo size

    # In late game, be aggressive
    if phase == "late":
        score -= 10  # Less blanket aggression

    # Response to opponent plays
    if trick_pile and trick_pile.type != PASS:
        # In late game, fight for control against strong plays
        if is_very_strong_play(trick_pile) and phase == "late":
            score -= 10  # Bonus to beat strong plays in late game
        elif trick_pile.type >= FOUR_KIND:  # Four of a kind or straight flush
            score += 25  # Strong penalty for unbeatable plays

    return score


def smart_strategy(combos: list[Combo], hand: list[int], trick_pile: Combo | None = None) -> Combo:
    """
    Smart rule-based strategy that makes strategic decisions about combo preservation,
    power card management, and situational play.

    Args:
        combos: List of legal combo actions
        hand: Current hand (list of card IDs)
        trick_pile: Current trick pile (None if starting a new trick)

    Returns:
        Selected Combo to play
    """
    # Filter out PASS first
    non_pass = [c for c in combos if c.type != PASS]

    # If must pass or only one option, take it
    if len(non_pass) == 0:
        # Find PASS combo
        for c in combos:
            if c.type == PASS:
                return c
        return combos[0]  # Fallback

    if len(combos) <= 1:
        return combos[0]

    # Score each combo based on heuristics
    scored_combos = [(score_combo(c, hand, trick_pile), c) for c in non_pass]

    # Check if we can pass
    can_pass = any(c.type == PASS for c in combos) and trick_pile is not None and trick_pile.type != PASS

    # Consider passing only in very specific situations
    if can_pass:
        best_score, best_combo = min(scored_combos)
        phase = game_phase(len(hand))

        # Only pass if:
        # 1. We're in early game AND the best play uses multiple power cards
        # 2. Opponent played an unbeatable 5-card combo (Four of a kind or Straight Flush)
        should_pass = False

        if phase == "early" and count_power_cards(best_combo.cards) >= 2:
            # Playing 2+ power cards in early game - consider passing
            should_pass = best_score > 30
        elif trick_pile and trick_pile.type >= FOUR_KIND:
            # Opponent played four of a kind or straight flush - hard to beat
            should_pass = True

        if should_pass:
            for c in combos:
                if c.type == PASS:
                    return c

    # Play lowest-scored (best strategic value) combo
    return min(scored_combos)[1]
