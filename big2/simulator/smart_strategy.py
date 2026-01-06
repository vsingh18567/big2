from big2.simulator.cards import (
    FOUR_KIND,
    PAIR,
    PASS,
    SINGLE,
    STRAIGHT,
    TRIPLE,
    Combo,
    card_rank,
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


def would_break_combo(action: Combo, hand: list[int]) -> bool:
    """
    Check if playing this action would break apart a pair or triple.
    Returns True if playing this combo uses cards that could form a higher-value combo.
    """
    if action.type == PASS:
        return False

    # Count occurrences of each rank in the hand
    rank_counts: dict[int, int] = {}
    for card in hand:
        rank = card_rank(card)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

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
                return True
            if remaining >= 3 and action.type in (SINGLE, PAIR):
                return True

    return False


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


def score_combo(combo: Combo, hand: list[int], trick_pile: Combo | None) -> float:
    """
    Score a combo based on strategic heuristics.
    Lower score is better (we want to play the lowest-scored combo).

    Hyperparameters optimized through grid search (Anti-Greedy config).
    """
    score = 0.0
    hand_size = len(hand)
    phase = game_phase(hand_size)

    # Base: card strength (reduced weight - less greedy-like behavior)
    base_strength = sum(card_rank(c) for c in combo.cards)
    score += base_strength * 0.5  # Lower weight = less emphasis on playing lowest cards

    # Moderate penalty for playing power cards (2s) early
    power_card_count = count_power_cards(combo.cards)
    if phase == "early":
        score += power_card_count * 10  # Moderate penalty for playing 2s early
    elif phase == "mid":
        score += power_card_count * 5  # Light penalty in mid game
    # Late game: no penalty, we need to get rid of cards

    # Light penalty for breaking apart combos (only in early game)
    if phase == "early" and would_break_combo(combo, hand):
        score += 8  # Light penalty for breaking pairs/triples

    # Strong bonus for getting rid of more cards
    score -= combo.size() * 6  # Prefer combos that play more cards

    # In late game, be aggressive
    if phase == "late":
        score -= 20  # Strong bonus to play anything in late game

    # Strong preference for combo types over singles
    if combo.type == PAIR:
        score -= 10  # Good bonus for pairs
    elif combo.type == TRIPLE:
        score -= 15  # Strong bonus for triples
    elif combo.type >= STRAIGHT:  # 5-card combos
        score -= 20  # Very strong bonus for 5-card combos

    # Response to opponent plays
    if trick_pile and trick_pile.type != PASS:
        # Only discourage playing against strong 5-card combos or 2s in late game
        if is_very_strong_play(trick_pile) and phase == "late":
            score += 15  # Light penalty when opponent plays strong in late game
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
