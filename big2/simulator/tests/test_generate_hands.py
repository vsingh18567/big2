from big2.simulator.cards import sort_cards
from big2.simulator.generate_hands import (
    generate_flushes,
    generate_fourkind,
    generate_fullhouses,
    generate_pairs,
    generate_singles,
    generate_straightflushes,
    generate_straights,
    generate_triples,
)


class TestGenerateHands:
    """Tests for generate_hands function"""

    def test_generate_hands(self):
        # Full 13-card hand
        hand = [3, 36, 28, 8, 32, 27, 51, 22, 41, 18, 29, 49, 45]
        hand = sort_cards(hand)
        singles = generate_singles(hand)
        assert len(singles) == 13
        pairs = generate_pairs(hand)
        assert len(pairs) == 2
        triples = generate_triples(hand)
        assert len(triples) == 0
        straights = generate_straights(hand)
        assert len(straights) == 8
        flushes = generate_flushes(hand)
        assert len(flushes) == 0
        fullhouses = generate_fullhouses(hand)
        assert len(fullhouses) == 0
        four_kinds = generate_fourkind(hand)
        assert len(four_kinds) == 0
        straight_flushes = generate_straightflushes(hand)
        assert len(straight_flushes) == 0

    def test_generate_hands_2(self):
        # Small hand with just a straight flush
        hand = [0, 4, 8, 12, 16]

        assert len(generate_straights(hand)) == 1
        assert len(generate_flushes(hand)) == 1
        assert len(generate_fullhouses(hand)) == 0
        assert len(generate_fourkind(hand)) == 0
        assert len(generate_straightflushes(hand)) == 1
