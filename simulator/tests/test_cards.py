from simulator.cards import (
    card_rank, card_suit, card_name, sort_cards, is_consecutive,
    cmp_single_hand, hand_to_combo, compare_combos,
    Combo, PASS, SINGLE, PAIR, TRIPLE, STRAIGHT, FLUSH, FULLHOUSE, 
    FOUR_KIND, STRAIGHT_FLUSH,
)


class TestCardBasics:
    """Tests for basic card functions: card_rank, card_suit, card_name"""
    
    def test_card_rank(self):
        # Test first card of each rank
        assert card_rank(0) == 0 
        assert card_rank(4) == 1 
        assert card_rank(48) == 12 
        
        # Test all suits of same rank
        assert card_rank(0) == card_rank(1) == card_rank(2) == card_rank(3) == 0
        
    def test_card_suit(self):
        assert card_suit(0) == 0 
        assert card_suit(1) == 1 
        assert card_suit(2) == 2 
        assert card_suit(3) == 3 
        
        # Test same suit across ranks
        assert card_suit(0) == card_suit(4) == card_suit(8) == 0 
        
    def test_card_name(self):
        # Test specific cards
        assert card_name(0) == "3♦"
        assert card_name(1) == "3♣"
        assert card_name(2) == "3♥"
        assert card_name(3) == "3♠"
        assert card_name(48) == "2♦"
        assert card_name(51) == "2♠"
        
        # Test face cards
        assert card_name(28) == "T♦"
        assert card_name(32) == "J♦"
        assert card_name(36) == "Q♦"
        assert card_name(40) == "K♦"
        assert card_name(44) == "A♦"


class TestSortCards:
    """Tests for sort_cards function"""
    
    def test_sort_empty(self):
        assert sort_cards([]) == []
    
    def test_sort_single(self):
        assert sort_cards([5]) == [5]
    
    def test_sort_already_sorted(self):
        assert sort_cards([0, 1, 2, 3]) == [0, 1, 2, 3]
    
    def test_sort_reverse(self):
        assert sort_cards([51, 48, 44, 40]) == [40, 44, 48, 51]
    
    def test_sort_mixed(self):
        assert sort_cards([10, 5, 20, 15]) == [5, 10, 15, 20]


class TestIsConsecutive:
    """Tests whether is_consecutive can detect straights"""
    
    def test_consecutive_valid(self):
        # 3, 4, 5, 6, 7
        assert is_consecutive([0, 1, 2, 3, 4]) == True
        
        # 9, 10, J, Q, K (ranks 6-10)
        assert is_consecutive([6, 7, 8, 9, 10]) == True
    
    def test_consecutive_with_ace_high(self):
        # 10, J, Q, K, A (ranks 7-11)
        assert is_consecutive([7, 8, 9, 10, 11]) == True
    
    def test_not_consecutive_with_2(self):
        # 2 (rank 12) cannot be in a straight in Big 2
        assert is_consecutive([8, 9, 10, 11, 12]) == False
        assert is_consecutive([12]) == False
    
    def test_not_consecutive_gap(self):
        # Has a gap
        assert is_consecutive([0, 1, 3, 4, 5]) == False
        assert is_consecutive([0, 2, 4, 6, 8]) == False
    
    def test_not_consecutive_duplicate(self):
        # Duplicates break consecutiveness
        assert is_consecutive([0, 0, 1, 2, 3]) == False
    
    def test_single_element(self):
        assert is_consecutive([5]) == True
    
    def test_two_consecutive(self):
        assert is_consecutive([5, 6]) == True
        assert is_consecutive([5, 7]) == False


class TestCmpSingleHand:
    """Tests for cmp_single_hand function"""
    
    def test_different_ranks(self):
        # 3♦ vs 4♦ (different ranks)
        assert cmp_single_hand([0], [4]) < 0
        assert cmp_single_hand([4], [0]) > 0
        
        # 2♠ vs 3♦ (2 is highest)
        assert cmp_single_hand([51], [0]) > 0
    
    def test_same_rank_different_suits(self):
        # 3♦ vs 3♣ vs 3♥ vs 3♠ (suits: ♦=0 < ♣=1 < ♥=2 < ♠=3)
        assert cmp_single_hand([0], [1]) < 0
        assert cmp_single_hand([1], [2]) < 0
        assert cmp_single_hand([2], [3]) < 0
        assert cmp_single_hand([3], [0]) > 0
    
    def test_same_card(self):
        assert cmp_single_hand([10], [10]) == 0


class TestCardToCombo:
    """Tests for hand_to_combo function"""
    
    def test_pass(self):
        combo = hand_to_combo([])
        assert combo.type == PASS
        assert combo.cards == []
        assert combo.key == ()
    
    def test_single(self):
        combo = hand_to_combo([5])
        assert combo.type == SINGLE
        assert combo.cards == [5]
        assert combo.key == (1, 1)
        
        combo = hand_to_combo([48])
        assert combo.type == SINGLE
        assert combo.cards == [48]
        assert combo.key == (12, 0)
    
    def test_pair_valid(self):
        # 3♦ 3♣ (rank=0, suits 0,1)
        combo = hand_to_combo([0, 1])
        assert combo.type == PAIR
        assert combo.cards == [0, 1]
        assert combo.key == (0, 1)
        
        # 2♥ 2♠ (rank=12, suits 2,3)
        combo = hand_to_combo([50, 51])
        assert combo.type == PAIR
        assert combo.key == (12, 3)
    
    def test_pair_invalid(self):
        # Different ranks
        combo = hand_to_combo([0, 4])
        assert combo is None
    
    def test_triple_valid(self):
        # 3♦ 3♣ 3♥ (rank=0, suits 0,1,2)
        combo = hand_to_combo([0, 1, 2])
        assert combo.type == TRIPLE
        assert combo.cards == [0, 1, 2]
        assert combo.key == (0)

        combo = hand_to_combo([48, 49, 50])
        assert combo.type == TRIPLE
        assert combo.key == (12)
    
    def test_triple_invalid(self):
        # Not all same rank
        combo = hand_to_combo([0, 1, 4])
        assert combo is None

        combo = hand_to_combo([47, 49, 51])
        assert combo is None
    
    def test_straight_valid(self):        
        # Mixed suits
        combo = hand_to_combo([0, 5, 8, 13, 16])
        assert combo.type == STRAIGHT
        assert combo.key == (4, 0)
    
    def test_straight_with_2_invalid(self):
        # 2 cannot be in a straight
        combo = hand_to_combo([32, 36, 41, 44, 48])
        assert combo is None
    
    def test_flush_valid(self):
        # All diamonds, not consecutive
        combo = hand_to_combo([0, 4, 8, 16, 20])
        assert combo.type == FLUSH
        assert len(combo.cards) == 5
        # Key: ranks sorted desc + max_suit
        assert combo.key == (0, 5, 4, 2, 1, 0)
    
    def test_fullhouse_valid(self):
        # 3♦ 3♣ 3♥ 4♦ 4♣ (triple 3s, pair 4s)
        combo = hand_to_combo([0, 1, 2, 4, 5])
        assert combo.type == FULLHOUSE
        assert combo.key == (0, 1)
        
        # 4♦ 4♣ 3♥ 3♠ 3♦ (triple 3s, pair 4s) - unsorted input
        combo = hand_to_combo([4, 5, 2, 3, 0])
        assert combo.type == FULLHOUSE
        assert combo.key == (0, 1)

    def test_fullhouse_invalid(self):
        combo = hand_to_combo([12, 13, 14, 20, 37])
        assert combo is None
    
    def test_four_kind_valid(self):
        # 3♦ 3♣ 3♥ 3♠ 4♦ (four 3s, kicker 4)
        combo = hand_to_combo([0, 1, 2, 3, 4])
        assert combo.type == FOUR_KIND
        assert combo.key == (0)
    
    def test_straight_flush_valid(self):
        # 3♦ 4♦ 5♦ 6♦ 7♦ (all diamonds, consecutive)
        combo = hand_to_combo([0, 4, 8, 12, 16])
        assert combo.type == STRAIGHT_FLUSH
        assert combo.key == (0, 4)
        
        # 9♠ 10♠ J♠ Q♠ K♠ (all spades, consecutive)
        combo = hand_to_combo([27, 31, 35, 39, 43])
        assert combo.type == STRAIGHT_FLUSH
        assert combo.key == (3, 10)
    
    def test_five_cards_invalid(self):
        # 5 cards that don't form any valid combo
        combo = hand_to_combo([0, 4, 8, 12, 21])
        assert combo is None
    
    def test_invalid_sizes(self):
        # 4 cards
        combo = hand_to_combo([0, 1, 2, 3])
        assert combo is None
        
        # 6 cards
        combo = hand_to_combo([0, 1, 2, 3, 4, 5])
        assert combo is None


class TestCompareCombos:
    """Tests for compare_combos function"""
    
    def test_different_types(self):
        # Different types return 0 (undefined)
        combo1 = hand_to_combo([0])
        combo2 = hand_to_combo([0, 1])
        assert combo1.type == SINGLE
        assert combo2.type == PAIR
        assert compare_combos(combo1, combo2) == 0

        # Different sizes return 0
        combo1 = hand_to_combo([0, 1])
        combo2 = hand_to_combo([0, 1, 2, 3, 4])
        assert compare_combos(combo1, combo2) == 0
            
    
    def test_compare_singles(self):
        combo1 = hand_to_combo([0])
        combo2 = hand_to_combo([4])
        assert compare_combos(combo1, combo2) < 0
        assert compare_combos(combo2, combo1) > 0
        
        # Same card
        combo3 = hand_to_combo([4])
        assert compare_combos(combo2, combo3) == 0
    
    def test_compare_pairs(self):
        combo1 = hand_to_combo([0, 1])
        combo2 = hand_to_combo([4, 5])
        assert compare_combos(combo1, combo2) < 0
        
        # Same rank, different max suits
        combo3 = hand_to_combo([0, 2])
        assert compare_combos(combo1, combo3) < 0
    
    def test_compare_triples(self):
        combo1 = hand_to_combo([0, 1, 2])
        combo2 = hand_to_combo([4, 5, 6])
        assert compare_combos(combo1, combo2) < 0
    
    def test_compare_straights(self):
        # Lower straight vs higher straight
        combo1 = hand_to_combo([0, 4, 8, 12, 17])
        combo2 = hand_to_combo([4, 8, 12, 16, 21])
        assert compare_combos(combo1, combo2) < 0

        combo3 = hand_to_combo([1, 5, 9, 13, 18])
        assert compare_combos(combo1, combo3) < 0
        assert compare_combos(combo2, combo3) > 0
    
    def test_compare_flushes(self):
        # Flushes compared by suit then ranks
        combo1 = hand_to_combo([0, 4, 8, 12, 16])
        combo2 = hand_to_combo([4, 8, 12, 16, 20])
        assert compare_combos(combo1, combo2) < 0

        combo3 = hand_to_combo([1, 5, 9, 13, 17])
        assert compare_combos(combo1, combo3) < 0
        assert compare_combos(combo2, combo3) < 0
    
    def test_compare_fullhouses(self):
        # Full house 3s over 4s vs 4s over 3s
        combo1 = hand_to_combo([0, 1, 2, 4, 5])
        combo2 = hand_to_combo([4, 5, 6, 0, 1])
        assert compare_combos(combo1, combo2) < 0
    
    def test_compare_four_kinds(self):
        combo1 = hand_to_combo([0, 1, 2, 3, 4])
        combo2 = hand_to_combo([4, 5, 6, 7, 0])
        assert compare_combos(combo1, combo2) < 0
    
    def test_compare_straight_flushes(self):
        # 3-7 straight flush vs 4-8 straight flush
        combo1 = hand_to_combo([0, 4, 8, 12, 16])
        combo2 = hand_to_combo([1, 5, 9, 13, 17])
        assert compare_combos(combo1, combo2) < 0


class TestComboClass:
    """Tests for the Combo dataclass"""
    
    def test_combo_size(self):
        combo1 = Combo(SINGLE, [5], (1, 1))
        assert combo1.size() == 1
        
        combo2 = Combo(PAIR, [0, 1], (0, 1))
        assert combo2.size() == 2
        
        combo3 = Combo(STRAIGHT, [0, 4, 8, 12, 16], (4, 0))
        assert combo3.size() == 5
        
        combo_pass = Combo(PASS, [], ())
        assert combo_pass.size() == 0

