# tests/test_card.py
import pytest
from simulator.card import Card


def test_card_initialization():
    card = Card("S", "A")
    assert card.suit == "S"
    assert card.rank == "A"

def test_card_string_representation():
    card = Card("H", "10")
    assert str(card) == "H10"
    assert repr(card) == "Card(H10)"

def test_card_equality():
    card1 = Card("S", "A")
    card2 = Card("S", "A")
    card3 = Card("H", "A")
    
    assert card1 == card2
    assert card1 != card3
    assert card1 != "not a card"

def test_card_hash():
    card1 = Card("S", "A")
    card2 = Card("S", "A")
    card_set = {card1, card2}
    assert len(card_set) == 1

@pytest.mark.parametrize("card1,card2,expected", [
    # Different ranks
    (Card("S", "2"), Card("S", "A"), False),  # 2 is highest rank
    (Card("S", "A"), Card("S", "K"), False),
    (Card("S", "3"), Card("S", "4"), True),
    
    # Same rank, different suits
    (Card("S", "A"), Card("H", "A"), False),  # Spades > Hearts
    (Card("H", "A"), Card("D", "A"), False),  # Hearts > Diamonds
    (Card("D", "A"), Card("C", "A"), False),  # Diamonds > Clubs
    
    # Complex comparisons
    (Card("S", "2"), Card("C", "2"), False),  # Same rank (2), Spades > Clubs
    (Card("D", "3"), Card("S", "2"), True),   # Different ranks, 3 < 2
])
def test_card_comparison(card1, card2, expected):
    assert (card1 < card2) == expected
    # Test the reverse comparison
    assert (card1 >= card2) == (not expected)

def test_rank_order():
    # Test that ranks are ordered correctly (3 is lowest, 2 is highest)
    ranks = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
    cards = [Card("S", rank) for rank in ranks]
    
    # Check that each card is less than the next card
    for i in range(len(cards) - 1):
        assert cards[i] < cards[i + 1], f"Failed: {cards[i]} should be < {cards[i + 1]}"

def test_suit_order():
    # Test that suits are ordered correctly (Clubs lowest, Spades highest)
    suits = ["C", "D", "H", "S"]
    cards = [Card(suit, "A") for suit in suits]
    
    # Check that each card is less than the next card
    for i in range(len(cards) - 1):
        assert cards[i] < cards[i + 1], f"Failed: {cards[i]} should be < {cards[i + 1]}"