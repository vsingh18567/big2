"""Card module for the Big 2 game simulator."""

from typing import Literal
from functools import total_ordering

Suit = Literal["D", "C", "H", "S"]
Rank = Literal["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]



@total_ordering
class Card:
    """Represents a playing card in the Big 2 game."""

    def __init__(self, suit: Suit, rank: Rank) -> None:
        self.suit = suit
        self.rank = rank

    def __str__(self) -> str:
        """Return string representation of the card."""
        return f"{self.suit}{self.rank}"

    def __repr__(self) -> str:
        """Return detailed string representation of the card."""
        return f"Card({self.suit}{self.rank})"
    
    def __eq__(self, other: object) -> bool:
        """Check if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    @staticmethod
    def _is_rank_smaller_than(rank: Rank, other_rank: Rank) -> bool:
        """Compare two ranks."""
        rank_order = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
        return rank_order.index(rank) < rank_order.index(other_rank)
    
    @staticmethod
    def _is_suit_smaller_than(suit: Suit, other_suit: Suit) -> bool:
        """Compare two suits."""
        suit_order = ["C", "D", "H", "S"]  # Clubs < Diamonds < Hearts < Spades
        return suit_order.index(suit) < suit_order.index(other_suit)

    def __lt__(self, other: object) -> bool:
        """Check if one card is less than another."""
        if not isinstance(other, Card):
            return False
        
        # First compare ranks
        if self.rank != other.rank:
            return self._is_rank_smaller_than(self.rank, other.rank)
        
        # If ranks are equal, compare suits
        return self._is_suit_smaller_than(self.suit, other.suit)
    
    def __hash__(self) -> int:
        """Return hash of the card."""
        return hash((self.suit, self.rank))
    
    
