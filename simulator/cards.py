from dataclasses import dataclass

CARDS_PER_DECK = 52

RANKS = list(range(13))  # Ordinal ranking from 0-12 where 2 = 12 (highest)
SUITS = list(range(4))  # 0: ♦, 1: ♣, 2: ♥, 3: ♠

RANK_NAMES = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]
SUIT_NAMES = ["♦", "♣", "♥", "♠"]

# Combo types ranked ordinally
PASS, SINGLE, PAIR, TRIPLE, STRAIGHT, FLUSH, FULLHOUSE, FOUR_KIND, STRAIGHT_FLUSH = range(9)


@dataclass
class Combo:
    type: int
    cards: list[int]  # cards used in this combo
    key: tuple  # used for comparing within same type

    def size(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return f"{self.type} {self.cards} {self.key}"


# Card utils


def card_rank(card_id: int) -> int:
    return card_id // 4  # 0..12


def card_suit(card_id: int) -> int:
    return card_id % 4  # 0..3


def card_name(card_id: int) -> str:
    return f"{RANK_NAMES[card_rank(card_id)]}{SUIT_NAMES[card_suit(card_id)]}"


# Helpers


def sort_cards(cards: list[int]) -> list[int]:
    return sorted(cards)


def is_consecutive(ranks_sorted: list[int]) -> bool:
    # Handle straights in Big 2: 2 (rank=12) cannot be in a straight.
    if 12 in ranks_sorted:
        return False
    return all(ranks_sorted[i] + 1 == ranks_sorted[i + 1] for i in range(len(ranks_sorted) - 1))


def cmp_single_hand(hand1: list[int], hand2: list[int]) -> int:
    # Compare two single cards by rank and suit
    card1, card2 = hand1[0], hand2[0]
    card1_rank, card2_rank = card_rank(card1), card_rank(card2)
    card1_suit, card2_suit = card_suit(card1), card_suit(card2)
    if card1_rank != card2_rank:
        return card1_rank - card2_rank
    return card1_suit - card2_suit


def hand_to_combo(cards: list[int]) -> Combo | None:
    # Convert a list of cards to the Combo it represents
    cards = sort_cards(cards)
    n = len(cards)
    if n == 0:
        return Combo(PASS, [], ())

    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]

    if n == 1:
        # Single key: (rank, suit)
        return Combo(SINGLE, cards, (ranks[0], suits[0]))

    if n == 2:
        # Only 2-card combo is a pair
        if ranks[0] == ranks[1]:
            # Pair key: (rank, max_suit)
            return Combo(PAIR, cards, (ranks[0], max(suits)))
        return None

    if n == 3:
        # Only 3-card combo is a triple
        if ranks[0] == ranks[1] == ranks[2]:
            # Triple key: (rank, max_suit)
            return Combo(TRIPLE, cards, (ranks[0],))
        return None

    if n == 5:
        # Only 5-card combo can be a straight flush, four-kind, full house, or flush
        counts: dict[int, int] = {}
        for r in ranks:
            counts[r] = counts.get(r, 0) + 1

        unique = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))  # sort by freq desc, rank desc
        is_flush = len(set(suits)) == 1
        is_straight = is_consecutive(sorted(set(ranks))) and len(set(ranks)) == 5

        if is_straight and is_flush:
            # Key: (highest rank of straight, suit of that highest card)
            high_rank = max(ranks)
            high_suit = max([card_suit(c) for c in cards if card_rank(c) == high_rank])
            return Combo(STRAIGHT_FLUSH, cards, (high_suit, high_rank))
        if unique[0][1] == 4:
            # Four-kind + kicker. Key: (rank of the quad)
            quad_rank = unique[0][0]
            return Combo(FOUR_KIND, cards, (quad_rank,))
        if unique[0][1] == 3 and unique[1][1] == 2:
            # Full house. Key: (rank of trip, rank of pair)
            trip_rank = unique[0][0]
            pair_rank = unique[1][0]
            return Combo(FULLHOUSE, cards, (trip_rank, pair_rank))
        if is_flush:
            # Flush key: (suit of the highest card, ranks sorted desc)
            key = (max(suits),) + tuple(sorted(ranks, reverse=True))
            return Combo(FLUSH, cards, key)
        if is_straight:
            # Key: (highest rank of straight, suit of that highest card)
            high_rank = max(ranks)
            high_suit = max([card_suit(c) for c in cards if card_rank(c) == high_rank])
            return Combo(STRAIGHT, cards, (high_rank, high_suit))

    return None


def compare_combos(a: Combo | None, b: Combo | None) -> int:
    if a is None or b is None:
        return 0
    # Only meaningful when a.type == b.type and len equal, per Big 2 trick rules.
    if a.size() != b.size():
        return 0  # undefined; caller ensures legality

    if a.type != b.type:
        return 1 if a.type > b.type else -1

    # Compare keys lexicographically
    if a.key == b.key:
        return 0
    return 1 if a.key > b.key else -1
