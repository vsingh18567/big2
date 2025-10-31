"""Computes all possible playable tricks from a hand."""

import itertools

from .cards import (
    FLUSH,
    FOUR_KIND,
    FULLHOUSE,
    PAIR,
    SINGLE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    SUITS,
    TRIPLE,
    card_rank,
    card_suit,
    sort_cards,
)


def generate_singles(hand: list[int]) -> list[list[int]]:
    return [[c] for c in hand]


def generate_pairs(hand: list[int]) -> list[list[int]]:
    pairs = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    for cards in by_rank.values():
        if len(cards) >= 2:
            for comb in itertools.combinations(sorted(cards), 2):
                pairs.append(list(comb))
    return pairs


def generate_triples(hand: list[int]) -> list[list[int]]:
    triples = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    for _r, cards in by_rank.items():
        if len(cards) >= 3:
            for comb in itertools.combinations(sorted(cards), 3):
                triples.append(list(comb))
    return triples


def generate_straights(hand: list[int]) -> list[list[int]]:
    # Build by rank sets (exclude rank=12 i.e., 2)
    by_rank_suits: dict[int, list[int]] = {}
    for c in hand:
        r = card_rank(c)
        if r == 12:
            continue
        by_rank_suits.setdefault(r, []).append(c)
    straights = []

    # We need sequences of length 5 of consecutive ranks
    for start in range(0, 8):  # 0..7 inclusive allows start at rank 7 -> 7,8,9,10,11
        seq = [start + i for i in range(5)]
        if all(r in by_rank_suits for r in seq):
            # choose one card for each rank
            for choice in itertools.product(*[sorted(by_rank_suits[r]) for r in seq]):
                straights.append(list(choice))

    return straights


def generate_flushes(hand: list[int]) -> list[list[int]]:
    by_suit: dict[int, list[int]] = {s: [] for s in SUITS}
    for c in hand:
        by_suit[card_suit(c)].append(c)
    flushes = []

    for cards in by_suit.values():
        if len(cards) >= 5:
            for comb in itertools.combinations(sorted(cards), 5):
                flushes.append(list(comb))

    return flushes


def generate_fullhouses(hand: list[int]) -> list[list[int]]:
    fulls = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    trips = []
    pairs = []
    for _r, cards in by_rank.items():
        if len(cards) >= 3:
            trips.extend([list(comb) for comb in itertools.combinations(sorted(cards), 3)])
        if len(cards) >= 2:
            pairs.extend([list(comb) for comb in itertools.combinations(sorted(cards), 2)])
    for t in trips:
        tr = card_rank(t[0])
        for p in pairs:
            pr = card_rank(p[0])
            if pr == tr:
                continue
            fulls.append(sort_cards(t + p))
    # Remove duplicates
    uniq = {tuple(sorted(x)): x for x in fulls}
    return list(uniq.values())


def generate_fourkind(hand: list[int]) -> list[list[int]]:
    quads = []
    by_rank: dict[int, list[int]] = {}
    for c in hand:
        by_rank.setdefault(card_rank(c), []).append(c)
    kickers = set(hand)
    for r, cards in by_rank.items():
        if len(cards) == 4:
            for k in kickers:
                if card_rank(k) != r:
                    combo = sort_cards(cards + [k])
                    quads.append(combo)
    # Remove duplicates
    uniq = {tuple(sorted(x)): x for x in quads}
    return list(uniq.values())


def generate_straightflushes(hand: list[int]) -> list[list[int]]:
    by_suit: dict[int, list[int]] = {s: [] for s in SUITS}
    for c in hand:
        by_suit[card_suit(c)].append(c)
    sflushes = []
    for _s, cards in by_suit.items():
        # map suit-specific ranks (exclude 2s)
        by_rank: dict[int, list[int]] = {}
        for c in cards:
            r = card_rank(c)
            if r == 12:
                continue
            by_rank.setdefault(r, []).append(c)
        for start in range(0, 8):
            seq = [start + i for i in range(5)]
            if all(r in by_rank for r in seq):
                for choice in itertools.product(*[sorted(by_rank[r]) for r in seq]):
                    sflushes.append(list(choice))
    return sflushes


COMBO_GENERATORS = {
    SINGLE: generate_singles,
    PAIR: generate_pairs,
    TRIPLE: generate_triples,
    STRAIGHT: generate_straights,
    FLUSH: generate_flushes,
    FULLHOUSE: generate_fullhouses,
    FOUR_KIND: generate_fourkind,
    STRAIGHT_FLUSH: generate_straightflushes,
}
