import random

import numpy as np

from big2.simulator.cards import (
    CARDS_PER_DECK,
    PAIR,
    PASS,
    SINGLE,
    TRIPLE,
    Combo,
    compare_combos,
    hand_to_combo,
)
from big2.simulator.generate_hands import COMBO_GENERATORS


class Big2Env:
    def __init__(self, n_players: int, cards_per_player: int | None = None):
        self.n_players = n_players
        max_cards_per_player = CARDS_PER_DECK // self.n_players
        self.cards_per_player = cards_per_player if cards_per_player is not None else max_cards_per_player
        if self.cards_per_player <= 0:
            raise ValueError("cards_per_player must be positive")
        if self.cards_per_player > max_cards_per_player:
            raise ValueError(
                f"cards_per_player={self.cards_per_player} exceeds max {max_cards_per_player} for n_players={n_players}"
            )
        self.total_cards_in_play = self.n_players * self.cards_per_player
        self.card_universe_size = self.total_cards_in_play
        self.reset()

    def reset(self):
        """Resets the environment, deals cards, and starts the game"""
        # Only use the first ordinal cards in the deck, then shuffle/deal those cards.
        deck = list(range(self.total_cards_in_play))
        random.shuffle(deck)
        self.hands = [deck[i * self.cards_per_player : (i + 1) * self.cards_per_player] for i in range(self.n_players)]

        # Sort each hand to keep consistent
        self.hands = [sorted(hand) for hand in self.hands]

        # Find the player with the 3 of diamonds, set as starting player
        self.current_player = 0
        for i, hand in enumerate(self.hands):
            if 0 in hand:
                self.current_player = i
                break

        self.seen: list[int] = [0] * self.card_universe_size
        self.trick_pile: Combo | None = None

        self.passes_in_row: int = 0
        self.done: bool = False
        self.winner: int | None = None
        self.cards_played_by_player: list[list[int]] = [[] for _ in range(self.n_players)]

        return self._obs(self.current_player)

    def _obs(self, player: int) -> np.ndarray:
        """Converts the internal game state into a numpy array"""
        hand = self.hands[player]
        hand_ids = hand + [-1] * (self.cards_per_player - len(hand))
        last_play = [0] * self.card_universe_size
        if self.trick_pile is not None:
            for c in self.trick_pile.cards:
                last_play[c] = 1

        seen: list[int] = self.seen.copy()

        # Opponent counts clockwise from current player
        counts = []
        for i in range(1, self.n_players):
            counts.append(len(self.hands[(player + i) % self.n_players]))

        # Per-opponent card vectors (which cards each opponent has played)
        opponent_cards = []
        for i in range(1, self.n_players):
            opp_id = (player + i) % self.n_players
            opp_cards_vec = [0] * self.card_universe_size
            for c in self.cards_played_by_player[opp_id]:
                opp_cards_vec[c] = 1
            opponent_cards.extend(opp_cards_vec)

        state = np.array(hand_ids + last_play + seen + counts + [self.passes_in_row] + opponent_cards, dtype=np.int32)

        return state

    def legal_candidates(self, player: int) -> list[Combo]:
        """Generates all legal moves for a player"""
        hand = self.hands[player]
        candidates: list[Combo] = []

        # Generate all possible combos from the current hand
        all_sets = []
        for _, gen in COMBO_GENERATORS.items():
            for cards in gen(hand):
                cmb = hand_to_combo(cards)
                if cmb is not None:
                    # If the player currently has the 3 of diamonds, they must include it to start
                    if 0 in hand and 0 not in cmb.cards:
                        continue
                    all_sets.append(cmb)

        # Filter by what matches the current trick
        if self.trick_pile is None or self.trick_pile.type == PASS:
            # Fresh trick: any non-PASS combo allowed
            candidates = all_sets
        else:
            # Must match size and type precedence rules (for five-card: same size and category must be >= in category)
            last = self.trick_pile
            # For 1/2/3: must be same size and compare by cmp
            if last.type in (SINGLE, PAIR, TRIPLE):
                for cmb in all_sets:
                    if cmb.type == last.type and compare_combos(cmb, last) > 0:
                        candidates.append(cmb)
            else:
                # five-card: must be same size (always 5). Category must be same or higher category.
                for cmb in all_sets:
                    if cmb.size() != 5:
                        continue
                    if cmb.type == last.type:
                        if compare_combos(cmb, last) > 0:
                            candidates.append(cmb)
                    elif cmb.type > last.type:
                        candidates.append(cmb)

        # PASS allowed if we are not opening a fresh trick
        if self.trick_pile is not None and self.trick_pile.type != PASS:
            candidates.append(Combo(PASS, [], ()))

        return candidates

    def step(self, action: Combo) -> tuple[np.ndarray, bool]:
        """Updates the game state according to a trick played by the current player"""
        player_hand = self.hands[self.current_player]
        if action.type == PASS:
            self.passes_in_row += 1
            # If this player was the last to pass, the next player has control
            if self.passes_in_row == self.n_players - 1:
                self.trick_pile = None
                self.passes_in_row = 0
        else:
            for c in action.cards:
                player_hand.remove(c)
                self.seen[c] = 1
            self.cards_played_by_player[self.current_player].extend(action.cards)
            self.trick_pile = action
            self.passes_in_row = 0
            if len(player_hand) == 0:
                self.done = True
                self.winner = self.current_player

        # If not done, advance the current player
        if not self.done:
            self.current_player = (self.current_player + 1) % self.n_players

        return self._obs(self.current_player), self.done
