import random
from dataclasses import dataclass

import numpy as np

from .cards import CARDS_PER_DECK, Combo, PASS, hand_to_combo, SINGLE, PAIR, TRIPLE, compare_combos
from .generate_hands import COMBO_GENERATORS
 
class Big2Env:
    def __init__(self, n_players: int):
        self.n_players = n_players
        self.reset()

    def reset(self):
        """ Resets the environment, deals cards, and starts the game """
        # Shuffle and deal the deck
        deck = list(range(CARDS_PER_DECK))
        random.shuffle(deck)
        cards_per_player = CARDS_PER_DECK // self.n_players
        self.hands = [deck[i*cards_per_player:(i+1)*cards_per_player] for i in range(self.n_players)]
        
        # Sort each hand to keep consistent
        self.hands = [sorted(hand) for hand in self.hands]
        
        # Find the player with the 3 of diamonds, set as starting player
        self.current_player = 0
        for i, hand in enumerate(self.hands):
            if 0 in hand:
                self.current_player = i
                break
        
        self.seen: list[int] = [0] * CARDS_PER_DECK
        self.trick_pile: Combo | None = None

        self.passes_in_row: int = 0
        self.done: bool = False
        self.winner: int | None = None

        return self._obs(self.current_player)

    def _obs(self, player: int) -> np.ndarray:
        """ Converts the internal game state into a numpy array """
        # 13 scalars for current hand, -1 padded; last-play 52 one-hot; seen 52; opponents counts 3
        hand = self.hands[player]
        hand_ids = hand + [-1]*(13-len(hand))
        last_play = [0]*52
        if self.trick_pile is not None:
            for c in self.trick_pile.cards:
                last_play[c] = 1
        
        seen: list[int] = self.seen.copy()
        
        # Opponent counts clockwise from current player
        counts = []
        for i in range(1,4):
            counts.append(len(self.hands[(player+i)%self.n_players]))
            state = np.array(hand_ids + last_play + seen + counts, dtype=np.int32)
        
        return state

    def legal_candidates(self, player: int) -> list[Combo]:
        """ Generates all legal moves for a player """
        hand = self.hands[player]
        candidates: list[Combo] = []
        
        # If the player currently has the 3 of diamonds, they must include it to start
        all_sets = []
        must_include_three_of_diamonds = False
        if 0 in hand:
            must_include_three_of_diamonds = True
        for t, gen in COMBO_GENERATORS.items():
            for cards in gen(hand):
                cmb = hand_to_combo(cards)
                if cmb is not None:
                    if must_include_three_of_diamonds and 0 not in cmb.cards:
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


    def step(self, action: Combo) -> tuple[np.ndarray, float, bool, dict]:
        """ Updates the game state according to a trick played by the current player """
        player_hand = self.hands[self.current_player]
        reward = 0
        if action.type == PASS:
            self.passes_in_row += 1
            # If this player was the 3rd pass in a row, the next player has control
            if self.passes_in_row >= self.n_players - 1:
                self.trick_pile = None
                self.passes_in_row = 0
        
        else:
            for c in action.cards:
                player_hand.remove(c)
                self.seen[c] = 1
            self.trick_pile = action
            self.passes_in_row = 0
            if len(player_hand) == 0:
                self.done = True
                self.winner = self.current_player
                reward = 1
        
        # If not done, advance the current player
        if not self.done:
            self.current_player = (self.current_player + 1) % self.n_players
        
        return self._obs(self.current_player), reward, self.done, {}
