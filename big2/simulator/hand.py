class Hand:
    def __init__(self, cards: list[int]):
        self.cards = cards

    def __str__(self):
        return "\n".join(str(card) for card in self.cards)

    def __repr__(self):
        return f"Hand({self.cards})"

    def __len__(self):
        return len(self.cards)
