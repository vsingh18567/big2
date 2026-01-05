from big2.simulator.cards import PASS, Combo


def greedy_strategy(combos: list[Combo]) -> Combo:
    if len(combos) <= 1:
        return combos[0]
    non_pass_combos = [c for c in combos if c.type != PASS]
    return min(non_pass_combos)
