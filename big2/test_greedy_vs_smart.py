"""Test script to simulate win rates between combinations of rule-based opponents.

Interesting benchmarks to run:
- 4 of the same strategies should have 25% win rate each.
- Greedy vs 3 Random
- Smart vs 3 Random
- Smart vs 3 Greedy
- 2 Smart vs 2 Greedy
"""

import argparse
import random
from typing import Protocol

from big2.simulator.cards import PASS, Combo
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.smart_strategy import smart_strategy


class StrategyFn(Protocol):
    def __call__(
        self,
        candidates: list[Combo],
        *,
        hand: list[int] | None = None,
        trick_pile: Combo | None = None,
    ) -> Combo: ...


def random_strategy(candidates: list[Combo]) -> Combo:
    return random.choice(candidates)


def _resolve_strategy(strategy_name: str) -> StrategyFn:
    name = strategy_name.strip().lower()
    if name == "greedy":
        return lambda candidates, **_: greedy_strategy(candidates)
    if name == "smart":
        return lambda candidates, *, hand=None, trick_pile=None: smart_strategy(
            candidates,
            hand or [],
            trick_pile,
        )
    if name == "random":
        return lambda candidates, **_: random_strategy(candidates)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def play_game(strategies: list[str]) -> int:
    """
    Play one game using the provided strategies.
    Returns:
        Winner player ID
    """
    try:
        env = Big2Env(len(strategies))
        state = env.reset()
        strategy_fns = [_resolve_strategy(name) for name in strategies]

        while not env.done:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]

            action = strategy_fns[p](
                candidates,
                hand=env.hands[p],
                trick_pile=env.trick_pile,
            )

            state, _ = env.step(action)

        return env.winner
    except Exception as e:
        print(f"Error in play_game: {e}")
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate Big2 wins between rule-based strategies.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["smart", "random", "random", "random"],
        help="Space-separated strategy list, e.g. greedy smart random random",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100000,
        help="Number of games to simulate.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N games (0 disables).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    strategies = args.strategies
    num_games = args.num_games
    progress_interval = args.progress_interval

    print(f"Playing {num_games} games with strategies: {strategies}")
    print("-" * 60)

    wins_by_player: dict[int, int] = {}
    wins_by_strategy: dict[str, int] = {}
    total_games = 0

    for game_num in range(1, num_games + 1):
        winner = play_game(strategies)
        wins_by_player[winner] = wins_by_player.get(winner, 0) + 1
        winner_strategy = strategies[winner]
        wins_by_strategy[winner_strategy] = wins_by_strategy.get(winner_strategy, 0) + 1
        total_games += 1

        # Progress update at configured interval
        if progress_interval > 0 and game_num % progress_interval == 0:
            summary = ", ".join(f"{name}={wins_by_strategy.get(name, 0)}" for name in sorted(set(strategies)))
            print(f"After {game_num} games: {summary}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print("\nWins by strategy:")
    for name in sorted(set(strategies)):
        wins = wins_by_strategy.get(name, 0)
        win_rate = wins / total_games
        print(f"  {name}: {wins}/{total_games} ({win_rate:.2%})")

    print("\nWins by player:")
    for player in sorted(wins_by_player.keys()):
        strategy_name = strategies[player]
        wins = wins_by_player[player]
        win_rate = wins / total_games
        print(f"  Player {player} ({strategy_name}): {wins} wins ({win_rate:.2%})")


if __name__ == "__main__":
    main()
