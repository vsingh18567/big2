"""Quick test script: Greedy strategy vs Smart strategy."""

from big2.simulator.cards import PASS, Combo
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator.smart_strategy import smart_strategy


def play_game(n_players: int = 4, greedy_player: int = 0) -> int:
    """
    Play one game where greedy_player uses greedy_strategy and others use smart_strategy.
    Returns:
        Winner player ID
    """
    try:
        env = Big2Env(n_players)
        state = env.reset()

        while not env.done:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]

            # Player 0 uses greedy, others use smart
            if p == greedy_player:
                action = greedy_strategy(candidates)
            else:
                action = smart_strategy(candidates, env.hands[p], env.trick_pile)

            state, _ = env.step(action)

        return env.winner
    except Exception as e:
        print(f"Error in play_game: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run 100 games and report results."""
    n_players = 4
    greedy_player = 0
    num_games = 100

    print(f"Playing {num_games} games: Player {greedy_player} (Greedy) vs Others (Smart)")
    print("-" * 60)

    wins_by_player: dict[int, int] = {}
    total_games = 0

    for game_num in range(1, num_games + 1):
        winner = play_game(n_players, greedy_player)
        wins_by_player[winner] = wins_by_player.get(winner, 0) + 1
        total_games += 1

        # Progress update every 25 games
        if game_num % 25 == 0:
            greedy_wins = wins_by_player.get(greedy_player, 0)
            print(f"After {game_num} games: Greedy wins = {greedy_wins} ({greedy_wins / game_num:.1%})")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    greedy_wins = wins_by_player.get(greedy_player, 0)
    greedy_win_rate = greedy_wins / total_games

    print(f"\nGreedy (Player {greedy_player}):")
    print(f"  Wins: {greedy_wins}/{total_games} ({greedy_win_rate:.2%})")

    print(f"\nSmart Strategy (Players {[i for i in range(n_players) if i != greedy_player]}):")
    smart_total_wins = sum(wins_by_player.get(i, 0) for i in range(n_players) if i != greedy_player)
    smart_win_rate = smart_total_wins / (total_games * (n_players - 1))
    print(f"  Total wins: {smart_total_wins}/{total_games * (n_players - 1)}")
    print(f"  Average win rate per player: {smart_win_rate:.2%}")

    print("\nWins by player:")
    for player in sorted(wins_by_player.keys()):
        strategy_name = "Greedy" if player == greedy_player else "Smart"
        wins = wins_by_player[player]
        win_rate = wins / total_games
        print(f"  Player {player} ({strategy_name}): {wins} wins ({win_rate:.2%})")


if __name__ == "__main__":
    main()
