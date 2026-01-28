#!/usr/bin/env python3
"""
Example usage of the LLM game system.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from big2.llm_game import GameConfig, GameRunner, LLMConfig, NNConfig, PlayerConfig


def run_single_game(game_id: int) -> dict:
    """Run a single game and return the results."""
    config = GameConfig(
        players=[
            PlayerConfig(type="llm", llm_config=LLMConfig(model="claude-sonnet-4-5-20250929", temperature=0.8)),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="models/big2_model_step_300.pt", device="cpu")),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="gpt-5.2", temperature=0.8)),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="models/big2_model_step_300.pt", device="cpu")),
        ]
    )

    runner = GameRunner(f"multithreaded-game-{game_id}", config)
    state = runner.start_game()

    return {
        "game_id": game_id,
        "winner": state["winner"],
        "done": state["done"],
        "history_length": len(runner.get_history()),
    }


def run_multiple_games(num_games: int = 20, concurrency: int = 3):
    """Run multiple games concurrently using ThreadPoolExecutor."""
    print(f"\nRunning {num_games} games with concurrency={concurrency}")
    print("=" * 60)

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all games
        future_to_game = {executor.submit(run_single_game, i): i for i in range(num_games)}

        # Process completed games as they finish
        for future in as_completed(future_to_game):
            game_id = future_to_game[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                winner = result["winner"]
                moves = result["history_length"]
                print(
                    f"[{completed}/{num_games}] Game {result['game_id']} finished: "
                    f"Winner=Player {winner}, Moves={moves}"
                )
            except Exception as exc:
                print(f"Game {game_id} generated an exception: {exc}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)

    winner_counts = {}
    for result in results:
        winner = result["winner"]
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    print(f"Total games completed: {len(results)}")
    print("\nWinner distribution:")
    for player_id in sorted(winner_counts.keys()):
        count = winner_counts[player_id]
        percentage = (count / len(results)) * 100
        player_type = "LLM" if player_id in [0, 2] else "NN"
        print(f"  Player {player_id} ({player_type}): {count} wins ({percentage:.1f}%)")

    avg_moves = sum(r["history_length"] for r in results) / len(results)
    print(f"\nAverage game length: {avg_moves:.1f} moves")

    return results


if __name__ == "__main__":
    # Run multiple games with multithreading
    run_multiple_games(num_games=100, concurrency=5)

    # Single game examples (commented out)
    # print("=" * 60)
    # print("Example 1: 2 LLM vs 2 NN")
    # print("=" * 60)
    # example_2_llm_vs_2_nn()

    # print("\n" + "=" * 60)
    # print("Example 2: Human vs 3 LLM")
    # print("=" * 60)
    # example_human_vs_3_llm()

    # print("\n" + "=" * 60)
    # print("Example 3: 4 NN players")
    # print("=" * 60)
    # example_all_nn()

    # print("\n" + "=" * 60)
    # print("Example 4: Mixed LLM providers")
    # print("=" * 60)
    # example_mixed_llm_providers()
