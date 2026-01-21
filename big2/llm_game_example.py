#!/usr/bin/env python3
"""
Example usage of the LLM game system.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from big2.llm_game import GameConfig, GameRunner, LLMConfig, NNConfig, PlayerConfig


def example_2_llm_vs_2_nn():
    """Example: 2 LLM players vs 2 neural network players."""
    config = GameConfig(
        players=[
            PlayerConfig(type="llm", llm_config=LLMConfig(model="claude-sonnet-4-5-20250929", temperature=0.8)),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="models/big2_model_step_100.pt", device="cpu")),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="claude-sonnet-4-5-20250929", temperature=0.8)),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="models/big2_model_step_100.pt", device="cpu")),
        ]
    )

    runner = GameRunner("example-game-1", config)
    state = runner.start_game()
    print(f"Game started. Done: {state['done']}, Winner: {state['winner']}, 'Players': {state['players']}")
    return runner


def example_human_vs_3_llm():
    """Example: 1 human player vs 3 LLM players."""
    config = GameConfig(
        players=[
            PlayerConfig(type="human"),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="gpt-4o-mini")),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="gpt-4o-mini")),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="gpt-4o-mini")),
        ]
    )

    runner = GameRunner("example-game-2", config)
    state = runner.start_game()

    print(f"Human hand: {state['human_hand']['display']}")
    print(f"Is human turn: {state['is_human_turn']}")

    if state["is_human_turn"] and state["legal_moves"]:
        print("\nLegal moves:")
        for i, move in enumerate(state["legal_moves"]):
            print(f"{i}: {move['type_name']} - {move['display']}")

    return runner


def example_all_nn():
    """Example: 4 neural network players (for benchmarking)."""
    config = GameConfig(
        players=[
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="big2_model.pt")),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="big2_model.pt")),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="big2_model.pt")),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="big2_model.pt")),
        ]
    )

    runner = GameRunner("example-game-3", config)
    state = runner.start_game()
    print(f"Game finished. Winner: Player {state['winner']}")
    print(f"Game history length: {len(runner.get_history())}")
    return runner


def example_mixed_llm_providers():
    """Example: Different LLM providers playing together."""
    config = GameConfig(
        players=[
            PlayerConfig(type="llm", llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.7)),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="claude-3-haiku-20240307", temperature=0.7)),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="gemini/gemini-1.5-flash", temperature=0.7)),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="big2_model.pt")),
        ]
    )

    runner = GameRunner("example-game-4", config)
    state = runner.start_game()
    print(f"Mixed LLM game complete. Winner: Player {state['winner']}")
    return runner


def run_single_game(game_id: int) -> dict:
    """Run a single game and return the results."""
    config = GameConfig(
        players=[
            PlayerConfig(type="llm", llm_config=LLMConfig(model="claude-sonnet-4-5-20250929", temperature=0.8)),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="models/big2_model_step_100.pt", device="cpu")),
            PlayerConfig(type="llm", llm_config=LLMConfig(model="claude-sonnet-4-5-20250929", temperature=0.8)),
            PlayerConfig(type="nn", nn_config=NNConfig(model_path="models/big2_model_step_100.pt", device="cpu")),
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
    run_multiple_games(num_games=20, concurrency=3)

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
