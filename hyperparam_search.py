"""Hyperparameter search for smart strategy."""

import copy
from big2.simulator.cards import PASS, Combo
from big2.simulator.env import Big2Env
from big2.simulator.greedy_strategy import greedy_strategy
from big2.simulator import smart_strategy

# Store original functions
original_score_combo = smart_strategy.score_combo


def create_score_combo_with_params(params):
    """Create a score_combo function with specific hyperparameters."""
    def score_combo(combo, hand, trick_pile):
        from big2.simulator.cards import SINGLE, PAIR, TRIPLE, STRAIGHT, FOUR_KIND, PASS
        from big2.simulator.smart_strategy import (
            card_rank, game_phase, count_power_cards, 
            would_break_combo, is_very_strong_play
        )
        
        score = 0.0
        hand_size = len(hand)
        phase = game_phase(hand_size)

        # Base: card strength
        base_strength = sum(card_rank(c) for c in combo.cards)
        score += base_strength * params['base_weight']

        # Power card penalties
        power_card_count = count_power_cards(combo.cards)
        if phase == "early":
            score += power_card_count * params['power_early_penalty']
        elif phase == "mid":
            score += power_card_count * params['power_mid_penalty']

        # Breaking combos penalty
        if phase == "early" and would_break_combo(combo, hand):
            score += params['break_combo_penalty']

        # Bonus for playing more cards
        score -= combo.size() * params['multi_card_bonus']

        # Late game aggression
        if phase == "late":
            score -= params['late_game_bonus']

        # Combo type preferences
        if combo.type == PAIR:
            score -= params['pair_bonus']
        elif combo.type == TRIPLE:
            score -= params['triple_bonus']
        elif combo.type >= STRAIGHT:
            score -= params['five_card_bonus']

        # Response to opponent plays
        if trick_pile and trick_pile.type != PASS:
            if is_very_strong_play(trick_pile) and phase == "late":
                score += params['strong_play_penalty']
            elif trick_pile.type >= FOUR_KIND:
                score += params['unbeatable_penalty']

        return score
    
    return score_combo


def play_game(n_players: int, greedy_player: int, params) -> int:
    """Play one game with custom hyperparameters."""
    # Monkey patch the score_combo function
    smart_strategy.score_combo = create_score_combo_with_params(params)
    
    try:
        env = Big2Env(n_players)
        state = env.reset()
        
        while not env.done:
            p = env.current_player
            candidates = env.legal_candidates(p)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            
            if p == greedy_player:
                action = greedy_strategy(candidates)
            else:
                action = smart_strategy.smart_strategy(candidates, env.hands[p], env.trick_pile)
            
            state, _ = env.step(action)
        
        return env.winner
    finally:
        # Restore original function
        smart_strategy.score_combo = original_score_combo


def evaluate_params(params, n_games=100):
    """Evaluate a set of hyperparameters."""
    n_players = 4
    greedy_player = 0
    wins_by_player = {}
    
    for _ in range(n_games):
        winner = play_game(n_players, greedy_player, params)
        wins_by_player[winner] = wins_by_player.get(winner, 0) + 1
    
    greedy_wins = wins_by_player.get(greedy_player, 0)
    smart_total_wins = sum(wins_by_player.get(i, 0) for i in range(n_players) if i != greedy_player)
    smart_avg_win_rate = smart_total_wins / (n_games * (n_players - 1))
    
    return {
        'greedy_win_rate': greedy_wins / n_games,
        'smart_avg_win_rate': smart_avg_win_rate,
        'smart_total_wins': smart_total_wins,
        'wins_by_player': wins_by_player
    }


# Define 15 hyperparameter configurations to test
HYPERPARAMETER_CONFIGS = [
    # Baseline (current settings)
    {
        'name': '1. Baseline',
        'base_weight': 1.0,
        'power_early_penalty': 15,
        'power_mid_penalty': 8,
        'break_combo_penalty': 8,
        'multi_card_bonus': 3,
        'late_game_bonus': 15,
        'pair_bonus': 5,
        'triple_bonus': 8,
        'five_card_bonus': 10,
        'strong_play_penalty': 20,
        'unbeatable_penalty': 15,
    },
    # More aggressive (play more cards)
    {
        'name': '2. Aggressive',
        'base_weight': 0.8,
        'power_early_penalty': 12,
        'power_mid_penalty': 6,
        'break_combo_penalty': 5,
        'multi_card_bonus': 5,
        'late_game_bonus': 20,
        'pair_bonus': 8,
        'triple_bonus': 12,
        'five_card_bonus': 15,
        'strong_play_penalty': 15,
        'unbeatable_penalty': 20,
    },
    # Conservative (save good cards)
    {
        'name': '3. Conservative',
        'base_weight': 1.2,
        'power_early_penalty': 20,
        'power_mid_penalty': 12,
        'break_combo_penalty': 12,
        'multi_card_bonus': 2,
        'late_game_bonus': 12,
        'pair_bonus': 3,
        'triple_bonus': 5,
        'five_card_bonus': 7,
        'strong_play_penalty': 25,
        'unbeatable_penalty': 15,
    },
    # Combo-focused
    {
        'name': '4. Combo Focus',
        'base_weight': 0.9,
        'power_early_penalty': 15,
        'power_mid_penalty': 8,
        'break_combo_penalty': 15,
        'multi_card_bonus': 4,
        'late_game_bonus': 15,
        'pair_bonus': 10,
        'triple_bonus': 15,
        'five_card_bonus': 18,
        'strong_play_penalty': 20,
        'unbeatable_penalty': 15,
    },
    # Late game specialist
    {
        'name': '5. Late Game Pro',
        'base_weight': 1.0,
        'power_early_penalty': 18,
        'power_mid_penalty': 10,
        'break_combo_penalty': 10,
        'multi_card_bonus': 3,
        'late_game_bonus': 25,
        'pair_bonus': 6,
        'triple_bonus': 10,
        'five_card_bonus': 12,
        'strong_play_penalty': 15,
        'unbeatable_penalty': 20,
    },
    # Balanced
    {
        'name': '6. Balanced',
        'base_weight': 1.0,
        'power_early_penalty': 12,
        'power_mid_penalty': 7,
        'break_combo_penalty': 6,
        'multi_card_bonus': 4,
        'late_game_bonus': 18,
        'pair_bonus': 7,
        'triple_bonus': 10,
        'five_card_bonus': 13,
        'strong_play_penalty': 18,
        'unbeatable_penalty': 18,
    },
    # Low base weight (less greedy-like)
    {
        'name': '7. Anti-Greedy',
        'base_weight': 0.5,
        'power_early_penalty': 10,
        'power_mid_penalty': 5,
        'break_combo_penalty': 8,
        'multi_card_bonus': 6,
        'late_game_bonus': 20,
        'pair_bonus': 10,
        'triple_bonus': 15,
        'five_card_bonus': 20,
        'strong_play_penalty': 15,
        'unbeatable_penalty': 25,
    },
    # High combo bonus
    {
        'name': '8. Combo King',
        'base_weight': 0.7,
        'power_early_penalty': 15,
        'power_mid_penalty': 8,
        'break_combo_penalty': 20,
        'multi_card_bonus': 7,
        'late_game_bonus': 15,
        'pair_bonus': 12,
        'triple_bonus': 18,
        'five_card_bonus': 25,
        'strong_play_penalty': 20,
        'unbeatable_penalty': 15,
    },
    # Power card saver
    {
        'name': '9. Power Saver',
        'base_weight': 1.0,
        'power_early_penalty': 25,
        'power_mid_penalty': 15,
        'break_combo_penalty': 8,
        'multi_card_bonus': 3,
        'late_game_bonus': 15,
        'pair_bonus': 5,
        'triple_bonus': 8,
        'five_card_bonus': 10,
        'strong_play_penalty': 18,
        'unbeatable_penalty': 12,
    },
    # Minimal penalties
    {
        'name': '10. Free Spirit',
        'base_weight': 1.0,
        'power_early_penalty': 5,
        'power_mid_penalty': 3,
        'break_combo_penalty': 3,
        'multi_card_bonus': 4,
        'late_game_bonus': 18,
        'pair_bonus': 8,
        'triple_bonus': 12,
        'five_card_bonus': 15,
        'strong_play_penalty': 10,
        'unbeatable_penalty': 15,
    },
    # Strategic passer
    {
        'name': '11. Strategic Pass',
        'base_weight': 1.0,
        'power_early_penalty': 15,
        'power_mid_penalty': 8,
        'break_combo_penalty': 8,
        'multi_card_bonus': 3,
        'late_game_bonus': 15,
        'pair_bonus': 5,
        'triple_bonus': 8,
        'five_card_bonus': 10,
        'strong_play_penalty': 30,
        'unbeatable_penalty': 25,
    },
    # Card efficiency
    {
        'name': '12. Efficient',
        'base_weight': 0.8,
        'power_early_penalty': 12,
        'power_mid_penalty': 7,
        'break_combo_penalty': 10,
        'multi_card_bonus': 8,
        'late_game_bonus': 22,
        'pair_bonus': 10,
        'triple_bonus': 15,
        'five_card_bonus': 20,
        'strong_play_penalty': 18,
        'unbeatable_penalty': 18,
    },
    # Greedy hybrid
    {
        'name': '13. Greedy+',
        'base_weight': 1.5,
        'power_early_penalty': 10,
        'power_mid_penalty': 5,
        'break_combo_penalty': 5,
        'multi_card_bonus': 2,
        'late_game_bonus': 20,
        'pair_bonus': 4,
        'triple_bonus': 6,
        'five_card_bonus': 8,
        'strong_play_penalty': 15,
        'unbeatable_penalty': 20,
    },
    # Ultra aggressive
    {
        'name': '14. All-In',
        'base_weight': 0.6,
        'power_early_penalty': 8,
        'power_mid_penalty': 4,
        'break_combo_penalty': 3,
        'multi_card_bonus': 10,
        'late_game_bonus': 30,
        'pair_bonus': 15,
        'triple_bonus': 20,
        'five_card_bonus': 25,
        'strong_play_penalty': 12,
        'unbeatable_penalty': 20,
    },
    # Adaptive
    {
        'name': '15. Adaptive',
        'base_weight': 0.9,
        'power_early_penalty': 14,
        'power_mid_penalty': 8,
        'break_combo_penalty': 7,
        'multi_card_bonus': 5,
        'late_game_bonus': 20,
        'pair_bonus': 8,
        'triple_bonus': 12,
        'five_card_bonus': 16,
        'strong_play_penalty': 17,
        'unbeatable_penalty': 17,
    },
]


def main():
    """Run hyperparameter search."""
    print("=" * 80)
    print("HYPERPARAMETER SEARCH FOR SMART STRATEGY")
    print("=" * 80)
    print(f"Testing {len(HYPERPARAMETER_CONFIGS)} configurations with 100 games each")
    print("=" * 80)
    print()
    
    results = []
    
    for i, config in enumerate(HYPERPARAMETER_CONFIGS, 1):
        params = {k: v for k, v in config.items() if k != 'name'}
        name = config['name']
        
        print(f"[{i}/{len(HYPERPARAMETER_CONFIGS)}] Testing: {name}...", end=' ', flush=True)
        
        result = evaluate_params(params, n_games=100)
        result['name'] = name
        result['params'] = params
        results.append(result)
        
        print(f"Smart avg: {result['smart_avg_win_rate']:.3f} | Greedy: {result['greedy_win_rate']:.3f}")
    
    print()
    print("=" * 80)
    print("RESULTS SUMMARY (sorted by smart average win rate)")
    print("=" * 80)
    
    # Sort by smart average win rate (descending)
    results.sort(key=lambda x: x['smart_avg_win_rate'], reverse=True)
    
    print(f"\n{'Rank':<6}{'Config':<20}{'Smart Avg WR':<15}{'Greedy WR':<12}{'Margin':<10}")
    print("-" * 80)
    
    for rank, result in enumerate(results, 1):
        margin = result['smart_avg_win_rate'] - result['greedy_win_rate']
        print(f"{rank:<6}{result['name']:<20}{result['smart_avg_win_rate']:.3f} ({result['smart_total_wins']}/300)    "
              f"{result['greedy_win_rate']:.3f}      {margin:+.3f}")
    
    print()
    print("=" * 80)
    print("TOP 3 CONFIGURATIONS")
    print("=" * 80)
    
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. {result['name']}")
        print(f"   Smart avg win rate: {result['smart_avg_win_rate']:.1%}")
        print(f"   Greedy win rate: {result['greedy_win_rate']:.1%}")
        print(f"   Margin: {result['smart_avg_win_rate'] - result['greedy_win_rate']:+.1%}")
        print("   Parameters:")
        for key, value in result['params'].items():
            print(f"      {key}: {value}")


if __name__ == "__main__":
    main()

