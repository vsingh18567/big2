#!/usr/bin/env python3
"""
Interactive Big 2 game where you play against trained AI models.
"""
import torch
import numpy as np
from typing import Optional

from nn import MLPPolicy, combo_to_action_vector
from simulator.cards import Combo, PASS, card_name, SINGLE, PAIR, TRIPLE
from simulator.env import Big2Env


def format_combo(combo: Combo) -> str:
    """Format a combo for display."""
    if combo.type == PASS:
        return "PASS"
    type_names = {
        SINGLE: "Single",
        PAIR: "Pair",
        TRIPLE: "Triple",
        4: "Straight",
        5: "Flush",
        6: "Full House",
        7: "Four of a Kind",
        8: "Straight Flush"
    }
    type_name = type_names.get(combo.type, f"Type {combo.type}")
    cards_str = ' '.join([card_name(c) for c in combo.cards])
    return f"{type_name}: {cards_str}"


def ai_select_action(policy: MLPPolicy, state: np.ndarray, candidates: list[Combo]) -> Combo:
    """Select an action for an AI player using the trained policy."""
    with torch.no_grad():
        st = torch.from_numpy(state[np.newaxis, :]).long().to(policy.device)
        action_feats = [[combo_to_action_vector(c) for c in candidates]]
        logits_list, values = policy(st, action_feats)
        logits = logits_list[0]
        # Use greedy action for AI (could also sample)
        idx = torch.argmax(logits).item()
        return candidates[idx]


def display_hand(hand: list[int]) -> None:
    """Display a player's hand."""
    sorted_hand = sorted(hand)
    cards = [card_name(c) for c in sorted_hand]
    print(f"  {' '.join(cards)}")


def display_trick(trick: Optional[Combo], player_who_played: int) -> None:
    """Display the current trick."""
    if trick is None or trick.type == PASS:
        print("\n🃏 Current trick: (empty - you can play anything)")
    else:
        print(f"\n🃏 Current trick (played by Player {player_who_played}):")
        print(f"  {format_combo(trick)}")


def get_human_choice(candidates: list[Combo], hand: list[int]) -> Combo:
    """Get the human player's choice from legal candidates."""
    print("\n📋 Legal moves:")
    for i, combo in enumerate(candidates):
        print(f"  {i+1}. {format_combo(combo)}")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-{}): ".format(len(candidates)))
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(candidates)}.")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Please enter a number.")
        except EOFError:
            print("\nExiting game.")
            exit(0)


def play_interactive_game(model_path: str = "big2_model.pt", device: str = 'cpu'):
    """Play an interactive game of Big 2 against trained AI."""
    print("=" * 80)
    print("🎴 Big 2 - Interactive Game")
    print("=" * 80)
    print("\nLoading trained model...")
    
    # Load the trained model
    policy = MLPPolicy(device=device).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy.eval()
    print(f"✓ Model loaded from {model_path}")
    
    # Create the environment
    env = Big2Env(4)
    state = env.reset()
    
    # Determine which player is human (the one who starts with 3♦)
    human_player = env.current_player
    print(f"\nYou are Player {human_player}")
    print(f"The player with 3♦ starts the game.")
    
    # Track who played the last non-pass trick
    last_trick_player = None
    
    print("\n" + "=" * 80)
    print("🎮 Game Start!")
    print("=" * 80)
    
    # Game loop
    turn_number = 0
    while not env.done:
        current_player = env.current_player
        turn_number += 1
        
        if current_player == human_player:
            # Human player's turn
            print(f"\n{'=' * 80}")
            print(f"Turn {turn_number} - YOUR TURN")
            print(f"{'=' * 80}")
            
            # Show current trick
            display_trick(env.trick_pile, last_trick_player if last_trick_player is not None else current_player)
            
            # Show hand
            print(f"\n🎴 Your hand ({len(env.hands[current_player])} cards):")
            display_hand(env.hands[current_player])
            
            # Show opponent card counts
            print("\n👥 Opponents:")
            for i in range(1, 4):
                opp_idx = (human_player + i) % 4
                print(f"  Player {opp_idx}: {len(env.hands[opp_idx])} cards")
            
            # Get legal moves
            candidates = env.legal_candidates(current_player)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            
            # Get human choice
            action = get_human_choice(candidates, env.hands[current_player])
            
            if action.type != PASS:
                last_trick_player = current_player
            
        else:
            # AI player's turn
            candidates = env.legal_candidates(current_player)
            if not candidates:
                candidates = [Combo(PASS, [], ())]
            
            action = ai_select_action(policy, state, candidates)
            
            print(f"\n🤖 Player {current_player} plays: {format_combo(action)}")
            
            if action.type != PASS:
                last_trick_player = current_player
        
        # Execute the action
        state, reward, done, _ = env.step(action)
        
        # Check if trick was cleared
        if env.trick_pile is None or env.trick_pile.type == PASS:
            if turn_number > 1:  # Don't show this on first turn
                print(f"\n🔄 Trick cleared! Player {env.current_player} starts a new trick.")
    
    # Game over
    print("\n" + "=" * 80)
    print("🏁 GAME OVER!")
    print("=" * 80)
    
    if env.winner == human_player:
        print("\n🎉 Congratulations! You won! 🎉")
    else:
        print(f"\n😔 Player {env.winner} won. Better luck next time!")
    
    print("\nFinal standings:")
    for i in range(4):
        cards_left = len(env.hands[i])
        status = "🏆 WINNER" if i == env.winner else f"{cards_left} cards left"
        player_label = "You" if i == human_player else f"Player {i}"
        print(f"  {player_label}: {status}")


def main():
    """Main entry point."""
    import sys
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = sys.argv[1] if len(sys.argv) > 1 else "big2_model.pt"
    
    try:
        play_interactive_game(model_path, device)
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Thanks for playing!")
    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{model_path}' not found.")
        print("Please train a model first by running: python train.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

