use big2_rust::action::MoveKind;
use big2_rust::card::{
    CARDS_PER_PLAYER, PLAYERS, THREE_DIAMONDS, card_bit, card_name, format_card_mask,
    mask_from_cards, mask_to_card_names,
};
use big2_rust::{ActionCatalog, Big2Env, GameState, RulesConfig, StepError};

fn state_with_hands(hands: [u64; PLAYERS], current_player: u8) -> GameState {
    let mut cards_remaining = [0u8; PLAYERS];
    for player in 0..PLAYERS {
        cards_remaining[player] = hands[player].count_ones() as u8;
    }
    GameState {
        hands,
        current_player,
        last_move_id: None,
        last_actor: None,
        passed: [false; PLAYERS],
        cards_played: 0,
        cards_remaining,
        finished: [false; PLAYERS],
        is_first_turn: true,
        done: false,
        rng_state: 1,
    }
}

#[test]
fn card_and_mask_formatting_is_human_readable() {
    let hand = mask_from_cards(&[0, 1, 48, 51]);

    assert_eq!(card_name(0), "3D");
    assert_eq!(card_name(51), "2S");
    assert_eq!(mask_to_card_names(hand), vec!["3D", "3C", "2D", "2S"]);
    assert_eq!(format_card_mask(hand), "[3D 3C 2D 2S]");
    assert_eq!(format_card_mask(0), "[]");
}

#[test]
fn move_formatting_includes_kind_and_cards() {
    let catalog = ActionCatalog::new();
    let pass = catalog.get(0).unwrap();
    let pair = catalog
        .get(catalog.move_id_for_mask(mask_from_cards(&[0, 1])).unwrap())
        .unwrap();
    let straight_flush = catalog
        .get(
            catalog
                .move_id_for_mask(mask_from_cards(&[0, 4, 8, 12, 16]))
                .unwrap(),
        )
        .unwrap();

    assert_eq!(pass.display(), "Pass");
    assert_eq!(pair.display(), "Pair [3D 3C]");
    assert_eq!(straight_flush.display(), "Straight flush [3D 4D 5D 6D 7D]");
    assert_eq!(
        catalog.format_move(pair.id).as_deref(),
        Some("Pair [3D 3C]")
    );
}

#[test]
fn reset_deals_all_cards_and_starts_with_three_diamonds_holder() {
    let env = Big2Env::new(42);
    let mut combined = 0u64;

    for player in 0..PLAYERS {
        assert_eq!(env.state.cards_remaining[player], CARDS_PER_PLAYER as u8);
        assert_eq!(
            env.state.hands[player].count_ones(),
            CARDS_PER_PLAYER as u32
        );
        assert_eq!(combined & env.state.hands[player], 0);
        combined |= env.state.hands[player];
    }

    assert_eq!(combined.count_ones(), 52);
    assert!(env.state.hands[env.state.current_player as usize] & card_bit(THREE_DIAMONDS) != 0);
}

#[test]
fn first_turn_legal_moves_must_include_three_diamonds() {
    let catalog = ActionCatalog::new();
    let hand = mask_from_cards(&[0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 32]);
    let mut env = Big2Env {
        state: state_with_hands([hand, 0, 0, 0], 0),
        rules: RulesConfig::default(),
    };

    let legal = env.legal_move_ids(&catalog);
    assert!(!legal.is_empty());
    for move_id in legal {
        let action = catalog.get(move_id).unwrap();
        assert!(!action.is_pass());
        assert!(action.mask & card_bit(THREE_DIAMONDS) != 0);
    }

    let single_three_diamonds = catalog.move_id_for_mask(mask_from_cards(&[0])).unwrap();
    env.step(single_three_diamonds, &catalog).unwrap();
    assert!(!env.state.is_first_turn);
}

#[test]
fn response_legal_moves_include_pass_and_only_moves_that_beat_previous() {
    let catalog = ActionCatalog::new();
    let previous = catalog.move_id_for_mask(mask_from_cards(&[0])).unwrap();
    let mut state = state_with_hands(
        [
            0,
            mask_from_cards(&[1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]),
            0,
            0,
        ],
        1,
    );
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);

    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };
    let legal = env.legal_move_ids(&catalog);

    assert!(legal.contains(&0));
    for move_id in legal.into_iter().filter(|&id| id != 0) {
        let action = catalog.get(move_id).unwrap();
        assert_eq!(action.kind, MoveKind::Single);
        assert!(catalog.beats(move_id, previous));
    }
}

#[test]
fn step_rejects_action_that_is_not_legal_for_current_state() {
    let catalog = ActionCatalog::new();
    let hand = mask_from_cards(&[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]);
    let mut env = Big2Env {
        state: state_with_hands([hand, 0, 0, 0], 0),
        rules: RulesConfig::default(),
    };

    let illegal_opening_move = catalog.move_id_for_mask(mask_from_cards(&[4])).unwrap();
    assert_eq!(
        env.step(illegal_opening_move, &catalog),
        Err(StepError::IllegalMove(illegal_opening_move))
    );
}

#[test]
fn straight_flush_comparison_prioritizes_rank_before_suit() {
    let catalog = ActionCatalog::new();
    let low_spade_straight_flush = catalog
        .move_id_for_mask(mask_from_cards(&[3, 7, 11, 15, 19]))
        .unwrap();
    let high_diamond_straight_flush = catalog
        .move_id_for_mask(mask_from_cards(&[4, 8, 12, 16, 20]))
        .unwrap();

    assert!(catalog.beats(high_diamond_straight_flush, low_spade_straight_flush));
    assert!(!catalog.beats(low_spade_straight_flush, high_diamond_straight_flush));
}
