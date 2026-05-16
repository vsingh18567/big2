use big2_rust::card::{CARDS_PER_PLAYER, PLAYERS, THREE_DIAMONDS, card_bit, mask_from_cards};
use big2_rust::{ActionCatalog, Big2Env, GameState, MoveId, RulesConfig, StepError};

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

fn env_with_hand(hand: &[u8]) -> Big2Env {
    Big2Env {
        state: state_with_hands([mask_from_cards(hand), 0, 0, 0], 0),
        rules: RulesConfig::default(),
    }
}

fn move_for(catalog: &ActionCatalog, cards: &[u8]) -> MoveId {
    catalog.move_id_for_mask(mask_from_cards(cards)).unwrap()
}

fn legal_formats(env: &Big2Env, catalog: &ActionCatalog) -> Vec<String> {
    env.legal_move_ids(catalog)
        .into_iter()
        .map(|id| catalog.format_move(id).unwrap())
        .collect()
}

#[test]
fn reset_deals_all_cards_disjointly_and_starts_with_three_diamonds_holder() {
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
    assert!(env.state.is_first_turn);
}

#[test]
fn reset_is_deterministic_for_same_seed_and_changes_for_different_seed() {
    let env_a = Big2Env::new(123);
    let env_b = Big2Env::new(123);
    let env_c = Big2Env::new(124);

    assert_eq!(env_a.state.hands, env_b.state.hands);
    assert_eq!(env_a.state.current_player, env_b.state.current_player);
    assert_ne!(env_a.state.hands, env_c.state.hands);
}

#[test]
fn done_state_has_no_legal_moves() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0]);
    env.state.done = true;

    assert!(env.legal_move_ids(&catalog).is_empty());
}

#[test]
fn opening_legal_moves_must_include_three_diamonds_and_cannot_pass() {
    let catalog = ActionCatalog::new();
    let env = env_with_hand(&[0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 32]);
    let legal = env.legal_move_ids(&catalog);

    assert!(!legal.is_empty());
    assert!(!legal.contains(&0));
    for move_id in legal {
        let action = catalog.get(move_id).unwrap();
        assert!(!action.is_pass());
        assert!(action.mask & card_bit(THREE_DIAMONDS) != 0);
    }
}

#[test]
fn rules_config_can_disable_three_diamonds_opening_requirement() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0, 4, 8, 12, 16]);
    env.rules = RulesConfig {
        require_three_diamond_open: false,
        ..RulesConfig::default()
    };
    let formatted = legal_formats(&env, &catalog);

    assert!(env.state.is_first_turn);
    assert!(formatted.contains(&"Single 4D".to_string()));
    assert!(formatted.contains(&"Straight flush [3D 4D 5D 6D 7D]".to_string()));
}

#[test]
fn non_opening_free_lead_allows_all_contained_non_pass_moves() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0, 1, 2, 3, 4, 5, 8, 12, 16]);
    env.state.is_first_turn = false;
    let legal = env.legal_move_ids(&catalog);
    let formatted = legal_formats(&env, &catalog);

    assert!(!legal.contains(&0));
    assert!(formatted.contains(&"Single 3D".to_string()));
    assert!(formatted.contains(&"Single 4D".to_string()));
    assert!(formatted.contains(&"Pair [3D 3C]".to_string()));
    assert!(formatted.contains(&"Triple [3D 3C 3H]".to_string()));
    assert!(formatted.contains(&"Straight flush [3D 4D 5D 6D 7D]".to_string()));
}

#[test]
fn following_single_allows_pass_and_only_higher_singles_from_hand() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[1]);
    let mut state = state_with_hands([0, mask_from_cards(&[0, 2, 4, 7, 8]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);
    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };
    let formatted = legal_formats(&env, &catalog);

    assert_eq!(
        formatted,
        vec![
            "Pass".to_string(),
            "Single 3H".to_string(),
            "Single 4D".to_string(),
            "Single 4S".to_string(),
            "Single 5D".to_string(),
        ]
    );
}

#[test]
fn following_pair_allows_pass_and_only_pairs_that_beat_previous_pair() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[0, 2]);
    let mut state = state_with_hands([0, mask_from_cards(&[0, 1, 2, 3, 4, 5, 8, 9, 10]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);
    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };
    let formatted = legal_formats(&env, &catalog);

    assert!(formatted.contains(&"Pass".to_string()));
    assert!(!formatted.contains(&"Pair [3D 3C]".to_string()));
    assert!(formatted.contains(&"Pair [3C 3S]".to_string()));
    assert!(formatted.contains(&"Pair [4D 4C]".to_string()));
    assert!(formatted.contains(&"Pair [5D 5C]".to_string()));
    assert!(formatted.contains(&"Pair [5D 5H]".to_string()));
    assert!(
        formatted
            .iter()
            .all(|text| text == "Pass" || text.starts_with("Pair "))
    );
}

#[test]
fn following_triple_allows_pass_and_only_higher_triples() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[4, 5, 6]);
    let mut state = state_with_hands(
        [
            0,
            mask_from_cards(&[0, 1, 2, 8, 9, 10, 11, 12, 13, 14]),
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
    let formatted = legal_formats(&env, &catalog);

    assert_eq!(
        formatted,
        vec![
            "Pass".to_string(),
            "Triple [5D 5C 5H]".to_string(),
            "Triple [5D 5C 5S]".to_string(),
            "Triple [5D 5H 5S]".to_string(),
            "Triple [5C 5H 5S]".to_string(),
            "Triple [6D 6C 6H]".to_string(),
        ]
    );
}

#[test]
fn following_five_card_allows_higher_same_category_and_higher_categories() {
    let catalog = ActionCatalog::new();
    let previous_straight = move_for(&catalog, &[0, 5, 8, 12, 16]);
    let mut state = state_with_hands(
        [
            0,
            mask_from_cards(&[
                4, 8, 12, 16, 20, // higher straight / straight flush
                1, 5, 9, 13, 21, // flush
                24, 25, 26, 28, 29, // full house
            ]),
            0,
            0,
        ],
        1,
    );
    state.is_first_turn = false;
    state.last_move_id = Some(previous_straight);
    state.last_actor = Some(0);
    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };
    let formatted = legal_formats(&env, &catalog);

    assert!(formatted.contains(&"Pass".to_string()));
    assert!(formatted.contains(&"Straight flush [4D 5D 6D 7D 8D]".to_string()));
    assert!(formatted.iter().any(|text| text.starts_with("Flush ")));
    assert!(formatted.contains(&"Full house [9D 9C 9H TD TC]".to_string()));
    assert!(formatted.iter().all(|text| text == "Pass"
        || text.starts_with("Straight ")
        || text.starts_with("Flush ")
        || text.starts_with("Full house ")
        || text.starts_with("Four of a kind ")
        || text.starts_with("Straight flush ")));
}

#[test]
fn player_who_already_passed_in_active_trick_can_only_pass() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[0]);
    let mut state = state_with_hands([0, mask_from_cards(&[4, 5, 6]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);
    state.passed[1] = true;

    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };
    assert_eq!(env.legal_move_ids(&catalog), vec![0]);
}

#[test]
fn rules_config_can_allow_passed_player_to_reenter_active_trick() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[0]);
    let mut state = state_with_hands([0, mask_from_cards(&[4, 5, 6]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);
    state.passed[1] = true;
    let env = Big2Env {
        state,
        rules: RulesConfig {
            passed_player_may_reenter: true,
            ..RulesConfig::default()
        },
    };
    let formatted = legal_formats(&env, &catalog);

    assert!(formatted.contains(&"Pass".to_string()));
    assert!(formatted.contains(&"Single 4D".to_string()));
}

#[test]
fn pass_is_illegal_on_free_lead() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0, 4, 8]);
    env.state.is_first_turn = false;

    assert_eq!(env.step(0, &catalog), Err(StepError::IllegalMove(0)));
}

#[test]
fn step_playing_move_updates_hand_public_state_and_turn() {
    let catalog = ActionCatalog::new();
    let hand = mask_from_cards(&[0, 1, 4]);
    let mut env = Big2Env {
        state: state_with_hands([hand, 0, 0, 0], 0),
        rules: RulesConfig::default(),
    };
    let pair = move_for(&catalog, &[0, 1]);

    env.step(pair, &catalog).unwrap();

    assert_eq!(env.state.hands[0], mask_from_cards(&[4]));
    assert_eq!(env.state.cards_played, mask_from_cards(&[0, 1]));
    assert_eq!(env.state.cards_remaining[0], 1);
    assert_eq!(env.state.last_move_id, Some(pair));
    assert_eq!(env.state.last_actor, Some(0));
    assert_eq!(env.state.passed, [false; PLAYERS]);
    assert!(!env.state.is_first_turn);
    assert_eq!(env.state.current_player, 1);
    assert!(!env.state.done);
}

#[test]
fn step_winning_move_marks_game_done_and_does_not_advance_turn() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0]);
    let single = move_for(&catalog, &[0]);

    env.step(single, &catalog).unwrap();

    assert!(env.state.done);
    assert!(env.state.finished[0]);
    assert_eq!(env.state.cards_remaining[0], 0);
    assert_eq!(env.state.current_player, 0);
    assert_eq!(env.step(single, &catalog), Err(StepError::GameDone));
}

#[test]
fn rules_config_can_continue_after_first_player_goes_out() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0]);
    env.rules = RulesConfig {
        game_ends_on_first_out: false,
        ..RulesConfig::default()
    };
    let single = move_for(&catalog, &[0]);

    env.step(single, &catalog).unwrap();

    assert!(!env.state.done);
    assert!(env.state.finished[0]);
    assert_eq!(env.state.cards_remaining[0], 0);
    assert_eq!(env.state.current_player, 1);
}

#[test]
fn step_unknown_move_id_is_rejected() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0]);

    assert_eq!(
        env.step(999_999, &catalog),
        Err(StepError::UnknownMoveId(999_999))
    );
}

#[test]
fn three_passes_clear_active_trick_and_return_control_to_last_actor() {
    let catalog = ActionCatalog::new();
    let mut env = Big2Env {
        state: state_with_hands(
            [
                mask_from_cards(&[0, 4]),
                mask_from_cards(&[8]),
                mask_from_cards(&[12]),
                mask_from_cards(&[16]),
            ],
            0,
        ),
        rules: RulesConfig::default(),
    };
    let opening = move_for(&catalog, &[0]);
    env.step(opening, &catalog).unwrap();
    assert_eq!(env.state.current_player, 1);

    env.step(0, &catalog).unwrap();
    assert_eq!(env.state.passed[1], true);
    assert_eq!(env.state.current_player, 2);

    env.step(0, &catalog).unwrap();
    assert_eq!(env.state.passed[2], true);
    assert_eq!(env.state.current_player, 3);

    env.step(0, &catalog).unwrap();
    assert_eq!(env.state.last_move_id, None);
    assert_eq!(env.state.last_actor, None);
    assert_eq!(env.state.passed, [false; PLAYERS]);
    assert_eq!(env.state.current_player, 0);

    let formatted = legal_formats(&env, &catalog);
    assert!(!formatted.contains(&"Pass".to_string()));
    assert!(formatted.contains(&"Single 4D".to_string()));
}

#[test]
fn beating_move_resets_passes_and_becomes_new_active_trick() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[0]);
    let mut state = state_with_hands([0, mask_from_cards(&[4, 8]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);
    state.passed[2] = true;
    let mut env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };
    let beating_single = move_for(&catalog, &[4]);

    env.step(beating_single, &catalog).unwrap();

    assert_eq!(env.state.last_move_id, Some(beating_single));
    assert_eq!(env.state.last_actor, Some(1));
    assert_eq!(env.state.passed, [false; PLAYERS]);
    assert_eq!(env.state.current_player, 2);
}

#[test]
fn legal_move_generation_never_returns_cards_outside_current_hand() {
    let catalog = ActionCatalog::new();
    let mut env = env_with_hand(&[0, 1, 4, 5, 8, 12, 16]);
    env.state.is_first_turn = false;

    for move_id in env.legal_move_ids(&catalog) {
        let action = catalog.get(move_id).unwrap();
        assert_eq!(action.mask & env.state.hands[0], action.mask);
    }
}

#[test]
fn legal_move_generation_only_returns_moves_that_beat_active_trick() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[0, 1]);
    let mut state = state_with_hands([0, mask_from_cards(&[0, 1, 2, 4, 5, 8, 9]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);
    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };

    for move_id in env
        .legal_move_ids(&catalog)
        .into_iter()
        .filter(|&id| id != 0)
    {
        assert!(catalog.beats(move_id, previous));
    }
}

#[test]
fn legal_move_generation_can_return_no_non_pass_when_player_cannot_beat() {
    let catalog = ActionCatalog::new();
    let previous = move_for(&catalog, &[51]);
    let mut state = state_with_hands([0, mask_from_cards(&[0, 1, 2, 3]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(previous);
    state.last_actor = Some(0);
    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };

    assert_eq!(env.legal_move_ids(&catalog), vec![0]);
}
