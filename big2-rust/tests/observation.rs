use big2_rust::card::{PLAYERS, card_bit, mask_from_cards};
use big2_rust::observation::{
    OBS_CARDS_PLAYED_START, OBS_CARDS_REMAINING_START, OBS_DIM, OBS_FIRST_TURN, OBS_FREE_LEAD,
    OBS_LAST_ACTOR_REL_START, OBS_LAST_MOVE_HIGH_SUIT_START, OBS_LAST_MOVE_KIND_START,
    OBS_LAST_MOVE_NUM_CARDS, OBS_LAST_MOVE_PRIMARY_RANK, OBS_LAST_MOVE_SECONDARY_RANK,
    OBS_OWN_HAND_START, OBS_PASSED_START, observe_current_player, observe_player,
};
use big2_rust::{ActionCatalog, GameState, MoveKind};

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

fn move_for(catalog: &ActionCatalog, cards: &[u8]) -> u32 {
    catalog.move_id_for_mask(mask_from_cards(cards)).unwrap()
}

#[test]
fn observation_has_stable_dimension_and_encodes_own_hand_and_played_cards() {
    let catalog = ActionCatalog::new();
    let mut state = state_with_hands(
        [
            mask_from_cards(&[0, 4]),
            mask_from_cards(&[8]),
            mask_from_cards(&[12]),
            mask_from_cards(&[16]),
        ],
        0,
    );
    state.cards_played = mask_from_cards(&[20, 21]);
    let obs = observe_current_player(&state, &catalog);

    assert_eq!(obs.features.len(), OBS_DIM);
    assert_eq!(obs.features[OBS_OWN_HAND_START], 1.0);
    assert_eq!(obs.features[OBS_OWN_HAND_START + 4], 1.0);
    assert_eq!(obs.features[OBS_OWN_HAND_START + 8], 0.0);
    assert_eq!(obs.features[OBS_CARDS_PLAYED_START + 20], 1.0);
    assert_eq!(obs.features[OBS_CARDS_PLAYED_START + 21], 1.0);
    assert_eq!(obs.features[OBS_CARDS_PLAYED_START], 0.0);
}

#[test]
fn observation_encodes_empty_last_move_as_pass_free_lead_and_no_last_actor() {
    let catalog = ActionCatalog::new();
    let state = state_with_hands([mask_from_cards(&[0]), 0, 0, 0], 0);
    let obs = observe_current_player(&state, &catalog);

    assert_eq!(
        obs.features[OBS_LAST_MOVE_KIND_START + MoveKind::Pass as usize],
        1.0
    );
    assert_eq!(obs.features[OBS_LAST_MOVE_NUM_CARDS], 0.0);
    assert_eq!(obs.features[OBS_FREE_LEAD], 1.0);
    assert_eq!(obs.features[OBS_FIRST_TURN], 1.0);
    assert_eq!(obs.features[OBS_LAST_ACTOR_REL_START], 1.0);
    assert_eq!(
        obs.features[OBS_LAST_ACTOR_REL_START + 1..OBS_LAST_ACTOR_REL_START + 5],
        [0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn observation_encodes_last_move_metadata() {
    let catalog = ActionCatalog::new();
    let pair = move_for(&catalog, &[4, 7]);
    let mut state = state_with_hands([0, mask_from_cards(&[8]), 0, 0], 1);
    state.is_first_turn = false;
    state.last_move_id = Some(pair);
    state.last_actor = Some(0);
    let obs = observe_current_player(&state, &catalog);

    assert_eq!(
        obs.features[OBS_LAST_MOVE_KIND_START + MoveKind::Pair as usize],
        1.0
    );
    assert_eq!(obs.features[OBS_LAST_MOVE_NUM_CARDS], 2.0 / 5.0);
    assert_eq!(obs.features[OBS_LAST_MOVE_PRIMARY_RANK], 1.0 / 12.0);
    assert_eq!(obs.features[OBS_LAST_MOVE_SECONDARY_RANK], 0.0);
    assert_eq!(obs.features[OBS_LAST_MOVE_HIGH_SUIT_START + 3], 1.0);
    assert_eq!(obs.features[OBS_FREE_LEAD], 0.0);
    assert_eq!(obs.features[OBS_FIRST_TURN], 0.0);
}

#[test]
fn observation_perspective_normalizes_counts_passes_and_last_actor() {
    let catalog = ActionCatalog::new();
    let mut state = state_with_hands(
        [
            mask_from_cards(&[0, 1, 2]),
            mask_from_cards(&[4, 5]),
            mask_from_cards(&[8, 9, 10, 11]),
            mask_from_cards(&[12]),
        ],
        2,
    );
    state.passed = [true, false, true, false];
    state.last_actor = Some(0);
    state.last_move_id = Some(move_for(&catalog, &[0]));
    let obs = observe_current_player(&state, &catalog);

    assert_eq!(obs.features[OBS_CARDS_REMAINING_START], 4.0 / 13.0);
    assert_eq!(obs.features[OBS_CARDS_REMAINING_START + 1], 1.0 / 13.0);
    assert_eq!(obs.features[OBS_CARDS_REMAINING_START + 2], 3.0 / 13.0);
    assert_eq!(obs.features[OBS_CARDS_REMAINING_START + 3], 2.0 / 13.0);

    assert_eq!(obs.features[OBS_PASSED_START], 1.0);
    assert_eq!(obs.features[OBS_PASSED_START + 1], 0.0);
    assert_eq!(obs.features[OBS_PASSED_START + 2], 1.0);
    assert_eq!(obs.features[OBS_PASSED_START + 3], 0.0);

    assert_eq!(obs.features[OBS_LAST_ACTOR_REL_START], 0.0);
    assert_eq!(obs.features[OBS_LAST_ACTOR_REL_START + 3], 1.0);
}

#[test]
fn observe_player_can_encode_a_non_current_player_perspective() {
    let catalog = ActionCatalog::new();
    let state = state_with_hands(
        [
            mask_from_cards(&[0]),
            mask_from_cards(&[4]),
            mask_from_cards(&[8]),
            mask_from_cards(&[12]),
        ],
        0,
    );
    let obs = observe_player(&state, &catalog, 3);

    assert_eq!(obs.features[OBS_OWN_HAND_START + 12], 1.0);
    assert_eq!(obs.features[OBS_OWN_HAND_START], 0.0);
}

#[test]
fn observation_does_not_encode_opponent_hands_as_own_hand() {
    let catalog = ActionCatalog::new();
    let state = state_with_hands(
        [
            mask_from_cards(&[0]),
            mask_from_cards(&[4]),
            mask_from_cards(&[8]),
            mask_from_cards(&[12]),
        ],
        0,
    );
    let obs = observe_current_player(&state, &catalog);

    assert_eq!(obs.features[OBS_OWN_HAND_START], 1.0);
    assert_eq!(obs.features[OBS_OWN_HAND_START + 4], 0.0);
    assert_eq!(obs.features[OBS_OWN_HAND_START + 8], 0.0);
    assert_eq!(obs.features[OBS_OWN_HAND_START + 12], 0.0);
    assert_eq!(state.hands[1] & card_bit(4), card_bit(4));
}
