use crate::action::{ActionCatalog, MoveKind};
use crate::card::{CARDS_PER_DECK, PLAYERS, card_bit};
use crate::env::GameState;

pub const OBS_OWN_HAND_START: usize = 0;
pub const OBS_CARDS_PLAYED_START: usize = OBS_OWN_HAND_START + CARDS_PER_DECK;
pub const OBS_LAST_MOVE_KIND_START: usize = OBS_CARDS_PLAYED_START + CARDS_PER_DECK;
pub const OBS_LAST_MOVE_NUM_CARDS: usize = OBS_LAST_MOVE_KIND_START + 9;
pub const OBS_LAST_MOVE_PRIMARY_RANK: usize = OBS_LAST_MOVE_NUM_CARDS + 1;
pub const OBS_LAST_MOVE_SECONDARY_RANK: usize = OBS_LAST_MOVE_PRIMARY_RANK + 1;
pub const OBS_LAST_MOVE_HIGH_SUIT_START: usize = OBS_LAST_MOVE_SECONDARY_RANK + 1;
pub const OBS_CARDS_REMAINING_START: usize = OBS_LAST_MOVE_HIGH_SUIT_START + 4;
pub const OBS_PASSED_START: usize = OBS_CARDS_REMAINING_START + PLAYERS;
pub const OBS_FREE_LEAD: usize = OBS_PASSED_START + PLAYERS;
pub const OBS_FIRST_TURN: usize = OBS_FREE_LEAD + 1;
pub const OBS_LAST_ACTOR_REL_START: usize = OBS_FIRST_TURN + 1;
pub const OBS_DIM: usize = OBS_LAST_ACTOR_REL_START + PLAYERS + 1;

#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub features: Vec<f32>,
}

impl Observation {
    pub fn as_slice(&self) -> &[f32] {
        &self.features
    }
}

pub fn observe_current_player(state: &GameState, catalog: &ActionCatalog) -> Observation {
    observe_player(state, catalog, state.current_player)
}

pub fn observe_player(state: &GameState, catalog: &ActionCatalog, player: u8) -> Observation {
    let player = player as usize;
    let mut features = vec![0.0; OBS_DIM];

    for card in 0..CARDS_PER_DECK as u8 {
        if state.hands[player] & card_bit(card) != 0 {
            features[OBS_OWN_HAND_START + card as usize] = 1.0;
        }
        if state.cards_played & card_bit(card) != 0 {
            features[OBS_CARDS_PLAYED_START + card as usize] = 1.0;
        }
    }

    if let Some(last_move_id) = state.last_move_id {
        if let Some(last_move) = catalog.get(last_move_id) {
            features[OBS_LAST_MOVE_KIND_START + last_move.kind as usize] = 1.0;
            features[OBS_LAST_MOVE_NUM_CARDS] = last_move.num_cards as f32 / 5.0;
            features[OBS_LAST_MOVE_PRIMARY_RANK] = last_move.primary_rank as f32 / 12.0;
            features[OBS_LAST_MOVE_SECONDARY_RANK] = last_move.secondary_rank as f32 / 12.0;
            if last_move.kind != MoveKind::Pass {
                features[OBS_LAST_MOVE_HIGH_SUIT_START + last_move.high_suit as usize] = 1.0;
            }
        }
    } else {
        features[OBS_LAST_MOVE_KIND_START + MoveKind::Pass as usize] = 1.0;
    }

    for relative in 0..PLAYERS {
        let absolute = (player + relative) % PLAYERS;
        features[OBS_CARDS_REMAINING_START + relative] =
            state.cards_remaining[absolute] as f32 / 13.0;
        features[OBS_PASSED_START + relative] = if state.passed[absolute] { 1.0 } else { 0.0 };
    }

    features[OBS_FREE_LEAD] = if state.last_move_id.is_none() {
        1.0
    } else {
        0.0
    };
    features[OBS_FIRST_TURN] = if state.is_first_turn { 1.0 } else { 0.0 };

    match state.last_actor {
        None => features[OBS_LAST_ACTOR_REL_START] = 1.0,
        Some(actor) => {
            let relative = (actor as usize + PLAYERS - player) % PLAYERS;
            features[OBS_LAST_ACTOR_REL_START + 1 + relative] = 1.0;
        }
    }

    features
        .iter()
        .all(|value| value.is_finite())
        .then_some(Observation { features })
        .expect("observation features are finite")
}
