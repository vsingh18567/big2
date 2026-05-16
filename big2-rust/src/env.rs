use crate::action::{ActionCatalog, MoveId, MoveKind};
use crate::card::{CARDS_PER_DECK, CARDS_PER_PLAYER, CardMask, PLAYERS, THREE_DIAMONDS, card_bit};
use crate::observation::{Observation, observe_player};
use crate::rules::RulesConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameState {
    pub hands: [CardMask; PLAYERS],
    pub current_player: u8,
    pub last_move_id: Option<MoveId>,
    pub last_actor: Option<u8>,
    pub passed: [bool; PLAYERS],
    pub cards_played: CardMask,
    pub cards_remaining: [u8; PLAYERS],
    pub finished: [bool; PLAYERS],
    pub is_first_turn: bool,
    pub done: bool,
    pub rng_state: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    GameDone,
    UnknownMoveId(MoveId),
    IllegalMove(MoveId),
}

#[derive(Debug, Clone)]
pub struct Big2Env {
    pub state: GameState,
    pub rules: RulesConfig,
}

impl Big2Env {
    pub fn new(seed: u64) -> Self {
        Self::new_with_rules(seed, RulesConfig::default())
    }

    pub fn new_with_rules(seed: u64, rules: RulesConfig) -> Self {
        let mut env = Self {
            state: empty_state(seed),
            rules,
        };
        env.reset(seed);
        env
    }

    pub fn reset(&mut self, seed: u64) {
        let mut deck = [0u8; CARDS_PER_DECK];
        for (idx, card) in deck.iter_mut().enumerate() {
            *card = idx as u8;
        }

        let mut rng = XorShift64::new(seed);
        for idx in (1..CARDS_PER_DECK).rev() {
            let swap_idx = rng.gen_range(idx + 1);
            deck.swap(idx, swap_idx);
        }

        let mut hands = [0u64; PLAYERS];
        for player in 0..PLAYERS {
            for card in &deck[player * CARDS_PER_PLAYER..(player + 1) * CARDS_PER_PLAYER] {
                hands[player] |= card_bit(*card);
            }
        }

        let current_player = hands
            .iter()
            .position(|hand| hand & card_bit(THREE_DIAMONDS) != 0)
            .unwrap_or(0) as u8;

        self.state = GameState {
            hands,
            current_player,
            last_move_id: None,
            last_actor: None,
            passed: [false; PLAYERS],
            cards_played: 0,
            cards_remaining: [CARDS_PER_PLAYER as u8; PLAYERS],
            finished: [false; PLAYERS],
            is_first_turn: true,
            done: false,
            rng_state: rng.state,
        };
    }

    pub fn legal_move_ids(&self, catalog: &ActionCatalog) -> Vec<MoveId> {
        legal_move_ids_for_state_with_rules(&self.state, catalog, self.rules)
    }

    pub fn observe_current_player(&self, catalog: &ActionCatalog) -> Observation {
        observe_player(&self.state, catalog, self.state.current_player)
    }

    pub fn step(&mut self, action_id: MoveId, catalog: &ActionCatalog) -> Result<(), StepError> {
        if self.state.done {
            return Err(StepError::GameDone);
        }
        let Some(action) = catalog.get(action_id) else {
            return Err(StepError::UnknownMoveId(action_id));
        };
        if !self.legal_move_ids(catalog).contains(&action_id) {
            return Err(StepError::IllegalMove(action_id));
        }

        let player = self.state.current_player as usize;
        if action.kind == MoveKind::Pass {
            self.state.passed[player] = true;
            let active_passes = self.state.passed.iter().filter(|&&passed| passed).count();
            if active_passes >= PLAYERS - 1 {
                self.state.last_move_id = None;
                self.state.last_actor = None;
                self.state.passed = [false; PLAYERS];
            }
        } else {
            self.state.hands[player] &= !action.mask;
            self.state.cards_played |= action.mask;
            self.state.cards_remaining[player] -= action.num_cards;
            self.state.last_move_id = Some(action_id);
            self.state.last_actor = Some(self.state.current_player);
            self.state.passed = [false; PLAYERS];
            self.state.is_first_turn = false;
            if self.state.cards_remaining[player] == 0 {
                self.state.finished[player] = true;
                if self.rules.game_ends_on_first_out {
                    self.state.done = true;
                }
            }
        }

        if !self.state.done {
            self.state.current_player = ((self.state.current_player as usize + 1) % PLAYERS) as u8;
        }
        Ok(())
    }
}

pub fn legal_move_ids_for_state(state: &GameState, catalog: &ActionCatalog) -> Vec<MoveId> {
    legal_move_ids_for_state_with_rules(state, catalog, RulesConfig::default())
}

pub fn legal_move_ids_for_state_with_rules(
    state: &GameState,
    catalog: &ActionCatalog,
    rules: RulesConfig,
) -> Vec<MoveId> {
    if state.done {
        return Vec::new();
    }

    let player = state.current_player as usize;
    if state.passed[player] && state.last_move_id.is_some() && !rules.passed_player_may_reenter {
        return vec![0];
    }

    let hand = state.hands[player];
    let mut legal = Vec::new();

    match state.last_move_id {
        None => {
            for action in catalog.all_moves() {
                if action.is_pass() {
                    continue;
                }
                if action.mask & hand != action.mask {
                    continue;
                }
                if rules.require_three_diamond_open
                    && state.is_first_turn
                    && action.mask & card_bit(THREE_DIAMONDS) == 0
                {
                    continue;
                }
                legal.push(action.id);
            }
        }
        Some(previous_id) => {
            legal.push(0);
            for action in catalog.all_moves() {
                if action.is_pass() {
                    continue;
                }
                if action.mask & hand != action.mask {
                    continue;
                }
                if catalog.beats(action.id, previous_id) {
                    legal.push(action.id);
                }
            }
        }
    }

    legal
}

fn empty_state(seed: u64) -> GameState {
    GameState {
        hands: [0; PLAYERS],
        current_player: 0,
        last_move_id: None,
        last_actor: None,
        passed: [false; PLAYERS],
        cards_played: 0,
        cards_remaining: [0; PLAYERS],
        finished: [false; PLAYERS],
        is_first_turn: true,
        done: false,
        rng_state: seed,
    }
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_range(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() as usize) % upper_exclusive
    }
}
