use big2_rust::card::{PLAYERS, mask_from_cards};
use big2_rust::{
    Big2Env, Big2VecEnv, DEFAULT_MAX_CANDIDATES, GameState, OBS_DIM, RulesConfig, VecEnvError,
    terminal_rewards,
};

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
        is_first_turn: false,
        done: false,
        rng_state: 1,
    }
}

#[test]
fn reset_returns_batched_observations_and_padded_candidates() {
    let mut env = Big2VecEnv::new(3, 123, RulesConfig::default(), 64);

    let batch = env.reset();

    assert_eq!(batch.num_envs, 3);
    assert_eq!(batch.obs_dim, OBS_DIM);
    assert_eq!(batch.max_candidates, 64);
    assert_eq!(batch.observations.len(), 3 * OBS_DIM);
    assert_eq!(batch.candidate_ids.len(), 3 * 64);
    assert_eq!(batch.candidate_mask.len(), 3 * 64);
    assert_eq!(batch.current_player.len(), 3);
    assert!(batch.current_player.iter().all(|&player| player < 4));
    assert_eq!(batch.done, vec![false, false, false]);
    assert_eq!(batch.final_rewards, vec![[0.0; PLAYERS]; 3]);
    assert_eq!(batch.truncated_candidate_lists, 0);

    for env_idx in 0..3 {
        let row = &batch.candidate_mask[env_idx * 64..(env_idx + 1) * 64];
        assert!(row.iter().any(|&is_valid| is_valid));
        for (slot, &is_valid) in row.iter().enumerate() {
            let id = batch.candidate_ids[env_idx * 64 + slot];
            if is_valid {
                assert!(id >= 0);
            } else {
                assert_eq!(id, -1);
            }
        }
    }
}

#[test]
fn step_advances_each_environment_with_its_own_action() {
    let mut env = Big2VecEnv::new(2, 77, RulesConfig::default(), DEFAULT_MAX_CANDIDATES);
    let first = env.reset();
    let actions = first
        .candidate_ids
        .chunks(first.max_candidates)
        .map(|row| row.iter().copied().find(|&id| id >= 0).unwrap() as u32)
        .collect::<Vec<_>>();

    let next = env.step(&actions).unwrap();

    assert_eq!(next.num_envs, 2);
    assert_eq!(next.observations.len(), 2 * OBS_DIM);
    assert_eq!(next.candidate_ids.len(), 2 * DEFAULT_MAX_CANDIDATES);
    assert_eq!(next.candidate_mask.len(), 2 * DEFAULT_MAX_CANDIDATES);
}

#[test]
fn reset_indices_replaces_only_requested_slots() {
    let mut env = Big2VecEnv::new(2, 77, RulesConfig::default(), DEFAULT_MAX_CANDIDATES);
    let first = env.reset();
    let first_hand_0 = env.env(0).unwrap().state.hands;
    let first_hand_1 = env.env(1).unwrap().state.hands;

    let batch = env.reset_indices(&[0]);

    assert_eq!(batch.num_envs, 2);
    assert_ne!(env.env(0).unwrap().state.hands, first_hand_0);
    assert_eq!(env.env(1).unwrap().state.hands, first_hand_1);
    assert_eq!(batch.current_player.len(), first.current_player.len());
}

#[test]
fn step_rejects_wrong_number_of_actions() {
    let mut env = Big2VecEnv::new(2, 99, RulesConfig::default(), 64);

    assert_eq!(
        env.step(&[0]).unwrap_err(),
        VecEnvError::WrongActionCount {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn step_reports_illegal_action_with_environment_index() {
    let mut env = Big2VecEnv::new(2, 99, RulesConfig::default(), 64);

    assert_eq!(
        env.step(&[0, 0]).unwrap_err(),
        VecEnvError::EnvStep {
            env_index: 0,
            source: big2_rust::StepError::IllegalMove(0),
        }
    );
}

#[test]
fn candidate_truncation_is_reported_and_uses_padding_contract() {
    let mut env = Big2VecEnv::new(
        1,
        123,
        RulesConfig {
            require_three_diamond_open: false,
            ..RulesConfig::default()
        },
        1,
    );

    let batch = env.reset();

    assert_eq!(batch.candidate_ids.len(), 1);
    assert_eq!(batch.candidate_mask, vec![true]);
    assert!(batch.candidate_ids[0] >= 0);
    assert_eq!(batch.truncated_candidate_lists, 1);
}

#[test]
fn terminal_rewards_score_first_out_and_remaining_cards() {
    let mut state = state_with_hands(
        [
            0,
            mask_from_cards(&[0, 1, 2]),
            mask_from_cards(&[4, 5]),
            mask_from_cards(&[8]),
        ],
        0,
    );
    state.done = true;
    state.finished[0] = true;
    state.cards_remaining = [0, 3, 2, 1];
    let env = Big2Env {
        state,
        rules: RulesConfig::default(),
    };

    assert_eq!(
        terminal_rewards(&env),
        [1.0, -3.0 / 13.0, -2.0 / 13.0, -1.0 / 13.0]
    );
}
