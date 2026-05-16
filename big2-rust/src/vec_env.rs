use crate::action::{ActionCatalog, MoveId};
use crate::card::PLAYERS;
use crate::env::{Big2Env, StepError};
use crate::observation::OBS_DIM;
use crate::rules::RulesConfig;

/// Default per-environment candidate row width.
///
/// Candidate IDs are returned as fixed-width padded rows so Python/PyTorch can
/// treat them as rectangular tensors. Legal Big 2 move counts should fit well
/// below this value; truncation is still counted explicitly in `VecEnvBatch`.
pub const DEFAULT_MAX_CANDIDATES: usize = 2048;

/// Batched observation and legal-action output for a vectorized environment.
///
/// `observations` is row-major with shape `[num_envs, obs_dim]`.
/// `candidate_ids` and `candidate_mask` are row-major with shape
/// `[num_envs, max_candidates]`. Invalid candidate slots use ID `-1` and mask
/// `false`.
#[derive(Debug, Clone, PartialEq)]
pub struct VecEnvBatch {
    pub num_envs: usize,
    pub obs_dim: usize,
    pub max_candidates: usize,
    pub observations: Vec<f32>,
    pub candidate_ids: Vec<i32>,
    pub candidate_mask: Vec<bool>,
    pub current_player: Vec<u8>,
    pub done: Vec<bool>,
    pub final_rewards: Vec<[f32; PLAYERS]>,
    pub truncated_candidate_lists: usize,
}

/// Errors that can stop a vectorized step before a full batch is produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VecEnvError {
    WrongActionCount { expected: usize, actual: usize },
    EnvStep { env_index: usize, source: StepError },
}

/// A fixed-size batch of independent Big 2 games sharing one action catalog.
///
/// This is the Rust-side shape expected by the eventual Python/PyO3 wrapper:
/// reset and step both return rectangular observation/candidate buffers, while
/// Python chooses one catalog `MoveId` per active environment.
#[derive(Debug, Clone)]
pub struct Big2VecEnv {
    envs: Vec<Big2Env>,
    catalog: ActionCatalog,
    rules: RulesConfig,
    base_seed: u64,
    reset_epoch: u64,
    max_candidates: usize,
}

impl Big2VecEnv {
    /// Create a vectorized environment with a caller-selected candidate row size.
    pub fn new(num_envs: usize, seed: u64, rules: RulesConfig, max_candidates: usize) -> Self {
        let catalog = ActionCatalog::new_with_rules(rules);
        let max_candidates = max_candidates.max(1);
        let envs = (0..num_envs)
            .map(|env_idx| Big2Env::new_with_rules(seed_for(seed, 0, env_idx), rules))
            .collect();

        Self {
            envs,
            catalog,
            rules,
            base_seed: seed,
            reset_epoch: 0,
            max_candidates,
        }
    }

    /// Create a vectorized environment using `DEFAULT_MAX_CANDIDATES`.
    pub fn with_default_max_candidates(num_envs: usize, seed: u64, rules: RulesConfig) -> Self {
        Self::new(num_envs, seed, rules, DEFAULT_MAX_CANDIDATES)
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn max_candidates(&self) -> usize {
        self.max_candidates
    }

    pub fn catalog(&self) -> &ActionCatalog {
        &self.catalog
    }

    pub fn env(&self, index: usize) -> Option<&Big2Env> {
        self.envs.get(index)
    }

    /// Reset every environment in the batch and return the first decision batch.
    pub fn reset(&mut self) -> VecEnvBatch {
        self.reset_epoch = self.reset_epoch.wrapping_add(1);
        for (env_idx, env) in self.envs.iter_mut().enumerate() {
            env.rules = self.rules;
            env.reset(seed_for(self.base_seed, self.reset_epoch, env_idx));
        }
        self.snapshot()
    }

    /// Reset only the environments whose indexes are provided.
    pub fn reset_indices(&mut self, env_indices: &[usize]) -> VecEnvBatch {
        for &env_idx in env_indices {
            if let Some(env) = self.envs.get_mut(env_idx) {
                env.rules = self.rules;
                env.reset(seed_for(
                    self.base_seed,
                    self.reset_epoch.wrapping_add(1),
                    env_idx,
                ));
            }
        }
        self.reset_epoch = self.reset_epoch.wrapping_add(1);
        self.snapshot()
    }

    /// Step each active environment once using one action ID per environment.
    ///
    /// Done environments are left unchanged, which lets a rollout collector
    /// consume terminal rewards before resetting the whole vectorized batch or
    /// replacing completed slots in a later API layer.
    pub fn step(&mut self, action_ids: &[MoveId]) -> Result<VecEnvBatch, VecEnvError> {
        if action_ids.len() != self.envs.len() {
            return Err(VecEnvError::WrongActionCount {
                expected: self.envs.len(),
                actual: action_ids.len(),
            });
        }

        for (env_index, (&action_id, env)) in
            action_ids.iter().zip(self.envs.iter_mut()).enumerate()
        {
            if env.state.done {
                continue;
            }
            env.step(action_id, &self.catalog)
                .map_err(|source| VecEnvError::EnvStep { env_index, source })?;
        }

        Ok(self.snapshot())
    }

    /// Return the current batched buffers without mutating any environment.
    pub fn snapshot(&self) -> VecEnvBatch {
        let num_envs = self.envs.len();
        let mut observations = Vec::with_capacity(num_envs * OBS_DIM);
        let mut candidate_ids = vec![-1; num_envs * self.max_candidates];
        let mut candidate_mask = vec![false; num_envs * self.max_candidates];
        let mut current_player = Vec::with_capacity(num_envs);
        let mut done = Vec::with_capacity(num_envs);
        let mut final_rewards = Vec::with_capacity(num_envs);
        let mut truncated_candidate_lists = 0;

        for (env_idx, env) in self.envs.iter().enumerate() {
            observations.extend(env.observe_current_player(&self.catalog).features);
            current_player.push(env.state.current_player);
            done.push(env.state.done);
            final_rewards.push(terminal_rewards(env));

            let legal = env.legal_move_ids(&self.catalog);
            if legal.len() > self.max_candidates {
                truncated_candidate_lists += 1;
            }
            for (slot, &move_id) in legal.iter().take(self.max_candidates).enumerate() {
                let offset = env_idx * self.max_candidates + slot;
                candidate_ids[offset] = move_id as i32;
                candidate_mask[offset] = true;
            }
        }

        VecEnvBatch {
            num_envs,
            obs_dim: OBS_DIM,
            max_candidates: self.max_candidates,
            observations,
            candidate_ids,
            candidate_mask,
            current_player,
            done,
            final_rewards,
            truncated_candidate_lists,
        }
    }
}

/// Terminal rewards for the current simple scoring rule.
///
/// The first player out receives `+1.0`; every other player receives the
/// negative fraction of cards still in hand. Non-terminal environments return
/// all zeros.
pub fn terminal_rewards(env: &Big2Env) -> [f32; PLAYERS] {
    if !env.state.done {
        return [0.0; PLAYERS];
    }

    let mut rewards = [0.0; PLAYERS];
    let winner = env
        .state
        .finished
        .iter()
        .position(|&finished| finished)
        .unwrap_or(env.state.current_player as usize);

    for (player, reward) in rewards.iter_mut().enumerate() {
        *reward = if player == winner {
            1.0
        } else {
            -(env.state.cards_remaining[player] as f32 / 13.0)
        };
    }
    rewards
}

fn seed_for(base_seed: u64, reset_epoch: u64, env_idx: usize) -> u64 {
    base_seed
        .wrapping_add(reset_epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add((env_idx as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9))
}
