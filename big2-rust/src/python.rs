use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rules::RulesConfig;
use crate::vec_env::{Big2VecEnv, DEFAULT_MAX_CANDIDATES, VecEnvBatch};
use crate::{MoveId, MoveMeta, OBS_DIM};

const MOVE_KIND_COUNT: usize = 9;
const MOVE_FEATURE_DIM: usize = 52 + MOVE_KIND_COUNT + 1 + 13 + 13 + 4 + 5;

type PyBatch = (
    usize,
    usize,
    usize,
    Vec<f32>,
    Vec<i32>,
    Vec<bool>,
    Vec<u32>,
    Vec<bool>,
    Vec<Vec<f32>>,
    usize,
);

type PyMoveMeta = (u32, u64, u8, u8, u8, u8, u8, Vec<u8>);
type PyMoveFeatures = (usize, Vec<f32>);

/// Python wrapper for the Rust vectorized Big 2 environment.
#[pyclass(name = "Big2VecEnv")]
pub struct PyBig2VecEnv {
    inner: Big2VecEnv,
}

#[pymethods]
impl PyBig2VecEnv {
    #[new]
    #[pyo3(signature = (
        num_envs,
        seed,
        max_candidates = DEFAULT_MAX_CANDIDATES,
        require_three_diamond_open = true,
        allow_wheel_straight = false,
        allow_two_in_straight = false,
        game_ends_on_first_out = true,
        passed_player_may_reenter = false,
    ))]
    fn new(
        num_envs: usize,
        seed: u64,
        max_candidates: usize,
        require_three_diamond_open: bool,
        allow_wheel_straight: bool,
        allow_two_in_straight: bool,
        game_ends_on_first_out: bool,
        passed_player_may_reenter: bool,
    ) -> Self {
        let rules = RulesConfig {
            require_three_diamond_open,
            allow_wheel_straight,
            allow_two_in_straight,
            game_ends_on_first_out,
            passed_player_may_reenter,
        };
        Self {
            inner: Big2VecEnv::new(num_envs, seed, rules, max_candidates),
        }
    }

    fn reset(&mut self) -> PyBatch {
        batch_to_python(self.inner.reset())
    }

    fn reset_done(&mut self, env_indices: Vec<usize>) -> PyBatch {
        batch_to_python(self.inner.reset_indices(&env_indices))
    }

    fn step(&mut self, action_ids: Vec<MoveId>) -> PyResult<PyBatch> {
        self.inner
            .step(&action_ids)
            .map(batch_to_python)
            .map_err(|error| PyValueError::new_err(format!("{error:?}")))
    }

    fn move_metadata(&self) -> Vec<PyMoveMeta> {
        self.inner
            .catalog()
            .all_moves()
            .iter()
            .map(|meta| {
                (
                    meta.id,
                    meta.mask,
                    meta.kind as u8,
                    meta.num_cards,
                    meta.primary_rank,
                    meta.secondary_rank,
                    meta.high_suit,
                    meta.ranks_desc.to_vec(),
                )
            })
            .collect()
    }

    fn move_features(&self) -> PyMoveFeatures {
        let mut features =
            Vec::with_capacity(self.inner.catalog().all_moves().len() * MOVE_FEATURE_DIM);
        for meta in self.inner.catalog().all_moves() {
            append_move_features(&mut features, meta);
        }
        (MOVE_FEATURE_DIM, features)
    }

    #[getter]
    fn num_actions(&self) -> usize {
        self.inner.catalog().all_moves().len()
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    #[getter]
    fn max_candidates(&self) -> usize {
        self.inner.max_candidates()
    }
}

fn append_move_features(features: &mut Vec<f32>, meta: &MoveMeta) {
    for card in 0..52 {
        let bit = 1u64 << card;
        features.push(if meta.mask & bit != 0 { 1.0 } else { 0.0 });
    }

    for kind in 0..MOVE_KIND_COUNT {
        features.push(if meta.kind as usize == kind { 1.0 } else { 0.0 });
    }

    features.push(meta.num_cards as f32 / 5.0);

    for rank in 0..13 {
        features.push(if meta.primary_rank as usize == rank {
            1.0
        } else {
            0.0
        });
    }

    for rank in 0..13 {
        features.push(if meta.secondary_rank as usize == rank {
            1.0
        } else {
            0.0
        });
    }

    for suit in 0..4 {
        features.push(if !meta.is_pass() && meta.high_suit as usize == suit {
            1.0
        } else {
            0.0
        });
    }

    for rank in meta.ranks_desc {
        features.push(if rank > 0 { rank as f32 / 12.0 } else { 0.0 });
    }
}

#[pymodule]
pub fn big2_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBig2VecEnv>()?;
    m.add("OBS_DIM", OBS_DIM)?;
    m.add("DEFAULT_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES)?;
    Ok(())
}

fn batch_to_python(batch: VecEnvBatch) -> PyBatch {
    (
        batch.num_envs,
        batch.obs_dim,
        batch.max_candidates,
        batch.observations,
        batch.candidate_ids,
        batch.candidate_mask,
        batch.current_player.into_iter().map(u32::from).collect(),
        batch.done,
        batch
            .final_rewards
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        batch.truncated_candidate_lists,
    )
}
