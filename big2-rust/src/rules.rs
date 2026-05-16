use std::cmp::Ordering;

use crate::action::{MoveKind, MoveMeta};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RulesConfig {
    pub require_three_diamond_open: bool,
    pub allow_wheel_straight: bool,
    pub allow_two_in_straight: bool,
    pub game_ends_on_first_out: bool,
    pub passed_player_may_reenter: bool,
}

impl Default for RulesConfig {
    fn default() -> Self {
        Self {
            require_three_diamond_open: true,
            allow_wheel_straight: false,
            allow_two_in_straight: false,
            game_ends_on_first_out: true,
            passed_player_may_reenter: false,
        }
    }
}

pub fn compare_moves_same_kind(left: &MoveMeta, right: &MoveMeta) -> Ordering {
    debug_assert_eq!(left.kind, right.kind);
    match left.kind {
        MoveKind::Pass => Ordering::Equal,
        MoveKind::Single
        | MoveKind::Pair
        | MoveKind::Triple
        | MoveKind::Straight
        | MoveKind::StraightFlush => {
            (left.primary_rank, left.high_suit).cmp(&(right.primary_rank, right.high_suit))
        }
        MoveKind::Flush => {
            (left.ranks_desc, left.high_suit).cmp(&(right.ranks_desc, right.high_suit))
        }
        MoveKind::FullHouse | MoveKind::FourOfKind => left.primary_rank.cmp(&right.primary_rank),
    }
}

pub fn straight_high_rank(unique_ranks: &[u8], rules: RulesConfig) -> Option<u8> {
    if unique_ranks.len() != 5 {
        return None;
    }
    if !rules.allow_two_in_straight && unique_ranks.contains(&12) {
        return None;
    }
    if rules.allow_wheel_straight && unique_ranks == [0, 1, 2, 11, 12] {
        return Some(2);
    }
    unique_ranks
        .windows(2)
        .all(|window| window[0] + 1 == window[1])
        .then_some(*unique_ranks.last().expect("five unique ranks"))
}
