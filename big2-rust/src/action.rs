use std::collections::HashMap;
use std::fmt;

use crate::card::{
    CARDS_PER_DECK, CardMask, card_name, card_rank, card_suit, cards_from_mask, format_card_mask,
    mask_from_cards,
};
use crate::rules::{RulesConfig, compare_moves_same_kind, straight_high_rank};

pub type MoveId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MoveKind {
    Pass = 0,
    Single = 1,
    Pair = 2,
    Triple = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfKind = 7,
    StraightFlush = 8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveMeta {
    pub id: MoveId,
    pub mask: CardMask,
    pub kind: MoveKind,
    pub num_cards: u8,
    pub primary_rank: u8,
    pub secondary_rank: u8,
    pub high_suit: u8,
    pub ranks_desc: [u8; 5],
}

impl MoveMeta {
    pub fn is_pass(&self) -> bool {
        self.kind == MoveKind::Pass
    }

    pub fn display(&self) -> String {
        if self.is_pass() {
            return "Pass".to_string();
        }
        if self.kind == MoveKind::Single {
            let card = cards_from_mask(self.mask)
                .into_iter()
                .next()
                .expect("single move has one card");
            return format!("{} {}", self.kind, card_name(card));
        }
        format!("{} {}", self.kind, format_card_mask(self.mask))
    }
}

impl fmt::Display for MoveKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            MoveKind::Pass => "Pass",
            MoveKind::Single => "Single",
            MoveKind::Pair => "Pair",
            MoveKind::Triple => "Triple",
            MoveKind::Straight => "Straight",
            MoveKind::Flush => "Flush",
            MoveKind::FullHouse => "Full house",
            MoveKind::FourOfKind => "Four of a kind",
            MoveKind::StraightFlush => "Straight flush",
        };
        f.write_str(name)
    }
}

impl fmt::Display for MoveMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display())
    }
}

#[derive(Debug, Clone)]
pub struct ActionCatalog {
    moves: Vec<MoveMeta>,
    move_by_mask: HashMap<CardMask, MoveId>,
    five_card_move_by_mask: HashMap<CardMask, MoveId>,
}

impl Default for ActionCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionCatalog {
    pub fn new() -> Self {
        Self::new_with_rules(RulesConfig::default())
    }

    pub fn new_with_rules(rules: RulesConfig) -> Self {
        let mut catalog = Self {
            moves: Vec::new(),
            move_by_mask: HashMap::new(),
            five_card_move_by_mask: HashMap::new(),
        };
        catalog.push(MoveMeta {
            id: 0,
            mask: 0,
            kind: MoveKind::Pass,
            num_cards: 0,
            primary_rank: 0,
            secondary_rank: 0,
            high_suit: 0,
            ranks_desc: [0; 5],
        });
        catalog.generate_small_moves();
        catalog.generate_five_card_moves(rules);
        catalog
    }

    pub fn all_moves(&self) -> &[MoveMeta] {
        &self.moves
    }

    pub fn get(&self, id: MoveId) -> Option<&MoveMeta> {
        self.moves.get(id as usize).filter(|meta| meta.id == id)
    }

    pub fn move_id_for_mask(&self, mask: CardMask) -> Option<MoveId> {
        self.move_by_mask.get(&mask).copied()
    }

    pub fn five_card_move_id_for_mask(&self, mask: CardMask) -> Option<MoveId> {
        self.five_card_move_by_mask.get(&mask).copied()
    }

    pub fn format_move(&self, id: MoveId) -> Option<String> {
        self.get(id).map(MoveMeta::display)
    }

    pub fn beats(&self, candidate_id: MoveId, previous_id: MoveId) -> bool {
        let Some(candidate) = self.get(candidate_id) else {
            return false;
        };
        let Some(previous) = self.get(previous_id) else {
            return false;
        };
        if candidate.is_pass() || previous.is_pass() || candidate.num_cards != previous.num_cards {
            return false;
        }
        if candidate.kind == previous.kind {
            return compare_moves_same_kind(candidate, previous).is_gt();
        }
        candidate.num_cards == 5 && previous.num_cards == 5 && candidate.kind > previous.kind
    }

    fn push(&mut self, mut meta: MoveMeta) -> MoveId {
        meta.id = self.moves.len() as MoveId;
        let id = meta.id;
        if meta.mask != 0 {
            self.move_by_mask.insert(meta.mask, id);
            if meta.num_cards == 5 {
                self.five_card_move_by_mask.insert(meta.mask, id);
            }
        }
        self.moves.push(meta);
        id
    }

    fn generate_small_moves(&mut self) {
        for card in 0..CARDS_PER_DECK as u8 {
            self.push(MoveMeta {
                id: 0,
                mask: mask_from_cards(&[card]),
                kind: MoveKind::Single,
                num_cards: 1,
                primary_rank: card_rank(card),
                secondary_rank: 0,
                high_suit: card_suit(card),
                ranks_desc: [card_rank(card), 0, 0, 0, 0],
            });
        }

        for rank in 0..13u8 {
            let base = rank * 4;
            for first_suit in 0..4u8 {
                for second_suit in (first_suit + 1)..4u8 {
                    self.push(MoveMeta {
                        id: 0,
                        mask: mask_from_cards(&[base + first_suit, base + second_suit]),
                        kind: MoveKind::Pair,
                        num_cards: 2,
                        primary_rank: rank,
                        secondary_rank: 0,
                        high_suit: second_suit,
                        ranks_desc: [rank, rank, 0, 0, 0],
                    });
                }
            }
        }

        for rank in 0..13u8 {
            let base = rank * 4;
            for a in 0..4u8 {
                for b in (a + 1)..4u8 {
                    for c in (b + 1)..4u8 {
                        self.push(MoveMeta {
                            id: 0,
                            mask: mask_from_cards(&[base + a, base + b, base + c]),
                            kind: MoveKind::Triple,
                            num_cards: 3,
                            primary_rank: rank,
                            secondary_rank: 0,
                            high_suit: c,
                            ranks_desc: [rank, rank, rank, 0, 0],
                        });
                    }
                }
            }
        }
    }

    fn generate_five_card_moves(&mut self, rules: RulesConfig) {
        for a in 0..48u8 {
            for b in (a + 1)..49u8 {
                for c in (b + 1)..50u8 {
                    for d in (c + 1)..51u8 {
                        for e in (d + 1)..52u8 {
                            let cards = [a, b, c, d, e];
                            if let Some(meta) = classify_five_cards_with_rules(&cards, rules) {
                                self.push(meta);
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn classify_five_cards(cards: &[u8; 5]) -> Option<MoveMeta> {
    classify_five_cards_with_rules(cards, RulesConfig::default())
}

pub fn classify_five_cards_with_rules(cards: &[u8; 5], rules: RulesConfig) -> Option<MoveMeta> {
    let mask = mask_from_cards(cards);
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    for &card in cards {
        rank_counts[card_rank(card) as usize] += 1;
        suit_counts[card_suit(card) as usize] += 1;
    }

    let is_flush = suit_counts.iter().any(|&count| count == 5);
    let unique_ranks: Vec<u8> = rank_counts
        .iter()
        .enumerate()
        .filter_map(|(rank, &count)| (count > 0).then_some(rank as u8))
        .collect();
    let straight_high_rank = straight_high_rank(&unique_ranks, rules);
    let is_straight = straight_high_rank.is_some();

    let mut ranks_desc = [0u8; 5];
    let mut expanded_ranks = Vec::with_capacity(5);
    for rank in (0..13u8).rev() {
        for _ in 0..rank_counts[rank as usize] {
            expanded_ranks.push(rank);
        }
    }
    for (idx, rank) in expanded_ranks.into_iter().enumerate() {
        ranks_desc[idx] = rank;
    }

    if is_straight && is_flush {
        let high_rank = straight_high_rank.expect("straight has high rank");
        return Some(MoveMeta {
            id: 0,
            mask,
            kind: MoveKind::StraightFlush,
            num_cards: 5,
            primary_rank: high_rank,
            secondary_rank: 0,
            high_suit: highest_suit_for_rank(cards, high_rank),
            ranks_desc,
        });
    }

    if let Some(quad_rank) = rank_with_count(&rank_counts, 4) {
        return Some(MoveMeta {
            id: 0,
            mask,
            kind: MoveKind::FourOfKind,
            num_cards: 5,
            primary_rank: quad_rank,
            secondary_rank: 0,
            high_suit: 0,
            ranks_desc,
        });
    }

    if let (Some(trip_rank), Some(pair_rank)) = (
        rank_with_count(&rank_counts, 3),
        rank_with_count(&rank_counts, 2),
    ) {
        return Some(MoveMeta {
            id: 0,
            mask,
            kind: MoveKind::FullHouse,
            num_cards: 5,
            primary_rank: trip_rank,
            secondary_rank: pair_rank,
            high_suit: 0,
            ranks_desc,
        });
    }

    if is_flush {
        let highest_rank = ranks_desc[0];
        return Some(MoveMeta {
            id: 0,
            mask,
            kind: MoveKind::Flush,
            num_cards: 5,
            primary_rank: highest_rank,
            secondary_rank: 0,
            high_suit: highest_suit_for_rank(cards, highest_rank),
            ranks_desc,
        });
    }

    if is_straight {
        let high_rank = straight_high_rank.expect("straight has high rank");
        return Some(MoveMeta {
            id: 0,
            mask,
            kind: MoveKind::Straight,
            num_cards: 5,
            primary_rank: high_rank,
            secondary_rank: 0,
            high_suit: highest_suit_for_rank(cards, high_rank),
            ranks_desc,
        });
    }

    None
}

fn rank_with_count(rank_counts: &[u8; 13], target: u8) -> Option<u8> {
    (0..13u8)
        .rev()
        .find(|&rank| rank_counts[rank as usize] == target)
}

fn highest_suit_for_rank(cards: &[u8; 5], rank: u8) -> u8 {
    cards
        .iter()
        .copied()
        .filter(|&card| card_rank(card) == rank)
        .map(card_suit)
        .max()
        .unwrap_or(0)
}
