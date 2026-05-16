pub type CardMask = u64;

pub const CARDS_PER_DECK: usize = 52;
pub const PLAYERS: usize = 4;
pub const CARDS_PER_PLAYER: usize = CARDS_PER_DECK / PLAYERS;
pub const THREE_DIAMONDS: u8 = 0;

pub const RANK_NAMES: [&str; 13] = [
    "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2",
];
pub const SUIT_NAMES: [&str; 4] = ["D", "C", "H", "S"];

#[inline]
pub const fn card_bit(card: u8) -> CardMask {
    1u64 << card
}

#[inline]
pub const fn card_rank(card: u8) -> u8 {
    card / 4
}

#[inline]
pub const fn card_suit(card: u8) -> u8 {
    card % 4
}

pub fn cards_from_mask(mask: CardMask) -> Vec<u8> {
    let mut cards = Vec::with_capacity(mask.count_ones() as usize);
    for card in 0..CARDS_PER_DECK as u8 {
        if mask & card_bit(card) != 0 {
            cards.push(card);
        }
    }
    cards
}

pub fn mask_from_cards(cards: &[u8]) -> CardMask {
    cards.iter().fold(0u64, |mask, &card| mask | card_bit(card))
}

pub fn card_name(card: u8) -> String {
    let rank = RANK_NAMES[card_rank(card) as usize];
    let suit = SUIT_NAMES[card_suit(card) as usize];
    format!("{rank}{suit}")
}

pub fn mask_to_card_names(mask: CardMask) -> Vec<String> {
    cards_from_mask(mask).into_iter().map(card_name).collect()
}

pub fn format_card_mask(mask: CardMask) -> String {
    let names = mask_to_card_names(mask);
    if names.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", names.join(" "))
    }
}
