use big2_rust::action::{MoveKind, classify_five_cards, classify_five_cards_with_rules};
use big2_rust::card::{
    card_bit, card_name, card_rank, card_suit, cards_from_mask, format_card_mask, mask_from_cards,
    mask_to_card_names,
};
use big2_rust::{ActionCatalog, RulesConfig};

fn move_for(catalog: &ActionCatalog, cards: &[u8]) -> u32 {
    catalog.move_id_for_mask(mask_from_cards(cards)).unwrap()
}

#[test]
fn card_ids_map_to_rank_suit_and_names() {
    assert_eq!(card_rank(0), 0);
    assert_eq!(card_suit(0), 0);
    assert_eq!(card_name(0), "3D");

    assert_eq!(card_rank(1), 0);
    assert_eq!(card_suit(1), 1);
    assert_eq!(card_name(1), "3C");

    assert_eq!(card_rank(48), 12);
    assert_eq!(card_suit(48), 0);
    assert_eq!(card_name(48), "2D");

    assert_eq!(card_rank(51), 12);
    assert_eq!(card_suit(51), 3);
    assert_eq!(card_name(51), "2S");
}

#[test]
fn masks_round_trip_to_sorted_cards() {
    let mask = card_bit(51) | card_bit(0) | card_bit(13);

    assert_eq!(cards_from_mask(mask), vec![0, 13, 51]);
    assert_eq!(mask_from_cards(&[51, 0, 13]), mask);
    assert_eq!(mask_to_card_names(mask), vec!["3D", "6C", "2S"]);
    assert_eq!(format_card_mask(mask), "[3D 6C 2S]");
    assert_eq!(format_card_mask(0), "[]");
}

#[test]
fn catalog_contains_pass_small_moves_and_no_invalid_small_sets() {
    let catalog = ActionCatalog::new();
    let pass = catalog.get(0).unwrap();
    let single = catalog.get(move_for(&catalog, &[5])).unwrap();
    let pair = catalog.get(move_for(&catalog, &[0, 1])).unwrap();
    let triple = catalog.get(move_for(&catalog, &[48, 49, 50])).unwrap();

    assert_eq!(pass.kind, MoveKind::Pass);
    assert_eq!(pass.num_cards, 0);
    assert_eq!(single.kind, MoveKind::Single);
    assert_eq!(single.primary_rank, 1);
    assert_eq!(single.high_suit, 1);
    assert_eq!(pair.kind, MoveKind::Pair);
    assert_eq!(pair.primary_rank, 0);
    assert_eq!(pair.high_suit, 1);
    assert_eq!(triple.kind, MoveKind::Triple);
    assert_eq!(triple.primary_rank, 12);

    assert_eq!(catalog.move_id_for_mask(mask_from_cards(&[0, 4])), None);
    assert_eq!(catalog.move_id_for_mask(mask_from_cards(&[0, 1, 4])), None);
}

#[test]
fn classify_five_card_categories_and_rejects_junk() {
    let straight = classify_five_cards(&[0, 5, 8, 13, 16]).unwrap();
    let flush = classify_five_cards(&[0, 4, 8, 16, 20]).unwrap();
    let full_house = classify_five_cards(&[0, 1, 2, 4, 5]).unwrap();
    let four_kind = classify_five_cards(&[0, 1, 2, 3, 4]).unwrap();
    let straight_flush = classify_five_cards(&[0, 4, 8, 12, 16]).unwrap();

    assert_eq!(straight.kind, MoveKind::Straight);
    assert_eq!(straight.primary_rank, 4);
    assert_eq!(flush.kind, MoveKind::Flush);
    assert_eq!(flush.ranks_desc, [5, 4, 2, 1, 0]);
    assert_eq!(full_house.kind, MoveKind::FullHouse);
    assert_eq!(full_house.primary_rank, 0);
    assert_eq!(full_house.secondary_rank, 1);
    assert_eq!(four_kind.kind, MoveKind::FourOfKind);
    assert_eq!(four_kind.primary_rank, 0);
    assert_eq!(straight_flush.kind, MoveKind::StraightFlush);
    assert_eq!(straight_flush.primary_rank, 4);

    assert_eq!(classify_five_cards(&[0, 4, 9, 15, 22]), None);
}

#[test]
fn twos_are_not_allowed_in_straights() {
    assert_eq!(classify_five_cards(&[32, 37, 42, 47, 48]), None);
    assert_eq!(classify_five_cards(&[35, 38, 43, 44, 51]), None);
}

#[test]
fn rules_config_can_allow_twos_and_wheel_straights() {
    let default_rules = RulesConfig::default();
    let permissive_rules = RulesConfig {
        allow_two_in_straight: true,
        allow_wheel_straight: true,
        ..RulesConfig::default()
    };

    assert_eq!(
        classify_five_cards_with_rules(&[32, 37, 42, 47, 49], default_rules),
        None
    );
    let ten_to_two =
        classify_five_cards_with_rules(&[32, 37, 42, 47, 49], permissive_rules).unwrap();
    let wheel = classify_five_cards_with_rules(&[0, 5, 10, 47, 49], permissive_rules).unwrap();

    assert_eq!(ten_to_two.kind, MoveKind::Straight);
    assert_eq!(ten_to_two.primary_rank, 12);
    assert_eq!(wheel.kind, MoveKind::Straight);
    assert_eq!(wheel.primary_rank, 2);
}

#[test]
fn catalog_respects_straight_rules_at_construction_time() {
    let default_catalog = ActionCatalog::new();
    let permissive_catalog = ActionCatalog::new_with_rules(RulesConfig {
        allow_two_in_straight: true,
        allow_wheel_straight: true,
        ..RulesConfig::default()
    });
    let ten_to_two_mask = mask_from_cards(&[32, 37, 42, 47, 49]);
    let wheel_mask = mask_from_cards(&[0, 5, 10, 47, 49]);

    assert_eq!(default_catalog.move_id_for_mask(ten_to_two_mask), None);
    assert_eq!(default_catalog.move_id_for_mask(wheel_mask), None);
    assert!(
        permissive_catalog
            .move_id_for_mask(ten_to_two_mask)
            .is_some()
    );
    assert!(permissive_catalog.move_id_for_mask(wheel_mask).is_some());
}

#[test]
fn catalog_lookup_for_five_card_moves_matches_classification() {
    let catalog = ActionCatalog::new();
    let straight_flush_mask = mask_from_cards(&[0, 4, 8, 12, 16]);
    let junk_mask = mask_from_cards(&[0, 4, 9, 15, 22]);

    let move_id = catalog
        .five_card_move_id_for_mask(straight_flush_mask)
        .unwrap();
    assert_eq!(catalog.get(move_id).unwrap().kind, MoveKind::StraightFlush);
    assert_eq!(catalog.five_card_move_id_for_mask(junk_mask), None);
    assert_eq!(catalog.move_id_for_mask(junk_mask), None);
}

#[test]
fn beats_compares_singles_pairs_and_triples_by_rank_then_suit_when_relevant() {
    let catalog = ActionCatalog::new();

    assert!(catalog.beats(move_for(&catalog, &[1]), move_for(&catalog, &[0])));
    assert!(catalog.beats(move_for(&catalog, &[4]), move_for(&catalog, &[3])));
    assert!(!catalog.beats(move_for(&catalog, &[0]), move_for(&catalog, &[1])));

    assert!(catalog.beats(move_for(&catalog, &[0, 2]), move_for(&catalog, &[0, 1])));
    assert!(catalog.beats(move_for(&catalog, &[4, 5]), move_for(&catalog, &[2, 3])));
    assert!(!catalog.beats(move_for(&catalog, &[0, 1]), move_for(&catalog, &[0, 2])));

    assert!(catalog.beats(
        move_for(&catalog, &[4, 5, 6]),
        move_for(&catalog, &[0, 1, 2])
    ));
    assert!(!catalog.beats(
        move_for(&catalog, &[0, 1, 2]),
        move_for(&catalog, &[4, 5, 6])
    ));
}

#[test]
fn beats_rejects_pass_unknown_and_different_sized_moves() {
    let catalog = ActionCatalog::new();

    assert!(!catalog.beats(0, move_for(&catalog, &[0])));
    assert!(!catalog.beats(move_for(&catalog, &[0]), 0));
    assert!(!catalog.beats(move_for(&catalog, &[0, 1]), move_for(&catalog, &[0])));
    assert!(!catalog.beats(999_999, move_for(&catalog, &[0])));
    assert!(!catalog.beats(move_for(&catalog, &[0]), 999_999));
}

#[test]
fn five_card_beats_allow_higher_category_or_higher_same_category() {
    let catalog = ActionCatalog::new();
    let low_straight = move_for(&catalog, &[0, 5, 8, 12, 16]);
    let high_straight = move_for(&catalog, &[4, 9, 12, 16, 20]);
    let flush = move_for(&catalog, &[0, 4, 8, 16, 20]);
    let full_house = move_for(&catalog, &[0, 1, 2, 4, 5]);
    let four_kind = move_for(&catalog, &[0, 1, 2, 3, 4]);
    let straight_flush = move_for(&catalog, &[1, 5, 9, 13, 17]);

    assert!(catalog.beats(high_straight, low_straight));
    assert!(!catalog.beats(low_straight, high_straight));
    assert!(catalog.beats(flush, high_straight));
    assert!(catalog.beats(full_house, flush));
    assert!(catalog.beats(four_kind, full_house));
    assert!(catalog.beats(straight_flush, four_kind));
}

#[test]
fn straight_flush_comparison_prioritizes_rank_before_suit() {
    let catalog = ActionCatalog::new();
    let low_spade_straight_flush = move_for(&catalog, &[3, 7, 11, 15, 19]);
    let high_diamond_straight_flush = move_for(&catalog, &[4, 8, 12, 16, 20]);

    assert!(catalog.beats(high_diamond_straight_flush, low_spade_straight_flush));
    assert!(!catalog.beats(low_spade_straight_flush, high_diamond_straight_flush));
}

#[test]
fn move_formatting_includes_kind_and_cards() {
    let catalog = ActionCatalog::new();
    let pass = catalog.get(0).unwrap();
    let pair = catalog.get(move_for(&catalog, &[0, 1])).unwrap();
    let straight_flush = catalog.get(move_for(&catalog, &[0, 4, 8, 12, 16])).unwrap();

    assert_eq!(pass.display(), "Pass");
    assert_eq!(pair.display(), "Pair [3D 3C]");
    assert_eq!(straight_flush.display(), "Straight flush [3D 4D 5D 6D 7D]");
    assert_eq!(
        catalog.format_move(pair.id).as_deref(),
        Some("Pair [3D 3C]")
    );
}
