"""Poker hand evaluator: given hole cards + board, returns PokerStars-style description."""

from itertools import combinations
from collections import Counter

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
RANK_NAMES = {2:'Twos',3:'Threes',4:'Fours',5:'Fives',6:'Sixes',7:'Sevens',
              8:'Eights',9:'Nines',10:'Tens',11:'Jacks',12:'Queens',13:'Kings',14:'Aces'}
RANK_SING = {2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',
             8:'Eight',9:'Nine',10:'Ten',11:'Jack',12:'Queen',13:'King',14:'Ace'}


def _parse_card(c: str):
    """Parse 'Ah' → (14, 'h')"""
    rank_str = c[:-1].upper()
    suit = c[-1].lower()
    return RANK_MAP.get(rank_str, 0), suit


def _eval5(cards):
    """Evaluate a 5-card hand. Returns (strength_tuple, description_str)."""
    ranks = sorted([r for r, s in cards], reverse=True)
    suits = [s for r, s in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    flush = len(set(suits)) == 1

    unique_ranks = sorted(rank_counts.keys(), reverse=True)
    # Straight detection (including A-2-3-4-5)
    straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:
            straight = True
            straight_high = 5  # wheel

    # Hand type
    if straight and flush:
        if straight_high == 14:
            return (9, 14), "a royal flush"
        return (8, straight_high), f"a straight flush, {RANK_SING[straight_high]} high"

    if counts[0] == 4:
        quad_rank = max(r for r, c in rank_counts.items() if c == 4)
        return (7, quad_rank), f"four of a kind, {RANK_NAMES[quad_rank]}"

    if counts[0] == 3 and counts[1] == 2:
        trip_rank = max(r for r, c in rank_counts.items() if c == 3)
        pair_rank = max(r for r, c in rank_counts.items() if c == 2)
        return (6, trip_rank, pair_rank), f"a full house, {RANK_NAMES[trip_rank]} full of {RANK_NAMES[pair_rank]}"

    if flush:
        return (5, *ranks), f"a flush, {RANK_SING[ranks[0]]} high"

    if straight:
        return (4, straight_high), f"a straight, {RANK_SING[straight_high]} high"

    if counts[0] == 3:
        trip_rank = max(r for r, c in rank_counts.items() if c == 3)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (3, trip_rank, *kickers), f"three of a kind, {RANK_NAMES[trip_rank]}"

    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = max(r for r, c in rank_counts.items() if c == 1)
        return (2, pairs[0], pairs[1], kicker), f"two pair, {RANK_NAMES[pairs[0]]} and {RANK_NAMES[pairs[1]]}"

    if counts[0] == 2:
        pair_rank = max(r for r, c in rank_counts.items() if c == 2)
        return (1, pair_rank), f"a pair of {RANK_NAMES[pair_rank]}"

    return (0, *ranks), f"high card {RANK_SING[ranks[0]]}"


def best_hand_desc(hole_cards: list, board_cards: list) -> str:
    """
    Given hole cards ['Ad', 'As'] and board ['8h', '2c', '6c', '8c', '9s'],
    return best hand description like 'two pair, Aces and Eights'.
    """
    try:
        all_cards = [_parse_card(c) for c in hole_cards + board_cards]
        all_cards = [(r, s) for r, s in all_cards if r > 0]
        if len(all_cards) < 5:
            return ''
        best = None
        for combo in combinations(all_cards, 5):
            result = _eval5(list(combo))
            if best is None or result[0] > best[0]:
                best = result
        return best[1] if best else ''
    except Exception:
        return ''
