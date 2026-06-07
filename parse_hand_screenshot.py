"""
Main parser: WPT Global hand history screenshot → PokerStars format.
Uses RapidOCR + image analysis. Amounts are in BB; converts to USD.
Usage: python parse_hand_screenshot.py <image_path> [--bb 0.10]
"""

import sys
import cv2
import re
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from bb_calibration import get_bb_value as _get_bb_value
except ImportError:
    _get_bb_value = None

from rapidocr_onnxruntime import RapidOCR
from src.detectors import detect_suit_hsv, normalize_rank, VALID_RANKS
from src.hh_writer_ps import format_hand_history

_ocr_instance = None

def _ocr():
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = RapidOCR()
    return _ocr_instance


# Portuguese → English
PT_TO_EN = {
    'desistir': 'folds',
    'passar': 'checks',
    'aposta': 'bets',
    'aumentar': 'raises',
    'pagar': 'calls',
    'fazercall': 'calls',
    'fazer': 'calls',
}

POSITIONS = {'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'HJ', 'CO', 'BTN', 'SB', 'BB', 'STR'}

# Portuguese action words that must NOT be treated as player names
PT_ACTION_WORDS = {
    'desistir', 'passar', 'aposta', 'aumentar', 'pagar', 'fazer', 'fazercall',
    'all-in', 'all-ln', 'allin', 'ante', 'desistis', 'winner',
    'blinds', 'ante', 'pre-flop', 'flop', 'turn', 'river',
}

# Standard seat assignment for 7-8 player: BTN=4, SB=5, BB=6, UTG=7, MP=1, HJ=2, CO=3
POS_TO_SEAT_7 = {'BTN': 4, 'SB': 5, 'BB': 6, 'STR': 7, 'MP': 1, 'HJ': 2, 'CO': 3}
POS_TO_SEAT_8 = {'BTN': 4, 'SB': 5, 'BB': 6, 'UTG': 7, 'MP': 1, 'HJ': 2, 'CO': 3}


def full_ocr(img) -> list:
    """Run OCR on full image, return sorted list of {x,y,text,conf}."""
    result, _ = _ocr()(img)
    if not result:
        return []
    items = []
    for item in result:
        if not item or len(item) < 2:
            continue
        bbox = item[0]
        text = item[1]
        conf = item[2] if len(item) > 2 else 1.0
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        if not text or not str(text).strip():
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({
            'x': int(min(xs)), 'y': int(min(ys)),
            'x2': int(max(xs)), 'y2': int(max(ys)),
            'text': str(text).strip(), 'conf': float(conf)
        })
    return sorted(items, key=lambda i: i['y'])


def parse_bb(text: str) -> float:
    """Parse European-format BB amount: '3,50BB' → 3.5, '-0,50 BB' → -0.5, '1.530BB' → 1530"""
    t = str(text).strip()
    sign = -1.0 if t.startswith('-') else 1.0
    t = t.lstrip('+-').strip()
    t = re.sub(r'\s*BB?\s*$', '', t, flags=re.IGNORECASE).strip()
    # Remove European thousands separator (period before exactly 3 digits)
    t = re.sub(r'\.(?=\d{3}(?:[.,]|$))', '', t)
    # Replace European decimal comma with period
    t = t.replace(',', '.')
    try:
        return sign * float(t)
    except ValueError:
        return 0.0


def is_amount(text: str) -> bool:
    return bool(re.match(r'^[+-]?[\d,.]+\s*BB?$', text, re.IGNORECASE))


def is_player_name(text: str) -> bool:
    if not text or len(text) < 2:
        return False
    t_upper = text.upper()
    t_lower = text.lower()
    if t_upper in POSITIONS | {'WINNER', 'ANTE', 'BB', 'SB', 'STR', 'ALL-IN', 'ALL', 'OBB'}:
        return False
    if t_lower in PT_ACTION_WORDS:
        return False
    if any(t_lower.startswith(w) for w in PT_ACTION_WORDS if len(w) > 3):
        return False
    if is_amount(text):
        return False
    if re.match(r'^\d{4}[-/]\d{2}', text):  # timestamp
        return False
    if t_upper.startswith('HAND') or t_upper.startswith('POTE') or t_upper.startswith('PRE'):
        return False
    # Reject OCR artifacts starting with / or + (e.g. /TG+1, /OBB)
    if text.startswith('/') or text.startswith('+'):
        return False
    # Must contain at least one letter/Chinese character, or be a pure numeric name (≥3 digits)
    if not re.search(r'[a-zA-Z一-鿿]', text):
        # Allow all-digit strings (e.g. player name "4480") with ≥3 digits
        if re.match(r'^\d{3,}$', text):
            return True
        return False
    return True


def items_in_band(items, x_min, x_max, y_min, y_max):
    return [i for i in items if x_min <= i['x'] < x_max and y_min <= i['y'] <= y_max]


# ─── Main Parser ────────────────────────────────────────────────────────────

def parse_hand(image_path: str, bb_value: float = None, debug: bool = False) -> str:
    if bb_value is None:
        if _get_bb_value is not None:
            bb_value = _get_bb_value(image_path)
        else:
            bb_value = 0.10
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]

    ocr = full_ocr(img)

    # Detect column boundaries from header label x positions (handles narrow layouts)
    C = _detect_column_boundaries(ocr, w)
    # C[0]=BLINDS, C[1]=PREFLOP, C[2]=FLOP, C[3]=TURN, C[4]=RIVER
    col_w = C[1][0] - C[0][0]  # approximate col width for debug

    if debug:
        print(f"[DEBUG] img={w}x{h}, col_w={col_w:.0f}")

    # ── Hand ID ──────────────────────────────────────────────────────
    hand_id = _extract_hand_id(ocr)

    # ── Timestamp ────────────────────────────────────────────────────
    timestamp = _extract_timestamp(ocr)

    # ── Find action table header ──────────────────────────────────────
    header_y = _find_header_y(ocr)
    pot_row_y = header_y + 35
    action_start_y = pot_row_y + 35  # lower threshold to catch first player row

    if debug:
        print(f"[DEBUG] hand_id={hand_id}, ts={timestamp}, header_y={header_y}")

    # ── Street pot amounts (row after column headers) ─────────────────
    pots_bb = _extract_pot_row(ocr, pot_row_y, action_start_y)

    # ── Player data from BLINDS column ───────────────────────────────
    blinds_col = _parse_blinds_column(ocr, C[0][0], C[0][1], action_start_y)

    # ── Detect results section: try RIVER first, fall back to TURN/FLOP ──
    # When a hand ends before the river, results appear in the last active column.
    act_y = header_y + 50
    result_col_idx = 4   # default: RIVER
    result_start_y = _find_result_start(ocr, C[4][0], w, action_start_y)
    if result_start_y == 99999:
        for ci in [3, 2]:   # try TURN then FLOP
            rsy = _find_result_start(ocr, C[ci][0], C[ci][1], action_start_y)
            if rsy < 99999:
                result_col_idx = ci
                result_start_y = rsy
                break

    # ── Parse each street ─────────────────────────────────────────────
    # Cut each column at result_start_y if that column holds the results.
    _cut = lambda ci, default_ymax: result_start_y if ci == result_col_idx else default_ymax
    preflop_acts = _parse_street(ocr, C[1][0], C[1][1], act_y, _cut(1, h), bb_value)
    flop_acts    = _parse_street(ocr, C[2][0], C[2][1], act_y, _cut(2, h), bb_value)
    turn_acts    = _parse_street(ocr, C[3][0], C[3][1], act_y, _cut(3, h), bb_value)
    river_acts   = _parse_street(ocr, C[4][0], C[4][1], act_y, _cut(4, result_start_y), bb_value)

    # ── Recover missing first-player names via focused OCR (PRE-FLOP only) ──
    # Only run for PRE-FLOP column to avoid excessive OCR calls.
    # _assign_anon_names will propagate the name to other streets via position.
    extra = _recover_first_player_names(img, ocr, [C[1]], act_y)
    if extra:
        ocr = ocr + extra
        # Re-parse streets with recovered names
        preflop_acts = _parse_street(ocr, C[1][0], C[1][1], act_y, _cut(1, h), bb_value)
        flop_acts    = _parse_street(ocr, C[2][0], C[2][1], act_y, _cut(2, h), bb_value)
        turn_acts    = _parse_street(ocr, C[3][0], C[3][1], act_y, _cut(3, h), bb_value)
        river_acts   = _parse_street(ocr, C[4][0], C[4][1], act_y, _cut(4, result_start_y), bb_value)

    # ── Parse results section (first pass, no position map yet) ──────
    # Use the detected result column (may be RIVER, TURN, or FLOP)
    res_x0 = C[result_col_idx][0]
    res_x1 = w if result_col_idx == 4 else C[result_col_idx][1]
    result_entries = _parse_results(ocr, res_x0, res_x1, result_start_y, bb_value=bb_value)

    if debug:
        print(f"[DEBUG] preflop={len(preflop_acts)}, flop={len(flop_acts)}, "
              f"turn={len(turn_acts)}, river={len(river_acts)}")
        print(f"[DEBUG] results={result_entries}")

    # ── Board cards ───────────────────────────────────────────────────
    board = _detect_board(img, w, h)
    if debug:
        print(f"[DEBUG] board={board}")

    # ── Build player map: {name: {position, net_bb}} ──────────────────
    player_map = _build_player_map(preflop_acts, flop_acts, turn_acts, river_acts,
                                    result_entries, blinds_col)

    # ── Fallback: all-anonymous table (use position labels as player names) ──
    # When no real names are found (e.g. all players show only UTG/BTN badges),
    # build a position-keyed player_map so _assign_anon_names can resolve actions.
    if not player_map:
        seen_positions = {}
        for acts in [preflop_acts, flop_acts, turn_acts, river_acts]:
            for a in acts:
                pos = a.get('position')
                if pos and pos not in seen_positions:
                    seen_positions[pos] = {'pos': pos, 'net_bb': None}
        if seen_positions:
            player_map = seen_positions
            if debug:
                print(f"[DEBUG] all-anon table, using positions as names: {list(player_map)}")

    # ── Assign names to anonymous actions using position lookup ────────
    _assign_anon_names([preflop_acts, flop_acts, turn_acts, river_acts], player_map)
    # Rebuild player_map now that anonymous actions have names
    player_map = _build_player_map(preflop_acts, flop_acts, turn_acts, river_acts,
                                    result_entries, blinds_col)

    # ── Deduplicate OCR artifact players ────────────────────────────────
    # If >8 players detected, some are likely OCR variants of the same person.
    # Remove players with no net_bb (not in results) that are near-duplicates
    # of a player who HAS net_bb.
    if len(player_map) > 8:
        from difflib import SequenceMatcher
        names_with_result = [n for n, d in player_map.items() if d.get('net_bb') is not None]
        names_without_result = [n for n, d in player_map.items() if d.get('net_bb') is None]
        to_remove = set()
        for artifact in names_without_result:
            for real in names_with_result:
                sim = SequenceMatcher(None, artifact, real).ratio()
                if sim > 0.7 and artifact != real:
                    to_remove.add(artifact)
                    break
        for n in to_remove:
            del player_map[n]
            # Replace artifact name in all action lists
            for acts in [preflop_acts, flop_acts, turn_acts, river_acts]:
                for a in acts:
                    if a.get('name') == n:
                        # Find the real player it was confused with
                        best_real = max(names_with_result,
                                        key=lambda r: SequenceMatcher(None, n, r).ratio())
                        a['name'] = best_real

    # ── Re-parse results with position→name map (recover anonymous winner) ──
    if any(r['name'] is None or r.get('net_bb', 0) == 0 for r in result_entries) or \
       not any(r.get('net_bb', 0) > 0 for r in result_entries):
        pos_to_name = {d.get('pos'): n for n, d in player_map.items() if d.get('pos')}
        # For all-anon tables (where names ARE positions), add self-mappings for all positions
        # so players who folded without captured actions (e.g. HJ) are still recovered
        is_anon_table = all(n in POSITIONS for n in player_map)
        if is_anon_table:
            for p in POSITIONS:
                if p not in pos_to_name:
                    pos_to_name[p] = p
        result_entries_v2 = _parse_results(ocr, res_x0, res_x1, result_start_y,
                                            pos_name_map=pos_to_name, bb_value=bb_value)
        if len(result_entries_v2) > len(result_entries):
            result_entries = result_entries_v2
            # Rebuild player_map once more with complete results
            player_map = _build_player_map(preflop_acts, flop_acts, turn_acts, river_acts,
                                            result_entries, blinds_col)

    # ── Remove OCR artifact players not grounded in stacks or results ──
    # Players like "NEPTIN" that appear once as noise in the action area
    # but have no stack data and no result entry are dropped here.
    _end_stacks_early = _extract_end_stacks(ocr, header_y)
    _is_anon = all(n in POSITIONS for n in player_map)
    if _end_stacks_early and not _is_anon and len(player_map) > len(_end_stacks_early):
        from difflib import SequenceMatcher
        anchor = (list(_end_stacks_early.keys()) +
                  [r['name'] for r in result_entries if r.get('name')])
        for pname in list(player_map.keys()):
            if pname in POSITIONS or player_map[pname].get('net_bb') is not None:
                continue
            if max((SequenceMatcher(None, pname, a).ratio() for a in anchor), default=0) < 0.5:
                del player_map[pname]
                for acts in [preflop_acts, flop_acts, turn_acts, river_acts]:
                    for a in acts:
                        if a.get('name') == pname:
                            a['name'] = None

    if debug:
        print(f"[DEBUG] players: {list(player_map.keys())}")

    # ── Determine seats ───────────────────────────────────────────────
    seats = _assign_seats(player_map, player_y=blinds_col.get('_player_y'))

    # ── Special players ───────────────────────────────────────────────
    sb_player  = next((n for n, d in player_map.items() if d.get('pos') == 'SB'), None)
    bb_player  = next((n for n, d in player_map.items() if d.get('pos') == 'BB'), None)
    str_player = next((n for n, d in player_map.items() if d.get('pos') == 'STR'), None)
    btn_player = next((n for n, d in player_map.items() if d.get('pos') == 'BTN'), None)

    # If positions not found from actions, try from blinds column
    if not sb_player or not bb_player or not str_player:
        for bp in blinds_col.get('_blind_players', []):
            if bp['type'] == 'sb' and not sb_player:
                sb_player = bp['name']
                if sb_player in player_map:
                    player_map[sb_player]['pos'] = 'SB'
            elif bp['type'] == 'bb' and not bb_player:
                bb_player = bp['name']
                if bb_player in player_map:
                    player_map[bb_player]['pos'] = 'BB'
            elif bp['type'] == 'str' and not str_player:
                str_player = bp['name']
                if str_player in player_map:
                    player_map[str_player]['pos'] = 'STR'

    # Get blind amounts from blinds column
    sb_bb  = blinds_col.get('sb_bb', 0.5)
    bb_bb  = blinds_col.get('bb_bb', 1.0)
    str_bb = blinds_col.get('str_bb', 2.0)
    # Calculate per-player ante from total ante / total players
    n_players = len(player_map)
    total_ante_bb = blinds_col.get('_total_ante_bb')
    if total_ante_bb and n_players > 0:
        ante_bb = round(total_ante_bb / n_players, 4)
    else:
        ante_bb = blinds_col.get('ante_per_player_bb', 0.5)

    # ── Get pot total from image (shown as "Pote Total: X BB" or "$ X.XX") ──
    pote_total_bb = _extract_pote_total(ocr, bb_value=bb_value)

    # ── Get end-of-hand stacks from table view ─────────────────────────
    end_stacks_bb = _extract_end_stacks(ocr, header_y)
    # For all-anonymous tables, position labels appear instead of real names
    if not end_stacks_bb and player_map and all(k in POSITIONS for k in player_map):
        end_stacks_bb = _extract_anon_stacks(ocr, header_y, bb_value=bb_value)
    if debug:
        print(f"[DEBUG] pote_total={pote_total_bb}, end_stacks={end_stacks_bb}")

    # ── Remove OCR artifact players not found in stacks or results ────────
    # Players like "NEPTIN" that appear once as OCR noise in the action area
    # but have no stack data and no result entry are dropped here.
    is_anon = all(n in POSITIONS for n in player_map)
    if end_stacks_bb and not is_anon:
        from difflib import SequenceMatcher
        anchor_names = list(end_stacks_bb.keys()) + [r['name'] for r in result_entries if r.get('name')]
        to_drop = set()
        for pname, pdata in list(player_map.items()):
            if pname in POSITIONS:
                continue
            if pdata.get('net_bb') is not None:
                continue  # has a result, keep
            best_sim = max((SequenceMatcher(None, pname, an).ratio() for an in anchor_names), default=0)
            if best_sim < 0.5:
                to_drop.add(pname)
        for pname in to_drop:
            del player_map[pname]
            for acts in [preflop_acts, flop_acts, turn_acts, river_acts]:
                for a in acts:
                    if a.get('name') == pname:
                        a['name'] = None

    # ── Calculate initial stacks (from net results) ───────────────────
    initial_stacks = _calc_initial_stacks(
        player_map, result_entries, ante_bb, sb_bb, bb_bb, str_bb,
        sb_player, bb_player, str_player, bb_value, pote_total_bb, end_stacks_bb,
        all_player_names=list(player_map.keys())
    )

    # ── Fix all-in call amounts using stack tracking ──────────────────
    initial_stacks_bb_vals = {k: v / bb_value for k, v in initial_stacks.items()}
    _fix_allin_call_amounts(
        preflop_acts, flop_acts, turn_acts, river_acts,
        initial_stacks_bb_vals, ante_bb, sb_bb, bb_bb, str_bb,
        sb_player, bb_player, str_player
    )

    # ── Convert actions to USD ────────────────────────────────────────
    def to_usd(x):
        return round(x * bb_value, 2)

    stakes = {'sb': to_usd(sb_bb), 'bb': to_usd(bb_bb)}
    # Preflop raises are relative to straddle (or BB if no straddle)
    preflop_init_bet_bb = str_bb if str_player else bb_bb
    preflop_usd = _convert_actions(preflop_acts, bb_value, initial_bet_bb=preflop_init_bet_bb)
    flop_usd    = _convert_actions(flop_acts, bb_value)
    turn_usd    = _convert_actions(turn_acts, bb_value)
    river_usd   = _convert_actions(river_acts, bb_value)

    # ── Showdown and winners ──────────────────────────────────────────
    # Extract hole cards from image for showdown players
    showdown_cards = _extract_showdown_cards(img, w, h, ocr, result_entries, result_start_y)
    showdown = _build_showdown(result_entries, showdown_cards, bb_value, board=board)
    winners = _build_winners(result_entries, showdown, bb_value)

    # ── Detect hero (player whose cards we see = the recording player) ──
    # Heuristic: loser who went all-in, or first player with a positive
    # all-in flag in river actions, or just the first showdown loser.
    hero = None
    hero_cards = []
    river_allins = {a['name'] for a in river_usd if a.get('allin')}
    for r in result_entries:
        if r['name'] in showdown_cards and r['net_bb'] < 0:
            if r['name'] in river_allins or not hero:
                hero = r['name']
                hero_cards = showdown_cards[r['name']]
                if r['name'] in river_allins:
                    break

    # ── Total pot ─────────────────────────────────────────────────────
    # Use pote_total_bb directly if available (most accurate source)
    if pote_total_bb and pote_total_bb > 0:
        total_pot = to_usd(pote_total_bb)
    else:
        total_losses_bb = sum(abs(r['net_bb']) for r in result_entries if r['net_bb'] < 0)
        winner_net_bb   = sum(r['net_bb'] for r in result_entries if r['net_bb'] > 0)
        winner_name = next((r['name'] for r in result_entries if r['net_bb'] > 0), None)
        winner_contrib = 0
        if winner_name:
            winner_contrib = _player_total_contribution(
                winner_name, preflop_acts + flop_acts + turn_acts + river_acts,
                ante_bb, sb_bb, bb_bb, str_bb, sb_player, bb_player, str_player)
        total_pot_bb = winner_net_bb + winner_contrib
        if total_pot_bb <= 0:
            total_pot_bb = total_losses_bb
        total_pot = to_usd(total_pot_bb)

    if winners:
        share = round(total_pot / len(winners), 2)
        for w in winners:
            w['amount'] = share

    # ── Seats list ────────────────────────────────────────────────────
    players_list = []
    for name, seat in sorted(seats.items(), key=lambda x: x[1]):
        players_list.append({'seat': seat, 'name': name, 'stack': initial_stacks.get(name, 0)})

    button_seat = seats.get(btn_player, 1) if btn_player else 1

    # ── Summary seats ─────────────────────────────────────────────────
    summary_seats = _build_summary_seats(
        seats, player_map, preflop_acts, flop_acts, turn_acts, river_acts,
        showdown, btn_player, sb_player, bb_player, initial_stacks, bb_value,
        result_entries=result_entries, pote_total_bb=pote_total_bb
    )

    # ── Blinds list ───────────────────────────────────────────────────
    blinds_list = []
    if sb_player:
        blinds_list.append({'name': sb_player, 'type': 'small blind', 'amount': to_usd(sb_bb)})
    if bb_player:
        blinds_list.append({'name': bb_player, 'type': 'big blind', 'amount': to_usd(bb_bb)})

    # ── Board sections ────────────────────────────────────────────────
    # Use partial boards when OCR can't detect all cards (e.g. Q obscured by text overlay)
    # but we have flop actions — show whatever cards we have
    if len(board) >= 3:
        flop_cards = board[:3]
    elif len(board) > 0 and flop_usd:
        flop_cards = board[:]  # partial flop (e.g. 1-2 cards when Q undetected)
    else:
        flop_cards = []
    turn_card   = board[3] if len(board) >= 4 else None
    river_card  = board[4] if len(board) >= 5 else None

    hand_data = {
        'hand_id': hand_id,
        'timestamp': timestamp,
        'stakes': stakes,
        'table_name': 'WPT Global',
        'max_players': 8,
        'button_seat': button_seat,
        'players': players_list,
        'ante': to_usd(ante_bb),
        'blinds': blinds_list,
        'straddle': {'name': str_player, 'amount': to_usd(str_bb)} if str_player else None,
        'hero': hero,
        'hole_cards': hero_cards,
        'preflop_actions': preflop_usd,
        'flop_cards': flop_cards,
        'flop_actions': flop_usd,
        'turn_card': turn_card,
        'turn_actions': turn_usd,
        'river_card': river_card,
        'river_actions': river_usd,
        'showdown': showdown,
        'winners': winners,
        'total_pot': total_pot,
        'rake': 0,
        'summary_seats': summary_seats,
    }

    return format_hand_history(hand_data)


# ── Extraction helpers ────────────────────────────────────────────────────

def _extract_hand_id(items) -> str:
    for item in items:
        if item['y'] > 80:
            break
        t = item['text']
        nums = re.findall(r'\d{10,}', t.replace(' ', ''))
        if nums:
            return nums[0]
    return '0'


def _extract_timestamp(items) -> str:
    for item in items:
        m = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})\s*(\d{2}:\d{2}:\d{2})', item['text'])
        if m:
            return f"{m.group(1)}/{m.group(2)}/{m.group(3)} {m.group(4)} ET"
    return '2026/01/01 00:00:00 ET'


def _find_header_y(items) -> int:
    for item in items:
        t = item['text'].upper().replace(' ', '').replace('&', '')
        if 'BLINDSANT' in t or ('BLINDS' in t and item['x'] < 200):
            return item['y']
        if 'PRE-FLOP' in t or 'PREFLOP' in t:
            return item['y']
    return int(items[0]['y'] + (items[-1]['y'] - items[0]['y']) * 0.3)  # fallback


def _detect_column_boundaries(items, w) -> list:
    """Detect column boundaries from header label x positions.

    Returns list of 5 (x_min, x_max) tuples for:
    C[0]=BLINDS, C[1]=PREFLOP, C[2]=FLOP, C[3]=TURN, C[4]=RIVER
    Falls back to w/5 equal widths if headers not found.
    """
    col_xs = [None] * 5
    for item in sorted(items, key=lambda i: i['y']):
        t_orig = item['text'].upper()
        t = t_orig.replace(' ', '').replace('&', '').replace('-', '')
        if col_xs[0] is None and ('BLINDSANTE' in t or (t_orig.startswith('BLINDS') and 'ANTE' in t_orig)):
            col_xs[0] = item['x']
        elif col_xs[1] is None and ('PREFLOP' in t):
            col_xs[1] = item['x']
        elif col_xs[2] is None and t_orig.strip() == 'FLOP':
            col_xs[2] = item['x']
        elif col_xs[3] is None and t_orig.strip() == 'TURN':
            col_xs[3] = item['x']
        elif col_xs[4] is None and t_orig.strip() == 'RIVER':
            col_xs[4] = item['x']
        if all(x is not None for x in col_xs):
            break

    if all(x is not None for x in col_xs):
        avg_spacing = (col_xs[4] - col_xs[0]) / 4
        if avg_spacing < 160:
            # Narrow layout: headers sit at left edge of each column;
            # use header x positions directly as column boundaries.
            xs = col_xs + [w]
            return [(xs[i], xs[i + 1]) for i in range(5)]

    col_w = w / 5
    return [(int(i * col_w), int((i + 1) * col_w)) for i in range(5)]


def _extract_pot_row(items, y_min, y_max) -> list:
    pot_items = [i for i in items if y_min <= i['y'] <= y_max and is_amount(i['text'])]
    return [parse_bb(i['text']) for i in sorted(pot_items, key=lambda i: i['x'])]


def _parse_blinds_column(items, x_min, x_max, y_start) -> dict:
    """Parse the BLINDS & ANTE column for ante/blind info."""
    col = sorted(items_in_band(items, x_min, x_max, y_start, 99999), key=lambda i: i['y'])
    result = {'ante_per_player_bb': 0.5, 'sb_bb': 0.5, 'bb_bb': 1.0, 'str_bb': 2.0,
              '_total_ante_bb': None, '_blind_players': [], '_player_y': {}}

    pending_type = None
    total_ante_bb = None
    blind_players = []  # players who posted blinds (SB, BB, STR)
    player_y = {}       # {player_name: first y-position in column}

    # Also track player+badge pairs by scanning for player→badge patterns
    # In the column: player name appears, then SB/BB/STR badge, then amount
    last_player_name = None

    for item in col:
        t = item['text']
        tl = t.lower()

        if tl == 'ante' or (tl.startswith('ante') and not is_amount(t)):
            pending_type = 'ante'
        elif t.upper() == 'SB':
            pending_type = 'sb'
            if last_player_name:
                blind_players.append({'name': last_player_name, 'type': 'sb'})
        elif t.upper() == 'BB' and pending_type not in ('ante',):
            pending_type = 'bb'
            if last_player_name:
                blind_players.append({'name': last_player_name, 'type': 'bb'})
        elif t.upper() == 'STR':
            pending_type = 'str'
            if last_player_name:
                blind_players.append({'name': last_player_name, 'type': 'str'})
        elif t.upper() == 'UTG' and pending_type == 'str':
            pass  # UTG appears with STR sometimes, ignore
        elif is_player_name(t):
            last_player_name = t
            if t not in player_y:
                player_y[t] = item['y']
        elif is_amount(t):
            val = abs(parse_bb(t))
            if pending_type == 'ante' and total_ante_bb is None:
                total_ante_bb = val
            elif pending_type == 'sb':
                result['sb_bb'] = val
            elif pending_type == 'bb':
                result['bb_bb'] = val
            elif pending_type == 'str':
                result['str_bb'] = val
            pending_type = None
            last_player_name = None  # reset after amount consumed

    result['_total_ante_bb'] = total_ante_bb
    result['_blind_players'] = blind_players
    result['_player_y'] = player_y
    # Note: ante_per_player_bb will be set after counting total players
    return result


def _find_result_start(items, x_min, x_max, y_min) -> int:
    """Find y where results start in RIVER column (BB or dollar format).
    Returns the y just before the first result player name/position label."""
    col = sorted(items_in_band(items, x_min, x_max, y_min, 99999), key=lambda i: i['y'])
    # Find first result amount (BB or dollar sign)
    first_amount_y = 99999
    for item in col:
        t = item['text']
        if (re.match(r'^[+-][\d,.]+\s*BB?$', t, re.IGNORECASE) or
                re.match(r'^[+-]\s*\$\s*[\d,.]+$', t)):
            first_amount_y = item['y']
            break
    if first_amount_y == 99999:
        return 99999
    # Find the earliest (lowest y) player name or position label within 150px
    # before the first amount — that's the start of the first result entry
    first_name_y = first_amount_y
    for item in [i for i in col if first_amount_y - 150 <= i['y'] < first_amount_y]:
        t = item['text']
        if t.upper() in POSITIONS or is_player_name(t):
            first_name_y = item['y']
            break  # take the lowest y (first encountered in ascending order)
    # Return 40px before the first name so action items (e.g. All-in) above it stay in actions
    return max(y_min, first_name_y - 40)


def _is_action_start(text: str) -> bool:
    """True if text is a Portuguese action word that can start a new cell."""
    tl = text.lower().strip()
    # All-in is a modifier (belongs to previous action), not a cell-starter
    if tl in ('all-in', 'all-ln', 'allin') or ('all' in tl and 'in' in tl and len(tl) <= 8):
        return False
    return any(tl.startswith(pt) for pt in PT_TO_EN if len(pt) > 3)


def _group_into_cells(col_items, gap=55) -> list:
    """
    Group OCR items into cells using player names and position-badge transitions.
    A new cell starts at:
      1. A player name (named player action)
      2. An action word that follows a position badge (anonymous player repeat action)
    Falls back to gap-based if no player names found.
    """
    if not col_items:
        return []

    has_names = any(is_player_name(i['text']) for i in col_items)

    if has_names:
        cells = []
        current = []
        last_was_badge = False

        for item in col_items:
            t = item['text']
            is_badge = t.upper() in POSITIONS
            is_allin = (t.lower().strip() in ('all-in', 'all-ln', 'allin') or
                        ('all' in t.lower() and 'in' in t.lower() and len(t) <= 8))

            if is_player_name(t) and current:
                cells.append(current)
                current = [item]
                last_was_badge = False
            elif last_was_badge and _is_action_start(t) and current:
                # Action after a position badge → new anonymous cell
                cells.append(current)
                current = [item]
                last_was_badge = False
            else:
                current.append(item)
                if is_badge:
                    last_was_badge = True
                elif not is_allin:
                    # Only non-allin non-badge items reset last_was_badge
                    last_was_badge = False

        if current:
            cells.append(current)
        return cells
    else:
        # Gap-based grouping
        cells = []
        current = [col_items[0]]
        for item in col_items[1:]:
            if item['y'] - current[-1]['y'] > gap:
                cells.append(current)
                current = [item]
            else:
                current.append(item)
        cells.append(current)
        return cells


def _parse_cell(cell, bb_value: float = 0.10) -> dict | None:
    """Parse one action cell → {name, action, amount_bb, position, allin}"""
    name = action = position = None
    amount_bb = None
    allin = False

    for item in sorted(cell, key=lambda i: i['y']):
        t = item['text']
        tl = t.lower().strip()

        if t.upper() in POSITIONS:
            position = t.upper()
            continue
        if tl == 'all-in' or tl == 'all-ln' or 'all' in tl and 'in' in tl:
            allin = True
            continue
        if is_amount(t) and not t.startswith('+'):
            amount_bb = abs(parse_bb(t))
            continue
        # Fallback: dollar-format amounts like '$4,04' or '$ 16,50' (some WPT display modes)
        if amount_bb is None and not t.startswith('+') and _is_dollar_amount(t):
            usd = _parse_dollar_amount(t)
            if usd is not None and usd > 0:
                amount_bb = usd / bb_value
            continue
        if is_player_name(t):
            if name is None:
                name = t
            continue
        # Action words
        for pt, en in PT_TO_EN.items():
            if tl.startswith(pt):
                action = en
                break

    if action is None:
        return None
    return {'name': name, 'action': action, 'amount_bb': amount_bb,
            'position': position, 'allin': allin}


def _parse_street(items, x_min, x_max, y_min, y_max, bb_value: float = 0.10) -> list:
    """Parse all actions in a street column."""
    col = sorted(items_in_band(items, x_min, x_max, y_min, y_max), key=lambda i: i['y'])
    cells = _group_into_cells(col, gap=55)
    acts = []
    for cell in cells:
        parsed = _parse_cell(cell, bb_value=bb_value)
        if parsed:
            acts.append(parsed)
    return acts


def _parse_results(items, x_min, x_max, y_min, pos_name_map=None, bb_value=0.10) -> list:
    """Parse result entries from RIVER column results section.
    pos_name_map: optional {position: player_name} for anonymous winner recovery.
    """
    col = sorted(items_in_band(items, x_min, x_max, y_min, 99999), key=lambda i: i['y'])
    cells = _group_into_cells(col, gap=70)
    results = []
    for cell in cells:
        name = pos = None
        net_bb = None
        unsigned_bb = None
        for item in sorted(cell, key=lambda i: i['y']):
            t = item['text']
            if t.upper() in POSITIONS:
                pos = t.upper()
            elif re.match(r'^[+-][\d,.]+\s*BB?$', t, re.IGNORECASE):
                net_bb = parse_bb(t)
            elif re.match(r'^\d[\d,.]+\s*BB?$', t, re.IGNORECASE):
                # Unsigned amount — minus sign likely dropped by OCR
                unsigned_bb = abs(parse_bb(t))
            elif re.match(r'^[+-]\s*\$\s*[\d,.]+$', t):
                # Dollar format: -$ 0,15 or -$0.15 or +$1.50
                sign = -1 if t.lstrip()[0] == '-' else 1
                num_str = re.sub(r'[^\d,.]', '', t)
                num_str = num_str.replace(',', '.')
                try:
                    net_bb = sign * float(num_str) / bb_value
                except ValueError:
                    pass
            elif re.match(r'^\$\s*[\d,.]+$', t):
                # Unsigned dollar format
                num_str = re.sub(r'[^\d,.]', '', t)
                num_str = num_str.replace(',', '.')
                try:
                    unsigned_bb = float(num_str) / bb_value
                except ValueError:
                    pass
            elif is_player_name(t):
                name = t
        # Use signed if available, else treat unsigned as negative (folders)
        if net_bb is None and unsigned_bb is not None:
            net_bb = -unsigned_bb
        # Recover anonymous winner via position lookup
        if name is None and pos is not None and pos_name_map and pos in pos_name_map:
            name = pos_name_map[pos]
        if name and net_bb is not None:
            results.append({'name': name, 'pos': pos, 'net_bb': net_bb})
    return results


def _recover_first_player_names(img, ocr_items, col_ranges, act_y) -> list:
    """
    For each action column, run focused OCR above the first detected action
    to recover player names that full-image OCR misses.
    Returns additional OCR items to inject into the main items list.
    """
    ocr_fn = _ocr()
    extra_items = []

    for col_x_min, col_x_max in col_ranges:
        # Find first item in this column at or after act_y
        col_items = sorted(
            [i for i in ocr_items if col_x_min <= i['x'] < col_x_max and i['y'] >= act_y],
            key=lambda i: i['y']
        )
        if not col_items:
            continue

        # Find the first ACTION WORD in this column (not just the first item, which could be an amount)
        first_action_y = None
        for ci in col_items:
            t = ci['text'].lower().strip()
            if any(t.startswith(pt) for pt in PT_ACTION_WORDS if len(pt) > 3):
                first_action_y = ci['y']
                break
        if first_action_y is None:
            first_action_y = col_items[0]['y']

        # Check if a player name already exists before the first action
        has_name = any(
            is_player_name(i['text']) for i in col_items if i['y'] < first_action_y
        )
        if has_name:
            continue

        # No player name before the first action → run focused OCR on that region
        y1 = max(0, act_y - 15)
        y2 = min(img.shape[0], first_action_y + 10)
        if y2 <= y1:
            continue
        crop = img[y1:y2, col_x_min:col_x_max]
        if crop.size < 50:
            continue

        big = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2), cv2.INTER_CUBIC)
        result, _ = ocr_fn(big)
        if not result:
            continue

        for item in result:
            if not item or len(item) < 2:
                continue
            bbox = item[0]
            text = str(item[1]).strip() if item[1] else ''
            if not text or not is_player_name(text):
                continue
            # Skip if this name already exists in the main items (avoid duplicates)
            if any(i['text'] == text for i in ocr_items):
                continue
            xs = [int(p[0]) // 2 + col_x_min for p in bbox]
            ys = [int(p[1]) // 2 + y1 for p in bbox]
            extra_items.append({
                'x': int(min(xs)), 'y': int(min(ys)),
                'x2': int(max(xs)), 'y2': int(max(ys)),
                'text': text, 'conf': 0.8
            })
            break  # Only take the first name found

    return extra_items


def _assign_anon_names(street_acts_list: list, player_map: dict):
    """
    Assign player names to anonymous action cells (name=None) using position matching.
    Modifies in-place. Removes cells that cannot be assigned a name.
    """
    pos_to_name = {}
    for name, data in player_map.items():
        if name and data.get('pos'):
            pos_to_name[data['pos']] = name

    for acts in street_acts_list:
        assigned = []
        for act in acts:
            if act['name'] is None:
                pos = act.get('position')
                if pos and pos in pos_to_name:
                    act['name'] = pos_to_name[pos]
                    assigned.append(act)
                # else: drop this anonymous action (can't identify player)
            else:
                assigned.append(act)
        acts[:] = assigned


def _build_player_map(preflop, flop, turn, river, results, blinds_col) -> dict:
    """Build {name: {pos, net_bb}} from all sources. Skips anonymous (name=None) entries."""
    pm = {}

    # From results (most complete list)
    for r in results:
        if r.get('name'):
            pm[r['name']] = {'pos': r.get('pos'), 'net_bb': r['net_bb']}

    # Fill in positions from named actions
    for acts in [preflop, flop, turn, river]:
        for a in acts:
            n = a.get('name')
            if not n:
                continue
            if n not in pm:
                pm[n] = {'pos': None, 'net_bb': None}
            if a.get('position') and pm[n].get('pos') is None:
                pm[n]['pos'] = a['position']

    return pm


def _assign_seats(player_map: dict, player_y: dict = None) -> dict:
    """
    Assign seat numbers based on position badges.
    Standard 7-player: BTN=4, SB=5, BB=6, STR/UTG=7, MP=1, HJ=2, CO=3
    """
    seat_base = {'BTN': 4, 'SB': 5, 'BB': 6, 'STR': 7, 'UTG': 7,
                 'UTG+1': 8, 'MP': 1, 'MP+1': 2, 'HJ': 2, 'CO': 3}

    seats = {}
    assigned = set()
    next_seat = 1

    # First pass: assign by known position
    for name, data in player_map.items():
        pos = data.get('pos')
        if pos and pos in seat_base:
            seat = seat_base[pos]
            # Handle conflicts: try up to 8 alternate seats, then overflow
            for _ in range(8):
                if seat not in assigned:
                    break
                seat = seat % 8 + 1
            else:
                # All 8 seats taken (extra OCR artifact player) - use overflow seat
                seat = 9
                while seat in assigned:
                    seat += 1
            seats[name] = seat
            assigned.add(seat)

    # Second pass: assign remaining players (overflow > 8 for extra OCR artifacts)
    for name in player_map:
        if name not in seats:
            while next_seat in assigned:
                next_seat += 1
                if next_seat > 99:  # safety guard
                    break
            seats[name] = next_seat
            assigned.add(next_seat)
            next_seat += 1

    return seats


def _player_total_contribution(name, all_acts, ante_bb, sb_bb, bb_bb, str_bb,
                                sb_player, bb_player, str_player) -> float:
    """Calculate total BB contribution for a player."""
    total = ante_bb  # every player pays ante
    pos_blind = 0
    if name == sb_player:
        pos_blind = sb_bb
    elif name == bb_player:
        pos_blind = bb_bb
    elif name == str_player:
        pos_blind = str_bb
    total += pos_blind

    # Track highest bet at each street (for raises)
    # Actually, just track how much they added with bets/calls/raises
    for act in all_acts:
        if act['name'] != name:
            continue
        a = act['action']
        amt = act.get('amount_bb') or 0
        if a in ('calls', 'bets'):
            total += amt
        elif a == 'raises':
            # amt_bb is the total raise target
            # They previously had some amount in - we can't easily recalculate here
            # Approximate: just track the total
            total += amt  # over-counts somewhat

    return total


def _extract_end_stacks(items, header_y) -> dict:
    """Extract end-of-hand player stacks from the poker table view (y < header_y).
    Uses x-proximity matching to correctly associate names with stacks.
    Handles truncated Chinese names (e.g. '淡淡的会...' → '淡淡的会顺顺')."""
    table_items = sorted([i for i in items if i['y'] < header_y - 20], key=lambda i: i['y'])
    stacks = {}

    for item in table_items:
        t = item['text']
        # Handle OCR errors: "OBB" → "0BB"
        if re.match(r'^[Oo0]BB?$', t):
            t = '0BB'
        if is_amount(t) and not t.startswith('+') and not t.startswith('-'):
            val = parse_bb(t)
            if val >= 0:
                # Find nearest player name nearby (same y ± 80px, similar x ± 200px)
                best_name = None
                best_dist = 9999
                for prev in table_items:
                    dy = abs(prev['y'] - item['y'])
                    dx = abs(prev['x'] - item['x'])
                    if dy <= 80 and dx <= 200 and is_player_name(prev['text']):
                        dist = dy + dx * 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_name = prev['text']
                if best_name:
                    stacks[best_name] = val
    return stacks


def _parse_dollar_amount(text: str) -> float | None:
    """Parse a dollar amount like '$ 38,20' or '$47.28' → float in USD, or None."""
    t = str(text).strip().lstrip('$').strip().replace(',', '.')
    try:
        v = float(t)
        return v if v >= 0 else None
    except ValueError:
        return None


def _is_dollar_amount(text: str) -> bool:
    """Match amounts in dollar format: '$ 38,20', '$47.28', '$ 9,15'."""
    return bool(re.match(r'^\$?\s*\d[\d,. ]*$', text.strip()))


def _extract_anon_stacks(items, header_y, bb_value: float = 0.10) -> dict:
    """Like _extract_end_stacks but for all-anonymous tables where the oval shows
    dollar amounts (e.g. '$ 38,20') next to position labels (e.g. 'BTN').
    Returns {position_label: bb_value} (amounts converted to BB)."""
    table_items = sorted([i for i in items if i['y'] < header_y - 20], key=lambda i: i['y'])
    stacks = {}
    for item in table_items:
        t = item['text'].strip()
        # Match BB format first, then dollar format
        if is_amount(t) and not t.startswith('+') and not t.startswith('-'):
            val_bb = parse_bb(t)
        elif _is_dollar_amount(t) and not t.startswith('+') and not t.startswith('-'):
            usd = _parse_dollar_amount(t)
            if usd is None or usd <= 0:
                continue
            val_bb = usd / bb_value  # convert USD to BB
        else:
            continue

        if val_bb > 0:
            best_pos = None
            best_dist = 9999
            for prev in table_items:
                dy = abs(prev['y'] - item['y'])
                dx = abs(prev['x'] - item['x'])
                if dy <= 80 and dx <= 200 and prev['text'].upper() in POSITIONS:
                    dist = dy + dx * 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = prev['text'].upper()
            if best_pos and best_pos not in stacks:
                stacks[best_pos] = val_bb
    return stacks


def _fuzzy_match_name(short_name: str, full_names: list) -> str | None:
    """Match a potentially truncated OCR name to the best full name.
    e.g. '淡淡的会...' → '淡淡的会顺顺'
    """
    if not short_name or not full_names:
        return None
    # Exact match first
    if short_name in full_names:
        return short_name
    # Prefix match (truncated name)
    clean = short_name.rstrip('.').strip()
    for fn in full_names:
        if fn.startswith(clean) or clean.startswith(fn[:min(4, len(fn))]):
            return fn
    # Similarity fallback
    from difflib import SequenceMatcher
    best = max(full_names, key=lambda fn: SequenceMatcher(None, clean, fn).ratio())
    if SequenceMatcher(None, clean, best).ratio() > 0.5:
        return best
    return None


def _extract_pote_total(items, bb_value: float = 0.10) -> float | None:
    """Extract total pot from image display (BB format: 'Pote Total: 518.7BB'
    or dollar format: 'PoteTotal:$51,87')."""
    for item in items:
        t = item['text']
        if 'Pote' in t or 'Total' in t:
            # BB format: 'Pote Total: 518.7 BB'
            m = re.search(r'(\d[\d,.]+)\s*BB', t, re.IGNORECASE)
            if m:
                val_str = m.group(1).replace('.', '').replace(',', '.')
                try:
                    val = float(val_str)
                    if val > 0:
                        return val
                except ValueError:
                    pass
            # Dollar format: 'PoteTotal:$51,87' or 'Pote Total: $ 51,87'
            m = re.search(r'\$\s*([\d,. ]+)', t)
            if m:
                val_str = m.group(1).strip().replace(' ', '').replace(',', '.')
                try:
                    usd = float(val_str)
                    if usd > 0.5:
                        return usd / bb_value
                except ValueError:
                    pass
    # Also look for large isolated BB amounts in top half of image
    mid_y = max(i['y'] for i in items) // 2 if items else 9999
    for item in items:
        if item['y'] > mid_y:
            break
        t = item['text']
        m = re.match(r'^(\d[\d,.]+)\s*BB?$', t, re.IGNORECASE)
        if m:
            val_str = m.group(1).replace('.', '').replace(',', '.')
            try:
                val = float(val_str)
                if val > 500:  # large pot
                    return val
            except ValueError:
                pass
    return None


def _calc_initial_stacks(player_map, results, ante_bb, sb_bb, bb_bb, str_bb,
                          sb_player, bb_player, str_player, bb_value,
                          pote_total_bb=None, end_stacks_bb=None,
                          all_player_names=None) -> dict:
    """
    Calculate initial USD stacks.
    Formula: initial = end_stack + abs(net_bb)  (for losers)
             initial = end_stack - net_bb = pote_total - net_bb  (for winner)
    end_stacks_bb: {player_name: end_stack_bb} from table view
    """
    result_map = {r['name']: r for r in results}
    if end_stacks_bb is None:
        end_stacks_bb = {}

    # Apply fuzzy matching for end_stacks that have truncated names
    if all_player_names:
        resolved_stacks = {}
        for short_name, val in end_stacks_bb.items():
            full = _fuzzy_match_name(short_name, all_player_names)
            if full:
                resolved_stacks[full] = val
            else:
                resolved_stacks[short_name] = val
        end_stacks_bb = resolved_stacks

    stacks = {}

    for name, data in player_map.items():
        net_bb = result_map.get(name, {}).get('net_bb')
        end_bb = end_stacks_bb.get(name)

        if net_bb is not None:
            if end_bb is not None:
                # Best case: initial = end - net = end + abs(net) for losers
                initial_bb = end_bb - net_bb
            elif net_bb > 0 and pote_total_bb:
                # Winner: end_stack = pote_total, initial = pote_total - net
                initial_bb = pote_total_bb - net_bb
            elif net_bb <= 0:
                # Loser (approx): initial ≈ abs(net)
                # This underestimates for folders (they keep their remaining stack)
                initial_bb = abs(net_bb)
            else:
                # Winner without pot total: approximate
                initial_bb = abs(net_bb) * 2  # rough guess
        else:
            # No result available (e.g. all-anonymous table or folder not in results)
            if end_bb is not None:
                # Use the oval stack directly as the best approximation of initial stack
                initial_bb = end_bb
            else:
                # Last resort: position-based estimate from blind amounts
                pos = data.get('pos', '')
                blind_bb = 0
                if pos == 'SB' or name == sb_player:
                    blind_bb = sb_bb
                elif pos == 'BB' or name == bb_player:
                    blind_bb = bb_bb
                elif pos == 'STR' or name == str_player:
                    blind_bb = str_bb
                initial_bb = ante_bb + blind_bb

        stacks[name] = round(max(initial_bb, 0) * bb_value, 2)

    return stacks


def _fix_allin_call_amounts(preflop, flop, turn, river, initial_stacks_bb,
                             ante_bb, sb_bb, bb_bb, str_bb,
                             sb_player, bb_player, str_player):
    """Fix all-in call amounts by tracking each player's cumulative committed chips."""
    total_in = {name: ante_bb for name in initial_stacks_bb}
    if sb_player and sb_player in total_in:
        total_in[sb_player] += sb_bb
    if bb_player and bb_player in total_in:
        total_in[bb_player] += bb_bb
    if str_player and str_player in total_in:
        total_in[str_player] += str_bb

    for street_acts in [preflop, flop, turn, river]:
        street_committed = {}
        for act in street_acts:
            name = act['name']
            action = act['action']
            amt = act.get('amount_bb') or 0.0
            allin = act.get('allin', False)

            if name not in total_in:
                total_in[name] = ante_bb

            if action == 'calls' and name in initial_stacks_bb:
                computed = max(0.0, initial_stacks_bb[name] - total_in[name])
                # Fix if OCR amount exceeds remaining stack, or player is all-in
                if computed > 0 and (allin or amt > computed + 2):
                    act['amount_bb'] = round(computed, 2)
                    amt = act['amount_bb']
                    act['allin'] = True

            if action == 'calls':
                total_in[name] = total_in.get(name, 0.0) + amt
                street_committed[name] = street_committed.get(name, 0.0) + amt
            elif action == 'bets':
                total_in[name] = total_in.get(name, 0.0) + amt
                street_committed[name] = amt
            elif action == 'raises':
                prev = street_committed.get(name, 0.0)
                delta = max(0.0, amt - prev)
                total_in[name] = total_in.get(name, 0.0) + delta
                street_committed[name] = amt


def _scan_board_strip(img, y1, y2, x_start, x_end) -> dict:
    """Run OCR on board strip and return {x_pos: rank_text}."""
    rank_items = {}
    strip = img[y1:y2, x_start:x_end]
    if strip.size < 100:
        return rank_items
    scale = 3
    strip_up = cv2.resize(strip, (strip.shape[1]*scale, strip.shape[0]*scale), cv2.INTER_CUBIC)
    result, _ = _ocr()(strip_up)
    if result:
        for ri in result:
            if not ri or len(ri) < 2: continue
            bbox, text = ri[0], ri[1]
            if isinstance(text, (list,tuple)): text = text[0] if text else ""
            text = str(text).strip()
            xs = [int(p[0])//scale + x_start for p in bbox]
            ys = [int(p[1])//scale + y1 for p in bbox]
            xmin, ymin = min(xs), min(ys)
            if ymin < y1 + 5: continue  # skip items at very edge of scan region
            if len(text) <= 4 and '.' not in text and ',' not in text:
                tup = text.upper()
                if tup in ('POTE', 'TOTAL', 'BB', 'SB', 'STR', 'UTG', 'HJ', 'CO',
                           'BTN', 'MP', 'HAND', 'ID', 'ANTE'): continue
                if len(tup) > 0 and tup[0].isdigit() or len(tup) == 1:
                    rank_items[xmin] = text
                elif len(tup) == 2:
                    # Skip two-char texts where both look like separate card ranks
                    # (e.g. 'KA','AK' = merged hole cards visible next to board)
                    if tup[0] in 'AKQJT' and tup[1] in 'AKQJT23456789':
                        continue
                    rank_items[xmin] = text
    return rank_items


def _detect_board(img, img_w, img_h) -> list:
    """Detect board cards from WPT Global screenshot.
    Board cards are at y≈10-15% of height (above the poker table view).
    5 cards laid horizontally: each has dark background + colored suit symbol.
    Suits: hearts=red(H<15), clubs=green(H=35-95), diamonds=blue(H=95-140), spades=dark.
    """
    x_start = int(img_w * 0.26)
    x_end   = int(img_w * 0.74)
    y1 = int(img_h * 0.10)

    # Try 15% first; extend to 18% if fewer than 5 card ranks found
    rank_items = _scan_board_strip(img, y1, int(img_h * 0.15), x_start, x_end)
    y2 = int(img_h * 0.15)
    if len(rank_items) < 5:
        rank_items_ext = _scan_board_strip(img, y1, int(img_h * 0.18), x_start, x_end)
        if len(rank_items_ext) >= len(rank_items):
            rank_items = rank_items_ext
            y2 = int(img_h * 0.18)

    # Fallback: scan board in two halves if still <5 cards found
    # (OCR sometimes misses extreme-left/right cards in the full-width strip)
    if len(rank_items) < 5:
        mid = (x_start + x_end) // 2
        for sx1, sx2 in [(x_start, mid + 60), (mid - 60, x_end)]:
            sub = _scan_board_strip(img, y1, y2, sx1, sx2)
            for xk, tv in sub.items():
                if not any(abs(xk - ex) < 30 for ex in rank_items):
                    rank_items[xk] = tv

    # If still <5 and we have ≥2 found, estimate missing positions from card spacing
    if 2 <= len(rank_items) < 5:
        xs_found = sorted(rank_items.keys())
        if len(xs_found) >= 2:
            # Estimate spacing from consecutive found cards
            spacings = [xs_found[i+1] - xs_found[i] for i in range(len(xs_found)-1)]
            spacing = int(sum(spacings) / len(spacings))
            # If spacing > 100px, cards between found positions may have been missed —
            # try halved spacing to fill in gaps (e.g. 8-?-5 with spacing 132 → try 66)
            if spacing > 100:
                half_sp = spacing // 2
                if 60 <= half_sp <= 160:
                    # Insert midpoints between all pairs of consecutive found positions
                    extra_xs = [xs_found[i] + half_sp for i in range(len(xs_found)-1)]
                    for mx in extra_xs:
                        if not any(abs(mx - ex) < 30 for ex in rank_items):
                            sx1 = max(x_start - 20, mx - 50)
                            sx2 = min(x_end + 20, mx + 80)
                            sub = _scan_board_strip(img, y1, y2, sx1, sx2)
                            for xk, tv in sub.items():
                                if not any(abs(xk - ex) < 30 for ex in rank_items):
                                    rank_items[xk] = tv
                    xs_found = sorted(rank_items.keys())
                    spacings = [xs_found[i+1] - xs_found[i] for i in range(len(xs_found)-1)]
                    spacing = int(sum(spacings) / len(spacings))
            if 60 <= spacing <= 160:
                # Extrapolate left and right to find up to 5 positions
                candidates = list(xs_found)
                # Extend left
                while len(candidates) < 5 and candidates[0] - spacing >= x_start - 20:
                    candidates.insert(0, candidates[0] - spacing)
                # Extend right
                while len(candidates) < 5 and candidates[-1] + spacing <= x_end + 20:
                    candidates.append(candidates[-1] + spacing)
                # For each estimated position without a card, do a focused scan
                for cx in candidates:
                    if any(abs(cx - ex) < 40 for ex in rank_items):
                        continue
                    sx1 = max(x_start - 20, cx - 50)
                    sx2 = min(x_end + 20, cx + 80)
                    sub = _scan_board_strip(img, y1, y2, sx1, sx2)
                    for xk, tv in sub.items():
                        if not any(abs(xk - ex) < 30 for ex in rank_items):
                            rank_items[xk] = tv

    # Step 2: cluster close x positions (deduplicate)
    sorted_xs = sorted(rank_items.keys())
    clusters = []
    for x in sorted_xs:
        if clusters and x - clusters[-1][0] < 15:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    # Representative x for each cluster (first x in cluster)
    card_positions = []
    for cluster in clusters:
        best_x = cluster[0]
        best_text = rank_items[best_x]
        # Prefer rank texts that look like single chars
        for cx in cluster:
            t = rank_items[cx]
            if len(t) == 1 and t.upper() in 'AKQJT98765432':
                best_x, best_text = cx, t
                break
        card_positions.append((best_x, best_text))

    # Step 3: detect suit for each card using bounded region
    def _detect_suit_board(img, num_x, y1, y2, prev_x, next_x):
        """Detect suit: sample region centered on card.
        Uses midpoint between prev/next card to prevent colour bleed from adjacent cards."""
        left  = max(x_start, num_x - 15)
        right = min(x_end,   num_x + 65)

        card_region = img[y1:y2, left:right]
        if card_region.size < 100:
            return 's'

        hsv = cv2.cvtColor(card_region, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        # Use top-30% most saturated colored pixels to filter out red felt contamination.
        # Suit symbols are more saturated than background felt; spades (dark) have
        # very few colored pixels → correctly fall through to 's'.
        colored = (s > 100) & (v > 80) & ~((h >= 15) & (h <= 40))
        n_colored = colored.sum()
        if n_colored < 5:
            return 's'

        # Use top 30% by saturation to isolate suit symbol pixels from background
        s_thresh = np.percentile(s[colored], 70)
        high_sat = colored & (s >= s_thresh)
        h_c = h[high_sat]
        if len(h_c) == 0:
            return 's'

        red   = ((h_c < 15) | (h_c > 160)).sum()
        green = ((h_c >= 35) & (h_c < 95)).sum()
        blue  = ((h_c >= 95) & (h_c <= 140)).sum()

        if red >= green and red >= blue and red > 3:
            return 'h'
        elif blue >= green and blue > 3:
            return 'd'
        elif green > 3:
            return 'c'
        return 's'

    def _normalize_board_rank(text):
        t = text.strip().upper()
        # Common OCR errors for card ranks
        if t in ('7G', '2G', '2C', '2H', '2S', '2T', 'ZG'): return '2'
        if t in ('0', 'O', 'Q0', 'OO'): return 'Q'  # Q misread as 0/O by OCR
        if t in ('1O', 'IO', '10'): return 'T'        # 10 misread
        if t in ('I', 'IJ', 'J1', 'LJ'): return 'J'  # J misread as I/l
        if re.match(r'^6[^0-9]$', t) or t == '60': return '6'
        if re.match(r'^8[^0-9]$', t) or t == '80': return '8'
        if re.match(r'^9[^0-9]$', t) or t == '90': return '9'
        if t == 'T': return 'T'
        if t.isdigit() and 2 <= int(t) <= 9: return t
        if t in ('A', 'K', 'Q', 'J'): return t
        if len(t) == 2 and t[0].isdigit() and not t[1].isdigit(): return t[0]
        return t if (len(t) == 1 and t in 'AKQJT98765432') else None

    cards = []
    for i, (nx, text) in enumerate(card_positions):
        rank = _normalize_board_rank(text)
        if not rank:
            continue
        prev_x = card_positions[i-1][0] if i > 0 else None
        next_x = card_positions[i+1][0] if i < len(card_positions)-1 else None
        suit = _detect_suit_board(img, nx, y1, y2, prev_x, next_x)
        cards.append(f"{rank}{suit}")

    return cards


def _extract_showdown_cards(img, img_w, img_h, ocr_items, results, result_start_y) -> dict:
    """
    Extract hole cards for showdown players from the RIVER column results section.
    Cards appear as small thumbnails between player name and position badge.
    Returns {player_name: [card1, card2]}.
    """
    ocr_fn = _ocr()
    col_x = int(4 * img_w / 5)

    # Map player name → approximate y (from results section only)
    name_ys = {}
    for item in ocr_items:
        if item['x'] >= col_x and item['y'] >= result_start_y:
            for r in results:
                nm = r['name']
                if nm == item['text'] or (len(nm) > 3 and item['text'].startswith(nm[:3])):
                    if nm not in name_ys:
                        name_ys[nm] = item['y']

    showdown_cards = {}

    for r in results:
        name = r['name']
        if name not in name_ys:
            continue
        # Skip obvious folders (tiny net loss = just posted blinds/antes)
        if abs(r.get('net_bb', 0)) < 3:
            continue
        ny = name_ys[name]

        # Try multiple y-windows for rank OCR to handle slight alignment variations
        found_ranks = []
        for dy_start, dy_end in [(18, 72), (20, 75), (15, 68)]:
            rank_y1 = ny + dy_start
            rank_y2 = ny + dy_end
            rank_x1 = col_x + 55   # skip avatar
            rank_x2 = min(col_x + 165, img_w)

            ocr_region = img[rank_y1:rank_y2, rank_x1:rank_x2]
            if ocr_region.size < 100:
                continue

            scale = 5
            big = cv2.resize(ocr_region, (ocr_region.shape[1]*scale, ocr_region.shape[0]*scale),
                             cv2.INTER_LANCZOS4)
            result_ocr, _ = ocr_fn(big)

            ranks = []
            if result_ocr:
                for item in result_ocr:
                    if not item or len(item) < 2:
                        continue
                    bbox = item[0]
                    txt = str(item[1]).strip().upper()
                    if txt in ('H', 'M', 'N', 'AA', 'AL', 'AH'): txt = 'A'
                    if txt in ('1O', 'IO', '10', 'T0', '1Q'): txt = 'T'
                    if txt == 'O': txt = '9'
                    if txt in 'AKQJT' or (txt.isdigit() and 2 <= int(txt) <= 9):
                        x_center = sum(p[0] for p in bbox) / 4
                        ranks.append((int(x_center), txt))

            ranks.sort()
            if len(ranks) >= 2:
                found_ranks = [r[1] for r in ranks[:2]]
                break

        if len(found_ranks) < 2:
            continue

        # Suit detection: use tight x windows to avoid avatar and badge contamination
        # Each card is ~40px wide; cards start ~62px from col_x
        card_w = 40
        suit_y1 = ny + 28
        suit_y2 = ny + 56  # avoid position badge (appears at ny+55-75)

        suits = []
        for i in range(2):
            cx1 = col_x + 62 + i * card_w
            cx2 = min(cx1 + card_w, img_w)
            card_crop = img[suit_y1:suit_y2, cx1:cx2]
            suits.append(_detect_hole_card_suit_tight(card_crop))

        showdown_cards[name] = [f"{found_ranks[0]}{suits[0]}", f"{found_ranks[1]}{suits[1]}"]

    return showdown_cards


def _detect_hole_card_suit_tight(region) -> str:
    """Detect suit from a tightly-cropped card thumbnail region.
    Thumbnail color: hearts=red, diamonds=red, clubs=dark/black, spades=dark/black.
    Distinction ♥/♦ uses shape; ♣/♠ both dark → default to 's' (spades).
    """
    if region.size < 20:
        return 's'

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Exclude orange-gold badge colors (H=12-55) to avoid badge contamination
    colored = (s > 90) & (v > 90) & ~((hh >= 12) & (hh <= 55))
    if colored.sum() < 3:
        return 's'

    h_c = hh[colored]
    red   = int(((h_c < 12) | (h_c > 155)).sum())
    green = int(((h_c >= 55) & (h_c < 100)).sum())
    blue  = int(((h_c >= 100) & (h_c <= 155)).sum())

    if green > red and green > blue and green > 3:
        return 'c'
    if blue > red and blue > green and blue > 3:
        return 'd'
    if red > 3:
        # Hearts and diamonds both appear red in thumbnails.
        # Key distinction: ♥ has TWO bumps creating a NARROW x-gap (2-3px);
        # rank letter 'A' legs also create bimodal but with a WIDER gap (4+ px);
        # ♦ body has no bimodal gap (solid rhombus).
        red_mask = (((hh < 12) | (hh > 155)) & (s > 90) & (v > 90)).astype(np.uint8)
        row_start = int(region.shape[0] * 0.35)
        row_end   = int(region.shape[0] * 0.82)
        heart_votes = 0
        diamond_votes = 0
        for row in range(row_start, row_end):
            if row >= red_mask.shape[0]:
                break
            x_positions = np.where(red_mask[row] > 0)[0]
            if len(x_positions) >= 4:
                gaps = np.diff(np.sort(x_positions))
                max_gap = int(gaps.max())
                if 2 <= max_gap <= 3:
                    heart_votes += 1    # narrow gap → ♥ bumps
                elif max_gap >= 4:
                    diamond_votes += 1  # wide gap → rank letter legs (not ♥)
        if heart_votes > 0 and heart_votes >= diamond_votes:
            return 'h'
        if diamond_votes > 0:
            return 'd'
        # No bimodal → solid symbol → diamond
        pts = cv2.findNonZero(red_mask)
        return 'd' if pts is not None and len(pts) >= 3 else 'h'
    return 's'


def _build_showdown(results, showdown_cards, bb_value, board=None) -> list:
    """Build showdown entries with hand descriptions."""
    from src.hand_eval import best_hand_desc
    showdown = []
    for r in results:
        if r['name'] in showdown_cards:
            cards = showdown_cards[r['name']]
            desc = best_hand_desc(cards, board or []) if board else ''
            showdown.append({
                'name': r['name'],
                'cards': cards,
                'hand_desc': desc,
                'won': r['net_bb'] > 0,
                'amount': 0,
            })
    return showdown


def _build_winners(results, showdown, bb_value) -> list:
    winners = []
    for r in results:
        if r['net_bb'] > 0:
            winners.append({'name': r['name'], 'amount': 0, 'pot_desc': 'pot'})
    return winners


def _convert_actions(raw_acts: list, bb_value: float, initial_bet_bb: float = 0.0) -> list:
    """Convert BB actions → USD PokerStars format."""
    converted = []
    current_bet_bb = initial_bet_bb

    for act in raw_acts:
        name = act['name']
        action = act['action']
        amount_bb = act.get('amount_bb')
        allin = act.get('allin', False)

        if action == 'folds':
            converted.append({'name': name, 'action': 'folds'})
        elif action == 'checks':
            converted.append({'name': name, 'action': 'checks'})
        elif action == 'calls':
            if amount_bb:
                converted.append({'name': name, 'action': 'calls',
                                   'amount': round(amount_bb * bb_value, 2), 'allin': allin})
            else:
                converted.append({'name': name, 'action': 'calls', 'amount': 0, 'allin': allin})
        elif action == 'bets':
            if amount_bb:
                current_bet_bb = amount_bb
                converted.append({'name': name, 'action': 'bets',
                                   'amount': round(amount_bb * bb_value, 2), 'allin': allin})
            else:
                converted.append({'name': name, 'action': 'bets', 'amount': 0, 'allin': allin})
        elif action == 'raises':
            if amount_bb:
                total_usd = round(amount_bb * bb_value, 2)
                inc_bb = amount_bb - current_bet_bb
                inc_usd = round(max(inc_bb, 0.01) * bb_value, 2)
                current_bet_bb = amount_bb
                converted.append({'name': name, 'action': 'raises',
                                   'amount': inc_usd, 'total': total_usd, 'allin': allin})
            else:
                converted.append({'name': name, 'action': 'raises', 'amount': 0, 'total': 0})
        else:
            converted.append({'name': name, 'action': action})

    return converted


def _build_summary_seats(seats, player_map, preflop, flop, turn, river,
                          showdown, btn_player, sb_player, bb_player,
                          initial_stacks, bb_value, result_entries=None,
                          pote_total_bb=None) -> list:
    """Build SUMMARY seats."""
    # Build result lookup
    result_map = {r['name']: r for r in (result_entries or [])}
    winners_set = {r['name'] for r in (result_entries or []) if r['net_bb'] > 0}
    losers_set  = {r['name'] for r in (result_entries or []) if r['net_bb'] < 0}

    # Determine fold streets
    folded = {}
    preflop_betters = {a['name'] for a in preflop if a['action'] in ('bets', 'raises', 'calls')}
    # SB/BB/straddle posters are treated as having "bet" preflop for fold classification
    _str_player = next((n for n, d in player_map.items() if d.get('pos') == 'STR'), None)
    for _p in [sb_player, bb_player, _str_player]:
        if _p: preflop_betters.add(_p)

    for a in river:
        if a['action'] == 'folds':
            folded[a['name']] = 'river'
    for a in turn:
        if a['action'] == 'folds' and a['name'] not in folded:
            folded[a['name']] = 'turn'
    for a in flop:
        if a['action'] == 'folds' and a['name'] not in folded:
            folded[a['name']] = 'flop'
    for a in preflop:
        if a['action'] == 'folds' and a['name'] not in folded:
            folded[a['name']] = 'preflop'

    # Track last street each player was non-fold active (for players missing from result_entries)
    last_active_street = {}
    for a in preflop:
        if a['action'] != 'folds':
            last_active_street[a['name']] = 'preflop'
    for a in flop:
        if a['action'] != 'folds':
            last_active_street[a['name']] = 'flop'
    for a in turn:
        if a['action'] != 'folds':
            last_active_street[a['name']] = 'turn'
    for a in river:
        if a['action'] != 'folds':
            last_active_street[a['name']] = 'river'

    # Infer river all-in caller as loser and last river aggressor as winner when result_entries missing
    inferred_losers = set()
    inferred_winners = set()
    river_players = {a['name'] for a in river}
    # Trigger when river participants are NOT covered by result_entries
    river_uncovered = river_players - winners_set - losers_set
    if river_uncovered:
        river_allin_callers = {a['name'] for a in river if a['action'] == 'calls' and a.get('allin')}
        inferred_losers = river_allin_callers & river_uncovered
        # Last non-allin river aggressor (bet/raise) is the probable winner
        for a in reversed(river):
            if a['action'] in ('bets', 'raises') and not a.get('allin') and a['name'] not in inferred_losers:
                inferred_winners.add(a['name'])
                break

    # Players who reached showdown: in result_entries (winner or loser) AND either
    # appeared in river actions OR went all-in on any earlier street.
    allin_players = set()
    for a in preflop + flop + turn + river:
        if a.get('allin'):
            allin_players.add(a['name'])
    showdown_players = (winners_set | losers_set) & (river_players | allin_players)

    showdown_map = {s['name']: s for s in showdown}
    summary = []

    for name, seat in sorted(seats.items(), key=lambda x: x[1]):
        role = ''
        if name == btn_player:
            role = 'button'
        elif name == sb_player:
            role = 'small blind'
        elif name == bb_player:
            role = 'big blind'

        if name in showdown_map:
            s = showdown_map[name]
            won = s.get('won', name in winners_set)
            outcome = 'showed_won' if won else 'showed_lost'
            # For winners: collected amount = total_pot divided by number of winners.
            # (net_bb is net gain only, not the total collected amount.)
            if won and pote_total_bb:
                amount = round(pote_total_bb * bb_value / max(len(winners_set), 1), 2)
            else:
                net_bb = result_map.get(name, {}).get('net_bb', 0)
                amount = round(initial_stacks.get(name, 0) + (net_bb * bb_value), 2)
            summary.append({
                'seat': seat, 'name': name, 'role': role,
                'outcome': outcome, 'cards': s.get('cards', []),
                'amount': amount,
                'hand_desc': s.get('hand_desc', ''),
            })
        elif name in winners_set:
            # Winner without hole cards in showdown; collected = pot / n_winners.
            if pote_total_bb:
                amount = round(pote_total_bb * bb_value / max(len(winners_set), 1), 2)
            else:
                amount = 0.0
            summary.append({
                'seat': seat, 'name': name, 'role': role,
                'outcome': 'showed_won', 'cards': [],
                'amount': amount, 'hand_desc': '',
            })
        elif name in showdown_players and name in losers_set:
            # Loser at showdown without hole cards
            summary.append({
                'seat': seat, 'name': name, 'role': role,
                'outcome': 'showed_lost', 'cards': [],
                'amount': 0, 'hand_desc': '',
            })
        elif name in inferred_winners:
            # Inferred winner from river aggression (no result_entries available)
            amount = round(pote_total_bb * bb_value, 2) if pote_total_bb else 0.0
            summary.append({
                'seat': seat, 'name': name, 'role': role,
                'outcome': 'showed_won', 'cards': [], 'amount': amount, 'hand_desc': '',
            })
        elif name in inferred_losers:
            # Inferred loser from river all-in call (no result_entries available)
            summary.append({
                'seat': seat, 'name': name, 'role': role,
                'outcome': 'showed_lost', 'cards': [], 'amount': 0, 'hand_desc': '',
            })
        else:
            # Use actual fold street from actions; fall back to last known active street
            if name in folded:
                street = folded[name]
            elif name in last_active_street:
                street = last_active_street[name]
            else:
                street = 'preflop'
            if street == 'preflop':
                had_bet = name in preflop_betters
                outcome = 'folded_preflop' if had_bet else 'folded_preflop_no_bet'
            elif street == 'flop':
                outcome = 'folded_flop'
            elif street == 'turn':
                outcome = 'folded_turn'
            else:
                outcome = 'folded_river'
            summary.append({'seat': seat, 'name': name, 'role': role, 'outcome': outcome})

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to WPT Global screenshot')
    parser.add_argument('--bb', type=float, default=None, help='USD value of 1 BB (auto-detected if omitted)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    result = parse_hand(args.image, bb_value=args.bb, debug=args.debug)
    print(result)
