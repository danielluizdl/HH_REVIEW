"""
Core OCR-based parser for WPT Global hand history screenshots.
Images are portrait (~988x2048), displayed in BB mode.
Actions are in Portuguese; amounts use European decimal (comma).
"""

import cv2
import numpy as np
import re
from rapidocr_onnxruntime import RapidOCR
from .detectors import detect_board_cards

_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr


# Portuguese → English action map
PT_ACTIONS = {
    'desistir': 'folds',
    'passar': 'checks',
    'aposta': 'bets',
    'aumentar': 'raises',
    'pagar': 'calls',
    'fazer call': 'calls',
    'fazercall': 'calls',
    'all-in': 'allin',
}

POSITION_MAP = {
    'UTG': 'UTG', 'UTG+1': 'UTG+1', 'UTG+2': 'UTG+2',
    'MP': 'MP', 'MP+1': 'MP+1', 'HJ': 'HJ', 'CO': 'CO',
    'BTN': 'BTN', 'SB': 'SB', 'BB': 'BB', 'STR': 'straddle',
}

KNOWN_POSITIONS = {'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'HJ', 'CO', 'BTN', 'SB', 'BB', 'STR', 'Co', 'co'}


def _parse_bb(text: str) -> float:
    """Parse BB amount from text like '3,50BB', '64,80 BB', '8,80BB', '-0,50 BB'."""
    t = text.strip()
    sign = -1 if t.startswith('-') else 1
    t = t.lstrip('+-')
    # Remove BB suffix
    t = re.sub(r'\s*BB?\s*$', '', t, flags=re.IGNORECASE).strip()
    # Replace European decimal comma with period
    t = t.replace(',', '.')
    # Remove thousand separators (periods before 3 digits)
    t = re.sub(r'\.(?=\d{3})', '', t)
    try:
        return sign * float(t)
    except ValueError:
        return 0.0


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def run_full_ocr(img) -> list:
    """Run OCR on full image. Returns list of (x, y, w, h, text, conf)."""
    ocr = _get_ocr()
    result, _ = ocr(img)
    if not result:
        return []
    items = []
    for item in result:
        if not item or len(item) < 3:
            continue
        bbox, text, conf = item[0], item[1], item[2]
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        if not text:
            continue
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        items.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text.strip(), 'conf': conf})
    return sorted(items, key=lambda i: i['y'])


def _items_in_col(items, x_min, x_max, y_min=0, y_max=99999):
    """Filter OCR items to a specific column region."""
    return [i for i in items if x_min <= i['x'] < x_max and y_min <= i['y'] <= y_max]


def find_action_table_y(items) -> int:
    """Find y position of the BLINDS & ANTE header."""
    for item in items:
        t = item['text'].upper().replace(' ', '').replace('&', '').replace('E', '')
        if 'BLINDSANT' in t or 'BLINDS' in t:
            return item['y']
    # Fallback: look for PRE-FLOP
    for item in items:
        if 'PRE-FLOP' in item['text'].upper() or 'PREFLOP' in item['text'].upper():
            return item['y']
    return 600  # fallback


def extract_hand_id(items) -> str:
    """Extract hand ID from top of image."""
    for item in items[:10]:  # hand ID is near top
        t = item['text']
        if 'HAND' in t.upper() or 'ID' in t.upper():
            # Extract number sequence
            nums = re.findall(r'\d+', t)
            if nums:
                return ''.join(nums)
    # Search all items for HANDID pattern
    for item in items:
        if item['y'] > 100:
            break
        nums = re.findall(r'\d{10,}', item['text'])
        if nums:
            return nums[0]
    return '0'


def extract_timestamp(items) -> str:
    """Extract timestamp, normalize to PokerStars format."""
    for item in items:
        t = item['text']
        # Look for date pattern
        m = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})\s*(\d{2}:\d{2}:\d{2})', t)
        if m:
            return f"{m.group(1)}/{m.group(2)}/{m.group(3)} {m.group(4)} ET"
    return '2026/01/01 00:00:00 ET'


def extract_player_end_stacks(items, img_w: int) -> dict:
    """
    Extract player stacks from the poker table view (y < action_table_y).
    Returns {name: stack_bb}.
    """
    result = {}
    action_y = find_action_table_y(items)
    table_items = [i for i in items if i['y'] < action_y - 50]

    # Look for (name, stack) pairs: name followed by "XXXBB" on next item
    for idx, item in enumerate(table_items):
        t = item['text']
        # Stack patterns: "546BB", "54,40 BB", "191BB", etc.
        if re.search(r'^\d[\d,.]+\s*BB?$', t, re.IGNORECASE):
            bb_val = _parse_bb(t)
            if bb_val > 0:
                # Find player name above this
                # Search nearby items
                for prev in table_items:
                    if abs(prev['y'] - item['y']) < 60 and prev['x'] is not None:
                        if abs(prev['x'] - item['x']) < 150:
                            name = prev['text']
                            if _is_player_name(name):
                                result[name] = bb_val
    return result


def _is_player_name(text: str) -> bool:
    """Heuristic: is this text a player name?"""
    if not text or len(text) < 2:
        return False
    # Not a keyword
    keywords = {'HAND', 'ID', 'BLINDS', 'ANTE', 'PRE-FLOP', 'FLOP', 'TURN', 'RIVER',
                'BB', 'SB', 'STR', 'UTG', 'MP', 'HJ', 'CO', 'BTN', 'WINNER',
                'Aposta', 'Aumentar', 'Pagar', 'Desistir', 'Passar', 'All-in',
                'Ante', 'Pote', 'Total', 'HANDID'}
    if text.upper() in {k.upper() for k in keywords}:
        return False
    if re.match(r'^[\d,.]+\s*BB?$', text, re.IGNORECASE):
        return False
    if re.match(r'^\d{4}[-/]\d{2}', text):
        return False
    return True


def extract_blinds_data(items, col_x_max: int, header_y: int) -> dict:
    """
    Parse the BLINDS & ANTE column to extract:
    - ante_per_player (BB)
    - blinds: list of {name, type, amount_bb}
    """
    col_items = _items_in_col(items, 0, col_x_max, header_y, 99999)
    col_items = sorted(col_items, key=lambda i: i['y'])

    result = {
        'ante_bb': 0.5,  # default
        'blinds': [],    # [{name, type, amount_bb}]
        'num_players': 0,
    }

    i = 0
    while i < len(col_items):
        item = col_items[i]
        t = item['text']

        # Detect ante block: "Ante" followed by "X,XXBB"
        if t.lower() == 'ante' or t.lower().startswith('ante'):
            # Next item should be amount
            if i + 1 < len(col_items):
                next_t = col_items[i+1]['text']
                ante_total = _parse_bb(next_t)
                result['_ante_total_bb'] = ante_total

        # Detect player blind posting: player name then SB/BB/STR then amount
        if _is_player_name(t) and len(t) > 1:
            # Look ahead for position badge and amount
            pos_type = None
            amount_bb = None
            for j in range(i+1, min(i+5, len(col_items))):
                jt = col_items[j]['text']
                if jt.upper() in ('SB',):
                    pos_type = 'small blind'
                elif jt.upper() in ('BB',):
                    pos_type = 'big blind'
                elif jt.upper() in ('STR',):
                    pos_type = 'straddle'
                elif re.match(r'^[\d,.]+\s*BB?$', jt, re.IGNORECASE):
                    amount_bb = _parse_bb(jt)

            if pos_type and amount_bb:
                result['blinds'].append({
                    'name': t,
                    'type': pos_type,
                    'amount_bb': amount_bb,
                })
        i += 1

    return result


def _parse_action_from_cell(cell_items: list) -> dict | None:
    """
    Parse a player action cell (list of OCR items at similar y position).
    Returns dict with name, action, amount_bb, total_bb, position, allin.
    """
    if not cell_items:
        return None

    # Sort by y within cell
    cell_items = sorted(cell_items, key=lambda i: i['y'])

    name = None
    action = None
    amount_bb = None
    position = None
    allin = False

    for item in cell_items:
        t = item['text']
        tl = t.lower().strip()

        # Player name (first non-keyword text)
        if name is None and _is_player_name(t) and t.upper() not in KNOWN_POSITIONS:
            name = t

        # Position badge
        if t.upper() in KNOWN_POSITIONS:
            position = t.upper()

        # Action
        if tl in PT_ACTIONS:
            action = PT_ACTIONS[tl]
        elif 'desistir' in tl:
            action = 'folds'
        elif 'passar' in tl:
            action = 'checks'
        elif 'aposta' in tl:
            action = 'bets'
        elif 'aumentar' in tl:
            action = 'raises'
        elif 'pagar' in tl or 'fazer' in tl:
            action = 'calls'
        elif 'all-in' in tl or 'allin' in tl:
            allin = True

        # Amount (BB)
        if re.match(r'^[+-]?[\d,.]+\s*BB?$', t, re.IGNORECASE) and 'all' not in tl:
            amount_bb = abs(_parse_bb(t))

    if name is None or action is None:
        return None

    return {
        'name': name,
        'action': action,
        'amount_bb': amount_bb,
        'position': position,
        'allin': allin,
    }


def extract_street_actions(items, col_x_min, col_x_max, header_y, next_section_y=99999) -> list:
    """
    Extract ordered action list from a street column.
    Groups OCR items into cells (one per player action) by y-distance.
    """
    pot_row_y = header_y + 40  # pot amount row
    action_start_y = pot_row_y + 40

    col_items = _items_in_col(items, col_x_min, col_x_max, action_start_y, next_section_y)
    col_items = sorted(col_items, key=lambda i: i['y'])

    if not col_items:
        return []

    # Group into cells by y-proximity (gap > 30px = new cell)
    cells = []
    current_cell = [col_items[0]]
    for item in col_items[1:]:
        if item['y'] - current_cell[-1]['y'] > 55:
            cells.append(current_cell)
            current_cell = [item]
        else:
            current_cell.append(item)
    if current_cell:
        cells.append(current_cell)

    actions = []
    for cell in cells:
        act = _parse_action_from_cell(cell)
        if act:
            actions.append(act)

    return actions


def extract_results(items, img_w: int) -> list:
    """
    Extract result entries from RIVER column (right side).
    Returns list of {name, position, net_bb, cards} sorted by y.
    """
    river_x_min = int(img_w * 0.8)
    result_items = [i for i in items if i['x'] >= river_x_min]
    result_items = sorted(result_items, key=lambda i: i['y'])

    results = []
    i = 0
    while i < len(result_items):
        item = result_items[i]
        t = item['text']
        # Net result pattern: "+XXXBB" or "-XXXBB"
        if re.match(r'^[+-][\d,.]+\s*BB?$', t, re.IGNORECASE):
            net_bb = _parse_bb(t)
            # Find name nearby
            name = None
            pos = None
            for prev in result_items:
                if abs(prev['y'] - item['y']) < 200 and prev['y'] <= item['y']:
                    if _is_player_name(prev['text']):
                        name = prev['text']
                    if prev['text'].upper() in KNOWN_POSITIONS:
                        pos = prev['text'].upper()
            if name:
                results.append({'name': name, 'position': pos, 'net_bb': net_bb})
        i += 1

    return results


def find_result_section_y(items, img_w, header_y) -> int:
    """Find where the result section starts in the RIVER column."""
    river_x_min = int(img_w * 0.8)
    river_items = [i for i in items if i['x'] >= river_x_min and i['y'] > header_y + 100]
    river_items = sorted(river_items, key=lambda i: i['y'])

    # Results contain "+XXXBB" or "-XXXBB" patterns
    for item in river_items:
        if re.match(r'^[+-][\d,.]+\s*BB?$', item['text'], re.IGNORECASE):
            return item['y']
    return 99999
