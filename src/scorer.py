"""Score parser output against ground truth."""
import re
from difflib import SequenceMatcher


def _extract_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ''


def _extract_seats(text: str) -> dict:
    seats = {}
    for m in re.finditer(r'Seat (\d+): (.+?) \(\$?([\d.]+) in chips\)', text):
        seats[m.group(2).strip()] = float(m.group(3))
    return seats


def _extract_actions(text: str) -> list:
    actions = []
    for line in text.split('\n'):
        line = line.strip()
        if re.match(r'.+: (folds|checks|calls|bets|raises|posts)', line):
            actions.append(line)
    return actions


def _extract_board(text: str) -> str:
    m = re.search(r'Board \[(.+?)\]', text)
    return m.group(1).strip() if m else ''


def _extract_total_pot(text: str) -> str:
    m = re.search(r'Total pot \$?([\d.]+)', text)
    return m.group(1) if m else ''


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_hand(predicted: str, ground_truth: str) -> dict:
    """
    Compare predicted hand history to ground truth.
    Returns dict with per-field accuracy and overall percentage.
    """
    results = {}

    # Hand ID
    pred_id = _extract_field(predicted, r'Hand #(\S+):')
    gt_id = _extract_field(ground_truth, r'Hand #(\S+):')
    results['hand_id'] = 1.0 if pred_id == gt_id else 0.0

    # Timestamp — compare date+time only (strip ET suffix, normalize single-digit hours)
    def _ts_normalized(text):
        m = re.search(r'(\d{4}/\d{2}/\d{2}) (\d{1,2}:\d{2}:\d{2})', text)
        if not m:
            return ''
        date, time = m.group(1), m.group(2)
        h, mi, s = time.split(':')
        return f"{date} {int(h):02d}:{mi}:{s}"
    pred_ts = _ts_normalized(predicted)
    gt_ts = _ts_normalized(ground_truth)
    results['timestamp'] = 1.0 if pred_ts == gt_ts else (_similarity(pred_ts, gt_ts) if pred_ts and gt_ts else 0.0)

    # Stakes — extract SB/BB amounts (ignore USD suffix, handle format variants)
    def _stakes_amounts(text):
        m = re.search(r'No Limit \(\$?([\d.]+)/\$?([\d.]+)', text)
        return (m.group(1), m.group(2)) if m else ('', '')
    pred_sb, pred_bb = _stakes_amounts(predicted)
    gt_sb, gt_bb = _stakes_amounts(ground_truth)
    results['stakes'] = 1.0 if (pred_sb == gt_sb and pred_bb == gt_bb) else 0.0

    # Button player — compare the player NAME sitting at the button seat
    # (WPT Global seat numbers can't be reliably derived from the image alone)
    def _btn_player(text):
        btn_seat = _extract_field(text, r'Seat #(\d+) is the button')
        if not btn_seat:
            return ''
        m = re.search(rf'Seat {btn_seat}: (.+?) \(', text)
        return m.group(1).strip() if m else ''
    pred_btn_player = _btn_player(predicted)
    gt_btn_player = _btn_player(ground_truth)
    if not pred_btn_player and not gt_btn_player:
        results['button_seat'] = 1.0
    elif not pred_btn_player or not gt_btn_player:
        results['button_seat'] = 0.0
    elif pred_btn_player == gt_btn_player:
        results['button_seat'] = 1.0
    else:
        results['button_seat'] = 1.0 if _similarity(pred_btn_player, gt_btn_player) >= 0.70 else 0.0

    # Players/stacks — fuzzy name matching for OCR character substitutions
    pred_seats = _extract_seats(predicted)
    gt_seats = _extract_seats(ground_truth)
    if gt_seats:
        def _stack_close(pred_s, gt_s):
            # Relative tolerance: 3% of stack OR absolute $2, whichever is larger.
            # Handles: BB rounding (±$0.05), starting vs ending stack mismatch (±a few BB).
            tol = max(2.0, abs(gt_s) * 0.03)
            return abs(pred_s - gt_s) <= tol
        def _find_stack(gt_name, gt_stack):
            if gt_name in pred_seats:
                return _stack_close(pred_seats[gt_name], gt_stack)
            # fuzzy fallback: accept if name similarity >= 0.70 and stack within tolerance
            for p_name, p_stack in pred_seats.items():
                if _similarity(gt_name, p_name) >= 0.70 and _stack_close(p_stack, gt_stack):
                    return True
            return False
        matched = sum(1 for name, stack in gt_seats.items() if _find_stack(name, stack))
        results['players_stacks'] = matched / len(gt_seats)
    else:
        results['players_stacks'] = 1.0 if not pred_seats else 0.0

    # Board cards
    pred_board = _extract_board(predicted)
    gt_board = _extract_board(ground_truth)
    results['board'] = 1.0 if pred_board == gt_board else _similarity(pred_board, gt_board)

    # Actions — fuzzy match player names to handle OCR character substitutions
    pred_actions = _extract_actions(predicted)
    gt_actions = _extract_actions(ground_truth)
    if gt_actions:
        def _action_matches(gt_a, pred_list):
            if gt_a in pred_list:
                return True
            # split "PlayerName: action rest" and fuzzy-match names
            m = re.match(r'^(.+?): (.+)$', gt_a)
            if not m:
                return False
            gt_name, gt_act = m.group(1), m.group(2)
            for p_a in pred_list:
                pm = re.match(r'^(.+?): (.+)$', p_a)
                if pm and _similarity(gt_name, pm.group(1)) >= 0.70 and gt_act == pm.group(2):
                    return True
            return False
        matched_actions = sum(1 for a in gt_actions if _action_matches(a, pred_actions))
        results['actions'] = matched_actions / len(gt_actions)
    else:
        results['actions'] = 1.0

    # Total pot
    pred_pot = _extract_total_pot(predicted)
    gt_pot = _extract_total_pot(ground_truth)
    results['total_pot'] = 1.0 if pred_pot == gt_pot else 0.0

    # Overall
    results['overall'] = sum(results.values()) / len(results) * 100

    return results


def print_score_report(image_name: str, predicted: str, ground_truth: str):
    scores = score_hand(predicted, ground_truth)
    print(f"\n=== Score for {image_name} ===")
    for field, score in scores.items():
        if field == 'overall':
            print(f"  OVERALL: {score:.1f}%")
        else:
            status = "OK" if score >= 0.99 else f"{score*100:.0f}%"
            print(f"  {field:<20}: {status}")
