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

    # Timestamp
    pred_ts = _extract_field(predicted, r'- (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ET)')
    gt_ts = _extract_field(ground_truth, r'- (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ET)')
    results['timestamp'] = 1.0 if pred_ts == gt_ts else _similarity(pred_ts, gt_ts)

    # Stakes
    pred_stakes = _extract_field(predicted, r'\((\$[\d.]+/\$[\d.]+ USD)\)')
    gt_stakes = _extract_field(ground_truth, r'\((\$[\d.]+/\$[\d.]+ USD)\)')
    results['stakes'] = 1.0 if pred_stakes == gt_stakes else 0.0

    # Button seat
    pred_btn = _extract_field(predicted, r'Seat #(\d+) is the button')
    gt_btn = _extract_field(ground_truth, r'Seat #(\d+) is the button')
    results['button_seat'] = 1.0 if pred_btn == gt_btn else 0.0

    # Players/stacks
    pred_seats = _extract_seats(predicted)
    gt_seats = _extract_seats(ground_truth)
    if gt_seats:
        matched = sum(
            1 for name, stack in gt_seats.items()
            if name in pred_seats and abs(pred_seats[name] - stack) < 0.01
        )
        results['players_stacks'] = matched / len(gt_seats)
    else:
        results['players_stacks'] = 1.0 if not pred_seats else 0.0

    # Board cards
    pred_board = _extract_board(predicted)
    gt_board = _extract_board(ground_truth)
    results['board'] = 1.0 if pred_board == gt_board else _similarity(pred_board, gt_board)

    # Actions
    pred_actions = _extract_actions(predicted)
    gt_actions = _extract_actions(ground_truth)
    if gt_actions:
        matched_actions = sum(1 for a in gt_actions if a in pred_actions)
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
