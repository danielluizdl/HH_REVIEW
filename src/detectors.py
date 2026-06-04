import cv2
import numpy as np
import re
from .ocr_engine import read_region, read_region_lines

# HSV ranges for card suit colors
SUIT_RANGES = {
    'h': [(0, 60, 80), (10, 255, 255), (160, 60, 80), (180, 255, 255)],   # red (hearts)
    'd': [(90, 60, 80), (130, 255, 255)],                                    # blue (diamonds)
    'c': [(40, 40, 60), (90, 255, 255)],                                     # green (clubs)
    's': [(0, 0, 0), (180, 60, 80)],                                         # dark/black (spades)
}

RANK_FIXES = {
    '0': 'T', 'O': 'T', 'o': 'T',
    '1O': 'T', '1o': 'T', '10': 'T',
    '60': '6', '6O': '6',
    'l': '1', 'I': '1',
    'A': 'A', 'K': 'K', 'Q': 'Q', 'J': 'J',
    'T': 'T', '9': '9', '8': '8', '7': '7',
    '6': '6', '5': '5', '4': '4', '3': '3', '2': '2',
}

VALID_RANKS = {'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'}


def detect_suit_hsv(img_region) -> str:
    """Detect card suit from colored region using HSV color analysis."""
    if img_region is None or img_region.size == 0:
        return 's'
    hsv = cv2.cvtColor(img_region, cv2.COLOR_BGR2HSV)

    scores = {}

    # Hearts: red hues (two ranges)
    r1 = cv2.inRange(hsv, np.array([0, 60, 80]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 60, 80]), np.array([180, 255, 255]))
    scores['h'] = cv2.countNonZero(r1) + cv2.countNonZero(r2)

    # Diamonds: blue
    scores['d'] = cv2.countNonZero(
        cv2.inRange(hsv, np.array([90, 60, 80]), np.array([130, 255, 255]))
    )

    # Clubs: green
    scores['c'] = cv2.countNonZero(
        cv2.inRange(hsv, np.array([40, 40, 60]), np.array([90, 255, 255]))
    )

    # Spades: dark pixels
    scores['s'] = cv2.countNonZero(
        cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 80]))
    )

    return max(scores, key=scores.get)


def normalize_rank(raw: str) -> str:
    """Normalize OCR-read card rank to standard format."""
    raw = raw.strip().upper()
    # Direct mapping
    if raw in RANK_FIXES:
        return RANK_FIXES[raw]
    # Try first character
    if len(raw) >= 1 and raw[0] in VALID_RANKS:
        return raw[0]
    # Numeric 10
    if raw in ('10', '1O', '1o'):
        return 'T'
    return raw


def detect_board_cards(img) -> list:
    """Detect the 5 board cards from the image (y=335-395, x=400-870)."""
    # Each card is approximately 94px wide across the 470px total
    cards = []
    y1, y2 = 335, 395
    total_x1, total_x2 = 400, 870
    card_width = (total_x2 - total_x1) // 5  # ~94px

    for i in range(5):
        cx1 = total_x1 + i * card_width
        cx2 = cx1 + card_width

        # Rank region: left portion of card
        rank_region = img[y1:y1 + 35, cx1:cx1 + 30]
        rank_text = read_region(img, cx1, y1, cx1 + 30, y1 + 35)
        rank = normalize_rank(rank_text) if rank_text else ''

        # Suit region: colored area of card
        suit_region = img[y1:y2, cx1:cx2]
        suit = detect_suit_hsv(suit_region)

        if rank and rank in VALID_RANKS:
            cards.append(f"{rank}{suit}")

    return cards


def _parse_amount(text: str) -> float:
    """Extract dollar amount from text string."""
    m = re.search(r'\$?\s*(\d+(?:\.\d+)?)', text.replace(',', ''))
    if m:
        return float(m.group(1))
    return 0.0


def detect_players_in_column(img, x_start: int, x_end: int, y_start: int = 920) -> list:
    """
    Detect player rows in a column by scanning horizontally for text.
    Returns list of dicts with keys: name, row_y, raw_text
    """
    h = img.shape[0]
    players = []
    row_height = 40  # approximate row height
    y = y_start

    while y + row_height <= h:
        text = read_region(img, x_start, y, x_end, y + row_height)
        if text and text.strip():
            players.append({'row_y': y, 'raw_text': text.strip()})
        y += row_height

    return players
