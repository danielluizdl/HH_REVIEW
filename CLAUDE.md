# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WPT Global Poker Hand History Parser — converts WPT Global poker hand history **screenshots** (PNG) into **PokerStars-format** text using OCR and image analysis. Written in pure Python with OpenCV + RapidOCR.

## Commands

```bash
# Parse a single screenshot
python parse_hand_screenshot.py "HAND HISTORY WPT/26-19.png" [--bb 0.10]

# Parse all images and save output
for img in "HAND HISTORY WPT"/*.png; do
  python parse_hand_screenshot.py "$img" > "parsed/$(basename "$img" .png).txt"
done

# Score output against ground truth
python -c "
from src.scorer import print_score_report
pred = open('parsed/26-19.txt').read()
gt   = open('ground_truth/26-19.txt').read()
print_score_report('26-19', pred, gt)
"

# Generate ground truths via Claude Vision API (requires ANTHROPIC_API_KEY env var)
python generate_ground_truths.py
```

There is no test suite, linter config, or build system — the project is script-based.

## Architecture

### Image Layout

Every WPT Global screenshot is portrait ~988×2048 px, divided into five vertical columns that correspond to poker streets:

| Column | X range (px) | Content |
|--------|-------------|---------|
| 0 | 0–256 | Player names, stacks, positions (BLINDS & ANTE header) |
| 1 | 256–512 | Preflop actions |
| 2 | 512–768 | Flop actions |
| 3 | 768–1024 | Turn actions |
| 4 | 1024–1280 | River actions + results section |

Board cards appear as a horizontal strip at approximately y=335–395 (10–15% image height).

### Processing Pipeline (`parse_hand_screenshot.py`)

1. **Full-image OCR** — RapidOCR produces a flat list of `{x, y, x2, y2, text, conf}` items sorted by `y`.
2. **Metadata extraction** — Hand ID (y<80), timestamp (regex `YYYY-MM-DD HH:MM:SS`), board cards (column strips + HSV suit detection).
3. **Column parsing** — Each column is identified by its header text ("PRE-FLOP", "FLOP", etc.); OCR items within the column are grouped into action cells by y-gap (≥55 px gap = new cell).
4. **Blinds column** — Extracts ante/SB/BB/straddle amounts and posters.
5. **Results section** — Located 150 px above the first `+/−BB` item (winner name is at y≈1201).
6. **Player map** — Consolidated across all sources; fuzzy name matching handles truncated/OCR-mangled names.
7. **Seat assignment** — Maps position badges (BTN, SB, BB, STR, UTG, MP, HJ, CO) to seat numbers using `POS_TO_SEAT_7` / `POS_TO_SEAT_8`.
8. **Stack calculation** — `initial_stack = end_stack − net_result`; avoids relying solely on in-image stack text (OCR unreliable).
9. **Action translation** — Portuguese → English (`desistir`→`folds`, `passar`→`checks`, `aposta`→`bets`, `aumentar`→`raises`, `pagar`/`fazer call`→`calls`).
10. **Amount parsing** — European format: comma is decimal (`3,50`→`3.5`), period is thousands separator (`1.530`→`1530`). All-in calls are clamped to the player's remaining stack when OCR over-reads.
11. **Showdown / hand evaluation** — Hole cards extracted from image; best 5-card hand described via `src/hand_eval.py`.
12. **Output** — `src/hh_writer_ps.py:format_hand_history()` converts the `hand_data` dict to PokerStars-format text.

### Module Responsibilities

| File | Role |
|------|------|
| `parse_hand_screenshot.py` | Orchestrator — all parsing logic, produces `hand_data` dict |
| `src/hh_writer_ps.py` | Formats `hand_data` → PokerStars text; pure formatter, no parsing |
| `src/detectors.py` | `detect_suit_hsv()` (HSV thresholds for ♥♣♦♠), `normalize_rank()` (fixes OCR rank errors like `7G`→`7`, `80`→`8`) |
| `src/ocr_engine.py` | OCR wrapper with region upscaling (3× zoom for small text like card ranks) |
| `src/scorer.py` | `score_hand(predicted, ground_truth)` — compares 8 fields and returns per-field accuracy + overall % |
| `src/hand_eval.py` | Evaluates best 5-card hand from hole cards + board |
| `generate_ground_truths.py` | Batch ground-truth generator via Claude Vision API; outputs to `ground_truth/` |

### Key Constants & Quirks

- **OCR singleton**: `_ocr_instance` in `parse_hand_screenshot.py` — initialized lazily via `_get_ocr()` to avoid loading the model multiple times.
- **Suit HSV ranges** (in `src/detectors.py`): hearts=red (H<15), clubs=green (H=35–95), diamonds=blue (H=95–140), spades=dark. Suit sampling window is `[num_x-15 : num_x+65]` to capture left-edge suit symbols without bleeding into adjacent cards.
- **Seat maps**: 7-player `POS_TO_SEAT_7 = {BTN:4, SB:5, BB:6, STR:7, MP:1, HJ:2, CO:3}`; 8-player adds UTG:7.
- **PT_ACTION_WORDS**: words that must never be treated as player names during column parsing.

## Current State & Known Limitations

- **Accuracy: 84.3%** on the single validated image (`HAND HISTORY WPT/26-19.png` vs `ground_truth/26-19.txt`). This is the practical ceiling for OCR-only on this image.
- **Hard OCR errors** (unfixable without external dictionary): Chinese character substitutions (飚→魇), hand ID extra digit, rare 4-cent rounding on all-in call amounts.
- **Ground truth coverage**: only `26-19.txt` exists; 44 more images need ground truths generated via `generate_ground_truths.py` (requires `ANTHROPIC_API_KEY`).
- **Target**: >90% overall accuracy across all 45 images.

## Dependencies

No requirements file exists; install manually:
```bash
pip install rapidocr-onnxruntime opencv-python numpy anthropic
```
