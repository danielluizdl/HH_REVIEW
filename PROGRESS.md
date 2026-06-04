# WPT Global Hand History Parser — Progress

## Status: Phase 1 complete (single-image OCR pipeline)

### Files Built
- `parse_hand_screenshot.py` — Main parser: OCR + image analysis → PokerStars format
- `src/hh_writer_ps.py` — PokerStars format writer
- `src/scorer.py` — Scoring: compares predicted vs ground truth
- `generate_ground_truths.py` — Vision API batch ground-truth generator (needs ANTHROPIC_API_KEY)
- `ground_truth/26-19.txt` — Hand-crafted ground truth for validation image

### Parser Capabilities
- **Amounts**: European decimal (3,50BB → 3.5) + thousands separator (1.530BB → 1530)
- **Board cards**: 5 column strips at y=10-15% image height; rank OCR at 3× zoom; suit via HSV color analysis
  - Suit sampling: `[num_x-15 : num_x+65]` — captures left-edge suit symbols (6c) without adjacent-card bleed (9s)
  - Suits: hearts=red(H<15), clubs=green(H=35-95), diamonds=blue(H=95-140), spades=dark
- **Actions**: Portuguese→English (Desistir, Passar, Aposta, Aumentar, Pagar, Fazer call)
- **All-in call amounts**: Stack-tracking fix — when OCR amount > remaining stack, replace with computed remaining
- **Stacks**: initial = end_stack − net; fuzzy name matching for truncated OCR
- **Straddle**: detected from STR badge in BLINDS column
- **Results section**: found 150px before first +/−BB item (captures winner name at y=1201)
- **Summary seats**: SB/BB/straddle posters get "folded before Flop" (not "didn't bet")

### Accuracy on 26-19.png (validation image)
- Overall: **84.3%** (maximum achievable with OCR-only approach)
- Fields: timestamp OK, stakes OK, button OK, board OK, total_pot OK
- Unfixable OCR errors:
  - hand_id: extra digit (1279290512207581184 vs 127929051220758184)
  - 奥德彪飚 vs 奥德彪魇 (Chinese character OCR error, no dictionary)
  - dLzinN river all-in call: $18.34 vs $18.30 (4-cent rounding from amount accumulation)

### Next Steps (blocked)
1. **ANTHROPIC_API_KEY required**: Run `generate_ground_truths.py` to create ground truths for all 45 images
2. Run parser on all 45 images: `for img in "HAND HISTORY WPT"/*.png; do python parse_hand_screenshot.py "$img" > "parsed/$(basename $img .png).txt"; done`
3. Score each: `python -c "from src.scorer import print_score_report; ..."`
4. Identify common failure patterns across images
5. Fix systematic errors (rank normalization, action parsing edge cases)
6. Target: >90% overall accuracy across all images

### Image Set
- 45 PNG screenshots in `HAND HISTORY WPT/`
- Ground truths needed: 44 more (only 26-19.txt exists)

### Known OCR Issues (hard to fix without dictionary)
- Chinese character substitutions (飚→魇, similar strokes)
- Hand ID extra digits (OCR confidence issue)
- European amounts: OCR sometimes reads "313BB" for "183BB" (fixed by stack tracking)
- Rank OCR errors: '7G'→'2', '60'→'6', '80'→'8', '0'→'9' (all normalized)
