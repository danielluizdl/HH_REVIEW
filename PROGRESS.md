# WPT Global Hand History Parser — Progress

## Status: Phase 2 - All 45 images parsing successfully

### Files Built
- `parse_hand_screenshot.py` — Main parser: OCR + image analysis → PokerStars format
- `src/hh_writer_ps.py` — PokerStars format writer
- `src/scorer.py` — Scoring: compares predicted vs ground truth
- `generate_ground_truths.py` — Vision API batch ground-truth generator (needs ANTHROPIC_API_KEY)
- `ground_truth/26-19.txt` — Hand-crafted ground truth for validation image
- `POKERSTARS_FORMAT.md` — Complete PokerStars hand history format reference

### Parser Capabilities
- **Amounts**: European decimal (3,50BB → 3.5) + thousands separator (1.530BB → 1530)
- **Board cards**: 5 column strips; 3x scaled OCR; suit via HSV color analysis
  - Adaptive y-range: tries 15% then extends to 18% if <5 cards found
  - Sub-region fallback: scans left/right halves + estimated positions when <5 cards detected
  - J/I OCR fix: 'I' and 'IJ' normalize to 'J' in board context
- **Actions**: Portuguese→English (Desistir, Passar, Aposta, Aumentar, Pagar, Fazer call)
- **All-in call amounts**: Stack-tracking fix — when OCR amount > remaining stack, replace with computed
- **Player recovery**: Focused OCR on pre-action area captures names hidden behind WINNER banner
  - Crops from act_y-15 to FIRST ACTION WORD y (not first item) to handle amount-before-name layout
- **Anonymous players**: Badge-based cell splitting; position→name mapping for repeated appearances
- **OCR artifacts**: Deduplication removes near-duplicate names (>0.7 SequenceMatcher similarity)
- **Seat assignment**: Overflow protection prevents infinite loop when >8 players detected

### Accuracy on 26-19.png (validation image)
- **4 differences** from ground truth (all known unfixable OCR errors):
  1. hand_id: extra digit (1279290512207581184 vs 127929051220758184)
  2. 奥德彪飚 vs 奥德彪魇 (last Chinese character OCR mismatch)
  3-4. Same character appearing in summary

### Parse Statistics (all 45 images)
- **45/45 images parsed** successfully (no timeouts or crashes)
- **37/45** have board cards detected
- **37/45** have FLOP actions
- **34/45** have TURN actions  
- **21/45** have RIVER actions
- **30/45** have SHOWDOWN section
- 8 images are preflop-only hands (normal — all-in or fold before flop)

### Known Issues
1. **46-20.png** (695x2048 narrow): Different image layout causes missing preflop actions
2. **Board suits**: Sometimes all detected as 'h' (hearts) — HSV suit detection needs tuning
3. **Stacks**: Some exact values off ($0.04 for OCR artifact players that slip through dedup)
4. **Summary amounts**: `collected` line sometimes shows end stack instead of net win

### Ground Truths Status
- `ground_truth/26-19.txt`: Hand-crafted ✓
- All other images: Need generation (requires ANTHROPIC_API_KEY or manual review)

### Next Steps
1. Generate ground truths for all 45 images (vision analysis of each screenshot)
2. Run scorer: `python -c "from src.scorer import ..."`
3. Fix board suit detection (HSV thresholds for different image brightness)
4. Fix 46-20.png narrow image layout parsing
5. Target: match ground truths on all structural elements (actions, board, players)

### Unfixable OCR Errors (accepted)
- Chinese character substitutions (飚→魇, similar strokes)
- Hand ID extra digits (OCR confidence issue with 18-digit numbers)
- All-in amounts: ±$0.03-0.05 rounding from BB accumulation
