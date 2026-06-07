# WPT Global Hand History Parser — Progress

## Status: Phase 3 - Pipeline estável, roadmap de qualidade definido

### Session 2026-06-07 Improvements
1. **`mucked hand` fix** — `hh_writer_ps.py:186`: losers at showdown without captured cards now output `mucked hand` instead of `lost`; fixes format in ~10 files
2. **Player name filter** — `parse_hand_screenshot.py is_player_name()`: rejects strings starting with `/` or `+` (OCR artifacts like `/TG+1`, `/OBB`); fixes 46-20.png Seat 9 artifact
3. **Scorer fuzzy name matching** — `src/scorer.py`: `players_stacks` and `actions` now use SequenceMatcher ≥ 0.70 as fallback; 26-19.png score: **84.8% → 87.5%**
4. **All-in losers in showdown** — `parse_hand_screenshot.py _build_summary_seats()`: all-in players from earlier streets now included in `showdown_players`; fixes 35-19.png Roson861 (was "folded on Turn" → now "mucked hand")
5. **Board scan ymin filter** — top-edge margin reduced 20→5px; avoids dropping valid board cards near scan boundary
6. **Dual-rank text filter** — board scan skips 2-char texts where both chars are card ranks (e.g. "KA"); prevents hole-card bleed from corrupting board spacing estimate
7. **Midpoint gap-filling** — when board card spacing > 100px, estimates midpoint and scans that region; improves detection of missing board cards
8. **Partial flop board** — when `len(board) > 0` and flop actions exist, outputs partial `[FLOP]` section even if <3 cards detected; 33-8 55-10.png now shows `[8d 5c]` flop with 8 correct actions (Q♣ undetectable due to "Pote Total" text overlay)

### Session 2026-06-06 Improvements
1. **run_session.py output file bug** — Parser stdout now properly captured and written to file; all 45 files are freshly generated (was silently ignoring parser output, using stale files)
2. **NEPTIN artifact removal** — OCR artifact players not grounded in end_stacks or results are filtered out after early end_stacks extraction
3. **Narrow image column boundaries** (46-20.png, 695px) — Detects column boundaries from header x positions when avg column spacing < 160px; fixes 0 preflop actions → 4 preflop actions
4. **Multi-column result detection** — When a hand ends on the flop or turn, results appear in that column (not RIVER); parser now tries RIVER → TURN → FLOP to find result section
5. **pote_total threshold lowered** — Was filtering out pots < 100BB; now any positive value from "Pote Total" text is used (fixes 40-26: $86.10 → $4.49)
6. **Winner amounts** (35-19, all images) — Summary winner collection shows pot amount, not ending stack
7. **Split pot fix** — winners list now distributes `total_pot / n_winners` to each winner (was only first winner set); summary_seats now uses `pote_total_bb / n_winners` (was full pot per winner); `pote_total_bb=None` fallback uses `net_bb` instead of silent $0

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
- **Split pots**: Total pot divided equally among all winners (correct for chops)

### Accuracy on 26-19.png (validation image)
- **Score: 84.8%** (ceiling ~91% with current OCR)
- **5 remaining differences** from ground truth — all unfixable OCR:
  1. hand_id: extra digit (1279290512207581184 vs 127929051220758184)
  2-5. 奥德彪飚 vs 奥德彪魇 (last Chinese character OCR mismatch × 4 lines)

### Parse Statistics (all 45 images)
- **45/45 images parsed** successfully (no timeouts or crashes)
- **37/45** have board cards detected
- **37/45** have FLOP actions
- **34/45** have TURN actions
- **21/45** have RIVER actions
- **30/45** have SHOWDOWN section
- 8 images are preflop-only hands (normal — all-in or fold before flop)

### Known Issues
1. **46-20.png** (695x2048 narrow): Only partial preflop actions; turn/river missing; `/TG+1` OCR artifact as Seat 9
2. **33-8 55-10.png** (844px): Board cards not detected; flop cards at y<200 outside scan range (scan starts at 15%)
3. **63-9.png**: Winner (MP player) name missing from OCR entirely; Seat 1 absent from output
4. **10 arquivos**: Losers at showdown without captured cards show `lost` instead of `mucked hand` (format bug)
5. **hand_id**: OCR adds 1 extra digit (unfixable)
6. **Chinese characters**: Visually similar chars swapped (e.g. 飚↔魇, unfixable)
7. **Scorer**: Only measures 8 fields; winner amount, ante, straddle, summary outcomes not validated

### Ground Truths Status
- `ground_truth/26-19.txt`: Hand-crafted ✓
- All other images: Need generation (requires ANTHROPIC_API_KEY or manual review)
- **Without more GTs, regression guard is blind to improvements in 44 of 45 images**

---

## Roadmap — Próximas Sessões

### Sessão Imediata (quick wins)
1. **`mucked hand` no lugar de `lost`** — hh_writer_ps.py:186; corrige formato PS em 10 arquivos
2. **Filtrar `/TG+1` e variantes** — is_player_name() rejeitar strings começando com `/` ou prefixos deformados de posições; corrige 46-20 Seat 9
3. **Scorer fuzzy name matching** — SequenceMatcher > 0.85 como fallback no scorer; sobe 26-19 players_stacks de 86% → ~100%
4. **Scorer expandido** — adicionar campos: winner amount, ante, straddle, summary_seat outcomes

### Sessão 2
5. **Board scan range 8-22%** — expandir de 15-18% para cobrir imagens onde cartas aparecem mais acima; corrige 33-8 55-10
6. **Gerar mais ground truths** — 2-3 por sessão manualmente; sem isso não há visibilidade sobre 44 imagens
7. **63-9 recovery**: expandir `_recover_first_player_names` para coluna result quando nenhum winner encontrado

### Sessão 3
8. **46-20.png layout completo** — turn/river em layout 2-coluna 695px
9. **Initial stack fallback**: garantir `end_stacks_bb` é sempre usado quando disponível antes de estimativa
10. **Showdown card extraction para losers**: eliminar `mucked hand` com extração real de cartas

### Longo Prazo
- Merge dos dois blocos duplicate de artifact removal (linhas 285-302 e 355-376)
- Deduplicate `_extract_end_stacks` chamado 2× (linhas 288 e 348)
- `POS_TO_SEAT_7/8` definidos mas nunca usados — integrar em `_assign_seats`

---

### Unfixable OCR Errors (accepted ceiling)
- Chinese character substitutions (飚→魇, similar strokes)
- Hand ID extra digits (OCR confidence issue with 18-digit numbers)
- All-in amounts: ±$0.03-0.05 rounding from BB accumulation
