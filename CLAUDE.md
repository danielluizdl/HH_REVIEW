# HH_REVIEW — Agent Briefing

## Project Goal
Parse WPT Global poker screenshots → PokerStars hand history format (`.txt`).
One screenshot per hand. Output files go in `parsed/`. Validate against `ground_truth/`.

## Session Bootstrap (run these first, every session)

```bash
# 1. Pull latest
git pull origin master

# 2. Install deps (fast, idempotent)
pip install rapidocr-onnxruntime opencv-python pillow -q

# 3. Run full pipeline (parse all + score + update state.json)
python3 run_session.py
```

If `run_session.py` doesn't exist yet: see "Legacy Commands" below.

## Key Files

| File | Purpose |
|------|---------|
| `parse_hand_screenshot.py` | Main parser (OCR + image → PS format) |
| `src/hh_writer_ps.py` | PokerStars format writer |
| `src/scorer.py` | Compares predicted vs ground truth |
| `run_session.py` | One-shot pipeline (parse → score → commit) |
| `state.json` | Machine-readable session state (read this before PROGRESS.md) |
| `PROGRESS.md` | Human-readable progress log |
| `POKERSTARS_FORMAT.md` | Complete PokerStars format reference |
| `ground_truth/26-19.txt` | Only hand-crafted ground truth |
| `HAND HISTORY WPT/*.png` | 45 source screenshots |
| `parsed/*.txt` | Parser outputs |

## Quick Commands

```bash
# Parse single image
python3 parse_hand_screenshot.py "HAND HISTORY WPT/26-19.png" parsed/26-19.txt

# Score 26-19 against ground truth
python3 -c "
from src.scorer import print_score_report
gt = open('ground_truth/26-19.txt').read()
pred = open('parsed/26-19.txt').read()
print_score_report('26-19.png', pred, gt)
"

# List all images
ls "HAND HISTORY WPT/"

# Check state
cat state.json 2>/dev/null || echo "state.json not created yet"
```

## Known Blockers (do not retry without new approach)

| Image | Issue | Status |
|-------|-------|--------|
| `46-20.png` | 695px wide — narrow layout, preflop=0, wrong stacks | open |
| `33-8 55-10.png` | Only 32 lines — likely layout issue | open |
| `53-8.png` | 46 lines, all-anon table mostly fixed; **summary wrong** (result_entries empty → everyone shows as "folded before Flop"), missing HJ+STR players | partial |
| `44-12.png` | NEPTIN artifact player in Seat 9 | open (similarity < 0.7, dedup misses it) |

### 53-8.png remaining issues:
- `result_entries=[]` for all-anonymous tables → `_build_summary_seats` shows everyone as "folded before Flop"
- Fix: `_parse_results` needs to handle position-label rows (BTN/CO/etc) as results when names are absent
- HJ and STR missing from player_map: check if their action cells are in correct x-range for preflop column

## Unfixable OCR Errors (accepted ceiling)

- **Hand ID**: OCR adds 1 extra digit (19 vs 18 digits) — always wrong
- **Chinese characters**: visually similar chars swapped (e.g. 飚↔魇) — random
- **All-in amounts**: ±$0.03–0.05 rounding from BB accumulation

## Current Accuracy (26-19.png — only GT available)

```
overall: 84.8%
hand_id:        0%   ← unfixable OCR
timestamp:      OK
stakes:         OK
button_seat:    OK
players_stacks: 86%  ← 1/7 wrong (name OCR → stack lookup fails)
board:          OK
actions:        93%  ← 1/14 wrong (same name OCR issue)
total_pot:      OK
```

Real ceiling ≈ 91% (hand_id is always wrong; characters are unfixable).

## Improvements Roadmap (implement in order)

- [x] `CLAUDE.md` — this file
- [x] `state.json` — machine-readable session state (auto-updated after each parse)
- [x] `run_session.py` — one-shot pipeline: parse → score → update state → commit
- [x] Incremental parsing — built into `run_session.py` (skip if mtime unchanged)
- [x] Debug image cleanup — built into `run_session.py`
- [x] All-anonymous table support — position labels as names, dollar-format amounts/stacks
- [ ] Fix 53-8.png summary (result_entries for anon tables)
- [ ] Fix 46-20.png narrow layout (col_w=139 vs 256)
- [ ] Ground truth semi-auto — use vision (Read PNG) to correct parsed output manually
- [ ] Regression guard — already in `run_session.py`, needs more GTs to be useful

## Legacy Commands (if run_session.py missing)

```bash
# Parse all 45 images
for f in "HAND HISTORY WPT/"*.png; do
    base=$(basename "$f" .png)
    python3 parse_hand_screenshot.py "$f" "parsed/$base.txt" 2>/dev/null && echo "$base: OK"
done

# Commit parsed outputs
git add parsed/ state.json PROGRESS.md
git commit -m "Session: parse all images + update state"
git push -u origin HEAD:master
```

## Environment Notes

- No `ANTHROPIC_API_KEY` → `generate_ground_truths.py` vision mode disabled
- No `gh` CLI → use GitHub MCP tools (`mcp__github__*`) for PR/issue work
- Remote execution: always commit + push before session ends
- Session timeout: images take 60–90s each; parse all 45 sequentially = ~45 min
