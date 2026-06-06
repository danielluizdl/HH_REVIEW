"""
One-shot session pipeline:
  1. Parse all images (incremental: skip if output is newer than image + parser)
  2. Score against all available ground truths
  3. Detect regressions vs previous session
  4. Update state.json
  5. Clean up debug images
  6. Print summary

Usage:
    python3 run_session.py              # incremental (skip unchanged)
    python3 run_session.py --force      # re-parse everything
    python3 run_session.py --score-only # just score, no parse
"""

import os
import sys
import glob
import json
import subprocess
import shutil
from datetime import date
from src.scorer import score_hand


IMG_DIR = "HAND HISTORY WPT"
OUT_DIR = "parsed"
GT_DIR = "ground_truth"
PARSER = "parse_hand_screenshot.py"
STATE_FILE = "state.json"

DEBUG_PATTERNS = [
    "debug_*.png", "scan_*.png", "region_*.png",
    "hand_id_row.png", "header_detail.png",
]


def _git_commit(msg: str) -> bool:
    try:
        subprocess.run(["git", "add", OUT_DIR, STATE_FILE, "PROGRESS.md"], check=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode == 0:
            print("  No changes to commit.")
            return True
        subprocess.run(["git", "commit", "-m", msg], check=True)
        subprocess.run(["git", "push", "-u", "origin", "HEAD:master"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Git error: {e}")
        return False


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        return "unknown"


def _should_parse(img_path: str, out_path: str, force: bool) -> bool:
    if force or not os.path.exists(out_path):
        return True
    img_mtime = os.path.getmtime(img_path)
    out_mtime = os.path.getmtime(out_path)
    parser_mtime = os.path.getmtime(PARSER)
    return img_mtime > out_mtime or parser_mtime > out_mtime


def parse_all(force: bool = False) -> dict:
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    skipped = 0

    print(f"\n{'='*50}")
    print(f"PARSE: {len(imgs)} images ({'force' if force else 'incremental'})")
    print("="*50)

    for img_path in imgs:
        base = os.path.basename(img_path).replace(".png", "")
        out_path = os.path.join(OUT_DIR, f"{base}.txt")

        if not _should_parse(img_path, out_path, force):
            skipped += 1
            with open(out_path) as f:
                lines = [l for l in f if l.strip()]
            results[base] = {"lines": len(lines), "status": "skipped"}
            continue

        try:
            proc = subprocess.run(
                ["python3", PARSER, img_path, out_path],
                capture_output=True, text=True, timeout=150
            )
            if os.path.exists(out_path):
                with open(out_path) as f:
                    lines = [l for l in f if l.strip()]
                n = len(lines)
                results[base] = {"lines": n, "status": "parsed"}
                print(f"  {base}: {n} lines")
            else:
                results[base] = {"lines": 0, "status": "error", "stderr": proc.stderr[-200:]}
                print(f"  {base}: ERROR (no output)")
        except subprocess.TimeoutExpired:
            results[base] = {"lines": 0, "status": "timeout"}
            print(f"  {base}: TIMEOUT")

    parsed = sum(1 for v in results.values() if v["status"] in ("parsed", "skipped"))
    print(f"\n  Done: {parsed}/{len(imgs)} OK, {skipped} skipped (up to date)")
    return results


def score_all(prev_scores: dict) -> dict:
    gts = sorted(glob.glob(os.path.join(GT_DIR, "*.txt")))
    if not gts:
        print("\nSCORE: No ground truths available.")
        return {}

    print(f"\n{'='*50}")
    print(f"SCORE: {len(gts)} ground truth(s)")
    print("="*50)

    scores = {}
    regressions = []

    for gt_path in gts:
        name = os.path.basename(gt_path).replace(".txt", "")
        pred_path = os.path.join(OUT_DIR, f"{name}.txt")

        if not os.path.exists(pred_path):
            print(f"  {name}: no parsed output")
            continue

        gt_text = open(gt_path, encoding="utf-8").read()
        pred_text = open(pred_path, encoding="utf-8").read()
        s = score_hand(pred_text, gt_text)
        scores[name] = s

        overall = s["overall"]
        prev = prev_scores.get(name, {}).get("overall", 0)
        delta = overall - prev

        if delta < -1.0:
            regressions.append((name, prev, overall))
            flag = " ⚠ REGRESSION"
        elif delta > 0.5:
            flag = f" ↑ (+{delta:.1f})"
        else:
            flag = ""

        print(f"  {name}: {overall:.1f}%{flag}")
        for field, val in s.items():
            if field == "overall":
                continue
            status = "OK" if val >= 0.99 else f"{val*100:.0f}%"
            if val < 0.99:
                print(f"    {field:<20}: {status}")

    if regressions:
        print("\n  !! REGRESSIONS DETECTED — review before committing !!")
        for name, prev, now in regressions:
            print(f"     {name}: {prev:.1f}% → {now:.1f}%")

    return scores


def cleanup_debug():
    removed = 0
    for pattern in DEBUG_PATTERNS:
        for f in glob.glob(pattern):
            os.remove(f)
            removed += 1
    if removed:
        print(f"\nCLEANUP: Removed {removed} debug image(s)")


def update_state(parse_results: dict, scores: dict):
    prev_state = _load_state()

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    parsed_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.txt")))
    gts = sorted(glob.glob(os.path.join(GT_DIR, "*.txt")))

    parse_stats = {}
    for base, res in parse_results.items():
        out_path = os.path.join(OUT_DIR, f"{base}.txt")
        if not os.path.exists(out_path):
            continue
        with open(out_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        parse_stats[base] = {
            "lines": len(lines),
            "has_flop": any("FLOP" in l for l in lines),
            "has_turn": any("TURN" in l for l in lines),
            "has_river": any("RIVER" in l for l in lines),
            "has_showdown": any("SHOWDOWN" in l for l in lines),
            "has_board": any("Board [" in l for l in lines),
            "seat9_overflow": any("Seat 9:" in l for l in lines),
        }

    n_flop = sum(1 for v in parse_stats.values() if v["has_flop"])
    n_turn = sum(1 for v in parse_stats.values() if v["has_turn"])
    n_river = sum(1 for v in parse_stats.values() if v["has_river"])
    n_show = sum(1 for v in parse_stats.values() if v["has_showdown"])
    n_seat9 = sum(1 for v in parse_stats.values() if v["seat9_overflow"])

    state = {
        "session_date": str(date.today()),
        "parser_commit": _git_commit_hash(),
        "images_total": len(imgs),
        "images_parsed": len(parsed_files),
        "ground_truths_available": len(gts),
        "parse_summary": {
            "with_flop": n_flop,
            "with_turn": n_turn,
            "with_river": n_river,
            "with_showdown": n_show,
            "with_seat9_overflow": n_seat9,
        },
        "scores": scores,
        "known_issues": prev_state.get("known_issues", [
            {"file": "46-20.png", "issue": "narrow_layout_695px", "status": "open"},
            {"file": "33-8 55-10.png", "issue": "short_output_32_lines", "status": "open"},
            {"file": "53-8.png", "issue": "very_short_9_lines", "status": "open"},
            {"file": "44-12.png", "issue": "artifact_player_NEPTIN_seat9", "status": "open"},
        ]),
        "improvements_done": prev_state.get("improvements_done", ["CLAUDE.md", "state.json", "run_session.py"]),
        "improvements_pending": prev_state.get("improvements_pending", [
            "incremental_parsing", "gt_semi_auto", "regression_guard", "debug_cleanup"
        ]),
        "next_steps": prev_state.get("next_steps", [
            "Generate more ground truths manually (1-2 per session)",
            "Fix 46-20.png narrow layout (col_w=139 vs 256)",
            "Fix 53-8.png (investigate why only 9 lines)",
        ]),
        "parse_details": parse_stats,
    }

    _save_state(state)
    print(f"\nSTATE: state.json updated")


def print_summary(parse_results: dict, scores: dict):
    total = len(parse_results)
    ok = sum(1 for v in parse_results.values() if v["status"] in ("parsed", "skipped") and v["lines"] > 20)
    short = [(k, v["lines"]) for k, v in parse_results.items() if v["lines"] <= 20]

    print(f"\n{'='*50}")
    print("SUMMARY")
    print("="*50)
    print(f"  Parsed: {ok}/{total} OK")
    if short:
        print(f"  Short outputs (<20 lines):")
        for name, n in short:
            print(f"    {name}: {n} lines")
    if scores:
        for name, s in scores.items():
            print(f"  Score {name}: {s['overall']:.1f}%")
    print()


def main():
    force = "--force" in sys.argv
    score_only = "--score-only" in sys.argv

    prev_state = _load_state()
    prev_scores = prev_state.get("scores", {})

    if score_only:
        parse_results = {}
        for p in glob.glob(os.path.join(OUT_DIR, "*.txt")):
            name = os.path.basename(p).replace(".txt", "")
            with open(p, encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            parse_results[name] = {"lines": len(lines), "status": "skipped"}
    else:
        parse_results = parse_all(force=force)

    scores = score_all(prev_scores)
    cleanup_debug()
    update_state(parse_results, scores)
    print_summary(parse_results, scores)

    # Auto-commit results
    commit_msg = f"Session {date.today()}: parse all images, score {len(scores)} GT(s)"
    _git_commit(commit_msg)


if __name__ == "__main__":
    main()
