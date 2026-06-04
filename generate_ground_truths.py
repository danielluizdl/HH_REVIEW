"""
Generate ground truth hand histories for all WPT Global screenshot images
using Claude Vision API (claude-sonnet-4-6). Skips already-processed images.
"""

import anthropic
import base64
import os
from pathlib import Path

client = anthropic.Anthropic()
images_dir = Path("HAND HISTORY WPT")
output_dir = Path("ground_truth")
output_dir.mkdir(exist_ok=True)

VISION_PROMPT = """This is a WPT Global poker hand history screenshot. Extract the complete hand history and format it EXACTLY in PokerStars format.

The image has 5 columns:
- Column 1 (BLINDS & ANTE, x=0-255): Player names, positions, stacks, antes/blinds
- Column 2 (PRE-FLOP, x=256-511): Pre-flop actions
- Column 3 (FLOP, x=512-767): Flop actions
- Column 4 (TURN, x=768-1023): Turn actions
- Column 5 (RIVER, x=1024-1279): River actions + showdown results

Board cards are shown at top (y=335-395). Hand ID at top (y=15-50).

Output ONLY the PokerStars hand history text, nothing else. Start with PokerStars Hand #...

Important:
- Preserve Chinese/Asian player names exactly
- Use $ for amounts
- Card format: rank+suit (Ah, Kd, Qc, Js, Ts, 9h, etc.)
- Ranks: A,K,Q,J,T,9,8,7,6,5,4,3,2
- Suits: h=hearts(red), d=diamonds(blue), c=clubs(green), s=spades(dark)
- The amounts shown in the image are in BB (big blinds). Convert to dollars.
  For example, if 1 BB = $0.10, then 10 BB = $1.00
- Look at the stakes (SB/BB amounts) to determine the BB dollar value
- Include ALL actions: antes, blinds, straddle if present, all street actions
- For raises: format as "raises $X to $Y" where X is the increment and Y is the total
- For calls: format as "calls $X" where X is the amount called
- Include SHOWDOWN section with hole cards shown
- Include SUMMARY section with all seats

The BLINDS & ANTE column shows:
- "Ante X BB" = total ante pool (divide by number of players for per-player ante)
- Player name + SB badge + amount = that player posts small blind
- Player name + BB badge + amount = that player posts big blind
- Player name + STR badge + amount = that player posts straddle

The results section (bottom of RIVER column) shows each player's net profit/loss in BB.
Winner has positive amount shown, losers have negative (red) amount."""


def process_image(img_path: Path, gt_path: Path):
    with open(img_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_data
            }},
            {"type": "text", "text": VISION_PROMPT}
        ]}]
    )

    gt_text = response.content[0].text
    gt_path.write_text(gt_text, encoding="utf-8")
    print(f"Generated ground truth for {img_path.name}")
    return gt_text


def main():
    images = sorted(images_dir.glob("*.png"))
    print(f"Found {len(images)} images")

    for img_path in images:
        gt_path = output_dir / (img_path.stem + ".txt")
        if gt_path.exists():
            print(f"Skipping {img_path.name} (already done)")
            continue
        try:
            process_image(img_path, gt_path)
        except Exception as e:
            print(f"ERROR processing {img_path.name}: {e}")


if __name__ == "__main__":
    main()
