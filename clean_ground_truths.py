"""Strip reasoning preamble from Claude Vision-generated ground truth files."""
from pathlib import Path

for gt_path in sorted(Path('ground_truth').glob('*.txt')):
    text = gt_path.read_text()
    lines = text.strip().split('\n')

    # Find first line starting with 'PokerStars Hand #'
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('PokerStars Hand #'):
            start = i
            break

    if start is None or start == 0:
        continue  # already clean

    # Extract from PS line onwards
    cleaned_lines = lines[start:]
    # Remove trailing markdown code block markers
    while cleaned_lines and cleaned_lines[-1].strip() in ('', '```'):
        cleaned_lines.pop()

    cleaned = '\n'.join(cleaned_lines) + '\n'
    gt_path.write_text(cleaned, encoding='utf-8')
    print(f'Cleaned {gt_path.name} (removed {start} preamble lines)')

print('Done')
