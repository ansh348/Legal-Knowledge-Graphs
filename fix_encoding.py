#!/usr/bin/env python3
"""
fix_encoding.py

1) Repairs existing broken JSON files (reads as cp1252, rewrites as clean UTF-8)
2) Shows the preventive patch to apply to run_iltur.py

Usage:
    python fix_encoding.py --dir iltur_graphs
    python fix_encoding.py --dir iltur_graphs --dry-run   # preview only
"""

import json
import argparse
from pathlib import Path

# The 10 known broken files (add more if needed)
KNOWN_BROKEN = [
    "1950_52", "1953_65", "1957_146", "1958_42", "1958_75",
    "1965_133", "1965_201", "1965_28", "1965_54", "1965_7",
]


def repair_file(filepath: Path, dry_run: bool = False) -> bool:
    """
    Try to read a broken JSON file using cp1252 fallback,
    then rewrite it as valid UTF-8.
    """
    raw = filepath.read_bytes()

    if len(raw) == 0:
        print(f"  SKIP  {filepath.name} — empty file (re-extract this one)")
        return False

    # Step 1: Try UTF-8 first (might already be fine)
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
        print(f"  OK    {filepath.name} — already valid UTF-8")
        return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    # Step 2: Decode as Windows-1252 (the actual encoding of the bad bytes)
    #   0x92 = right single quote, 0x93 = left double quote,
    #   0x97 = em dash, 0xa3 = pound sign, etc.
    try:
        text = raw.decode("cp1252")
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        # Step 3: Last resort — replace bad bytes
        text = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e2:
            print(f"  FAIL  {filepath.name} — can't parse even with fallback: {e2}")
            return False

    if dry_run:
        print(f"  WOULD FIX  {filepath.name} ({len(raw)} bytes)")
        return True

    # Rewrite as clean UTF-8 with ensure_ascii=True to be safe
    clean_json = json.dumps(data, indent=2, ensure_ascii=False)
    filepath.write_text(clean_json, encoding="utf-8")
    print(f"  FIXED {filepath.name} ({len(raw)} → {len(clean_json.encode('utf-8'))} bytes)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix encoding in broken IL-TUR JSON files")
    parser.add_argument("--dir", type=str, default="iltur_graph", help="Path to graph output directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--all", action="store_true", help="Scan ALL files, not just known broken ones")
    args = parser.parse_args()

    graph_dir = Path(args.dir)
    if not graph_dir.exists():
        print(f"Directory not found: {graph_dir}")
        return

    if args.all:
        files = sorted(graph_dir.glob("*.json"))
        files = [f for f in files if f.name != "checkpoint.json"]
    else:
        files = [graph_dir / f"{cid}.json" for cid in KNOWN_BROKEN]
        files = [f for f in files if f.exists()]

    print(f"{'DRY RUN — ' if args.dry_run else ''}Checking {len(files)} files in {graph_dir}/\n")

    fixed = 0
    failed = 0
    for f in files:
        if repair_file(f, dry_run=args.dry_run):
            fixed += 1
        else:
            failed += 1

    print(f"\nDone: {fixed} fixed, {failed} failed")

    # Show the preventive patch
    print("\n" + "=" * 60)
    print("PREVENTIVE FIX — apply to run_iltur.py")
    print("=" * 60)
    print("""
Add this helper function near the top of run_iltur.py:

    def sanitize_text(text: str) -> str:
        \"\"\"Fix Windows-1252 chars that sneak through IL-TUR data.\"\"\"
        # Common cp1252 → unicode replacements
        replacements = {
            '\\u0092': '\\u2019',  # right single quote
            '\\u0093': '\\u201c',  # left double quote
            '\\u0094': '\\u201d',  # right double quote
            '\\u0097': '\\u2014',  # em dash
            '\\u0096': '\\u2013',  # en dash
            '\\u0091': '\\u2018',  # left single quote
            '\\u0085': '\\u2026',  # ellipsis
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

Then in extract_one(), sanitize BEFORE passing to the extractor:

    text = sanitize_text(text)   # <-- add this line
    graph = await extractor.extract(text=text, case_id=case_id)
""")


if __name__ == "__main__":
    main()
