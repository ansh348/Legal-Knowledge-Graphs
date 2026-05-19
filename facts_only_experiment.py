#!/usr/bin/env python3
"""
Steps 1-2: Extract facts-only sections from 50 ILDC cases.
"""

import json
import re
import sys
import random
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from datasets import load_dataset

GRAPH_DIR = Path("graphs_enriched_v2")

# Patterns that signal the START of court reasoning (end of facts)
REASONING_START_PATTERNS = [
    r"heard\s+(?:the\s+)?learned\s+counsel",
    r"we\s+have\s+heard",
    r"having\s+heard",
    r"the\s+submissions?\s+of\s+(?:the\s+)?(?:learned\s+)?counsel",
    r"the\s+contentions?\s+raised",
    r"it\s+was\s+(?:submitted|contended|argued|urged)",
    r"the\s+learned\s+counsel\s+for\s+the\s+respondent\s+(?:argued|submitted|contended)",
    r"the\s+learned\s+counsel\s+for\s+the\s+appellant\s+(?:argued|submitted|contended)",
    r"learned\s+counsel\s+(?:for|appearing)\s+(?:the\s+)?(?:appellant|respondent|petitioner)",
    r"mr\.\s+\w+,?\s+(?:the\s+)?learned\s+(?:senior\s+)?counsel",
    r"shri\.?\s+\w+,?\s+(?:the\s+)?learned\s+counsel",
    r"(?:the\s+)?(?:main|principal|primary)\s+(?:contention|submission|argument)\s+(?:of|is|was)",
    r"we\s+(?:shall\s+)?(?:now\s+)?(?:proceed\s+to\s+)?(?:consider|examine|deal\s+with)",
    r"the\s+(?:only|short|main)\s+(?:question|point|issue)\s+(?:that|which|for)",
    r"(?:on\s+)?(?:a\s+)?careful\s+(?:consideration|perusal|examination|reading)",
    r"the\s+question\s+(?:that\s+)?arises\s+(?:for|is)",
    r"before\s+we\s+(?:proceed|deal|consider)",
    r"we\s+have\s+(?:carefully\s+)?(?:considered|perused|examined|gone\s+through)",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REASONING_START_PATTERNS]


def extract_facts_section(text):
    """Extract facts section from Indian SC judgment text.

    Returns facts text using the heuristic:
    - Everything before the first reasoning-start pattern
    - OR first 40% of text
    - Whichever is shorter
    """
    max_chars = int(len(text) * 0.4)

    # Find earliest reasoning-start pattern
    earliest_pos = len(text)
    earliest_pattern = None
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match and match.start() > 500:  # Must be at least 500 chars in (skip headers)
            if match.start() < earliest_pos:
                earliest_pos = match.start()
                earliest_pattern = match.group()

    # Take the shorter of: text before reasoning marker, or 40% of text
    if earliest_pos < len(text):
        cut_point = min(earliest_pos, max_chars)
    else:
        cut_point = max_chars

    # Try to cut at a sentence boundary
    facts = text[:cut_point]
    last_period = facts.rfind(". ")
    if last_period > cut_point * 0.8:  # Don't cut too much
        facts = facts[:last_period + 1]

    return facts.strip(), earliest_pattern


def sanitize_case_id(case_id):
    case_id = str(case_id or "").strip()
    case_id = case_id.replace("/", "_").replace("\\", "_")
    case_id = re.sub(r"[^0-9A-Za-z._-]+", "_", case_id)
    case_id = re.sub(r"_+", "_", case_id).strip("_")
    return case_id or "case"


def main():
    print("Loading HuggingFace dataset...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe", split="single_train")

    # Build lookup
    raw_data = {}
    for row in ds:
        cid = sanitize_case_id(row.get("id", ""))
        raw_data[cid] = {
            "text": row.get("text", ""),
            "label": row.get("label", None),
        }

    # Get cases that also exist in graphs_enriched_v2/
    graph_ids = {f.stem for f in GRAPH_DIR.glob("*.json") if f.name != "checkpoint.json"}
    available = sorted(set(raw_data.keys()) & graph_ids)
    print(f"Available cases (in both HF and graphs): {len(available)}")

    # Sample 50
    rng = random.Random(42)
    sample_ids = rng.sample(available, 50)

    # Extract facts
    results = []
    for cid in sample_ids:
        text = raw_data[cid]["text"]
        label = raw_data[cid]["label"]
        facts, split_pattern = extract_facts_section(text)

        results.append({
            "case_id": cid,
            "label": label,
            "full_length": len(text),
            "facts_length": len(facts),
            "pct_kept": len(facts) / len(text) * 100 if text else 0,
            "split_pattern": split_pattern,
            "facts_text": facts,
        })

    # =========================================================================
    # STEP 2: Show examples and stats
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"FACTS EXTRACTION RESULTS (n=50)")
    print(f"{'='*70}")

    lengths = [(r["full_length"], r["facts_length"], r["pct_kept"]) for r in results]
    import statistics
    avg_full = statistics.mean([l[0] for l in lengths])
    avg_facts = statistics.mean([l[1] for l in lengths])
    avg_pct = statistics.mean([l[2] for l in lengths])

    print(f"  Avg full text:    {avg_full:.0f} chars")
    print(f"  Avg facts only:   {avg_facts:.0f} chars")
    print(f"  Avg % kept:       {avg_pct:.1f}%")

    # Split pattern distribution
    pattern_counts = {}
    for r in results:
        p = r["split_pattern"] or "(40% cutoff)"
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
    print(f"\n  Split triggers:")
    for p, c in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {p[:60]:<60} {c:>3}")

    # All 50 cases
    print(f"\n  {'Case':<14} {'Full':>6} {'Facts':>6} {'%':>5} {'Split trigger'}")
    print(f"  {'-'*70}")
    for r in sorted(results, key=lambda x: x["case_id"]):
        trigger = (r["split_pattern"] or "(40%)") [:35]
        print(f"  {r['case_id']:<14} {r['full_length']:>6} {r['facts_length']:>6} {r['pct_kept']:>5.1f} {trigger}")

    # 3 examples
    print(f"\n{'='*70}")
    print(f"3 EXAMPLE CASES")
    print(f"{'='*70}")

    for r in results[:3]:
        print(f"\n  --- {r['case_id']} (full={r['full_length']}, facts={r['facts_length']}, {r['pct_kept']:.0f}%) ---")
        print(f"  Split at: {r['split_pattern'] or '40% cutoff'}")
        print(f"\n  FACTS SECTION (last 300 chars):")
        print(f"    ...{r['facts_text'][-300:]}")
        print(f"\n  FULL TEXT CONTINUES WITH (next 300 chars after cut):")
        full_text = raw_data[r['case_id']]["text"]
        cut_pos = len(r['facts_text'])
        continuation = full_text[cut_pos:cut_pos+300]
        print(f"    {continuation}...")

    # Save facts texts for pipeline
    output_dir = Path("facts_only_texts")
    output_dir.mkdir(exist_ok=True)

    manifest = []
    for r in results:
        # Save as .txt file for the pipeline
        txt_path = output_dir / f"{r['case_id']}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(r["facts_text"])
        manifest.append({
            "case_id": r["case_id"],
            "label": r["label"],
            "full_length": r["full_length"],
            "facts_length": r["facts_length"],
        })

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Saved {len(results)} facts-only texts to {output_dir}/")
    print(f"  Manifest: {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
