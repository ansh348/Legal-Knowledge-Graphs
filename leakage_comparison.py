#!/usr/bin/env python3
"""
Leakage comparison audit: raw text vs graph prompts.
Counts outcome-signaling patterns in both representations.
"""

import json
import re
import sys
import random
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from eval_graph_vs_raw import build_graph_prompt, blind_raw_text, build_raw_prompt

GRAPH_DIR = Path("graphs_enriched_v2")
EVAL_RESULTS = Path("graph_vs_raw_n2449_noscrub_grok-4-1-fast-reasoning.json")

# =========================================================================
# Leakage patterns
# =========================================================================

NEGATIVE_SIGNALS = [
    r"no\s+force\s+in",
    r"without\s+merit",
    r"devoid\s+of\s+merit",
    r"fails?\s+to\s+establish",
    r"failed?\s+to\s+establish",
    r"cannot\s+be\s+sustained",
    r"no\s+substance",
    r"rightly\s+held",
    r"rightly\s+observed",
    r"correctly\s+decided",
    r"we\s+find\s+no\s+reason\s+to\s+interfere",
    r"no\s+infirmity",
    r"deserves\s+to\s+be\s+dismissed",
    r"bereft\s+of\s+merit",
    r"no\s+merit\s+in",
    r"is\s+dismissed",
    r"appeal\s+fails",
    r"petition\s+is\s+dismissed",
    r"devoid\s+of\s+any\s+merit",
    r"lacks\s+merit",
    r"no\s+illegality",
    r"rightly\s+rejected",
]

POSITIVE_SIGNALS = [
    r"has\s+merit",
    r"wrongly\s+decided",
    r"erred\s+in",
    r"failed\s+to\s+consider",
    r"vitiated",
    r"perverse",
    r"illegal(?:ly)?",
    r"set\s+aside",
    r"deserves\s+to\s+succeed",
    r"is\s+allowed",
    r"appeal\s+is\s+allowed",
    r"appeal\s+succeeds",
    r"petition\s+is\s+allowed",
    r"wrongly\s+rejected",
    r"erroneously\s+held",
    r"erroneous(?:ly)?",
]

DISCOURSE_MARKERS = [
    r"unfortunately\s+for\s+the\s+appellant",
    r"we\s+have\s+no\s+hesitation",
    r"clearly\s+untenable",
    r"manifestly\s+(unjust|wrong|illegal|erroneous)",
    r"patently\s+(illegal|erroneous|wrong)",
    r"we\s+are\s+of\s+the\s+(view|opinion)\s+that",
    r"in\s+our\s+(considered\s+)?(view|opinion)",
    r"we\s+(find|hold)\s+that",
    r"this\s+court\s+(finds|holds)\s+that",
    r"we\s+do\s+not\s+agree",
    r"we\s+agree\s+with",
]

TREATMENT_LABELS = [
    r"\bdistinguished\b",
    r"\bfollowed\b",
    r"\bapproved\b",
    r"\boverruled\b",
    r"\bapplied\b",
    r"\bnot\s+followed\b",
    r"\bdoubted\b",
]

ALL_PATTERNS = {
    "negative": NEGATIVE_SIGNALS,
    "positive": POSITIVE_SIGNALS,
    "discourse": DISCOURSE_MARKERS,
    "treatment": TREATMENT_LABELS,
}


def count_leakage(text):
    """Count leakage signals in text. Returns (total, breakdown_dict, matches_list)."""
    text_lower = text.lower()
    total = 0
    breakdown = {"negative": 0, "positive": 0, "discourse": 0, "treatment": 0}
    matches = []

    for category, patterns in ALL_PATTERNS.items():
        for pattern in patterns:
            found = list(re.finditer(pattern, text_lower))
            count = len(found)
            if count > 0:
                breakdown[category] += count
                total += count
                for m in found:
                    start = max(0, m.start() - 20)
                    end = min(len(text), m.end() + 20)
                    context = text[start:end].replace("\n", " ").strip()
                    matches.append((category, m.group(), context))

    return total, breakdown, matches


def main():
    # Load eval results for labels
    with open(EVAL_RESULTS, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_lookup = {r["case_id"]: r for r in eval_data["cases"]}

    # Load raw texts from HuggingFace
    from datasets import load_dataset
    print("Loading raw texts from HuggingFace...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe", split="single_train")
    raw_texts = {}
    for row in ds:
        cid = str(row.get("id", "")).replace("/", "_").replace("\\", "_")
        cid = re.sub(r"[^0-9A-Za-z._-]+", "_", cid).strip("_")
        raw_texts[cid] = row.get("text", "")

    # Sample 30 random cases
    rng = random.Random(42)
    available = sorted(set(eval_lookup.keys()) & set(raw_texts.keys()))
    sample_ids = rng.sample(available, 30)

    print(f"Auditing {len(sample_ids)} cases...\n")

    # Analyze each case
    results = []
    raw_totals = []
    graph_totals = []
    raw_densities = []
    graph_densities = []

    for cid in sample_ids:
        gpath = GRAPH_DIR / f"{cid}.json"
        if not gpath.exists():
            continue

        with open(gpath, "r", encoding="utf-8") as f:
            g = json.load(f)

        # Build both prompts
        graph_prompt = build_graph_prompt(g, no_scrub=True)
        blinded_raw = blind_raw_text(raw_texts[cid], max_chars=8000)
        raw_prompt = build_raw_prompt(blinded_raw)

        # Count leakage
        raw_total, raw_bd, raw_matches = count_leakage(raw_prompt)
        graph_total, graph_bd, graph_matches = count_leakage(graph_prompt)

        # Density (per 1000 chars)
        raw_density = raw_total / (len(raw_prompt) / 1000) if len(raw_prompt) > 0 else 0
        graph_density = graph_total / (len(graph_prompt) / 1000) if len(graph_prompt) > 0 else 0

        raw_totals.append(raw_total)
        graph_totals.append(graph_total)
        raw_densities.append(raw_density)
        graph_densities.append(graph_density)

        eval_r = eval_lookup.get(cid, {})
        label = "ACC" if eval_r.get("true_label") == 1 else "REJ"

        results.append({
            "case_id": cid,
            "label": label,
            "raw_chars": len(raw_prompt),
            "graph_chars": len(graph_prompt),
            "raw_leakage": raw_total,
            "graph_leakage": graph_total,
            "raw_breakdown": raw_bd,
            "graph_breakdown": graph_bd,
            "raw_density": round(raw_density, 3),
            "graph_density": round(graph_density, 3),
            "raw_matches": raw_matches[:5],
            "graph_matches": graph_matches[:5],
        })

    import statistics

    # =========================================================================
    # Print results
    # =========================================================================
    print(f"{'='*90}")
    print(f"LEAKAGE COMPARISON: RAW TEXT vs GRAPH PROMPT")
    print(f"{'='*90}")

    # Summary stats
    print(f"\n  AGGREGATE STATISTICS (n={len(results)}):")
    print(f"  {'Metric':<40} {'Raw Text':>12} {'Graph':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Avg prompt length (chars):':<40} {statistics.mean([r['raw_chars'] for r in results]):>12.0f} {statistics.mean([r['graph_chars'] for r in results]):>12.0f}")
    print(f"  {'Avg leakage signals:':<40} {statistics.mean(raw_totals):>12.2f} {statistics.mean(graph_totals):>12.2f}")
    print(f"  {'Median leakage signals:':<40} {statistics.median(raw_totals):>12.1f} {statistics.median(graph_totals):>12.1f}")
    print(f"  {'Max leakage signals:':<40} {max(raw_totals):>12} {max(graph_totals):>12}")
    print(f"  {'Cases with 0 leakage:':<40} {sum(1 for t in raw_totals if t == 0):>12} {sum(1 for t in graph_totals if t == 0):>12}")
    print(f"  {'Avg density (signals/1000 chars):':<40} {statistics.mean(raw_densities):>12.3f} {statistics.mean(graph_densities):>12.3f}")
    print(f"  {'Median density (signals/1000 chars):':<40} {statistics.median(raw_densities):>12.3f} {statistics.median(graph_densities):>12.3f}")

    # Breakdown by type
    print(f"\n  LEAKAGE BY TYPE (total across {len(results)} cases):")
    print(f"  {'Type':<20} {'Raw':>8} {'Graph':>8} {'Ratio':>8}")
    print(f"  {'-'*48}")
    for ltype in ["negative", "positive", "discourse", "treatment"]:
        raw_sum = sum(r["raw_breakdown"][ltype] for r in results)
        graph_sum = sum(r["graph_breakdown"][ltype] for r in results)
        ratio_str = f"{raw_sum/graph_sum:.1f}x" if graph_sum > 0 else "inf"
        print(f"  {ltype:<20} {raw_sum:>8} {graph_sum:>8} {ratio_str:>8}")
    raw_all = sum(raw_totals)
    graph_all = sum(graph_totals)
    ratio_all = f"{raw_all/graph_all:.1f}x" if graph_all > 0 else "inf"
    print(f"  {'TOTAL':<20} {raw_all:>8} {graph_all:>8} {ratio_all:>8}")

    # Per-case comparison table
    print(f"\n  PER-CASE COMPARISON:")
    print(f"  {'Case':<14} {'Label':>5} {'Raw#':>5} {'Graph#':>6} {'RawD':>6} {'GraphD':>6} {'Winner':>8}")
    print(f"  {'-'*55}")
    for r in results:
        winner = "GRAPH" if r["raw_leakage"] > r["graph_leakage"] else ("RAW" if r["graph_leakage"] > r["raw_leakage"] else "TIE")
        print(f"  {r['case_id']:<14} {r['label']:>5} {r['raw_leakage']:>5} {r['graph_leakage']:>6} "
              f"{r['raw_density']:>6.2f} {r['graph_density']:>6.2f} {winner:>8}")

    # Count wins
    graph_cleaner = sum(1 for r in results if r["raw_leakage"] > r["graph_leakage"])
    raw_cleaner = sum(1 for r in results if r["graph_leakage"] > r["raw_leakage"])
    ties = sum(1 for r in results if r["raw_leakage"] == r["graph_leakage"])

    print(f"\n  VERDICT:")
    print(f"  Graph is cleaner:  {graph_cleaner}/{len(results)} cases ({graph_cleaner/len(results)*100:.0f}%)")
    print(f"  Raw is cleaner:    {raw_cleaner}/{len(results)} cases ({raw_cleaner/len(results)*100:.0f}%)")
    print(f"  Tied:              {ties}/{len(results)} cases ({ties/len(results)*100:.0f}%)")

    density_ratio = statistics.mean(raw_densities) / statistics.mean(graph_densities) if statistics.mean(graph_densities) > 0 else float("inf")
    print(f"\n  Raw text has {density_ratio:.1f}x higher leakage density than graph prompts.")
    if density_ratio > 1.5:
        print(f"  -> Graph representation provides SUBSTANTIALLY cleaner blinding.")
        print(f"  -> The graph's accuracy advantage is UNDERSTATED, not overstated.")
    elif density_ratio > 1.1:
        print(f"  -> Graph representation provides MODERATELY cleaner blinding.")
    else:
        print(f"  -> Leakage is comparable between representations.")

    # Show a few example leakage matches from raw
    print(f"\n  SAMPLE RAW TEXT LEAKAGE (first 3 cases with leakage):")
    shown = 0
    for r in results:
        if r["raw_matches"] and shown < 3:
            print(f"    {r['case_id']}:")
            for cat, match, ctx in r["raw_matches"][:3]:
                print(f"      [{cat}] \"{match}\" -> ...{ctx}...")
            shown += 1

    # Save
    output = {
        "n_cases": len(results),
        "aggregate": {
            "raw_avg_leakage": round(statistics.mean(raw_totals), 3),
            "graph_avg_leakage": round(statistics.mean(graph_totals), 3),
            "raw_avg_density": round(statistics.mean(raw_densities), 4),
            "graph_avg_density": round(statistics.mean(graph_densities), 4),
            "density_ratio": round(density_ratio, 2),
            "graph_cleaner_pct": round(graph_cleaner / len(results), 3),
        },
        "cases": [{k: v for k, v in r.items() if k != "raw_matches" and k != "graph_matches"} for r in results],
    }
    with open("leakage_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to leakage_comparison_results.json")


if __name__ == "__main__":
    main()
