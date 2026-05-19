#!/usr/bin/env python3
"""
ILDC SOTA fairness audit.
1. Dataset overlap with ILDC test set
2. Input text comparison (ILDC raw facts vs our blinded graph prompt)
3. Outcome leakage audit on 20 random correctly-predicted cases
"""

import json
import re
import sys
import random
from pathlib import Path
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from datasets import load_dataset

GRAPH_DIR = Path("graphs_enriched_v2")
EVAL_RESULTS = Path("graph_vs_raw_n2449_noscrub_grok-4-1-fast-reasoning.json")


# Leakage patterns
MERIT_LEAKAGE = [
    r"without\s+merit", r"devoid\s+of\s+merit", r"no\s+force\s+in",
    r"rightly\s+held", r"wrongly\s+decided", r"rightly\s+decided",
    r"no\s+substance", r"without\s+substance", r"lacks\s+merit",
    r"utterly\s+without", r"devoid\s+of\s+substance",
    r"cannot\s+be\s+sustained", r"deserves\s+to\s+be\s+(allowed|dismissed)",
    r"is\s+(liable|fit)\s+to\s+be\s+(dismissed|allowed|set\s+aside)",
    r"must\s+(succeed|fail)", r"bound\s+to\s+(succeed|fail)",
    r"no\s+merit\s+in", r"bereft\s+of\s+merit",
]

TREATMENT_LEAKAGE = [
    "distinguished", "overruled", "followed", "approved",
    "doubted", "not applicable", "misapplied",
]

COURT_ANALYSIS_PATTERNS = [
    r"we\s+(find|hold|are\s+of\s+the\s+(view|opinion))",
    r"this\s+court\s+(finds|holds|is\s+of\s+the\s+(view|opinion))",
    r"in\s+our\s+(view|opinion|considered\s+view)",
    r"we\s+are\s+(satisfied|not\s+satisfied|convinced|not\s+convinced)",
    r"the\s+learned\s+judge\s+(rightly|wrongly|correctly|erroneously)",
    r"the\s+high\s+court\s+(rightly|wrongly|correctly|erroneously)",
    r"we\s+do\s+not\s+agree", r"we\s+agree\s+with",
    r"the\s+contention\s+is\s+(rejected|accepted|well-founded|ill-founded)",
    r"the\s+submission\s+(has|deserves)\s+(no|some)\s+force",
]

JUDICIAL_OPINION_PATTERNS = [
    r"we\s+have\s+no\s+hesitation",
    r"clearly\s+(untenable|unsustainable|erroneous|correct)",
    r"patently\s+(illegal|erroneous|wrong)",
    r"manifestly\s+(unjust|wrong|illegal)",
    r"the\s+appeal\s+(is|stands|deserves)\s+to\s+be",
]


def audit_prompt_for_leakage(prompt_text):
    """Check a blinded graph prompt for outcome leakage. Returns list of findings."""
    findings = []
    text_lower = prompt_text.lower()

    # Merit language
    for pattern in MERIT_LEAKAGE:
        matches = re.findall(pattern, text_lower)
        if matches:
            # Find the actual text around the match
            for m in re.finditer(pattern, text_lower):
                start = max(0, m.start() - 30)
                end = min(len(text_lower), m.end() + 30)
                context = prompt_text[start:end].replace("\n", " ")
                findings.append(("MERIT_LANGUAGE", m.group(), context))

    # Treatment labels in precedent text
    for treatment in TREATMENT_LEAKAGE:
        if treatment.lower() in text_lower:
            # Find context
            idx = text_lower.find(treatment.lower())
            if idx >= 0:
                start = max(0, idx - 30)
                end = min(len(text_lower), idx + len(treatment) + 30)
                context = prompt_text[start:end].replace("\n", " ")
                findings.append(("TREATMENT_LABEL", treatment, context))

    # Court analysis language
    for pattern in COURT_ANALYSIS_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            for m in re.finditer(pattern, text_lower):
                start = max(0, m.start() - 30)
                end = min(len(text_lower), m.end() + 30)
                context = prompt_text[start:end].replace("\n", " ")
                findings.append(("COURT_ANALYSIS", m.group(), context))

    # Judicial opinion
    for pattern in JUDICIAL_OPINION_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            for m in re.finditer(pattern, text_lower):
                start = max(0, m.start() - 30)
                end = min(len(text_lower), m.end() + 30)
                context = prompt_text[start:end].replace("\n", " ")
                findings.append(("JUDICIAL_OPINION", m.group(), context))

    return findings


def main():
    # =================================================================
    # 1. DATASET OVERLAP
    # =================================================================
    print(f"{'='*70}")
    print(f"1. DATASET OVERLAP WITH ILDC")
    print(f"{'='*70}")

    # Load IL-TUR / ILDC dataset
    print("Loading Exploration-Lab/IL-TUR (cjpe config)...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe")

    # Show available splits
    print(f"  Available splits: {list(ds.keys())}")
    for split_name in ds:
        print(f"    {split_name}: {len(ds[split_name])} cases")

    # Build ID sets per split
    split_ids = {}
    for split_name in ds:
        ids = set()
        for row in ds[split_name]:
            cid = str(row.get("id", "")).replace("/", "_").replace("\\", "_")
            cid = re.sub(r"[^0-9A-Za-z._-]+", "_", cid).strip("_")
            ids.add(cid)
        split_ids[split_name] = ids

    # Our evaluation case IDs
    our_graph_ids = set()
    for gf in GRAPH_DIR.glob("*.json"):
        if gf.name != "checkpoint.json":
            our_graph_ids.add(gf.stem)

    # Load eval results case IDs
    with open(EVAL_RESULTS, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_ids = {r["case_id"] for r in eval_data["cases"]}

    print(f"\n  Our graphs: {len(our_graph_ids)} cases")
    print(f"  Our eval results: {len(eval_ids)} cases")

    for split_name, ids in split_ids.items():
        overlap_graphs = our_graph_ids & ids
        overlap_eval = eval_ids & ids
        print(f"\n  vs {split_name} ({len(ids)} cases):")
        print(f"    Graph overlap: {len(overlap_graphs)}/{len(ids)} ({len(overlap_graphs)/len(ids)*100:.1f}%)")
        print(f"    Eval overlap:  {len(overlap_eval)}/{len(ids)} ({len(overlap_eval)/len(ids)*100:.1f}%)")

    # Check if there's a dedicated test split
    if "test" in split_ids:
        test_ids = split_ids["test"]
        test_in_our_eval = eval_ids & test_ids
        test_NOT_in_eval = test_ids - eval_ids
        eval_NOT_in_test = eval_ids - test_ids
        print(f"\n  CRITICAL: ILDC test set analysis:")
        print(f"    Test cases in our eval: {len(test_in_our_eval)}/{len(test_ids)}")
        print(f"    Test cases NOT in our eval: {len(test_NOT_in_eval)}")
        print(f"    Our eval cases NOT in test: {len(eval_NOT_in_test)}")
        if len(test_NOT_in_eval) > 0:
            print(f"    -> We are NOT evaluating on the same set as ILDC!")

    # Check single_train (the full set we used)
    if "single_train" in split_ids:
        st_ids = split_ids["single_train"]
        print(f"\n  single_train split ({len(st_ids)} cases):")
        print(f"    Contains our eval cases: {len(eval_ids & st_ids)}/{len(eval_ids)}")
        print(f"    -> We trained/evaluated on single_train, not the official test split")

    # =================================================================
    # 2. INPUT TEXT COMPARISON
    # =================================================================
    print(f"\n{'='*70}")
    print(f"2. INPUT TEXT COMPARISON (5 sample cases)")
    print(f"{'='*70}")

    from eval_graph_vs_raw import build_graph_prompt

    # Load the dataset for raw text
    if "single_train" in ds:
        raw_data = {
            re.sub(r"[^0-9A-Za-z._-]+", "_", str(row.get("id", ""))).strip("_"): row
            for row in ds["single_train"]
        }
    else:
        raw_data = {
            re.sub(r"[^0-9A-Za-z._-]+", "_", str(row.get("id", ""))).strip("_"): row
            for row in ds[list(ds.keys())[0]]
        }

    rng = random.Random(42)
    sample_ids = rng.sample(sorted(eval_ids & set(raw_data.keys())), 5)

    for cid in sample_ids:
        gpath = GRAPH_DIR / f"{cid}.json"
        if not gpath.exists():
            continue

        with open(gpath, "r", encoding="utf-8") as f:
            g = json.load(f)

        row = raw_data[cid]
        ildc_text = row.get("text", "")

        # Build our blinded graph prompt
        graph_prompt = build_graph_prompt(g, no_scrub=True)

        print(f"\n  --- Case: {cid} ---")
        print(f"  ILDC raw text: {len(ildc_text)} chars")
        print(f"  Our graph prompt: {len(graph_prompt)} chars")

        # Check what our prompt contains that goes beyond facts
        has_arguments = "PARTY ARGUMENTS:" in graph_prompt
        has_precedents = "CITED PRECEDENTS:" in graph_prompt
        has_concepts = "LEGAL CONCEPTS:" in graph_prompt
        has_issues = "ISSUES BEFORE THE COURT:" in graph_prompt

        print(f"  Graph prompt contains:")
        print(f"    FACTS:              {'YES' if 'FACTS:' in graph_prompt else 'NO'}")
        print(f"    LEGAL CONCEPTS:     {'YES' if has_concepts else 'NO'}")
        print(f"    ISSUES:             {'YES' if has_issues else 'NO'}")
        print(f"    PARTY ARGUMENTS:    {'YES' if has_arguments else 'NO'}")
        print(f"    CITED PRECEDENTS:   {'YES' if has_precedents else 'NO'}")

        # Check if arguments contain court analysis (not just party claims)
        args = g.get("arguments", [])
        court_args = [a for a in args if a.get("actor") == "court"]
        party_args = [a for a in args if a.get("actor") != "court"]
        print(f"    Court-actor arguments: {len(court_args)} (SHOULD BE 0 — excluded from prompt)")
        print(f"    Party arguments: {len(party_args)}")

        # Show first 200 chars of ILDC text vs our prompt
        print(f"\n  ILDC text (first 300 chars):")
        print(f"    {ildc_text[:300]}...")
        print(f"\n  Our graph prompt (first 500 chars):")
        print(f"    {graph_prompt[:500]}...")

    # =================================================================
    # 3. OUTCOME LEAKAGE AUDIT
    # =================================================================
    print(f"\n{'='*70}")
    print(f"3. OUTCOME LEAKAGE AUDIT (20 random correct predictions)")
    print(f"{'='*70}")

    # Get correctly predicted cases
    correct_cases = [
        r for r in eval_data["cases"]
        if r["graph_pred"] == r["true_label"] and r["graph_pred"] in (0, 1)
    ]
    rng2 = random.Random(42)
    audit_sample = rng2.sample(correct_cases, min(20, len(correct_cases)))

    total_leaky = 0
    leakage_type_counts = Counter()
    all_findings_by_case = []

    for case in audit_sample:
        cid = case["case_id"]
        gpath = GRAPH_DIR / f"{cid}.json"
        if not gpath.exists():
            continue

        with open(gpath, "r", encoding="utf-8") as f:
            g = json.load(f)

        prompt = build_graph_prompt(g, no_scrub=True)
        findings = audit_prompt_for_leakage(prompt)

        if findings:
            total_leaky += 1

        for ftype, match, context in findings:
            leakage_type_counts[ftype] += 1

        all_findings_by_case.append({
            "case_id": cid,
            "true_label": case["true_label"],
            "graph_pred": case["graph_pred"],
            "n_findings": len(findings),
            "findings": [(t, m, c) for t, m, c in findings],
        })

    print(f"  Cases audited: {len(audit_sample)}")
    print(f"  Cases with potential leakage: {total_leaky}/{len(audit_sample)} "
          f"({total_leaky/len(audit_sample)*100:.0f}%)")

    print(f"\n  Leakage type breakdown:")
    for ltype, count in leakage_type_counts.most_common():
        print(f"    {ltype:<25} {count:>3} instances")

    print(f"\n  Per-case details:")
    for entry in all_findings_by_case:
        label = "ACC" if entry["true_label"] == 1 else "REJ"
        pred = "ACC" if entry["graph_pred"] == 1 else "REJ"
        status = "CLEAN" if entry["n_findings"] == 0 else f"LEAKY ({entry['n_findings']})"
        print(f"    {entry['case_id']:12s} TRUE={label} PRED={pred}  {status}")
        for ftype, match, context in entry["findings"][:3]:
            print(f"      [{ftype}] \"{match}\"")
            print(f"        ...{context}...")
        if len(entry["findings"]) > 3:
            print(f"      ... and {len(entry['findings']) - 3} more")

    # =================================================================
    # SUMMARY VERDICT
    # =================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY VERDICT")
    print(f"{'='*70}")

    # Dataset overlap
    if "test" in split_ids:
        test_overlap = len(eval_ids & split_ids["test"])
        test_total = len(split_ids["test"])
        if test_overlap < test_total * 0.9:
            print(f"  [WARN] DATASET: We evaluate on single_train, NOT the ILDC test split.")
            print(f"         Only {test_overlap}/{test_total} test cases are in our eval.")
            print(f"         -> Direct SOTA comparison requires re-eval on test split only.")
        else:
            print(f"  [OK] DATASET: {test_overlap}/{test_total} test cases overlap.")

    # Information advantage
    print(f"\n  [INFO] INFORMATION ADVANTAGE:")
    print(f"    Our graph prompt includes: facts + concepts + issues + arguments + precedents")
    print(f"    ILDC task provides: full judgment text (facts through disposition)")
    print(f"    Key difference: our graph has STRUCTURED EXTRACTION with typed nodes,")
    print(f"    but ILDC models see the FULL text (including court reasoning sections)")
    print(f"    that we deliberately blind. This is a FAIRNESS TRADEOFF, not a clear advantage.")

    # Leakage
    leakage_rate = total_leaky / len(audit_sample) * 100
    if leakage_rate > 50:
        print(f"\n  [FAIL] LEAKAGE: {leakage_rate:.0f}% of cases show potential outcome leakage.")
    elif leakage_rate > 20:
        print(f"\n  [WARN] LEAKAGE: {leakage_rate:.0f}% of cases show potential outcome leakage.")
    else:
        print(f"\n  [OK] LEAKAGE: Only {leakage_rate:.0f}% of cases show potential leakage.")


if __name__ == "__main__":
    main()
