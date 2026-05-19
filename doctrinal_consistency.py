#!/usr/bin/env python3
"""
Doctrinal consistency analysis across enriched Indian graphs.

Measures whether cases sharing the same doctrinal concepts (ontology concept IDs)
tend to have the same outcome — a key prerequisite for graph-RAG exemplar retrieval.
"""

import json
import sys
import random
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

GRAPH_DIR = Path("graphs_enriched_v2")


def extract_case_info(g):
    """Extract ontology concept set and outcome from a graph dict."""
    # Outcome
    outcome_node = g.get("outcome")
    if outcome_node:
        binary = outcome_node.get("binary")  # "accepted" or "rejected"
    else:
        binary = None

    if binary not in ("accepted", "rejected"):
        return None, None

    # Ontology concept IDs from cluster_summary (exclude UNLISTED_*)
    cluster_summary = (g.get("_meta") or {}).get("cluster_summary") or {}
    concept_ids = frozenset(
        cid for cid in cluster_summary.keys()
        if not cid.startswith("UNLISTED_")
    )

    return concept_ids, binary


def main():
    graph_files = sorted([
        f for f in GRAPH_DIR.glob("*.json")
        if f.name != "checkpoint.json"
    ])
    print(f"Loading {len(graph_files)} graphs...\n")

    # Extract case info
    cases = []  # list of (case_id, concept_set, outcome)
    for gf in graph_files:
        with open(gf, "r", encoding="utf-8") as f:
            g = json.load(f)
        concepts, outcome = extract_case_info(g)
        if concepts is not None and outcome is not None:
            cases.append((gf.stem, concepts, outcome))

    print(f"Cases with valid outcome + concepts: {len(cases)}")
    n_accepted = sum(1 for _, _, o in cases if o == "accepted")
    n_rejected = sum(1 for _, _, o in cases if o == "rejected")
    print(f"  Accepted: {n_accepted} ({n_accepted/len(cases)*100:.1f}%)")
    print(f"  Rejected: {n_rejected} ({n_rejected/len(cases)*100:.1f}%)")

    # =========================================================================
    # 1 & 2. DOCTRINAL SIGNATURES + CONSISTENCY
    # =========================================================================
    sig_cases = defaultdict(list)  # signature -> list of (case_id, outcome)
    for case_id, concepts, outcome in cases:
        sig_cases[concepts].append((case_id, outcome))

    total_sigs = len(sig_cases)
    sigs_5 = {s: cs for s, cs in sig_cases.items() if len(cs) >= 5}
    sigs_10 = {s: cs for s, cs in sig_cases.items() if len(cs) >= 10}

    # Consistency per signature
    def consistency(case_list):
        outcomes = [o for _, o in case_list]
        majority = max(set(outcomes), key=outcomes.count)
        return outcomes.count(majority) / len(outcomes), majority

    consistencies_5 = {}
    for sig, cs in sigs_5.items():
        rate, maj = consistency(cs)
        consistencies_5[sig] = {"rate": rate, "majority": maj, "n": len(cs),
                                "accepted": sum(1 for _, o in cs if o == "accepted"),
                                "rejected": sum(1 for _, o in cs if o == "rejected")}

    rates_5 = [v["rate"] for v in consistencies_5.values()]

    import statistics

    print(f"\n{'='*70}")
    print(f"DOCTRINAL SIGNATURE ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total unique signatures:        {total_sigs}")
    print(f"  Signatures with >=5 cases:      {len(sigs_5)}")
    print(f"  Signatures with >=10 cases:     {len(sigs_10)}")

    if rates_5:
        print(f"\n  CONSISTENCY (signatures with >=5 cases):")
        print(f"  Mean consistency rate:          {statistics.mean(rates_5):.4f}")
        print(f"  Median consistency rate:        {statistics.median(rates_5):.4f}")
        print(f"  Stdev:                          {statistics.stdev(rates_5):.4f}" if len(rates_5) > 1 else "")

        thresholds = [0.9, 0.8, 0.7, 0.6]
        for t in thresholds:
            count = sum(1 for r in rates_5 if r >= t)
            print(f"  Consistency >= {t}:              {count}/{len(rates_5)} ({count/len(rates_5)*100:.1f}%)")

    # Top 15 most common signatures
    by_count = sorted(consistencies_5.items(), key=lambda x: -x[1]["n"])
    print(f"\n  TOP 15 MOST COMMON SIGNATURES:")
    print(f"  {'#':<4} {'Cases':>5} {'Consist':>8} {'Maj':>5} {'ACC':>4}:{'REJ':>4}  Concepts")
    print(f"  {'-'*90}")
    for i, (sig, info) in enumerate(by_count[:15]):
        concepts_str = ", ".join(sorted(sig)) if sig else "(empty)"
        if len(concepts_str) > 55:
            concepts_str = concepts_str[:52] + "..."
        maj_str = "ACC" if info["majority"] == "accepted" else "REJ"
        print(f"  {i+1:<4} {info['n']:>5} {info['rate']:>8.3f} {maj_str:>5} "
              f"{info['accepted']:>4}:{info['rejected']:>4}  {concepts_str}")

    # Top 5 most inconsistent (>=10 cases)
    consistencies_10 = {s: v for s, v in consistencies_5.items() if v["n"] >= 10}
    by_inconsistency = sorted(consistencies_10.items(), key=lambda x: x[1]["rate"])
    print(f"\n  TOP 5 MOST INCONSISTENT SIGNATURES (>=10 cases):")
    print(f"  {'#':<4} {'Cases':>5} {'Consist':>8} {'ACC':>4}:{'REJ':>4}  Concepts")
    print(f"  {'-'*90}")
    for i, (sig, info) in enumerate(by_inconsistency[:5]):
        concepts_str = ", ".join(sorted(sig)) if sig else "(empty)"
        if len(concepts_str) > 55:
            concepts_str = concepts_str[:52] + "..."
        print(f"  {i+1:<4} {info['n']:>5} {info['rate']:>8.3f} "
              f"{info['accepted']:>4}:{info['rejected']:>4}  {concepts_str}")

    # =========================================================================
    # 4. PAIRWISE JACCARD SIMILARITY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PAIRWISE JACCARD SIMILARITY ANALYSIS")
    print(f"{'='*70}")

    # Sample for computational feasibility (2450^2 = 6M pairs is too many)
    rng = random.Random(42)
    sample_size = min(len(cases), 2000)
    sample = rng.sample(cases, sample_size)
    print(f"  Sampling {sample_size} cases for pairwise analysis...")

    same_outcome_high_sim = 0
    total_high_sim = 0
    same_outcome_random = 0
    total_random = 0

    # High similarity pairs (Jaccard >= 0.7)
    for i in range(len(sample)):
        for j in range(i + 1, min(i + 200, len(sample))):  # limit comparisons
            cid_i, concepts_i, outcome_i = sample[i]
            cid_j, concepts_j, outcome_j = sample[j]

            if not concepts_i or not concepts_j:
                continue

            intersection = len(concepts_i & concepts_j)
            union = len(concepts_i | concepts_j)
            if union == 0:
                continue
            jaccard = intersection / union

            if jaccard >= 0.7:
                total_high_sim += 1
                if outcome_i == outcome_j:
                    same_outcome_high_sim += 1

    # Random baseline (sample 10,000 random pairs)
    n_random_pairs = 10000
    for _ in range(n_random_pairs):
        i, j = rng.sample(range(len(sample)), 2)
        _, _, outcome_i = sample[i]
        _, _, outcome_j = sample[j]
        total_random += 1
        if outcome_i == outcome_j:
            same_outcome_random += 1

    high_sim_rate = same_outcome_high_sim / total_high_sim if total_high_sim > 0 else 0
    random_rate = same_outcome_random / total_random if total_random > 0 else 0

    print(f"  High similarity pairs (Jaccard >= 0.7): {total_high_sim}")
    print(f"    Same outcome rate:            {high_sim_rate:.4f} ({same_outcome_high_sim}/{total_high_sim})")
    print(f"  Random pairs baseline:          {total_random}")
    print(f"    Same outcome rate:            {random_rate:.4f} ({same_outcome_random}/{total_random})")
    print(f"  Lift (high_sim / random):       {high_sim_rate/random_rate:.2f}x" if random_rate > 0 else "")

    # =========================================================================
    # 5. PER-CONCEPT PREDICTIVE POWER
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"PER-CONCEPT PREDICTIVE POWER")
    print(f"{'='*70}")

    concept_outcomes = defaultdict(lambda: {"accepted": 0, "rejected": 0})
    for case_id, concepts, outcome in cases:
        for cid in concepts:
            concept_outcomes[cid][outcome] += 1

    # Filter to >=20 cases
    concept_stats = {}
    for cid, counts in concept_outcomes.items():
        total = counts["accepted"] + counts["rejected"]
        if total >= 20:
            majority = max(counts, key=counts.get)
            skew = counts[majority] / total
            concept_stats[cid] = {
                "total": total,
                "accepted": counts["accepted"],
                "rejected": counts["rejected"],
                "majority": majority,
                "skew": skew,
            }

    print(f"  Concepts with >=20 cases:       {len(concept_stats)}")

    # Overall stats
    all_skews = [v["skew"] for v in concept_stats.values()]
    if all_skews:
        print(f"  Mean skew:                      {statistics.mean(all_skews):.4f}")
        print(f"  Median skew:                    {statistics.median(all_skews):.4f}")

    # Top 10 most predictive
    by_skew = sorted(concept_stats.items(), key=lambda x: -x[1]["skew"])
    print(f"\n  TOP 10 MOST PREDICTIVE CONCEPTS:")
    print(f"  {'Concept':<50} {'N':>5} {'ACC':>4}:{'REJ':>4} {'Skew':>6} {'Maj':>5}")
    print(f"  {'-'*85}")
    for cid, info in by_skew[:10]:
        maj_str = "ACC" if info["majority"] == "accepted" else "REJ"
        print(f"  {cid:<50} {info['total']:>5} {info['accepted']:>4}:{info['rejected']:>4} "
              f"{info['skew']:>6.3f} {maj_str:>5}")

    # Top 10 least predictive (most balanced)
    by_balance = sorted(concept_stats.items(), key=lambda x: x[1]["skew"])
    print(f"\n  TOP 10 LEAST PREDICTIVE (most balanced):")
    print(f"  {'Concept':<50} {'N':>5} {'ACC':>4}:{'REJ':>4} {'Skew':>6}")
    print(f"  {'-'*80}")
    for cid, info in by_balance[:10]:
        print(f"  {cid:<50} {info['total']:>5} {info['accepted']:>4}:{info['rejected']:>4} "
              f"{info['skew']:>6.3f}")

    # =========================================================================
    # SAVE
    # =========================================================================
    output = {
        "config": {
            "graph_dir": str(GRAPH_DIR),
            "n_cases": len(cases),
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
        },
        "signatures": {
            "total_unique": total_sigs,
            "with_5_plus_cases": len(sigs_5),
            "with_10_plus_cases": len(sigs_10),
            "mean_consistency_5": round(statistics.mean(rates_5), 4) if rates_5 else None,
            "median_consistency_5": round(statistics.median(rates_5), 4) if rates_5 else None,
            "pct_above_90": round(sum(1 for r in rates_5 if r >= 0.9) / len(rates_5), 4) if rates_5 else None,
            "pct_above_80": round(sum(1 for r in rates_5 if r >= 0.8) / len(rates_5), 4) if rates_5 else None,
            "pct_above_70": round(sum(1 for r in rates_5 if r >= 0.7) / len(rates_5), 4) if rates_5 else None,
            "top_15_signatures": [
                {
                    "concepts": sorted(sig),
                    "n": info["n"],
                    "consistency": round(info["rate"], 4),
                    "majority": info["majority"],
                    "accepted": info["accepted"],
                    "rejected": info["rejected"],
                }
                for sig, info in by_count[:15]
            ],
            "top_5_inconsistent": [
                {
                    "concepts": sorted(sig),
                    "n": info["n"],
                    "consistency": round(info["rate"], 4),
                    "accepted": info["accepted"],
                    "rejected": info["rejected"],
                }
                for sig, info in by_inconsistency[:5]
            ],
        },
        "pairwise_jaccard": {
            "high_sim_pairs": total_high_sim,
            "high_sim_same_outcome_rate": round(high_sim_rate, 4),
            "random_pairs": total_random,
            "random_same_outcome_rate": round(random_rate, 4),
            "lift": round(high_sim_rate / random_rate, 4) if random_rate > 0 else None,
        },
        "per_concept": {
            "concepts_with_20_plus": len(concept_stats),
            "mean_skew": round(statistics.mean(all_skews), 4) if all_skews else None,
            "median_skew": round(statistics.median(all_skews), 4) if all_skews else None,
            "top_10_predictive": [
                {"concept": cid, **{k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()}}
                for cid, info in by_skew[:10]
            ],
            "top_10_balanced": [
                {"concept": cid, **{k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()}}
                for cid, info in by_balance[:10]
            ],
        },
    }

    out_path = Path("doctrinal_consistency_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
