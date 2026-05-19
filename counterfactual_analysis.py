#!/usr/bin/env python3
"""Counterfactual structure analysis of justification sets in enriched graphs."""
import json, sys, os
from pathlib import Path
from collections import Counter, defaultdict
import statistics

sys.stdout.reconfigure(encoding="utf-8")

GRAPH_DIR = Path("graphs_enriched_v2")


def analyze_graph(g):
    """Analyze one graph's justification structure. Returns metrics dict."""
    js_list = g.get("justification_sets", [])
    edges = g.get("edges", [])
    holdings = g.get("holdings", [])
    concepts = g.get("concepts", [])

    if not js_list:
        return None

    # Build: js_id -> list of source node IDs (from edges with that support_group_id)
    js_members = defaultdict(set)
    for e in edges:
        for sg in (e.get("support_group_ids") or []):
            js_members[sg].add(e["source"])

    # JS stats
    js_logics = [js.get("logic", "and") for js in js_list]
    n_and = sum(1 for l in js_logics if l == "and")
    n_or = sum(1 for l in js_logics if l == "or")

    # Evidence edges per JS
    edges_per_js = [len(js_members.get(js["id"], set())) for js in js_list]

    # Holding -> list of JS
    holding_js = defaultdict(list)
    for js in js_list:
        holding_js[js["target_id"]].append(js)

    # Holding fragility classification
    fragile = 0
    robust = 0
    mixed = 0
    holdings_with_js = 0
    for h_id, js_group in holding_js.items():
        if not js_group:
            continue
        holdings_with_js += 1
        logics = set(js.get("logic", "and") for js in js_group)
        if logics == {"and"}:
            fragile += 1
        elif logics == {"or"}:
            robust += 1
        else:
            mixed += 1

    # Concept criticality
    concept_ids = {c["id"] for c in concepts}

    # For each concept, track which AND/OR justification sets it's in
    concept_and_js = defaultdict(set)  # concept_id -> set of AND JS ids
    concept_or_js = defaultdict(set)   # concept_id -> set of OR JS ids

    for js in js_list:
        js_id = js["id"]
        logic = js.get("logic", "and")
        members = js_members.get(js_id, set())
        for m in members:
            if m in concept_ids:
                if logic == "and":
                    concept_and_js[m].add(js_id)
                else:
                    concept_or_js[m].add(js_id)

    # Load-bearing: participates in at least one AND JS
    load_bearing = set()
    redundant = set()
    for c_id in concept_ids:
        if concept_and_js[c_id]:
            load_bearing.add(c_id)
        elif concept_or_js[c_id]:
            redundant.add(c_id)
        # else: not in any JS (neither load-bearing nor redundant)

    # Concept criticality scores: for each concept in AND JS,
    # how many holdings would lose ALL justification if this concept were removed?
    concept_criticality = {}
    for c_id in load_bearing:
        # Which AND JSs does this concept participate in?
        affected_js_ids = concept_and_js[c_id]
        # Which holdings would be affected?
        holdings_killed = 0
        for h_id, js_group in holding_js.items():
            # Check if removing this concept kills ALL justification for this holding
            all_killed = True
            for js in js_group:
                js_id = js["id"]
                logic = js.get("logic", "and")
                members = js_members.get(js_id, set()) & concept_ids
                if logic == "and" and c_id in members:
                    # This AND JS is broken by removing c_id
                    continue
                elif logic == "or":
                    # OR JS survives if at least one other member remains
                    remaining = members - {c_id}
                    if remaining:
                        all_killed = False
                        break
                    else:
                        # Only member, so this OR JS is also killed
                        continue
                else:
                    # JS not affected by this concept
                    all_killed = False
                    break
            if all_killed and js_group:
                holdings_killed += 1
        concept_criticality[c_id] = holdings_killed

    # Fragility score: fraction of holdings killed by removing the single most critical concept
    most_critical_concept = None
    max_killed = 0
    for c_id, killed in concept_criticality.items():
        if killed > max_killed:
            max_killed = killed
            most_critical_concept = c_id

    fragility_score = max_killed / holdings_with_js if holdings_with_js > 0 else 0.0

    # Get concept labels for reporting
    concept_labels = {}
    for c in concepts:
        concept_labels[c["id"]] = c.get("concept_id", c["id"])

    return {
        "case_id": g.get("case_id", "?"),
        "n_js": len(js_list),
        "n_and": n_and,
        "n_or": n_or,
        "edges_per_js": edges_per_js,
        "holdings_with_js": holdings_with_js,
        "fragile": fragile,
        "robust": robust,
        "mixed": mixed,
        "n_concepts": len(concept_ids),
        "load_bearing": len(load_bearing),
        "redundant": len(redundant),
        "load_bearing_ids": {c: concept_labels.get(c, c) for c in load_bearing},
        "fragility_score": fragility_score,
        "most_critical_concept": concept_labels.get(most_critical_concept, most_critical_concept),
        "most_critical_holdings_killed": max_killed,
        "concept_criticality": {concept_labels.get(k, k): v for k, v in concept_criticality.items()},
    }


def main():
    graph_files = sorted([
        f for f in GRAPH_DIR.glob("*.json")
        if f.name != "checkpoint.json"
    ])
    print(f"Analyzing {len(graph_files)} graphs in {GRAPH_DIR}/...\n")

    results = []
    for gf in graph_files:
        with open(gf, "r", encoding="utf-8") as f:
            g = json.load(f)
        r = analyze_graph(g)
        if r:
            results.append(r)

    n = len(results)
    print(f"Graphs with justification sets: {n}/{len(graph_files)}")

    # =========================================================================
    # 1. JUSTIFICATION SET STATISTICS
    # =========================================================================
    total_js = sum(r["n_js"] for r in results)
    total_and = sum(r["n_and"] for r in results)
    total_or = sum(r["n_or"] for r in results)
    all_edges_per_js = [e for r in results for e in r["edges_per_js"]]

    print(f"\n{'='*70}")
    print(f"1. JUSTIFICATION SET STATISTICS")
    print(f"{'='*70}")
    print(f"  Total justification sets:       {total_js}")
    print(f"  Avg per graph:                  {total_js/n:.1f}")
    print(f"  AND logic:                      {total_and} ({total_and/total_js*100:.1f}%)")
    print(f"  OR logic:                       {total_or} ({total_or/total_js*100:.1f}%)")
    print(f"  Avg evidence edges per JS:      {statistics.mean(all_edges_per_js):.1f}")
    print(f"  Median evidence edges per JS:   {statistics.median(all_edges_per_js):.1f}")

    # =========================================================================
    # 2. HOLDING FRAGILITY ANALYSIS
    # =========================================================================
    total_fragile = sum(r["fragile"] for r in results)
    total_robust = sum(r["robust"] for r in results)
    total_mixed = sum(r["mixed"] for r in results)
    total_h_with_js = sum(r["holdings_with_js"] for r in results)

    print(f"\n{'='*70}")
    print(f"2. HOLDING FRAGILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"  Holdings with justification:    {total_h_with_js}")
    print(f"  Fragile (AND only):             {total_fragile} ({total_fragile/total_h_with_js*100:.1f}%)")
    print(f"  Robust (OR only):               {total_robust} ({total_robust/total_h_with_js*100:.1f}%)")
    print(f"  Mixed (AND + OR):               {total_mixed} ({total_mixed/total_h_with_js*100:.1f}%)")

    # =========================================================================
    # 3. CONCEPT CRITICALITY
    # =========================================================================
    all_lb = [r["load_bearing"] for r in results]
    all_rd = [r["redundant"] for r in results]
    all_nc = [r["n_concepts"] for r in results]

    # Global concept criticality counter
    global_criticality = Counter()
    for r in results:
        for concept_label, killed in r["concept_criticality"].items():
            if killed > 0:
                global_criticality[concept_label] += 1

    total_concepts = sum(all_nc)
    total_lb = sum(all_lb)
    total_rd = sum(all_rd)
    in_any_js = total_lb + total_rd

    print(f"\n{'='*70}")
    print(f"3. CONCEPT CRITICALITY")
    print(f"{'='*70}")
    print(f"  Total concepts across corpus:   {total_concepts}")
    print(f"  In any justification set:       {in_any_js} ({in_any_js/total_concepts*100:.1f}%)")
    print(f"  Load-bearing (in AND JS):       {total_lb} ({total_lb/in_any_js*100:.1f}% of JS-participating)")
    print(f"  Redundant (OR only):            {total_rd} ({total_rd/in_any_js*100:.1f}% of JS-participating)")
    print(f"  Avg load-bearing per graph:     {statistics.mean(all_lb):.1f}")
    print(f"  Avg redundant per graph:        {statistics.mean(all_rd):.1f}")
    print(f"\n  Top 10 most frequently load-bearing concepts:")
    for concept, count in global_criticality.most_common(10):
        print(f"    {concept:<45s} {count:>5} cases")

    # =========================================================================
    # 4. COUNTERFACTUAL IMPACT SCORES
    # =========================================================================
    frag_scores = [r["fragility_score"] for r in results]
    high_frag = sum(1 for f in frag_scores if f > 0.5)
    very_high = sum(1 for f in frag_scores if f > 0.8)

    print(f"\n{'='*70}")
    print(f"4. COUNTERFACTUAL IMPACT SCORES")
    print(f"{'='*70}")
    print(f"  Mean fragility score:           {statistics.mean(frag_scores):.3f}")
    print(f"  Median fragility score:         {statistics.median(frag_scores):.3f}")
    print(f"  Std dev:                        {statistics.stdev(frag_scores):.3f}")
    print(f"  Fragility > 0.5 (majority):     {high_frag} ({high_frag/n*100:.1f}%)")
    print(f"  Fragility > 0.8 (critical):     {very_high} ({very_high/n*100:.1f}%)")

    # Distribution buckets
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"\n  Distribution:")
    for lo, hi in buckets:
        count = sum(1 for f in frag_scores if lo <= f < hi)
        bar = "#" * (count * 40 // n)
        print(f"    [{lo:.1f}-{hi:.1f}): {count:>5} ({count/n*100:5.1f}%)  {bar}")

    # =========================================================================
    # 5. EXAMPLE CASES
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"5. EXAMPLE CASES")
    print(f"{'='*70}")

    # Sort by fragility
    by_frag = sorted(results, key=lambda r: r["fragility_score"], reverse=True)

    # Highly fragile
    fragile_ex = next((r for r in by_frag if r["fragility_score"] > 0.8 and r["holdings_with_js"] >= 3), by_frag[0])
    # Highly robust
    robust_ex = next((r for r in reversed(by_frag) if r["fragility_score"] < 0.1 and r["holdings_with_js"] >= 3), by_frag[-1])
    # Non-obvious: many concepts but only 1 load-bearing
    interesting = sorted(
        [r for r in results if r["load_bearing"] == 1 and r["n_concepts"] >= 8],
        key=lambda r: r["n_concepts"], reverse=True
    )
    interesting_ex = interesting[0] if interesting else results[len(results)//2]

    for label, ex in [
        ("A. HIGHLY FRAGILE (fragility > 0.8)", fragile_ex),
        ("B. HIGHLY ROBUST (fragility < 0.1)", robust_ex),
        ("C. NON-OBVIOUS (many concepts, 1 load-bearing)", interesting_ex),
    ]:
        print(f"\n  {label}")
        print(f"  {'~'*60}")
        print(f"  Case:              {ex['case_id']}.json")
        print(f"  Fragility score:   {ex['fragility_score']:.3f}")
        print(f"  Holdings w/ JS:    {ex['holdings_with_js']}")
        print(f"  Total concepts:    {ex['n_concepts']}")
        print(f"  Load-bearing:      {ex['load_bearing']}")
        print(f"  Redundant:         {ex['redundant']}")
        print(f"  JS breakdown:      {ex['n_and']} AND, {ex['n_or']} OR")
        print(f"  Most critical:     {ex['most_critical_concept']}")
        print(f"  Holdings killed:   {ex['most_critical_holdings_killed']}/{ex['holdings_with_js']}")
        # Show concept criticality details
        crit = {k: v for k, v in ex["concept_criticality"].items() if v > 0}
        if crit:
            print(f"  Concept criticality:")
            for c, v in sorted(crit.items(), key=lambda x: -x[1]):
                print(f"    {c:<40s} kills {v} holding(s)")


if __name__ == "__main__":
    main()
