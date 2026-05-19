#!/usr/bin/env python3
"""
Group cases by primary critical concept and by AND/OR satisfaction pattern.
"""

import json
import sys
import statistics
from collections import Counter, defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

GRAPH_DIR = Path("graphs_enriched_v2")


def analyze_graph(g):
    """Extract critical concept, structure stats, and satisfaction pattern."""
    js_list = g.get("justification_sets", [])
    edges = g.get("edges", [])
    concepts = g.get("concepts", [])
    holdings = g.get("holdings", [])
    cluster_summary = (g.get("_meta") or {}).get("cluster_summary") or {}
    outcome_node = g.get("outcome")

    if not js_list or not outcome_node:
        return None

    binary = outcome_node.get("binary")
    if binary not in ("accepted", "rejected"):
        return None

    concept_ids = {c["id"] for c in concepts}

    # js_id -> source nodes
    js_members = defaultdict(set)
    for e in edges:
        for sg in (e.get("support_group_ids") or []):
            js_members[sg].add(e["source"])

    # holding -> JS list
    holding_js = defaultdict(list)
    for js in js_list:
        holding_js[js["target_id"]].append(js)

    holdings_with_js = [h_id for h_id, jsg in holding_js.items() if jsg]

    # --- Critical cluster (same logic as counterfactual_analysis.py) ---
    cluster_criticality = {}
    for cluster_id, cdata in cluster_summary.items():
        cluster_nodes = set()
        for k in ["facts", "concepts", "issues", "arguments", "holdings", "precedents"]:
            cluster_nodes.update(cdata.get(k) or [])
        if len(cluster_nodes) < 2:
            continue

        holdings_killed = 0
        for h_id in holdings_with_js:
            js_group = holding_js[h_id]
            all_killed = True
            for js in js_group:
                js_id = js["id"]
                logic = js.get("logic", "and")
                members = js_members.get(js_id, set())
                if logic == "and":
                    if members & cluster_nodes:
                        continue
                    else:
                        all_killed = False
                        break
                else:
                    remaining = members - cluster_nodes
                    if remaining:
                        all_killed = False
                        break
                    else:
                        continue
            if all_killed and js_group:
                holdings_killed += 1
        cluster_criticality[cluster_id] = holdings_killed

    if not cluster_criticality:
        return None

    critical_cluster = max(cluster_criticality, key=cluster_criticality.get)

    # Fragility score
    max_killed = cluster_criticality[critical_cluster]
    fragility = max_killed / len(holdings_with_js) if holdings_with_js else 0

    # --- AND/OR satisfaction pattern for primary holding ---
    # Find primary holding (first with JS, or holding with most JS)
    primary_h = max(holding_js, key=lambda h: len(holding_js[h])) if holding_js else None
    pattern = None
    if primary_h:
        primary_js_list = holding_js[primary_h]
        n_and = sum(1 for js in primary_js_list if js.get("logic", "and") == "and")
        n_or = sum(1 for js in primary_js_list if js.get("logic", "and") == "or")

        # Count evidence prongs per AND JS
        and_prong_counts = []
        for js in primary_js_list:
            if js.get("logic", "and") == "and":
                members = js_members.get(js["id"], set())
                and_prong_counts.append(len(members))

        # Build pattern string
        if n_and > 0 and n_or == 0:
            avg_prongs = sum(and_prong_counts) / len(and_prong_counts) if and_prong_counts else 0
            if avg_prongs <= 3:
                pattern = f"AND-only({n_and}JS, <=3prongs)"
            elif avg_prongs <= 6:
                pattern = f"AND-only({n_and}JS, 4-6prongs)"
            else:
                pattern = f"AND-only({n_and}JS, 7+prongs)"
        elif n_or > 0 and n_and == 0:
            pattern = f"OR-only({n_or}JS)"
        else:
            pattern = f"Mixed(AND={n_and},OR={n_or})"

    # All ontology concept IDs (non-UNLISTED)
    all_concepts = sorted(
        cid for cid in cluster_summary.keys()
        if not cid.startswith("UNLISTED_")
    )

    # Secondary concepts (everything except the critical one)
    secondary = [c for c in all_concepts if c != critical_cluster]

    return {
        "case_id": g.get("case_id", "?"),
        "outcome": binary,
        "critical_cluster": critical_cluster,
        "n_edges": len(edges),
        "n_js": len(js_list),
        "n_and": sum(1 for js in js_list if js.get("logic", "and") == "and"),
        "n_or": sum(1 for js in js_list if js.get("logic", "and") == "or"),
        "fragility": fragility,
        "n_holdings_with_js": len(holdings_with_js),
        "pattern": pattern,
        "secondary_concepts": secondary,
    }


def main():
    graph_files = sorted([
        f for f in GRAPH_DIR.glob("*.json")
        if f.name != "checkpoint.json"
    ])
    print(f"Analyzing {len(graph_files)} graphs...\n")

    results = []
    for gf in graph_files:
        with open(gf, "r", encoding="utf-8") as f:
            g = json.load(f)
        r = analyze_graph(g)
        if r:
            results.append(r)

    print(f"Cases with valid analysis: {len(results)}")

    # =========================================================================
    # PART 1: GROUP BY PRIMARY CRITICAL CONCEPT
    # =========================================================================
    by_concept = defaultdict(list)
    for r in results:
        by_concept[r["critical_cluster"]].append(r)

    # Filter to >=20 cases
    big_groups = {k: v for k, v in by_concept.items() if len(v) >= 20}
    big_groups_sorted = sorted(big_groups.items(), key=lambda x: -len(x[1]))

    print(f"\n{'='*90}")
    print(f"PART 1: CASES GROUPED BY PRIMARY CRITICAL CONCEPT (>= 20 cases)")
    print(f"{'='*90}")
    print(f"  Groups with >=20 cases: {len(big_groups)}")

    print(f"\n  {'Concept':<42} {'N':>4} {'ACC%':>5} {'REJ%':>5} {'Consist':>7} "
          f"{'Edges':>5} {'JS':>3} {'Frag':>5}  Top secondary concepts")
    print(f"  {'-'*120}")

    for concept, group in big_groups_sorted:
        n = len(group)
        n_acc = sum(1 for r in group if r["outcome"] == "accepted")
        n_rej = n - n_acc
        majority = max(n_acc, n_rej)
        consist = majority / n

        avg_edges = statistics.mean([r["n_edges"] for r in group])
        avg_js = statistics.mean([r["n_js"] for r in group])
        avg_frag = statistics.mean([r["fragility"] for r in group])

        # Most common secondary concepts
        sec_counter = Counter()
        for r in group:
            for s in r["secondary_concepts"]:
                sec_counter[s] += 1
        top_sec = [c for c, _ in sec_counter.most_common(3)]
        sec_str = ", ".join(c[:20] for c in top_sec)

        concept_short = concept[:40]
        print(f"  {concept_short:<42} {n:>4} {n_acc/n*100:>5.1f} {n_rej/n*100:>5.1f} {consist:>7.3f} "
              f"{avg_edges:>5.0f} {avg_js:>3.0f} {avg_frag:>5.3f}  {sec_str}")

    # =========================================================================
    # PART 2: GROUP BY AND/OR PATTERN
    # =========================================================================
    by_pattern = defaultdict(list)
    for r in results:
        if r["pattern"]:
            by_pattern[r["pattern"]].append(r)

    pattern_groups = sorted(by_pattern.items(), key=lambda x: -len(x[1]))

    print(f"\n{'='*90}")
    print(f"PART 2: CASES GROUPED BY AND/OR SATISFACTION PATTERN")
    print(f"{'='*90}")

    print(f"\n  {'Pattern':<35} {'N':>5} {'ACC%':>5} {'REJ%':>5} {'Consist':>7} "
          f"{'AvgEdge':>7} {'AvgJS':>5} {'AvgFrag':>7}")
    print(f"  {'-'*85}")

    for pattern, group in pattern_groups:
        n = len(group)
        if n < 10:
            continue
        n_acc = sum(1 for r in group if r["outcome"] == "accepted")
        n_rej = n - n_acc
        majority = max(n_acc, n_rej)
        consist = majority / n

        avg_edges = statistics.mean([r["n_edges"] for r in group])
        avg_js = statistics.mean([r["n_js"] for r in group])
        avg_frag = statistics.mean([r["fragility"] for r in group])

        print(f"  {pattern:<35} {n:>5} {n_acc/n*100:>5.1f} {n_rej/n*100:>5.1f} {consist:>7.3f} "
              f"{avg_edges:>7.0f} {avg_js:>5.0f} {avg_frag:>7.3f}")

    # =========================================================================
    # PART 3: CROSS-TAB — PATTERN x OUTCOME, CONTROLLING FOR BASE RATE
    # =========================================================================
    base_acc = sum(1 for r in results if r["outcome"] == "accepted") / len(results)

    print(f"\n{'='*90}")
    print(f"PART 3: PATTERN PREDICTIVE LIFT OVER BASE RATE")
    print(f"{'='*90}")
    print(f"  Base acceptance rate: {base_acc*100:.1f}%")

    print(f"\n  {'Pattern':<35} {'N':>5} {'ACC%':>6} {'Lift':>6} {'Direction':>10}")
    print(f"  {'-'*70}")

    for pattern, group in pattern_groups:
        n = len(group)
        if n < 20:
            continue
        acc_rate = sum(1 for r in group if r["outcome"] == "accepted") / n
        lift = acc_rate / base_acc if base_acc > 0 else 0

        direction = "neutral"
        if lift > 1.15:
            direction = "PRO-ACC"
        elif lift < 0.85:
            direction = "PRO-REJ"

        print(f"  {pattern:<35} {n:>5} {acc_rate*100:>6.1f} {lift:>6.2f} {direction:>10}")

    # =========================================================================
    # PART 4: CONCEPT x PATTERN INTERACTION (most interesting combos)
    # =========================================================================
    print(f"\n{'='*90}")
    print(f"PART 4: CONCEPT x PATTERN INTERACTION")
    print(f"{'='*90}")

    combo_groups = defaultdict(list)
    for r in results:
        if r["pattern"]:
            # Simplify pattern to just AND-only/OR-only/Mixed
            if r["pattern"].startswith("AND-only"):
                simple_pattern = "AND-only"
            elif r["pattern"].startswith("OR-only"):
                simple_pattern = "OR-only"
            else:
                simple_pattern = "Mixed"
            combo_groups[(r["critical_cluster"], simple_pattern)].append(r)

    combo_sorted = sorted(
        [(k, v) for k, v in combo_groups.items() if len(v) >= 10],
        key=lambda x: -len(x[1])
    )

    print(f"\n  {'Concept':<35} {'Pattern':<10} {'N':>4} {'ACC%':>5} {'REJ%':>5} {'Consist':>7}")
    print(f"  {'-'*75}")

    for (concept, pattern), group in combo_sorted[:20]:
        n = len(group)
        n_acc = sum(1 for r in group if r["outcome"] == "accepted")
        n_rej = n - n_acc
        consist = max(n_acc, n_rej) / n
        concept_short = concept[:33]
        print(f"  {concept_short:<35} {pattern:<10} {n:>4} {n_acc/n*100:>5.1f} {n_rej/n*100:>5.1f} {consist:>7.3f}")


if __name__ == "__main__":
    main()
