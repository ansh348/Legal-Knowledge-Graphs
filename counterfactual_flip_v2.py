#!/usr/bin/env python3
"""
Counterfactual prediction flip test v2 — CLUSTER-LEVEL REMOVAL.

Removes the entire cluster (concept + all facts, arguments, precedents
assigned to it) rather than a single concept node.

Usage:
    python counterfactual_flip_v2.py --concurrent 200
"""

import argparse
import asyncio
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from eval_graph_vs_raw import (
    build_graph_prompt,
    build_blinded_graph_summary,
    llm_predict,
    _SYSTEM_PROMPT,
)

GRAPH_DIR = Path("graphs_enriched_v2")
INTACT_RESULTS = Path("graph_vs_raw_n2449_noscrub_grok-4-1-fast-reasoning.json")


# =============================================================================
# Find critical cluster
# =============================================================================

def find_critical_cluster(g):
    """Find the cluster whose removal breaks the most AND justification sets.

    Returns (critical_cluster_id, concept_criticality_dict) or (None, {}).
    """
    js_list = g.get("justification_sets", [])
    edges = g.get("edges", [])
    concepts = g.get("concepts", [])
    cluster_summary = (g.get("_meta") or {}).get("cluster_summary") or {}

    if not js_list or not cluster_summary:
        return None, {}

    concept_ids = {c["id"] for c in concepts}

    # js_id -> set of source node IDs
    js_members = defaultdict(set)
    for e in edges:
        for sg in (e.get("support_group_ids") or []):
            js_members[sg].add(e["source"])

    # holding_id -> list of JS
    holding_js = defaultdict(list)
    for js in js_list:
        holding_js[js["target_id"]].append(js)

    holdings_with_js = [h_id for h_id, jsg in holding_js.items() if jsg]

    # For each cluster, compute how many holdings would lose ALL justification
    # if the entire cluster were removed
    cluster_criticality = {}
    for cluster_id, cdata in cluster_summary.items():
        # All node IDs in this cluster
        cluster_nodes = set()
        for k in ["facts", "concepts", "issues", "arguments", "holdings", "precedents"]:
            cluster_nodes.update(cdata.get(k) or [])

        if len(cluster_nodes) < 3:
            continue

        holdings_killed = 0
        for h_id in holdings_with_js:
            # If the holding itself is in this cluster, it's "killed" trivially
            # But more meaningfully: check if all its JSs lose support
            js_group = holding_js[h_id]
            all_killed = True
            for js in js_group:
                js_id = js["id"]
                logic = js.get("logic", "and")
                members = js_members.get(js_id, set())

                # Remove cluster nodes from members
                remaining = members - cluster_nodes
                if logic == "and":
                    # AND JS: if any member removed, JS breaks
                    if members & cluster_nodes:
                        continue  # broken
                    else:
                        all_killed = False
                        break
                else:
                    # OR JS: survives if any member remains
                    if remaining:
                        all_killed = False
                        break
                    else:
                        continue  # broken
            if all_killed and js_group:
                holdings_killed += 1

        cluster_criticality[cluster_id] = holdings_killed

    if not cluster_criticality:
        return None, {}

    critical = max(cluster_criticality, key=cluster_criticality.get)
    return critical, cluster_criticality


def get_cluster_size(cluster_summary, cluster_id):
    """Count total nodes in a cluster."""
    cdata = cluster_summary.get(cluster_id, {})
    return sum(len(cdata.get(k) or []) for k in
               ["facts", "concepts", "issues", "arguments", "holdings", "precedents"])


def remove_cluster_from_graph(graph_dict, cluster_id):
    """Remove all nodes in a cluster and their edges from the graph."""
    g = copy.deepcopy(graph_dict)
    cluster_summary = (g.get("_meta") or {}).get("cluster_summary") or {}
    cdata = cluster_summary.get(cluster_id, {})

    # Collect all node IDs to remove
    remove_ids = set()
    for k in ["facts", "concepts", "issues", "arguments", "holdings", "precedents"]:
        remove_ids.update(cdata.get(k) or [])

    if not remove_ids:
        return g, 0

    # Remove nodes
    g["facts"] = [n for n in (g.get("facts") or []) if n.get("id") not in remove_ids]
    g["concepts"] = [n for n in (g.get("concepts") or []) if n.get("id") not in remove_ids]
    g["issues"] = [n for n in (g.get("issues") or []) if n.get("id") not in remove_ids]
    g["arguments"] = [n for n in (g.get("arguments") or []) if n.get("id") not in remove_ids]
    g["holdings"] = [n for n in (g.get("holdings") or []) if n.get("id") not in remove_ids]
    g["precedents"] = [n for n in (g.get("precedents") or []) if n.get("id") not in remove_ids]

    # Remove edges touching any removed node
    g["edges"] = [
        e for e in (g.get("edges") or [])
        if e.get("source") not in remove_ids and e.get("target") not in remove_ids
    ]

    return g, len(remove_ids)


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Counterfactual flip test v2 — cluster removal")
    parser.add_argument("--concurrent", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cases", type=int, default=0)
    args = parser.parse_args()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    model = "grok-4-1-fast-reasoning"

    # 1. Load intact results
    print(f"Loading intact results from {INTACT_RESULTS}...")
    with open(INTACT_RESULTS, "r", encoding="utf-8") as f:
        intact_data = json.load(f)

    intact_cases = {r["case_id"]: r for r in intact_data["cases"]}
    correct_cases = {
        cid: r for cid, r in intact_cases.items()
        if r["graph_pred"] == r["true_label"] and r["graph_pred"] in (0, 1)
    }
    print(f"  {len(correct_cases)} cases with correct graph predictions")

    # 2. Analyze each graph and build test cases
    print(f"\nAnalyzing cluster structure...")
    eligible = []
    rng = random.Random(args.seed)

    for cid in sorted(correct_cases.keys()):
        gpath = GRAPH_DIR / f"{cid}.json"
        if not gpath.exists():
            continue
        with open(gpath, "r", encoding="utf-8") as f:
            g = json.load(f)

        cluster_summary = (g.get("_meta") or {}).get("cluster_summary") or {}
        if len(cluster_summary) < 3:
            continue

        critical_id, criticality = find_critical_cluster(g)
        if critical_id is None:
            continue

        crit_size = get_cluster_size(cluster_summary, critical_id)
        if crit_size < 3:
            continue

        # Find a random cluster of similar size (±2 nodes), not the critical one
        candidates = []
        for cl_id in cluster_summary:
            if cl_id == critical_id:
                continue
            sz = get_cluster_size(cluster_summary, cl_id)
            if sz >= max(3, crit_size - 2) and sz <= crit_size + 2:
                candidates.append(cl_id)

        # Fallback: any cluster with >=3 nodes
        if not candidates:
            candidates = [
                cl_id for cl_id in cluster_summary
                if cl_id != critical_id and get_cluster_size(cluster_summary, cl_id) >= 3
            ]

        if not candidates:
            continue

        rand_cluster = rng.choice(candidates)
        rand_size = get_cluster_size(cluster_summary, rand_cluster)

        # Build modified graphs and prompts
        g_targeted, n_removed_t = remove_cluster_from_graph(g, critical_id)
        g_random, n_removed_r = remove_cluster_from_graph(g, rand_cluster)

        prompt_intact = build_graph_prompt(g, no_scrub=True)
        prompt_targeted = build_graph_prompt(g_targeted, no_scrub=True)
        prompt_random = build_graph_prompt(g_random, no_scrub=True)

        # Compute % of prompt removed
        pct_removed_t = 1.0 - len(prompt_targeted) / len(prompt_intact) if len(prompt_intact) > 0 else 0
        pct_removed_r = 1.0 - len(prompt_random) / len(prompt_intact) if len(prompt_intact) > 0 else 0

        eligible.append({
            "case_id": cid,
            "true_label": correct_cases[cid]["true_label"],
            "intact_pred": correct_cases[cid]["graph_pred"],
            "intact_conf": correct_cases[cid]["graph_conf"],
            "critical_cluster": critical_id,
            "critical_cluster_size": crit_size,
            "critical_holdings_killed": criticality.get(critical_id, 0),
            "random_cluster": rand_cluster,
            "random_cluster_size": rand_size,
            "nodes_removed_targeted": n_removed_t,
            "nodes_removed_random": n_removed_r,
            "pct_prompt_removed_targeted": round(pct_removed_t, 3),
            "pct_prompt_removed_random": round(pct_removed_r, 3),
            "prompt_targeted": prompt_targeted,
            "prompt_random": prompt_random,
        })

    if args.max_cases > 0:
        eligible = eligible[:args.max_cases]

    print(f"  {len(eligible)} eligible cases")

    # Stats on removal sizes
    avg_t = sum(c["nodes_removed_targeted"] for c in eligible) / len(eligible)
    avg_r = sum(c["nodes_removed_random"] for c in eligible) / len(eligible)
    avg_pct_t = sum(c["pct_prompt_removed_targeted"] for c in eligible) / len(eligible)
    avg_pct_r = sum(c["pct_prompt_removed_random"] for c in eligible) / len(eligible)
    print(f"  Avg nodes removed (targeted): {avg_t:.1f}")
    print(f"  Avg nodes removed (random):   {avg_r:.1f}")
    print(f"  Avg % prompt removed (targeted): {avg_pct_t*100:.1f}%")
    print(f"  Avg % prompt removed (random):   {avg_pct_r*100:.1f}%")

    total_calls = len(eligible) * 2
    est_cost = total_calls * 3000 * 0.20 / 1e6 + total_calls * 1000 * 0.50 / 1e6
    print(f"  API calls: {total_calls} (est. cost: ${est_cost:.2f})")

    # 3. Run predictions
    semaphore = asyncio.Semaphore(args.concurrent)
    results = []
    errors = 0
    t0 = time.time()

    async def predict_one(case, idx):
        nonlocal errors
        async with semaphore:
            targeted_result = await llm_predict(
                api_key, _SYSTEM_PROMPT, case["prompt_targeted"], model=model
            )
            random_result = await llm_predict(
                api_key, _SYSTEM_PROMPT, case["prompt_random"], model=model
            )

        tp = targeted_result.get("prediction", -1)
        rp = random_result.get("prediction", -1)
        if tp not in (0, 1): errors += 1
        if rp not in (0, 1): errors += 1

        true_label = case["true_label"]
        targeted_flipped = (tp != true_label) and tp in (0, 1)
        random_flipped = (rp != true_label) and rp in (0, 1)

        if targeted_flipped and not random_flipped:
            tag = "TARGETED_ONLY"
        elif random_flipped and not targeted_flipped:
            tag = "RANDOM_ONLY"
        elif targeted_flipped and random_flipped:
            tag = "BOTH_FLIP"
        else:
            tag = "NEITHER"

        if idx % 100 == 0 or targeted_flipped:
            label_str = "ACC" if true_label == 1 else "REJ"
            tp_str = "ACC" if tp == 1 else ("REJ" if tp == 0 else "ERR")
            rp_str = "ACC" if rp == 1 else ("REJ" if rp == 0 else "ERR")
            print(f"  [{idx:4d}] {case['case_id']:12s} TRUE={label_str}  "
                  f"tgt={tp_str} rnd={rp_str}  {tag}  "
                  f"crit={case['critical_cluster'][:35]}  "
                  f"removed={case['nodes_removed_targeted']}n/{case['pct_prompt_removed_targeted']*100:.0f}%")

        return {
            "case_id": case["case_id"],
            "true_label": true_label,
            "intact_pred": case["intact_pred"],
            "intact_conf": case["intact_conf"],
            "targeted_pred": tp,
            "targeted_conf": targeted_result.get("confidence", 0),
            "targeted_reasoning": targeted_result.get("reasoning", "")[:200],
            "random_pred": rp,
            "random_conf": random_result.get("confidence", 0),
            "random_reasoning": random_result.get("reasoning", "")[:200],
            "targeted_flipped": targeted_flipped,
            "random_flipped": random_flipped,
            "critical_cluster": case["critical_cluster"],
            "critical_cluster_size": case["critical_cluster_size"],
            "random_cluster": case["random_cluster"],
            "random_cluster_size": case["random_cluster_size"],
            "nodes_removed_targeted": case["nodes_removed_targeted"],
            "nodes_removed_random": case["nodes_removed_random"],
            "pct_prompt_removed_targeted": case["pct_prompt_removed_targeted"],
            "pct_prompt_removed_random": case["pct_prompt_removed_random"],
        }

    # Process in batches
    batch_size = 200
    for batch_start in range(0, len(eligible), batch_size):
        batch = eligible[batch_start:batch_start + batch_size]
        tasks = [
            asyncio.create_task(predict_one(c, batch_start + i))
            for i, c in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        done = len(results)
        t_flips = sum(1 for r in results if r["targeted_flipped"])
        r_flips = sum(1 for r in results if r["random_flipped"])
        print(f"  --- {done}/{len(eligible)} done | "
              f"targeted_flips={t_flips} random_flips={r_flips} ---")

    elapsed = time.time() - t0

    # 4. Metrics
    valid = [r for r in results
             if r["targeted_pred"] in (0, 1) and r["random_pred"] in (0, 1)]
    n = len(valid)

    targeted_flips = sum(1 for r in valid if r["targeted_flipped"])
    random_flips = sum(1 for r in valid if r["random_flipped"])
    both_flips = sum(1 for r in valid if r["targeted_flipped"] and r["random_flipped"])
    targeted_only = sum(1 for r in valid if r["targeted_flipped"] and not r["random_flipped"])
    random_only = sum(1 for r in valid if r["random_flipped"] and not r["targeted_flipped"])
    neither = sum(1 for r in valid if not r["targeted_flipped"] and not r["random_flipped"])

    targeted_flip_rate = targeted_flips / n if n else 0
    random_flip_rate = random_flips / n if n else 0
    flip_ratio = targeted_flip_rate / random_flip_rate if random_flip_rate > 0 else float("inf")

    targeted_acc = sum(1 for r in valid if r["targeted_pred"] == r["true_label"]) / n if n else 0
    random_acc = sum(1 for r in valid if r["random_pred"] == r["true_label"]) / n if n else 0

    # McNemar test on 2x2 table: targeted flip vs random flip
    # b = targeted flipped but random didn't, c = random flipped but targeted didn't
    b = targeted_only
    c = random_only
    if b + c > 0:
        mcnemar_chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        mcnemar_sig = "p<0.05" if mcnemar_chi2 > 3.841 else "n.s."
        mcnemar_p = "significant" if mcnemar_chi2 > 3.841 else "not significant"
    else:
        mcnemar_chi2 = 0
        mcnemar_sig = "n.s."
        mcnemar_p = "undefined"

    # Avg prompt removal sizes
    avg_pct_t = sum(r["pct_prompt_removed_targeted"] for r in valid) / n if n else 0
    avg_pct_r = sum(r["pct_prompt_removed_random"] for r in valid) / n if n else 0

    # Top concepts causing flips
    flip_concepts = Counter()
    for r in valid:
        if r["targeted_flipped"]:
            flip_concepts[r["critical_cluster"]] += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"COUNTERFACTUAL FLIP TEST v2 — CLUSTER-LEVEL REMOVAL")
    print(f"{'='*70}")
    print(f"  Cases tested:                   {n}")
    print(f"  Elapsed:                        {elapsed:.0f}s")
    print(f"  API errors:                     {errors}")

    print(f"\n  REMOVAL SIZES (fairness check):")
    print(f"  {'Avg % prompt removed (targeted):':<40} {avg_pct_t*100:.1f}%")
    print(f"  {'Avg % prompt removed (random):':<40} {avg_pct_r*100:.1f}%")

    print(f"\n  FLIP RATES:")
    print(f"  {'Targeted cluster removal:':<40} {targeted_flip_rate:.4f} ({targeted_flips}/{n})")
    print(f"  {'Random cluster removal:':<40} {random_flip_rate:.4f} ({random_flips}/{n})")
    print(f"  {'Flip ratio (targeted/random):':<40} {flip_ratio:.2f}x")

    print(f"\n  ACCURACY AFTER REMOVAL:")
    print(f"  {'Intact (baseline):':<40} 1.000")
    print(f"  {'After targeted cluster removal:':<40} {targeted_acc:.4f}")
    print(f"  {'After random cluster removal:':<40} {random_acc:.4f}")

    print(f"\n  CONCORDANCE TABLE:")
    print(f"  {'Neither flipped:':<40} {neither:>5} ({neither/n*100:.1f}%)")
    print(f"  {'Targeted only flipped:':<40} {targeted_only:>5} ({targeted_only/n*100:.1f}%)")
    print(f"  {'Random only flipped:':<40} {random_only:>5} ({random_only/n*100:.1f}%)")
    print(f"  {'Both flipped:':<40} {both_flips:>5} ({both_flips/n*100:.1f}%)")

    print(f"\n  McNEMAR TEST (targeted vs random flips):")
    print(f"  {'b (targeted only):':<40} {b}")
    print(f"  {'c (random only):':<40} {c}")
    print(f"  {'chi2:':<40} {mcnemar_chi2:.4f}")
    print(f"  {'significance:':<40} {mcnemar_sig}")

    print(f"\n  TOP 10 CLUSTERS CAUSING TARGETED FLIPS:")
    for concept, count in flip_concepts.most_common(10):
        print(f"    {concept:<45s} {count:>4} flips")

    # 5. Save
    output = {
        "config": {
            "model": model,
            "n_cases": n,
            "seed": args.seed,
            "graph_dir": str(GRAPH_DIR),
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": {
            "targeted_flip_rate": round(targeted_flip_rate, 4),
            "random_flip_rate": round(random_flip_rate, 4),
            "flip_ratio": round(flip_ratio, 2),
            "targeted_accuracy": round(targeted_acc, 4),
            "random_accuracy": round(random_acc, 4),
            "targeted_flips": targeted_flips,
            "random_flips": random_flips,
            "targeted_only": targeted_only,
            "random_only": random_only,
            "both_flips": both_flips,
            "neither": neither,
            "mcnemar_chi2": round(mcnemar_chi2, 4),
            "mcnemar_sig": mcnemar_sig,
            "avg_pct_prompt_removed_targeted": round(avg_pct_t, 4),
            "avg_pct_prompt_removed_random": round(avg_pct_r, 4),
            "top_flip_concepts": flip_concepts.most_common(20),
        },
        "cases": valid,
    }

    out_path = Path("counterfactual_flip_v2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
