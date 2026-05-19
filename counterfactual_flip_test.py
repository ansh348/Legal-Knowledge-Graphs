#!/usr/bin/env python3
"""
Counterfactual prediction flip test.

Tests whether AND/OR justification structure identifies genuinely causal
concepts by measuring prediction changes when critical vs random concepts
are removed from graph prompts.

Usage:
    python counterfactual_flip_test.py --concurrent 200
"""

import argparse
import asyncio
import copy
import hashlib
import json
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
    llm_predict,
    _SYSTEM_PROMPT,
    _extract_prediction_json,
)

GRAPH_DIR = Path("graphs_enriched_v2")
INTACT_RESULTS = Path("graph_vs_raw_n2449_noscrub_grok-4-1-fast-reasoning.json")


# =============================================================================
# Counterfactual concept identification (from counterfactual_analysis.py)
# =============================================================================

def find_critical_and_random_concepts(g, seed=42):
    """Find the most critical concept and a random JS-participating concept.

    Returns (critical_concept_id, random_concept_id, n_js_participating)
    or (None, None, 0) if not enough concepts.
    """
    js_list = g.get("justification_sets", [])
    edges = g.get("edges", [])
    concepts = g.get("concepts", [])

    if not js_list or len(concepts) < 3:
        return None, None, 0

    concept_ids = {c["id"] for c in concepts}

    # Build: js_id -> set of source node IDs from edges
    js_members = defaultdict(set)
    for e in edges:
        for sg in (e.get("support_group_ids") or []):
            js_members[sg].add(e["source"])

    # holding_id -> list of JS dicts
    holding_js = defaultdict(list)
    for js in js_list:
        holding_js[js["target_id"]].append(js)

    # Find concepts participating in any JS
    js_participating = set()
    concept_and_js = defaultdict(set)
    for js in js_list:
        members = js_members.get(js["id"], set()) & concept_ids
        js_participating |= members
        if js.get("logic", "and") == "and":
            for m in members:
                concept_and_js[m].add(js["id"])

    if len(js_participating) < 3:
        return None, None, len(js_participating)

    # Concept criticality: how many holdings killed if removed
    holdings_with_js = [h_id for h_id, jsg in holding_js.items() if jsg]

    concept_criticality = {}
    for c_id in js_participating:
        holdings_killed = 0
        for h_id in holdings_with_js:
            js_group = holding_js[h_id]
            all_killed = True
            for js in js_group:
                js_id = js["id"]
                logic = js.get("logic", "and")
                members = js_members.get(js_id, set()) & concept_ids
                if logic == "and" and c_id in members:
                    continue  # This AND JS broken
                elif logic == "or":
                    remaining = members - {c_id}
                    if remaining:
                        all_killed = False
                        break
                    else:
                        continue  # Only member, OR JS also broken
                else:
                    all_killed = False
                    break
            if all_killed and js_group:
                holdings_killed += 1
        concept_criticality[c_id] = holdings_killed

    # Most critical
    critical = max(concept_criticality, key=concept_criticality.get)

    # Random (not the critical one)
    rng = random.Random(seed)
    candidates = sorted(js_participating - {critical})
    if not candidates:
        return None, None, len(js_participating)
    rand_concept = rng.choice(candidates)

    return critical, rand_concept, len(js_participating)


def remove_concept_from_graph(graph_dict, concept_id_to_remove):
    """Return a modified copy of graph_dict with one concept and its edges removed."""
    g = copy.deepcopy(graph_dict)

    # Remove concept node
    g["concepts"] = [c for c in (g.get("concepts") or []) if c.get("id") != concept_id_to_remove]

    # Remove edges involving this concept
    g["edges"] = [
        e for e in (g.get("edges") or [])
        if e.get("source") != concept_id_to_remove and e.get("target") != concept_id_to_remove
    ]

    return g


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Counterfactual prediction flip test")
    parser.add_argument("--concurrent", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cases", type=int, default=0, help="0 = all eligible")
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
    print(f"  {len(intact_cases)} cases with intact predictions")

    # Filter to correct graph predictions
    correct_cases = {
        cid: r for cid, r in intact_cases.items()
        if r["graph_pred"] == r["true_label"] and r["graph_pred"] in (0, 1)
    }
    print(f"  {len(correct_cases)} with correct graph predictions")

    # 2. Load graphs and find critical/random concepts
    print(f"\nAnalyzing counterfactual structure...")
    eligible = []
    concept_labels = {}  # concept node id -> concept_id label

    for cid in sorted(correct_cases.keys()):
        gpath = GRAPH_DIR / f"{cid}.json"
        if not gpath.exists():
            continue
        with open(gpath, "r", encoding="utf-8") as f:
            g = json.load(f)

        critical, rand_c, n_participating = find_critical_and_random_concepts(g, seed=args.seed)
        if critical is None or n_participating < 3:
            continue

        # Get concept labels
        for c in g.get("concepts", []):
            concept_labels[c["id"]] = c.get("concept_id", c["id"])

        # Build modified graph prompts
        g_targeted = remove_concept_from_graph(g, critical)
        g_random = remove_concept_from_graph(g, rand_c)

        prompt_targeted = build_graph_prompt(g_targeted, no_scrub=True)
        prompt_random = build_graph_prompt(g_random, no_scrub=True)

        eligible.append({
            "case_id": cid,
            "true_label": correct_cases[cid]["true_label"],
            "intact_pred": correct_cases[cid]["graph_pred"],
            "intact_conf": correct_cases[cid]["graph_conf"],
            "critical_concept_id": critical,
            "critical_concept_label": concept_labels.get(critical, critical),
            "random_concept_id": rand_c,
            "random_concept_label": concept_labels.get(rand_c, rand_c),
            "n_js_participating": n_participating,
            "prompt_targeted": prompt_targeted,
            "prompt_random": prompt_random,
        })

    if args.max_cases > 0:
        eligible = eligible[:args.max_cases]

    print(f"  {len(eligible)} eligible cases (correct pred + >=3 JS-participating concepts)")

    total_api_calls = len(eligible) * 2
    est_cost = total_api_calls * 3000 * 0.20 / 1e6 + total_api_calls * 1000 * 0.50 / 1e6
    print(f"  API calls needed: {total_api_calls} (est. cost: ${est_cost:.2f})")

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

        if tp not in (0, 1):
            errors += 1
        if rp not in (0, 1):
            errors += 1

        true_label = case["true_label"]
        targeted_flipped = (tp != true_label) and tp in (0, 1)
        random_flipped = (rp != true_label) and rp in (0, 1)

        tag = ""
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
                  f"crit={case['critical_concept_label'][:30]}")

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
            "critical_concept": case["critical_concept_label"],
            "random_concept": case["random_concept_label"],
            "n_js_participating": case["n_js_participating"],
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

    # 4. Compute metrics
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

    # Top concepts causing targeted flips
    flip_concepts = Counter()
    for r in valid:
        if r["targeted_flipped"]:
            flip_concepts[r["critical_concept"]] += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"COUNTERFACTUAL PREDICTION FLIP TEST")
    print(f"{'='*70}")
    print(f"  Cases tested:                   {n}")
    print(f"  (all had correct intact predictions)")
    print(f"  Elapsed:                        {elapsed:.0f}s")
    print(f"  API errors:                     {errors}")

    print(f"\n  FLIP RATES:")
    print(f"  {'Targeted removal flip rate:':<35} {targeted_flip_rate:.4f} ({targeted_flips}/{n})")
    print(f"  {'Random removal flip rate:':<35} {random_flip_rate:.4f} ({random_flips}/{n})")
    print(f"  {'Flip ratio (targeted/random):':<35} {flip_ratio:.2f}x")

    print(f"\n  ACCURACY AFTER REMOVAL:")
    print(f"  {'Intact (baseline):':<35} 1.000 (by construction)")
    print(f"  {'After targeted removal:':<35} {targeted_acc:.4f}")
    print(f"  {'After random removal:':<35} {random_acc:.4f}")

    print(f"\n  CONCORDANCE TABLE:")
    print(f"  {'Neither flipped:':<35} {neither:>5} ({neither/n*100:.1f}%)")
    print(f"  {'Targeted only flipped:':<35} {targeted_only:>5} ({targeted_only/n*100:.1f}%)")
    print(f"  {'Random only flipped:':<35} {random_only:>5} ({random_only/n*100:.1f}%)")
    print(f"  {'Both flipped:':<35} {both_flips:>5} ({both_flips/n*100:.1f}%)")

    print(f"\n  TOP 10 CONCEPTS CAUSING TARGETED FLIPS:")
    for concept, count in flip_concepts.most_common(10):
        print(f"    {concept:<45s} {count:>4} flips")

    # 5. Save results
    output = {
        "config": {
            "model": model,
            "n_cases": n,
            "seed": args.seed,
            "graph_dir": str(GRAPH_DIR),
            "intact_results": str(INTACT_RESULTS),
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
            "top_flip_concepts": flip_concepts.most_common(20),
        },
        "cases": valid,
    }

    out_path = Path("counterfactual_flip_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
