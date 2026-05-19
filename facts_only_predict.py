#!/usr/bin/env python3
"""
Step 4: Predict on 50 cases with 3 conditions:
1. Raw facts text (no graph)
2. Facts-only graph
3. Full graph (from graphs_enriched_v2/)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from eval_graph_vs_raw import (
    build_graph_prompt, build_raw_prompt,
    llm_predict, _SYSTEM_PROMPT,
)

FACTS_TEXT_DIR = Path("facts_only_texts")
FACTS_GRAPH_DIR = Path("facts_only_graphs")
FULL_GRAPH_DIR = Path("graphs_enriched_v2")
MANIFEST = FACTS_TEXT_DIR / "manifest.json"


async def main():
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set")
        return

    model = "grok-4-1-fast-reasoning"

    # Load manifest
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Cases: {len(manifest)}")

    # Build prompts for all 3 conditions
    cases = []
    for entry in manifest:
        cid = entry["case_id"]
        label = entry["label"]

        # 1. Raw facts text
        facts_path = FACTS_TEXT_DIR / f"{cid}.txt"
        if not facts_path.exists():
            continue
        with open(facts_path, "r", encoding="utf-8") as f:
            facts_text = f.read()

        raw_prompt = build_raw_prompt(facts_text)

        # 2. Facts-only graph
        fg_path = FACTS_GRAPH_DIR / f"{cid}.json"
        if not fg_path.exists():
            continue
        with open(fg_path, "r", encoding="utf-8") as f:
            facts_graph = json.load(f)
        facts_graph_prompt = build_graph_prompt(facts_graph, no_scrub=True)

        # 3. Full graph
        full_path = FULL_GRAPH_DIR / f"{cid}.json"
        if not full_path.exists():
            continue
        with open(full_path, "r", encoding="utf-8") as f:
            full_graph = json.load(f)
        full_graph_prompt = build_graph_prompt(full_graph, no_scrub=True)

        cases.append({
            "case_id": cid,
            "label": label,
            "raw_prompt": raw_prompt,
            "facts_graph_prompt": facts_graph_prompt,
            "full_graph_prompt": full_graph_prompt,
        })

    print(f"Cases with all 3 conditions: {len(cases)}")

    # Run all predictions (3 per case = 150 calls)
    semaphore = asyncio.Semaphore(50)
    results = []
    t0 = time.time()

    async def predict_case(case, idx):
        async with semaphore:
            raw_result = await llm_predict(api_key, _SYSTEM_PROMPT, case["raw_prompt"], model=model)
            fg_result = await llm_predict(api_key, _SYSTEM_PROMPT, case["facts_graph_prompt"], model=model)
            full_result = await llm_predict(api_key, _SYSTEM_PROMPT, case["full_graph_prompt"], model=model)

        label = case["label"]
        rp = raw_result.get("prediction", -1)
        fgp = fg_result.get("prediction", -1)
        fp = full_result.get("prediction", -1)

        r_ok = "Y" if rp == label else "N"
        fg_ok = "Y" if fgp == label else "N"
        f_ok = "Y" if fp == label else "N"

        label_str = "ACC" if label == 1 else "REJ"
        print(f"  [{idx:2d}] {case['case_id']:12s} TRUE={label_str}  "
              f"raw={r_ok} facts_graph={fg_ok} full_graph={f_ok}")

        return {
            "case_id": case["case_id"],
            "true_label": label,
            "raw_pred": rp,
            "raw_conf": raw_result.get("confidence", 0),
            "facts_graph_pred": fgp,
            "facts_graph_conf": fg_result.get("confidence", 0),
            "full_graph_pred": fp,
            "full_graph_conf": full_result.get("confidence", 0),
        }

    tasks = [asyncio.create_task(predict_case(c, i)) for i, c in enumerate(cases)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - t0

    # Compute metrics
    valid = [r for r in results if all(r[k] in (0, 1) for k in ["raw_pred", "facts_graph_pred", "full_graph_pred"])]
    n = len(valid)

    raw_correct = sum(1 for r in valid if r["raw_pred"] == r["true_label"])
    fg_correct = sum(1 for r in valid if r["facts_graph_pred"] == r["true_label"])
    full_correct = sum(1 for r in valid if r["full_graph_pred"] == r["true_label"])

    raw_acc = raw_correct / n if n else 0
    fg_acc = fg_correct / n if n else 0
    full_acc = full_correct / n if n else 0

    print(f"\n{'='*60}")
    print(f"FACTS-ONLY EXPERIMENT RESULTS (n={n})")
    print(f"{'='*60}")
    print(f"  {'Condition':<25} {'Accuracy':>10} {'Correct':>8}")
    print(f"  {'-'*45}")
    print(f"  {'Raw facts text':<25} {raw_acc:>10.4f} {raw_correct:>5}/{n}")
    print(f"  {'Facts-only graph':<25} {fg_acc:>10.4f} {fg_correct:>5}/{n}")
    print(f"  {'Full graph':<25} {full_acc:>10.4f} {full_correct:>5}/{n}")
    print(f"\n  Delta (facts_graph - raw): {fg_acc - raw_acc:+.4f}")
    print(f"  Delta (full_graph - raw):  {full_acc - raw_acc:+.4f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Win/loss table
    print(f"\n  FACTS_GRAPH vs RAW:")
    fg_wins = sum(1 for r in valid if r["facts_graph_pred"] == r["true_label"] and r["raw_pred"] != r["true_label"])
    raw_wins = sum(1 for r in valid if r["raw_pred"] == r["true_label"] and r["facts_graph_pred"] != r["true_label"])
    both_ok = sum(1 for r in valid if r["facts_graph_pred"] == r["true_label"] and r["raw_pred"] == r["true_label"])
    both_bad = sum(1 for r in valid if r["facts_graph_pred"] != r["true_label"] and r["raw_pred"] != r["true_label"])
    print(f"    Facts-graph wins: {fg_wins}  Raw wins: {raw_wins}  Both correct: {both_ok}  Both wrong: {both_bad}")

    # Save
    output = {
        "config": {"model": model, "n_cases": n, "elapsed": round(elapsed, 1)},
        "summary": {
            "raw_accuracy": round(raw_acc, 4),
            "facts_graph_accuracy": round(fg_acc, 4),
            "full_graph_accuracy": round(full_acc, 4),
        },
        "cases": valid,
    }
    with open("facts_only_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to facts_only_results.json")


if __name__ == "__main__":
    asyncio.run(main())
