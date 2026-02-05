#!/usr/bin/env python3
"""
eval_diagnose.py - Side-by-side diagnostic of zero-shot vs retrieval predictions.

Runs both methods on the SAME cases and dumps:
1. What the LLM actually sees (full prompts)
2. Neighbor quality (similarity scores, labels, shared concepts)
3. Full reasoning from both methods
4. Where they agree/disagree and who's right

Usage:
    python eval_diagnose.py --graph_dir iltur_graphs --n 10 --k 5
"""

from __future__ import annotations
import argparse, asyncio, json, os, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv

from eval_concept_retrieval import (
    load_json, iter_graphs, load_labels_hf,
    extract_concept_profile, compute_idf_multi,
    FuzzyConceptIndex, CaseTextSimilarity, combined_similarity,
)
from eval_hybrid import (
    _compact_graph_summary,
    _build_zeroshot_prompt,
    _build_prediction_prompt,
    ConceptNeighborIndex,
    llm_predict,
)

load_dotenv()


def fmt_label(label):
    return "ACC" if label == 1 else "REJ"


async def run_diagnostic(graph_dir: str, n_cases: int = 10, k: int = 5,
                         model: str = "grok-4-1-fast-reasoning"):
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    graphs = iter_graphs(Path(graph_dir))
    labels = load_labels_hf()
    corpus = [(extract_concept_profile(g, labels[c]), g) for c, g in graphs if c in labels]
    n = len(corpus)

    # Build concept index
    concept_index = ConceptNeighborIndex(corpus)

    # Same stratified sample as eval_hybrid
    rng = np.random.RandomState(42)
    acc_idx = [i for i, (p, _) in enumerate(corpus) if p.label == 1]
    rej_idx = [i for i, (p, _) in enumerate(corpus) if p.label == 0]
    n_acc = max(1, int(n_cases * len(acc_idx) / n))
    n_rej = n_cases - n_acc
    selected = sorted(
        list(rng.choice(acc_idx, min(n_acc, len(acc_idx)), replace=False)) +
        list(rng.choice(rej_idx, min(n_rej, len(rej_idx)), replace=False))
    )

    print(f"\nDiagnostic: {len(selected)} cases, k={k}, model={model}")
    print(f"=" * 80)

    results = []

    for case_num, idx in enumerate(selected, 1):
        qp, qg = corpus[idx]
        true_label = qp.label

        print(f"\n{'#' * 80}")
        print(f"# CASE {case_num}/{len(selected)}: {qp.case_id}  (TRUE: {fmt_label(true_label)})")
        print(f"{'#' * 80}")

        # ── Show blinded query summary ──
        blinded_summary = _compact_graph_summary(qg, max_facts=6, max_args=6,
                                                  max_holdings=0, max_precedents=4,
                                                  blind=True)
        print(f"\n{'─' * 40}")
        print("BLINDED QUERY (what both methods see):")
        print(f"{'─' * 40}")
        print(blinded_summary[:800])
        if len(blinded_summary) > 800:
            print(f"  ... [{len(blinded_summary)} chars total]")

        # ── Get neighbors ──
        nbr_ids = concept_index.get_neighbors(qp.case_id, k=k)
        neighbor_data = []
        for nbr_cid, sim_score in nbr_ids:
            for p, g in corpus:
                if p.case_id == nbr_cid:
                    neighbor_data.append((g, p.label, sim_score))
                    break

        print(f"\n{'─' * 40}")
        print(f"RETRIEVED NEIGHBORS ({k} by concept similarity):")
        print(f"{'─' * 40}")
        nbr_labels = []
        for i, (ng, nlabel, nsim) in enumerate(neighbor_data, 1):
            nbr_labels.append(nlabel)
            # Count how much court behavior info is available
            n_court_resp = sum(1 for a in (ng.get("arguments") or [])
                             if isinstance(a, dict) and a.get("court_response"))
            n_holdings = len(ng.get("holdings") or [])
            n_issue_answers = sum(1 for iss in (ng.get("issues") or [])
                                 if isinstance(iss, dict) and iss.get("answer"))

            print(f"  {i}. {ng.get('case_id', '?'):12s}  sim={nsim:.3f}  "
                  f"label={fmt_label(nlabel)}  "
                  f"court_resp={n_court_resp}  holdings={n_holdings}  "
                  f"issue_ans={n_issue_answers}")

            # Show what court behavior looks like for this neighbor
            nbr_summary = _compact_graph_summary(ng, max_facts=2, max_args=3,
                                                  max_holdings=2, max_precedents=1,
                                                  hide_outcome=True)
            # Just show argument court responses and holdings
            for line in nbr_summary.split("\n"):
                if "Court:" in line or "HOLDINGS:" in line or (line.startswith("  ") and "Reasoning:" in line):
                    print(f"       {line.strip()}")

        n_acc_nbrs = sum(1 for l in nbr_labels if l == 1)
        n_rej_nbrs = sum(1 for l in nbr_labels if l == 0)
        print(f"\n  Neighbor distribution: {n_acc_nbrs} ACC / {n_rej_nbrs} REJ")
        majority_nbr = "ACC" if n_acc_nbrs > n_rej_nbrs else "REJ"
        print(f"  Neighbor majority vote: {majority_nbr}")

        # ── Run zero-shot ──
        zs_system, zs_prompt = _build_zeroshot_prompt(qg)
        zs_result = await llm_predict(api_key, zs_system, zs_prompt, model=model)
        zs_pred = zs_result.get("prediction", -1)
        zs_conf = zs_result.get("confidence", 0)
        zs_reasoning = zs_result.get("reasoning", "")

        # ── Run concept behavior ──
        cb_system, cb_prompt = _build_prediction_prompt(qg, neighbor_data, behavior=True)
        cb_result = await llm_predict(api_key, cb_system, cb_prompt, model=model)
        cb_pred = cb_result.get("prediction", -1)
        cb_conf = cb_result.get("confidence", 0)
        cb_reasoning = cb_result.get("reasoning", "")

        # ── Run concept hybrid (with outcome labels) ──
        ch_system, ch_prompt = _build_prediction_prompt(qg, neighbor_data, behavior=False)
        ch_result = await llm_predict(api_key, ch_system, ch_prompt, model=model)
        ch_pred = ch_result.get("prediction", -1)
        ch_conf = ch_result.get("confidence", 0)
        ch_reasoning = ch_result.get("reasoning", "")

        # ── Print results ──
        print(f"\n{'─' * 40}")
        print("PREDICTIONS:")
        print(f"{'─' * 40}")

        def status(pred):
            return "✓" if pred == true_label else "✗"

        print(f"\n  ZERO-SHOT:         {status(zs_pred)} {fmt_label(zs_pred)} (conf={zs_conf:.2f})")
        print(f"    Reasoning: {zs_reasoning}")

        print(f"\n  CONCEPT+BEHAVIOR:  {status(cb_pred)} {fmt_label(cb_pred)} (conf={cb_conf:.2f})")
        print(f"    Reasoning: {cb_reasoning}")

        print(f"\n  CONCEPT+OUTCOMES:  {status(ch_pred)} {fmt_label(ch_pred)} (conf={ch_conf:.2f})")
        print(f"    Reasoning: {ch_reasoning}")

        # Flag interesting patterns
        if zs_pred == true_label and cb_pred != true_label:
            print(f"\n  ⚠️  NEIGHBORS HURT: zero-shot correct, behavior wrong")
        elif zs_pred != true_label and cb_pred == true_label:
            print(f"\n  ✨ NEIGHBORS HELPED: zero-shot wrong, behavior correct")
        elif zs_pred != true_label and cb_pred != true_label:
            print(f"\n  ❌ BOTH WRONG")
        else:
            print(f"\n  ✅ BOTH CORRECT")

        results.append({
            "case_id": qp.case_id,
            "true_label": true_label,
            "zs_pred": zs_pred, "zs_conf": zs_conf, "zs_reasoning": zs_reasoning,
            "cb_pred": cb_pred, "cb_conf": cb_conf, "cb_reasoning": cb_reasoning,
            "ch_pred": ch_pred, "ch_conf": ch_conf, "ch_reasoning": ch_reasoning,
            "nbr_labels": nbr_labels,
            "nbr_sims": [s for _, _, s in neighbor_data],
        })

    # ── Summary ──
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    zs_correct = sum(1 for r in results if r["zs_pred"] == r["true_label"])
    cb_correct = sum(1 for r in results if r["cb_pred"] == r["true_label"])
    ch_correct = sum(1 for r in results if r["ch_pred"] == r["true_label"])
    n_total = len(results)

    print(f"\n  Zero-shot:         {zs_correct}/{n_total} = {zs_correct/n_total:.3f}")
    print(f"  Concept+behavior:  {cb_correct}/{n_total} = {cb_correct/n_total:.3f}")
    print(f"  Concept+outcomes:  {ch_correct}/{n_total} = {ch_correct/n_total:.3f}")

    nbr_helped = sum(1 for r in results if r["zs_pred"] != r["true_label"] and r["cb_pred"] == r["true_label"])
    nbr_hurt = sum(1 for r in results if r["zs_pred"] == r["true_label"] and r["cb_pred"] != r["true_label"])
    both_right = sum(1 for r in results if r["zs_pred"] == r["true_label"] and r["cb_pred"] == r["true_label"])
    both_wrong = sum(1 for r in results if r["zs_pred"] != r["true_label"] and r["cb_pred"] != r["true_label"])

    print(f"\n  Neighbors helped:  {nbr_helped}")
    print(f"  Neighbors hurt:    {nbr_hurt}")
    print(f"  Both correct:      {both_right}")
    print(f"  Both wrong:        {both_wrong}")

    # Neighbor quality analysis
    print(f"\n  Avg neighbor similarity: {np.mean([s for r in results for s in r['nbr_sims']]):.3f}")
    print(f"  Avg neighbor label match with query:")
    match_rates = []
    for r in results:
        matches = sum(1 for nl in r["nbr_labels"] if nl == r["true_label"])
        match_rates.append(matches / len(r["nbr_labels"]))
    print(f"    {np.mean(match_rates):.3f} (1.0 = all neighbors same label as query)")

    # Save full results
    out_path = Path("diagnose_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Diagnostic comparison of prediction methods")
    p.add_argument("--graph_dir", required=True)
    p.add_argument("--n", type=int, default=10, help="Number of cases to diagnose")
    p.add_argument("--k", type=int, default=5, help="Number of neighbors")
    p.add_argument("--model", default="grok-4-1-fast-reasoning")
    args = p.parse_args()

    asyncio.run(run_diagnostic(args.graph_dir, n_cases=args.n, k=args.k, model=args.model))


if __name__ == "__main__":
    main()
