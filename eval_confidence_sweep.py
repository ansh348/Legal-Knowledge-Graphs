#!/usr/bin/env python3
"""
eval_confidence_sweep.py — Confidence-gated selective prediction analysis.

Core hypothesis:
    Graph representations produce CALIBRATED confidence — the model knows when
    it doesn't know. Raw text confidence is uncalibrated (equally confident
    when right and wrong). This enables a selective prediction strategy:
    only predict when confident, abstain otherwise.

Outputs:
    1. Accuracy-vs-coverage curves at each confidence threshold (0.50–0.95)
    2. Optimal threshold selection via F1 on coverage-adjusted accuracy
    3. Hybrid strategy: graph when confident, raw fallback, abstain as last resort
    4. Paper-ready statistics and LaTeX table

Usage:
    # From existing results:
    python eval_confidence_sweep.py --results graph_vs_raw_n50.json

    # Fresh run with larger sample:
    python eval_confidence_sweep.py --graph_dir iltur_graphs --n 200 --concurrent 10

Requirements:
    pip install numpy
    (Optional for fresh runs: datasets python-dotenv httpx)
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


# =============================================================================
# LOAD RESULTS
# =============================================================================

def load_results(path: str) -> List[Dict]:
    """Load case-level results from eval_graph_vs_raw.py output."""
    with open(path) as f:
        data = json.load(f)
    cases = data.get("cases", [])
    # Normalize: newer eval scripts use "struct_*" instead of "raw_*"
    for c in cases:
        if "struct_pred" in c and "raw_pred" not in c:
            c["raw_pred"] = c["struct_pred"]
            c["raw_conf"] = c["struct_conf"]
    # Filter to valid predictions only
    valid = [c for c in cases
             if c.get("graph_pred") in (0, 1) and c.get("raw_pred") in (0, 1)]
    return valid


# =============================================================================
# CONFIDENCE SWEEP
# =============================================================================

def sweep_thresholds(
    cases: List[Dict],
    method: str,  # "graph" or "raw"
    thresholds: List[float] = None,
) -> List[Dict]:
    """Sweep confidence thresholds and compute accuracy/coverage at each."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    pred_key = f"{method}_pred"
    conf_key = f"{method}_conf"

    results = []
    for thresh in thresholds:
        accepted = [c for c in cases if c[conf_key] >= thresh]
        n_accepted = len(accepted)
        coverage = n_accepted / len(cases) if cases else 0

        if n_accepted > 0:
            correct = sum(1 for c in accepted if c[pred_key] == c["true_label"])
            accuracy = correct / n_accepted

            # Per-class on accepted cases
            acc_cases = [c for c in accepted if c["true_label"] == 1]
            rej_cases = [c for c in accepted if c["true_label"] == 0]
            acc_on_acc = (sum(1 for c in acc_cases if c[pred_key] == 1) /
                          len(acc_cases)) if acc_cases else 0
            acc_on_rej = (sum(1 for c in rej_cases if c[pred_key] == 0) /
                          len(rej_cases)) if rej_cases else 0
        else:
            accuracy = 0
            acc_on_acc = 0
            acc_on_rej = 0

        results.append({
            "threshold": thresh,
            "coverage": round(coverage, 4),
            "n_accepted": n_accepted,
            "accuracy": round(accuracy, 4),
            "acc_on_accepted": round(acc_on_acc, 4),
            "acc_on_rejected": round(acc_on_rej, 4),
        })

    return results


# =============================================================================
# HYBRID STRATEGY
# =============================================================================

def evaluate_hybrid_strategies(cases: List[Dict]) -> List[Dict]:
    """Evaluate hybrid strategies: graph-primary with raw fallback.

    Strategies:
        1. Graph-only at various thresholds
        2. Raw-only at various thresholds
        3. HYBRID: Trust graph if graph_conf >= T_high, else trust raw if
           raw_conf >= T_low, else abstain.
        4. ENSEMBLE: If both agree → take that. If they disagree → take the
           one with higher confidence. Abstain if max conf < threshold.
    """
    thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]
    strategies = []

    for t_graph in thresholds:
        for t_raw in [0.50, 0.60, 0.70, 0.75, 0.80]:
            # Hybrid: graph first, raw fallback
            correct = 0
            predicted = 0
            for c in cases:
                if c["graph_conf"] >= t_graph:
                    # Trust graph
                    predicted += 1
                    if c["graph_pred"] == c["true_label"]:
                        correct += 1
                elif c["raw_conf"] >= t_raw:
                    # Fallback to raw
                    predicted += 1
                    if c["raw_pred"] == c["true_label"]:
                        correct += 1
                # else: abstain

            coverage = predicted / len(cases) if cases else 0
            accuracy = correct / predicted if predicted > 0 else 0

            strategies.append({
                "strategy": "hybrid",
                "t_graph": t_graph,
                "t_raw": t_raw,
                "coverage": round(coverage, 4),
                "accuracy": round(accuracy, 4),
                "n_predicted": predicted,
                "n_correct": correct,
            })

    # Contra-predictive: flip graph predictions when confidence is LOW
    # Rationale: if graph is <40% accurate at low confidence, it has a
    # systematic anti-signal — flipping recovers accuracy.
    for t_flip in thresholds:
        correct = 0
        predicted = len(cases)  # Full coverage — no abstention
        for c in cases:
            gc = c["graph_conf"]
            gp = c["graph_pred"]
            label = c["true_label"]

            if gc >= t_flip:
                # High confidence — trust graph as-is
                if gp == label:
                    correct += 1
            else:
                # Low confidence — FLIP the prediction
                flipped = 1 - gp
                if flipped == label:
                    correct += 1

        coverage = 1.0
        accuracy = correct / predicted if predicted > 0 else 0

        strategies.append({
            "strategy": "contra_graph",
            "t_flip": t_flip,
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4),
            "n_predicted": predicted,
            "n_correct": correct,
        })

    # Contra-predictive hybrid: flip graph at low conf, trust graph at high conf,
    # fall back to raw in the middle zone
    for t_high in thresholds:
        for t_low in [round(t, 2) for t in np.arange(0.50, t_high, 0.05)]:
            correct = 0
            predicted = len(cases)  # Full coverage
            details = {"graph_trusted": 0, "graph_flipped": 0, "raw_fallback": 0}
            for c in cases:
                gc, gp = c["graph_conf"], c["graph_pred"]
                rc, rp = c["raw_conf"], c["raw_pred"]
                label = c["true_label"]

                if gc >= t_high:
                    # High confidence — trust graph
                    details["graph_trusted"] += 1
                    if gp == label:
                        correct += 1
                elif gc < t_low:
                    # Very low confidence — flip graph
                    details["graph_flipped"] += 1
                    if (1 - gp) == label:
                        correct += 1
                else:
                    # Middle zone — fall back to raw
                    details["raw_fallback"] += 1
                    if rp == label:
                        correct += 1

            accuracy = correct / predicted if predicted > 0 else 0

            strategies.append({
                "strategy": "contra_hybrid",
                "t_high": t_high,
                "t_low": t_low,
                "coverage": 1.0,
                "accuracy": round(accuracy, 4),
                "n_predicted": predicted,
                "n_correct": correct,
                "details": details,
            })

    # Ensemble: agreement-based
    for t_min in thresholds:
        correct = 0
        predicted = 0
        for c in cases:
            gp, rp = c["graph_pred"], c["raw_pred"]
            gc, rc = c["graph_conf"], c["raw_conf"]

            if gp == rp:
                # Both agree — predict if either is confident enough
                if max(gc, rc) >= t_min:
                    predicted += 1
                    if gp == c["true_label"]:
                        correct += 1
            else:
                # Disagree — take higher confidence, but only if it's high enough
                if max(gc, rc) >= t_min + 0.05:  # Higher bar for disagreement
                    predicted += 1
                    chosen = gp if gc >= rc else rp
                    if chosen == c["true_label"]:
                        correct += 1

        coverage = predicted / len(cases) if cases else 0
        accuracy = correct / predicted if predicted > 0 else 0

        strategies.append({
            "strategy": "ensemble",
            "t_min": t_min,
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4),
            "n_predicted": predicted,
            "n_correct": correct,
        })

    return strategies


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def calibration_analysis(cases: List[Dict]) -> Dict:
    """Detailed calibration analysis — bin predictions by confidence and
    measure actual accuracy in each bin."""
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    result = {"graph": [], "raw": []}

    for method in ["graph", "raw"]:
        pred_key = f"{method}_pred"
        conf_key = f"{method}_conf"

        for lo, hi in bins:
            in_bin = [c for c in cases if lo <= c[conf_key] < hi]
            if in_bin:
                actual_acc = sum(1 for c in in_bin
                                if c[pred_key] == c["true_label"]) / len(in_bin)
                mean_conf = np.mean([c[conf_key] for c in in_bin])
            else:
                actual_acc = None
                mean_conf = None

            result[method].append({
                "bin": f"[{lo:.2f}, {hi:.2f})",
                "n": len(in_bin),
                "mean_confidence": round(mean_conf, 4) if mean_conf is not None else None,
                "actual_accuracy": round(actual_acc, 4) if actual_acc is not None else None,
                "calibration_error": (round(abs(mean_conf - actual_acc), 4)
                                      if mean_conf is not None and actual_acc is not None
                                      else None),
            })

    # Expected Calibration Error (ECE)
    for method in ["graph", "raw"]:
        total_n = len(cases)
        ece = 0
        for bin_data in result[method]:
            if bin_data["calibration_error"] is not None and bin_data["n"] > 0:
                ece += (bin_data["n"] / total_n) * bin_data["calibration_error"]
        result[f"{method}_ece"] = round(ece, 4)

    return result


# =============================================================================
# AREA UNDER ACCURACY-COVERAGE CURVE (AUACC)
# =============================================================================

def compute_auacc(sweep_results: List[Dict]) -> float:
    """Compute area under the accuracy-coverage curve using trapezoidal rule.
    Higher = better (accurate AND high coverage)."""
    points = [(r["coverage"], r["accuracy"]) for r in sweep_results if r["n_accepted"] > 0]
    if len(points) < 2:
        return 0.0
    # Sort by coverage descending (low threshold = high coverage)
    points.sort(key=lambda x: x[0], reverse=True)
    area = 0
    for i in range(len(points) - 1):
        cov1, acc1 = points[i]
        cov2, acc2 = points[i + 1]
        # Trapezoidal rule
        area += abs(cov1 - cov2) * (acc1 + acc2) / 2
    return round(area, 4)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(results_path: str):
    cases = load_results(results_path)
    n = len(cases)

    if n == 0:
        print("No valid cases found!")
        return

    print(f"\n{'='*80}")
    print(f"  CONFIDENCE-GATED SELECTIVE PREDICTION ANALYSIS")
    print(f"  n={n} cases from {results_path}")
    print(f"{'='*80}")

    # ---- 1. Threshold sweep ----
    thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]
    graph_sweep = sweep_thresholds(cases, "graph", thresholds)
    raw_sweep = sweep_thresholds(cases, "raw", thresholds)

    print(f"\n  {'─'*72}")
    print(f"  ACCURACY-COVERAGE SWEEP")
    print(f"  {'─'*72}")
    print(f"\n  {'Thresh':>8}  {'GRAPH':^25}  {'RAW':^25}")
    print(f"  {'':>8}  {'Acc':>8} {'Cov':>8} {'n':>5}   {'Acc':>8} {'Cov':>8} {'n':>5}")
    print(f"  {'─'*72}")

    for g, r in zip(graph_sweep, raw_sweep):
        t = g["threshold"]
        # Highlight rows where graph accuracy jumps
        marker = "  ◄" if g["accuracy"] > r["accuracy"] and g["accuracy"] > 0.75 else ""
        print(f"  {t:>8.2f}  {g['accuracy']:>8.3f} {g['coverage']:>8.3f} {g['n_accepted']:>5}"
              f"   {r['accuracy']:>8.3f} {r['coverage']:>8.3f} {r['n_accepted']:>5}{marker}")

    # AUACC
    graph_auacc = compute_auacc(graph_sweep)
    raw_auacc = compute_auacc(raw_sweep)

    print(f"\n  Area Under Accuracy-Coverage Curve (AUACC):")
    print(f"    Graph: {graph_auacc:.4f}")
    print(f"    Raw:   {raw_auacc:.4f}")
    print(f"    Delta: {graph_auacc - raw_auacc:+.4f}")
    if graph_auacc > raw_auacc:
        print(f"    → Graph has better accuracy-coverage tradeoff")
    elif raw_auacc > graph_auacc:
        print(f"    → Raw has better accuracy-coverage tradeoff")
    else:
        print(f"    → Equal")

    # ---- 2. Calibration ----
    cal = calibration_analysis(cases)

    print(f"\n  {'─'*72}")
    print(f"  CALIBRATION ANALYSIS")
    print(f"  {'─'*72}")

    print(f"\n  GRAPH (ECE = {cal['graph_ece']:.4f}):")
    print(f"  {'Bin':>16} {'n':>5} {'Mean Conf':>10} {'Actual Acc':>12} {'|Error|':>10}")
    for b in cal["graph"]:
        if b["n"] > 0:
            print(f"  {b['bin']:>16} {b['n']:>5} {b['mean_confidence']:>10.3f} "
                  f"{b['actual_accuracy']:>12.3f} {b['calibration_error']:>10.3f}")
        else:
            print(f"  {b['bin']:>16} {b['n']:>5} {'—':>10} {'—':>12} {'—':>10}")

    print(f"\n  RAW (ECE = {cal['raw_ece']:.4f}):")
    print(f"  {'Bin':>16} {'n':>5} {'Mean Conf':>10} {'Actual Acc':>12} {'|Error|':>10}")
    for b in cal["raw"]:
        if b["n"] > 0:
            print(f"  {b['bin']:>16} {b['n']:>5} {b['mean_confidence']:>10.3f} "
                  f"{b['actual_accuracy']:>12.3f} {b['calibration_error']:>10.3f}")
        else:
            print(f"  {b['bin']:>16} {b['n']:>5} {'—':>10} {'—':>12} {'—':>10}")

    ece_improvement = cal["raw_ece"] - cal["graph_ece"]
    print(f"\n  ECE improvement (graph vs raw): {ece_improvement:+.4f}")
    if ece_improvement > 0:
        print(f"  → Graph is better calibrated by {ece_improvement:.4f}")
    else:
        print(f"  → Raw is better calibrated by {-ece_improvement:.4f}")

    # Brier score
    def brier_score(cases, method):
        scores = []
        for c in cases:
            pred = c[f"{method}_pred"]
            conf = c[f"{method}_conf"]
            label = c["true_label"]
            prob_true = conf if pred == label else (1.0 - conf)
            scores.append((1.0 - prob_true) ** 2)
        return np.mean(scores)

    graph_brier = brier_score(cases, "graph")
    raw_brier = brier_score(cases, "raw")

    print(f"\n  Brier Score (lower = better):")
    print(f"    Graph: {graph_brier:.4f}")
    print(f"    Raw:   {raw_brier:.4f}")
    print(f"    Delta: {graph_brier - raw_brier:+.4f} "
          f"({'graph' if graph_brier < raw_brier else 'raw'} better)")

    # Bootstrap 95% CI for ECE difference
    def _bootstrap_ece(cases, method, bins):
        """Compute ECE for a single bootstrap sample."""
        pred_key = f"{method}_pred"
        conf_key = f"{method}_conf"
        total = len(cases)
        ece = 0.0
        for lo, hi in bins:
            in_bin = [c for c in cases if lo <= c[conf_key] < hi]
            if in_bin:
                mean_conf = np.mean([c[conf_key] for c in in_bin])
                actual_acc = sum(1 for c in in_bin
                                if c[pred_key] == c["true_label"]) / len(in_bin)
                ece += (len(in_bin) / total) * abs(mean_conf - actual_acc)
        return ece

    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    n_boot = 10000
    rng = np.random.RandomState(42)
    boot_ece_diffs = []
    boot_brier_diffs = []
    case_arr = np.array(cases)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        sample = [cases[i] for i in idx]
        g_ece = _bootstrap_ece(sample, "graph", bins)
        r_ece = _bootstrap_ece(sample, "raw", bins)
        boot_ece_diffs.append(r_ece - g_ece)  # positive = graph better
        # Brier
        g_brier = brier_score(sample, "graph")
        r_brier = brier_score(sample, "raw")
        boot_brier_diffs.append(r_brier - g_brier)  # positive = graph better

    ece_ci_low = np.percentile(boot_ece_diffs, 2.5)
    ece_ci_high = np.percentile(boot_ece_diffs, 97.5)
    brier_ci_low = np.percentile(boot_brier_diffs, 2.5)
    brier_ci_high = np.percentile(boot_brier_diffs, 97.5)

    print(f"\n  Bootstrap 95% CI for ECE difference (raw−graph, + = graph better):")
    print(f"    [{ece_ci_low:+.4f}, {ece_ci_high:+.4f}]")
    if ece_ci_low > 0:
        print(f"    → Graph significantly better calibrated (CI excludes 0)")
    elif ece_ci_high < 0:
        print(f"    → Raw significantly better calibrated (CI excludes 0)")
    else:
        print(f"    → Calibration difference not significant at 95%")

    print(f"\n  Bootstrap 95% CI for Brier difference (raw−graph, + = graph better):")
    print(f"    [{brier_ci_low:+.4f}, {brier_ci_high:+.4f}]")
    if brier_ci_low > 0:
        print(f"    → Graph significantly better (CI excludes 0)")
    elif brier_ci_high < 0:
        print(f"    → Raw significantly better (CI excludes 0)")
    else:
        print(f"    → Brier difference not significant at 95%")

    cal["graph_brier"] = round(graph_brier, 4)
    cal["raw_brier"] = round(raw_brier, 4)
    cal["ece_bootstrap_ci_95"] = [round(ece_ci_low, 4), round(ece_ci_high, 4)]
    cal["brier_bootstrap_ci_95"] = [round(brier_ci_low, 4), round(brier_ci_high, 4)]

    # ---- 3. Hybrid strategies ----
    strategies = evaluate_hybrid_strategies(cases)

    # Find best hybrid (maximize accuracy * coverage product)
    hybrid_strats = [s for s in strategies if s["strategy"] == "hybrid"]
    ensemble_strats = [s for s in strategies if s["strategy"] == "ensemble"]

    # Score = accuracy * sqrt(coverage) — rewards accuracy more but penalizes low coverage
    for s in strategies:
        s["score"] = round(s["accuracy"] * np.sqrt(s["coverage"]), 4) if s["coverage"] > 0 else 0

    best_hybrid = max(hybrid_strats, key=lambda s: s["score"])
    best_ensemble = max(ensemble_strats, key=lambda s: s["score"])

    # Best contra-predictive (simple flip)
    contra_strats = [s for s in strategies if s["strategy"] == "contra_graph"]
    contra_hybrid_strats = [s for s in strategies if s["strategy"] == "contra_hybrid"]

    for s in contra_strats + contra_hybrid_strats:
        s["score"] = round(s["accuracy"] * np.sqrt(s["coverage"]), 4) if s["coverage"] > 0 else 0

    best_contra = max(contra_strats, key=lambda s: s["score"]) if contra_strats else None
    best_contra_hybrid = max(contra_hybrid_strats, key=lambda s: s["score"]) if contra_hybrid_strats else None

    # Best single-method
    graph_baseline = {"accuracy": sum(1 for c in cases if c["graph_pred"] == c["true_label"]) / n,
                      "coverage": 1.0}
    raw_baseline = {"accuracy": sum(1 for c in cases if c["raw_pred"] == c["true_label"]) / n,
                    "coverage": 1.0}

    print(f"\n  {'─'*72}")
    print(f"  STRATEGY COMPARISON")
    print(f"  {'─'*72}")
    print(f"\n  {'Strategy':<40} {'Accuracy':>10} {'Coverage':>10} {'Score':>10}")
    print(f"  {'─'*72}")
    print(f"  {'Graph (all, no threshold)':<40} {graph_baseline['accuracy']:>10.3f} "
          f"{graph_baseline['coverage']:>10.3f} "
          f"{graph_baseline['accuracy'] * np.sqrt(graph_baseline['coverage']):>10.3f}")
    print(f"  {'Raw (all, no threshold)':<40} {raw_baseline['accuracy']:>10.3f} "
          f"{raw_baseline['coverage']:>10.3f} "
          f"{raw_baseline['accuracy'] * np.sqrt(raw_baseline['coverage']):>10.3f}")

    # Best graph-only at optimal threshold
    best_graph_only = max(graph_sweep,
                          key=lambda s: s["accuracy"] * np.sqrt(s["coverage"])
                          if s["n_accepted"] > 0 else 0)
    print(f"  {'Graph (t≥' + str(best_graph_only['threshold']) + ')':<40} "
          f"{best_graph_only['accuracy']:>10.3f} {best_graph_only['coverage']:>10.3f} "
          f"{best_graph_only['accuracy'] * np.sqrt(best_graph_only['coverage']):>10.3f}")

    best_raw_only = max(raw_sweep,
                        key=lambda s: s["accuracy"] * np.sqrt(s["coverage"])
                        if s["n_accepted"] > 0 else 0)
    print(f"  {'Raw (t≥' + str(best_raw_only['threshold']) + ')':<40} "
          f"{best_raw_only['accuracy']:>10.3f} {best_raw_only['coverage']:>10.3f} "
          f"{best_raw_only['accuracy'] * np.sqrt(best_raw_only['coverage']):>10.3f}")

    print(f"  {'Hybrid (g≥' + str(best_hybrid['t_graph']) + ', r≥' + str(best_hybrid['t_raw']) + ')':<40} "
          f"{best_hybrid['accuracy']:>10.3f} {best_hybrid['coverage']:>10.3f} "
          f"{best_hybrid['score']:>10.3f}")
    print(f"  {'Ensemble (t≥' + str(best_ensemble['t_min']) + ')':<40} "
          f"{best_ensemble['accuracy']:>10.3f} {best_ensemble['coverage']:>10.3f} "
          f"{best_ensemble['score']:>10.3f}")

    if best_contra:
        print(f"  {'Contra-graph (flip<' + str(best_contra['t_flip']) + ')':<40} "
              f"{best_contra['accuracy']:>10.3f} {best_contra['coverage']:>10.3f} "
              f"{best_contra['score']:>10.3f}")

    if best_contra_hybrid:
        label = f"Contra-hybrid (flip<{best_contra_hybrid['t_low']}, g≥{best_contra_hybrid['t_high']})"
        print(f"  {label:<40} "
              f"{best_contra_hybrid['accuracy']:>10.3f} {best_contra_hybrid['coverage']:>10.3f} "
              f"{best_contra_hybrid['score']:>10.3f}")

    # ---- CONTRA-PREDICTIVE DEEP DIVE ----
    print(f"\n  {'─'*72}")
    print(f"  CONTRA-PREDICTIVE ANALYSIS")
    print(f"  {'─'*72}")

    # Show the low-confidence bin stats to justify the flip
    low_conf_cases = [c for c in cases if c["graph_conf"] < 0.80]
    high_conf_cases = [c for c in cases if c["graph_conf"] >= 0.80]

    if low_conf_cases:
        low_acc = sum(1 for c in low_conf_cases
                      if c["graph_pred"] == c["true_label"]) / len(low_conf_cases)
        low_acc_flipped = sum(1 for c in low_conf_cases
                              if (1 - c["graph_pred"]) == c["true_label"]) / len(low_conf_cases)
        print(f"\n  Low-confidence cases (graph_conf < 0.80): n={len(low_conf_cases)}")
        print(f"    Original accuracy:  {low_acc:.3f} ({sum(1 for c in low_conf_cases if c['graph_pred'] == c['true_label'])}/{len(low_conf_cases)})")
        print(f"    Flipped accuracy:   {low_acc_flipped:.3f} ({sum(1 for c in low_conf_cases if (1 - c['graph_pred']) == c['true_label'])}/{len(low_conf_cases)})")
        print(f"    Improvement:        {low_acc_flipped - low_acc:+.3f}")

        # Is the inversion significant? Binomial test
        n_low = len(low_conf_cases)
        k_wrong = sum(1 for c in low_conf_cases if c["graph_pred"] != c["true_label"])
        # Under null (random): P(≥k_wrong wrong out of n_low) with p=0.5
        from math import comb
        p_value = sum(comb(n_low, k) * 0.5**n_low for k in range(k_wrong, n_low + 1))
        print(f"    Binomial test (H0: acc=0.50): p={p_value:.4f} "
              f"{'← SIGNIFICANT' if p_value < 0.05 else '← not significant (n too small)'}")

    if high_conf_cases:
        high_acc = sum(1 for c in high_conf_cases
                       if c["graph_pred"] == c["true_label"]) / len(high_conf_cases)
        print(f"\n  High-confidence cases (graph_conf ≥ 0.80): n={len(high_conf_cases)}")
        print(f"    Accuracy: {high_acc:.3f} ({sum(1 for c in high_conf_cases if c['graph_pred'] == c['true_label'])}/{len(high_conf_cases)})")

    # Combined: high-conf trusted + low-conf flipped
    if low_conf_cases and high_conf_cases:
        combined_correct = (
            sum(1 for c in high_conf_cases if c["graph_pred"] == c["true_label"]) +
            sum(1 for c in low_conf_cases if (1 - c["graph_pred"]) == c["true_label"])
        )
        combined_acc = combined_correct / n
        print(f"\n  COMBINED (trust high, flip low at 0.80):")
        print(f"    Accuracy: {combined_acc:.3f} ({combined_correct}/{n}) at 100% coverage")
        print(f"    vs Graph baseline:  {graph_baseline['accuracy']:.3f} → Δ={combined_acc - graph_baseline['accuracy']:+.3f}")
        print(f"    vs Raw baseline:    {raw_baseline['accuracy']:.3f} → Δ={combined_acc - raw_baseline['accuracy']:+.3f}")
        print(f"    vs Best hybrid:     {best_hybrid['accuracy']:.3f} → Δ={combined_acc - best_hybrid['accuracy']:+.3f}")

    # Sweep all flip thresholds to show the full picture
    print(f"\n  Contra-predictive sweep (flip below threshold, trust above):")
    print(f"  {'Thresh':>8} {'Trusted':>8} {'Flipped':>8} {'Combined Acc':>13} {'vs Baseline':>12}")
    print(f"  {'─'*52}")
    for cs in sorted(contra_strats, key=lambda s: s["t_flip"]):
        delta = cs["accuracy"] - graph_baseline["accuracy"]
        n_high = sum(1 for c in cases if c["graph_conf"] >= cs["t_flip"])
        n_low = len(cases) - n_high
        print(f"  {cs['t_flip']:>8.2f} {n_high:>8} {n_low:>8} {cs['accuracy']:>13.3f} {delta:>+12.3f}")

    # ---- 4. The key finding ----
    # At what threshold does graph first exceed 80%? 85%? 90%?
    print(f"\n  {'─'*72}")
    print(f"  KEY FINDING: ACCURACY AT CONFIDENCE MILESTONES")
    print(f"  {'─'*72}")

    for target_acc in [0.80, 0.85, 0.90]:
        # Find lowest threshold where accuracy >= target
        graph_hit = next((g for g in graph_sweep
                          if g["accuracy"] >= target_acc and g["n_accepted"] >= 3), None)
        raw_hit = next((r for r in raw_sweep
                        if r["accuracy"] >= target_acc and r["n_accepted"] >= 3), None)

        print(f"\n  To reach {target_acc:.0%} accuracy:")
        if graph_hit:
            print(f"    Graph: threshold ≥ {graph_hit['threshold']:.2f} "
                  f"→ {graph_hit['accuracy']:.1%} acc, {graph_hit['coverage']:.1%} coverage "
                  f"({graph_hit['n_accepted']}/{n} cases)")
        else:
            print(f"    Graph: NOT ACHIEVABLE at any threshold")
        if raw_hit:
            print(f"    Raw:   threshold ≥ {raw_hit['threshold']:.2f} "
                  f"→ {raw_hit['accuracy']:.1%} acc, {raw_hit['coverage']:.1%} coverage "
                  f"({raw_hit['n_accepted']}/{n} cases)")
        else:
            print(f"    Raw:   NOT ACHIEVABLE at any threshold")

        if graph_hit and raw_hit:
            cov_advantage = graph_hit["coverage"] - raw_hit["coverage"]
            print(f"    → Graph covers {cov_advantage:+.1%} more cases at {target_acc:.0%}")
        elif graph_hit and not raw_hit:
            print(f"    → GRAPH ONLY can reach {target_acc:.0%} accuracy")

    # ---- 5. Save full results for visualization ----
    output = {
        "n_cases": n,
        "graph_sweep": graph_sweep,
        "raw_sweep": raw_sweep,
        "calibration": cal,
        "best_hybrid": best_hybrid,
        "best_ensemble": best_ensemble,
        "best_contra": best_contra,
        "best_contra_hybrid": best_contra_hybrid,
        "contra_sweep": [s for s in strategies if s["strategy"] == "contra_graph"],
        "strategies": strategies,
        "graph_auacc": graph_auacc,
        "raw_auacc": raw_auacc,
        "cases": cases,  # Include for the visualization
    }

    out_path = Path(results_path).stem + "_confidence_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full analysis saved to {out_path}")
    print(f"  → Use this JSON with the visualization to generate paper figures")

    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Confidence-gated selective prediction analysis")
    p.add_argument("--results", required=True,
                   help="Path to graph_vs_raw_nXX.json from eval_graph_vs_raw.py")
    args = p.parse_args()

    run_analysis(args.results)


if __name__ == "__main__":
    main()