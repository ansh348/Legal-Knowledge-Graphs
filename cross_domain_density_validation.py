#!/usr/bin/env python3
"""
cross_domain_density_validation.py

Cross-domain validation: does the density-accuracy relationship from
multi-document QA (Paper A) replicate in legal judgment prediction?

Paper A finding (Liu et al. protocol, 50 docs, 3500 examples):
  - Gold doc POSITION does not predict accuracy (r=-0.009, p=0.65)
  - Gold doc DENSITY positively predicts accuracy (r=+0.128, p<0.0001)
  - Gold doc TOKEN COUNT negatively predicts accuracy (r=-0.168, p<0.0001)

This script tests the same density/token-count relationships on LexCGraph
(Indian Supreme Court case prediction, 2 models x 2 conditions).

Usage:
    python cross_domain_density_validation.py
"""

import json
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SAMPLE_N = 500
SEED = 42
GROK_FILE = "graph_vs_raw_n2517_noscrub.json"
SONNET_FILE = "graph_vs_raw_n2517_noscrub_sonnet.json"
CACHE_FILE = "density_cache_n500.json"

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_texts_hf():
    """Load raw judgment texts from HuggingFace IL-TUR dataset."""
    from datasets import load_dataset
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe")["single_train"]
    return {str(ex["id"]): ex["text"] for ex in ds}


# ---------------------------------------------------------------------------
# DENSITY COMPUTATION
# ---------------------------------------------------------------------------

def compute_density_spacy(texts_dict, case_ids):
    """Compute information density for each case using SpaCy."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 5_000_000

    # Prepare texts in order
    ordered_ids = [cid for cid in case_ids if cid in texts_dict]
    ordered_texts = [texts_dict[cid] for cid in ordered_ids]

    print(f"  Processing {len(ordered_ids)} documents with SpaCy...")
    results = {}
    t0 = time.time()

    for i, doc in enumerate(nlp.pipe(ordered_texts, batch_size=25)):
        cid = ordered_ids[i]
        tokens = [t for t in doc if not t.is_space and not t.is_punct]
        token_count = len(tokens)
        if token_count == 0:
            results[cid] = {"density": 0.0, "token_count": 0,
                            "entity_count": 0, "relation_count": 0, "clause_count": 0}
            continue

        entity_count = len(doc.ents)
        relation_count = len([t for t in doc if t.dep_ not in ("punct", "")])
        clause_count = len([t for t in doc if t.dep_ in
                            ("ccomp", "xcomp", "advcl", "acl", "relcl")])
        density = (entity_count + relation_count + clause_count) / token_count

        results[cid] = {
            "density": density,
            "token_count": token_count,
            "entity_count": entity_count,
            "relation_count": relation_count,
            "clause_count": clause_count,
        }

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ordered_ids) - i - 1) / rate
            print(f"    {i+1}/{len(ordered_ids)} done  ({rate:.1f} docs/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(results)/elapsed:.1f} docs/s)")
    return results


# ---------------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------------

def sig_label(p):
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_correlations(density_arr, token_arr, accuracy_arr, label):
    """Run point-biserial correlations and print results."""
    r_d, p_d = stats.pointbiserialr(accuracy_arr, density_arr)
    r_t, p_t = stats.pointbiserialr(accuracy_arr, token_arr)
    print(f"  {label}")
    print(f"    density:  r={r_d:+.4f}, p={p_d:.6f} ({sig_label(p_d)})")
    print(f"    tokens:   r={r_t:+.4f}, p={p_t:.6f} ({sig_label(p_t)})")
    return {"density_r": r_d, "density_p": p_d,
            "tokens_r": r_t, "tokens_p": p_t}


def quartile_analysis(density_arr, accuracy_arr, label):
    """Split into density quartiles and report accuracy per quartile."""
    quartile_edges = np.percentile(density_arr, [25, 50, 75])
    q_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    bins = np.digitize(density_arr, quartile_edges)  # 0,1,2,3

    print(f"\n  Quartile analysis — {label}:")
    print(f"    {'Quartile':<16} {'N':>5} {'Accuracy':>10} {'Mean Density':>14} {'Mean Tokens':>13}")
    print(f"    {'-'*60}")

    quartile_accs = []
    for q in range(4):
        mask = bins == q
        n_q = mask.sum()
        acc_q = accuracy_arr[mask].mean() if n_q > 0 else 0
        den_q = density_arr[mask].mean() if n_q > 0 else 0
        tok_q = 0  # filled below
        quartile_accs.append(acc_q)
        print(f"    {q_labels[q]:<16} {n_q:>5} {acc_q:>10.4f} {den_q:>14.4f}")

    # Chi-square test: Q1 vs Q4 accuracy difference
    q1_mask = bins == 0
    q4_mask = bins == 3
    q1_correct = accuracy_arr[q1_mask].sum()
    q1_total = q1_mask.sum()
    q4_correct = accuracy_arr[q4_mask].sum()
    q4_total = q4_mask.sum()

    if q1_total > 0 and q4_total > 0:
        # Two-proportion z-test
        p1 = q1_correct / q1_total
        p4 = q4_correct / q4_total
        p_pool = (q1_correct + q4_correct) / (q1_total + q4_total)
        if p_pool > 0 and p_pool < 1:
            se = np.sqrt(p_pool * (1 - p_pool) * (1/q1_total + 1/q4_total))
            z = (p4 - p1) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            print(f"    Q1 vs Q4: delta={p4-p1:+.4f}, z={z:.3f}, p={p_val:.6f} ({sig_label(p_val)})")

    return quartile_accs


def correct_vs_incorrect(density_arr, token_arr, accuracy_arr, label):
    """Compare density and tokens between correct and incorrect predictions."""
    correct_mask = accuracy_arr == 1
    incorrect_mask = accuracy_arr == 0

    d_correct = density_arr[correct_mask]
    d_incorrect = density_arr[incorrect_mask]
    t_correct = token_arr[correct_mask]
    t_incorrect = token_arr[incorrect_mask]

    print(f"\n  Correct vs Incorrect — {label}:")
    print(f"    {'Metric':<12} {'Correct (n={})'.format(correct_mask.sum()):>20} "
          f"{'Incorrect (n={})'.format(incorrect_mask.sum()):>22} {'t':>8} {'p':>10} {'d':>8}")
    print(f"    {'-'*82}")

    # Density
    t_d, p_d = stats.ttest_ind(d_correct, d_incorrect)
    d_d = cohens_d(d_correct, d_incorrect)
    print(f"    {'density':<12} {d_correct.mean():>20.4f} {d_incorrect.mean():>22.4f} "
          f"{t_d:>8.3f} {p_d:>10.6f} {d_d:>+8.3f}")

    # Tokens
    t_t, p_t = stats.ttest_ind(t_correct, t_incorrect)
    d_t = cohens_d(t_correct, t_incorrect)
    print(f"    {'tokens':<12} {t_correct.mean():>20.1f} {t_incorrect.mean():>22.1f} "
          f"{t_t:>8.3f} {p_t:>10.6f} {d_t:>+8.3f}")

    return {
        "density_correct_mean": d_correct.mean(), "density_incorrect_mean": d_incorrect.mean(),
        "density_t": t_d, "density_p": p_d, "density_d": d_d,
        "tokens_correct_mean": t_correct.mean(), "tokens_incorrect_mean": t_incorrect.mean(),
        "tokens_t": t_t, "tokens_p": p_t, "tokens_d": d_t,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CROSS-DOMAIN VALIDATION: Density-Accuracy on LexCGraph")
    print(f"  Sample: {SAMPLE_N} cases (seed={SEED})")
    print("=" * 70)

    # --- Load prediction results ---
    print("\n[1] Loading prediction results...")
    grok = load_results(GROK_FILE)
    sonnet = load_results(SONNET_FILE)

    grok_cases = {c["case_id"]: c for c in grok["cases"]}
    sonnet_cases = {c["case_id"]: c for c in sonnet["cases"]}

    # Cases present in BOTH result files
    common_ids = sorted(set(grok_cases.keys()) & set(sonnet_cases.keys()))
    print(f"  Grok: {len(grok_cases)} cases, Sonnet: {len(sonnet_cases)} cases")
    print(f"  Common: {len(common_ids)} cases")

    # --- Sample ---
    sample_ids = sorted(random.sample(common_ids, min(SAMPLE_N, len(common_ids))))
    print(f"  Sampled: {len(sample_ids)} cases")

    # --- Load or compute density ---
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        print(f"\n[2] Loading cached density from {CACHE_FILE}...")
        with open(cache_path) as f:
            density_data = json.load(f)
        # Check coverage
        missing = [cid for cid in sample_ids if cid not in density_data]
        if missing:
            print(f"  Cache missing {len(missing)} cases — recomputing all...")
            cache_path.unlink()
        else:
            print(f"  Cache hit: {len(density_data)} cases")

    if not cache_path.exists():
        print("\n[2] Loading raw texts from HuggingFace IL-TUR...")
        raw_texts = load_raw_texts_hf()
        print(f"  Loaded {len(raw_texts)} raw texts")

        # Check coverage
        matched = [cid for cid in sample_ids if cid in raw_texts]
        print(f"  Matched to sample: {len(matched)}/{len(sample_ids)}")

        if len(matched) < len(sample_ids):
            missing = [cid for cid in sample_ids if cid not in raw_texts]
            print(f"  Missing case IDs (first 10): {missing[:10]}")
            sample_ids = matched

        print("\n[3] Computing information density with SpaCy...")
        density_data = compute_density_spacy(raw_texts, sample_ids)

        # Cache results
        with open(cache_path, "w") as f:
            json.dump(density_data, f, indent=2)
        print(f"  Cached to {CACHE_FILE}")

    # --- Build arrays ---
    print(f"\n[4] Building analysis arrays...")

    # Filter to cases we have density for
    valid_ids = [cid for cid in sample_ids if cid in density_data]
    print(f"  Valid cases with density data: {len(valid_ids)}")

    density_arr = np.array([density_data[cid]["density"] for cid in valid_ids])
    token_arr = np.array([density_data[cid]["token_count"] for cid in valid_ids])

    # Accuracy arrays: 1 if pred == true_label, else 0
    grok_raw_acc = np.array([int(grok_cases[cid]["raw_pred"] == grok_cases[cid]["true_label"])
                             for cid in valid_ids])
    grok_graph_acc = np.array([int(grok_cases[cid]["graph_pred"] == grok_cases[cid]["true_label"])
                               for cid in valid_ids])
    sonnet_raw_acc = np.array([int(sonnet_cases[cid]["raw_pred"] == sonnet_cases[cid]["true_label"])
                               for cid in valid_ids])
    sonnet_graph_acc = np.array([int(sonnet_cases[cid]["graph_pred"] == sonnet_cases[cid]["true_label"])
                                 for cid in valid_ids])

    # Summary stats
    print(f"\n  Density:  mean={density_arr.mean():.4f}, std={density_arr.std():.4f}, "
          f"min={density_arr.min():.4f}, max={density_arr.max():.4f}")
    print(f"  Tokens:   mean={token_arr.mean():.0f}, std={token_arr.std():.0f}, "
          f"min={token_arr.min()}, max={token_arr.max()}")
    print(f"  Grok  raw acc:   {grok_raw_acc.mean():.4f}  |  graph acc: {grok_graph_acc.mean():.4f}")
    print(f"  Sonnet raw acc:  {sonnet_raw_acc.mean():.4f}  |  graph acc: {sonnet_graph_acc.mean():.4f}")

    # --- Correlations ---
    print("\n" + "=" * 70)
    print("POINT-BISERIAL CORRELATIONS")
    print("=" * 70)

    print("\n  Paper A — Multi-doc QA (reference):")
    print("    position:  r=-0.009, p=0.6500 (ns)")
    print("    density:   r=+0.128, p<0.0001 (****)")
    print("    tokens:    r=-0.168, p<0.0001 (****)")

    print()
    results = {}
    results["raw_grok"] = run_correlations(density_arr, token_arr, grok_raw_acc,
                                           "Legal prediction — Raw text (Grok)")
    results["raw_sonnet"] = run_correlations(density_arr, token_arr, sonnet_raw_acc,
                                             "Legal prediction — Raw text (Sonnet)")
    results["graph_grok"] = run_correlations(density_arr, token_arr, grok_graph_acc,
                                             "Legal prediction — Graph (Grok)")
    results["graph_sonnet"] = run_correlations(density_arr, token_arr, sonnet_graph_acc,
                                               "Legal prediction — Graph (Sonnet)")

    # --- Quartile analysis ---
    print("\n" + "=" * 70)
    print("QUARTILE ANALYSIS (density quartiles)")
    print("=" * 70)

    quartile_analysis(density_arr, grok_raw_acc, "Raw text (Grok)")
    quartile_analysis(density_arr, sonnet_raw_acc, "Raw text (Sonnet)")
    quartile_analysis(density_arr, grok_graph_acc, "Graph (Grok)")
    quartile_analysis(density_arr, sonnet_graph_acc, "Graph (Sonnet)")

    # --- Correct vs Incorrect ---
    print("\n" + "=" * 70)
    print("CORRECT vs INCORRECT — t-tests")
    print("=" * 70)

    cvi = {}
    cvi["raw_grok"] = correct_vs_incorrect(density_arr, token_arr, grok_raw_acc,
                                            "Raw text (Grok)")
    cvi["raw_sonnet"] = correct_vs_incorrect(density_arr, token_arr, sonnet_raw_acc,
                                              "Raw text (Sonnet)")
    cvi["graph_grok"] = correct_vs_incorrect(density_arr, token_arr, grok_graph_acc,
                                              "Graph (Grok)")
    cvi["graph_sonnet"] = correct_vs_incorrect(density_arr, token_arr, sonnet_graph_acc,
                                                "Graph (Sonnet)")

    # --- Save full results ---
    print("\n" + "=" * 70)
    out = {
        "config": {"sample_n": len(valid_ids), "seed": SEED, "total_available": len(common_ids)},
        "density_summary": {
            "mean": float(density_arr.mean()), "std": float(density_arr.std()),
            "min": float(density_arr.min()), "max": float(density_arr.max()),
        },
        "token_summary": {
            "mean": float(token_arr.mean()), "std": float(token_arr.std()),
            "min": int(token_arr.min()), "max": int(token_arr.max()),
        },
        "correlations": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        "correct_vs_incorrect": {
            k: {kk: float(vv) for kk, vv in v.items()} for k, v in cvi.items()
        },
    }
    out_path = "cross_domain_density_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
