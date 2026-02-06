#!/usr/bin/env python3
"""
eval_ablation.py — Ablation study: What part of graph structure helps prediction?

Isolates contributions of:
    1. EXTRACTION:  selecting relevant information from raw text
    2. TYPE LABELS: semantic annotation (fact, concept, issue, argument)
    3. STRUCTURE:   sectional organization, relevance scores, actor mapping

Conditions (each adds one layer):
    A) Raw Text       — baseline, full text excerpt (ALREADY RUN)
    B) Flat Extracted  — same content as graph, dumped as plain prose
    C) Typed List     — extracted content with type labels, flat list
    D) Full Graph     — structured sections with all metadata (ALREADY RUN)

Reuses results from existing eval run for conditions A and D.

Usage:
    python eval_ablation.py --results graph_vs_raw_n452_noscrub.json --graph_dir iltur_graphs
    python eval_ablation.py --results graph_vs_raw_n452_noscrub.json --graph_dir iltur_graphs --concurrent 10
"""

from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# DATA LOADING (reused from main eval)
# =============================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_graphs(graph_dir):
    results = []
    for p in sorted(graph_dir.glob("*.json")):
        if p.name.startswith("checkpoint") or p.name.startswith("eval_"):
            continue
        try:
            data = load_json(p)
            case_id = data.get("case_id") or p.stem
            results.append((case_id, data))
        except Exception:
            continue
    return results


# =============================================================================
# ABLATION PROMPT BUILDERS
# =============================================================================

def _extract_all_text(graph: dict) -> List[str]:
    """Extract all text content from graph nodes (blinded: no holdings/outcome/court_response)."""
    texts = []

    # Facts
    facts = graph.get("facts") or []
    material = [f for f in facts if isinstance(f, dict) and f.get("fact_type") == "material"]
    other = [f for f in facts if isinstance(f, dict) and f.get("fact_type") != "material"]
    for f in (material + other)[:8]:
        text = f.get("text", "")[:300]
        if text:
            texts.append(text)

    # Concepts
    concepts = graph.get("concepts") or []
    central = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "central"]
    supporting = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "supporting"]
    for c in central + supporting[:4]:
        label = c.get("unlisted_label") or c.get("concept_id", "").replace("UNLISTED_", "").replace("_", " ")
        interp = c.get("interpretation", "")
        desc = c.get("unlisted_description", "")
        extra = interp or desc
        if label:
            texts.append(f"{label}: {extra[:200]}" if extra else label)

    # Issues (questions only)
    issues = graph.get("issues") or []
    for iss in issues[:5]:
        if isinstance(iss, dict):
            text = iss.get("text", "")[:250]
            if text:
                texts.append(text)

    # Arguments — party claims only, NO court actor
    arguments = graph.get("arguments") or []
    party_actors = ("petitioner", "appellant", "complainant", "prosecution",
                    "respondent", "accused")
    for a in arguments:
        if isinstance(a, dict) and a.get("actor") in party_actors:
            claim = a.get("claim", "")[:250]
            if claim:
                texts.append(claim)

    # Precedents
    precedents = graph.get("precedents") or []
    for pr in precedents[:5]:
        if isinstance(pr, dict):
            name = pr.get("case_name") or pr.get("citation", "")
            prop = pr.get("cited_proposition", "")
            if name:
                texts.append(f"{name} — {prop[:150]}" if prop else name)

    return texts


def build_flat_prompt(graph: dict) -> str:
    """Condition B: Flat Extracted — all extracted text as plain prose, zero structure.

    Same content as full graph, but:
    - No section headers (FACTS:, LEGAL CONCEPTS:, etc.)
    - No type labels ([material], [central], etc.)
    - No relevance scores or metadata
    - No actor attribution on arguments
    - Just concatenated text separated by periods
    """
    texts = _extract_all_text(graph)
    prose = ". ".join(t.rstrip(". ") for t in texts if t.strip())

    return (
        "Predict the outcome of this Indian Supreme Court case.\n"
        "Below is extracted information from the case. "
        "The court's decision has been removed — predict from the merits alone.\n\n"
        f"{prose}\n\n"
        "Predict: {\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
    )


def build_typed_prompt(graph: dict) -> str:
    """Condition C: Typed List — extracted content with type labels, flat list.

    Same content as full graph, but:
    - Type labels present (FACT, CONCEPT, ISSUE, ARGUMENT, PRECEDENT)
    - No section grouping — just a flat list
    - No relevance scores, no fact_type sub-labels
    - No actor attribution on arguments
    - No argument schemes
    - No concept kind labels
    """
    items = []

    # Facts
    facts = graph.get("facts") or []
    material = [f for f in facts if isinstance(f, dict) and f.get("fact_type") == "material"]
    other = [f for f in facts if isinstance(f, dict) and f.get("fact_type") != "material"]
    for f in (material + other)[:8]:
        text = f.get("text", "")[:300]
        if text:
            items.append(f"FACT: {text}")

    # Concepts
    concepts = graph.get("concepts") or []
    central = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "central"]
    supporting = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "supporting"]
    for c in central + supporting[:4]:
        label = c.get("unlisted_label") or c.get("concept_id", "").replace("UNLISTED_", "").replace("_", " ")
        interp = c.get("interpretation", "")
        desc = c.get("unlisted_description", "")
        extra = interp or desc
        entry = f"CONCEPT: {label}: {extra[:200]}" if extra else f"CONCEPT: {label}"
        items.append(entry)

    # Issues
    issues = graph.get("issues") or []
    for iss in issues[:5]:
        if isinstance(iss, dict):
            text = iss.get("text", "")[:250]
            if text:
                items.append(f"ISSUE: {text}")

    # Arguments — party claims only, NO court actor, no actor label
    arguments = graph.get("arguments") or []
    party_actors = ("petitioner", "appellant", "complainant", "prosecution",
                    "respondent", "accused")
    for a in arguments:
        if isinstance(a, dict) and a.get("actor") in party_actors:
            claim = a.get("claim", "")[:250]
            if claim:
                items.append(f"ARGUMENT: {claim}")

    # Precedents
    precedents = graph.get("precedents") or []
    for pr in precedents[:5]:
        if isinstance(pr, dict):
            name = pr.get("case_name") or pr.get("citation", "")
            prop = pr.get("cited_proposition", "")
            if name:
                entry = f"PRECEDENT: {name} — {prop[:150]}" if prop else f"PRECEDENT: {name}"
                items.append(entry)

    flat_list = "\n".join(items)

    return (
        "Predict the outcome of this Indian Supreme Court case.\n"
        "Below is extracted information from the case, labeled by type. "
        "The court's decision has been removed — predict from the merits alone.\n\n"
        f"{flat_list}\n\n"
        "Predict: {\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
    )


# =============================================================================
# LLM CLIENT (reused from main eval)
# =============================================================================

_SYSTEM_PROMPT = """You are an expert legal analyst specializing in Indian Supreme Court cases.

Your task: Given a case summary — facts, legal concepts, issues, and arguments from both 
parties — predict whether the appeal will be ACCEPTED (label=1) or REJECTED (label=0).

IMPORTANT: You do NOT know how the court ruled. You see only what was presented to the 
court, not its response. You must predict the outcome from the legal merits alone.

ACCEPTED means: allowed, partly_allowed, set_aside, remanded, or modified.
REJECTED means: dismissed.

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{
    "prediction": 0 or 1,
    "confidence": float between 0.0 and 1.0,
    "reasoning": "2-3 sentence explanation"
}

You MUST respond with valid JSON only. No markdown, no ```json blocks."""


async def llm_predict(api_key: str, system: str, prompt: str,
                      model: str = "grok-4-1-fast-reasoning",
                      temperature: float = 0.1) -> dict:
    """Call Grok API and parse JSON response."""
    import httpx

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 1024,
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)

            if not content.startswith("{"):
                json_match = re.search(r'\{[^{}]*"prediction"[^{}]*\}', content)
                if json_match:
                    content = json_match.group(0)

            result = json.loads(content)
            pred = result.get("prediction")
            if pred not in (0, 1):
                return {"prediction": -1, "confidence": 0.0,
                        "reasoning": f"Invalid prediction value: {pred}"}
            return result

        except Exception as e:
            return {"prediction": -1, "confidence": 0.0,
                    "reasoning": f"Error: {str(e)[:150]}"}


# =============================================================================
# CHECKPOINTING
# =============================================================================

def _ckpt_path(condition: str) -> Path:
    return Path(f"ablation_checkpoint_{condition}.json")


def save_checkpoint(results: list, config: dict, path: Path):
    data = {"config": config, "completed": results}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


def load_checkpoint(path: Path, config: dict) -> list:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        saved = data.get("config", {})
        if saved.get("model") != config.get("model"):
            print(f"    ⚠️  Checkpoint model mismatch — starting fresh")
            return []
        completed = data.get("completed", [])
        print(f"    ✅ Resuming: {len(completed)} cases done")
        return completed
    except Exception as e:
        print(f"    ⚠️  Checkpoint corrupted ({e}) — starting fresh")
        return []


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(predictions, true_labels, confidences=None):
    """Compute accuracy, F1, macro F1, Brier score."""
    n = len(predictions)
    if n == 0:
        return {}

    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    acc = correct / n

    # F1 (accepted = positive)
    tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    # F1 for class 0 (rejected = positive)
    tp0 = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    fp0 = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    fn0 = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0
    rec0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0

    macro_f1 = (f1 + f1_0) / 2

    result = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "n": n,
    }

    if confidences:
        scores = []
        for p, c, t in zip(predictions, confidences, true_labels):
            prob_true = c if p == t else (1.0 - c)
            scores.append((1.0 - prob_true) ** 2)
        result["brier"] = round(np.mean(scores), 4)

    return result


def mcnemar_test(preds_a, preds_b, true_labels):
    """McNemar's test between two prediction sets."""
    a_right_b_wrong = sum(1 for a, b, t in zip(preds_a, preds_b, true_labels)
                          if a == t and b != t)
    a_wrong_b_right = sum(1 for a, b, t in zip(preds_a, preds_b, true_labels)
                          if a != t and b == t)
    denom = a_right_b_wrong + a_wrong_b_right
    if denom == 0:
        return 0.0, "n/a"
    chi2 = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / denom
    sig = "p<0.05" if chi2 > 3.841 else "p≥0.05 (n.s.)"
    return round(chi2, 3), sig


def bootstrap_ci(preds_a, preds_b, true_labels, n_boot=10000, seed=42):
    """Bootstrap 95% CI for accuracy difference (A - B)."""
    rng = np.random.RandomState(seed)
    n = len(true_labels)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        acc_a = np.mean([1 if preds_a[i] == true_labels[i] else 0 for i in idx])
        acc_b = np.mean([1 if preds_b[i] == true_labels[i] else 0 for i in idx])
        diffs.append(acc_a - acc_b)
    return round(np.percentile(diffs, 2.5), 4), round(np.percentile(diffs, 97.5), 4)


# =============================================================================
# MAIN ABLATION RUNNER
# =============================================================================

async def run_ablation(
    results_file: str,
    graph_dir: str,
    model: str = "grok-4-1-fast-reasoning",
    concurrent: int = 5,
):
    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set XAI_API_KEY or GROK_API_KEY in .env")

    # ── Load existing results ────────────────────────────────────────────
    print("Loading existing results...")
    existing = load_json(results_file)
    existing_results = existing["results"]
    case_ids_order = [r["case_id"] for r in existing_results]
    true_labels = {r["case_id"]: r["true_label"] for r in existing_results}

    # Extract existing predictions
    graph_preds = {r["case_id"]: r["graph_pred"] for r in existing_results}
    raw_preds = {r["case_id"]: r["raw_pred"] for r in existing_results}
    graph_confs = {r["case_id"]: r["graph_conf"] for r in existing_results}
    raw_confs = {r["case_id"]: r["raw_conf"] for r in existing_results}

    print(f"  {len(case_ids_order)} cases from {results_file}")
    print(f"  Full Graph accuracy: {existing['summary']['graph_accuracy']}")
    print(f"  Raw Text accuracy:   {existing['summary']['raw_accuracy']}")

    # ── Load graphs ──────────────────────────────────────────────────────
    print("\nLoading graphs...")
    gdir = Path(graph_dir)
    all_graphs = {cid: g for cid, g in iter_graphs(gdir)}
    print(f"  {len(all_graphs)} graphs loaded")

    # Filter to cases in existing results
    cases = []
    for cid in case_ids_order:
        if cid in all_graphs:
            cases.append({
                "case_id": cid,
                "graph": all_graphs[cid],
                "label": true_labels[cid],
            })
    print(f"  {len(cases)} cases matched")

    # ── Build prompts ────────────────────────────────────────────────────
    print("\nBuilding ablation prompts...")
    for c in cases:
        c["flat_prompt"] = build_flat_prompt(c["graph"])
        c["typed_prompt"] = build_typed_prompt(c["graph"])

    # Prompt size comparison
    flat_sizes = [len(c["flat_prompt"]) for c in cases]
    typed_sizes = [len(c["typed_prompt"]) for c in cases]
    print(f"  Flat:  {np.mean(flat_sizes):.0f} chars avg "
          f"(min={min(flat_sizes)}, max={max(flat_sizes)})")
    print(f"  Typed: {np.mean(typed_sizes):.0f} chars avg "
          f"(min={min(typed_sizes)}, max={max(typed_sizes)})")

    # ── Run ablation conditions ──────────────────────────────────────────
    CONDITIONS = ["flat", "typed"]
    condition_results = {}

    for condition in CONDITIONS:
        print(f"\n{'='*70}")
        print(f"  RUNNING CONDITION: {condition.upper()}")
        print(f"{'='*70}")

        ckpt_config = {"model": model, "condition": condition}
        ckpt = _ckpt_path(condition)
        completed = load_checkpoint(ckpt, ckpt_config)
        completed_ids = {r["case_id"] for r in completed}
        remaining = [c for c in cases if c["case_id"] not in completed_ids]

        if remaining:
            print(f"  {len(remaining)} to run ({len(completed)} done)")
        else:
            print(f"  All {len(completed)} already done — skipping")

        semaphore = asyncio.Semaphore(concurrent)
        results = list(completed)
        t0 = time.time()

        prompt_key = f"{condition}_prompt"

        async def predict_one(case: dict, num: int):
            async with semaphore:
                result = await llm_predict(
                    api_key, _SYSTEM_PROMPT, case[prompt_key], model=model
                )

            pred = result.get("prediction", -1)
            conf = result.get("confidence", 0.0)
            label = case["label"]
            correct = pred == label

            label_str = "ACC" if label == 1 else "REJ"
            pred_str = "ACC" if pred == 1 else ("REJ" if pred == 0 else "ERR")
            mark = "✓" if correct else "✗"

            # Compare to full graph and raw
            g_correct = graph_preds.get(case["case_id"], -1) == label
            r_correct = raw_preds.get(case["case_id"], -1) == label

            tag = ""
            if correct and not g_correct:
                tag = f"  ←{condition.upper()} beats GRAPH"
            elif not correct and g_correct:
                tag = f"  ←GRAPH beats {condition.upper()}"

            print(f"  [{num:3d}] {case['case_id']:12s} TRUE={label_str} "
                  f"{condition}={mark}{pred_str} conf={conf:.2f}{tag}")

            return {
                "case_id": case["case_id"],
                "true_label": label,
                "prediction": pred,
                "confidence": conf,
                "reasoning": result.get("reasoning", ""),
            }

        # Process in batches with checkpointing
        for batch_start in range(0, len(remaining), concurrent):
            batch = remaining[batch_start:batch_start + concurrent]
            offset = len(completed) + batch_start
            tasks = [
                asyncio.create_task(predict_one(c, offset + i + 1))
                for i, c in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            save_checkpoint(results, ckpt_config, ckpt)
            done = len(results)
            total = len(cases)
            print(f"  --- checkpoint: {done}/{total} ({done/total*100:.0f}%) ---")

        elapsed = time.time() - t0
        if remaining:
            print(f"  Done in {elapsed:.1f}s")

        # Clean up checkpoint
        if ckpt.exists():
            ckpt.unlink()

        condition_results[condition] = results

    # ── ANALYSIS ─────────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("  ABLATION RESULTS")
    print(f"{'='*80}")

    # Build aligned prediction arrays (same case order)
    all_labels = [true_labels[cid] for cid in case_ids_order]

    methods = {}
    methods["raw"] = {
        "preds": [raw_preds[cid] for cid in case_ids_order],
        "confs": [raw_confs[cid] for cid in case_ids_order],
    }
    methods["flat"] = {
        "preds": [],
        "confs": [],
    }
    methods["typed"] = {
        "preds": [],
        "confs": [],
    }
    methods["graph"] = {
        "preds": [graph_preds[cid] for cid in case_ids_order],
        "confs": [graph_confs[cid] for cid in case_ids_order],
    }

    # Map ablation results by case_id
    for condition in CONDITIONS:
        result_map = {r["case_id"]: r for r in condition_results[condition]}
        for cid in case_ids_order:
            r = result_map.get(cid, {})
            methods[condition]["preds"].append(r.get("prediction", -1))
            methods[condition]["confs"].append(r.get("confidence", 0.0))

    # Filter to cases where ALL conditions have valid predictions
    valid_idx = []
    for i, cid in enumerate(case_ids_order):
        if all(methods[m]["preds"][i] in (0, 1) for m in ["raw", "flat", "typed", "graph"]):
            valid_idx.append(i)

    n_valid = len(valid_idx)
    print(f"\n  Valid cases (all 4 conditions produced predictions): {n_valid}/{len(case_ids_order)}")

    labels = [all_labels[i] for i in valid_idx]

    method_names = ["raw", "flat", "typed", "graph"]
    display_names = {
        "raw": "Raw Text",
        "flat": "Flat Extracted",
        "typed": "Typed List",
        "graph": "Full Graph",
    }
    adds_what = {
        "raw": "(baseline)",
        "flat": "(+ extraction)",
        "typed": "(+ type labels)",
        "graph": "(+ structure)",
    }

    metrics = {}
    for m in method_names:
        preds = [methods[m]["preds"][i] for i in valid_idx]
        confs = [methods[m]["confs"][i] for i in valid_idx]
        metrics[m] = compute_metrics(preds, labels, confs)
        metrics[m]["preds"] = preds
        metrics[m]["confs"] = confs

    # ── Results table ────────────────────────────────────────────────────
    print(f"\n  {'Condition':<20} {'Layer':<18} {'Acc':>7} {'Macro F1':>9} "
          f"{'Brier':>7} {'Δ Acc vs Raw':>13}")
    print(f"  {'─'*20} {'─'*18} {'─'*7} {'─'*9} {'─'*7} {'─'*13}")

    for m in method_names:
        met = metrics[m]
        delta = met["accuracy"] - metrics["raw"]["accuracy"]
        delta_str = f"{delta:+.3f}" if m != "raw" else "—"
        print(f"  {display_names[m]:<20} {adds_what[m]:<18} "
              f"{met['accuracy']:>7.3f} {met['macro_f1']:>9.3f} "
              f"{met.get('brier', 0):>7.3f} {delta_str:>13}")

    # ── Incremental contribution ─────────────────────────────────────────
    print(f"\n  INCREMENTAL CONTRIBUTION (each step adds what?):")
    print(f"  {'─'*65}")

    steps = [
        ("raw→flat", "raw", "flat", "Extraction effect"),
        ("flat→typed", "flat", "typed", "Type label effect"),
        ("typed→graph", "typed", "graph", "Structure effect"),
        ("raw→graph", "raw", "graph", "Total improvement"),
    ]

    for step_name, from_m, to_m, desc in steps:
        acc_from = metrics[from_m]["accuracy"]
        acc_to = metrics[to_m]["accuracy"]
        delta = acc_to - acc_from

        chi2, sig = mcnemar_test(
            metrics[to_m]["preds"], metrics[from_m]["preds"], labels
        )
        ci_lo, ci_hi = bootstrap_ci(
            metrics[to_m]["preds"], metrics[from_m]["preds"], labels
        )

        sig_marker = "***" if chi2 > 6.635 else ("**" if chi2 > 3.841 else "n.s.")

        print(f"  {desc:<22} {acc_from:.3f} → {acc_to:.3f}  "
              f"Δ={delta:+.3f}  χ²={chi2:>6.3f} ({sig_marker})  "
              f"CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # ── Discordant analysis between adjacent conditions ──────────────────
    print(f"\n  DISCORDANT CASES (adjacent conditions):")
    print(f"  {'─'*55}")

    adj_pairs = [("raw", "flat"), ("flat", "typed"), ("typed", "graph")]
    for m_a, m_b in adj_pairs:
        a_wins = sum(1 for i in range(n_valid)
                     if metrics[m_b]["preds"][i] == labels[i]
                     and metrics[m_a]["preds"][i] != labels[i])
        b_wins = sum(1 for i in range(n_valid)
                     if metrics[m_a]["preds"][i] == labels[i]
                     and metrics[m_b]["preds"][i] != labels[i])
        print(f"  {display_names[m_a]:>20} → {display_names[m_b]:<20} "
              f"gained={a_wins:3d}  lost={b_wins:3d}  net={a_wins - b_wins:+3d}")

    # ── Per-class breakdown ──────────────────────────────────────────────
    acc_idx = [i for i in range(n_valid) if labels[i] == 1]
    rej_idx = [i for i in range(n_valid) if labels[i] == 0]

    print(f"\n  PER-CLASS ACCURACY:")
    print(f"  {'Condition':<20} {'Accepted':>10} {'Rejected':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")

    for m in method_names:
        acc_acc = np.mean([1 if metrics[m]["preds"][i] == labels[i] else 0 for i in acc_idx])
        rej_acc = np.mean([1 if metrics[m]["preds"][i] == labels[i] else 0 for i in rej_idx])
        print(f"  {display_names[m]:<20} {acc_acc:>10.3f} {rej_acc:>10.3f}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "n": n_valid,
        "model": model,
        "conditions": {},
    }
    for m in method_names:
        met = metrics[m]
        output["conditions"][m] = {
            "display_name": display_names[m],
            "adds": adds_what[m],
            "accuracy": met["accuracy"],
            "macro_f1": met["macro_f1"],
            "f1": met["f1"],
            "precision": met["precision"],
            "recall": met["recall"],
            "brier": met.get("brier", None),
        }

    output["incremental"] = {}
    for step_name, from_m, to_m, desc in steps:
        chi2, sig = mcnemar_test(
            metrics[to_m]["preds"], metrics[from_m]["preds"], labels
        )
        ci_lo, ci_hi = bootstrap_ci(
            metrics[to_m]["preds"], metrics[from_m]["preds"], labels
        )
        output["incremental"][step_name] = {
            "description": desc,
            "delta_accuracy": round(metrics[to_m]["accuracy"] - metrics[from_m]["accuracy"], 4),
            "mcnemar_chi2": chi2,
            "mcnemar_sig": sig,
            "bootstrap_ci_95": [ci_lo, ci_hi],
        }

    # Per-condition results
    output["case_results"] = []
    for i, idx in enumerate(valid_idx):
        cid = case_ids_order[idx]
        entry = {"case_id": cid, "true_label": labels[i]}
        for m in method_names:
            entry[f"{m}_pred"] = metrics[m]["preds"][i]
            entry[f"{m}_conf"] = metrics[m]["confs"][i]
        output["case_results"].append(entry)

    out_file = "ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    print(f"\n{'='*80}")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Ablation study: what part of graph structure helps?")
    p.add_argument("--results", required=True,
                   help="Path to existing graph_vs_raw results JSON")
    p.add_argument("--graph_dir", required=True,
                   help="Directory of extracted graph JSONs")
    p.add_argument("--model", default="grok-4-1-fast-reasoning")
    p.add_argument("--concurrent", type=int, default=5)
    args = p.parse_args()

    asyncio.run(run_ablation(
        results_file=args.results,
        graph_dir=args.graph_dir,
        model=args.model,
        concurrent=args.concurrent,
    ))


if __name__ == "__main__":
    main()
