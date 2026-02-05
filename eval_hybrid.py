#!/usr/bin/env python3
"""
eval_hybrid.py - Hybrid LLM evaluation for legal reasoning graphs

Three evaluation modes:
  1. embed_retrieval  - Dense embeddings replace TF-IDF for retrieval → kNN predict
  2. llm_zeroshot     - LLM predicts from case graph alone (no retrieval)  
  3. llm_hybrid       - Retrieve neighbors with embeddings → LLM reasons over graphs → predict

Usage:
    # Dense embedding retrieval (fast, no API needed)
    python eval_hybrid.py embed_retrieval --graph_dir iltur_graphs --k 10

    # LLM zero-shot (needs XAI_API_KEY, 452 API calls)
    python eval_hybrid.py llm_zeroshot --graph_dir iltur_graphs --n 50

    # Full hybrid (best results, needs API key)
    python eval_hybrid.py llm_hybrid --graph_dir iltur_graphs --k 5 --n 50

    # All three for paper comparison
    python eval_hybrid.py paper_table --graph_dir iltur_graphs --k 5 --n 100

Requirements:
    pip install sentence-transformers datasets python-dotenv httpx
"""

from __future__ import annotations
import argparse, asyncio, json, math, os, re, sys, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from dotenv import load_dotenv

# Import base components from eval_concept_retrieval
from eval_concept_retrieval import (
    load_json, iter_graphs, load_labels_hf, load_ontology,
    extract_concept_profile, ConceptProfile,
    compute_idf_multi, FuzzyConceptIndex,
    CaseTextSimilarity, combined_similarity, retrieve_similar,
    predict_from_neighbors, RetrievalResult,
    find_optimal_threshold, _compute_metrics_at_threshold,
    bootstrap_ci,
)

load_dotenv()


# =============================================================================
# DENSE EMBEDDING SIMILARITY
# =============================================================================

class EmbeddingSimilarity:
    """Dense semantic embeddings using sentence-transformers.

    This replaces TF-IDF with proper semantic similarity.
    'all-MiniLM-L6-v2' is fast and good. 'BAAI/bge-large-en-v1.5' is better but slower.
    """

    def __init__(self, profiles: list, model_name: str = "all-MiniLM-L6-v2",
                 use_full_text: bool = True, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.idx = {p.case_id: i for i, p in enumerate(profiles)}
        self.model = SentenceTransformer(model_name)

        texts = []
        for p in profiles:
            text = p.full_case_text if (use_full_text and p.full_case_text.strip()) else p.concept_text
            # Truncate to model's max length (most models handle 512 tokens)
            # Use first ~2000 chars (covers most key content)
            if len(text) > 6000:
                # Keep beginning (facts, issues) + end (holdings, outcome)
                text = text[:3000] + " ... " + text[-3000:]
            texts.append(text if text.strip() else "empty case")

        print(f"  [embeddings] encoding {len(texts)} cases with {model_name}...")
        t0 = time.time()
        self.embeddings = self.model.encode(texts, batch_size=batch_size,
                                            show_progress_bar=True, normalize_embeddings=True)
        elapsed = time.time() - t0
        print(f"  [embeddings] done in {elapsed:.1f}s, shape={self.embeddings.shape}")

    def similarity(self, case_id_a: str, case_id_b: str) -> float:
        i, j = self.idx.get(case_id_a), self.idx.get(case_id_b)
        if i is None or j is None:
            return 0.0
        # Dot product of normalized vectors = cosine similarity
        return float(self.embeddings[i] @ self.embeddings[j])

    def get_neighbors(self, query_id: str, k: int = 10, exclude: set = None) -> List[Tuple[str, float]]:
        """Get k nearest neighbors by embedding similarity. Returns [(case_id, sim), ...]"""
        exclude = exclude or set()
        qi = self.idx.get(query_id)
        if qi is None:
            return []
        sims = self.embeddings @ self.embeddings[qi]  # all cosine similarities
        # Build (index, sim) pairs, excluding self and excluded
        scored = []
        for case_id, idx in self.idx.items():
            if case_id == query_id or case_id in exclude:
                continue
            scored.append((case_id, float(sims[idx])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# =============================================================================
# CONCEPT-BASED SIMILARITY (fuzzy concepts + TF-IDF + statute families)
# =============================================================================

class ConceptNeighborIndex:
    """Graph-structure-aware retrieval using legal concept similarity.

    Uses combined_similarity from eval_concept_retrieval which blends:
      - Fuzzy token-Jaccard on concept IDs (w=0.35)
      - Full case text TF-IDF (w=0.55)
      - Statute family overlap (w=0.10)

    This retrieves neighbors that share legal provisions and argument structures,
    not just surface text similarity like MiniLM embeddings.
    """

    def __init__(self, corpus: list):
        """
        Args:
            corpus: list of (ConceptProfile, graph_dict) tuples
        """
        self.corpus = corpus
        self.profiles = [p for p, _ in corpus]
        self.idx = {p.case_id: i for i, p in enumerate(self.profiles)}

        print(f"  [concept index] building similarity components for {len(self.profiles)} cases...")
        t0 = time.time()

        # Build all the similarity components
        self.onto_idf, self.family_idf = compute_idf_multi(self.profiles)
        self.fuzzy_index = FuzzyConceptIndex(self.profiles)
        try:
            self.text_sim = CaseTextSimilarity(self.profiles, use_full_text=True)
        except Exception as e:
            print(f"  [concept index] warning: text similarity failed ({e}), using fuzzy only")
            self.text_sim = None

        elapsed = time.time() - t0
        print(f"  [concept index] ready in {elapsed:.1f}s")

    def get_neighbors(self, query_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """Get k nearest neighbors by concept similarity. Returns [(case_id, sim), ...]"""
        qi = self.idx.get(query_id)
        if qi is None:
            return []

        query_profile = self.profiles[qi]
        sim_kwargs = {
            "onto_idf": self.onto_idf,
            "family_idf": self.family_idf,
            "text_sim": self.text_sim,
            "fuzzy_index": self.fuzzy_index,
        }

        scored = []
        for p in self.profiles:
            if p.case_id == query_id:
                continue
            sim = combined_similarity(query_profile, p, **sim_kwargs)
            scored.append((p.case_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# =============================================================================
# COMPACT GRAPH SERIALIZATION FOR LLM CONTEXT
# =============================================================================

def _compact_graph_summary(graph: dict, max_facts: int = 5, max_args: int = 4,
                           max_holdings: int = 3, max_precedents: int = 3,
                           blind: bool = False, hide_outcome: bool = False) -> str:
    """Serialize a graph into a compact text summary for LLM context.

    Prioritizes central/high-confidence elements. Targets ~200-400 tokens per case.

    Args:
        blind: If True, strip all outcome-revealing fields from the summary.
               This removes: issue answers, argument court_response, holdings,
               and outcome. Used for query cases where the task is genuine
               prediction — the model should predict from facts, concepts,
               and party arguments alone, without seeing how the court ruled.
               Neighbor cases should NOT be blinded (their outcomes are known).
        hide_outcome: If True, strip only the final OUTCOME disposition line
               but keep all court behavior (court_response, holdings, issue
               answers). Used for "behavior mode" neighbors where the LLM
               should see HOW the court reasoned but not the final verdict.
    """
    parts = []

    # Key facts (prioritize material facts)
    facts = graph.get("facts") or []
    material = [f for f in facts if isinstance(f, dict) and f.get("fact_type") == "material"]
    other = [f for f in facts if isinstance(f, dict) and f.get("fact_type") != "material"]
    selected_facts = (material + other)[:max_facts]
    if selected_facts:
        parts.append("FACTS:")
        for f in selected_facts:
            text = f.get("text", "")[:200]
            ftype = f.get("fact_type", "")
            parts.append(f"  [{ftype}] {text}")

    # Legal concepts (prioritize central)
    concepts = graph.get("concepts") or []
    central = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "central"]
    supporting = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "supporting"]
    selected_concepts = central + supporting[:3]
    if selected_concepts:
        parts.append("LEGAL CONCEPTS:")
        for c in selected_concepts:
            cid = c.get("concept_id", "unknown")
            label = c.get("unlisted_label") or cid.replace("UNLISTED_", "").replace("_", " ")
            interp = c.get("interpretation", "")
            rel = c.get("relevance", "")
            line = f"  [{rel}] {label}"
            if interp and not blind:
                line += f" — {interp[:150]}"
            parts.append(line)

    # Issues — suppress answers when blinded (answers reveal outcome)
    issues = graph.get("issues") or []
    if issues:
        parts.append("ISSUES:")
        for iss in issues[:4]:
            if isinstance(iss, dict):
                text = iss.get("text", "")[:200]
                parts.append(f"  Q: {text}")
                if not blind:
                    answer = iss.get("answer", "")
                    if answer:
                        parts.append(f"  A: {answer}")

    # Arguments — suppress court_response when blinded (court's response reveals outcome)
    arguments = graph.get("arguments") or []
    pet_args = [a for a in arguments if isinstance(a, dict) and
                a.get("actor") in ("petitioner", "appellant", "complainant", "prosecution")]
    resp_args = [a for a in arguments if isinstance(a, dict) and
                 a.get("actor") in ("respondent", "accused")]
    court_args = [a for a in arguments if isinstance(a, dict) and a.get("actor") == "court"]

    if pet_args or resp_args:
        parts.append("ARGUMENTS:")
        for a in pet_args[:max_args // 2]:
            claim = a.get("claim", "")[:200]
            parts.append(f"  [Petitioner] {claim}")
            if not blind:
                resp = a.get("court_response", "")
                if resp:
                    parts.append(f"    → Court: {resp}")
        for a in resp_args[:max_args // 2]:
            claim = a.get("claim", "")[:200]
            parts.append(f"  [Respondent] {claim}")
            if not blind:
                resp = a.get("court_response", "")
                if resp:
                    parts.append(f"    → Court: {resp}")

    # Holdings — suppress entirely when blinded (holdings ARE the court's decision)
    if not blind:
        holdings = graph.get("holdings") or []
        if holdings:
            parts.append("HOLDINGS:")
            for h in holdings[:max_holdings]:
                if isinstance(h, dict):
                    text = h.get("text", "")[:200]
                    reasoning = h.get("reasoning_summary", "")
                    parts.append(f"  {text}")
                    if reasoning:
                        parts.append(f"    Reasoning: {reasoning[:150]}")

    # Precedents (compact)
    precedents = graph.get("precedents") or []
    if precedents:
        prec_strs = []
        for pr in precedents[:max_precedents]:
            if isinstance(pr, dict):
                name = pr.get("case_name") or pr.get("citation", "")
                if blind:
                    prec_strs.append(name)
                else:
                    treatment = pr.get("treatment", "cited")
                    prec_strs.append(f"{name} ({treatment})")
        if prec_strs:
            parts.append(f"PRECEDENTS: {'; '.join(prec_strs)}")

    # Outcome (only for neighbor cases, not query; suppress in behavior mode)
    if not hide_outcome:
        outcome = graph.get("outcome")
        if isinstance(outcome, dict):
            disp = outcome.get("disposition", "unknown")
            parts.append(f"OUTCOME: {disp}")

    return "\n".join(parts)


def _build_prediction_prompt(query_graph: dict, neighbor_graphs: List[Tuple[dict, int, float]],
                             behavior: bool = False) -> Tuple[str, str]:
    """Build system + user prompt for LLM prediction.

    Args:
        query_graph: the case to predict
        neighbor_graphs: list of (graph_dict, label, similarity_score)
        behavior: If True, show neighbors' court reasoning (court_response,
                  holdings, issue answers) but hide the final outcome label.
                  Forces the LLM to reason from judicial behavior patterns
                  rather than counting outcome votes.

    Returns:
        (system_prompt, user_prompt)
    """
    if behavior:
        system = """You are an expert legal analyst specializing in Indian Supreme Court cases.

Your task: Given a query case (with facts, legal concepts, issues, and arguments from both 
parties — but WITHOUT knowing how the court ruled) and similar previously-decided cases 
(showing how the court responded to arguments, what it held, and how it reasoned — but NOT 
the final outcome), predict whether the appeal will be ACCEPTED (label=1) or REJECTED (label=0).

ACCEPTED means: allowed, partly_allowed, set_aside, remanded, or modified.
REJECTED means: dismissed.

The query case shows what the parties argued but NOT the court's response. The similar cases 
show how courts responded to similar arguments and what they held — study these judicial 
reasoning patterns to predict the query case outcome.

Pay attention to:
- How courts responded to similar arguments (accepted vs rejected specific claims)
- What courts held when dealing with the same legal provisions
- Whether holdings in similar cases favored petitioners or respondents
- The reasoning patterns — did courts find merit in similar constitutional challenges?

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{
    "prediction": 0 or 1,
    "confidence": float between 0.0 and 1.0,
    "reasoning": "2-3 sentence explanation of why"
}"""
    else:
        system = """You are an expert legal analyst specializing in Indian Supreme Court cases.

Your task: Given a query case (with facts, legal concepts, issues, and arguments from both 
parties — but WITHOUT knowing how the court ruled) and similar previously-decided cases 
(with full outcomes), predict whether the appeal will be ACCEPTED (label=1) or REJECTED (label=0).

ACCEPTED means: allowed, partly_allowed, set_aside, remanded, or modified.
REJECTED means: dismissed.

The query case shows what the parties argued but NOT the court's response. Use the similar 
cases to reason by analogy — if similar arguments under similar legal provisions led to 
acceptance or rejection, the query case is likely to follow the same pattern.

Pay attention to:
- Which legal provisions are invoked and how they were treated in similar cases
- Whether the petitioner's arguments align more with accepted or rejected case patterns
- The court's treatment of precedents in similar cases
- The strength and nature of the holdings in similar cases

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{
    "prediction": 0 or 1,
    "confidence": float between 0.0 and 1.0,
    "reasoning": "2-3 sentence explanation of why"
}"""

    # Build user prompt
    parts = []
    parts.append("=" * 60)
    parts.append("QUERY CASE (predict the outcome — court responses NOT shown):")
    parts.append("=" * 60)

    # Query case: BLINDED — strip court_response, issue answers, holdings
    query_summary = _compact_graph_summary(query_graph, max_facts=6, max_args=6,
                                           max_holdings=0, max_precedents=4,
                                           blind=True)
    # Also strip any outcome line that might have leaked in
    query_lines = [l for l in query_summary.split("\n") if not l.startswith("OUTCOME:")]
    parts.append("\n".join(query_lines))

    parts.append("")
    parts.append("=" * 60)

    if behavior:
        parts.append(f"SIMILAR CASES ({len(neighbor_graphs)} — showing court reasoning, NOT final outcomes):")
    else:
        parts.append(f"SIMILAR CASES ({len(neighbor_graphs)} most similar previously-decided cases):")
    parts.append("=" * 60)

    for i, (ng, label, sim) in enumerate(neighbor_graphs, 1):
        if behavior:
            # Show court behavior but NOT the outcome label
            parts.append(f"\n--- Similar Case {i} (similarity: {sim:.3f}) ---")
            parts.append(_compact_graph_summary(ng, max_facts=3, max_args=3,
                                                max_holdings=2, max_precedents=2,
                                                hide_outcome=True))
        else:
            outcome_str = "ACCEPTED" if label == 1 else "REJECTED"
            parts.append(f"\n--- Similar Case {i} (outcome: {outcome_str}, similarity: {sim:.3f}) ---")
            parts.append(_compact_graph_summary(ng, max_facts=3, max_args=3,
                                                max_holdings=2, max_precedents=2))

    parts.append("")
    parts.append("=" * 60)
    parts.append("Based on the query case and the similar cases above, predict the outcome.")
    parts.append("Respond with JSON only: {\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}")

    return system, "\n".join(parts)


def _build_zeroshot_prompt(query_graph: dict) -> Tuple[str, str]:
    """Build prompt for zero-shot prediction (no retrieved neighbors).

    The query case is BLINDED — no court_response, issue answers, or holdings.
    This is genuine prediction: the model sees facts, concepts, and party arguments,
    but not how the court actually ruled.
    """
    system = """You are an expert legal analyst specializing in Indian Supreme Court cases.

Your task: Given a case summary with the facts, legal concepts, issues, and arguments from 
both parties — but WITHOUT knowing how the court ruled — predict whether the appeal 
will be ACCEPTED (label=1) or REJECTED (label=0).

ACCEPTED means: allowed, partly_allowed, set_aside, remanded, or modified.
REJECTED means: dismissed.

You see what the parties argued but NOT the court's response to those arguments.
Analyze the legal concepts and argument strength to make your prediction.

Respond with ONLY this JSON (no markdown, no explanation outside the JSON):
{
    "prediction": 0 or 1,
    "confidence": float between 0.0 and 1.0,
    "reasoning": "2-3 sentence explanation of why"
}"""

    parts = []
    parts.append("Predict the outcome of this Indian Supreme Court case.")
    parts.append("NOTE: Court responses to arguments are NOT shown — you must predict from the facts and arguments alone.")
    parts.append("")
    # BLINDED: strip court_response, issue answers, holdings, interpretations, treatments
    summary = _compact_graph_summary(query_graph, max_facts=8, max_args=8,
                                     max_holdings=0, max_precedents=5,
                                     blind=True)
    # Strip outcome
    lines = [l for l in summary.split("\n") if not l.startswith("OUTCOME:")]
    parts.append("\n".join(lines))
    parts.append("")
    parts.append("Predict: {\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}")

    return system, "\n".join(parts)


# =============================================================================
# LLM CLIENT (simplified, reuses GrokClient interface)
# =============================================================================

async def llm_predict(api_key: str, system: str, prompt: str,
                      model: str = "grok-4-1-fast-reasoning", temperature: float = 0.1) -> dict:
    """Call LLM API and parse JSON response.

    Uses grok-4-1-fast-reasoning for strong legal reasoning capability.
    """
    import httpx

    system_full = system + "\n\nYou MUST respond with valid JSON only. No markdown, no ```json blocks."

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
                        {"role": "system", "content": system_full},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 16384
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response (handle markdown wrapping, thinking tokens, etc.)
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)

            # If reasoning model includes thinking before JSON, extract the JSON object
            if not content.startswith("{"):
                # Find the last JSON object in the response
                json_match = re.search(r'\{[^{}]*"prediction"[^{}]*\}', content)
                if json_match:
                    content = json_match.group(0)

            return json.loads(content)

        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
            return {"prediction": -1, "confidence": 0.0, "reasoning": f"Error: {str(e)[:100]}"}


# =============================================================================
# EMBEDDING-BASED RETRIEVAL EVALUATION
# =============================================================================

def run_embedding_retrieval(graphs, labels, k=10, model_name="all-MiniLM-L6-v2",
                            methods=None, do_bootstrap=False):
    """LOO evaluation using dense embedding retrieval instead of TF-IDF/fuzzy."""
    if methods is None:
        methods = ["majority_vote", "weighted_vote", "distance_decay_vote"]

    corpus = [(extract_concept_profile(g, labels[c]), g) for c, g in graphs if c in labels]
    n = len(corpus)
    profiles = [p for p, _ in corpus]
    _, family_idf = compute_idf_multi(profiles)

    print(f"Evaluating {n} cases (LOO, k={k}) with dense embeddings")
    print(f"Labels: {sum(1 for p in profiles if p.label == 1)} accepted, "
          f"{sum(1 for p in profiles if p.label == 0)} rejected")

    # Build embedding index
    embed_sim = EmbeddingSimilarity(profiles, model_name=model_name, use_full_text=True)

    # LOO evaluation
    results = {m: {"probs": [], "trues": []} for m in methods}

    for i, (qp, qg) in enumerate(corpus):
        # Get neighbors by embedding similarity
        nbr_ids = embed_sim.get_neighbors(qp.case_id, k=k)

        # Build RetrievalResult objects for prediction functions
        nbrs = []
        for nbr_cid, sim_score in nbr_ids:
            # Find the profile and graph for this neighbor
            for p, g in corpus:
                if p.case_id == nbr_cid:
                    nbrs.append(RetrievalResult(
                        case_id=nbr_cid, similarity=sim_score,
                        label=p.label, outcome=p.outcome,
                        shared_concepts=[], shared_families=[],
                        shared_precedents=[],
                        reasoning_chains=[],
                    ))
                    break

        for m in methods:
            pred, prob, _ = predict_from_neighbors(nbrs, m, family_idf)
            results[m]["probs"].append(prob)
            results[m]["trues"].append(qp.label)

        if (i + 1) % 50 == 0:
            bm = methods[1] if len(methods) > 1 else methods[0]
            trues_so_far = np.array(results[bm]["trues"])
            probs_so_far = np.array(results[bm]["probs"])
            preds = (probs_so_far >= 0.5).astype(int)
            acc = float(np.mean(preds == trues_so_far))
            print(f"  [{i + 1}/{n}] {bm} acc: {acc:.3f}")

    # Compute metrics
    final = {}
    for m in methods:
        trues = np.array(results[m]["trues"])
        probs = np.array(results[m]["probs"])

        # Fixed threshold
        preds_50 = (probs >= 0.5).astype(int)
        acc_50 = float(np.mean(preds_50 == trues))

        # Optimal threshold
        opt_t, _ = find_optimal_threshold(trues, probs)
        m_opt = _compute_metrics_at_threshold(trues, probs, opt_t)

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(trues, probs)
        except:
            auc = float("nan")

        m_50 = _compute_metrics_at_threshold(trues, probs, 0.5)

        entry = {
            "accuracy": round(acc_50, 4),
            "f1": round(m_50["f1"], 4),
            "precision": round(m_50["precision"], 4),
            "recall": round(m_50["recall"], 4),
            "opt_threshold": round(opt_t, 2),
            "opt_accuracy": round(m_opt["accuracy"], 4),
            "opt_f1": round(m_opt["f1"], 4),
            "opt_precision": round(m_opt["precision"], 4),
            "opt_recall": round(m_opt["recall"], 4),
            "auc": round(auc, 4) if not math.isnan(auc) else "N/A",
            "n": len(trues),
        }
        if do_bootstrap:
            entry.update(bootstrap_ci(trues, preds_50, probs))
        final[m] = entry

    return final


# =============================================================================
# LLM PREDICTION EVALUATION
# =============================================================================

async def run_llm_evaluation(graphs, labels, api_key: str, mode: str = "hybrid",
                             k: int = 5, n_cases: int = 50, model: str = "grok-4-1-fast-reasoning",
                             embed_model: str = "all-MiniLM-L6-v2",
                             concurrent: int = 10, start: int = 0):
    """Run LLM-based prediction evaluation.

    Args:
        mode: "zeroshot" | "hybrid" (embed + outcome labels) |
              "concept_hybrid" (concept + outcome labels) |
              "embed_behavior" (embed + court behavior, no labels) |
              "concept_behavior" (concept + court behavior, no labels)
        k: number of neighbors for hybrid/behavior modes
        n_cases: how many cases to evaluate (API calls = n_cases)
        model: LLM model to use
        concurrent: max concurrent API calls
    """
    corpus = [(extract_concept_profile(g, labels[c]), g) for c, g in graphs if c in labels]
    n = len(corpus)
    print(f"\nLLM {mode} evaluation: {n_cases} cases, model={model}")
    print(f"Labels: {sum(1 for p, _ in corpus if p.label == 1)} accepted, "
          f"{sum(1 for p, _ in corpus if p.label == 0)} rejected")

    # Build retrieval index based on mode
    neighbor_index = None
    if mode in ("hybrid", "embed_behavior"):
        profiles = [p for p, _ in corpus]
        neighbor_index = EmbeddingSimilarity(profiles, model_name=embed_model, use_full_text=True)
    elif mode in ("concept_hybrid", "concept_behavior"):
        neighbor_index = ConceptNeighborIndex(corpus)

    # Select cases (stratified sample if n_cases < n)
    if n_cases < n:
        rng = np.random.RandomState(42)
        acc_idx = [i for i, (p, _) in enumerate(corpus) if p.label == 1]
        rej_idx = [i for i, (p, _) in enumerate(corpus) if p.label == 0]
        # Proportional stratified sample
        n_acc = max(1, int(n_cases * len(acc_idx) / n))
        n_rej = n_cases - n_acc
        selected = sorted(
            list(rng.choice(acc_idx, min(n_acc, len(acc_idx)), replace=False)) +
            list(rng.choice(rej_idx, min(n_rej, len(rej_idx)), replace=False))
        )
    else:
        selected = list(range(n))

    print(f"Selected {len(selected)} cases (stratified)")

    # Run predictions
    semaphore = asyncio.Semaphore(concurrent)
    predictions = []
    errors = 0

    async def predict_one(idx):
        nonlocal errors
        qp, qg = corpus[idx]

        if mode in ("hybrid", "concept_hybrid", "embed_behavior", "concept_behavior") and neighbor_index:
            # Get neighbors from whichever index was built
            nbr_ids = neighbor_index.get_neighbors(qp.case_id, k=k)
            neighbor_data = []
            for nbr_cid, sim_score in nbr_ids:
                for p, g in corpus:
                    if p.case_id == nbr_cid:
                        neighbor_data.append((g, p.label, sim_score))
                        break
            use_behavior = mode in ("embed_behavior", "concept_behavior")
            system, prompt = _build_prediction_prompt(qg, neighbor_data, behavior=use_behavior)
        else:
            system, prompt = _build_zeroshot_prompt(qg)

        async with semaphore:
            result = await llm_predict(api_key, system, prompt, model=model)

        pred = result.get("prediction", -1)
        conf = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")

        if pred not in (0, 1):
            errors += 1
            return None

        return {
            "case_id": qp.case_id,
            "true_label": qp.label,
            "prediction": pred,
            "confidence": conf,
            "correct": pred == qp.label,
            "reasoning": reasoning,
        }

    # Process in batches
    t0 = time.time()
    for batch_start in range(0, len(selected), concurrent):
        batch = selected[batch_start:batch_start + concurrent]
        tasks = [asyncio.create_task(predict_one(idx)) for idx in batch]
        batch_results = await asyncio.gather(*tasks)
        for r in batch_results:
            if r is not None:
                predictions.append(r)

        done = len(predictions) + errors
        if done % 10 == 0 or done == len(selected):
            acc = sum(1 for p in predictions if p["correct"]) / max(len(predictions), 1)
            print(f"  [{done}/{len(selected)}] acc: {acc:.3f} (errors: {errors})")

    elapsed = time.time() - t0

    # Compute metrics
    if not predictions:
        print("No valid predictions!")
        return {}

    trues = np.array([p["true_label"] for p in predictions])
    preds = np.array([p["prediction"] for p in predictions])
    confs = np.array([p["confidence"] for p in predictions])

    # Use confidence as probability for AUC
    # Flip confidence for predicted-0 to get P(accepted)
    probs = np.where(preds == 1, confs, 1.0 - confs)

    acc = float(np.mean(preds == trues))
    tp = int(np.sum((preds == 1) & (trues == 1)))
    fp = int(np.sum((preds == 1) & (trues == 0)))
    fn = int(np.sum((preds == 0) & (trues == 1)))
    tn = int(np.sum((preds == 0) & (trues == 0)))
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(trues, probs)
    except:
        auc = float("nan")

    result = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(pr, 4),
        "recall": round(rc, 4),
        "auc": round(auc, 4) if not math.isnan(auc) else "N/A",
        "n": len(predictions),
        "errors": errors,
        "time_s": round(elapsed, 1),
        "model": model,
        "mode": mode,
    }

    # Print confusion matrix
    print(f"\n  Confusion matrix:")
    print(f"             Pred=Acc  Pred=Rej")
    print(f"  True=Acc     {tp:4d}      {fn:4d}")
    print(f"  True=Rej     {fp:4d}      {tn:4d}")

    # Show some example predictions
    print(f"\n  Example predictions:")
    correct = [p for p in predictions if p["correct"]][:3]
    wrong = [p for p in predictions if not p["correct"]][:3]
    for p in correct:
        label = "ACC" if p["true_label"] == 1 else "REJ"
        print(f"    ✓ {p['case_id']}: {label} (conf={p['confidence']:.2f}) {p['reasoning'][:80]}")
    for p in wrong:
        label = "ACC" if p["true_label"] == 1 else "REJ"
        pred_s = "ACC" if p["prediction"] == 1 else "REJ"
        print(f"    ✗ {p['case_id']}: true={label} pred={pred_s} (conf={p['confidence']:.2f}) {p['reasoning'][:80]}")

    return result


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_embed_retrieval(args):
    """Evaluate dense embedding retrieval."""
    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()

    r = run_embedding_retrieval(graphs, labels, k=args.k, model_name=args.embed_model)

    print(f"\n{'=' * 95}")
    print(f"EMBEDDING RETRIEVAL RESULTS (model={args.embed_model}, k={args.k})")
    print(f"{'=' * 95}")
    hdr = f"{'Method':<35} {'Acc@.5':>7} {'F1@.5':>7} {'AUC':>7} | {'Acc*':>7} {'F1*':>7} {'P*':>7} {'R*':>7} {'t*':>5}"
    print(hdr)
    print("-" * 95)
    for m, met in r.items():
        a_s = f"{met['auc']:>7.4f}" if isinstance(met['auc'], float) else f"{'N/A':>7}"
        print(f"{'Embed: ' + m:<35} {met['accuracy']:>7.4f} {met['f1']:>7.4f} {a_s} | "
              f"{met['opt_accuracy']:>7.4f} {met['opt_f1']:>7.4f} {met['opt_precision']:>7.4f} "
              f"{met['opt_recall']:>7.4f} {met['opt_threshold']:>5.2f}")
    print("-" * 95)

    # K-sweep
    if args.k_sweep:
        print(f"\nK-SWEEP (weighted_vote):")
        print(f"{'k':>4} {'AUC':>8} {'Acc*':>8} {'F1*':>8}")
        print("-" * 32)
        for k in [3, 5, 7, 10, 15, 20]:
            r = run_embedding_retrieval(graphs, labels, k=k, model_name=args.embed_model,
                                        methods=["weighted_vote"])
            m = r["weighted_vote"]
            a_s = f"{m['auc']:>8.4f}" if isinstance(m['auc'], float) else f"{'N/A':>8}"
            print(f"{k:>4} {a_s} {m['opt_accuracy']:>8.4f} {m['opt_f1']:>8.4f}")


def cmd_llm_zeroshot(args):
    """Evaluate LLM zero-shot prediction."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()

    result = asyncio.run(run_llm_evaluation(
        graphs, labels, api_key, mode="zeroshot",
        n_cases=args.n, model=args.model, concurrent=args.concurrent
    ))

    _print_llm_result("LLM ZERO-SHOT", result)


def cmd_llm_hybrid(args):
    """Evaluate LLM hybrid (retrieve + reason)."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()

    result = asyncio.run(run_llm_evaluation(
        graphs, labels, api_key, mode="hybrid",
        k=args.k, n_cases=args.n, model=args.model,
        embed_model=args.embed_model, concurrent=args.concurrent
    ))

    _print_llm_result(f"LLM HYBRID (k={args.k})", result)


def cmd_llm_concept_hybrid(args):
    """Evaluate LLM concept hybrid (concept-based retrieval + LLM reason)."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()

    result = asyncio.run(run_llm_evaluation(
        graphs, labels, api_key, mode="concept_hybrid",
        k=args.k, n_cases=args.n, model=args.model,
        concurrent=args.concurrent
    ))

    _print_llm_result(f"LLM CONCEPT HYBRID (k={args.k})", result)


def cmd_llm_concept_behavior(args):
    """Evaluate LLM concept behavior (concept retrieval + court behavior, no outcome labels)."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()

    result = asyncio.run(run_llm_evaluation(
        graphs, labels, api_key, mode="concept_behavior",
        k=args.k, n_cases=args.n, model=args.model,
        concurrent=args.concurrent
    ))

    _print_llm_result(f"LLM CONCEPT BEHAVIOR (k={args.k})", result)


def cmd_llm_embed_behavior(args):
    """Evaluate LLM embed behavior (embed retrieval + court behavior, no outcome labels)."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()

    result = asyncio.run(run_llm_evaluation(
        graphs, labels, api_key, mode="embed_behavior",
        k=args.k, n_cases=args.n, model=args.model,
        embed_model=args.embed_model, concurrent=args.concurrent
    ))

    _print_llm_result(f"LLM EMBED BEHAVIOR (k={args.k})", result)


def cmd_paper_table(args):
    """Run all methods and produce the paper comparison table."""
    api_key = os.getenv("XAI_API_KEY")
    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()

    labeled = [(c, g) for c, g in graphs if c in labels]
    ar = sum(labels[c] for c, _ in labeled) / len(labeled)
    maj_acc = ar if ar >= 0.5 else 1 - ar

    # 1. Embedding retrieval (full LOO, no API needed)
    print("\n[1/4] Embedding retrieval (full LOO)...")
    embed_r = run_embedding_retrieval(graphs, labels, k=args.k, model_name=args.embed_model,
                                      methods=["weighted_vote", "distance_decay_vote"])

    all_results = {
        "Majority class": {"accuracy": round(maj_acc, 4), "f1": "---", "auc": "---"},
        **{f"Embed kNN: {m}": v for m, v in embed_r.items()},
    }

    # 2-3. LLM methods (if API key available)
    if api_key and args.n > 0:
        print(f"\n[2/4] LLM zero-shot ({args.n} cases)...")
        zs_r = asyncio.run(run_llm_evaluation(
            graphs, labels, api_key, mode="zeroshot",
            n_cases=args.n, model=args.model, concurrent=args.concurrent
        ))
        if zs_r:
            all_results["LLM zero-shot"] = zs_r

        print(f"\n[3/4] LLM hybrid ({args.n} cases, k={args.k})...")
        hy_r = asyncio.run(run_llm_evaluation(
            graphs, labels, api_key, mode="hybrid",
            k=args.k, n_cases=args.n, model=args.model,
            embed_model=args.embed_model, concurrent=args.concurrent
        ))
        if hy_r:
            all_results[f"LLM hybrid (k={args.k})"] = hy_r
    else:
        if not api_key:
            print("\n[2-3/4] Skipping LLM methods (no XAI_API_KEY)")
        else:
            print("\n[2-3/4] Skipping LLM methods (--n 0)")

    # Print final table
    print(f"\n{'=' * 100}")
    print(f"PAPER TABLE: ALL METHODS COMPARISON")
    print(f"{'=' * 100}")
    hdr = f"{'Method':<40} {'Acc':>7} {'F1':>7} {'AUC':>7} {'P':>7} {'R':>7} {'n':>5}"
    print(hdr)
    print("-" * 80)

    for name, r in all_results.items():
        if isinstance(r.get("f1"), str):
            # Majority class
            print(f"{name:<40} {r['accuracy']:>7.4f} {'---':>7} {'---':>7} {'---':>7} {'---':>7}")
            continue

        # For embedding methods, use optimal threshold results
        acc = r.get("opt_accuracy", r.get("accuracy", 0))
        f1 = r.get("opt_f1", r.get("f1", 0))
        pr = r.get("opt_precision", r.get("precision", 0))
        rc = r.get("opt_recall", r.get("recall", 0))
        auc = r.get("auc", "N/A")
        n_val = r.get("n", "?")

        a_s = f"{auc:>7.4f}" if isinstance(auc, float) else f"{'N/A':>7}"
        print(f"{name:<40} {acc:>7.4f} {f1:>7.4f} {a_s} {pr:>7.4f} {rc:>7.4f} {n_val:>5}")

    print("-" * 80)


def _print_llm_result(title, result):
    if not result:
        print("No results.")
        return
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    auc_s = f"{result['auc']:.4f}" if isinstance(result.get('auc'), float) else "N/A"
    print(f"  Accuracy:  {result['accuracy']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  AUC:       {auc_s}")
    print(f"  n={result['n']}, errors={result.get('errors', 0)}, time={result.get('time_s', '?')}s")
    print(f"  Model: {result.get('model', '?')}")


def main():
    p = argparse.ArgumentParser(description="Hybrid LLM evaluation for legal reasoning graphs")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common args
    def add_common(sp):
        sp.add_argument("--graph_dir", required=True)
        sp.add_argument("--k", type=int, default=5)
        sp.add_argument("--embed_model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")

    # Embedding retrieval
    sp = sub.add_parser("embed_retrieval", help="Dense embedding retrieval evaluation")
    add_common(sp)
    sp.add_argument("--k_sweep", action="store_true")

    # LLM zero-shot
    sp = sub.add_parser("llm_zeroshot", help="LLM zero-shot prediction")
    add_common(sp)
    sp.add_argument("--n", type=int, default=50, help="Number of cases to evaluate")
    sp.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    sp.add_argument("--concurrent", type=int, default=10)

    # LLM hybrid (embedding retrieval)
    sp = sub.add_parser("llm_hybrid", help="LLM hybrid (embed retrieval + reason)")
    add_common(sp)
    sp.add_argument("--n", type=int, default=50, help="Number of cases to evaluate")
    sp.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    sp.add_argument("--concurrent", type=int, default=10)

    # LLM concept hybrid (concept-based retrieval)
    sp = sub.add_parser("llm_concept_hybrid", help="LLM concept hybrid (concept retrieval + reason)")
    add_common(sp)
    sp.add_argument("--n", type=int, default=50, help="Number of cases to evaluate")
    sp.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    sp.add_argument("--concurrent", type=int, default=10)

    # LLM concept behavior (concept retrieval + court behavior, no outcome labels)
    sp = sub.add_parser("llm_concept_behavior", help="LLM concept + court behavior (no outcome labels)")
    add_common(sp)
    sp.add_argument("--n", type=int, default=50, help="Number of cases to evaluate")
    sp.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    sp.add_argument("--concurrent", type=int, default=10)

    # LLM embed behavior (embed retrieval + court behavior, no outcome labels)
    sp = sub.add_parser("llm_embed_behavior", help="LLM embed + court behavior (no outcome labels)")
    add_common(sp)
    sp.add_argument("--n", type=int, default=50, help="Number of cases to evaluate")
    sp.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    sp.add_argument("--concurrent", type=int, default=10)

    # Paper table (all methods)
    sp = sub.add_parser("paper_table", help="Run all methods for paper comparison")
    add_common(sp)
    sp.add_argument("--n", type=int, default=100, help="Number of cases for LLM methods")
    sp.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    sp.add_argument("--concurrent", type=int, default=10)

    args = p.parse_args()
    cmds = {
        "embed_retrieval": cmd_embed_retrieval,
        "llm_zeroshot": cmd_llm_zeroshot,
        "llm_hybrid": cmd_llm_hybrid,
        "llm_concept_hybrid": cmd_llm_concept_hybrid,
        "llm_concept_behavior": cmd_llm_concept_behavior,
        "llm_embed_behavior": cmd_llm_embed_behavior,
        "paper_table": cmd_paper_table,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()