#!/usr/bin/env python3
"""
eval_retrieval_baselines.py — Concept retrieval evaluation for SIGIR 2025.

Given a legal concept as a query, retrieve the most relevant cases from a
corpus of ~2,518 Indian Supreme Court case graphs.  Compares 5 retrieval
methods using standard IR metrics (nDCG@10, MAP, P@10).

Methods:
  1. TF-IDF         — sklearn TfidfVectorizer + cosine similarity
  2. BM25            — rank_bm25 BM25Okapi
  3. E5-Large        — SentenceTransformer dense retrieval (dot product)
  4. BM25+E5 (RRF)   — Reciprocal Rank Fusion of BM25 + E5
  5. ConceptSet      — exact/fuzzy concept_id matching with graded relevance

Usage:
  python eval_retrieval_baselines.py --graph_dir iltur_graphs [--n_queries 50]
         [--cache_dir .] [--skip_e5]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Reused helpers from eval_concept_retrieval.py
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_graphs(graph_dir, tier_filter=None):
    results, skipped = [], 0
    for p in sorted(graph_dir.glob("*.json")):
        if p.name == "checkpoint.json":
            continue
        try:
            g = load_json(p)
            if not (isinstance(g, dict) and isinstance(g.get("case_id"), str)):
                continue
            if tier_filter:
                if g.get("quality_tier", "").lower() not in tier_filter:
                    skipped += 1
                    continue
            results.append((g["case_id"], g))
        except Exception:
            continue
    if tier_filter and skipped:
        print(f"  [tier filter] kept {len(results)}, skipped {skipped}")
    return results


def build_full_case_text(graph):
    """Build rich document text from all graph fields."""
    parts = []
    for f in (graph.get("facts") or []):
        if isinstance(f, dict):
            t = f.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    for iss in (graph.get("issues") or []):
        if isinstance(iss, dict):
            t = iss.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    for a in (graph.get("arguments") or []):
        if isinstance(a, dict):
            t = a.get("claim")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
            cr = a.get("court_reasoning")
            if isinstance(cr, str) and cr.strip():
                parts.append(cr.strip())
    for h in (graph.get("holdings") or []):
        if isinstance(h, dict):
            t = h.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
            rs = h.get("reasoning_summary")
            if isinstance(rs, str) and rs.strip():
                parts.append(rs.strip())
    for c in (graph.get("concepts") or []):
        if isinstance(c, dict):
            for fld in ("unlisted_label", "unlisted_description", "interpretation"):
                val = c.get(fld)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
    for p in (graph.get("precedents") or []):
        if isinstance(p, dict):
            prop = p.get("cited_proposition")
            if isinstance(prop, str) and prop.strip():
                parts.append(prop.strip())
    return " ".join(parts)


# Fuzzy concept matching ---------------------------------------------------

_ABBREV_MAP = {
    "ipc": "indian_penal_code", "crpc": "code_criminal_procedure",
    "cpc": "code_civil_procedure", "coa": "constitution_india",
    "con": "constitution", "const": "constitution", "art": "article",
    "sec": "section", "s": "section", "subs": "subsection",
    "cl": "clause", "r": "rule", "o": "order", "sch": "schedule",
    "para": "paragraph", "amdt": "amendment",
}

_LEGAL_STOP = {
    "act", "the", "of", "for", "and", "in", "to", "under",
    "with", "by", "a", "an", "on", "or",
}

_SECTION_RE = re.compile(
    r'^(?:s|sec|section|art|article|rule|order|cl|clause|para|subs|subsection)(\d+[a-z]?)$',
    re.I,
)


def _normalize_concept_tokens(concept_id: str) -> Set[str]:
    raw = concept_id
    if raw.startswith("UNLISTED_"):
        raw = raw[len("UNLISTED_"):]
    tokens = re.split(r'[_\-\s/]+', raw.lower())
    normalized: Set[str] = set()
    for tok in tokens:
        if not tok or tok in _LEGAL_STOP:
            continue
        m = _SECTION_RE.match(tok)
        if m:
            normalized.add(f"sec_{m.group(1)}")
            continue
        expanded = _ABBREV_MAP.get(tok)
        if expanded is not None:
            for sub in expanded.split("_"):
                if sub and sub not in _LEGAL_STOP:
                    normalized.add(sub)
            continue
        if tok.isdigit():
            normalized.add(f"sec_{tok}")
            continue
        normalized.add(tok)
    return normalized


def _fuzzy_token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    shared = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(shared) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Graded relevance weights
# ---------------------------------------------------------------------------

RELEVANCE_GRADE = {"central": 3, "supporting": 2, "mentioned": 1, "obiter": 1}


# ---------------------------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_corpus(graphs):
    """Return (case_ids, doc_texts, graphs_list)."""
    case_ids, doc_texts, graph_list = [], [], []
    for cid, g in graphs:
        text = build_full_case_text(g)
        if not text.strip():
            text = "empty"
        case_ids.append(cid)
        doc_texts.append(text)
        graph_list.append(g)
    return case_ids, doc_texts, graph_list


def collect_concepts(graph_list, case_ids):
    """Scan all concepts across all cases.

    Returns
    -------
    concept_info : dict[concept_id -> {df, cases, metadata}]
        df: number of distinct cases containing this concept
        cases: dict[case_idx -> max_grade]
        metadata: dict with unlisted_label, unlisted_description, interpretation
    """
    concept_info: Dict[str, Dict[str, Any]] = {}

    for idx, g in enumerate(graph_list):
        seen_in_case: Dict[str, int] = {}  # concept_id -> max grade in this case
        for c in (g.get("concepts") or []):
            if not isinstance(c, dict):
                continue
            cid = c.get("concept_id", "")
            if not isinstance(cid, str) or not cid:
                continue
            grade = RELEVANCE_GRADE.get(c.get("relevance", "mentioned"), 1)
            seen_in_case[cid] = max(seen_in_case.get(cid, 0), grade)

            # Store metadata (first seen wins for label/description)
            if cid not in concept_info:
                concept_info[cid] = {
                    "df": 0,
                    "cases": {},
                    "unlisted_label": None,
                    "unlisted_description": None,
                    "interpretation": None,
                }
            info = concept_info[cid]
            for fld in ("unlisted_label", "unlisted_description", "interpretation"):
                if info[fld] is None:
                    val = c.get(fld)
                    if isinstance(val, str) and val.strip():
                        info[fld] = val.strip()

        for cid, grade in seen_in_case.items():
            concept_info[cid]["df"] += 1
            concept_info[cid]["cases"][idx] = max(
                concept_info[cid]["cases"].get(idx, 0), grade
            )

    return concept_info


def select_queries(concept_info, n_corpus, n_queries=50):
    """Filter concepts with 3 <= df <= 50% of corpus, sort by df desc, take top n_queries."""
    max_df = n_corpus // 2
    eligible = [
        (cid, info)
        for cid, info in concept_info.items()
        if 3 <= info["df"] <= max_df
    ]
    eligible.sort(key=lambda x: x[1]["df"], reverse=True)
    return eligible[:n_queries]


def build_query_text(cid, info):
    """Build a rich query string from concept metadata."""
    parts = []

    # Primary: label + description
    if info["unlisted_label"]:
        parts.append(info["unlisted_label"])
    if info["unlisted_description"]:
        parts.append(info["unlisted_description"])

    # Fallback: clean concept_id
    if not parts:
        clean = cid
        if clean.startswith("UNLISTED_"):
            clean = clean[len("UNLISTED_"):]
        clean = clean.replace("_", " ")
        parts.append(clean)

    # Enrich: append interpretation (truncated)
    if info["interpretation"]:
        interp = info["interpretation"][:200]
        parts.append(interp)

    return " ".join(parts)


def build_qrels(queries, binary=True):
    """Build qrels: list of dicts {doc_idx: relevance} for each query.

    binary=True  -> relevance is 0 or 1
    binary=False -> graded relevance (1, 2, or 3)
    """
    qrels = []
    for cid, info in queries:
        rel = {}
        for doc_idx, grade in info["cases"].items():
            rel[doc_idx] = 1 if binary else grade
        qrels.append(rel)
    return qrels


# ---------------------------------------------------------------------------
# 2. RETRIEVAL METHODS
# ---------------------------------------------------------------------------

def run_tfidf(doc_texts, query_texts):
    """TF-IDF retrieval. Returns scores matrix (n_queries x n_docs)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    doc_matrix = vectorizer.fit_transform(doc_texts)
    query_matrix = vectorizer.transform(query_texts)
    scores = cosine_similarity(query_matrix, doc_matrix)
    return scores  # shape (n_queries, n_docs)


def run_bm25(doc_texts, query_texts):
    """BM25 retrieval. Returns scores matrix (n_queries x n_docs)."""
    from rank_bm25 import BM25Okapi

    tokenized_docs = [doc.lower().split() for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)

    scores = np.zeros((len(query_texts), len(doc_texts)))
    for qi, qt in enumerate(query_texts):
        tokenized_query = qt.lower().split()
        scores[qi] = bm25.get_scores(tokenized_query)
    return scores


def run_e5(doc_texts, query_texts, cache_dir="."):
    """E5-Large dense retrieval. Returns scores matrix (n_queries x n_docs).

    Caches document embeddings as .npy to avoid re-encoding.
    Falls back through smaller models if needed.
    """
    cache_path = Path(cache_dir)

    # Try models in order of preference
    model_configs = [
        ("intfloat/e5-large-v2", True, "e5_large_v2"),
        ("intfloat/e5-base-v2", True, "e5_base_v2"),
        ("sentence-transformers/all-MiniLM-L6-v2", False, "minilm_l6_v2"),
    ]

    model = None
    use_prefix = True
    cache_name = ""

    for model_name, prefix, cname in model_configs:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading {model_name}...")
            model = SentenceTransformer(model_name)
            use_prefix = prefix
            cache_name = cname
            print(f"  Loaded {model_name}")
            break
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            continue

    if model is None:
        print("  ERROR: Could not load any sentence transformer model")
        return None

    # Encode documents (with caching)
    doc_emb_path = cache_path / f"doc_embeddings_{cache_name}.npy"
    if doc_emb_path.exists():
        print(f"  Loading cached doc embeddings from {doc_emb_path}")
        doc_embeddings = np.load(str(doc_emb_path))
        if doc_embeddings.shape[0] != len(doc_texts):
            print(f"  Cache size mismatch ({doc_embeddings.shape[0]} vs {len(doc_texts)}), re-encoding...")
            doc_embeddings = None
        else:
            print(f"  Loaded {doc_embeddings.shape}")
    else:
        doc_embeddings = None

    if doc_embeddings is None:
        # Truncate docs to ~2000 chars for transformer input
        truncated_docs = [d[:2000] for d in doc_texts]
        if use_prefix:
            truncated_docs = [f"passage: {d}" for d in truncated_docs]
        print(f"  Encoding {len(truncated_docs)} documents...")
        doc_embeddings = model.encode(truncated_docs, show_progress_bar=True, batch_size=32)
        np.save(str(doc_emb_path), doc_embeddings)
        print(f"  Saved doc embeddings to {doc_emb_path}")

    # Encode queries
    if use_prefix:
        prefixed_queries = [f"query: {q}" for q in query_texts]
    else:
        prefixed_queries = query_texts
    print(f"  Encoding {len(prefixed_queries)} queries...")
    query_embeddings = model.encode(prefixed_queries, show_progress_bar=False)

    # Dot product similarity
    scores = query_embeddings @ doc_embeddings.T
    return scores


def run_rrf(scores_a, scores_b, k=60):
    """Reciprocal Rank Fusion of two score matrices. Returns fused scores."""
    n_queries, n_docs = scores_a.shape
    fused = np.zeros((n_queries, n_docs))

    for qi in range(n_queries):
        # Rank by score descending (0-based ranks)
        rank_a = np.argsort(-scores_a[qi])
        rank_b = np.argsort(-scores_b[qi])

        # Convert to rank positions
        pos_a = np.empty(n_docs, dtype=int)
        pos_b = np.empty(n_docs, dtype=int)
        pos_a[rank_a] = np.arange(n_docs)
        pos_b[rank_b] = np.arange(n_docs)

        # RRF: score = sum of 1/(k + rank + 1) for 0-based rank
        fused[qi] = 1.0 / (k + pos_a + 1) + 1.0 / (k + pos_b + 1)

    return fused


def run_concept_set(queries, graph_list, case_ids, fuzzy_threshold=0.25):
    """ConceptSet retrieval: match query concept against case concepts.

    For each query concept, check all cases for exact concept_id match or
    fuzzy match (token Jaccard >= threshold).  Score = graded relevance
    weight of matched concept.

    Returns scores matrix (n_queries x n_docs).
    """
    n_docs = len(graph_list)

    # Pre-compute normalized tokens for all concepts in all cases
    # case_concept_tokens[doc_idx] = list of (concept_id, tokens, grade)
    case_concept_data: List[List[Tuple[str, Set[str], int]]] = []
    for idx, g in enumerate(graph_list):
        concept_entries = []
        seen: Dict[str, int] = {}
        for c in (g.get("concepts") or []):
            if not isinstance(c, dict):
                continue
            cid = c.get("concept_id", "")
            if not isinstance(cid, str) or not cid:
                continue
            grade = RELEVANCE_GRADE.get(c.get("relevance", "mentioned"), 1)
            if cid in seen:
                seen[cid] = max(seen[cid], grade)
            else:
                seen[cid] = grade
        for cid, grade in seen.items():
            tokens = _normalize_concept_tokens(cid)
            concept_entries.append((cid, tokens, grade))
        case_concept_data.append(concept_entries)

    scores = np.zeros((len(queries), n_docs))

    for qi, (query_cid, query_info) in enumerate(queries):
        query_tokens = _normalize_concept_tokens(query_cid)

        for doc_idx in range(n_docs):
            best_score = 0.0
            for cid, tokens, grade in case_concept_data[doc_idx]:
                # Exact match
                if cid == query_cid:
                    best_score = max(best_score, float(grade))
                    continue
                # Fuzzy match
                sim = _fuzzy_token_jaccard(query_tokens, tokens)
                if sim >= fuzzy_threshold:
                    best_score = max(best_score, float(grade) * sim)
            scores[qi, doc_idx] = best_score

    return scores


# ---------------------------------------------------------------------------
# 3. IR METRICS
# ---------------------------------------------------------------------------

def dcg_at_k(relevances, k=10):
    """Compute DCG@k from a list of relevances in rank order."""
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(ranking_rels, qrel_graded, k=10):
    """Compute nDCG@k.

    ranking_rels: relevance scores in rank order
    qrel_graded: dict of {doc_idx: graded_relevance} for the ideal ranking
    """
    actual = dcg_at_k(ranking_rels, k)
    # Ideal: sort all relevant docs by grade descending
    ideal_rels = sorted(qrel_graded.values(), reverse=True)
    ideal = dcg_at_k(ideal_rels, k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def average_precision(ranking_doc_ids, qrel_binary):
    """Compute AP over the full ranking (binary relevance)."""
    if not qrel_binary:
        return 0.0
    n_rel = len(qrel_binary)
    hits = 0
    sum_prec = 0.0
    for rank, doc_id in enumerate(ranking_doc_ids, 1):
        if doc_id in qrel_binary:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / n_rel if n_rel > 0 else 0.0


def precision_at_k(ranking_doc_ids, qrel_binary, k=10):
    """Compute P@k (binary relevance)."""
    top_k = ranking_doc_ids[:k]
    hits = sum(1 for d in top_k if d in qrel_binary)
    return hits / k


def evaluate_method(scores_matrix, qrels_binary, qrels_graded, k=10, seed=42):
    """Evaluate a retrieval method.

    Parameters
    ----------
    scores_matrix : ndarray (n_queries, n_docs)
    qrels_binary : list of dicts {doc_idx: 1}
    qrels_graded : list of dicts {doc_idx: grade}
    k : int
    seed : int — for tie-breaking reproducibility

    Returns
    -------
    per_query : list of dicts with ndcg, ap, p10
    mean_metrics : dict with nDCG@10, MAP, P@10
    """
    rng = np.random.RandomState(seed)
    n_queries = scores_matrix.shape[0]
    per_query = []

    for qi in range(n_queries):
        query_scores = scores_matrix[qi].copy()
        # Add tiny random noise for tie-breaking
        query_scores += rng.uniform(0, 1e-10, size=query_scores.shape)
        # Rank by score descending
        ranked_indices = np.argsort(-query_scores).tolist()

        # nDCG@10 (graded)
        ranking_rels = [qrels_graded[qi].get(idx, 0) for idx in ranked_indices[:k]]
        ndcg = ndcg_at_k(ranking_rels, qrels_graded[qi], k)

        # MAP (binary, full ranking)
        ap = average_precision(ranked_indices, qrels_binary[qi])

        # P@10 (binary)
        p10 = precision_at_k(ranked_indices, qrels_binary[qi], k)

        per_query.append({"ndcg": ndcg, "ap": ap, "p10": p10})

    ndcgs = [pq["ndcg"] for pq in per_query]
    aps = [pq["ap"] for pq in per_query]
    p10s = [pq["p10"] for pq in per_query]

    mean_metrics = {
        "nDCG@10": float(np.mean(ndcgs)),
        "MAP": float(np.mean(aps)),
        "P@10": float(np.mean(p10s)),
    }
    return per_query, mean_metrics


# ---------------------------------------------------------------------------
# 4. STATISTICAL SIGNIFICANCE
# ---------------------------------------------------------------------------

def paired_ttest(scores_a, scores_b):
    """Paired t-test on per-query metric arrays. Returns (t_stat, p_value)."""
    from scipy.stats import ttest_rel
    a = np.array(scores_a)
    b = np.array(scores_b)
    if np.allclose(a, b):
        return 0.0, 1.0
    t_stat, p_value = ttest_rel(a, b)
    return float(t_stat), float(p_value)


def significance_label(p_value):
    if p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Concept retrieval evaluation for SIGIR 2025"
    )
    parser.add_argument("--graph_dir", required=True, help="Directory of case graph JSONs")
    parser.add_argument("--n_queries", type=int, default=50, help="Number of query concepts")
    parser.add_argument("--cache_dir", default=".", help="Directory for E5 embedding cache")
    parser.add_argument("--skip_e5", action="store_true", help="Skip E5 (dense) retrieval")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    print("Loading case graphs...")
    graphs = iter_graphs(Path(args.graph_dir))
    print(f"  Loaded {len(graphs)} cases")

    case_ids, doc_texts, graph_list = prepare_corpus(graphs)
    n_corpus = len(case_ids)

    # ------------------------------------------------------------------
    # Collect concepts and select queries
    # ------------------------------------------------------------------
    print("Collecting concepts...")
    concept_info = collect_concepts(graph_list, case_ids)
    print(f"  {len(concept_info)} unique concept_ids across corpus")

    queries = select_queries(concept_info, n_corpus, args.n_queries)
    n_queries = len(queries)
    dfs = [info["df"] for _, info in queries]
    print(f"  Selected {n_queries} query concepts (df range: {min(dfs)}-{max(dfs)}, "
          f"median df: {int(np.median(dfs))})")

    # Build query texts
    query_texts = [build_query_text(cid, info) for cid, info in queries]

    # Build qrels
    qrels_binary = build_qrels(queries, binary=True)
    qrels_graded = build_qrels(queries, binary=False)

    avg_rel = float(np.mean([len(qb) for qb in qrels_binary]))

    print(f"\nQueries: {n_queries} concepts (df range: {min(dfs)}-{max(dfs)}, "
          f"median df: {int(np.median(dfs))})")
    print(f"Corpus: {n_corpus} cases")
    print(f"Avg relevant docs/query: {avg_rel:.1f}")

    # ------------------------------------------------------------------
    # Run retrieval methods
    # ------------------------------------------------------------------
    methods_results = {}  # name -> (per_query, mean_metrics)

    # 1. TF-IDF
    print("\n[1/5] TF-IDF...")
    tfidf_scores = run_tfidf(doc_texts, query_texts)
    pq, mm = evaluate_method(tfidf_scores, qrels_binary, qrels_graded)
    methods_results["TF-IDF"] = (pq, mm)
    print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")

    # 2. BM25
    print("\n[2/5] BM25...")
    bm25_scores = run_bm25(doc_texts, query_texts)
    pq, mm = evaluate_method(bm25_scores, qrels_binary, qrels_graded)
    methods_results["BM25"] = (pq, mm)
    print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")

    # 3. E5-Large (optional)
    e5_scores = None
    if not args.skip_e5:
        print("\n[3/5] E5-Large...")
        e5_scores = run_e5(doc_texts, query_texts, cache_dir=args.cache_dir)
        if e5_scores is not None:
            pq, mm = evaluate_method(e5_scores, qrels_binary, qrels_graded)
            methods_results["E5-Large"] = (pq, mm)
            print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")
        else:
            print("  Skipped (no model available)")
    else:
        print("\n[3/5] E5-Large... SKIPPED (--skip_e5)")

    # 4. Hybrid RRF (BM25 + E5)
    if e5_scores is not None:
        print("\n[4/5] BM25+E5 (RRF)...")
        rrf_scores = run_rrf(bm25_scores, e5_scores, k=60)
        pq, mm = evaluate_method(rrf_scores, qrels_binary, qrels_graded)
        methods_results["BM25+E5 (RRF)"] = (pq, mm)
        print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")
    else:
        print("\n[4/5] BM25+E5 (RRF)... SKIPPED (no E5 scores)")

    # 5. ConceptSet
    print("\n[5/5] ConceptSet...")
    concept_scores = run_concept_set(queries, graph_list, case_ids)
    pq, mm = evaluate_method(concept_scores, qrels_binary, qrels_graded)
    methods_results["ConceptSet"] = (pq, mm)
    print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")

    # ------------------------------------------------------------------
    # Statistical significance (ConceptSet vs each baseline)
    # ------------------------------------------------------------------
    print("\nComputing statistical significance...")
    concept_ndcgs = [pq["ndcg"] for pq in methods_results["ConceptSet"][0]]
    sig_results = {}
    for name in methods_results:
        if name == "ConceptSet":
            continue
        baseline_ndcgs = [pq["ndcg"] for pq in methods_results[name][0]]
        t_stat, p_val = paired_ttest(concept_ndcgs, baseline_ndcgs)
        sig_results[name] = (p_val, significance_label(p_val))
        print(f"  ConceptSet vs {name}: t={t_stat:.3f}, p={p_val:.4f} {sig_results[name][1]}")

    # ------------------------------------------------------------------
    # Output table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"Queries: {n_queries} concepts (df range: {min(dfs)}-{max(dfs)}, "
          f"median df: {int(np.median(dfs))})")
    print(f"Corpus: {n_corpus} cases")
    print(f"Avg relevant docs/query: {avg_rel:.1f}")
    print(f"{'=' * 65}")

    header = f"{'Method':<17} {'nDCG@10':>8} {'MAP':>8} {'P@10':>8}    {'sig':>4}"
    print(header)
    print("-" * 52)

    # Display order
    display_order = ["TF-IDF", "BM25", "E5-Large", "BM25+E5 (RRF)", "ConceptSet"]
    for name in display_order:
        if name not in methods_results:
            continue
        _, mm = methods_results[name]
        if name == "ConceptSet":
            sig_str = "--"
        else:
            sig_str = sig_results.get(name, (1.0, ""))[1]
        print(f"{name:<17} {mm['nDCG@10']:>8.3f} {mm['MAP']:>8.3f} {mm['P@10']:>8.3f}    {sig_str:>4}")

    print("-" * 52)
    print()


if __name__ == "__main__":
    main()
