#!/usr/bin/env python3
"""
eval_concept_retrieval.py  (v4 - fuzzy matching + threshold calibration)

FIXES from v3:
1. FUZZY CONCEPT MATCHING - token-Jaccard on UNLISTED_ IDs instead of exact string match
2. FULL CASE TEXT TF-IDF - uses fact.text + argument.claim + holding.text (not just concept metadata)
3. THRESHOLD CALIBRATION - sweeps thresholds, reports optimal F1 instead of forcing 0.5
4. SIMILARITY DIAGNOSTICS - shows what's actually contributing signal
5. RICHER CONCEPT TEXT - includes interpretation + holding reasoning summaries
"""

from __future__ import annotations
import argparse, json, math, re, sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)


def iter_graphs(graph_dir, tier_filter=None):
    results, skipped = [], 0
    for p in sorted(graph_dir.glob("*.json")):
        if p.name == "checkpoint.json": continue
        try:
            g = load_json(p)
            if not (isinstance(g, dict) and isinstance(g.get("case_id"), str)): continue
            if tier_filter:
                if g.get("quality_tier", "").lower() not in tier_filter: skipped += 1; continue
            results.append((g["case_id"], g))
        except:
            continue
    if tier_filter and skipped: print(f"  [tier filter] kept {len(results)}, skipped {skipped}")
    return results


def load_labels_hf(name="Exploration-Lab/IL-TUR", config="cjpe", split="single_train"):
    from datasets import load_dataset
    return {str(ex["id"]): int(ex["label"]) for ex in load_dataset(name, config)[split]}


def load_ontology(path):
    if not path or not path.exists(): return {}
    data = load_json(path)
    return data.get("concepts", {}) if isinstance(data, dict) and "concepts" in data else (
        data if isinstance(data, dict) else {})


# =============================================================================
# FIX #1: FUZZY CONCEPT MATCHING
# =============================================================================
# The core problem: 99.3% of concept_ids are UNLISTED_ free-text strings.
# Two cases about the same statute get different IDs like:
#   UNLISTED_INDIAN_PENAL_CODE_S302_MURDER
#   UNLISTED_IPC_SEC_302_PUNISHMENT_FOR_MURDER
# Exact match sees these as completely different. Token-Jaccard catches the overlap.

# Common abbreviations in Indian legal concept IDs
_ABBREV_MAP = {
    "ipc": "indian_penal_code",
    "crpc": "code_criminal_procedure",
    "cpc": "code_civil_procedure",
    "coa": "constitution_india",
    "con": "constitution",
    "const": "constitution",
    "art": "article",
    "sec": "section",
    "s": "section",
    "subs": "subsection",
    "cl": "clause",
    "r": "rule",
    "o": "order",
    "sch": "schedule",
    "para": "paragraph",
    "amdt": "amendment",
}

# Legal noise words to discard
_LEGAL_STOP = {"act", "the", "of", "for", "and", "in", "to", "under", "with", "by", "a", "an", "on", "or"}

# Section number pattern - extract just the number for matching
_SECTION_RE = re.compile(r'^(?:s|sec|section|art|article|rule|order|cl|clause|para|subs|subsection)(\d+[a-z]?)$', re.I)


def _normalize_concept_tokens(concept_id: str) -> Set[str]:
    """Convert a concept_id into a normalized token set for fuzzy matching."""
    raw = concept_id
    if raw.startswith("UNLISTED_"):
        raw = raw[len("UNLISTED_"):]

    # Split on underscores and other separators
    tokens = re.split(r'[_\-\s/]+', raw.lower())
    normalized = set()

    for tok in tokens:
        if not tok or tok in _LEGAL_STOP:
            continue

        # Check if it's a section number reference
        m = _SECTION_RE.match(tok)
        if m:
            normalized.add(f"sec_{m.group(1)}")  # canonical: sec_302
            continue

        # Expand abbreviations
        expanded = _ABBREV_MAP.get(tok)
        if expanded is not None:
            for sub in expanded.split("_"):
                if sub and sub not in _LEGAL_STOP:
                    normalized.add(sub)
            continue

        # Standalone numbers - in legal concept IDs these are almost always
        # section/article numbers, so emit sec_N to match combined forms like S302
        if tok.isdigit():
            normalized.add(f"sec_{tok}")
            continue

        normalized.add(tok)

    return normalized


def _fuzzy_token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Token-level Jaccard between two normalized concept token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    shared = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(shared) / len(union) if union else 0.0


class FuzzyConceptIndex:
    """Pre-computes token sets for all concepts across all cases for fast fuzzy matching."""

    def __init__(self, profiles: list, threshold: float = 0.25):
        """
        Args:
            profiles: list of ConceptProfile
            threshold: minimum token-Jaccard to count as a match (lowered from 0.4)
        """
        self.threshold = threshold

        # For each case, store {concept_id: token_set}
        self._case_tokens: Dict[str, Dict[str, Set[str]]] = {}
        for p in profiles:
            ct = {}
            for cid in p.all_concepts:
                ct[cid] = _normalize_concept_tokens(cid)
            self._case_tokens[p.case_id] = ct

    def fuzzy_concept_similarity(self, prof_a: 'ConceptProfile', prof_b: 'ConceptProfile') -> float:
        """Compute soft concept overlap between two cases using fuzzy matching.

        For each concept in A, find the best-matching concept in B (by token Jaccard).
        If the best match exceeds threshold, count it as a (partial) match weighted by
        relevance weights from both sides.

        Returns a score in [0, 1] representing fuzzy concept overlap.
        """
        tokens_a = self._case_tokens.get(prof_a.case_id, {})
        tokens_b = self._case_tokens.get(prof_b.case_id, {})

        if not tokens_a or not tokens_b:
            return 0.0

        # Compute best-match similarity for each concept in A against all in B
        match_scores = []
        for cid_a, toks_a in tokens_a.items():
            w_a = prof_a.all_concepts.get(cid_a, 0.5)
            best_sim = 0.0
            best_w_b = 0.0
            for cid_b, toks_b in tokens_b.items():
                sim = _fuzzy_token_jaccard(toks_a, toks_b)
                if sim > best_sim:
                    best_sim = sim
                    best_w_b = prof_b.all_concepts.get(cid_b, 0.5)

            if best_sim >= self.threshold:
                # Weight by min of both relevance weights * match quality
                match_scores.append(best_sim * min(w_a, best_w_b))

        # Normalize by total possible weight
        total_possible = sum(prof_a.all_concepts.values())
        if total_possible == 0:
            return 0.0

        return min(sum(match_scores) / total_possible, 1.0)


# =============================================================================
# STATUTE FAMILY EXTRACTION (unchanged from v3)
# =============================================================================

def extract_statute_family(concept_id):
    if not concept_id.startswith("UNLISTED_"): return None
    raw = concept_id[len("UNLISTED_"):]
    m = re.match(r"^(.+?)_(?:S\d|ART\d|SCHEDULE|SUBS\d|RULE\d|ORDER\d|SEC\d|CL\d|PARA\d|ITEM\d)", raw, re.IGNORECASE)
    return "FAMILY_" + (m.group(1).upper() if m else raw.upper())


def build_family_profile(concepts):
    RW = {"central": 3.0, "supporting": 1.5, "mentioned": 0.5, "obiter": 0.25}
    families = {}
    for c in concepts:
        if not isinstance(c, dict): continue
        cid = c.get("concept_id", "")
        family = extract_statute_family(cid)
        if family is None: continue
        w = RW.get(c.get("relevance", "mentioned"), 0.5)
        families[family] = max(families.get(family, 0), w)
    return families


# =============================================================================
# FIX #2: FULL CASE TEXT (not just concept metadata)
# =============================================================================

def build_concept_text(concepts):
    """Original v3 concept text - just concept metadata."""
    parts = []
    for c in concepts:
        if not isinstance(c, dict): continue
        for fld in ("unlisted_label", "unlisted_description", "interpretation"):
            val = c.get(fld)
            if isinstance(val, str) and val.strip(): parts.append(val.strip())
        cid = c.get("concept_id", "")
        if cid: parts.append(cid.replace("UNLISTED_", "").replace("_", " "))
    return " ".join(parts)


def build_full_case_text(graph):
    """FIX: Use ALL available text from the graph - facts, arguments, holdings, issues.

    This gives TF-IDF a much richer signal to work with compared to just concept metadata.
    """
    parts = []

    # Facts - the primary content
    for f in (graph.get("facts") or []):
        if isinstance(f, dict):
            t = f.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())

    # Issues - what the court is deciding
    for iss in (graph.get("issues") or []):
        if isinstance(iss, dict):
            t = iss.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())

    # Arguments - claims made by parties
    for a in (graph.get("arguments") or []):
        if isinstance(a, dict):
            t = a.get("claim")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
            # Also include court reasoning if present
            cr = a.get("court_reasoning")
            if isinstance(cr, str) and cr.strip():
                parts.append(cr.strip())

    # Holdings - court determinations
    for h in (graph.get("holdings") or []):
        if isinstance(h, dict):
            t = h.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
            rs = h.get("reasoning_summary")
            if isinstance(rs, str) and rs.strip():
                parts.append(rs.strip())

    # Concepts - interpretation and labels
    for c in (graph.get("concepts") or []):
        if isinstance(c, dict):
            for fld in ("unlisted_label", "unlisted_description", "interpretation"):
                val = c.get(fld)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())

    # Precedents - cited propositions
    for p in (graph.get("precedents") or []):
        if isinstance(p, dict):
            prop = p.get("cited_proposition")
            if isinstance(prop, str) and prop.strip():
                parts.append(prop.strip())

    return " ".join(parts)


# =============================================================================
# CONCEPT PROFILE
# =============================================================================

RELEVANCE_WEIGHT = {"central": 3.0, "supporting": 1.5, "mentioned": 0.5, "obiter": 0.25}


@dataclass
class ConceptProfile:
    case_id: str
    ontology_concepts: Dict[str, float] = field(default_factory=dict)
    statute_families: Dict[str, float] = field(default_factory=dict)
    concept_text: str = ""         # concept-only text (v3 compat)
    full_case_text: str = ""       # FIX: all graph text
    all_concepts: Dict[str, float] = field(default_factory=dict)
    schemes: Counter = field(default_factory=Counter)
    precedent_citations: Set[str] = field(default_factory=set)
    fact_types: Counter = field(default_factory=Counter)
    edge_types: Counter = field(default_factory=Counter)
    outcome: Optional[str] = None
    label: Optional[int] = None


def extract_concept_profile(graph, label=None):
    p = ConceptProfile(case_id=graph.get("case_id", "unknown"))
    p.label = label
    concepts_raw = graph.get("concepts", []) or []
    for c in concepts_raw:
        if not isinstance(c, dict): continue
        cid = c.get("concept_id", "")
        if not isinstance(cid, str): continue
        w = RELEVANCE_WEIGHT.get(c.get("relevance", "mentioned"), 0.5)
        p.all_concepts[cid] = max(p.all_concepts.get(cid, 0), w)
        if not cid.startswith("UNLISTED_"):
            p.ontology_concepts[cid] = max(p.ontology_concepts.get(cid, 0), w)
    p.statute_families = build_family_profile(concepts_raw)
    p.concept_text = build_concept_text(concepts_raw)
    p.full_case_text = build_full_case_text(graph)
    for a in graph.get("arguments", []) or []:
        if not isinstance(a, dict): continue
        for s in (a.get("schemes", []) or []):
            if isinstance(s, str): p.schemes[s] += 1
    for pr in graph.get("precedents", []) or []:
        if not isinstance(pr, dict): continue
        cit = pr.get("citation")
        if isinstance(cit, str): p.precedent_citations.add(cit)
    for f in graph.get("facts", []) or []:
        if isinstance(f, dict) and isinstance(f.get("fact_type"), str): p.fact_types[f["fact_type"]] += 1
    for e in graph.get("edges", []) or []:
        if isinstance(e, dict) and isinstance(e.get("relation"), str): p.edge_types[e["relation"]] += 1
    outcome = graph.get("outcome")
    if isinstance(outcome, dict): p.outcome = outcome.get("disposition")
    return p


# =============================================================================
# IDF
# =============================================================================

def compute_idf_multi(corpus):
    N = len(corpus)
    if N == 0: return {}, {}
    onto_df, family_df = Counter(), Counter()
    for p in corpus:
        for c in p.ontology_concepts: onto_df[c] += 1
        for f in p.statute_families: family_df[f] += 1
    return ({c: math.log(N / (1 + d)) for c, d in onto_df.items()},
            {f: math.log(N / (1 + d)) for f, d in family_df.items()})


# =============================================================================
# TEXT TF-IDF SIMILARITY (now with full case text option)
# =============================================================================

class CaseTextSimilarity:
    """TF-IDF similarity using full case text from graphs."""

    def __init__(self, profiles, use_full_text=True, max_features=5000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.idx = {p.case_id: i for i, p in enumerate(profiles)}

        if use_full_text:
            texts = [p.full_case_text if p.full_case_text.strip() else "empty" for p in profiles]
            label = "full-case-text"
        else:
            texts = [p.concept_text if p.concept_text.strip() else "empty" for p in profiles]
            label = "concept-text"

        X = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,  # log(1+tf) - helps with long documents
        ).fit_transform(texts)
        self.sim = cosine_similarity(X)
        print(f"  [{label} TF-IDF] {X.shape[0]} cases x {X.shape[1]} features")

    def similarity(self, a, b):
        i, j = self.idx.get(a), self.idx.get(b)
        return float(self.sim[i, j]) if i is not None and j is not None else 0.0


# Keep backward compat alias
ConceptTextSimilarity = CaseTextSimilarity


# =============================================================================
# SIMILARITY METRICS
# =============================================================================

def _weighted_jaccard(dict_a, dict_b, idf=None):
    set_a, set_b = set(dict_a), set(dict_b)
    shared, union = set_a & set_b, set_a | set_b
    if not union: return 0.0
    if idf and shared:
        n = sum(idf.get(c, 1.0) * min(dict_a[c], dict_b[c]) for c in shared)
        d = sum(idf.get(c, 1.0) * max(dict_a.get(c, 0), dict_b.get(c, 0)) for c in union)
        return n / d if d > 0 else 0.0
    return len(shared) / len(union)


def _cosine(counter_a, counter_b):
    keys = set(counter_a) | set(counter_b)
    if not keys: return 0.0
    dot = sum(counter_a.get(k, 0) * counter_b.get(k, 0) for k in keys)
    ma = math.sqrt(sum(v ** 2 for v in counter_a.values())) or 1
    mb = math.sqrt(sum(v ** 2 for v in counter_b.values())) or 1
    return dot / (ma * mb)


def precedent_overlap(a, b):
    if not a.precedent_citations and not b.precedent_citations: return 0.0
    s = a.precedent_citations & b.precedent_citations
    u = a.precedent_citations | b.precedent_citations
    return len(s) / len(u) if u else 0.0


def combined_similarity(a, b,
                        # v4.1 weights - DATA-DRIVEN from diagnostics
                        # Only components with discriminative variance get weight.
                        # scheme/fact_type/edge_type are near-constant (0.64/0.74/0.93 mean)
                        #   = pure noise that drowns signal. ZERO weight.
                        # precedent overlap is literally 0 across all pairs. ZERO weight.
                        w_text=0.55,      # full case text TF-IDF (best discriminative range)
                        w_fuzzy=0.35,     # fuzzy concept matching (sparse but discriminative)
                        w_family=0.10,    # statute families (very sparse but occasionally helpful)
                        w_precedent=0.0,  # DEAD: zero overlap across all sampled pairs
                        w_scheme=0.0,     # NOISE: mean=0.64, near-constant across all pairs
                        w_edge_type=0.0,  # NOISE: mean=0.93, near-constant across all pairs
                        w_fact_type=0.0,  # NOISE: mean=0.74, near-constant across all pairs
                        w_ontology=0.0,   # only 90 concepts total, 93/452 cases have zero
                        # Injected dependencies
                        onto_idf=None, family_idf=None, text_sim=None,
                        fuzzy_index=None,
                        **_kw):

    sim = 0.0

    # FUZZY concept matching (NEW - the big fix)
    if fuzzy_index and w_fuzzy > 0:
        sim += w_fuzzy * fuzzy_index.fuzzy_concept_similarity(a, b)

    # Exact ontology match (legacy, low weight)
    if w_ontology > 0:
        sim += w_ontology * _weighted_jaccard(a.ontology_concepts, b.ontology_concepts, onto_idf)

    # Statute family overlap
    if w_family > 0:
        sim += w_family * _weighted_jaccard(a.statute_families, b.statute_families, family_idf)

    # Text similarity (now using full case text)
    if text_sim and w_text > 0:
        sim += w_text * text_sim.similarity(a.case_id, b.case_id)
    elif w_text > 0:
        # Fallback: use family overlap again (same as v3)
        sim += w_text * _weighted_jaccard(a.statute_families, b.statute_families, family_idf)

    # Precedent citation overlap
    if w_precedent > 0:
        sim += w_precedent * precedent_overlap(a, b)

    # Structural features
    if w_scheme > 0:
        sim += w_scheme * _cosine(a.schemes, b.schemes)
    if w_fact_type > 0:
        sim += w_fact_type * _cosine(a.fact_types, b.fact_types)
    if w_edge_type > 0:
        sim += w_edge_type * _cosine(a.edge_types, b.edge_types)

    return sim


# =============================================================================
# RETRIEVAL
# =============================================================================

@dataclass
class RetrievalResult:
    case_id: str
    similarity: float
    label: Optional[int]
    outcome: Optional[str]
    shared_concepts: List[str]
    shared_families: List[str]
    shared_precedents: List[str]
    reasoning_chains: List[Dict]


def retrieve_similar(query, corpus, k=10, sim_kwargs=None):
    sim_kwargs = sim_kwargs or {}
    scored = [(combined_similarity(query, p, **sim_kwargs), p, g)
              for p, g in corpus if p.case_id != query.case_id]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, p, g in scored[:k]:
        results.append(RetrievalResult(
            case_id=p.case_id, similarity=sim, label=p.label, outcome=p.outcome,
            shared_concepts=sorted(set(query.ontology_concepts) & set(p.ontology_concepts)),
            shared_families=sorted(set(query.statute_families) & set(p.statute_families)),
            shared_precedents=sorted(query.precedent_citations & p.precedent_citations),
            reasoning_chains=[c for c in (g.get("reasoning_chains") or []) if isinstance(c, dict)],
        ))
    return results


# =============================================================================
# FIX #3: PREDICTION WITH THRESHOLD SWEEP
# =============================================================================

def predict_from_neighbors(neighbors, method="weighted_vote", family_idf=None, threshold=0.5):
    """Predict outcome from k nearest neighbors. Returns (pred, prob, meta)."""
    if not neighbors: return 0, 0.5, {"method": method}

    if method == "majority_vote":
        votes = [n.label for n in neighbors if n.label is not None]
        if not votes: return 0, 0.5, {"method": method}
        prob = sum(votes) / len(votes)
        return (1 if prob >= threshold else 0), prob, {"method": method}

    elif method == "weighted_vote":
        wa, wr = 0.0, 0.0
        for n in neighbors:
            if n.label is None: continue
            w = max(n.similarity, 0.0)
            if n.label == 1:
                wa += w
            else:
                wr += w
        t = wa + wr
        if t == 0: return 0, 0.5, {"method": method}
        prob = wa / t
        return (1 if prob >= threshold else 0), prob, {"method": method}

    elif method == "distance_decay_vote":
        wa, wr = 0.0, 0.0
        for rank, n in enumerate(neighbors):
            if n.label is None: continue
            w = max(n.similarity, 0.0) * math.exp(-0.3 * rank)
            if n.label == 1:
                wa += w
            else:
                wr += w
        t = wa + wr
        if t == 0: return 0, 0.5, {"method": method}
        prob = wa / t
        return (1 if prob >= threshold else 0), prob, {"method": method}

    elif method == "family_cluster_aggregate":
        fs = defaultdict(lambda: {"a": 0.0, "r": 0.0})
        for n in neighbors:
            if n.label is None: continue
            for fid in n.shared_families:
                fw = family_idf.get(fid, 1.0) if family_idf else 1.0
                if n.label == 1:
                    fs[fid]["a"] += fw
                else:
                    fs[fid]["r"] += fw
        ta = sum(s["a"] for s in fs.values())
        tr = sum(s["r"] for s in fs.values())
        t = ta + tr
        prob = ta / t if t > 0 else 0.5
        return (1 if prob >= threshold else 0), prob, {"method": method}

    raise ValueError(f"Unknown: {method}")


def _compute_metrics_at_threshold(trues, probs, threshold):
    """Compute accuracy, F1, P, R at a given threshold."""
    preds = (probs >= threshold).astype(int)
    acc = float(np.mean(preds == trues))
    tp = int(np.sum((preds == 1) & (trues == 1)))
    fp = int(np.sum((preds == 1) & (trues == 0)))
    fn = int(np.sum((preds == 0) & (trues == 1)))
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
    return {"accuracy": acc, "f1": f1, "precision": pr, "recall": rc}


def find_optimal_threshold(trues, probs, metric="f1"):
    """Sweep thresholds to find optimal for given metric."""
    best_t, best_val = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.01):
        m = _compute_metrics_at_threshold(trues, probs, t)
        if m[metric] > best_val:
            best_val = m[metric]
            best_t = float(t)
    return best_t, best_val


# =============================================================================
# TEXT BASELINE (improved)
# =============================================================================

def build_text_baseline(graphs, labels, k=10):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    cids, texts, ys = [], [], []
    for cid, g in graphs:
        if cid not in labels: continue
        ft = [f["text"] for f in (g.get("facts") or []) if isinstance(f, dict) and isinstance(f.get("text"), str)]
        t = " ".join(ft)
        if not t.strip(): continue
        cids.append(cid)
        texts.append(t)
        ys.append(labels[cid])
    if len(texts) < 10: return {"error": "not enough texts"}

    X = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True).fit_transform(texts)
    sm = cosine_similarity(X)
    ys = np.array(ys)

    # Collect probs first, then find optimal threshold
    all_probs = []
    for i in range(len(cids)):
        s = sm[i].copy()
        s[i] = -1
        tk = np.argsort(s)[-k:][::-1]
        wp = sum(s[j] for j in tk if ys[j] == 1)
        wn = sum(s[j] for j in tk if ys[j] == 0)
        t = wp + wn
        prob = wp / t if t > 0 else 0.5
        all_probs.append(prob)
    probs = np.array(all_probs)

    # Fixed threshold (0.5) results
    preds_50 = (probs >= 0.5).astype(int)

    # Optimal threshold results
    opt_t, _ = find_optimal_threshold(ys, probs)
    preds_opt = (probs >= opt_t).astype(int)

    def _metrics(preds_arr):
        tp = int(np.sum((preds_arr == 1) & (ys == 1)))
        fp = int(np.sum((preds_arr == 1) & (ys == 0)))
        fn = int(np.sum((preds_arr == 0) & (ys == 1)))
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        acc = float(np.mean(preds_arr == ys))
        try:
            auc = roc_auc_score(ys, probs)
        except:
            auc = float("nan")
        return {"accuracy": round(acc, 4), "f1": round(f1, 4),
                "precision": round(pr, 4), "recall": round(rc, 4),
                "auc": round(auc, 4) if not math.isnan(auc) else "N/A", "n": len(ys)}

    return {
        "fixed_50": _metrics(preds_50),
        "optimal": _metrics(preds_opt),
        "optimal_threshold": round(opt_t, 2),
    }


# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap_ci(trues, preds, probs, n_boot=1000):
    rng = np.random.RandomState(42)
    n = len(trues)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, True)
        t, p, pr = trues[idx], preds[idx], probs[idx]
        accs.append(float(np.mean(t == p)))
        tp = np.sum((p == 1) & (t == 1))
        fp = np.sum((p == 1) & (t == 0))
        fn = np.sum((p == 0) & (t == 1))
        pc = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pc * rc / (pc + rc) if pc + rc else 0)
        if len(set(t)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                aucs.append(roc_auc_score(t, pr))
            except:
                pass
    return {"accuracy_ci": (round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)),
            "f1_ci": (round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)),
            "auc_ci": (round(float(np.percentile(aucs, 2.5)), 4),
                       round(float(np.percentile(aucs, 97.5)), 4)) if aucs else ("N/A", "N/A")}


# =============================================================================
# COUNTERFACTUAL (unchanged from v3)
# =============================================================================

def run_counterfactual_sensitivity(graphs, labels, ontology):
    req_breaks, rand_breaks = [], []
    for cid, g in graphs:
        if cid not in labels: continue
        js_list = g.get("justification_sets") or []
        edges = g.get("edges") or []
        holdings = g.get("holdings") or []
        concepts = g.get("concepts") or []
        if not js_list or not holdings or not concepts: continue
        and_members = set()
        for js in js_list:
            if not isinstance(js, dict) or js.get("logic") != "and": continue
            jid = js.get("id")
            for e in edges:
                if isinstance(e, dict) and jid in (e.get("support_group_ids") or []):
                    and_members.add(e.get("source"))
        concept_ids = {c["id"] for c in concepts if isinstance(c, dict) and isinstance(c.get("id"), str)}
        for concept_id in concept_ids:
            broken = 0
            for h in holdings:
                if not isinstance(h, dict): continue
                hid = h.get("id")
                js_h = [j for j in js_list if isinstance(j, dict) and j.get("target_id") == hid]
                if not js_h:
                    if any(isinstance(e, dict) and e.get("source") == concept_id and e.get("target") == hid for e in
                           edges): broken += 1
                    continue
                all_b = True
                for j in js_h:
                    jid = j.get("id")
                    members = [e.get("source") for e in edges if
                               isinstance(e, dict) and jid in (e.get("support_group_ids") or [])]
                    if concept_id not in members: all_b = False; break
                    if j.get("logic") == "or" and any(m != concept_id for m in members): all_b = False; break
                if all_b and js_h: broken += 1
            entry = {"case_id": cid, "concept_id": concept_id, "holdings_broken": broken,
                     "total_holdings": len(holdings)}
            (req_breaks if concept_id in and_members else rand_breaks).append(entry)
    rr = [e["holdings_broken"] / max(e["total_holdings"], 1) for e in req_breaks]
    rn = [e["holdings_broken"] / max(e["total_holdings"], 1) for e in rand_breaks]
    return {"n_required": len(req_breaks), "n_random": len(rand_breaks),
            "req_avg": float(np.mean(rr)) if rr else 0, "rand_avg": float(np.mean(rn)) if rn else 0,
            "req_any": sum(1 for r in rr if r > 0) / max(len(rr), 1),
            "rand_any": sum(1 for r in rn if r > 0) / max(len(rn), 1)}


# =============================================================================
# FIX #4: SIMILARITY DIAGNOSTICS
# =============================================================================

def run_diagnostics(profiles, fuzzy_index, text_sim, onto_idf, family_idf):
    """Print diagnostic information about what's driving similarity."""
    print("\n" + "=" * 70)
    print("SIMILARITY DIAGNOSTICS")
    print("=" * 70)

    # Concept coverage stats
    n = len(profiles)
    onto_counts = [len(p.ontology_concepts) for p in profiles]
    unlisted_counts = [sum(1 for c in p.all_concepts if c.startswith("UNLISTED_")) for p in profiles]
    family_counts = [len(p.statute_families) for p in profiles]
    precedent_counts = [len(p.precedent_citations) for p in profiles]
    text_lens = [len(p.full_case_text.split()) for p in profiles]

    print(f"\n  Ontology concepts/case:  mean={np.mean(onto_counts):.1f}, median={np.median(onto_counts):.0f}, "
          f"max={max(onto_counts)}, zero={sum(1 for x in onto_counts if x == 0)}/{n}")
    print(f"  UNLISTED concepts/case:  mean={np.mean(unlisted_counts):.1f}, median={np.median(unlisted_counts):.0f}")
    print(f"  Statute families/case:   mean={np.mean(family_counts):.1f}, median={np.median(family_counts):.0f}")
    print(f"  Precedent cites/case:    mean={np.mean(precedent_counts):.1f}, median={np.median(precedent_counts):.0f}")
    print(f"  Full text words/case:    mean={np.mean(text_lens):.0f}, median={np.median(text_lens):.0f}")

    # Sample pairwise similarities to understand distributions
    rng = np.random.RandomState(42)
    sample_size = min(200, n * (n - 1) // 2)
    pairs = set()
    while len(pairs) < sample_size:
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i != j: pairs.add((min(i, j), max(i, j)))

    sim_components = {"fuzzy": [], "family": [], "text": [], "precedent": [], "scheme": [],
                      "fact_type": [], "edge_type": [], "total": []}
    for i, j in pairs:
        a, b = profiles[i], profiles[j]
        if fuzzy_index:
            sim_components["fuzzy"].append(fuzzy_index.fuzzy_concept_similarity(a, b))
        sim_components["family"].append(_weighted_jaccard(a.statute_families, b.statute_families, family_idf))
        if text_sim:
            sim_components["text"].append(text_sim.similarity(a.case_id, b.case_id))
        sim_components["precedent"].append(precedent_overlap(a, b))
        sim_components["scheme"].append(_cosine(a.schemes, b.schemes))
        sim_components["fact_type"].append(_cosine(a.fact_types, b.fact_types))
        sim_components["edge_type"].append(_cosine(a.edge_types, b.edge_types))
        sim_components["total"].append(combined_similarity(
            a, b, onto_idf=onto_idf, family_idf=family_idf, text_sim=text_sim, fuzzy_index=fuzzy_index))

    print(f"\n  Pairwise similarity distributions (n={sample_size} random pairs):")
    print(f"  {'Component':<15} {'Mean':>8} {'Std':>8} {'>0':>6} {'Max':>8}")
    print(f"  {'-'*50}")
    for name in ["fuzzy", "family", "text", "precedent", "scheme", "fact_type", "edge_type", "total"]:
        vals = sim_components[name]
        if not vals: continue
        v = np.array(vals)
        print(f"  {name:<15} {np.mean(v):>8.4f} {np.std(v):>8.4f} {np.sum(v > 0):>5d} {np.max(v):>8.4f}")

    # Check if accepted vs rejected cases are separable
    acc_profs = [p for p in profiles if p.label == 1]
    rej_profs = [p for p in profiles if p.label == 0]
    print(f"\n  Label balance: {len(acc_profs)} accepted, {len(rej_profs)} rejected")


# =============================================================================
# EVALUATION (with threshold sweep)
# =============================================================================

def run_leave_one_out(graphs, labels, k=10, methods=None, use_text_sim=True, sim_override=None,
                      do_bootstrap=False, threshold=0.5, use_fuzzy=True, use_full_text=True,
                      show_diagnostics=False):
    if methods is None:
        methods = ["majority_vote", "weighted_vote", "distance_decay_vote", "family_cluster_aggregate"]

    corpus = [(extract_concept_profile(g, labels[c]), g) for c, g in graphs if c in labels]
    n = len(corpus)
    profiles = [p for p, _ in corpus]
    onto_idf, family_idf = compute_idf_multi(profiles)

    all_fam = set()
    all_onto = set()
    for p in profiles:
        all_fam.update(p.statute_families)
        all_onto.update(p.ontology_concepts)
    print(f"Evaluating {n} cases (LOO, k={k})")
    print(f"Labels: {sum(1 for p in profiles if p.label == 1)} accepted, "
          f"{sum(1 for p in profiles if p.label == 0)} rejected")
    print(f"Ontology: {len(all_onto)} | Families: {len(all_fam)}")

    # Build fuzzy concept index
    fuzzy_index = None
    if use_fuzzy:
        fuzzy_index = FuzzyConceptIndex(profiles)
        print(f"  [fuzzy concept index] built for {n} cases")

    # Build text similarity
    text_sim = None
    if use_text_sim:
        try:
            text_sim = CaseTextSimilarity(profiles, use_full_text=use_full_text)
        except ImportError:
            print("  [warn] no sklearn for text sim")

    if show_diagnostics:
        run_diagnostics(profiles, fuzzy_index, text_sim, onto_idf, family_idf)

    sim_kw = {"onto_idf": onto_idf, "family_idf": family_idf, "text_sim": text_sim, "fuzzy_index": fuzzy_index}
    if sim_override:
        sim_kw.update(sim_override)

    # Collect probabilities for all methods
    results = {m: {"correct": 0, "total": 0, "probs": [], "trues": []} for m in methods}
    for i, (qp, qg) in enumerate(corpus):
        rem = [(p, g) for j, (p, g) in enumerate(corpus) if j != i]
        nbrs = retrieve_similar(qp, rem, k, sim_kw)
        for m in methods:
            pred, prob, _ = predict_from_neighbors(nbrs, m, family_idf, threshold)
            results[m]["correct"] += int(pred == qp.label)
            results[m]["total"] += 1
            results[m]["probs"].append(prob)
            results[m]["trues"].append(qp.label)
        if (i + 1) % 50 == 0:
            bm = methods[1] if len(methods) > 1 else methods[0]
            print(f"  [{i + 1}/{n}] {bm} acc: {results[bm]['correct'] / results[bm]['total']:.3f}")

    final = {}
    for m in methods:
        r = results[m]
        trues, probs = np.array(r["trues"]), np.array(r["probs"])

        # Fixed threshold metrics
        acc_fixed = r["correct"] / r["total"] if r["total"] else 0
        preds_fixed = (probs >= threshold).astype(int)
        tp = int(np.sum((preds_fixed == 1) & (trues == 1)))
        fp = int(np.sum((preds_fixed == 1) & (trues == 0)))
        fn = int(np.sum((preds_fixed == 0) & (trues == 1)))
        pr_f = tp / (tp + fp) if tp + fp else 0
        rc_f = tp / (tp + fn) if tp + fn else 0
        f1_f = 2 * pr_f * rc_f / (pr_f + rc_f) if pr_f + rc_f else 0

        # Optimal threshold metrics
        opt_t, _ = find_optimal_threshold(trues, probs, metric="f1")
        m_opt = _compute_metrics_at_threshold(trues, probs, opt_t)

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(trues, probs)
        except:
            auc = float("nan")

        entry = {
            # Fixed threshold (0.5) results
            "accuracy": round(acc_fixed, 4),
            "f1": round(f1_f, 4),
            "precision": round(pr_f, 4),
            "recall": round(rc_f, 4),
            # Optimal threshold results
            "opt_threshold": round(opt_t, 2),
            "opt_accuracy": round(m_opt["accuracy"], 4),
            "opt_f1": round(m_opt["f1"], 4),
            "opt_precision": round(m_opt["precision"], 4),
            "opt_recall": round(m_opt["recall"], 4),
            # AUC (threshold-independent)
            "auc": round(auc, 4) if not math.isnan(auc) else "N/A",
            "n": r["total"],
        }
        if do_bootstrap:
            entry.update(bootstrap_ci(trues, preds_fixed, probs))
        final[m] = entry

    return final


# =============================================================================
# ABLATION CONFIGS (updated for v4)
# =============================================================================

ABLATION_CONFIGS = {
    # --- Isolate each DISCRIMINATIVE signal ---
    "text_only":        {"w_fuzzy": 0, "w_family": 0, "w_text": 1.0, "w_precedent": 0, "w_scheme": 0, "w_fact_type": 0, "w_edge_type": 0, "w_ontology": 0},
    "fuzzy_only":       {"w_fuzzy": 1.0, "w_family": 0, "w_text": 0, "w_precedent": 0, "w_scheme": 0, "w_fact_type": 0, "w_edge_type": 0, "w_ontology": 0},
    "family_only":      {"w_fuzzy": 0, "w_family": 1.0, "w_text": 0, "w_precedent": 0, "w_scheme": 0, "w_fact_type": 0, "w_edge_type": 0, "w_ontology": 0},
    # --- Show that noise components hurt ---
    "noise_only":       {"w_fuzzy": 0, "w_family": 0, "w_text": 0, "w_precedent": 0, "w_scheme": 0.4, "w_fact_type": 0.2, "w_edge_type": 0.4, "w_ontology": 0},
    # --- Additive ablation (signal only) ---
    "text+fuzzy":       {"w_fuzzy": 0.40, "w_family": 0, "w_text": 0.60, "w_precedent": 0, "w_scheme": 0, "w_fact_type": 0, "w_edge_type": 0, "w_ontology": 0},
    "text+fuzzy+fam":   {"w_fuzzy": 0.35, "w_family": 0.10, "w_text": 0.55, "w_precedent": 0, "w_scheme": 0, "w_fact_type": 0, "w_edge_type": 0, "w_ontology": 0},
    # --- Show adding noise back hurts ---
    "+noise_components": {"w_fuzzy": 0.25, "w_family": 0.05, "w_text": 0.40, "w_precedent": 0, "w_scheme": 0.12, "w_fact_type": 0.06, "w_edge_type": 0.12, "w_ontology": 0},
    # --- Full default ---
    "full_default":     {},
}


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_baselines(args):
    graphs = iter_graphs(Path(args.graph_dir), set(args.tier.lower().split(",")) if args.tier else None)
    labels = load_labels_hf()
    print("Running all baselines...\n")
    labeled = [(c, g) for c, g in graphs if c in labels]
    ar = sum(labels[c] for c, _ in labeled) / len(labeled)
    maj_acc = ar if ar >= 0.5 else 1 - ar

    # TF-IDF baseline (fact text)
    print("TF-IDF baseline (fact text)...")
    tr = build_text_baseline(graphs, labels, args.k)

    # Graph v4.1 (fuzzy + full text, noise-free weights)
    print("\nGraph v4.1 (signal-only weights)...")
    v4 = run_leave_one_out(graphs, labels, args.k, use_text_sim=True, use_fuzzy=True, use_full_text=True,
                           do_bootstrap=getattr(args, "bootstrap", False), show_diagnostics=True)

    # Print results
    print(f"\n{'=' * 100}")
    print(f"BASELINES COMPARISON (SIGIR) - v4.1 signal-only weights")
    print(f"{'=' * 100}")

    hdr = f"{'Method':<42} {'Acc@.5':>7} {'F1@.5':>7} {'AUC':>7} | {'Acc*':>7} {'F1*':>7} {'P*':>7} {'R*':>7} {'t*':>5}"
    print(hdr)
    print("-" * 100)

    # Majority class
    print(f"{'Majority class':<42} {maj_acc:>7.4f} {'---':>7} {'---':>7} | {'---':>7} {'---':>7} {'---':>7} {'---':>7} {'---':>5}")

    # TF-IDF baseline
    if isinstance(tr, dict) and "fixed_50" in tr:
        r50 = tr["fixed_50"]
        ropt = tr["optimal"]
        a_s = f"{r50['auc']:>7.4f}" if isinstance(r50.get('auc'), float) else f"{'N/A':>7}"
        print(f"{'TF-IDF fact text kNN':<42} {r50['accuracy']:>7.4f} {r50['f1']:>7.4f} {a_s} | "
              f"{ropt['accuracy']:>7.4f} {ropt['f1']:>7.4f} {ropt['precision']:>7.4f} {ropt['recall']:>7.4f} "
              f"{tr['optimal_threshold']:>5.2f}")
    elif isinstance(tr, dict) and "accuracy" in tr:
        a_s = f"{tr['auc']:>7.4f}" if isinstance(tr.get('auc'), float) else f"{'N/A':>7}"
        print(f"{'TF-IDF fact text kNN':<42} {tr['accuracy']:>7.4f} {tr.get('f1', 0):>7.4f} {a_s} | "
              f"{'---':>7} {'---':>7} {'---':>7} {'---':>7} {'---':>5}")

    # Graph methods
    for m, r in v4.items():
        a_s = f"{r['auc']:>7.4f}" if isinstance(r['auc'], float) else f"{'N/A':>7}"
        print(f"{'Graph v4: ' + m:<42} {r['accuracy']:>7.4f} {r['f1']:>7.4f} {a_s} | "
              f"{r['opt_accuracy']:>7.4f} {r['opt_f1']:>7.4f} {r['opt_precision']:>7.4f} {r['opt_recall']:>7.4f} "
              f"{r['opt_threshold']:>5.2f}")

    print("-" * 100)
    print(f"n={len(labeled)}, k={args.k}")
    print(f"Acc@.5 = accuracy at threshold 0.5 | * = optimal threshold (sweeping 0.20-0.80)")
    print(f"AUC is threshold-independent and the most reliable metric here.")

    # Multi-k sweep for best method
    ks_to_try = [3, 5, 7, 10, 15]
    print(f"\n{'=' * 70}")
    print(f"K-SWEEP (weighted_vote, optimal threshold)")
    print(f"{'=' * 70}")
    print(f"{'k':>4} {'AUC':>8} {'Acc*':>8} {'F1*':>8} {'t*':>6}")
    print("-" * 38)
    for k in ks_to_try:
        r = run_leave_one_out(graphs, labels, k, ["weighted_vote"],
                              use_text_sim=True, use_fuzzy=True, use_full_text=True)
        m = r["weighted_vote"]
        a_s = f"{m['auc']:>8.4f}" if isinstance(m['auc'], float) else f"{'N/A':>8}"
        print(f"{k:>4} {a_s} {m['opt_accuracy']:>8.4f} {m['opt_f1']:>8.4f} {m['opt_threshold']:>6.2f}")


def cmd_evaluate(args):
    graphs = iter_graphs(Path(args.graph_dir), set(args.tier.lower().split(",")) if args.tier else None)
    labels = load_labels_hf()
    ks = [3, 5, 7, 10, 15, 20] if args.k_sweep else [args.k]
    for k in ks:
        if args.k_sweep: print(f"\n{'=' * 60}\n  k={k}\n{'=' * 60}")
        r = run_leave_one_out(graphs, labels, k, use_text_sim=True, use_fuzzy=True, use_full_text=True,
                              do_bootstrap=args.bootstrap)
        for m, met in r.items():
            a_s = met['auc'] if isinstance(met['auc'], float) else 'N/A'
            print(f"  {m}: acc={met['accuracy']:.4f} f1={met['f1']:.4f} auc={a_s}"
                  f"  |  opt: acc={met['opt_accuracy']:.4f} f1={met['opt_f1']:.4f} t={met['opt_threshold']:.2f}")


def cmd_explain(args):
    graphs = iter_graphs(Path(args.graph_dir))
    labels = load_labels_hf()
    ontology = load_ontology(Path(args.ontology) if args.ontology else None)
    qg = load_json(Path(args.query))
    qcid = qg.get("case_id", "?")
    corpus = [(extract_concept_profile(g, labels[c]), g) for c, g in graphs if c != qcid and c in labels]
    profiles = [p for p, _ in corpus]
    oi, fi = compute_idf_multi(profiles)
    fuzzy_index = FuzzyConceptIndex(profiles)
    try:
        ts = CaseTextSimilarity(profiles, use_full_text=True)
    except:
        ts = None
    qp = extract_concept_profile(qg)
    nbrs = retrieve_similar(qp, corpus, args.k,
                            {"onto_idf": oi, "family_idf": fi, "text_sim": ts, "fuzzy_index": fuzzy_index})
    pred, prob, _ = predict_from_neighbors(nbrs, "weighted_vote", fi)
    print(f"PREDICTION: {'ACCEPTED' if pred == 1 else 'REJECTED'} (conf={prob:.2f})")
    print(f"Families: {len(qp.statute_families)} | Ontology concepts: {len(qp.ontology_concepts)} | "
          f"All concepts: {len(qp.all_concepts)}")
    for i, n in enumerate(nbrs[:10], 1):
        lab = "acc" if n.label == 1 else "rej"
        print(f"  {i}. {n.case_id} sim={n.similarity:.3f} {lab} "
              f"({len(n.shared_families)} fam, {len(n.shared_precedents)} prec)")
    if qcid in labels: print(f"\nGROUND TRUTH: {'ACCEPTED' if labels[qcid] == 1 else 'REJECTED'}")


def cmd_counterfactual(args):
    graphs = iter_graphs(Path(args.graph_dir), set(args.tier.lower().split(",")) if args.tier else None)
    labels = load_labels_hf()
    ontology = load_ontology(Path(args.ontology) if args.ontology else None)
    r = run_counterfactual_sensitivity(graphs, labels, ontology)
    ratio = r["req_avg"] / max(r["rand_avg"], 0.001)
    print(f"\nRequired: n={r['n_required']}, avg_break={r['req_avg']:.3f}, any_break={r['req_any']:.1%}")
    print(f"Random:   n={r['n_random']}, avg_break={r['rand_avg']:.3f}, any_break={r['rand_any']:.1%}")
    print(f"Impact ratio: {ratio:.1f}x")


def cmd_ablation(args):
    graphs = iter_graphs(Path(args.graph_dir), set(args.tier.lower().split(",")) if args.tier else None)
    labels = load_labels_hf()
    print(f"Ablation ({len(ABLATION_CONFIGS)} configs)...\n")
    results = {}
    for name, w in ABLATION_CONFIGS.items():
        w = dict(w)
        print(f"  {name}...")
        use_text = w.get("w_text", 0.25) > 0 if w else True
        use_fuzzy = w.get("w_fuzzy", 0.30) > 0 if w else True
        r = run_leave_one_out(graphs, labels, args.k, ["weighted_vote"],
                              use_text_sim=use_text, sim_override=w if w else None,
                              use_fuzzy=use_fuzzy, use_full_text=True)
        results[name] = r["weighted_vote"]

    print(f"\n{'=' * 95}")
    print(f"ABLATION (Table 2)")
    print(f"{'=' * 95}")
    print(f"{'Config':<22} {'Acc@.5':>7} {'F1@.5':>7} {'AUC':>7} | {'Acc*':>7} {'F1*':>7} {'t*':>5}")
    print("-" * 70)
    for name, m in results.items():
        a_s = f"{m['auc']:>7.4f}" if isinstance(m['auc'], float) else f"{'N/A':>7}"
        print(f"{name:<22} {m['accuracy']:>7.4f} {m['f1']:>7.4f} {a_s} | "
              f"{m['opt_accuracy']:>7.4f} {m['opt_f1']:>7.4f} {m['opt_threshold']:>5.2f}")


def cmd_diagnose(args):
    """Run diagnostics only - show what's happening with concept coverage and similarity."""
    graphs = iter_graphs(Path(args.graph_dir), set(args.tier.lower().split(",")) if args.tier else None)
    labels = load_labels_hf()
    corpus = [(extract_concept_profile(g, labels.get(c)), g) for c, g in graphs if c in labels]
    profiles = [p for p, _ in corpus]
    onto_idf, family_idf = compute_idf_multi(profiles)
    fuzzy_index = FuzzyConceptIndex(profiles)
    try:
        text_sim = CaseTextSimilarity(profiles, use_full_text=True)
    except:
        text_sim = None
    run_diagnostics(profiles, fuzzy_index, text_sim, onto_idf, family_idf)

    # Show some example fuzzy matches
    print(f"\n{'=' * 70}")
    print("EXAMPLE FUZZY CONCEPT MATCHES")
    print("=" * 70)
    all_concepts = set()
    for p in profiles:
        all_concepts.update(p.all_concepts.keys())

    unlisted = sorted([c for c in all_concepts if c.startswith("UNLISTED_")])
    if len(unlisted) > 20:
        # Sample and show token-level matches
        rng = np.random.RandomState(42)
        sample = rng.choice(unlisted, min(20, len(unlisted)), replace=False)
        for cid in sample:
            tokens = _normalize_concept_tokens(cid)
            # Find best matches in the corpus
            best_sim, best_other = 0, ""
            for other in unlisted:
                if other == cid: continue
                s = _fuzzy_token_jaccard(tokens, _normalize_concept_tokens(other))
                if s > best_sim:
                    best_sim = s
                    best_other = other
            if best_sim > 0.3:
                print(f"  {cid[:60]:<60}")
                print(f"    -> {best_other[:60]:<60} (sim={best_sim:.2f})")
                print(f"    tokens: {sorted(tokens)}")
                print()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(sp):
        sp.add_argument("--graph_dir", required=True)
        sp.add_argument("--ontology", default=None)
        sp.add_argument("--tier", default=None)

    for name in ["evaluate", "explain", "counterfactual", "baselines", "ablation", "diagnose"]:
        sp = sub.add_parser(name)
        add(sp)
        sp.add_argument("--k", type=int, default=10)

    sub.choices["evaluate"].add_argument("--k_sweep", action="store_true")
    sub.choices["evaluate"].add_argument("--bootstrap", action="store_true")
    sub.choices["baselines"].add_argument("--bootstrap", action="store_true")
    sub.choices["explain"].add_argument("--query", required=True)

    args = p.parse_args()
    cmds = {
        "evaluate": cmd_evaluate,
        "explain": cmd_explain,
        "counterfactual": cmd_counterfactual,
        "baselines": cmd_baselines,
        "ablation": cmd_ablation,
        "diagnose": cmd_diagnose,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()