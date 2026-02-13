#!/usr/bin/env python3
"""
eval_retrieval_v2.py — Non-circular concept retrieval evaluation.

Default mode (--qrel_mode regex): relevance judged by regex-matching
concept identifiers in raw IL-TUR text. Ground truth is independent
of the extraction pipeline. BM25 searches raw text; ConceptSet matches
graph annotations. This evaluates whether structured extraction captures
concept relevance beyond what surface-text methods achieve.

Usage:
    # BM25 + ConceptSet only (original)
    python eval_retrieval_v2.py --graph_dir iltur_graphs

    # With dense retrieval baselines (E5-large, BGE-large, ColBERTv2)
    python eval_retrieval_v2.py --graph_dir iltur_graphs --dense_baselines

    # With domain-adapted legal baselines (Legal-BERT, InLegalBERT)
    python eval_retrieval_v2.py --graph_dir iltur_graphs --legal_baselines

    # Full evaluation: all baselines
    python eval_retrieval_v2.py --graph_dir iltur_graphs --dense_baselines --legal_baselines

    # Dense baselines without ColBERT (faster — E5 + BGE only)
    python eval_retrieval_v2.py --graph_dir iltur_graphs --dense_baselines --skip_colbert

    # Full evaluation with qrel comparison
    python eval_retrieval_v2.py --graph_dir iltur_graphs --dense_baselines --legal_baselines --compare_qrels

    # Annotation-derived qrels
    python eval_retrieval_v2.py --graph_dir iltur_graphs --qrel_mode annotation

Requirements for dense baselines:
    pip install sentence-transformers transformers torch

Requirements for legal baselines:
    pip install transformers torch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reused helpers
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


# ---------------------------------------------------------------------------
# TEXT SOURCES: Raw IL-TUR text vs graph-derived text
# ---------------------------------------------------------------------------

def load_raw_iltur_texts(case_ids: List[str], raw_text_dir: Optional[str] = None) -> Dict[str, str]:
    """Load raw case texts from IL-TUR.

    Tries local directory first (if provided), then HuggingFace.
    Returns {case_id: raw_text}.
    """
    case_id_set = set(case_ids)
    raw_texts = {}

    # Try local directory first
    if raw_text_dir:
        raw_dir = Path(raw_text_dir)
        if raw_dir.exists():
            print(f"  Loading raw texts from {raw_dir}...")
            for p in raw_dir.glob("*.txt"):
                cid = p.stem
                if cid in case_id_set:
                    raw_texts[cid] = p.read_text(encoding="utf-8")
            if raw_texts:
                print(f"  Loaded {len(raw_texts)}/{len(case_ids)} from local dir")
                return raw_texts
            print(f"  No matching texts found in {raw_dir}, falling back to HuggingFace")

    # Fall back to HuggingFace
    from datasets import load_dataset

    print("  Loading raw IL-TUR texts from HuggingFace...")
    ds = load_dataset("Exploration-Lab/IL-TUR", "cjpe")
    all_cases = ds['single_train']

    for i in range(len(all_cases)):
        case = all_cases[i]
        if case['id'] in case_id_set:
            raw_texts[case['id']] = case['text']

    print(f"  Loaded {len(raw_texts)}/{len(case_ids)} raw texts")
    return raw_texts


def build_graph_text(graph):
    """Build document text from graph fields (ORIGINAL method — for ConceptSet only)."""
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


# ---------------------------------------------------------------------------
# Concept matching (reused from original)
# ---------------------------------------------------------------------------

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


RELEVANCE_GRADE = {"central": 3, "supporting": 2, "mentioned": 1, "obiter": 1}


# ---------------------------------------------------------------------------
# QREL SOURCES
# ---------------------------------------------------------------------------

def build_annotation_qrels(graph_list, case_ids, n_queries=50):
    """Build qrels from graph annotations (original method).
    Returns (queries, qrels_binary, qrels_graded).
    """
    concept_info: Dict[str, Dict[str, Any]] = {}

    for idx, g in enumerate(graph_list):
        seen_in_case: Dict[str, int] = {}
        for c in (g.get("concepts") or []):
            if not isinstance(c, dict):
                continue
            cid = c.get("concept_id", "")
            if not isinstance(cid, str) or not cid:
                continue
            grade = RELEVANCE_GRADE.get(c.get("relevance", "mentioned"), 1)
            seen_in_case[cid] = max(seen_in_case.get(cid, 0), grade)
            if cid not in concept_info:
                concept_info[cid] = {
                    "df": 0, "cases": {},
                    "unlisted_label": None, "unlisted_description": None,
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

    n_corpus = len(case_ids)
    max_df = int(n_corpus * 0.25)  # 25% cap — removes overly broad concepts
    eligible = [
        (cid, info) for cid, info in concept_info.items()
        if 3 <= info["df"] <= max_df
    ]
    eligible.sort(key=lambda x: x[1]["df"], reverse=True)
    queries = eligible[:n_queries]

    # Build query texts
    query_texts = []
    for cid, info in queries:
        parts = []
        if info["unlisted_label"]:
            parts.append(info["unlisted_label"])
        if info["unlisted_description"]:
            parts.append(info["unlisted_description"])
        if not parts:
            clean = cid
            if clean.startswith("UNLISTED_"):
                clean = clean[len("UNLISTED_"):]
            clean = clean.replace("_", " ")
            parts.append(clean)
        if info["interpretation"]:
            parts.append(info["interpretation"][:200])
        query_texts.append(" ".join(parts))

    # Build qrels
    qrels_binary = []
    qrels_graded = []
    for cid, info in queries:
        qrels_binary.append({idx: 1 for idx in info["cases"]})
        qrels_graded.append({idx: grade for idx, grade in info["cases"].items()})

    return queries, query_texts, qrels_binary, qrels_graded


def build_ik_qrels(ik_qrels_path: str, case_ids: List[str], n_queries: int = 50):
    """Build qrels from Indian Kanoon statute tags.
    Returns (queries, query_texts, qrels_binary, qrels_graded).
    """
    data = load_json(ik_qrels_path)
    ik_queries = data.get("queries", {})

    # Map case_id -> index in our corpus
    cid_to_idx = {cid: idx for idx, cid in enumerate(case_ids)}

    # Filter and sort by document frequency
    eligible = []
    for concept_id, info in ik_queries.items():
        # Map case IDs to corpus indices
        cases_by_idx = {}
        for cjpe_id, rel in info.get("cases", {}).items():
            if cjpe_id in cid_to_idx:
                cases_by_idx[cid_to_idx[cjpe_id]] = rel
        if len(cases_by_idx) >= 3:
            eligible.append((concept_id, info, cases_by_idx))

    eligible.sort(key=lambda x: len(x[2]), reverse=True)
    eligible = eligible[:n_queries]

    queries = [(cid, info) for cid, info, _ in eligible]
    query_texts = [
        info.get("label", cid).replace("_", " ")
        for cid, info in queries
    ]
    qrels_binary = [{idx: 1 for idx in cases} for _, _, cases in eligible]
    qrels_graded = [{idx: rel for idx, rel in cases.items()} for _, _, cases in eligible]

    return queries, query_texts, qrels_binary, qrels_graded


# ---------------------------------------------------------------------------
# REGEX-BASED INDEPENDENT QRELS
# ---------------------------------------------------------------------------

# Statute/article prefix variants
_REGEX_PREFIX_MAP = {
    "article": r"(?:Article|Art\.?)\s*",
    "section": r"(?:Section|Sec\.?|S\.?)\s*",
    "rule":    r"(?:Rule)\s*",
    "order":   r"(?:Order)\s*",
    "clause":  r"(?:Clause|Cl\.?)\s*",
}

_REGEX_ACT_ALIASES = {
    "ipc": [r"I\.?P\.?C\.?", r"Indian\s+Penal\s+Code"],
    "indian_penal_code": [r"I\.?P\.?C\.?", r"Indian\s+Penal\s+Code"],
    "crpc": [r"Cr\.?P\.?C\.?", r"Code\s+of\s+Criminal\s+Procedure"],
    "code_criminal_procedure": [r"Cr\.?P\.?C\.?", r"Code\s+of\s+Criminal\s+Procedure"],
    "cpc": [r"C\.?P\.?C\.?", r"Code\s+of\s+Civil\s+Procedure"],
    "code_civil_procedure": [r"C\.?P\.?C\.?", r"Code\s+of\s+Civil\s+Procedure"],
    "constitution": [r"Constitution", r"Constitution\s+of\s+India"],
    "constitution_india": [r"Constitution", r"Constitution\s+of\s+India"],
    "evidence_act": [r"Evidence\s+Act", r"Indian\s+Evidence\s+Act"],
    "bns": [r"B\.?N\.?S\.?", r"Bharatiya\s+Nyaya\s+Sanhita"],
    "bnss": [r"B\.?N\.?S\.?S\.?", r"Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita"],
    "ida": [r"I\.?D\.?\s*Act", r"Industrial\s+Disputes\s+Act"],
    "industrial_disputes_act": [r"I\.?D\.?\s*Act", r"Industrial\s+Disputes\s+Act"],
    "it_act": [r"I\.?T\.?\s*Act", r"Information\s+Technology\s+Act", r"Income[\s-]?Tax\s+Act"],
    "ndps": [r"N\.?D\.?P\.?S\.?", r"Narcotic\s+Drugs"],
    "ndps_act": [r"N\.?D\.?P\.?S\.?", r"Narcotic\s+Drugs"],
    "mv_act": [r"M\.?V\.?\s*Act", r"Motor\s+Vehicles?\s+Act"],
    "posh": [r"POSH", r"Sexual\s+Harassment"],
    "sarfaesi": [r"SARFAESI", r"Securitisation"],
    "rera": [r"RERA", r"Real\s+Estate"],
    "arms_act": [r"Arms\s+Act"],
    "sc_st_act": [r"SC/?ST", r"Scheduled\s+Castes?\s+and\s+Scheduled\s+Tribes?"],
    "pocso": [r"POCSO", r"Protection\s+of\s+Children"],
    "tada": [r"TADA", r"Terrorist.*?Disruptive"],
    "pota": [r"POTA", r"Prevention\s+of\s+Terrorism"],
    "uapa": [r"UAPA", r"Unlawful\s+Activities"],
    "nia": [r"N\.?I\.?A\.?\s*Act", r"National\s+Investigation\s+Agency"],
    "ni_act": [r"N\.?I\.?\s*Act", r"Negotiable\s+Instruments?\s+Act"],
    "transfer_property": [r"T\.?P\.?\s*Act", r"Transfer\s+of\s+Property"],
    "arbitration": [r"Arbitration.*?Conciliation\s+Act", r"Arbitration\s+Act"],
    "hindu_marriage": [r"Hindu\s+Marriage\s+Act", r"H\.?M\.?A\.?"],
    "companies_act": [r"Companies\s+Act"],
    "consumer_protection": [r"Consumer\s+Protection\s+Act"],
    "land_acquisition": [r"Land\s+Acquisition\s+Act"],
}

_DOCTRINE_KEYWORDS = {
    "wednesbury": "Wednesbury",
    "basic_structure": "basic structure",
    "maneka_gandhi": "Maneka Gandhi",
    "rarest_of_rare": r"rarest of (?:the )?rare",
    "reasonable_classification": "reasonable classification",
    "natural_justice": "natural justice",
    "res_judicata": "res judicata",
    "stare_decisis": "stare decisis",
    "proportionality": "proportionality",
    "legitimate_expectation": "legitimate expectation",
    "promissory_estoppel": "promissory estoppel",
    "due_process": "due process",
    "right_to_life": "right to life",
    "right_to_privacy": "right to privacy",
    "right_to_livelihood": "right to livelihood",
    "free_speech": r"free(?:dom of)? speech",
    "double_jeopardy": "double jeopardy",
    "habeas_corpus": "habeas corpus",
    "eminent_domain": "eminent domain",
    "adverse_possession": "adverse possession",
    "specific_performance": "specific performance",
    "anticipatory_bail": "anticipatory bail",
    "dying_declaration": "dying declaration",
    "dowry_death": "dowry death",
    "cruelty": "cruelty",
    "murder": r"\bmurder\b",
    "culpable_homicide": "culpable homicide",
    "cheating": r"\bcheating\b",
    "defamation": "defamation",
    "kidnapping": "kidnapping",
    "robbery": r"\brobbery\b",
    "dacoity": "dacoity",
    "forgery": r"\bforgery\b",
    "writ": r"\bwrit\b",
    "common_intention": "common intention",
    "common_object": "common object",
    "abetment": r"\babetment\b",
    "criminal_conspiracy": "criminal conspiracy",
    "attempt_to_murder": "attempt to (?:commit )?murder",
    "hurt": r"\bgrevious hurt\b|\bhurt\b",
    "negligence": r"\bnegligence\b",
    "strict_liability": "strict liability",
    "vicarious_liability": "vicarious liability",
    "ultra_vires": "ultra vires",
    "locus_standi": "locus standi",
    "certiorari": r"\bcertiorari\b",
    "mandamus": r"\bmandamus\b",
    "prohibition": r"\bprohibition\b",
    "quo_warranto": "quo warranto",
    "judicial_review": "judicial review",
    "separation_of_powers": "separation of powers",
    "pith_and_substance": "pith and substance",
    "colourable_legislation": "colourable legislation",
    "doctrine_of_eclipse": "doctrine of eclipse",
    "doctrine_of_severability": "doctrine of severability",
    "bail": r"\bbail\b",
    "quashing": r"\bquashing\b",
    "discharge": r"\bdischarge\b",
    "acquittal": r"\bacquittal\b",
    "compensation": r"\bcompensation\b",
    "restitution": r"\brestitution\b",
    "injunction": r"\binjunction\b",
    "stay": r"\bstay\b",
    "arbitration": r"\barbitration\b",
    "mediation": r"\bmediation\b",
    "harmonious_construction": "harmonious construction",
    "rational_nexus": "rational nexus",
    "criminal_breach_trust": "criminal breach of trust",
    "criminal_breach": "criminal breach",
    "breach_of_trust": "breach of trust",
    "mischief": r"\bmischief\b",
    "trespass": r"\btrespass\b",
    "extortion": r"\bextortion\b",
    "misappropriation": r"\bmisappropriation\b",
}


def _concept_id_to_patterns(concept_id: str, label: Optional[str] = None) -> List[str]:
    """Generate regex patterns for matching a concept in raw text."""
    raw = concept_id
    # Strip common prefixes
    for prefix in ("UNLISTED_", "CONCEPT_", "DOCTRINE_", "TEST_"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break

    tokens = raw.lower().split("_")
    tokens = [t for t in tokens if t]
    patterns = []

    # Strategy 0: Handle abbreviated forms like "S109", "S307", "ART14"
    # Expand them into separate prefix + number tokens before main logic
    expanded_tokens = []
    for tok in tokens:
        # "s109" → "section", "109"
        m = re.match(r'^(s|sec)(\d+[a-z]?)$', tok)
        if m:
            expanded_tokens.extend(["section", m.group(2)])
            continue
        # "art14" / "art226" → "article", "14"
        m = re.match(r'^(art)(\d+[a-z]?)$', tok)
        if m:
            expanded_tokens.extend(["article", m.group(2)])
            continue
        expanded_tokens.append(tok)
    tokens = expanded_tokens

    # Also handle "const" as a token → expand to "constitution"
    tokens = ["constitution" if t == "const" else t for t in tokens]

    # Strategy 1: Statute pattern (prefix + number)
    prefix_type = None
    number = None
    act_tokens = []

    for i, tok in enumerate(tokens):
        if tok in _REGEX_PREFIX_MAP and not prefix_type:
            prefix_type = tok
            for j in range(i + 1, len(tokens)):
                if re.match(r'^\d+[a-z]?$', tokens[j]):
                    number = tokens[j]
                    act_tokens = [t for k, t in enumerate(tokens)
                                  if k != i and k != j
                                  and t not in ("of", "the", "and", "to", "for", "in", "under")]
                    break
            break
        m = re.match(r'^(article|section|rule|order|clause)(\d+[a-z]?)$', tok)
        if m:
            prefix_key = m.group(1)
            if prefix_key in _REGEX_PREFIX_MAP:
                prefix_type = prefix_key
                number = m.group(2)
                act_tokens = [t for k, t in enumerate(tokens) if k != i
                              and t not in ("of", "the", "and", "to", "for", "in", "under")]
                break

    # Bare number + act name: e.g., "ipc_302"
    if not prefix_type:
        for i, tok in enumerate(tokens):
            if re.match(r'^\d+[a-z]?$', tok):
                remaining = [t for j, t in enumerate(tokens) if j != i
                             and t not in ("of", "the", "and", "to", "for", "in", "under")]
                remaining_str = "_".join(remaining)
                if remaining_str in _REGEX_ACT_ALIASES:
                    prefix_type = "section"
                    number = tok
                    act_tokens = remaining
                    break

    if prefix_type and number:
        prefix_re = _REGEX_PREFIX_MAP[prefix_type]
        # Check for subsection: next numeric token after number
        subsection = None
        for t in act_tokens:
            if re.match(r'^\d+$', t) and t != number:
                subsection = t
                act_tokens = [a for a in act_tokens if a != t]
                break

        act_str = "_".join(act_tokens)
        has_act_context = act_str in _REGEX_ACT_ALIASES

        # Always include bare pattern: "Section 302", "Article 21"
        patterns.append(f"{prefix_re}{number}\\b")
        if subsection:
            patterns.append(f"{prefix_re}{number}\\s*\\({subsection}\\)")

        # Add act-qualified patterns when available
        if has_act_context:
            for alias in _REGEX_ACT_ALIASES[act_str]:
                patterns.append(f"{prefix_re}{number}\\b.{{0,30}}{alias}")
                patterns.append(f"\\b{number}\\s+{alias}")
                if subsection:
                    patterns.append(f"{prefix_re}{number}\\s*\\({subsection}\\).{{0,30}}{alias}")

    # Strategy 2: Doctrine/keyword matching
    raw_lower = raw.lower()
    for key, keyword_re in _DOCTRINE_KEYWORDS.items():
        if key in raw_lower:
            patterns.append(keyword_re)

    # Strategy 3: Use label if available
    if label and len(label) > 4:
        stops = {"the", "and", "for", "with", "from", "under", "that", "this",
                 "have", "been", "case", "court", "section", "article", "act"}
        label_words = [w for w in re.split(r'\s+', label)
                       if len(w) > 3 and w.lower() not in stops]
        if len(label_words) >= 2:
            phrase_re = r"\s+".join(re.escape(w) for w in label_words[:3])
            patterns.append(phrase_re)

    # Strategy 4: Fallback — multi-word phrase
    if not patterns:
        clean_tokens = [t for t in tokens if len(t) > 2
                        and t not in ("the", "and", "for", "with", "from", "under")]
        if len(clean_tokens) >= 2:
            phrase_re = r"\s+".join(re.escape(t) for t in clean_tokens[:3])
            patterns.append(phrase_re)
        elif len(clean_tokens) == 1 and len(clean_tokens[0]) > 5:
            patterns.append(r"\b" + re.escape(clean_tokens[0]) + r"\b")

    return patterns


def build_regex_qrels(
    queries: List[tuple],
    raw_texts: Dict[str, str],
    case_ids: List[str],
) -> Tuple[List[Dict[int, int]], List[Dict[int, int]], Dict[str, Any]]:
    """Build independent binary qrels by regex-matching concepts in raw text.

    Ground truth is derived from raw case text, not from extraction annotations.
    A case is relevant (grade=1) if any generated pattern matches its text.
    """
    qrels_binary = []
    qrels_graded = []
    diagnostics = {}

    for qi, (cid, info) in enumerate(queries):
        label = info.get("unlisted_label") if isinstance(info, dict) else None
        patterns = _concept_id_to_patterns(cid, label=label)

        compiled = []
        for p in patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error:
                continue

        if not compiled:
            qrels_binary.append({})
            qrels_graded.append({})
            diagnostics[cid] = {"n_matches": 0, "patterns": [], "skipped": True}
            continue

        matches = {}
        for idx, case_id in enumerate(case_ids):
            text = raw_texts.get(case_id, "")
            for regex in compiled:
                if regex.search(text):
                    matches[idx] = 1
                    break

        qrels_binary.append(matches)
        qrels_graded.append(matches.copy())
        diagnostics[cid] = {
            "n_matches": len(matches),
            "patterns": [p for p in patterns],
            "skipped": False,
        }

    return qrels_binary, qrels_graded, diagnostics


# ---------------------------------------------------------------------------
# RETRIEVAL METHODS
# ---------------------------------------------------------------------------

def run_tfidf(doc_texts, query_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        max_features=50000, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True,
    )
    doc_matrix = vectorizer.fit_transform(doc_texts)
    query_matrix = vectorizer.transform(query_texts)
    return cosine_similarity(query_matrix, doc_matrix)


def run_bm25(doc_texts, query_texts):
    from rank_bm25 import BM25Okapi

    tokenized_docs = [doc.lower().split() for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)
    scores = np.zeros((len(query_texts), len(doc_texts)))
    for qi, qt in enumerate(query_texts):
        scores[qi] = bm25.get_scores(qt.lower().split())
    return scores


def run_concept_set(queries, graph_list, case_ids, fuzzy_threshold=0.25):
    n_docs = len(graph_list)
    case_concept_data = []
    for idx, g in enumerate(graph_list):
        concept_entries = []
        seen = {}
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
                if cid == query_cid:
                    best_score = max(best_score, float(grade))
                    continue
                sim = _fuzzy_token_jaccard(query_tokens, tokens)
                if sim >= fuzzy_threshold:
                    best_score = max(best_score, float(grade) * sim)
            scores[qi, doc_idx] = best_score
    return scores


# ---------------------------------------------------------------------------
# DENSE RETRIEVAL BASELINES
# ---------------------------------------------------------------------------

def _batch_encode(model, texts: List[str], batch_size: int = 64,
                  prefix: str = "", show_progress: bool = True) -> np.ndarray:
    """Encode texts in batches using a SentenceTransformer model."""
    if prefix:
        texts = [prefix + t for t in texts]
    all_embs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False,
                            normalize_embeddings=True)
        all_embs.append(embs)
        if show_progress and (i // batch_size) % 5 == 0:
            print(f"    batch {i // batch_size + 1}/{n_batches}", end="\r")
    if show_progress:
        print(f"    encoded {len(texts)} texts" + " " * 20)
    return np.vstack(all_embs)


def _truncate_texts(texts: List[str], max_tokens: int = 400) -> List[str]:
    """Rough truncation to keep texts within model context limits.
    Uses whitespace tokenization as a proxy (~1.3 tokens per word).
    """
    truncated = []
    max_words = int(max_tokens / 1.3)
    for t in texts:
        words = t.split()
        if len(words) > max_words:
            truncated.append(" ".join(words[:max_words]))
        else:
            truncated.append(t)
    return truncated


def run_e5(doc_texts: List[str], query_texts: List[str],
           model_name: str = "intfloat/e5-large-v2",
           batch_size: int = 64) -> np.ndarray:
    """Dense retrieval using E5-large-v2.
    E5 requires 'query: ' and 'passage: ' prefixes.
    """
    from sentence_transformers import SentenceTransformer

    print(f"    Loading {model_name}...")
    model = SentenceTransformer(model_name)

    trunc_docs = _truncate_texts(doc_texts, max_tokens=400)

    print(f"    Encoding {len(trunc_docs)} documents...")
    doc_embs = _batch_encode(model, trunc_docs, batch_size=batch_size,
                             prefix="passage: ")

    print(f"    Encoding {len(query_texts)} queries...")
    query_embs = _batch_encode(model, query_texts, batch_size=batch_size,
                               prefix="query: ")

    # Cosine similarity (embeddings are already normalized)
    scores = query_embs @ doc_embs.T
    return scores


def run_bge(doc_texts: List[str], query_texts: List[str],
            model_name: str = "BAAI/bge-large-en-v1.5",
            batch_size: int = 64) -> np.ndarray:
    """Dense retrieval using BGE-large-en-v1.5.
    BGE uses 'Represent this sentence for searching relevant passages: ' prefix for queries.
    """
    from sentence_transformers import SentenceTransformer

    print(f"    Loading {model_name}...")
    model = SentenceTransformer(model_name)

    trunc_docs = _truncate_texts(doc_texts, max_tokens=400)

    print(f"    Encoding {len(trunc_docs)} documents...")
    doc_embs = _batch_encode(model, trunc_docs, batch_size=batch_size)

    print(f"    Encoding {len(query_texts)} queries...")
    query_prefix = "Represent this sentence for searching relevant passages: "
    query_embs = _batch_encode(model, query_texts, batch_size=batch_size,
                               prefix=query_prefix)

    scores = query_embs @ doc_embs.T
    return scores


def run_colbert(doc_texts: List[str], query_texts: List[str],
                model_name: str = "colbert-ir/colbertv2.0",
                batch_size: int = 32, doc_max_len: int = 300) -> np.ndarray:
    """Dense retrieval using ColBERTv2 with late-interaction MaxSim scoring.

    Unlike bi-encoders, ColBERT produces per-token embeddings and scores
    via MaxSim: for each query token, find the max similarity across all
    document tokens, then sum across query tokens.
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"    Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def _encode_colbert(texts: List[str], max_length: int = 512,
                        is_query: bool = False) -> List[np.ndarray]:
        """Encode texts into per-token normalized embeddings."""
        all_token_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**encoded)
            # Use last hidden state as token embeddings
            token_embs = outputs.last_hidden_state  # (B, seq_len, dim)
            attention_mask = encoded["attention_mask"]  # (B, seq_len)

            for j in range(token_embs.size(0)):
                mask = attention_mask[j].bool()
                emb = token_embs[j][mask]  # (n_tokens, dim)
                # L2 normalize each token embedding
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                all_token_embs.append(emb.cpu().numpy())

            if (i // batch_size) % 5 == 0:
                print(f"    batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}",
                      end="\r")
        print(f"    encoded {len(texts)} texts" + " " * 20)
        return all_token_embs

    # Truncate documents for manageable token counts
    trunc_docs = _truncate_texts(doc_texts, max_tokens=doc_max_len)

    print(f"    Encoding {len(trunc_docs)} documents (token-level)...")
    doc_token_embs = _encode_colbert(trunc_docs, max_length=512)

    print(f"    Encoding {len(query_texts)} queries (token-level)...")
    query_token_embs = _encode_colbert(query_texts, max_length=128, is_query=True)

    # MaxSim scoring
    print(f"    Computing MaxSim scores ({len(query_texts)}x{len(trunc_docs)})...")
    scores = np.zeros((len(query_texts), len(trunc_docs)))
    for qi, q_emb in enumerate(query_token_embs):
        q_tensor = torch.from_numpy(q_emb)  # (n_qtokens, dim)
        for di, d_emb in enumerate(doc_token_embs):
            d_tensor = torch.from_numpy(d_emb)  # (n_dtokens, dim)
            # MaxSim: for each query token, max similarity to any doc token
            sim_matrix = q_tensor @ d_tensor.T  # (n_qtokens, n_dtokens)
            max_sims = sim_matrix.max(dim=1).values  # (n_qtokens,)
            scores[qi, di] = max_sims.sum().item()
        if qi % 5 == 0:
            print(f"    query {qi + 1}/{len(query_texts)}", end="\r")
    print(f"    scored all queries" + " " * 20)

    return scores


# ---------------------------------------------------------------------------
# DOMAIN-ADAPTED LEGAL RETRIEVAL BASELINES
# ---------------------------------------------------------------------------

def _mean_pool_encode(model, tokenizer, texts: List[str],
                      batch_size: int = 64, max_length: int = 512,
                      show_progress: bool = True) -> np.ndarray:
    """Encode texts via mean pooling over last hidden state (for base BERT models
    that are not natively sentence-transformers).
    Returns L2-normalized embeddings.
    """
    device = next(model.parameters()).device
    all_embs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        # Mean pooling: average non-padding token embeddings
        token_embs = outputs.last_hidden_state          # (B, seq, dim)
        mask = encoded["attention_mask"].unsqueeze(-1)   # (B, seq, 1)
        summed = (token_embs * mask).sum(dim=1)          # (B, dim)
        counts = mask.sum(dim=1).clamp(min=1)            # (B, 1)
        mean_emb = summed / counts                       # (B, dim)
        # L2 normalize
        mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=-1)
        all_embs.append(mean_emb.cpu().numpy())
        if show_progress and (i // batch_size) % 5 == 0:
            print(f"    batch {i // batch_size + 1}/{n_batches}", end="\r")
    if show_progress:
        print(f"    encoded {len(texts)} texts" + " " * 20)
    return np.vstack(all_embs)


def run_legal_bert(doc_texts: List[str], query_texts: List[str],
                   model_name: str = "nlpaueb/legal-bert-base-uncased",
                   batch_size: int = 64) -> np.ndarray:
    """Domain-adapted dense retrieval using Legal-BERT (Chalkidis et al.).

    Legal-BERT is pre-trained on English legal corpora (EU legislation,
    UK legislation, ECHR cases, US court opinions, contracts).
    Used as a bi-encoder with mean pooling.
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"    Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    trunc_docs = _truncate_texts(doc_texts, max_tokens=400)

    print(f"    Encoding {len(trunc_docs)} documents...")
    doc_embs = _mean_pool_encode(model, tokenizer, trunc_docs,
                                 batch_size=batch_size)

    print(f"    Encoding {len(query_texts)} queries...")
    query_embs = _mean_pool_encode(model, tokenizer, query_texts,
                                   batch_size=batch_size)

    # Cosine similarity (embeddings are already normalized)
    scores = query_embs @ doc_embs.T
    return scores


def run_inlegal_bert(doc_texts: List[str], query_texts: List[str],
                     model_name: str = "law-ai/InLegalBERT",
                     batch_size: int = 64) -> np.ndarray:
    """Domain-adapted dense retrieval using InLegalBERT (IIT Kharagpur).

    InLegalBERT is pre-trained on Indian legal documents including
    Supreme Court and High Court judgments, making it the closest
    domain match for IL-TUR retrieval. Used as a bi-encoder with
    mean pooling.

    Reference: Paul et al., "Pre-trained Language Models for the
    Indian Legal Domain" (ICAIL 2023).
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"    Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    trunc_docs = _truncate_texts(doc_texts, max_tokens=400)

    print(f"    Encoding {len(trunc_docs)} documents...")
    doc_embs = _mean_pool_encode(model, tokenizer, trunc_docs,
                                 batch_size=batch_size)

    print(f"    Encoding {len(query_texts)} queries...")
    query_embs = _mean_pool_encode(model, tokenizer, query_texts,
                                   batch_size=batch_size)

    scores = query_embs @ doc_embs.T
    return scores


# ---------------------------------------------------------------------------
# IR METRICS
# ---------------------------------------------------------------------------

def dcg_at_k(relevances, k=10):
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(ranking_rels, qrel_graded, k=10):
    actual = dcg_at_k(ranking_rels, k)
    ideal_rels = sorted(qrel_graded.values(), reverse=True)
    ideal = dcg_at_k(ideal_rels, k)
    return actual / ideal if ideal > 0 else 0.0


def average_precision(ranking_doc_ids, qrel_binary):
    if not qrel_binary:
        return 0.0
    n_rel = len(qrel_binary)
    hits = 0
    sum_prec = 0.0
    for rank, doc_id in enumerate(ranking_doc_ids, 1):
        if doc_id in qrel_binary:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / n_rel


def precision_at_k(ranking_doc_ids, qrel_binary, k=10):
    top_k = ranking_doc_ids[:k]
    hits = sum(1 for d in top_k if d in qrel_binary)
    return hits / k


def evaluate_method(scores_matrix, qrels_binary, qrels_graded, k=10, seed=42):
    rng = np.random.RandomState(seed)
    n_queries = scores_matrix.shape[0]
    per_query = []

    for qi in range(n_queries):
        query_scores = scores_matrix[qi].copy()
        query_scores += rng.uniform(0, 1e-10, size=query_scores.shape)
        ranked_indices = np.argsort(-query_scores).tolist()

        ranking_rels = [qrels_graded[qi].get(idx, 0) for idx in ranked_indices[:k]]
        ndcg = ndcg_at_k(ranking_rels, qrels_graded[qi], k)
        ap = average_precision(ranked_indices, qrels_binary[qi])
        p10 = precision_at_k(ranked_indices, qrels_binary[qi], k)

        per_query.append({"ndcg": ndcg, "ap": ap, "p10": p10})

    mean_metrics = {
        "nDCG@10": float(np.mean([pq["ndcg"] for pq in per_query])),
        "MAP": float(np.mean([pq["ap"] for pq in per_query])),
        "P@10": float(np.mean([pq["p10"] for pq in per_query])),
    }
    return per_query, mean_metrics


def paired_ttest(scores_a, scores_b):
    from scipy.stats import ttest_rel
    a, b = np.array(scores_a), np.array(scores_b)
    if np.allclose(a, b):
        return 0.0, 1.0
    return ttest_rel(a, b)


def significance_label(p_value):
    if p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Non-circular concept retrieval evaluation"
    )
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--ik_qrels", default=None,
                        help="Path to Indian Kanoon qrels (ik_ground_truth/ik_qrels.json)")
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--raw_text_mode", action="store_true",
                        help="Use raw IL-TUR text for BM25 baseline (recommended)")
    parser.add_argument("--raw_text_dir", default=None,
                        help="Local directory with raw case texts (*.txt, named by case_id)")
    parser.add_argument("--qrel_mode", default="regex",
                        choices=["regex", "annotation", "ik"],
                        help="Qrel source: regex (independent, default), annotation, or ik")
    parser.add_argument("--compare_qrels", action="store_true",
                        help="Run with both regex and annotation qrels for comparison")
    parser.add_argument("--dense_baselines", action="store_true",
                        help="Run dense retrieval baselines (E5, BGE, ColBERT)")
    parser.add_argument("--e5_model", default="intfloat/e5-large-v2",
                        help="E5 model name (default: intfloat/e5-large-v2)")
    parser.add_argument("--bge_model", default="BAAI/bge-large-en-v1.5",
                        help="BGE model name (default: BAAI/bge-large-en-v1.5)")
    parser.add_argument("--colbert_model", default="colbert-ir/colbertv2.0",
                        help="ColBERT model name (default: colbert-ir/colbertv2.0)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for dense encoding (default: 64)")
    parser.add_argument("--skip_colbert", action="store_true",
                        help="Skip ColBERT (slow MaxSim) — run only E5 + BGE")
    parser.add_argument("--legal_baselines", action="store_true",
                        help="Run domain-adapted legal retrieval baselines "
                             "(Legal-BERT, InLegalBERT)")
    parser.add_argument("--legal_bert_model", default="nlpaueb/legal-bert-base-uncased",
                        help="Legal-BERT model (default: nlpaueb/legal-bert-base-uncased)")
    parser.add_argument("--inlegal_bert_model", default="law-ai/InLegalBERT",
                        help="InLegalBERT model (default: law-ai/InLegalBERT)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load graphs
    # ------------------------------------------------------------------
    print("Loading case graphs...")
    graphs = iter_graphs(Path(args.graph_dir))
    print(f"  Loaded {len(graphs)} cases")

    case_ids = [cid for cid, _ in graphs]
    graph_list = [g for _, g in graphs]
    n_corpus = len(case_ids)

    # ------------------------------------------------------------------
    # Select queries (always from annotations — this is query selection,
    # not ground truth)
    # ------------------------------------------------------------------
    print("\nSelecting query concepts from graph annotations...")
    ann_queries, ann_query_texts, _, _ = build_annotation_qrels(
        graph_list, case_ids, args.n_queries
    )
    queries = ann_queries
    query_texts = ann_query_texts
    n_queries = len(queries)

    # ------------------------------------------------------------------
    # Load raw texts (needed for both regex qrels and BM25)
    # ------------------------------------------------------------------
    print("\nLoading raw IL-TUR texts...")
    raw_texts = load_raw_iltur_texts(case_ids, raw_text_dir=args.raw_text_dir)
    doc_texts = [raw_texts.get(cid, "empty") for cid in case_ids]

    # ------------------------------------------------------------------
    # Build qrels
    # ------------------------------------------------------------------
    if args.qrel_mode == "regex":
        print("\n[QRELS] Regex-matched in raw text (independent ground truth)")
        regex_qb, regex_qg, diag = build_regex_qrels(queries, raw_texts, case_ids)
        qrels_binary = regex_qb
        qrels_graded = regex_qg
        qrel_source = "Regex-matched (independent)"

        # Report diagnostics
        matched_queries = sum(1 for d in diag.values() if d["n_matches"] >= 3)
        skipped = sum(1 for d in diag.values() if d.get("skipped") or d["n_matches"] < 3)
        total_matches = sum(d["n_matches"] for d in diag.values())
        print(f"  {matched_queries}/{n_queries} queries have ≥3 matching cases")
        print(f"  {total_matches} total case-concept matches")
        print(f"  Avg matches/query: {total_matches/n_queries:.1f}")

        # Show per-query match counts
        print(f"\n  Per-query regex matches (top 10 + bottom 5):")
        sorted_diag = sorted(diag.items(), key=lambda x: x[1]["n_matches"], reverse=True)
        for cid, d in sorted_diag[:10]:
            short_cid = cid[:40] + "..." if len(cid) > 40 else cid
            print(f"    {short_cid:<43} {d['n_matches']:>5} cases")
        if len(sorted_diag) > 15:
            print(f"    {'...':<43}")
        for cid, d in sorted_diag[-5:]:
            short_cid = cid[:40] + "..." if len(cid) > 40 else cid
            status = "SKIP" if d["n_matches"] < 3 else ""
            print(f"    {short_cid:<43} {d['n_matches']:>5} cases  {status}")

    elif args.qrel_mode == "ik" and args.ik_qrels:
        print(f"\nUsing Indian Kanoon ground truth from {args.ik_qrels}")
        queries, query_texts, qrels_binary, qrels_graded = build_ik_qrels(
            args.ik_qrels, case_ids, args.n_queries
        )
        qrel_source = "Indian Kanoon (independent)"
    else:
        print("\nUsing annotation-derived qrels")
        _, _, qrels_binary, qrels_graded = build_annotation_qrels(
            graph_list, case_ids, args.n_queries
        )
        qrel_source = "Annotation-derived"

    # Filter out queries with <3 or too many (>40% corpus) relevant docs
    max_regex_df = int(n_corpus * 0.4)
    valid_mask = [i for i in range(len(queries))
                  if 3 <= len(qrels_binary[i]) <= max_regex_df]
    if len(valid_mask) < len(queries):
        dropped_low = sum(1 for i in range(len(queries)) if len(qrels_binary[i]) < 3)
        dropped_high = sum(1 for i in range(len(queries)) if len(qrels_binary[i]) > max_regex_df)
        print(f"\n  Filtering: {len(queries)} -> {len(valid_mask)} queries "
              f"(3 ≤ df ≤ {max_regex_df})")
        if dropped_low:
            print(f"    Dropped {dropped_low} queries with <3 matches")
        if dropped_high:
            print(f"    Dropped {dropped_high} queries with >{max_regex_df} matches (too broad)")
        queries = [queries[i] for i in valid_mask]
        query_texts = [query_texts[i] for i in valid_mask]
        qrels_binary = [qrels_binary[i] for i in valid_mask]
        qrels_graded = [qrels_graded[i] for i in valid_mask]

    n_queries = len(queries)
    if n_queries == 0:
        print("ERROR: No queries with sufficient relevant documents. Exiting.")
        sys.exit(1)

    dfs = [len(qb) for qb in qrels_binary]
    avg_rel = float(np.mean(dfs))

    print(f"\nQrel source: {qrel_source}")
    print(f"Queries: {n_queries} concepts (df range: {min(dfs)}-{max(dfs)}, "
          f"median df: {int(np.median(dfs))})")
    print(f"Corpus: {n_corpus} cases")
    print(f"Avg relevant docs/query: {avg_rel:.1f}")

    # ------------------------------------------------------------------
    # Run retrieval methods
    # ------------------------------------------------------------------
    methods_results = {}
    method_order = ["BM25"]

    # 1. BM25 over raw case text
    print(f"\n[1/?] BM25 (Raw IL-TUR)...")
    bm25_scores = run_bm25(doc_texts, query_texts)
    pq, mm = evaluate_method(bm25_scores, qrels_binary, qrels_graded)
    methods_results["BM25"] = (pq, mm)
    print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")

    # 2-4. Dense retrieval baselines (optional)
    if args.dense_baselines:
        step = 2

        # E5-large
        print(f"\n[{step}/?] E5-large-v2 (Dense, Raw IL-TUR)...")
        try:
            e5_scores = run_e5(doc_texts, query_texts,
                               model_name=args.e5_model,
                               batch_size=args.batch_size)
            pq, mm = evaluate_method(e5_scores, qrels_binary, qrels_graded)
            methods_results["E5-large"] = (pq, mm)
            method_order.append("E5-large")
            print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")
        except Exception as e:
            print(f"  ERROR: E5 failed — {e}")
        step += 1

        # BGE-large
        print(f"\n[{step}/?] BGE-large-en-v1.5 (Dense, Raw IL-TUR)...")
        try:
            bge_scores = run_bge(doc_texts, query_texts,
                                 model_name=args.bge_model,
                                 batch_size=args.batch_size)
            pq, mm = evaluate_method(bge_scores, qrels_binary, qrels_graded)
            methods_results["BGE-large"] = (pq, mm)
            method_order.append("BGE-large")
            print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")
        except Exception as e:
            print(f"  ERROR: BGE failed — {e}")
        step += 1

        # ColBERTv2
        if not args.skip_colbert:
            print(f"\n[{step}/?] ColBERTv2 (Late Interaction, Raw IL-TUR)...")
            try:
                colbert_scores = run_colbert(doc_texts, query_texts,
                                             model_name=args.colbert_model,
                                             batch_size=args.batch_size // 2)
                pq, mm = evaluate_method(colbert_scores, qrels_binary, qrels_graded)
                methods_results["ColBERTv2"] = (pq, mm)
                method_order.append("ColBERTv2")
                print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")
            except Exception as e:
                print(f"  ERROR: ColBERT failed — {e}")
            step += 1
        else:
            print(f"\n  [Skipping ColBERT (--skip_colbert)]")

    # 5-6. Domain-adapted legal retrieval baselines (optional)
    if args.legal_baselines:
        step = len(method_order) + 1

        # Legal-BERT (Chalkidis et al.)
        print(f"\n[{step}/?] Legal-BERT (Domain-adapted bi-encoder, Raw IL-TUR)...")
        try:
            lb_scores = run_legal_bert(doc_texts, query_texts,
                                       model_name=args.legal_bert_model,
                                       batch_size=args.batch_size)
            pq, mm = evaluate_method(lb_scores, qrels_binary, qrels_graded)
            methods_results["Legal-BERT"] = (pq, mm)
            method_order.append("Legal-BERT")
            print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")
        except Exception as e:
            print(f"  ERROR: Legal-BERT failed — {e}")
        step += 1

        # InLegalBERT (Indian legal domain)
        print(f"\n[{step}/?] InLegalBERT (Indian legal bi-encoder, Raw IL-TUR)...")
        try:
            ilb_scores = run_inlegal_bert(doc_texts, query_texts,
                                           model_name=args.inlegal_bert_model,
                                           batch_size=args.batch_size)
            pq, mm = evaluate_method(ilb_scores, qrels_binary, qrels_graded)
            methods_results["InLegalBERT"] = (pq, mm)
            method_order.append("InLegalBERT")
            print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")
        except Exception as e:
            print(f"  ERROR: InLegalBERT failed — {e}")
        step += 1

    # N. ConceptSet (always uses graph annotations)
    method_order.append("ConceptSet")
    print(f"\n[last] ConceptSet (graph annotations)...")
    concept_scores = run_concept_set(queries, graph_list, case_ids)
    pq, mm = evaluate_method(concept_scores, qrels_binary, qrels_graded)
    methods_results["ConceptSet"] = (pq, mm)
    print(f"  nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}  P@10={mm['P@10']:.3f}")

    # ------------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------------
    print("\nStatistical significance (ConceptSet vs each baseline):")
    concept_ndcgs = [pq["ndcg"] for pq in methods_results["ConceptSet"][0]]
    sig_results = {}
    for name in method_order:
        if name == "ConceptSet":
            continue
        if name not in methods_results:
            continue
        baseline_ndcgs = [pq["ndcg"] for pq in methods_results[name][0]]
        t_stat, p_val = paired_ttest(concept_ndcgs, baseline_ndcgs)
        sig_results[name] = (t_stat, p_val, significance_label(p_val))
        print(f"  ConceptSet vs {name:<12}: t={t_stat:>7.3f}, p={p_val:.4f} {sig_results[name][2]}")

    # Pairwise: Dense vs BM25 (useful for the narrative)
    if args.dense_baselines:
        print("\nStatistical significance (Dense baselines vs BM25):")
        bm25_ndcgs = [pq["ndcg"] for pq in methods_results["BM25"][0]]
        for name in method_order:
            if name in ("BM25", "ConceptSet") or name not in methods_results:
                continue
            baseline_ndcgs = [pq["ndcg"] for pq in methods_results[name][0]]
            t_stat, p_val = paired_ttest(bm25_ndcgs, baseline_ndcgs)
            label = significance_label(p_val)
            direction = "BM25 > " + name if t_stat > 0 else name + " > BM25"
            print(f"  BM25 vs {name:<12}: t={t_stat:>7.3f}, p={p_val:.4f} {label}  ({direction})")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"CONCEPT RETRIEVAL EVALUATION")
    print(f"  Qrel source:  {qrel_source}")
    print(f"  Text source:  Raw IL-TUR")
    print(f"  Queries:      {n_queries} concepts (df {min(dfs)}-{max(dfs)}, "
          f"median {int(np.median(dfs))})")
    print(f"  Corpus:       {n_corpus} cases")
    print(f"  Avg rel/query: {avg_rel:.1f}")
    print(f"{'=' * 78}")

    header = f"{'Method':<17} {'Type':<13} {'nDCG@10':>8} {'MAP':>8} {'P@10':>8}    {'sig':>4}"
    print(header)
    print("-" * 70)

    method_types = {
        "BM25": "Sparse",
        "E5-large": "Dense (bi)",
        "BGE-large": "Dense (bi)",
        "ColBERTv2": "Dense (late)",
        "Legal-BERT": "Dense (legal)",
        "InLegalBERT": "Dense (legal)",
        "ConceptSet": "Structured",
    }

    for name in method_order:
        if name not in methods_results:
            continue
        _, mm = methods_results[name]
        mtype = method_types.get(name, "")
        if name == "ConceptSet":
            sig_str = "--"
        else:
            sig_str = sig_results.get(name, (0, 1.0, ""))[2]
        print(f"{name:<17} {mtype:<13} {mm['nDCG@10']:>8.3f} {mm['MAP']:>8.3f} {mm['P@10']:>8.3f}    {sig_str:>4}")

    print("-" * 70)
    if args.qrel_mode == "regex":
        print("\n  Note: Relevance judged by regex-matching concept identifiers")
        print("  in raw IL-TUR text (independent of extraction pipeline).")
        print("  BM25 and dense baselines search raw text; ConceptSet matches")
        print("  graph annotations. †p<0.05 *p<0.05 **p<0.01 (paired t-test vs ConceptSet)")
    else:
        print("\n  Note: BM25 and dense baselines operate over raw IL-TUR case text;")
        print("  ConceptSet matches against ontology-normalized graph annotations.")

    # ------------------------------------------------------------------
    # Optional: compare regex vs annotation qrels
    # ------------------------------------------------------------------
    if args.compare_qrels:
        print(f"\n{'=' * 70}")
        print("COMPARISON: Regex qrels vs Annotation-derived qrels")
        print(f"{'=' * 70}")

        # Get annotation qrels for the same queries
        ann_queries_full, ann_qtexts_full, ann_qb, ann_qg = build_annotation_qrels(
            graph_list, case_ids, args.n_queries
        )

        # Agreement analysis
        if args.qrel_mode == "regex":
            # Rebuild regex qrels for all queries (before filtering) for comparison
            regex_qb_all, _, _ = build_regex_qrels(ann_queries_full, raw_texts, case_ids)
            print("\nQuery-level agreement (regex vs annotation):")
            agreements = []
            for qi in range(min(len(regex_qb_all), len(ann_qb))):
                regex_set = set(regex_qb_all[qi].keys())
                ann_set = set(ann_qb[qi].keys())
                if regex_set or ann_set:
                    jaccard = len(regex_set & ann_set) / len(regex_set | ann_set)
                    agreements.append(jaccard)
            if agreements:
                print(f"  Mean Jaccard overlap: {np.mean(agreements):.3f}")
                print(f"  Median Jaccard overlap: {np.median(agreements):.3f}")

        # Re-run with annotation qrels
        print("\n[Annotation-derived qrels]")
        ann_valid = [i for i in range(len(ann_qb)) if len(ann_qb[i]) >= 3]
        ann_qb_f = [ann_qb[i] for i in ann_valid]
        ann_qg_f = [ann_qg[i] for i in ann_valid]
        ann_qtexts_f = [ann_qtexts_full[i] for i in ann_valid]
        ann_queries_f = [ann_queries_full[i] for i in ann_valid]

        bm25_ann = run_bm25(doc_texts, ann_qtexts_f)
        _, mm = evaluate_method(bm25_ann, ann_qb_f, ann_qg_f)
        print(f"  BM25              nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}")

        cs = run_concept_set(ann_queries_f, graph_list, case_ids)
        _, mm = evaluate_method(cs, ann_qb_f, ann_qg_f)
        print(f"  ConceptSet        nDCG@10={mm['nDCG@10']:.3f}  MAP={mm['MAP']:.3f}")

    print()


if __name__ == "__main__":
    main()