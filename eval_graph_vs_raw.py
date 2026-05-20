#!/usr/bin/env python3
"""
eval_graph_vs_raw.py — Head-to-head: Graph-structured vs Raw-text zero-shot prediction.

The core question for the SIGIR paper:
    Does structuring a judgment into a legal reasoning graph improve
    LLM zero-shot prediction compared to reading the raw text?

Design:
    BOTH methods are fully blinded — the LLM never sees how the court ruled.
    The only difference is how the case is *presented*:
        A) GRAPH:    Structured extraction (facts, concepts, issues, party arguments)
                     with type labels, relevance scores, and explicit issue framing.
                     Holdings, court_response, issue answers, outcome ALL stripped.
                     Fact text scrubbed for court-finding language.
                     Concept interpretations excluded. Precedent treatment excluded.
        B) RAW TEXT: First N chars of the original judgment, with aggressive
                     outcome-phrase stripping. Headnotes removed. Short-judgment
                     safety. Given MORE text than graph (generous to baseline)
                     to isolate the effect of *structure* not *volume*.

    Same LLM, same system prompt logic, same cases, same seed.

Usage:
    python eval_graph_vs_raw.py --graph_dir iltur_graphs --n 50 --k_chars 4000
    python eval_graph_vs_raw.py --graph_dir iltur_graphs --n 100 --concurrent 10

Requirements:
    pip install datasets python-dotenv httpx numpy
"""

from __future__ import annotations
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_graphs(graph_dir):
    results = []
    for p in sorted(graph_dir.glob("*.json")):
        if p.name == "checkpoint.json":
            continue
        try:
            g = load_json(p)
            if isinstance(g, dict) and isinstance(g.get("case_id"), str):
                results.append((g["case_id"], g))
        except Exception:
            continue
    return results


def load_labels_hf(name="Exploration-Lab/IL-TUR", config="cjpe", split="single_train"):
    from datasets import load_dataset
    return {str(ex["id"]): int(ex["label"]) for ex in load_dataset(name, config)[split]}


def load_raw_texts_hf(name="Exploration-Lab/IL-TUR", config="cjpe", split="single_train"):
    """Load raw judgment texts keyed by case ID."""
    from datasets import load_dataset
    return {str(ex["id"]): ex["text"] for ex in load_dataset(name, config)[split]}


def _sanitize_case_id_for_match(case_id: str) -> str:
    """Reproduce run_iltur._sanitize_case_id so loader keys match graph case_ids.

    The extractor sanitizes filenames (Turkish letters İ/Ş etc. → "_") when
    producing the graph JSON name. Labels and raw_texts loaded here must use
    the SAME sanitized stem so iter_graphs() finds matches.
    """
    s = str(case_id or "").strip()
    if not s:
        return "case"
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "case"


def load_echr_turkey_labels(local_dir: str = "echr_turkey_cases") -> Dict[str, int]:
    """Mirror of run_iltur.load_echr_turkey_labels for the eval side.

    Reads echr_turkey_cases/metadata.csv and returns {sanitized_stem: 0|1}
    keyed by the same sanitized case_ids the extractor uses for graph filenames.
    Cases that are admissibility-only / struck out / friendly-settled get
    skipped (no merits finding).
    """
    import csv
    from pathlib import Path as _Path
    p = _Path(local_dir)
    csv_path = p / "metadata.csv"
    if not csv_path.exists():
        return {}
    itemid_to_conclusion: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = (row.get("itemid") or "").strip()
            if iid:
                itemid_to_conclusion[iid] = (row.get("conclusion") or "").strip()

    def _derive(conclusion: str):
        if not conclusion:
            return None
        has_v = has_nv = False
        for finding in conclusion.split(";"):
            fl = finding.strip().lower()
            if fl.startswith("no violation of"):
                has_nv = True
            elif fl.startswith("violation of") or fl.startswith("violations of"):
                has_v = True
        if has_v:
            return 1
        if has_nv:
            return 0
        return None

    out: Dict[str, int] = {}
    for tf in p.glob("*.txt"):
        stem = tf.stem
        if "__" not in stem:
            continue
        iid = stem.rsplit("__", 1)[1]
        lab = _derive(itemid_to_conclusion.get(iid, ""))
        if lab in (0, 1):
            out[_sanitize_case_id_for_match(stem)] = lab
    return out


_HUDOC_HEADER_RE = re.compile(
    r"^={5,}\s*\n.*?\n={5,}\s*\n+",
    re.DOTALL,
)


def _strip_hudoc_metadata_header(text: str) -> str:
    """Remove the metadata banner written by scrape_turkey_echr.py.

    The scraper injects a banner containing CASE/CONCLUSION/LABEL/HUDOC fields
    at the top of every .txt file, bracketed by lines of '=' characters. Those
    fields contain the literal outcome label, so they must be removed before
    any blinding — otherwise the raw-text baseline trivially copies the answer.
    """
    return _HUDOC_HEADER_RE.sub("", text, count=1)


def load_echr_turkey_raw_texts(local_dir: str = "echr_turkey_cases") -> Dict[str, str]:
    """Read full HUDOC text for each Turkey ECHR case keyed by sanitized stem
    (matches extractor's graph filenames).

    Strips the HUDOC metadata banner (CASE/CONCLUSION/LABEL/HUDOC etc.) before
    returning so the raw baseline doesn't get the answer key for free.
    """
    from pathlib import Path as _Path
    out: Dict[str, str] = {}
    for tf in _Path(local_dir).glob("*.txt"):
        try:
            txt = tf.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = tf.read_text(encoding="latin-1")
        out[_sanitize_case_id_for_match(tf.stem)] = _strip_hudoc_metadata_header(txt)
    return out


def load_echr_labels(name="lex_glue", config="ecthr_b", split="train"):
    """Load ECHR binary violation labels keyed by case ID.

    Case IDs follow the convention used by run_iltur.py's _adapt_case for the
    lex_glue/ecthr_b dataset (which has no explicit ID field): "echr_{idx}".
    Labels are multi-hot vectors of violated articles; we collapse to binary:
    any violation → 1, empty list → 0. Matches _adapt_case's logic exactly so
    case IDs and labels stay in sync between extraction and eval.
    """
    from datasets import load_dataset
    ds = load_dataset(name, config)[split]
    out = {}
    for i, ex in enumerate(ds):
        labels = ex.get("labels") or []
        label = 1 if len(labels) > 0 else 0
        out[f"echr_{i}"] = label
    return out


def load_echr_raw_texts(name="lex_glue", config="ecthr_b", split="train"):
    """Load ECHR raw text keyed by case ID (matches _adapt_case in run_iltur.py).

    lex_glue/ecthr_b stores text as a list of paragraphs under the "text" key;
    we join with newlines to reconstruct the document, identical to how the
    extractor's _adapt_case sees the input.
    """
    from datasets import load_dataset
    ds = load_dataset(name, config)[split]
    out = {}
    for i, ex in enumerate(ds):
        text = ex.get("text")
        if isinstance(text, list):
            text = "\n".join(str(x) for x in text if x is not None)
        elif text is None:
            text = ""
        out[f"echr_{i}"] = str(text)
    return out


# =============================================================================
# CHECKPOINTING
# =============================================================================

def _model_slug(model: str) -> str:
    """Short slug from model name for file naming."""
    m = model.lower()
    if "sonnet" in m: return "sonnet"
    if "opus" in m: return "opus"
    if "haiku" in m: return "haiku"
    if "grok" in m: return m.split("/")[-1].replace(" ", "-")
    if "gpt" in m or m.startswith(("o1", "o3")):
        return m.replace("/", "-").replace(" ", "-").replace(".", "-")[:30]
    return m.replace("/", "-").replace(" ", "-")[:30]


def _checkpoint_path(n_cases: int, no_scrub: bool, model: str = "") -> Path:
    scrub_tag = "_noscrub" if no_scrub else ""
    model_tag = f"_{_model_slug(model)}" if model else ""
    return Path(f"eval_checkpoint_n{n_cases}{scrub_tag}{model_tag}.json")


def save_checkpoint(results: list, config: dict, path: Path):
    """Save intermediate results so we can resume after crashes."""
    data = {"config": config, "completed": results}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)  # atomic rename, works on Windows too


def load_checkpoint(path: Path, config: dict) -> list:
    """Load checkpoint if it exists and config matches."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        # Verify config matches (model, seed, chars must agree)
        saved_cfg = data.get("config", {})
        for key in ("model", "seed", "max_raw_chars", "no_scrub"):
            if saved_cfg.get(key) != config.get(key):
                print(f"  ⚠️  Checkpoint config mismatch on '{key}' — starting fresh")
                return []
        completed = data.get("completed", [])
        print(f"  ✅ Resuming from checkpoint: {len(completed)} cases already done")
        return completed
    except Exception as e:
        print(f"  ⚠️  Checkpoint corrupted ({e}) — starting fresh")
        return []


# =============================================================================
# BLINDING: RAW TEXT
# =============================================================================

# Outcome-revealing patterns in Indian SC judgments.
# These leak the result even in early paragraphs.
_OUTCOME_PATTERNS = [
    # Direct dispositions
    r"(?:appeal|petition|writ|application|suit|complaint|reference|review)\s+"
    r"(?:is|are|was|were|shall\s+be|stands?|be|has\s+been|have\s+been)\s+"
    r"(?:dismissed|allowed|partly\s+allowed|set\s+aside|remanded|rejected|"
    r"granted|refused|disposed\s+of|accepted|decreed|negatived|overruled|"
    r"answered\s+in\s+the\s+(?:affirmative|negative))",

    # "We dismiss / We allow / We hold"
    r"(?:we|court|bench|i)\s+(?:hereby\s+)?(?:dismiss|allow|reject|grant|refuse|"
    r"set\s+aside|remand|uphold|affirm|reverse|modify|quash|restore|hold\s+that|"
    r"are\s+of\s+the\s+(?:view|opinion)\s+that)",

    # "In the result" / "For the foregoing reasons" + any disposition
    r"(?:in\s+the\s+result|for\s+(?:the\s+)?(?:foregoing|above|aforesaid)\s+reasons?|"
    r"accordingly|in\s+(?:the\s+)?(?:light|view)\s+of\s+the\s+above|"
    r"for\s+(?:all\s+)?(?:these|the\s+above)\s+reasons|"
    r"in\s+conclusion|to\s+sum\s+up|summing\s+up)",

    # Order/decree language
    r"(?:ordered?\s+accordingly|(?:the\s+)?(?:order|decree|judgment|conviction|sentence)\s+"
    r"(?:is|shall\s+be|stands?)\s+(?:affirmed|reversed|modified|set\s+aside|upheld|"
    r"restored|quashed|maintained|confirmed))",

    # "The appeal fails" / "The appeal succeeds" / "The appeal has no merit"
    r"(?:appeal|petition|writ|complaint)\s+(?:fails?|succeeds?|is\s+(?:without|with)\s+merit|"
    r"must\s+(?:fail|succeed)|deserves?\s+to\s+be\s+(?:dismissed|allowed)|"
    r"is\s+(?:devoid|bereft)\s+of\s+(?:merit|substance))",

    # Cost language (appears at very end)
    r"(?:no\s+order\s+as\s+to\s+costs?|costs?\s+(?:shall|to)\s+(?:be\s+)?(?:borne|paid)|"
    r"parties?\s+(?:shall|to|will)\s+bear\s+(?:their\s+)?own\s+costs)",

    # "We see no merit" / "We find merit" / "We see no reason"
    r"(?:we|court)\s+(?:see|find|perceive|discern)\s+(?:no\s+)?(?:merit|substance|force|"
    r"reason\s+to\s+(?:interfere|intervene|disturb))",

    # "The conviction is upheld" / "The sentence is reduced"
    r"(?:conviction|sentence|acquittal)\s+(?:is|shall\s+be|stands?)\s+"
    r"(?:upheld|confirmed|maintained|set\s+aside|reversed|reduced|modified|altered)",

    # "We answer the question" / "The question is answered"
    r"(?:we\s+answer|(?:the\s+)?question\s+(?:is|are)\s+(?:hereby\s+)?answered)",

    # Headnote summary phrases
    r"(?:held\s*[-:–]|per\s+curiam\s*[-:–]|the\s+court\s+held\s+that)",

    # ECHR-specific dispositions (gate-free; these patterns simply don't match
    # Indian SC text, so adding them is safe across both jurisdictions).
    # Generalized from a leak audit on Turkey HUDOC: prior patterns missed
    # past-tense GC narrations of Chamber rulings, standalone "violation of
    # Art N" references, and naked voting splits.

    # 1. Any "(no) violation of Article/Art. N" — catches headnote chains,
    #    GC referral narrations, dispositive paragraphs, anywhere it appears.
    r"\b(?:no\s+)?violation\s+of\s+(?:Article|Art\.?)\s*\d+",

    # 2. "there had/has/was been (no) violation" — past and present tense
    r"\bthere\s+(?:had\s+been|has\s+been|was)\s+(?:a\s+|no\s+)?violation\b",

    # 3. Naked voting splits — strongest signal that an outcome is being narrated
    r"\bby\s+(?:a\s+majority\s+of\s+)?\w+\s+votes?\s+to\s+\w+\b",

    # 4. Unanimous / vote-split finding verbs (both tenses)
    r"(?:the\s+court\s+)?(?:unanimously|by\s+\d+\s+votes?\s+to\s+\d+)\s+"
    r"(?:holds|finds|declares|decides|rules|held|found|declared|decided|ruled)",

    # 5. Operative clause marker
    r"FOR\s+THESE\s+REASONS,?\s+THE\s+COURT",

    # 6. Admissibility rulings (both tenses, plus passive)
    r"(?:declares|holds|rules|declared|held|ruled)\s+(?:that\s+)?the\s+(?:application|complaint|request|"
    r"applicant'?s?\s+claim)\s+(?:admissible|inadmissible|manifestly\s+ill[\s-]?founded|"
    r"incompatible|premature)",
    r"(?:application|complaint|claim|remainder).{0,60}(?:was|were|is|has\s+been|have\s+been)\s+"
    r"declared\s+(?:partly\s+)?(?:in)?admissible",

    # 7. Strike-out
    r"(?:strikes?|struck)\s+(?:the\s+(?:application|case)\s+)?out\s+of\s+(?:its\s+)?list",

    # 8. Article 41 just satisfaction (signals merits decided with award)
    r"(?:just\s+satisfaction|article\s+41)\s*[-:–]?",
]

_OUTCOME_RE = re.compile("|".join(_OUTCOME_PATTERNS), re.IGNORECASE)

# Sentence-level words that are almost always dispositive
_DISPOSITIVE_SENTENCE_WORDS = {
    "dismissed", "allowed", "remanded", "set aside", "disposed of",
    "affirmed", "reversed", "upheld", "quashed", "restored",
    "conviction upheld", "conviction set aside", "acquitted",
    "sentence reduced", "sentence modified", "appeal fails",
    "appeal succeeds", "petition granted", "writ issued",
    "decreed", "negatived",
    # ECHR-specific (additive across jurisdictions):
    "violation found", "no violation", "violation of article",
    "no violation of article", "inadmissible", "struck out",
    "manifestly ill-founded", "manifestly ill founded",
}

# Headnote markers in Indian SC judgments
_HEADNOTE_RE = re.compile(
    r"^[\s\S]*?(?:HEAD\s*NOTE|HEADNOTE)\s*[-:–\n]",
    re.IGNORECASE
)

# ECHR Registry-style headnote bullet chains. Modern HUDOC judgments (since ~2018)
# start with lines like:
#   "Art 3 (procedural) • Positive obligation • Ineffective investigation into..."
# These literally narrate the doctrine + outcome and must be stripped before
# sentence-level filtering (they're not sentence-shaped, so _OUTCOME_RE misses them).
_ECHR_BULLET_HEADNOTE_RE = re.compile(
    r"(?:^|\n)[ \t]*Art\s*\d+(?:\s*§\s*\d+)?\s*(?:\([^)]+\))?\s*•[^\n]*(?:\n[ \t]*•[^\n]*)*",
    re.MULTILINE | re.IGNORECASE,
)

# Court reasoning phrases that can appear even in early text
_COURT_REASONING_IN_TEXT_RE = re.compile(
    r"(?:the\s+court\s+(?:held|found|observed|concluded|noted|opined|was\s+of\s+the\s+view)|"
    r"it\s+was\s+(?:held|found|observed|concluded)\s+(?:that|by)|"
    r"(?:we|this\s+court)\s+(?:hold|find|observe|conclude|are\s+of\s+the\s+(?:view|opinion))\s+that|"
    r"the\s+(?:learned\s+)?(?:judge|magistrate|tribunal|high\s+court|sessions?\s+court)\s+"
    r"(?:held|found|observed|concluded|was\s+(?:right|wrong|justified)|erred)|"
    r"(?:rightly|wrongly|correctly|erroneously)\s+(?:held|found|decided|concluded|dismissed|allowed))",
    re.IGNORECASE
)


def blind_raw_text(text: str, max_chars: int = 4000) -> str:
    """Blind a raw judgment text for zero-shot prediction.

    Strategy:
        1. Strip headnotes (Indian SC judgments often start with outcome summaries).
        2. Handle short judgments safely (don't take 85% of a short text).
        3. Take first `max_chars` characters (facts/history come first).
        4. Strip the final 15% of the taken text (court reasoning creeps in).
        5. Strip sentences containing explicit outcome/disposition language.
        6. Filter individual sentences with court-reasoning language.

    This is intentionally GENEROUS to the raw baseline — it gets 3-5x more text
    than the graph summary. If the graph still wins, structure matters.
    """
    if not text:
        return ""

    # Step 0: Strip headnotes — they summarize the outcome at the top
    headnote_match = _HEADNOTE_RE.search(text[:2000])
    if headnote_match:
        text = text[headnote_match.end():]

    # Step 0b: Strip ECHR Registry bullet-headnote chains anywhere in the text.
    # These "Art X • doctrine • outcome cue" lines appear right after JUDGMENT
    # in modern HUDOC judgments and literally narrate the disposition.
    text = _ECHR_BULLET_HEADNOTE_RE.sub("\n", text)

    # Step 1: Handle short judgments
    # If the full text is less than 1.5x our budget, taking max_chars gets almost
    # everything including the outcome. Scale down proportionally.
    total_len = len(text)
    if total_len < max_chars * 1.5:
        effective_max = int(total_len * 0.50)
    elif total_len < max_chars * 2.0:
        effective_max = int(total_len * 0.65)
    else:
        effective_max = max_chars

    # Step 2: Take beginning of judgment (facts/history come first)
    chunk = text[:effective_max]

    # Step 3: Remove last 15% (where reasoning/outcome starts leaking)
    cutoff = int(len(chunk) * 0.85)
    chunk = chunk[:cutoff]

    # Step 4: Sentence-level filtering
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    cleaned = []
    for sent in sentences:
        # Skip sentences with outcome patterns
        if _OUTCOME_RE.search(sent):
            continue

        # Skip short dispositive sentences
        lower = sent.lower().strip()
        if any(dw in lower for dw in _DISPOSITIVE_SENTENCE_WORDS) and len(sent) < 150:
            continue

        # Skip sentences with court reasoning language
        if _COURT_REASONING_IN_TEXT_RE.search(sent):
            continue

        cleaned.append(sent)

    result = " ".join(cleaned).strip()

    # Step 5: Truncate trailing incomplete sentence
    last_period = result.rfind(".")
    if last_period > len(result) * 0.5:
        result = result[:last_period + 1]

    return result


# =============================================================================
# BLINDING: GRAPH SUMMARY
# =============================================================================

# Phrases in fact text that reveal court's findings
_FACT_COURT_LEAK_RE = re.compile(
    r"(?:the\s+court\s+(?:held|found|observed|concluded|noted|opined|directed)|"
    r"it\s+was\s+(?:held|found|observed|concluded)\s+(?:that|by)|"
    r"(?:rightly|wrongly|correctly|erroneously)\s+(?:held|found|decided|concluded)|"
    r"the\s+(?:learned\s+)?(?:judge|magistrate|tribunal|high\s+court)\s+"
    r"(?:held|found|observed|concluded|was\s+(?:right|wrong|justified)|erred)|"
    r"(?:we|this\s+court)\s+(?:hold|find|observe|conclude)\s+that|"
    r"(?:conviction|acquittal|sentence)\s+(?:was|is|has\s+been)\s+"
    r"(?:upheld|set\s+aside|reversed|confirmed|modified)|"
    r"(?:appeal|petition|writ)\s+(?:was|is|has\s+been)\s+"
    r"(?:dismissed|allowed|granted|refused|rejected))",
    re.IGNORECASE
)


def _scrub_fact_text(text: str) -> str:
    """Remove court-finding language from fact text.

    Facts should describe what happened, not how the court ruled on it.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s for s in sentences if not _FACT_COURT_LEAK_RE.search(s)]
    result = " ".join(cleaned).strip()
    if not result and text:
        return text[:100] + "..."
    return result


def build_blinded_graph_summary(graph: dict, no_scrub: bool = False) -> str:
    """Build blinded graph summary — NO holdings, NO court responses, NO outcome.

    If no_scrub=True: minimal blinding. Facts, concepts, precedents get their
    full text. Only holdings, outcome, court_response, and issue answers are
    stripped. This trusts the extractor's schema separation.

    If no_scrub=False (default): aggressive scrubbing. Fact text is regex-filtered,
    concept interpretations excluded, precedent propositions excluded.
    """
    parts = []

    # Facts
    facts = graph.get("facts") or []
    material = [f for f in facts if isinstance(f, dict) and f.get("fact_type") == "material"]
    other = [f for f in facts if isinstance(f, dict) and f.get("fact_type") != "material"]
    selected = (material + other)[:8]
    if selected:
        parts.append("FACTS:")
        for f in selected:
            raw_text = f.get("text", "")[:300]
            text = raw_text if no_scrub else _scrub_fact_text(raw_text)
            ftype = f.get("fact_type", "")
            if text:
                parts.append(f"  [{ftype}] {text}")

    # Legal concepts
    concepts = graph.get("concepts") or []
    central = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "central"]
    supporting = [c for c in concepts if isinstance(c, dict) and c.get("relevance") == "supporting"]
    selected_c = central + supporting[:4]
    if selected_c:
        parts.append("LEGAL CONCEPTS:")
        for c in selected_c:
            cid = c.get("concept_id", "unknown")
            label = c.get("unlisted_label") or cid.replace("UNLISTED_", "").replace("_", " ")
            rel = c.get("relevance", "")
            kind = c.get("kind", "")
            kind_str = f" ({kind})" if kind else ""
            # In no_scrub mode, include interpretation and description
            if no_scrub:
                interp = c.get("interpretation", "")
                desc = c.get("unlisted_description", "")
                extra = interp or desc
                extra_str = f": {extra[:200]}" if extra else ""
            else:
                extra_str = ""
            parts.append(f"  [{rel}]{kind_str} {label}{extra_str}")

    # Issues (questions only — NO answers regardless of mode)
    issues = graph.get("issues") or []
    if issues:
        parts.append("ISSUES BEFORE THE COURT:")
        for iss in issues[:5]:
            if isinstance(iss, dict):
                text = iss.get("text", "")[:250]
                parts.append(f"  - {text}")

    # Arguments — party claims only, NO court actor, NO court_response regardless of mode
    arguments = graph.get("arguments") or []
    pet_args = [a for a in arguments if isinstance(a, dict) and
                a.get("actor") in ("petitioner", "appellant", "complainant", "prosecution")]
    resp_args = [a for a in arguments if isinstance(a, dict) and
                 a.get("actor") in ("respondent", "accused")]

    if pet_args or resp_args:
        parts.append("PARTY ARGUMENTS:")
        for a in pet_args[:4]:
            claim = a.get("claim", "")[:250]
            actor = a.get("actor", "petitioner")
            schemes = a.get("schemes") or []
            scheme_str = f" [{', '.join(str(s) for s in schemes[:2])}]" if schemes else ""
            parts.append(f"  [{actor.upper()}]{scheme_str} {claim}")

        for a in resp_args[:4]:
            claim = a.get("claim", "")[:250]
            actor = a.get("actor", "respondent")
            schemes = a.get("schemes") or []
            scheme_str = f" [{', '.join(str(s) for s in schemes[:2])}]" if schemes else ""
            parts.append(f"  [{actor.upper()}]{scheme_str} {claim}")

    # Precedents
    precedents = graph.get("precedents") or []
    if precedents:
        prec_parts = []
        for pr in precedents[:5]:
            if isinstance(pr, dict):
                name = pr.get("case_name") or pr.get("citation", "")
                if name:
                    if no_scrub:
                        # Include cited_proposition in no_scrub mode
                        prop = pr.get("cited_proposition", "")
                        prop_str = f" — {prop[:150]}" if prop else ""
                        prec_parts.append(f"{name}{prop_str}")
                    else:
                        prec_parts.append(name)
        if prec_parts:
            parts.append(f"CITED PRECEDENTS: {'; '.join(prec_parts)}")

    return "\n".join(parts)


# =============================================================================
# BLINDING SANITY CHECK
# =============================================================================

_SANITY_PATTERNS = re.compile(
    r"(?:dismissed|allowed|set\s+aside|remanded|reversed|affirmed|upheld|quashed|"
    r"conviction\s+(?:upheld|set\s+aside)|appeal\s+(?:fails|succeeds)|"
    r"(?:we|court)\s+(?:hold|find|dismiss|allow|reject)\s+that|"
    r"in\s+the\s+result|for\s+the\s+foregoing\s+reasons|ordered?\s+accordingly)",
    re.IGNORECASE
)


def blinding_sanity_check(text: str, label: str, case_id: str) -> List[str]:
    """Check if blinded text still contains suspicious outcome language."""
    warnings = []
    for match in _SANITY_PATTERNS.finditer(text):
        ctx_start = max(0, match.start() - 30)
        ctx_end = min(len(text), match.end() + 30)
        context = text[ctx_start:ctx_end].replace("\n", " ")
        warnings.append(f"  [{case_id}] {label}: ...{context}...")
    return warnings


# =============================================================================
# PROMPT CONSTRUCTION
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


_JURISDICTION_LABELS = {
    "iltur":    "Indian Supreme Court case",
    "echr":     "European Court of Human Rights case",
    "echr_hf":  "European Court of Human Rights case",
}
# For ECHR the prediction is "violation found" vs "no violation". For IL-TUR
# it's "appeal accepted" vs "rejected". The 0/1 binary semantics are identical
# but a small label-explanation helps mini distinguish.
_PREDICTION_LABEL_HINTS = {
    "iltur":    "1 = appeal accepted, 0 = appeal rejected",
    "echr":     "1 = violation found, 0 = no violation",
    "echr_hf":  "1 = violation found, 0 = no violation",
}


def build_graph_prompt(graph: dict, no_scrub: bool = False,
                       jurisdiction_label: str = "Indian Supreme Court case",
                       prediction_hint: str = "1 = appeal accepted, 0 = appeal rejected") -> str:
    """Prompt for graph-structured zero-shot prediction."""
    summary = build_blinded_graph_summary(graph, no_scrub=no_scrub)
    return (
        f"Predict the outcome of this {jurisdiction_label}.\n"
        f"({prediction_hint})\n"
        "The case has been analyzed into structured components below.\n"
        "Court responses to arguments are NOT shown — predict from the facts, "
        "legal framework, and party arguments alone.\n\n"
        f"{summary}\n\n"
        "Predict: {\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
    )


def build_raw_prompt(text: str,
                     jurisdiction_label: str = "Indian Supreme Court case",
                     prediction_hint: str = "1 = appeal accepted, 0 = appeal rejected") -> str:
    """Prompt for raw-text zero-shot prediction."""
    return (
        f"Predict the outcome of this {jurisdiction_label}.\n"
        f"({prediction_hint})\n"
        "Below is an excerpt from the judgment covering the facts, background, and "
        "party arguments. The court's decision and reasoning have been removed.\n"
        "Predict from the facts and arguments alone.\n\n"
        "--- JUDGMENT EXCERPT ---\n"
        f"{text}\n"
        "--- END EXCERPT ---\n\n"
        "Predict: {\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
    )


# =============================================================================
# LLM CLIENT
# =============================================================================

def _extract_prediction_json(content: str) -> dict:
    """Parse LLM response content into a prediction dict.

    Handles markdown wrapping, reasoning preamble, and nested JSON.
    Returns {"prediction": -1, ...} on parse failure.
    """
    content = content.strip()

    # Strip markdown wrapping
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

    # If content starts with JSON, try direct parse first
    if content.startswith("{"):
        try:
            result = json.loads(content)
            if result.get("prediction") in (0, 1):
                return result
        except json.JSONDecodeError:
            pass

    # Handle reasoning model thinking before JSON — find the JSON block
    # Use a greedy approach that handles nested braces in "reasoning"
    json_start = content.find("{")
    if json_start >= 0:
        depth = 0
        for i in range(json_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[json_start:i + 1]
                    try:
                        result = json.loads(candidate)
                        if result.get("prediction") in (0, 1):
                            return result
                    except json.JSONDecodeError:
                        continue

    # Fallback: original simple regex (no nested braces)
    json_match = re.search(r'\{[^{}]*"prediction"[^{}]*\}', content)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if result.get("prediction") in (0, 1):
                return result
        except json.JSONDecodeError:
            pass

    return {"prediction": -1, "confidence": 0.0,
            "reasoning": f"Failed to parse response: {content[:200]}"}


# Retryable HTTP/API errors (non-deterministic failures worth retrying)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


async def llm_predict(api_key: str, system: str, prompt: str,
                      model: str = "grok-4-1-fast-reasoning",
                      temperature: float = 0.1) -> dict:
    """Call Grok API and parse JSON response. Retries on transient failures."""
    import httpx

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
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

                # Retry on transient HTTP errors
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                result = _extract_prediction_json(content)

                # Retry on parse failures (model sometimes produces garbage)
                if result["prediction"] == -1 and attempt < _MAX_RETRIES - 1:
                    last_error = f"Parse failure: {result['reasoning'][:80]}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue

                return result

        except Exception as e:
            last_error = str(e)[:150]
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
            continue

    return {"prediction": -1, "confidence": 0.0,
            "reasoning": f"Failed after {_MAX_RETRIES} attempts: {last_error}"}


def _is_anthropic_model(model: str) -> bool:
    """Check if the model string refers to an Anthropic/Claude model."""
    return any(tag in model.lower() for tag in ("claude", "sonnet", "haiku", "opus"))


def _is_openai_model(model: str) -> bool:
    """Check if the model string refers to an OpenAI model (gpt-*, o1, o3, etc.)."""
    m = model.lower()
    return any(tag in m for tag in ("gpt-", "gpt4", "openai", "o1", "o3", "o4"))


def _is_openai_reasoning_model(model: str) -> bool:
    """Reasoning-tier OpenAI models (o1, o3, o4-mini, gpt-5*) require
    max_completion_tokens instead of max_tokens. They also produce internal
    reasoning tokens that count against output billing.
    """
    m = model.lower()
    if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return True
    # gpt-5 and onward (gpt-5, gpt-5.1, ..., gpt-5.4, etc.) are reasoning-tier.
    return bool(re.match(r"^gpt-[5-9](?:\.\d+)?", m))


async def llm_predict_openai(api_key: str, system: str, prompt: str,
                              model: str = "gpt-4.1-mini",
                              temperature: float = 0.1,
                              max_output_tokens: int = 1024) -> dict:
    """Call OpenAI chat/completions API and parse JSON response.

    Uses response_format=json_object for server-side JSON enforcement, which
    dramatically reduces parse failures on smaller models like gpt-4.1-mini.

    For reasoning-tier models (o1, o3, o4-mini, gpt-5*) uses max_completion_tokens
    instead of max_tokens. Those models also produce hidden reasoning tokens
    that count against output billing — pass a generous max_output_tokens.
    """
    import httpx

    is_reasoning = _is_openai_reasoning_model(model)
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                }
                if is_reasoning:
                    payload["max_completion_tokens"] = max_output_tokens
                else:
                    payload["max_tokens"] = max_output_tokens
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                )

                # Retry on transient HTTP errors
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                result = _extract_prediction_json(content)

                # Retry on parse failures (rare with response_format=json_object)
                if result["prediction"] == -1 and attempt < _MAX_RETRIES - 1:
                    last_error = f"Parse failure: {result['reasoning'][:80]}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue

                return result

        except Exception as e:
            last_error = str(e)[:150]
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
            continue

    return {"prediction": -1, "confidence": 0.0,
            "reasoning": f"Failed after {_MAX_RETRIES} attempts: {last_error}"}


async def llm_predict_anthropic(api_key: str, system: str, prompt: str,
                                model: str = "claude-sonnet-4-5-20250929",
                                temperature: float = 0.1) -> dict:
    """Call Anthropic Messages API and parse JSON response. Retries on transient failures."""
    import httpx

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "system": system,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": 1024,
                    }
                )

                # Retry on transient HTTP errors
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()

                # Anthropic returns content as a list of blocks
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")

                result = _extract_prediction_json(content)

                # Retry on parse failures
                if result["prediction"] == -1 and attempt < _MAX_RETRIES - 1:
                    last_error = f"Parse failure: {result['reasoning'][:80]}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue

                return result

        except Exception as e:
            last_error = str(e)[:150]
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    ⚠ {last_error}, retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
            continue

    return {"prediction": -1, "confidence": 0.0,
            "reasoning": f"Failed after {_MAX_RETRIES} attempts: {last_error}"}


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

async def run_comparison(
    graph_dir: str,
    n_cases: int = 50,
    max_raw_chars: int = 4000,
    model: str = "grok-4-1-fast-reasoning",
    concurrent: int = 10,
    seed: int = 42,
    no_scrub: bool = False,
    dataset: str = "iltur",
    echr_local_dir: str = "echr_turkey_cases",
):
    # Three-way API selection: Grok (xAI), OpenAI (gpt/o1/o3), or Anthropic (Claude).
    use_anthropic = _is_anthropic_model(model)
    use_openai = _is_openai_model(model)

    if use_anthropic:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set in .env (needed for Claude models)")
            return None
        print(f"  Using Anthropic API (model={model})")
    elif use_openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set in .env (needed for OpenAI/GPT models)")
            return None
        print(f"  Using OpenAI API (model={model})")
    else:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            print("Error: XAI_API_KEY not set in .env")
            return None
        print(f"  Using xAI Grok API (model={model})")

    # Load data
    print("Loading graphs...")
    graphs = iter_graphs(Path(graph_dir))
    print(f"  Loaded {len(graphs)} graphs")

    if dataset == "echr":
        # Default ECHR path: local Turkey HUDOC corpus (full judgments + parsed labels).
        print(f"Loading labels from local Turkey ECHR metadata: {echr_local_dir}/metadata.csv ...")
        labels = load_echr_turkey_labels(echr_local_dir)
        print(f"  {len(labels)} merits cases with binary labels "
              f"(viol={sum(1 for v in labels.values() if v==1)}, "
              f"noviol={sum(1 for v in labels.values() if v==0)})")
        print(f"Loading raw judgment text from: {echr_local_dir}/*.txt "
              f"(stripping HUDOC metadata banner) ...")
        raw_texts = load_echr_turkey_raw_texts(echr_local_dir)
        print(f"  {len(raw_texts)} text files loaded")
    elif dataset == "echr_hf":
        print("Loading labels from lex_glue/ecthr_b (HF)...")
        labels = load_echr_labels()
        print("Loading raw texts from lex_glue/ecthr_b (HF)...")
        raw_texts = load_echr_raw_texts()
    else:
        print("Loading labels from IL-TUR...")
        labels = load_labels_hf()
        print("Loading raw texts from IL-TUR...")
        raw_texts = load_raw_texts_hf()

    jurisdiction_label = _JURISDICTION_LABELS.get(dataset, "court case")
    prediction_hint = _PREDICTION_LABEL_HINTS.get(dataset, "1 = accepted, 0 = rejected")
    print(f"Prompt jurisdiction label: {jurisdiction_label!r}")
    print(f"Prediction label hint:     {prediction_hint!r}")

    # Filter to cases with both graph + label + raw text
    corpus = []
    for case_id, graph in graphs:
        if case_id in labels and case_id in raw_texts:
            corpus.append((case_id, graph, labels[case_id], raw_texts[case_id]))
    print(f"  {len(corpus)} cases with graph + label + raw text")

    n_acc = sum(1 for _, _, l, _ in corpus if l == 1)
    n_rej = sum(1 for _, _, l, _ in corpus if l == 0)
    print(f"  Labels: {n_acc} accepted, {n_rej} rejected")

    # Stratified sample
    rng = np.random.RandomState(seed)
    acc_idx = [i for i, (_, _, l, _) in enumerate(corpus) if l == 1]
    rej_idx = [i for i, (_, _, l, _) in enumerate(corpus) if l == 0]

    if n_cases < len(corpus):
        n_acc_sample = max(1, int(n_cases * len(acc_idx) / len(corpus)))
        n_rej_sample = n_cases - n_acc_sample
        selected = sorted(
            list(rng.choice(acc_idx, min(n_acc_sample, len(acc_idx)), replace=False)) +
            list(rng.choice(rej_idx, min(n_rej_sample, len(rej_idx)), replace=False))
        )
    else:
        selected = list(range(len(corpus)))

    print(f"\nSelected {len(selected)} cases (stratified, seed={seed})")
    sel_acc = sum(1 for i in selected if corpus[i][2] == 1)
    sel_rej = sum(1 for i in selected if corpus[i][2] == 0)
    print(f"  {sel_acc} accepted, {sel_rej} rejected")

    print(f"\nModel: {model}")
    print(f"Graph blinding: {'MINIMAL (no_scrub)' if no_scrub else 'AGGRESSIVE (scrubbed)'}")
    print(f"Raw text budget: {max_raw_chars} chars (pre-stripping)")
    print(f"Concurrent: {concurrent}")

    # Prepare prompts + sanity checks
    eval_cases = []
    graph_char_counts = []
    raw_char_counts = []
    sanity_warnings = []

    for idx in selected:
        case_id, graph, label, raw_text = corpus[idx]
        label_str = "ACC" if label == 1 else "REJ"

        graph_prompt = build_graph_prompt(graph, no_scrub=no_scrub,
                                          jurisdiction_label=jurisdiction_label,
                                          prediction_hint=prediction_hint)
        blinded_raw = blind_raw_text(raw_text, max_chars=max_raw_chars)
        raw_prompt = build_raw_prompt(blinded_raw,
                                      jurisdiction_label=jurisdiction_label,
                                      prediction_hint=prediction_hint)

        # Sanity check both blinded outputs
        graph_summary = build_blinded_graph_summary(graph, no_scrub=no_scrub)
        sanity_warnings.extend(
            blinding_sanity_check(graph_summary, f"GRAPH-{label_str}", case_id))
        sanity_warnings.extend(
            blinding_sanity_check(blinded_raw, f"RAW-{label_str}", case_id))

        eval_cases.append({
            "case_id": case_id,
            "label": label,
            "graph_prompt": graph_prompt,
            "raw_prompt": raw_prompt,
            "raw_text_len": len(raw_text),
            "blinded_raw_len": len(blinded_raw),
        })

        graph_char_counts.append(len(graph_prompt))
        raw_char_counts.append(len(raw_prompt))

    # Print sanity check results
    if sanity_warnings:
        print(f"\n⚠️  BLINDING SANITY CHECK: {len(sanity_warnings)} warnings")
        for w in sanity_warnings[:20]:
            print(w)
        if len(sanity_warnings) > 20:
            print(f"  ... and {len(sanity_warnings) - 20} more")
    else:
        print(f"\n✅ BLINDING SANITY CHECK: Clean — no outcome language detected")

    short_cases = [c for c in eval_cases if c["raw_text_len"] < max_raw_chars * 1.5]
    if short_cases:
        print(f"\n📏 Short judgment safety: {len(short_cases)}/{len(eval_cases)} cases "
              f"had text < {max_raw_chars * 1.5:.0f} chars — budget reduced to avoid outcome")

    print(f"\nPrompt sizes:")
    print(f"  Graph: {np.mean(graph_char_counts):.0f} chars avg "
          f"(min={np.min(graph_char_counts)}, max={np.max(graph_char_counts)})")
    print(f"  Raw:   {np.mean(raw_char_counts):.0f} chars avg "
          f"(min={np.min(raw_char_counts)}, max={np.max(raw_char_counts)})")
    print(f"  Raw/Graph ratio: {np.mean(raw_char_counts)/np.mean(graph_char_counts):.1f}x")

    # Run predictions (with checkpointing)
    print(f"\n{'='*80}")
    print("RUNNING PREDICTIONS")
    print(f"{'='*80}")

    ckpt_config = {
        "model": model, "seed": seed,
        "max_raw_chars": max_raw_chars, "no_scrub": no_scrub,
    }
    ckpt_path = _checkpoint_path(n_cases, no_scrub, model)
    completed = load_checkpoint(ckpt_path, ckpt_config)
    completed_ids = {r["case_id"] for r in completed}
    remaining = [c for c in eval_cases if c["case_id"] not in completed_ids]

    if remaining:
        print(f"  {len(remaining)} cases to run ({len(completed)} already done)")
    else:
        print(f"  All {len(completed)} cases already completed — skipping API calls")

    semaphore = asyncio.Semaphore(concurrent)
    results = list(completed)
    errors = {"graph": 0, "raw": 0}
    t0 = time.time()

    async def predict_pair(case: dict, case_num: int):
        """Run both graph and raw prediction for one case."""
        # Reasoning-tier OpenAI models burn internal reasoning tokens that count
        # against the output budget — bump for those so the JSON answer fits.
        extra_kwargs = {}
        if use_openai and _is_openai_reasoning_model(model):
            extra_kwargs["max_output_tokens"] = 4096
        if use_anthropic:
            predict_fn = llm_predict_anthropic
        elif use_openai:
            predict_fn = llm_predict_openai
        else:
            predict_fn = llm_predict
        async with semaphore:
            graph_result = await predict_fn(api_key, _SYSTEM_PROMPT,
                                             case["graph_prompt"], model=model, **extra_kwargs)
            raw_result = await predict_fn(api_key, _SYSTEM_PROMPT,
                                           case["raw_prompt"], model=model, **extra_kwargs)

        gp = graph_result.get("prediction", -1)
        rp = raw_result.get("prediction", -1)

        if gp not in (0, 1):
            errors["graph"] += 1
        if rp not in (0, 1):
            errors["raw"] += 1

        label = case["label"]
        label_str = "ACC" if label == 1 else "REJ"

        g_correct = gp == label
        r_correct = rp == label
        g_str = "✓" if g_correct else "✗"
        r_str = "✓" if r_correct else "✗"
        gp_str = "ACC" if gp == 1 else ("REJ" if gp == 0 else "ERR")
        rp_str = "ACC" if rp == 1 else ("REJ" if rp == 0 else "ERR")

        if g_correct and r_correct:
            tag = "both✓"
        elif g_correct and not r_correct:
            tag = "GRAPH+"
        elif not g_correct and r_correct:
            tag = "RAW+"
        else:
            tag = "both✗"

        print(f"  [{case_num:3d}] {case['case_id']:12s} TRUE={label_str}  "
              f"graph={g_str}{gp_str} raw={r_str}{rp_str}  "
              f"gconf={graph_result.get('confidence', 0):.2f} "
              f"rconf={raw_result.get('confidence', 0):.2f}  {tag}")

        return {
            "case_id": case["case_id"],
            "true_label": label,
            "graph_pred": gp,
            "graph_conf": graph_result.get("confidence", 0),
            "graph_reasoning": graph_result.get("reasoning", ""),
            "raw_pred": rp,
            "raw_conf": raw_result.get("confidence", 0),
            "raw_reasoning": raw_result.get("reasoning", ""),
        }

    # Process remaining cases in batches with checkpoint after each batch
    batch_num = 0
    for batch_start in range(0, len(remaining), concurrent):
        batch = remaining[batch_start:batch_start + concurrent]
        global_offset = len(completed) + batch_start
        tasks = [
            asyncio.create_task(predict_pair(c, global_offset + i + 1))
            for i, c in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # Checkpoint after every batch
        batch_num += 1
        save_checkpoint(results, ckpt_config, ckpt_path)
        done = len(results)
        total = len(eval_cases)
        print(f"  --- checkpoint saved: {done}/{total} cases "
              f"({done/total*100:.0f}%) ---")

    elapsed = time.time() - t0

    # Clean up checkpoint on successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  ✅ Checkpoint cleaned up (run complete)")

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================

    valid = [r for r in results if r["graph_pred"] in (0, 1) and r["raw_pred"] in (0, 1)]
    n = len(valid)

    if n == 0:
        print("\nNo valid results!")
        return None

    graph_correct = sum(1 for r in valid if r["graph_pred"] == r["true_label"])
    raw_correct = sum(1 for r in valid if r["raw_pred"] == r["true_label"])

    graph_acc = graph_correct / n
    raw_acc = raw_correct / n

    acc_cases = [r for r in valid if r["true_label"] == 1]
    rej_cases = [r for r in valid if r["true_label"] == 0]

    graph_acc_on_acc = sum(1 for r in acc_cases if r["graph_pred"] == 1) / max(len(acc_cases), 1)
    graph_acc_on_rej = sum(1 for r in rej_cases if r["graph_pred"] == 0) / max(len(rej_cases), 1)
    raw_acc_on_acc = sum(1 for r in acc_cases if r["raw_pred"] == 1) / max(len(acc_cases), 1)
    raw_acc_on_rej = sum(1 for r in rej_cases if r["raw_pred"] == 0) / max(len(rej_cases), 1)

    def compute_f1(predictions, labels):
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    graph_preds = [r["graph_pred"] for r in valid]
    raw_preds = [r["raw_pred"] for r in valid]
    true_labels = [r["true_label"] for r in valid]

    graph_f1_info = compute_f1(graph_preds, true_labels)
    raw_f1_info = compute_f1(raw_preds, true_labels)

    # Macro F1: average of per-class F1 scores
    def compute_macro_f1(predictions, labels):
        """Macro F1 = mean(F1_class0, F1_class1)."""
        # F1 for class 1 (accepted = positive)
        f1_cls1 = compute_f1(predictions, labels)
        # F1 for class 0 (rejected = positive): flip labels and preds
        flipped_preds = [1 - p for p in predictions]
        flipped_labels = [1 - l for l in labels]
        f1_cls0 = compute_f1(flipped_preds, flipped_labels)
        return (f1_cls1["f1"] + f1_cls0["f1"]) / 2

    graph_macro_f1 = compute_macro_f1(graph_preds, true_labels)
    raw_macro_f1 = compute_macro_f1(raw_preds, true_labels)

    # Cohen's kappa between graph and raw predictions
    def cohens_kappa(pred_a, pred_b):
        """Inter-rater agreement between two prediction sets."""
        n_total = len(pred_a)
        if n_total == 0:
            return 0.0
        agree = sum(1 for a, b in zip(pred_a, pred_b) if a == b)
        p_o = agree / n_total
        # Expected agreement by chance
        a1 = sum(pred_a) / n_total
        b1 = sum(pred_b) / n_total
        p_e = a1 * b1 + (1 - a1) * (1 - b1)
        if p_e == 1.0:
            return 1.0
        return (p_o - p_e) / (1 - p_e)

    kappa_graph_raw = cohens_kappa(graph_preds, raw_preds)

    # Brier score: mean squared error of confidence-weighted predictions
    def brier_score(predictions, confidences, labels):
        """Brier score = mean( (prob_assigned_to_true_class - 1)^2 ).
        Lower is better. Measures both calibration and discrimination."""
        scores = []
        for pred, conf, label in zip(predictions, confidences, labels):
            if pred == label:
                # Confidence was assigned to the correct class
                prob_true = conf
            else:
                # Confidence was assigned to the wrong class
                prob_true = 1.0 - conf
            scores.append((1.0 - prob_true) ** 2)
        return np.mean(scores) if scores else 0.0

    graph_confs = [r["graph_conf"] for r in valid]
    raw_confs = [r["raw_conf"] for r in valid]

    graph_brier = brier_score(graph_preds, graph_confs, true_labels)
    raw_brier = brier_score(raw_preds, raw_confs, true_labels)

    both_correct = sum(1 for r in valid if r["graph_pred"] == r["true_label"]
                       and r["raw_pred"] == r["true_label"])
    graph_only = sum(1 for r in valid if r["graph_pred"] == r["true_label"]
                     and r["raw_pred"] != r["true_label"])
    raw_only = sum(1 for r in valid if r["graph_pred"] != r["true_label"]
                   and r["raw_pred"] == r["true_label"])
    both_wrong = sum(1 for r in valid if r["graph_pred"] != r["true_label"]
                     and r["raw_pred"] != r["true_label"])

    graph_conf_correct = np.mean([r["graph_conf"] for r in valid
                                  if r["graph_pred"] == r["true_label"]] or [0])
    graph_conf_wrong = np.mean([r["graph_conf"] for r in valid
                                if r["graph_pred"] != r["true_label"]] or [0])
    raw_conf_correct = np.mean([r["raw_conf"] for r in valid
                                if r["raw_pred"] == r["true_label"]] or [0])
    raw_conf_wrong = np.mean([r["raw_conf"] for r in valid
                              if r["raw_pred"] != r["true_label"]] or [0])

    # McNemar's test
    b, c = graph_only, raw_only
    if b + c > 0:
        mcnemar_chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        mcnemar_sig = "p<0.05" if mcnemar_chi2 > 3.84 else "n.s."
    else:
        mcnemar_chi2 = 0.0
        mcnemar_sig = "n.s. (no discordant pairs)"

    # Bootstrap CI
    n_boot = 10000
    rng_boot = np.random.RandomState(seed + 1)
    graph_correct_arr = np.array([1 if r["graph_pred"] == r["true_label"] else 0 for r in valid])
    raw_correct_arr = np.array([1 if r["raw_pred"] == r["true_label"] else 0 for r in valid])
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng_boot.choice(n, n, replace=True)
        boot_diffs.append(float(graph_correct_arr[idx].mean() - raw_correct_arr[idx].mean()))
    boot_diffs = np.array(boot_diffs)
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================

    print(f"\n\n{'='*80}")
    print(f"  RESULTS: GRAPH-STRUCTURED vs RAW-TEXT ZERO-SHOT PREDICTION")
    print(f"{'='*80}")

    print(f"\n  {'Metric':<30} {'Graph':>10} {'Raw':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<30} {graph_acc:>10.3f} {raw_acc:>10.3f} {graph_acc - raw_acc:>+10.3f}")
    print(f"  {'F1':<30} {graph_f1_info['f1']:>10.3f} {raw_f1_info['f1']:>10.3f} "
          f"{graph_f1_info['f1'] - raw_f1_info['f1']:>+10.3f}")
    print(f"  {'Precision':<30} {graph_f1_info['precision']:>10.3f} {raw_f1_info['precision']:>10.3f} "
          f"{graph_f1_info['precision'] - raw_f1_info['precision']:>+10.3f}")
    print(f"  {'Recall':<30} {graph_f1_info['recall']:>10.3f} {raw_f1_info['recall']:>10.3f} "
          f"{graph_f1_info['recall'] - raw_f1_info['recall']:>+10.3f}")
    print(f"  {'Macro F1':<30} {graph_macro_f1:>10.3f} {raw_macro_f1:>10.3f} "
          f"{graph_macro_f1 - raw_macro_f1:>+10.3f}")
    print(f"  {'Brier Score (↓ better)':<30} {graph_brier:>10.3f} {raw_brier:>10.3f} "
          f"{graph_brier - raw_brier:>+10.3f}")

    print(f"\n  {'Per-class accuracy:':<30}")
    print(f"  {'  Accepted cases':<30} {graph_acc_on_acc:>10.3f} {raw_acc_on_acc:>10.3f} "
          f"{graph_acc_on_acc - raw_acc_on_acc:>+10.3f}")
    print(f"  {'  Rejected cases':<30} {graph_acc_on_rej:>10.3f} {raw_acc_on_rej:>10.3f} "
          f"{graph_acc_on_rej - raw_acc_on_rej:>+10.3f}")

    print(f"\n  {'Confidence (correct)':<30} {graph_conf_correct:>10.3f} {raw_conf_correct:>10.3f}")
    print(f"  {'Confidence (wrong)':<30} {graph_conf_wrong:>10.3f} {raw_conf_wrong:>10.3f}")
    print(f"  {'Conf. calibration gap':<30} "
          f"{graph_conf_correct - graph_conf_wrong:>10.3f} "
          f"{raw_conf_correct - raw_conf_wrong:>10.3f}")

    print(f"\n  AGREEMENT MATRIX:")
    print(f"  {'':>20} {'Raw ✓':>10} {'Raw ✗':>10}")
    print(f"  {'Graph ✓':<20} {both_correct:>10} {graph_only:>10}")
    print(f"  {'Graph ✗':<20} {raw_only:>10} {both_wrong:>10}")

    print(f"\n  Graph wins (graph✓ raw✗):  {graph_only}")
    print(f"  Raw wins   (graph✗ raw✓):  {raw_only}")
    print(f"  Both correct:              {both_correct}")
    print(f"  Both wrong:                {both_wrong}")

    print(f"\n  STATISTICAL TESTS:")
    print(f"  McNemar's χ² = {mcnemar_chi2:.3f}  ({mcnemar_sig})")
    print(f"  Bootstrap 95% CI for accuracy difference: [{ci_low:+.3f}, {ci_high:+.3f}]")
    if ci_low > 0:
        print(f"  → Graph significantly better (CI excludes 0)")
    elif ci_high < 0:
        print(f"  → Raw significantly better (CI excludes 0)")
    else:
        print(f"  → Difference not significant at 95% (CI includes 0)")

    print(f"\n  Cohen's κ (graph vs raw agreement): {kappa_graph_raw:.3f}")
    if kappa_graph_raw < 0.20:
        print(f"  → Slight agreement — methods behave very differently")
    elif kappa_graph_raw < 0.40:
        print(f"  → Fair agreement")
    elif kappa_graph_raw < 0.60:
        print(f"  → Moderate agreement")
    elif kappa_graph_raw < 0.80:
        print(f"  → Substantial agreement")
    else:
        print(f"  → Near-perfect agreement")

    print(f"\n  Brier Score: graph={graph_brier:.4f}, raw={raw_brier:.4f} "
          f"(Δ={graph_brier - raw_brier:+.4f}, {'graph' if graph_brier < raw_brier else 'raw'} better)")

    print(f"\n  n={n}, errors: graph={errors['graph']}, raw={errors['raw']}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(n, 1):.1f}s per case pair)")
    print(f"  Model: {model}")
    print(f"  Prompt ratio (raw/graph): {np.mean(raw_char_counts)/np.mean(graph_char_counts):.1f}x")
    if sanity_warnings:
        print(f"  ⚠️  Blinding warnings: {len(sanity_warnings)} (check above)")
    print(f"{'='*80}")

    # ==========================================================================
    # DISCORDANT CASES
    # ==========================================================================

    discordant = [r for r in valid if (r["graph_pred"] == r["true_label"]) !=
                  (r["raw_pred"] == r["true_label"])]

    if discordant:
        print(f"\n\n{'='*80}")
        print(f"  DISCORDANT CASES (where methods disagree)")
        print(f"{'='*80}")

        for r in discordant:
            label_str = "ACC" if r["true_label"] == 1 else "REJ"
            g_correct = r["graph_pred"] == r["true_label"]
            winner = "GRAPH" if g_correct else "RAW"
            gp_str = "ACC" if r["graph_pred"] == 1 else "REJ"
            rp_str = "ACC" if r["raw_pred"] == 1 else "REJ"

            print(f"\n  {r['case_id']}  TRUE={label_str}  → {winner} wins")
            print(f"    Graph: {gp_str} (conf={r['graph_conf']:.2f})")
            print(f"      {r['graph_reasoning'][:200]}")
            print(f"    Raw:   {rp_str} (conf={r['raw_conf']:.2f})")
            print(f"      {r['raw_reasoning'][:200]}")

    # Save full results
    output = {
        "config": {
            "model": model,
            "n_cases": n,
            "max_raw_chars": max_raw_chars,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blinding_warnings": len(sanity_warnings),
        },
        "summary": {
            "graph_accuracy": round(graph_acc, 4),
            "raw_accuracy": round(raw_acc, 4),
            "accuracy_delta": round(graph_acc - raw_acc, 4),
            "graph_f1": round(graph_f1_info["f1"], 4),
            "raw_f1": round(raw_f1_info["f1"], 4),
            "graph_macro_f1": round(graph_macro_f1, 4),
            "raw_macro_f1": round(raw_macro_f1, 4),
            "graph_precision": round(graph_f1_info["precision"], 4),
            "raw_precision": round(raw_f1_info["precision"], 4),
            "graph_recall": round(graph_f1_info["recall"], 4),
            "raw_recall": round(raw_f1_info["recall"], 4),
            "graph_brier": round(graph_brier, 4),
            "raw_brier": round(raw_brier, 4),
            "cohens_kappa_graph_raw": round(kappa_graph_raw, 4),
            "mcnemar_chi2": round(mcnemar_chi2, 4),
            "mcnemar_sig": mcnemar_sig,
            "bootstrap_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "graph_only_wins": graph_only,
            "raw_only_wins": raw_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "cases": valid,
    }

    scrub_tag = "_noscrub" if no_scrub else ""
    model_tag = f"_{_model_slug(model)}" if model else ""
    out_path = Path(f"graph_vs_raw_n{n}{scrub_tag}{model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved to {out_path}")

    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Graph-structured vs Raw-text zero-shot prediction comparison"
    )
    p.add_argument("--graph_dir", required=True, help="Directory with extracted graph JSONs")
    p.add_argument("--n", type=int, default=50, help="Number of cases to evaluate")
    p.add_argument("--k_chars", type=int, default=4000,
                   help="Max chars of raw text to use (pre-stripping)")
    p.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    p.add_argument("--concurrent", type=int, default=10, help="Max concurrent API calls")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--no_scrub", action="store_true",
                   help="Minimal blinding: strip holdings/outcome/court_response only. "
                        "No regex scrubbing of fact text or concept fields.")
    p.add_argument("--dataset", choices=["iltur", "echr", "echr_hf"], default="iltur",
                   help="Which dataset's labels and raw texts to load. "
                        "'iltur' (default) → Exploration-Lab/IL-TUR cjpe. "
                        "'echr' → local Turkey HUDOC corpus (full judgments). "
                        "'echr_hf' → lex_glue ecthr_b (facts-only).")
    p.add_argument("--echr_local_dir", default="echr_turkey_cases",
                   help="Path to local Turkey HUDOC directory (for --dataset echr).")

    args = p.parse_args()

    asyncio.run(run_comparison(
        graph_dir=args.graph_dir,
        n_cases=args.n,
        max_raw_chars=args.k_chars,
        model=args.model,
        concurrent=args.concurrent,
        seed=args.seed,
        no_scrub=args.no_scrub,
        dataset=args.dataset,
        echr_local_dir=args.echr_local_dir,
    ))


if __name__ == "__main__":
    main()