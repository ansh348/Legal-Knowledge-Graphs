#!/usr/bin/env python3
"""
eval_graph_vs_structured.py — Head-to-head: Graph vs Structured (non-graph) prediction.

Both methods are blinded — the LLM never sees how the court ruled.
The difference is how the case is *presented*:
    A) GRAPH:      Ontology-driven extraction (facts, concepts, issues, party arguments)
                   with type labels, relevance scores, explicit issue framing.
                   Holdings, court_response, outcome, issue answers ALL stripped.
    B) STRUCTURED: Single-pass LLM structuring (facts, issues, party arguments,
                   court_reasoning, precedents, statutes).
                   Holdings, outcome, court_reasoning, key_quotes ALL stripped.

Both use NO regex scrubbing of fact text — the comparison isolates
the effect of the extraction *architecture*, not the scrubbing strategy.

Same LLM, same system prompt, same cases, same seed.

Usage:
    python eval_graph_vs_structured.py --n 50
    python eval_graph_vs_structured.py --n 100 --concurrent 10 --model grok-4-1-fast-reasoning
    python eval_graph_vs_structured.py --n 200 --model claude-sonnet-4-5-20250929

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


# =============================================================================
# CONFIG
# =============================================================================

GRAPH_DIR = Path("iltur_graphs")
STRUCT_DIR = Path("structured-nongraph-cases")
COMMONS_FILE = Path("commons.json")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_both_extractions(common_ids: list[str]) -> list[tuple[str, dict, dict]]:
    """Load graph + structured extraction for each common case ID."""
    pairs = []
    for cid in common_ids:
        gpath = GRAPH_DIR / f"{cid}.json"
        spath = STRUCT_DIR / f"{cid}.json"
        if not gpath.exists() or not spath.exists():
            continue
        try:
            graph = load_json(gpath)
            struct = load_json(spath)
            pairs.append((cid, graph, struct))
        except Exception:
            continue
    return pairs


def load_labels_hf(name="Exploration-Lab/IL-TUR", config="cjpe", split="single_train"):
    from datasets import load_dataset
    return {str(ex["id"]): int(ex["label"]) for ex in load_dataset(name, config)[split]}


# =============================================================================
# CHECKPOINTING
# =============================================================================

def _model_slug(model: str) -> str:
    m = model.lower()
    if "sonnet" in m: return "sonnet"
    if "opus" in m: return "opus"
    if "haiku" in m: return "haiku"
    if "grok" in m: return m.split("/")[-1].replace(" ", "-")
    return m.replace("/", "-").replace(" ", "-")[:30]


def _checkpoint_path(n_cases: int, model: str = "") -> Path:
    model_tag = f"_{_model_slug(model)}" if model else ""
    return Path(f"eval_gvs_checkpoint_n{n_cases}{model_tag}.json")


def save_checkpoint(results: list, config: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"config": config, "completed": results}, f)
    tmp.replace(path)


def load_checkpoint(path: Path, config: dict) -> list:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        saved_cfg = data.get("config", {})
        for key in ("model", "seed"):
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
# BLINDING: GRAPH SUMMARY (no scrub — but strip holdings/outcome/court_response)
# =============================================================================

def build_blinded_graph_summary(graph: dict) -> str:
    """Build blinded graph summary — no holdings, no outcome, no court_response.

    No regex scrubbing of fact text. Concepts get full text including
    interpretation. Precedent names included but treatment excluded.
    This is the 'no_scrub' philosophy — trust schema separation.
    """
    parts = []

    # Facts — full text, no scrubbing
    facts = graph.get("facts") or []
    material = [f for f in facts if isinstance(f, dict) and f.get("fact_type") == "material"]
    other = [f for f in facts if isinstance(f, dict) and f.get("fact_type") != "material"]
    selected = (material + other)[:12]
    if selected:
        parts.append("FACTS:")
        for f in selected:
            text = f.get("text", "")[:300]
            ftype = f.get("fact_type", "")
            if text:
                parts.append(f"  [{ftype}] {text}")

    # Legal concepts — full text including interpretation
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
            interp = c.get("interpretation", "")
            desc = c.get("unlisted_description", "")
            extra = interp or desc
            extra_str = f": {extra[:200]}" if extra else ""
            parts.append(f"  [{rel}]{kind_str} {label}{extra_str}")

    # Issues — questions only, NO answers
    issues = graph.get("issues") or []
    if issues:
        parts.append("ISSUES BEFORE THE COURT:")
        for iss in issues[:6]:
            if isinstance(iss, dict):
                text = iss.get("text", "")[:250]
                parts.append(f"  - {text}")

    # Arguments — party claims only, NO court actor, NO court_response
    arguments = graph.get("arguments") or []
    pet_args = [a for a in arguments if isinstance(a, dict) and
                a.get("actor") in ("petitioner", "appellant", "complainant", "prosecution")]
    resp_args = [a for a in arguments if isinstance(a, dict) and
                 a.get("actor") in ("respondent", "accused")]

    if pet_args or resp_args:
        parts.append("PARTY ARGUMENTS:")
        for a in pet_args[:5]:
            claim = a.get("claim", "")[:250]
            scheme = a.get("scheme", "")
            scheme_str = f" [{scheme}]" if scheme else ""
            if claim:
                parts.append(f"  Petitioner{scheme_str}: {claim}")
        for a in resp_args[:5]:
            claim = a.get("claim", "")[:250]
            scheme = a.get("scheme", "")
            scheme_str = f" [{scheme}]" if scheme else ""
            if claim:
                parts.append(f"  Respondent{scheme_str}: {claim}")

    # Precedents — names only, NO treatment (followed/distinguished/overruled is outcome-leaking)
    precedents = graph.get("precedents") or []
    if precedents:
        prec_names = []
        for p in precedents[:8]:
            if isinstance(p, dict):
                name = p.get("case_name", "")
                if name:
                    prec_names.append(name)
        if prec_names:
            parts.append(f"CITED PRECEDENTS: {'; '.join(prec_names)}")

    return "\n".join(parts)


# =============================================================================
# BLINDING: STRUCTURED (non-graph) SUMMARY
# =============================================================================

# Outcome-leaking patterns to catch in fact/argument text
_OUTCOME_LEAK_RE = re.compile(
    r"(?:appeal|petition|writ|application)\s+"
    r"(?:is|are|was|were|shall\s+be|stands?)\s+"
    r"(?:dismissed|allowed|partly\s+allowed|set\s+aside|remanded|rejected|"
    r"granted|refused|disposed\s+of|accepted)",
    re.IGNORECASE
)


def build_blinded_structured_summary(struct: dict) -> str:
    """Build blinded structured summary — no outcome, no holdings, no court_reasoning.

    Strips:
        - outcome (disposition, summary, relief, costs)
        - holdings (court's legal determinations)
        - court_reasoning (court's own analysis)
        - key_quotes from court (may reveal reasoning)

    Keeps:
        - metadata (case name, court, year)
        - facts (full text)
        - legal_issues (questions only)
        - petitioner_arguments (full text + legal basis)
        - respondent_arguments (full text + legal basis)
        - precedents_cited (names + citation, NO treatment)
        - statutes_cited (names + sections)
    """
    parts = []

    # Metadata header
    meta = struct.get("metadata") or {}
    case_name = meta.get("case_name", "")
    court = meta.get("court", "")
    year = meta.get("case_year", "")
    if case_name:
        header = case_name
        if court:
            header += f" ({court}"
            if year:
                header += f", {year}"
            header += ")"
        parts.append(f"CASE: {header}")

    # Facts — full text, no scrubbing, but filter any that leak outcome
    facts = struct.get("facts") or []
    if facts:
        parts.append("FACTS:")
        for f in facts[:15]:
            if isinstance(f, dict):
                text = f.get("text", "")[:300]
                ftype = f.get("type", "")
                source = f.get("source", "")
                # Skip facts that explicitly state the outcome
                if _OUTCOME_LEAK_RE.search(text):
                    continue
                source_str = f" ({source})" if source else ""
                if text:
                    parts.append(f"  [{ftype}]{source_str} {text}")

    # Legal issues — questions only
    issues = struct.get("legal_issues") or []
    if issues:
        parts.append("ISSUES BEFORE THE COURT:")
        for iss in issues[:6]:
            if isinstance(iss, dict):
                text = iss.get("text", "")[:250]
                parts.append(f"  - {text}")

    # Petitioner arguments — full text + legal basis
    pet_args = struct.get("petitioner_arguments") or []
    if pet_args:
        parts.append("PETITIONER ARGUMENTS:")
        for a in pet_args[:6]:
            if isinstance(a, dict):
                text = a.get("text", "")[:300]
                basis = a.get("legal_basis", "")
                if _OUTCOME_LEAK_RE.search(text):
                    continue
                basis_str = f" [Basis: {basis}]" if basis else ""
                if text:
                    parts.append(f"  - {text}{basis_str}")

    # Respondent arguments
    resp_args = struct.get("respondent_arguments") or []
    if resp_args:
        parts.append("RESPONDENT ARGUMENTS:")
        for a in resp_args[:6]:
            if isinstance(a, dict):
                text = a.get("text", "")[:300]
                basis = a.get("legal_basis", "")
                if _OUTCOME_LEAK_RE.search(text):
                    continue
                basis_str = f" [Basis: {basis}]" if basis else ""
                if text:
                    parts.append(f"  - {text}{basis_str}")

    # Precedents — name + citation only, NO treatment
    precs = struct.get("precedents_cited") or []
    if precs:
        prec_strs = []
        for p in precs[:8]:
            if isinstance(p, dict):
                name = p.get("case_name", "")
                cite = p.get("citation", "")
                if name:
                    s = name
                    if cite:
                        s += f" ({cite})"
                    prec_strs.append(s)
        if prec_strs:
            parts.append(f"CITED PRECEDENTS: {'; '.join(prec_strs)}")

    # Statutes — names + sections (these don't leak outcome)
    stats = struct.get("statutes_cited") or []
    if stats:
        stat_strs = []
        for s in stats[:6]:
            if isinstance(s, dict):
                name = s.get("name", "")
                sections = s.get("sections") or []
                if name:
                    sec_str = f" ({', '.join(sections[:4])})" if sections else ""
                    stat_strs.append(f"{name}{sec_str}")
        if stat_strs:
            parts.append(f"STATUTES: {'; '.join(stat_strs)}")

    # key_quotes — ONLY from counsel, never from court
    quotes = struct.get("key_quotes") or []
    counsel_quotes = [q for q in quotes if isinstance(q, dict)
                      and q.get("speaker") in ("petitioner_counsel", "respondent_counsel")]
    if counsel_quotes:
        parts.append("KEY COUNSEL QUOTES:")
        for q in counsel_quotes[:3]:
            text = q.get("text", "")[:200]
            speaker = q.get("speaker", "")
            if text:
                parts.append(f"  [{speaker}] \"{text}\"")

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


def blinding_sanity_check(text: str, label: str, case_id: str) -> list[str]:
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


def build_graph_prompt(graph: dict) -> str:
    summary = build_blinded_graph_summary(graph)
    return (
        "Predict the outcome of this Indian Supreme Court case.\n"
        "The case has been analyzed into a structured legal reasoning graph.\n"
        "Court holdings and outcome are NOT shown — predict from the facts, "
        "legal framework, and party arguments alone.\n\n"
        f"{summary}\n\n"
        "Predict: {{\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    )


def build_structured_prompt(struct: dict) -> str:
    summary = build_blinded_structured_summary(struct)
    return (
        "Predict the outcome of this Indian Supreme Court case.\n"
        "The case has been organized into structured sections by an AI.\n"
        "The court's holdings, reasoning, and outcome have been removed — "
        "predict from the facts, issues, and party arguments alone.\n\n"
        f"{summary}\n\n"
        "Predict: {{\"prediction\": 0 or 1, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    )


# =============================================================================
# LLM CLIENT
# =============================================================================

def _extract_prediction_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
    if content.startswith("{"):
        try:
            result = json.loads(content)
            if result.get("prediction") in (0, 1):
                return result
        except json.JSONDecodeError:
            pass
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
    json_match = re.search(r'\{[^{}]*"prediction"[^{}]*\}', content)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if result.get("prediction") in (0, 1):
                return result
        except json.JSONDecodeError:
            pass
    return {"prediction": -1, "confidence": 0.0,
            "reasoning": f"Failed to parse: {content[:200]}"}


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0


def _is_anthropic_model(model: str) -> bool:
    return any(tag in model.lower() for tag in ("claude", "sonnet", "haiku", "opus"))


async def llm_predict(api_key: str, system: str, prompt: str,
                      model: str = "grok-4-1-fast-reasoning",
                      temperature: float = 0.1) -> dict:
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
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s ({attempt+1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                result = _extract_prediction_json(content)
                if result["prediction"] == -1 and attempt < _MAX_RETRIES - 1:
                    last_error = f"Parse failure: {result['reasoning'][:80]}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s ({attempt+1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue
                return result
        except Exception as e:
            last_error = str(e)[:150]
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    ⚠ {last_error}, retrying in {delay:.0f}s ({attempt+1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
            continue
    return {"prediction": -1, "confidence": 0.0,
            "reasoning": f"Failed after {_MAX_RETRIES} attempts: {last_error}"}


async def llm_predict_anthropic(api_key: str, system: str, prompt: str,
                                model: str = "claude-sonnet-4-5-20250929",
                                temperature: float = 0.1) -> dict:
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
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 1024,
                    }
                )
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s ({attempt+1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                data = response.json()
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                result = _extract_prediction_json(content)
                if result["prediction"] == -1 and attempt < _MAX_RETRIES - 1:
                    last_error = f"Parse failure: {result['reasoning'][:80]}"
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ {last_error}, retrying in {delay:.0f}s ({attempt+1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue
                return result
        except Exception as e:
            last_error = str(e)[:150]
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    ⚠ {last_error}, retrying in {delay:.0f}s ({attempt+1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
            continue
    return {"prediction": -1, "confidence": 0.0,
            "reasoning": f"Failed after {_MAX_RETRIES} attempts: {last_error}"}


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

async def run_comparison(
    n_cases: int = 50,
    model: str = "grok-4-1-fast-reasoning",
    concurrent: int = 10,
    seed: int = 42,
):
    api_key = os.getenv("XAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    use_anthropic = _is_anthropic_model(model)

    if use_anthropic:
        if not anthropic_key:
            print("Error: ANTHROPIC_API_KEY not set in .env")
            return None
        api_key = anthropic_key
        print(f"  Using Anthropic API (model={model})")
    else:
        if not api_key:
            print("Error: XAI_API_KEY not set in .env")
            return None

    # Load commons
    if not COMMONS_FILE.exists():
        print(f"Error: {COMMONS_FILE} not found. Run tally_commons.py first.")
        return None

    commons = load_json(COMMONS_FILE)
    common_ids = commons.get("common_case_ids", [])
    print(f"Common cases available: {len(common_ids)}")

    if not common_ids:
        print("No common cases found!")
        return None

    # Load both extractions
    print("Loading graph + structured extractions...")
    pairs = load_both_extractions(common_ids)
    print(f"  Loaded {len(pairs)} valid pairs")

    # Load labels
    print("Loading labels from IL-TUR...")
    labels = load_labels_hf()

    # Filter to cases with labels
    corpus = []
    for case_id, graph, struct in pairs:
        if case_id in labels:
            corpus.append((case_id, graph, struct, labels[case_id]))
    print(f"  {len(corpus)} cases with graph + structured + label")

    n_acc = sum(1 for _, _, _, l in corpus if l == 1)
    n_rej = sum(1 for _, _, _, l in corpus if l == 0)
    print(f"  Labels: {n_acc} accepted, {n_rej} rejected")

    # Stratified sample
    rng = np.random.RandomState(seed)
    acc_idx = [i for i, (_, _, _, l) in enumerate(corpus) if l == 1]
    rej_idx = [i for i, (_, _, _, l) in enumerate(corpus) if l == 0]

    actual_n = min(n_cases, len(corpus))
    if actual_n < len(corpus):
        n_acc_sample = max(1, int(actual_n * len(acc_idx) / len(corpus)))
        n_rej_sample = actual_n - n_acc_sample
        selected = sorted(
            list(rng.choice(acc_idx, min(n_acc_sample, len(acc_idx)), replace=False)) +
            list(rng.choice(rej_idx, min(n_rej_sample, len(rej_idx)), replace=False))
        )
    else:
        selected = list(range(len(corpus)))

    n = len(selected)
    print(f"\nSelected {n} cases (stratified, seed={seed})")
    sel_acc = sum(1 for i in selected if corpus[i][3] == 1)
    sel_rej = sum(1 for i in selected if corpus[i][3] == 0)
    print(f"  {sel_acc} accepted, {sel_rej} rejected")

    print(f"\nModel: {model}")
    print(f"Blinding: NO scrub on both — strip holdings/outcome/court_reasoning only")
    print(f"Concurrent: {concurrent}")

    # Build prompts + sanity checks
    eval_cases = []
    graph_char_counts = []
    struct_char_counts = []
    sanity_warnings = []

    for idx in selected:
        case_id, graph, struct, label = corpus[idx]
        label_str = "ACC" if label == 1 else "REJ"

        graph_prompt = build_graph_prompt(graph)
        struct_prompt = build_structured_prompt(struct)

        graph_summary = build_blinded_graph_summary(graph)
        struct_summary = build_blinded_structured_summary(struct)
        sanity_warnings.extend(blinding_sanity_check(graph_summary, f"GRAPH-{label_str}", case_id))
        sanity_warnings.extend(blinding_sanity_check(struct_summary, f"STRUCT-{label_str}", case_id))

        eval_cases.append({
            "case_id": case_id,
            "label": label,
            "graph_prompt": graph_prompt,
            "struct_prompt": struct_prompt,
        })
        graph_char_counts.append(len(graph_prompt))
        struct_char_counts.append(len(struct_prompt))

    if sanity_warnings:
        print(f"\n⚠️  BLINDING SANITY CHECK: {len(sanity_warnings)} warnings")
        for w in sanity_warnings[:20]:
            print(w)
        if len(sanity_warnings) > 20:
            print(f"  ... and {len(sanity_warnings) - 20} more")
    else:
        print(f"\n✅ BLINDING SANITY CHECK: Clean — no outcome language detected")

    print(f"\nPrompt sizes:")
    print(f"  Graph:      {np.mean(graph_char_counts):.0f} chars avg "
          f"(min={np.min(graph_char_counts)}, max={np.max(graph_char_counts)})")
    print(f"  Structured: {np.mean(struct_char_counts):.0f} chars avg "
          f"(min={np.min(struct_char_counts)}, max={np.max(struct_char_counts)})")
    print(f"  Struct/Graph ratio: {np.mean(struct_char_counts)/max(np.mean(graph_char_counts),1):.1f}x")

    # Run predictions with checkpointing
    print(f"\n{'='*80}")
    print("RUNNING PREDICTIONS")
    print(f"{'='*80}")

    ckpt_config = {"model": model, "seed": seed}
    ckpt_path = _checkpoint_path(n_cases, model)
    completed = load_checkpoint(ckpt_path, ckpt_config)
    completed_ids = {r["case_id"] for r in completed}
    remaining = [c for c in eval_cases if c["case_id"] not in completed_ids]

    if remaining:
        print(f"  {len(remaining)} cases to run ({len(completed)} already done)")
    else:
        print(f"  All {len(completed)} cases already completed — skipping API calls")

    semaphore = asyncio.Semaphore(concurrent)
    results = list(completed)
    errors = {"graph": 0, "struct": 0}
    t0 = time.time()

    async def predict_pair(case: dict, case_num: int):
        predict_fn = llm_predict_anthropic if use_anthropic else llm_predict
        async with semaphore:
            graph_result = await predict_fn(api_key, _SYSTEM_PROMPT,
                                            case["graph_prompt"], model=model)
            struct_result = await predict_fn(api_key, _SYSTEM_PROMPT,
                                             case["struct_prompt"], model=model)

        gp = graph_result.get("prediction", -1)
        sp = struct_result.get("prediction", -1)

        if gp not in (0, 1):
            errors["graph"] += 1
        if sp not in (0, 1):
            errors["struct"] += 1

        label = case["label"]
        label_str = "ACC" if label == 1 else "REJ"
        g_correct = gp == label
        s_correct = sp == label
        g_str = "✓" if g_correct else "✗"
        s_str = "✓" if s_correct else "✗"
        gp_str = "ACC" if gp == 1 else ("REJ" if gp == 0 else "ERR")
        sp_str = "ACC" if sp == 1 else ("REJ" if sp == 0 else "ERR")

        if g_correct and s_correct:
            tag = "both✓"
        elif g_correct and not s_correct:
            tag = "GRAPH+"
        elif not g_correct and s_correct:
            tag = "STRUCT+"
        else:
            tag = "both✗"

        print(f"  [{case_num:3d}] {case['case_id']:12s} TRUE={label_str}  "
              f"graph={g_str}{gp_str} struct={s_str}{sp_str}  "
              f"gconf={graph_result.get('confidence', 0):.2f} "
              f"sconf={struct_result.get('confidence', 0):.2f}  {tag}")

        return {
            "case_id": case["case_id"],
            "true_label": label,
            "graph_pred": gp,
            "graph_conf": graph_result.get("confidence", 0),
            "graph_reasoning": graph_result.get("reasoning", ""),
            "struct_pred": sp,
            "struct_conf": struct_result.get("confidence", 0),
            "struct_reasoning": struct_result.get("reasoning", ""),
        }

    for batch_start in range(0, len(remaining), concurrent):
        batch = remaining[batch_start:batch_start + concurrent]
        global_offset = len(completed) + batch_start
        tasks = [
            asyncio.create_task(predict_pair(c, global_offset + i + 1))
            for i, c in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        save_checkpoint(results, ckpt_config, ckpt_path)
        done = len(results)
        total = len(eval_cases)
        print(f"  --- checkpoint saved: {done}/{total} ({done/total*100:.0f}%) ---")

    elapsed = time.time() - t0

    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  ✅ Checkpoint cleaned up")

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================

    valid = [r for r in results if r["graph_pred"] in (0, 1) and r["struct_pred"] in (0, 1)]
    n = len(valid)

    if n == 0:
        print("\nNo valid results!")
        return None

    graph_correct = sum(1 for r in valid if r["graph_pred"] == r["true_label"])
    struct_correct = sum(1 for r in valid if r["struct_pred"] == r["true_label"])

    graph_acc = graph_correct / n
    struct_acc = struct_correct / n

    acc_cases = [r for r in valid if r["true_label"] == 1]
    rej_cases = [r for r in valid if r["true_label"] == 0]

    graph_acc_on_acc = sum(1 for r in acc_cases if r["graph_pred"] == 1) / max(len(acc_cases), 1)
    graph_acc_on_rej = sum(1 for r in rej_cases if r["graph_pred"] == 0) / max(len(rej_cases), 1)
    struct_acc_on_acc = sum(1 for r in acc_cases if r["struct_pred"] == 1) / max(len(acc_cases), 1)
    struct_acc_on_rej = sum(1 for r in rej_cases if r["struct_pred"] == 0) / max(len(rej_cases), 1)

    def compute_f1(predictions, lbls):
        tp = sum(1 for p, l in zip(predictions, lbls) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, lbls) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, lbls) if p == 0 and l == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    graph_preds = [r["graph_pred"] for r in valid]
    struct_preds = [r["struct_pred"] for r in valid]
    true_labels = [r["true_label"] for r in valid]

    graph_f1_info = compute_f1(graph_preds, true_labels)
    struct_f1_info = compute_f1(struct_preds, true_labels)

    def compute_macro_f1(predictions, lbls):
        f1_cls1 = compute_f1(predictions, lbls)
        flipped_preds = [1 - p for p in predictions]
        flipped_labels = [1 - l for l in lbls]
        f1_cls0 = compute_f1(flipped_preds, flipped_labels)
        return (f1_cls1["f1"] + f1_cls0["f1"]) / 2

    graph_macro_f1 = compute_macro_f1(graph_preds, true_labels)
    struct_macro_f1 = compute_macro_f1(struct_preds, true_labels)

    def cohens_kappa(pred_a, pred_b):
        n_total = len(pred_a)
        if n_total == 0:
            return 0.0
        agree = sum(1 for a, b in zip(pred_a, pred_b) if a == b)
        p_o = agree / n_total
        a1 = sum(pred_a) / n_total
        b1 = sum(pred_b) / n_total
        p_e = a1 * b1 + (1 - a1) * (1 - b1)
        if p_e == 1.0:
            return 1.0
        return (p_o - p_e) / (1 - p_e)

    kappa = cohens_kappa(graph_preds, struct_preds)

    def brier_score(predictions, confidences, lbls):
        scores = []
        for pred, conf, label in zip(predictions, confidences, lbls):
            prob_true = conf if pred == label else 1.0 - conf
            scores.append((1.0 - prob_true) ** 2)
        return np.mean(scores) if scores else 0.0

    graph_confs = [r["graph_conf"] for r in valid]
    struct_confs = [r["struct_conf"] for r in valid]

    graph_brier = brier_score(graph_preds, graph_confs, true_labels)
    struct_brier = brier_score(struct_preds, struct_confs, true_labels)

    both_correct = sum(1 for r in valid if r["graph_pred"] == r["true_label"]
                       and r["struct_pred"] == r["true_label"])
    graph_only = sum(1 for r in valid if r["graph_pred"] == r["true_label"]
                     and r["struct_pred"] != r["true_label"])
    struct_only = sum(1 for r in valid if r["graph_pred"] != r["true_label"]
                      and r["struct_pred"] == r["true_label"])
    both_wrong = sum(1 for r in valid if r["graph_pred"] != r["true_label"]
                     and r["struct_pred"] != r["true_label"])

    graph_conf_correct = np.mean([r["graph_conf"] for r in valid
                                  if r["graph_pred"] == r["true_label"]] or [0])
    graph_conf_wrong = np.mean([r["graph_conf"] for r in valid
                                if r["graph_pred"] != r["true_label"]] or [0])
    struct_conf_correct = np.mean([r["struct_conf"] for r in valid
                                   if r["struct_pred"] == r["true_label"]] or [0])
    struct_conf_wrong = np.mean([r["struct_conf"] for r in valid
                                  if r["struct_pred"] != r["true_label"]] or [0])

    # McNemar's test
    b, c = graph_only, struct_only
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
    struct_correct_arr = np.array([1 if r["struct_pred"] == r["true_label"] else 0 for r in valid])
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng_boot.choice(n, n, replace=True)
        boot_diffs.append(float(graph_correct_arr[idx].mean() - struct_correct_arr[idx].mean()))
    boot_diffs = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================

    print(f"\n\n{'='*80}")
    print(f"  RESULTS: GRAPH-STRUCTURED vs STRUCTURED (non-graph) PREDICTION")
    print(f"{'='*80}")

    print(f"\n  {'Metric':<30} {'Graph':>10} {'Struct':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Accuracy':<30} {graph_acc:>10.3f} {struct_acc:>10.3f} {graph_acc - struct_acc:>+10.3f}")
    print(f"  {'F1':<30} {graph_f1_info['f1']:>10.3f} {struct_f1_info['f1']:>10.3f} "
          f"{graph_f1_info['f1'] - struct_f1_info['f1']:>+10.3f}")
    print(f"  {'Precision':<30} {graph_f1_info['precision']:>10.3f} {struct_f1_info['precision']:>10.3f} "
          f"{graph_f1_info['precision'] - struct_f1_info['precision']:>+10.3f}")
    print(f"  {'Recall':<30} {graph_f1_info['recall']:>10.3f} {struct_f1_info['recall']:>10.3f} "
          f"{graph_f1_info['recall'] - struct_f1_info['recall']:>+10.3f}")
    print(f"  {'Macro F1':<30} {graph_macro_f1:>10.3f} {struct_macro_f1:>10.3f} "
          f"{graph_macro_f1 - struct_macro_f1:>+10.3f}")
    print(f"  {'Brier Score (↓ better)':<30} {graph_brier:>10.3f} {struct_brier:>10.3f} "
          f"{graph_brier - struct_brier:>+10.3f}")

    print(f"\n  {'Per-class accuracy:':<30}")
    print(f"  {'  Accepted cases':<30} {graph_acc_on_acc:>10.3f} {struct_acc_on_acc:>10.3f} "
          f"{graph_acc_on_acc - struct_acc_on_acc:>+10.3f}")
    print(f"  {'  Rejected cases':<30} {graph_acc_on_rej:>10.3f} {struct_acc_on_rej:>10.3f} "
          f"{graph_acc_on_rej - struct_acc_on_rej:>+10.3f}")

    print(f"\n  {'Confidence (correct)':<30} {graph_conf_correct:>10.3f} {struct_conf_correct:>10.3f}")
    print(f"  {'Confidence (wrong)':<30} {graph_conf_wrong:>10.3f} {struct_conf_wrong:>10.3f}")
    print(f"  {'Conf. calibration gap':<30} "
          f"{graph_conf_correct - graph_conf_wrong:>10.3f} "
          f"{struct_conf_correct - struct_conf_wrong:>10.3f}")

    print(f"\n  AGREEMENT MATRIX:")
    print(f"  {'':>20} {'Struct ✓':>10} {'Struct ✗':>10}")
    print(f"  {'Graph ✓':<20} {both_correct:>10} {graph_only:>10}")
    print(f"  {'Graph ✗':<20} {struct_only:>10} {both_wrong:>10}")

    print(f"\n  Graph wins  (graph✓ struct✗): {graph_only}")
    print(f"  Struct wins (graph✗ struct✓): {struct_only}")
    print(f"  Both correct:                 {both_correct}")
    print(f"  Both wrong:                   {both_wrong}")

    print(f"\n  STATISTICAL TESTS:")
    print(f"  McNemar's χ² = {mcnemar_chi2:.3f}  ({mcnemar_sig})")
    print(f"  Bootstrap 95% CI for accuracy difference: [{ci_low:+.3f}, {ci_high:+.3f}]")
    if ci_low > 0:
        print(f"  → Graph significantly better (CI excludes 0)")
    elif ci_high < 0:
        print(f"  → Structured significantly better (CI excludes 0)")
    else:
        print(f"  → Difference not significant at 95% (CI includes 0)")

    print(f"\n  Cohen's κ (graph vs struct agreement): {kappa:.3f}")
    if kappa < 0.20:
        print(f"  → Slight agreement — methods behave very differently")
    elif kappa < 0.40:
        print(f"  → Fair agreement")
    elif kappa < 0.60:
        print(f"  → Moderate agreement")
    elif kappa < 0.80:
        print(f"  → Substantial agreement")
    else:
        print(f"  → Near-perfect agreement")

    print(f"\n  Brier Score: graph={graph_brier:.4f}, struct={struct_brier:.4f} "
          f"(Δ={graph_brier - struct_brier:+.4f}, "
          f"{'graph' if graph_brier < struct_brier else 'struct'} better)")

    print(f"\n  n={n}, errors: graph={errors['graph']}, struct={errors['struct']}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(n, 1):.1f}s per case pair)")
    print(f"  Model: {model}")
    print(f"  Prompt ratio (struct/graph): {np.mean(struct_char_counts)/max(np.mean(graph_char_counts),1):.1f}x")
    if sanity_warnings:
        print(f"  ⚠️  Blinding warnings: {len(sanity_warnings)} (check above)")
    print(f"{'='*80}")

    # ==========================================================================
    # DISCORDANT CASES
    # ==========================================================================

    discordant = [r for r in valid if (r["graph_pred"] == r["true_label"]) !=
                  (r["struct_pred"] == r["true_label"])]

    if discordant:
        print(f"\n\n{'='*80}")
        print(f"  DISCORDANT CASES (where methods disagree)")
        print(f"{'='*80}")
        for r in discordant:
            label_str = "ACC" if r["true_label"] == 1 else "REJ"
            g_correct = r["graph_pred"] == r["true_label"]
            winner = "GRAPH" if g_correct else "STRUCT"
            gp_str = "ACC" if r["graph_pred"] == 1 else "REJ"
            sp_str = "ACC" if r["struct_pred"] == 1 else "REJ"
            print(f"\n  {r['case_id']}  TRUE={label_str}  → {winner} wins")
            print(f"    Graph:  {gp_str} (conf={r['graph_conf']:.2f})")
            print(f"      {r['graph_reasoning'][:200]}")
            print(f"    Struct: {sp_str} (conf={r['struct_conf']:.2f})")
            print(f"      {r['struct_reasoning'][:200]}")

    # Save full results
    output = {
        "config": {
            "model": model,
            "n_cases": n,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "blinding": "no_scrub on both — holdings/outcome/court_reasoning stripped",
            "blinding_warnings": len(sanity_warnings),
        },
        "summary": {
            "graph_accuracy": round(graph_acc, 4),
            "struct_accuracy": round(struct_acc, 4),
            "accuracy_delta": round(graph_acc - struct_acc, 4),
            "graph_f1": round(graph_f1_info["f1"], 4),
            "struct_f1": round(struct_f1_info["f1"], 4),
            "graph_macro_f1": round(graph_macro_f1, 4),
            "struct_macro_f1": round(struct_macro_f1, 4),
            "graph_precision": round(graph_f1_info["precision"], 4),
            "struct_precision": round(struct_f1_info["precision"], 4),
            "graph_recall": round(graph_f1_info["recall"], 4),
            "struct_recall": round(struct_f1_info["recall"], 4),
            "graph_brier": round(graph_brier, 4),
            "struct_brier": round(struct_brier, 4),
            "cohens_kappa": round(kappa, 4),
            "mcnemar_chi2": round(mcnemar_chi2, 4),
            "mcnemar_sig": mcnemar_sig,
            "bootstrap_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "graph_only_wins": graph_only,
            "struct_only_wins": struct_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "cases": valid,
    }

    model_tag = f"_{_model_slug(model)}" if model else ""
    out_path = Path(f"graph_vs_structured_n{n}{model_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved to {out_path}")

    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Graph vs Structured (non-graph) zero-shot prediction comparison"
    )
    p.add_argument("--n", type=int, default=50, help="Number of cases to evaluate")
    p.add_argument("--model", default="grok-4-1-fast-reasoning", help="LLM model")
    p.add_argument("--concurrent", type=int, default=10, help="Max concurrent API calls")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = p.parse_args()

    asyncio.run(run_comparison(
        n_cases=args.n,
        model=args.model,
        concurrent=args.concurrent,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()