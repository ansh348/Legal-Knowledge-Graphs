#!/usr/bin/env python3
"""
test_iltur_graphs.py

Unified, robust test suite for IL-TUR Legal Reasoning Graph output JSONs.

Validates ~380 extracted graph JSONs against the v2.1 schema with:
  - Structural integrity (required fields, types, enums)
  - Referential integrity (edge endpoints, justification sets, chains)
  - Anchor validity (char offsets, hashes, doc_id)
  - Semantic constraints (Toulmin structure, holdings→issues, outcome wiring)
  - Cross-graph aggregate statistics & anomaly detection
  - Quality tier distribution & outcome prediction accuracy

Usage:
    python test_iltur_graphs.py [--dir iltur_graphs] [--strict] [--report report.json]
    python -m pytest test_iltur_graphs.py -v          # via pytest
    python -m pytest test_iltur_graphs.py -v -k node  # run only node tests

Environment:
    ILTUR_GRAPH_DIR  - path to graph directory (default: iltur_graphs)
    ILTUR_STRICT     - set to "1" to treat warnings as errors

Requires:
    pip install pytest --break-system-packages
"""

import os
import re
import sys
import json
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# =============================================================================
# SCHEMA CONSTANTS (mirrored from schema_v2_1.py — keep in sync)
# =============================================================================

SCHEMA_VERSION_PREFIX = "2.1"

NODE_TYPES = {"fact", "concept", "issue", "argument", "holding", "precedent",
              "outcome", "justification_set"}

ACTOR_TYPES = {"petitioner", "respondent", "court", "lower_court", "amicus",
               "third_party", "appellant", "complainant", "accused", "prosecution"}

CONCEPT_KINDS = {"statute_article", "statute_section", "order_rule", "doctrine",
                 "test", "standard", "right", "definition"}

FACT_TYPES = {"material", "procedural", "background", "disputed", "admitted",
              "judicial_notice"}

ARGUMENT_SCHEMES = {
    "rule_application", "rule_exception", "precedent_following",
    "precedent_analogy", "precedent_distinction", "textual", "purposive",
    "harmonious", "proportionality", "balancing", "evidence_sufficiency",
    "evidence_credibility", "procedural", "jurisdiction", "limitation",
    "policy_consequence", "public_interest", "natural_justice", "other",
}

EDGE_RELATIONS = {
    "triggers", "negates", "partially_satisfies", "satisfies", "claims_satisfies",
    "supports", "rebuts", "undercuts", "grounds", "establishes", "enables",
    "constrains", "requires", "excludes", "specializes", "conflicts_with",
    "addresses", "concedes", "attacks", "supports_arg", "responds_to",
    "resolves", "partially_resolves", "determines", "contributes_to", "follows",
    "applies", "distinguishes", "overrules", "doubts", "explains", "member_of",
}

JUSTIFICATION_LOGIC = {"and", "or"}

CONFIDENCE_LEVELS = {"high", "medium", "low", "inferred"}

RELEVANCE_LEVELS = {"central", "supporting", "mentioned", "obiter"}

PRECEDENT_TREATMENTS = {"followed", "applied", "distinguished", "overruled",
                        "doubted", "explained", "cited"}

DISPOSITIONS = {"allowed", "dismissed", "partly_allowed", "remanded",
                "modified", "set_aside"}

EXTRACTION_METHODS = {"regex", "llm", "rule", "inference", "manual"}

QUALITY_TIERS = {"gold", "silver", "bronze", "reject"}

STRENGTH_VALUES = {"strong", "moderate", "weak"}

# ── Nullable enums: None handled separately by allow_none param ──
COURT_FINDINGS = {"accepted", "rejected", "not_decided"}
COURT_RESPONSES = {"accepted", "rejected", "partly_accepted", "not_addressed"}
ISSUE_ANSWERS = {"yes", "no", "partly", "not_decided"}
OUTCOME_BINARY = {"accepted", "rejected"}
COST_VALUES = {"petitioner", "respondent", "none", "shared"}

# Disposition → expected binary mapping
DISPOSITION_TO_BINARY = {
    "allowed": "accepted",
    "partly_allowed": "accepted",
    "set_aside": "accepted",
    "remanded": "accepted",
    "modified": "accepted",
    "dismissed": "rejected",
}

# Edge relation → expected target node types (where semantically strict)
EDGE_TARGET_CONSTRAINTS = {
    "determines": {"outcome"},
    "resolves": {"issue"},
    "partially_resolves": {"issue"},
    "addresses": {"issue"},
    "member_of": {"justification_set"},
}

# Edge relation → expected source node types
EDGE_SOURCE_CONSTRAINTS = {
    "determines": {"holding"},
    "resolves": {"holding"},
    "partially_resolves": {"holding"},
}


# =============================================================================
# GRAPH LOADER
# =============================================================================

MIN_FILE_SIZE_BYTES = 50  # Files smaller than this are almost certainly corrupt


def load_graphs(graph_dir: str) -> Dict[str, dict]:
    """Load all JSON graph files from directory. Returns {filename_stem: graph_dict}."""
    graphs = {}
    graph_path = Path(graph_dir)

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")

    json_files = sorted(graph_path.glob("*.json"))
    json_files = [f for f in json_files if f.name != "checkpoint.json"]

    for fp in json_files:
        try:
            file_size = fp.stat().st_size
            if file_size < MIN_FILE_SIZE_BYTES:
                graphs[fp.stem] = {
                    "_parse_error": f"File too small ({file_size} bytes), likely corrupt/truncated",
                    "_file": str(fp),
                }
                continue

            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graphs[fp.stem] = data
        except json.JSONDecodeError as e:
            graphs[fp.stem] = {"_parse_error": str(e), "_file": str(fp)}
        except Exception as e:
            graphs[fp.stem] = {"_parse_error": f"{type(e).__name__}: {e}", "_file": str(fp)}

    return graphs


# =============================================================================
# HELPERS — safe type/enum checks (never crash on None-containing sets)
# =============================================================================

def _check_enum(val, allowed: set, path: str, allow_none: bool = False) -> Optional[str]:
    """Check value is in allowed set. Safe: sorts as strings, handles None."""
    if val is None:
        if allow_none:
            return None
        return f"{path}: value is None (expected one of {sorted(str(x) for x in allowed)})"
    if val not in allowed:
        return f"{path}: '{val}' not in {sorted(str(x) for x in allowed)}"
    return None


def _check_type(val, expected_type, path: str, allow_none: bool = False) -> Optional[str]:
    """Return error string if val is not of expected_type, else None."""
    if val is None and allow_none:
        return None
    if not isinstance(val, expected_type):
        return f"{path}: expected {expected_type.__name__}, got {type(val).__name__}"
    return None


def _is_nonempty_string(val) -> bool:
    """Check val is a non-empty, non-whitespace-only string."""
    return isinstance(val, str) and len(val.strip()) > 0


def _check_bool(val, path: str, allow_none: bool = False) -> Optional[str]:
    """Validate a boolean field. Tolerates 0/1 ints (common JSON quirk)."""
    if val is None and allow_none:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int) and val in (0, 1):
        return None  # tolerate 0/1 from JSON serializers
    return f"{path}: expected bool, got {type(val).__name__} ({val!r})"


# =============================================================================
# PER-GRAPH VALIDATOR
# =============================================================================

@dataclass
class ValidationResult:
    case_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def _validate_anchor(anchor: dict, path: str, errors: list, warnings: list):
    """Validate an anchor object."""
    if anchor is None:
        return  # Some anchors are optional

    if not isinstance(anchor, dict):
        errors.append(f"{path}: anchor must be dict, got {type(anchor).__name__}")
        return

    # Required fields
    for req in ("doc_id", "start_char", "end_char"):
        if req not in anchor:
            errors.append(f"{path}.{req}: missing required field")

    # doc_id type check
    doc_id = anchor.get("doc_id")
    if doc_id is not None and not isinstance(doc_id, str):
        errors.append(f"{path}.doc_id: must be string, got {type(doc_id).__name__}")

    # Type checks for offsets
    sc = anchor.get("start_char")
    ec = anchor.get("end_char")

    if sc is not None and not isinstance(sc, (int, float)):
        errors.append(f"{path}.start_char: must be numeric, got {type(sc).__name__}")
        sc = None  # prevent downstream checks on bad type
    if ec is not None and not isinstance(ec, (int, float)):
        errors.append(f"{path}.end_char: must be numeric, got {type(ec).__name__}")
        ec = None

    # Float offsets are suspicious (should be ints)
    if isinstance(sc, float) and sc != int(sc):
        warnings.append(f"{path}.start_char: fractional value {sc} (expected int)")
    if isinstance(ec, float) and ec != int(ec):
        warnings.append(f"{path}.end_char: fractional value {ec} (expected int)")

    # Semantic offset checks
    if sc is not None and ec is not None:
        if sc < 0:
            errors.append(f"{path}: start_char ({sc}) < 0")
        if ec < sc:
            warnings.append(f"{path}: end_char ({ec}) < start_char ({sc})")
        if ec == sc:
            warnings.append(f"{path}: zero-length span (start_char == end_char == {sc})")

    # text_hash
    th = anchor.get("text_hash")
    if th is not None and not isinstance(th, str):
        warnings.append(f"{path}.text_hash: expected string, got {type(th).__name__}")

    # secondary_spans
    spans = anchor.get("secondary_spans")
    if spans is not None:
        if not isinstance(spans, list):
            errors.append(f"{path}.secondary_spans: must be list, got {type(spans).__name__}")
        else:
            for i, span in enumerate(spans):
                if not isinstance(span, (list, tuple)):
                    warnings.append(f"{path}.secondary_spans[{i}]: must be [start, end], got {type(span).__name__}")
                elif len(span) != 2:
                    warnings.append(f"{path}.secondary_spans[{i}]: must be [start, end], got length {len(span)}")
                else:
                    if not isinstance(span[0], (int, float)) or not isinstance(span[1], (int, float)):
                        warnings.append(f"{path}.secondary_spans[{i}]: elements must be numeric")
                    elif span[1] < span[0]:
                        warnings.append(f"{path}.secondary_spans[{i}]: end ({span[1]}) < start ({span[0]})")


def _validate_provenance(prov: dict, path: str, errors: list, warnings: list):
    """Validate a provenance object."""
    if prov is None:
        return

    if not isinstance(prov, dict):
        errors.append(f"{path}: provenance must be dict")
        return

    method = prov.get("extraction_method")
    if method is not None:
        err = _check_enum(method, EXTRACTION_METHODS, f"{path}.extraction_method")
        if err:
            warnings.append(err)  # downgrade: older runs may use custom values

    temp = prov.get("temperature")
    if temp is not None:
        if not isinstance(temp, (int, float)):
            warnings.append(f"{path}.temperature: expected number, got {type(temp).__name__}")
        elif not (0.0 <= temp <= 2.0):
            warnings.append(f"{path}.temperature: {temp} outside expected range [0.0, 2.0]")


def validate_graph(case_id: str, g: dict, strict: bool = False) -> ValidationResult:
    """
    Validate a single graph JSON against the v2.1 schema.

    strict=True treats warnings as errors.
    """
    result = ValidationResult(case_id=case_id)
    errors = result.errors
    warnings = result.warnings

    # ── Parse error from loader ──
    if "_parse_error" in g:
        errors.append(f"JSON parse error: {g['_parse_error']}")
        return result

    # ── Root type ──
    if not isinstance(g, dict):
        errors.append(f"Root is not a JSON object, got {type(g).__name__}")
        return result

    # ── Top-level scalar fields ──
    if "case_id" not in g:
        errors.append("Missing required field: case_id")
    elif not isinstance(g["case_id"], str):
        errors.append(f"case_id must be string, got {type(g['case_id']).__name__}")

    cn = g.get("case_name")
    if cn is not None and not isinstance(cn, str):
        warnings.append(f"case_name: expected string, got {type(cn).__name__}")

    court = g.get("court")
    if court is not None and not isinstance(court, str):
        warnings.append(f"court: expected string, got {type(court).__name__}")

    cy = g.get("case_year")
    if cy is not None:
        if not isinstance(cy, int):
            warnings.append(f"case_year: expected int, got {type(cy).__name__} ({cy!r})")
        elif cy < 1800 or cy > 2030:
            warnings.append(f"case_year: {cy} outside plausible range [1800, 2030]")

    judges = g.get("judges")
    if judges is not None:
        if not isinstance(judges, list):
            warnings.append(f"judges: expected list, got {type(judges).__name__}")
        else:
            for ji, j in enumerate(judges):
                if not isinstance(j, str):
                    warnings.append(f"judges[{ji}]: expected string, got {type(j).__name__}")

    # ── Meta block ──
    meta = g.get("_meta", {})
    if not isinstance(meta, dict):
        warnings.append(f"_meta: expected dict, got {type(meta).__name__}")
        meta = {}

    if not meta:
        warnings.append("Missing _meta block")
    else:
        sv = meta.get("schema_version", "")
        if sv and not str(sv).startswith("2."):
            warnings.append(f"Unexpected schema_version: {sv}")

        qt = meta.get("quality_tier")
        if qt:
            err = _check_enum(qt, QUALITY_TIERS, "_meta.quality_tier")
            if err:
                errors.append(err)

        vw = meta.get("validation_warnings")
        if vw is not None and not isinstance(vw, list):
            warnings.append(f"_meta.validation_warnings: expected list, got {type(vw).__name__}")

    # ── Collect all node IDs for referential integrity ──
    all_node_ids: Set[str] = set()
    id_to_type: Dict[str, str] = {}
    seen_ids: Set[str] = set()

    def _register_id(nid: str, ntype: str, path: str):
        if not isinstance(nid, str) or not nid.strip():
            errors.append(f"{path}: id must be a non-empty string, got {nid!r}")
            return
        if nid in seen_ids:
            errors.append(f"{path}: duplicate node ID '{nid}'")
        seen_ids.add(nid)
        all_node_ids.add(nid)
        id_to_type[nid] = ntype

    # ── Facts ──
    facts = g.get("facts", [])
    if not isinstance(facts, list):
        errors.append(f"facts: expected list, got {type(facts).__name__}")
        facts = []

    for i, f in enumerate(facts):
        path = f"facts[{i}]"
        if not isinstance(f, dict):
            errors.append(f"{path}: must be dict, got {type(f).__name__}")
            continue

        fid = f.get("id")
        if not fid or not isinstance(fid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            _register_id(fid, "fact", path)

        if not _is_nonempty_string(f.get("text")):
            errors.append(f"{path}: missing or empty text")

        err = _check_enum(f.get("type"), {"fact"}, f"{path}.type")
        if err:
            warnings.append(err)

        err = _check_enum(f.get("fact_type"), FACT_TYPES, f"{path}.fact_type")
        if err:
            errors.append(err)

        err = _check_enum(f.get("actor_source"), ACTOR_TYPES, f"{path}.actor_source", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(f.get("court_finding"), COURT_FINDINGS, f"{path}.court_finding", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(f.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            errors.append(err)

        err = _check_bool(f.get("date_approximate"), f"{path}.date_approximate", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(f.get("disputed_by"), ACTOR_TYPES, f"{path}.disputed_by", allow_none=True)
        if err:
            warnings.append(err)

        _validate_anchor(f.get("anchor"), f"{path}.anchor", errors, warnings)
        _validate_provenance(f.get("provenance"), f"{path}.provenance", errors, warnings)

    # ── Concepts ──
    concepts = g.get("concepts", [])
    if not isinstance(concepts, list):
        errors.append(f"concepts: expected list, got {type(concepts).__name__}")
        concepts = []

    concept_ids_in_graph: Set[str] = set()
    for i, c in enumerate(concepts):
        path = f"concepts[{i}]"
        if not isinstance(c, dict):
            errors.append(f"{path}: must be dict, got {type(c).__name__}")
            continue

        cid = c.get("id")
        if not cid or not isinstance(cid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            _register_id(cid, "concept", path)

        concept_id_val = c.get("concept_id")
        if concept_id_val:
            if not isinstance(concept_id_val, str):
                warnings.append(f"{path}.concept_id: expected string, got {type(concept_id_val).__name__}")
            else:
                concept_ids_in_graph.add(concept_id_val)

        err = _check_enum(c.get("type"), {"concept"}, f"{path}.type")
        if err:
            warnings.append(err)

        err = _check_enum(c.get("relevance"), RELEVANCE_LEVELS, f"{path}.relevance")
        if err:
            errors.append(err)

        err = _check_enum(c.get("kind"), CONCEPT_KINDS, f"{path}.kind", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(c.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            errors.append(err)

        _validate_anchor(c.get("anchor"), f"{path}.anchor", errors, warnings)
        _validate_provenance(c.get("provenance"), f"{path}.provenance", errors, warnings)
        if c.get("interpretation_anchor"):
            _validate_anchor(c["interpretation_anchor"], f"{path}.interpretation_anchor", errors, warnings)

    # ── Issues ──
    issues = g.get("issues", [])
    if not isinstance(issues, list):
        errors.append(f"issues: expected list, got {type(issues).__name__}")
        issues = []

    for i, iss in enumerate(issues):
        path = f"issues[{i}]"
        if not isinstance(iss, dict):
            errors.append(f"{path}: must be dict, got {type(iss).__name__}")
            continue

        iid = iss.get("id")
        if not iid or not isinstance(iid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            _register_id(iid, "issue", path)

        if not _is_nonempty_string(iss.get("text")):
            errors.append(f"{path}: missing or empty text")

        err = _check_enum(iss.get("type"), {"issue"}, f"{path}.type")
        if err:
            warnings.append(err)

        err = _check_enum(iss.get("framed_by"), ACTOR_TYPES, f"{path}.framed_by")
        if err:
            warnings.append(err)

        err = _check_enum(iss.get("answer"), ISSUE_ANSWERS, f"{path}.answer", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(iss.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            errors.append(err)

        inum = iss.get("issue_number")
        if inum is not None and not isinstance(inum, int):
            warnings.append(f"{path}.issue_number: expected int, got {type(inum).__name__}")

        pcs = iss.get("primary_concepts")
        if pcs is not None:
            if not isinstance(pcs, list):
                warnings.append(f"{path}.primary_concepts: expected list, got {type(pcs).__name__}")
            else:
                for pc in pcs:
                    if not isinstance(pc, str):
                        warnings.append(f"{path}.primary_concepts: element must be string")
                    elif pc not in concept_ids_in_graph:
                        warnings.append(f"{path}: primary_concept '{pc}' not in graph concepts")

        _validate_anchor(iss.get("anchor"), f"{path}.anchor", errors, warnings)
        _validate_provenance(iss.get("provenance"), f"{path}.provenance", errors, warnings)

    # ── Arguments ──
    arguments = g.get("arguments", [])
    if not isinstance(arguments, list):
        errors.append(f"arguments: expected list, got {type(arguments).__name__}")
        arguments = []

    for i, a in enumerate(arguments):
        path = f"arguments[{i}]"
        if not isinstance(a, dict):
            errors.append(f"{path}: must be dict, got {type(a).__name__}")
            continue

        aid = a.get("id")
        if not aid or not isinstance(aid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            _register_id(aid, "argument", path)

        if not _is_nonempty_string(a.get("claim")):
            errors.append(f"{path}: missing or empty claim")

        err = _check_enum(a.get("type"), {"argument"}, f"{path}.type")
        if err:
            warnings.append(err)

        err = _check_enum(a.get("actor"), ACTOR_TYPES, f"{path}.actor")
        if err:
            errors.append(err)

        schemes = a.get("schemes")
        if schemes is None or not isinstance(schemes, list):
            errors.append(f"{path}.schemes: must be a list, got {type(schemes).__name__}")
        elif len(schemes) == 0:
            warnings.append(f"{path}.schemes: empty (should have at least one)")
        else:
            for si, s in enumerate(schemes):
                err = _check_enum(s, ARGUMENT_SCHEMES, f"{path}.schemes[{si}]")
                if err:
                    errors.append(err)

        err = _check_enum(a.get("court_response"), COURT_RESPONSES, f"{path}.court_response", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(a.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            errors.append(err)

        _validate_anchor(a.get("anchor"), f"{path}.anchor", errors, warnings)
        _validate_provenance(a.get("provenance"), f"{path}.provenance", errors, warnings)
        if a.get("court_response_anchor"):
            _validate_anchor(a["court_response_anchor"], f"{path}.court_response_anchor", errors, warnings)

    # ── Holdings ──
    holdings = g.get("holdings", [])
    if not isinstance(holdings, list):
        errors.append(f"holdings: expected list, got {type(holdings).__name__}")
        holdings = []

    issue_ids_set = {iss.get("id") for iss in issues if isinstance(iss, dict) and isinstance(iss.get("id"), str)}

    for i, h in enumerate(holdings):
        path = f"holdings[{i}]"
        if not isinstance(h, dict):
            errors.append(f"{path}: must be dict, got {type(h).__name__}")
            continue

        hid = h.get("id")
        if not hid or not isinstance(hid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            _register_id(hid, "holding", path)

        if not _is_nonempty_string(h.get("text")):
            errors.append(f"{path}: missing or empty text")

        err = _check_enum(h.get("type"), {"holding"}, f"{path}.type")
        if err:
            warnings.append(err)

        ri = h.get("resolves_issue")
        if ri is not None:
            if not isinstance(ri, str):
                warnings.append(f"{path}.resolves_issue: expected string, got {type(ri).__name__}")
            elif ri not in issue_ids_set:
                warnings.append(f"{path}: resolves_issue '{ri}' not found in issues")

        err = _check_bool(h.get("is_ratio"), f"{path}.is_ratio", allow_none=True)
        if err:
            warnings.append(err)
        err = _check_bool(h.get("novel"), f"{path}.novel", allow_none=True)
        if err:
            warnings.append(err)

        h_schemes = h.get("schemes")
        if h_schemes is not None:
            if not isinstance(h_schemes, list):
                warnings.append(f"{path}.schemes: expected list, got {type(h_schemes).__name__}")
            else:
                for si, s in enumerate(h_schemes):
                    err = _check_enum(s, ARGUMENT_SCHEMES, f"{path}.schemes[{si}]")
                    if err:
                        warnings.append(err)

        err = _check_enum(h.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            errors.append(err)

        _validate_anchor(h.get("anchor"), f"{path}.anchor", errors, warnings)
        _validate_provenance(h.get("provenance"), f"{path}.provenance", errors, warnings)

    # ── Precedents ──
    precedents = g.get("precedents", [])
    if not isinstance(precedents, list):
        errors.append(f"precedents: expected list, got {type(precedents).__name__}")
        precedents = []

    for i, p in enumerate(precedents):
        path = f"precedents[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{path}: must be dict, got {type(p).__name__}")
            continue

        pid = p.get("id")
        if not pid or not isinstance(pid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            _register_id(pid, "precedent", path)

        if not _is_nonempty_string(p.get("citation")):
            errors.append(f"{path}: missing or empty citation")

        err = _check_enum(p.get("type"), {"precedent"}, f"{path}.type")
        if err:
            warnings.append(err)

        err = _check_enum(p.get("treatment"), PRECEDENT_TREATMENTS, f"{path}.treatment", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(p.get("relevance"), RELEVANCE_LEVELS, f"{path}.relevance")
        if err:
            warnings.append(err)

        err = _check_enum(p.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            errors.append(err)

        pcy = p.get("case_year")
        if pcy is not None:
            if not isinstance(pcy, int):
                warnings.append(f"{path}.case_year: expected int, got {type(pcy).__name__}")
            elif pcy < 1800 or pcy > 2030:
                warnings.append(f"{path}.case_year: {pcy} outside plausible range")

        _validate_anchor(p.get("anchor"), f"{path}.anchor", errors, warnings)
        _validate_provenance(p.get("provenance"), f"{path}.provenance", errors, warnings)

    # ── Justification Sets ──
    js_list = g.get("justification_sets", [])
    if not isinstance(js_list, list):
        errors.append(f"justification_sets: expected list, got {type(js_list).__name__}")
        js_list = []

    js_ids: Set[str] = set()
    js_targets: Dict[str, str] = {}

    for i, js in enumerate(js_list):
        path = f"justification_sets[{i}]"
        if not isinstance(js, dict):
            errors.append(f"{path}: must be dict, got {type(js).__name__}")
            continue

        jsid = js.get("id")
        if not jsid or not isinstance(jsid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            _register_id(jsid, "justification_set", path)
            js_ids.add(jsid)
            js_targets[jsid] = js.get("target_id", "")

        err = _check_enum(js.get("type"), {"justification_set"}, f"{path}.type")
        if err:
            warnings.append(err)

        if not js.get("target_id"):
            errors.append(f"{path}: missing target_id")
        elif not isinstance(js.get("target_id"), str):
            errors.append(f"{path}.target_id: expected string, got {type(js.get('target_id')).__name__}")

        err = _check_enum(js.get("logic"), JUSTIFICATION_LOGIC, f"{path}.logic")
        if err:
            errors.append(err)

        err = _check_bool(js.get("is_primary"), f"{path}.is_primary", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_enum(js.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            warnings.append(err)

    # ── Outcome ──
    outcome = g.get("outcome")
    if outcome is not None:
        if not isinstance(outcome, dict):
            errors.append(f"outcome: expected dict or null, got {type(outcome).__name__}")
            outcome = None
        else:
            path = "outcome"
            oid = outcome.get("id", "outcome")
            _register_id(oid, "outcome", path)

            err = _check_enum(outcome.get("type"), {"outcome"}, f"{path}.type")
            if err:
                warnings.append(err)

            disp = outcome.get("disposition")
            err = _check_enum(disp, DISPOSITIONS, f"{path}.disposition")
            if err:
                errors.append(err)

            binary = outcome.get("binary")
            err = _check_enum(binary, OUTCOME_BINARY, f"{path}.binary")
            if err:
                errors.append(err)

            # Disposition ↔ binary consistency
            if disp in DISPOSITION_TO_BINARY and binary in OUTCOME_BINARY:
                expected_binary = DISPOSITION_TO_BINARY[disp]
                if binary != expected_binary:
                    warnings.append(
                        f"{path}: disposition '{disp}' implies binary='{expected_binary}', got '{binary}'"
                    )

            err = _check_enum(outcome.get("costs"), COST_VALUES, f"{path}.costs", allow_none=True)
            if err:
                warnings.append(err)

            directions = outcome.get("directions")
            if directions is not None:
                if not isinstance(directions, list):
                    warnings.append(f"{path}.directions: expected list, got {type(directions).__name__}")
                else:
                    for di, d in enumerate(directions):
                        if not isinstance(d, str):
                            warnings.append(f"{path}.directions[{di}]: expected string, got {type(d).__name__}")

            _validate_anchor(outcome.get("anchor"), f"{path}.anchor", errors, warnings)
            _validate_provenance(outcome.get("provenance"), f"{path}.provenance", errors, warnings)
    else:
        warnings.append("No outcome extracted")

    # ── Edges ──
    edges = g.get("edges", [])
    if not isinstance(edges, list):
        errors.append(f"edges: expected list, got {type(edges).__name__}")
        edges = []

    edge_ids_seen: Set[str] = set()
    edge_triples: Set[Tuple[str, str, str]] = set()

    for i, e in enumerate(edges):
        path = f"edges[{i}]"
        if not isinstance(e, dict):
            errors.append(f"{path}: must be dict, got {type(e).__name__}")
            continue

        eid = e.get("id")
        if not eid or not isinstance(eid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            if eid in edge_ids_seen:
                errors.append(f"{path}: duplicate edge ID '{eid}'")
            edge_ids_seen.add(eid)

        src = e.get("source")
        tgt = e.get("target")
        rel = e.get("relation")

        if not isinstance(src, str) or not src:
            errors.append(f"{path}: missing or invalid source")
        elif src not in all_node_ids:
            errors.append(f"{path}: source '{src}' not found in node IDs")

        if not isinstance(tgt, str) or not tgt:
            errors.append(f"{path}: missing or invalid target")
        elif tgt not in all_node_ids:
            errors.append(f"{path}: target '{tgt}' not found in node IDs")

        if isinstance(src, str) and isinstance(tgt, str) and src == tgt:
            errors.append(f"{path}: self-loop (source == target == '{src}')")

        err = _check_enum(rel, EDGE_RELATIONS, f"{path}.relation")
        if err:
            errors.append(err)

        if isinstance(src, str) and isinstance(tgt, str) and isinstance(rel, str):
            triple = (src, tgt, rel)
            if triple in edge_triples:
                warnings.append(f"{path}: duplicate edge ({src} --{rel}--> {tgt})")
            edge_triples.add(triple)

        err = _check_enum(e.get("confidence"), CONFIDENCE_LEVELS, f"{path}.confidence")
        if err:
            errors.append(err)

        err = _check_enum(e.get("strength"), STRENGTH_VALUES, f"{path}.strength", allow_none=True)
        if err:
            warnings.append(err)

        err = _check_bool(e.get("is_critical"), f"{path}.is_critical", allow_none=True)
        if err:
            warnings.append(err)

        sgids = e.get("support_group_ids")
        if sgids is not None:
            if not isinstance(sgids, list):
                errors.append(f"{path}.support_group_ids: expected list, got {type(sgids).__name__}")
            else:
                for sg in sgids:
                    if not isinstance(sg, str):
                        errors.append(f"{path}.support_group_ids: element must be string")
                    elif sg not in js_ids:
                        errors.append(f"{path}: support_group_id '{sg}' not in justification_sets")
                    elif isinstance(tgt, str) and js_targets.get(sg) and tgt != js_targets[sg]:
                        warnings.append(f"{path}: edge target '{tgt}' != JS '{sg}' target '{js_targets[sg]}'")

        conf = e.get("confidence")
        if conf in ("high", "medium") and e.get("anchor") is None:
            warnings.append(f"{path}: {conf} confidence edge lacks anchor")
        if conf == "inferred" and not e.get("explanation"):
            warnings.append(f"{path}: inferred edge lacks explanation")

        # Edge type compatibility
        if isinstance(rel, str) and isinstance(tgt, str) and tgt in id_to_type:
            tgt_type = id_to_type[tgt]
            expected_targets = EDGE_TARGET_CONSTRAINTS.get(rel)
            if expected_targets and tgt_type not in expected_targets:
                warnings.append(f"{path}: '{rel}' targets '{tgt_type}', expected {expected_targets}")
        if isinstance(rel, str) and isinstance(src, str) and src in id_to_type:
            src_type = id_to_type[src]
            expected_sources = EDGE_SOURCE_CONSTRAINTS.get(rel)
            if expected_sources and src_type not in expected_sources:
                warnings.append(f"{path}: '{rel}' from '{src_type}', expected {expected_sources}")

        _validate_anchor(e.get("anchor"), f"{path}.anchor", errors, warnings)
        _validate_provenance(e.get("provenance"), f"{path}.provenance", errors, warnings)

    # ── Justification Set referential checks (deferred — all nodes now registered) ──
    for i, js in enumerate(js_list):
        if not isinstance(js, dict):
            continue
        path = f"justification_sets[{i}]"
        tid = js.get("target_id")
        if isinstance(tid, str) and tid not in all_node_ids:
            errors.append(f"{path}: target_id '{tid}' not found in node IDs")

        jsid = js.get("id")
        if isinstance(jsid, str):
            member_edges = [
                e for e in edges
                if isinstance(e, dict) and jsid in (e.get("support_group_ids") or [])
            ]
            if not member_edges:
                warnings.append(f"{path}: no edges reference justification_set '{jsid}'")

    # ── Reasoning Chains ──
    chains = g.get("reasoning_chains", [])
    if not isinstance(chains, list):
        errors.append(f"reasoning_chains: expected list, got {type(chains).__name__}")
        chains = []

    chain_ids_seen: Set[str] = set()
    for i, rc in enumerate(chains):
        path = f"reasoning_chains[{i}]"
        if not isinstance(rc, dict):
            errors.append(f"{path}: must be dict, got {type(rc).__name__}")
            continue

        rcid = rc.get("id")
        if not rcid or not isinstance(rcid, str):
            errors.append(f"{path}: missing or invalid id")
        else:
            if rcid in chain_ids_seen:
                errors.append(f"{path}: duplicate chain ID '{rcid}'")
            chain_ids_seen.add(rcid)

        for ref_name in ("issue_id", "holding_id", "justification_set_id"):
            ref_val = rc.get(ref_name)
            if ref_val is not None:
                if not isinstance(ref_val, str):
                    warnings.append(f"{path}.{ref_name}: expected string, got {type(ref_val).__name__}")
                elif ref_name == "justification_set_id" and ref_val not in js_ids:
                    warnings.append(f"{path}.{ref_name}: '{ref_val}' not found in JS")
                elif ref_name != "justification_set_id" and ref_val not in all_node_ids:
                    warnings.append(f"{path}.{ref_name}: '{ref_val}' not found in nodes")

        for ref_field in ("fact_ids", "concept_ids", "argument_ids", "edge_ids", "critical_nodes"):
            ref_list = rc.get(ref_field)
            if ref_list is not None:
                if not isinstance(ref_list, list):
                    warnings.append(f"{path}.{ref_field}: expected list, got {type(ref_list).__name__}")
                    continue
                for ref_id in ref_list:
                    if not isinstance(ref_id, str):
                        warnings.append(f"{path}.{ref_field}: element must be string")
                    elif ref_field == "edge_ids":
                        if ref_id not in edge_ids_seen:
                            warnings.append(f"{path}.{ref_field}: edge '{ref_id}' not found")
                    else:
                        if ref_id not in all_node_ids:
                            warnings.append(f"{path}.{ref_field}: node '{ref_id}' not found")

    # ── Outcome wiring: at least one DETERMINES edge → outcome ──
    if outcome and isinstance(outcome, dict):
        oid = outcome.get("id", "outcome")
        has_determines = any(
            isinstance(e, dict) and e.get("target") == oid and e.get("relation") == "determines"
            for e in edges
        )
        if not has_determines:
            warnings.append("Outcome exists but no DETERMINES edge targets it")

    # ── Orphan detection ──
    nodes_in_edges = set()
    for e in edges:
        if isinstance(e, dict):
            s = e.get("source")
            t = e.get("target")
            if isinstance(s, str):
                nodes_in_edges.add(s)
            if isinstance(t, str):
                nodes_in_edges.add(t)

    for nid in all_node_ids:
        ntype = id_to_type.get(nid)
        if ntype in ("outcome", "justification_set"):
            continue
        if nid not in nodes_in_edges:
            warnings.append(f"Orphan node '{nid}' ({ntype}): no edges reference it")

    # ── Compute statistics ──
    orphan_count = sum(
        1 for nid in all_node_ids
        if id_to_type.get(nid) not in ("outcome", "justification_set")
        and nid not in nodes_in_edges
    )

    result.stats = {
        "n_facts": len(facts),
        "n_concepts": len(concepts),
        "n_issues": len(issues),
        "n_arguments": len(arguments),
        "n_holdings": len(holdings),
        "n_precedents": len(precedents),
        "n_justification_sets": len(js_list),
        "n_edges": len(edges),
        "n_chains": len(chains),
        "n_nodes_total": len(all_node_ids),
        "has_outcome": outcome is not None and isinstance(outcome, dict),
        "quality_tier": meta.get("quality_tier") if isinstance(meta, dict) else None,
        "disposition": outcome.get("disposition") if isinstance(outcome, dict) else None,
        "n_orphans": orphan_count,
    }

    # In strict mode, warnings become errors
    if strict:
        errors.extend(warnings)
        warnings.clear()

    # Record counts AFTER strict conversion
    result.stats["n_errors"] = len(errors)
    result.stats["n_warnings"] = len(warnings)

    return result


# =============================================================================
# AGGREGATE VALIDATOR
# =============================================================================

@dataclass
class AggregateResult:
    total_graphs: int = 0
    parsed_ok: int = 0
    parse_failures: int = 0
    validation_passed: int = 0
    validation_failed: int = 0
    total_errors: int = 0
    total_warnings: int = 0

    per_graph: Dict[str, ValidationResult] = field(default_factory=dict)

    quality_tier_dist: Counter = field(default_factory=Counter)
    disposition_dist: Counter = field(default_factory=Counter)
    fact_type_dist: Counter = field(default_factory=Counter)
    edge_relation_dist: Counter = field(default_factory=Counter)
    argument_scheme_dist: Counter = field(default_factory=Counter)
    confidence_dist: Counter = field(default_factory=Counter)

    stat_distributions: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    anomalies: List[str] = field(default_factory=list)
    error_frequency: Counter = field(default_factory=Counter)
    duplicate_case_ids: List[Tuple[str, list]] = field(default_factory=list)


def run_aggregate_validation(
        graphs: Dict[str, dict],
        strict: bool = False
) -> AggregateResult:
    """Run validation across all graphs and compute aggregate metrics."""
    agg = AggregateResult()
    agg.total_graphs = len(graphs)

    case_id_to_files: Dict[str, List[str]] = defaultdict(list)

    for file_stem, g in graphs.items():
        if "_parse_error" in g:
            agg.parse_failures += 1
            result = ValidationResult(case_id=file_stem, errors=[f"Parse error: {g['_parse_error']}"])
            agg.per_graph[file_stem] = result
            agg.validation_failed += 1
            continue

        agg.parsed_ok += 1

        graph_case_id = g.get("case_id")
        if isinstance(graph_case_id, str):
            case_id_to_files[graph_case_id].append(file_stem)

        result = validate_graph(file_stem, g, strict=strict)
        agg.per_graph[file_stem] = result

        if result.passed:
            agg.validation_passed += 1
        else:
            agg.validation_failed += 1

        agg.total_errors += len(result.errors)
        agg.total_warnings += len(result.warnings)

        for err in result.errors:
            if ": " in err:
                pattern = err.split(": ", 1)[0]
                pattern = re.sub(r'\[\d+\]', '[*]', pattern)       # facts[3] → facts[*]
                pattern = re.sub(r"'[^']{1,60}'", "'…'", pattern)  # 'f3' → '…'
            else:
                pattern = err
            agg.error_frequency[pattern] += 1

        for key, val in result.stats.items():
            if isinstance(val, (int, float)) and key.startswith("n_"):
                agg.stat_distributions[key].append(val)

        qt = result.stats.get("quality_tier")
        if qt:
            agg.quality_tier_dist[qt] += 1

        disp = result.stats.get("disposition")
        if disp:
            agg.disposition_dist[disp] += 1

        for f_node in g.get("facts", []):
            if isinstance(f_node, dict):
                ft = f_node.get("fact_type")
                if ft:
                    agg.fact_type_dist[ft] += 1
                conf = f_node.get("confidence")
                if conf:
                    agg.confidence_dist[f"fact:{conf}"] += 1

        for e_node in g.get("edges", []):
            if isinstance(e_node, dict):
                rel_val = e_node.get("relation")
                if rel_val:
                    agg.edge_relation_dist[rel_val] += 1

        for a_node in g.get("arguments", []):
            if isinstance(a_node, dict):
                for s in (a_node.get("schemes") or []):
                    if isinstance(s, str):
                        agg.argument_scheme_dist[s] += 1

    # Cross-graph duplicate case_ids
    for cid, files in case_id_to_files.items():
        if len(files) > 1:
            agg.duplicate_case_ids.append((cid, files))
            agg.anomalies.append(f"Duplicate case_id '{cid}' in files: {files}")

    # Z-score anomaly detection
    for key, values in agg.stat_distributions.items():
        if len(values) < 10:
            continue
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        if stdev == 0:
            continue
        for file_stem, result in agg.per_graph.items():
            if not result.stats:
                continue
            val = result.stats.get(key)
            if val is None:
                continue
            z_score = (val - mean) / stdev
            if abs(z_score) > 3.0:
                direction = "abnormally high" if z_score > 0 else "abnormally low"
                agg.anomalies.append(
                    f"{file_stem}: {key}={val} {direction} (z={z_score:.1f}, μ={mean:.1f}, σ={stdev:.1f})"
                )

    # Degenerate graphs
    for file_stem, result in agg.per_graph.items():
        s = result.stats
        if not s:
            continue
        if s.get("n_facts", 0) == 0 and s.get("n_nodes_total", 0) > 0:
            agg.anomalies.append(f"{file_stem}: 0 facts ({s.get('n_nodes_total', 0)} total nodes)")
        if s.get("n_edges", 0) == 0 and s.get("n_nodes_total", 0) > 2:
            agg.anomalies.append(f"{file_stem}: 0 edges but {s.get('n_nodes_total', 0)} nodes")
        if s.get("n_nodes_total", 0) == 0:
            agg.anomalies.append(f"{file_stem}: completely empty graph")

    return agg


# =============================================================================
# PYTEST INTEGRATION
# =============================================================================

_GRAPH_DIR = os.environ.get("ILTUR_GRAPH_DIR", "iltur_graphs")
_STRICT = os.environ.get("ILTUR_STRICT", "0") == "1"
_GRAPHS = None
_AGG = None


def _get_graphs():
    global _GRAPHS
    if _GRAPHS is None:
        try:
            _GRAPHS = load_graphs(_GRAPH_DIR)
        except FileNotFoundError:
            _GRAPHS = {}
    return _GRAPHS


def _get_agg():
    global _AGG
    if _AGG is None:
        _AGG = run_aggregate_validation(_get_graphs(), strict=_STRICT)
    return _AGG


def _skip_if_no_graphs():
    """Skip test if no graphs loaded (avoids vacuous passes)."""
    if not _get_graphs():
        try:
            import pytest
            pytest.skip(f"No graph files found in {_GRAPH_DIR}/")
        except ImportError:
            pass  # running outside pytest


def _valid_graphs():
    """Iterate only over non-parse-error graphs."""
    for cid, g in _get_graphs().items():
        if "_parse_error" not in g:
            yield cid, g


def _collect_node_ids(g: dict) -> Set[str]:
    ids = set()
    for lst_name in ("facts", "concepts", "issues", "arguments",
                     "holdings", "precedents", "justification_sets"):
        node_list = g.get(lst_name, [])
        if not isinstance(node_list, list):
            continue
        for node in node_list:
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                ids.add(node["id"])
    outcome = g.get("outcome")
    if isinstance(outcome, dict):
        ids.add(outcome.get("id", "outcome"))
    return ids


def _build_id_type_map(g: dict) -> Dict[str, str]:
    m = {}
    for lst_name, ntype in [("facts", "fact"), ("concepts", "concept"),
                             ("issues", "issue"), ("arguments", "argument"),
                             ("holdings", "holding"), ("precedents", "precedent"),
                             ("justification_sets", "justification_set")]:
        node_list = g.get(lst_name, [])
        if not isinstance(node_list, list):
            continue
        for node in node_list:
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                m[node["id"]] = ntype
    outcome = g.get("outcome")
    if isinstance(outcome, dict):
        m[outcome.get("id", "outcome")] = "outcome"
    return m


def _iter_anchors(g: dict):
    for lst_name in ("facts", "concepts", "issues", "arguments",
                     "holdings", "precedents"):
        node_list = g.get(lst_name, [])
        if not isinstance(node_list, list):
            continue
        for i, node in enumerate(node_list):
            if not isinstance(node, dict):
                continue
            for field_name in ("anchor", "interpretation_anchor", "court_response_anchor"):
                a = node.get(field_name)
                if isinstance(a, dict):
                    yield a, f"{lst_name}[{i}].{field_name}"
    outcome = g.get("outcome")
    if isinstance(outcome, dict):
        a = outcome.get("anchor")
        if isinstance(a, dict):
            yield a, "outcome.anchor"
    edges_list = g.get("edges", [])
    if isinstance(edges_list, list):
        for i, e in enumerate(edges_list):
            if isinstance(e, dict):
                a = e.get("anchor")
                if isinstance(a, dict):
                    yield a, f"edges[{i}].anchor"


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestJsonParsing:
    """All output files must parse as valid JSON."""

    def test_at_least_one_graph_loaded(self):
        assert len(_get_graphs()) > 0, f"No graph files found in {_GRAPH_DIR}/"

    def test_no_parse_failures(self):
        _skip_if_no_graphs()
        agg = _get_agg()
        failures = {cid: r.errors[0] for cid, r in agg.per_graph.items()
                     if any("Parse error" in e for e in r.errors)}
        assert len(failures) == 0, (
            f"{len(failures)} parse failures:\n" +
            "\n".join(f"  {k}: {v}" for k, v in list(failures.items())[:10])
        )

    def test_all_files_are_dicts(self):
        _skip_if_no_graphs()
        for cid, g in _valid_graphs():
            assert isinstance(g, dict), f"{cid}: root is {type(g).__name__}, expected dict"


class TestTopLevelStructure:
    """Every graph must have required top-level fields."""

    def test_case_id_present_and_string(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            if "case_id" not in g:
                bad.append(f"{cid}: missing")
            elif not isinstance(g["case_id"], str):
                bad.append(f"{cid}: type={type(g['case_id']).__name__}")
        assert not bad, f"case_id problems: {bad[:10]}"

    def test_no_duplicate_case_ids_across_files(self):
        _skip_if_no_graphs()
        agg = _get_agg()
        assert not agg.duplicate_case_ids, (
            f"{len(agg.duplicate_case_ids)} duplicate case_ids: {agg.duplicate_case_ids[:10]}"
        )

    def test_meta_block_present(self):
        _skip_if_no_graphs()
        missing = [cid for cid, g in _valid_graphs()
                    if not isinstance(g.get("_meta"), dict) or not g["_meta"]]
        assert not missing, f"{len(missing)} graphs missing _meta: {missing[:10]}"

    def test_schema_version(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            meta = g.get("_meta", {})
            if isinstance(meta, dict):
                sv = str(meta.get("schema_version", ""))
                if sv and not sv.startswith("2."):
                    bad.append((cid, sv))
        assert not bad, f"Unexpected schema versions: {bad[:10]}"

    def test_quality_tier_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            qt = (g.get("_meta") or {}).get("quality_tier")
            if qt and qt not in QUALITY_TIERS:
                bad.append((cid, qt))
        assert not bad, f"Invalid quality tiers: {bad[:10]}"

    def test_required_lists_present(self):
        _skip_if_no_graphs()
        required = ["facts", "concepts", "issues", "arguments", "holdings", "precedents", "edges"]
        bad = []
        for cid, g in _valid_graphs():
            for fn in required:
                if fn not in g:
                    bad.append(f"{cid}: missing '{fn}'")
                elif not isinstance(g[fn], list):
                    bad.append(f"{cid}: '{fn}' is {type(g[fn]).__name__}")
        assert not bad, f"Required list problems:\n" + "\n".join(bad[:15])

    def test_case_year_type(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            cy = g.get("case_year")
            if cy is not None and not isinstance(cy, int):
                bad.append((cid, type(cy).__name__, cy))
        assert not bad, f"Non-int case_year: {bad[:10]}"

    def test_judges_type(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            j = g.get("judges")
            if j is not None:
                if not isinstance(j, list):
                    bad.append((cid, f"judges is {type(j).__name__}"))
                else:
                    for i, item in enumerate(j):
                        if not isinstance(item, str):
                            bad.append((cid, f"judges[{i}] is {type(item).__name__}"))
                            break
        assert not bad, f"judges type problems: {bad[:10]}"

    def test_case_id_matches_filename(self):
        """The runner saves as {case_id}.json, so internal case_id must match."""
        _skip_if_no_graphs()
        mismatches = []
        for file_stem, g in _valid_graphs():
            internal_id = g.get("case_id")
            if isinstance(internal_id, str) and internal_id != file_stem:
                mismatches.append((file_stem, internal_id))
        if mismatches:
            pct = len(mismatches) / sum(1 for _ in _valid_graphs()) * 100
            assert pct < 5, (
                f"{len(mismatches)} case_id/filename mismatches ({pct:.0f}%): {mismatches[:10]}"
            )

    def test_meta_extraction_timestamp_type(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            meta = g.get("_meta")
            if isinstance(meta, dict):
                ts = meta.get("extraction_timestamp")
                if ts is not None and not isinstance(ts, str):
                    bad.append((cid, type(ts).__name__))
        assert not bad, f"Non-string extraction_timestamps: {bad[:10]}"

    def test_meta_validation_warnings_type(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            meta = g.get("_meta")
            if isinstance(meta, dict):
                vw = meta.get("validation_warnings")
                if vw is not None:
                    if not isinstance(vw, list):
                        bad.append((cid, f"not a list: {type(vw).__name__}"))
                    else:
                        for i, w in enumerate(vw):
                            if not isinstance(w, str):
                                bad.append((cid, f"[{i}] is {type(w).__name__}"))
                                break
        assert not bad, f"validation_warnings type problems: {bad[:10]}"


class TestNodeIntegrity:
    """All nodes must have valid IDs, types, and required fields."""

    def test_no_duplicate_node_ids(self):
        _skip_if_no_graphs()
        dups = []
        for cid, g in _valid_graphs():
            seen = set()
            for lst_name in ("facts", "concepts", "issues", "arguments",
                             "holdings", "precedents", "justification_sets"):
                for node in g.get(lst_name, []):
                    if isinstance(node, dict) and isinstance(node.get("id"), str):
                        if node["id"] in seen:
                            dups.append((cid, node["id"]))
                        seen.add(node["id"])
        assert not dups, f"{len(dups)} duplicate IDs: {dups[:15]}"

    def test_all_facts_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, f in enumerate(g.get("facts", [])):
                if not isinstance(f, dict):
                    bad.append((cid, i, "not a dict")); continue
                if not _is_nonempty_string(f.get("id")):
                    bad.append((cid, i, "missing id"))
                if not _is_nonempty_string(f.get("text")):
                    bad.append((cid, i, "empty text"))
                if f.get("fact_type") not in FACT_TYPES:
                    bad.append((cid, i, f"bad fact_type: {f.get('fact_type')}"))
                if f.get("confidence") not in CONFIDENCE_LEVELS:
                    bad.append((cid, i, f"bad confidence: {f.get('confidence')}"))
        assert not bad, f"{len(bad)} fact errors: {bad[:15]}"

    def test_all_concepts_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, c in enumerate(g.get("concepts", [])):
                if not isinstance(c, dict):
                    bad.append((cid, i, "not a dict")); continue
                if not _is_nonempty_string(c.get("id")):
                    bad.append((cid, i, "missing id"))
                if c.get("relevance") not in RELEVANCE_LEVELS:
                    bad.append((cid, i, f"bad relevance: {c.get('relevance')}"))
                if c.get("confidence") not in CONFIDENCE_LEVELS:
                    bad.append((cid, i, f"bad confidence: {c.get('confidence')}"))
        assert not bad, f"{len(bad)} concept errors: {bad[:15]}"

    def test_all_issues_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, iss in enumerate(g.get("issues", [])):
                if not isinstance(iss, dict):
                    bad.append((cid, i, "not a dict")); continue
                if not _is_nonempty_string(iss.get("id")):
                    bad.append((cid, i, "missing id"))
                if not _is_nonempty_string(iss.get("text")):
                    bad.append((cid, i, "empty text"))
                if iss.get("confidence") not in CONFIDENCE_LEVELS:
                    bad.append((cid, i, f"bad confidence: {iss.get('confidence')}"))
        assert not bad, f"{len(bad)} issue errors: {bad[:15]}"

    def test_all_arguments_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, a in enumerate(g.get("arguments", [])):
                if not isinstance(a, dict):
                    bad.append((cid, i, "not a dict")); continue
                if not _is_nonempty_string(a.get("id")):
                    bad.append((cid, i, "missing id"))
                if not _is_nonempty_string(a.get("claim")):
                    bad.append((cid, i, "empty claim"))
                if a.get("actor") not in ACTOR_TYPES:
                    bad.append((cid, i, f"bad actor: {a.get('actor')}"))
                schemes = a.get("schemes")
                if not isinstance(schemes, list):
                    bad.append((cid, i, "schemes not list"))
                elif schemes:
                    for s in schemes:
                        if s not in ARGUMENT_SCHEMES:
                            bad.append((cid, i, f"bad scheme: {s}")); break
                if a.get("confidence") not in CONFIDENCE_LEVELS:
                    bad.append((cid, i, f"bad confidence: {a.get('confidence')}"))
        assert not bad, f"{len(bad)} argument errors: {bad[:15]}"

    def test_all_holdings_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, h in enumerate(g.get("holdings", [])):
                if not isinstance(h, dict):
                    bad.append((cid, i, "not a dict")); continue
                if not _is_nonempty_string(h.get("id")):
                    bad.append((cid, i, "missing id"))
                if not _is_nonempty_string(h.get("text")):
                    bad.append((cid, i, "empty text"))
                if h.get("confidence") not in CONFIDENCE_LEVELS:
                    bad.append((cid, i, f"bad confidence: {h.get('confidence')}"))
        assert not bad, f"{len(bad)} holding errors: {bad[:15]}"

    def test_all_precedents_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, p in enumerate(g.get("precedents", [])):
                if not isinstance(p, dict):
                    bad.append((cid, i, "not a dict")); continue
                if not _is_nonempty_string(p.get("id")):
                    bad.append((cid, i, "missing id"))
                if not _is_nonempty_string(p.get("citation")):
                    bad.append((cid, i, "empty citation"))
                if p.get("confidence") not in CONFIDENCE_LEVELS:
                    bad.append((cid, i, f"bad confidence: {p.get('confidence')}"))
        assert not bad, f"{len(bad)} precedent errors: {bad[:15]}"

    def test_no_whitespace_only_ids(self):
        """IDs that are just spaces/tabs will silently break lookups."""
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for lst_name in ("facts", "concepts", "issues", "arguments",
                             "holdings", "precedents", "justification_sets"):
                for node in (g.get(lst_name) or []):
                    if isinstance(node, dict):
                        nid = node.get("id")
                        if isinstance(nid, str) and len(nid) > 0 and len(nid.strip()) == 0:
                            bad.append((cid, lst_name, repr(nid)))
        assert not bad, f"{len(bad)} whitespace-only IDs: {bad[:10]}"

    def test_holding_booleans_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for h in (g.get("holdings") or []):
                if not isinstance(h, dict):
                    continue
                for bf in ("is_ratio", "novel"):
                    val = h.get(bf)
                    if val is not None and not isinstance(val, (bool, int)):
                        bad.append((cid, h.get("id", "?"), bf, type(val).__name__))
                    elif isinstance(val, int) and val not in (0, 1):
                        bad.append((cid, h.get("id", "?"), bf, val))
        assert not bad, f"{len(bad)} bad holding booleans: {bad[:10]}"


class TestEdgeIntegrity:
    """All edges must reference valid nodes with valid relations."""

    def test_no_dangling_edge_sources(self):
        _skip_if_no_graphs()
        dangling = []
        for cid, g in _valid_graphs():
            node_ids = _collect_node_ids(g)
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    src = e.get("source")
                    if isinstance(src, str) and src not in node_ids:
                        dangling.append((cid, e.get("id", "?"), src))
        assert not dangling, f"{len(dangling)} dangling sources: {dangling[:15]}"

    def test_no_dangling_edge_targets(self):
        _skip_if_no_graphs()
        dangling = []
        for cid, g in _valid_graphs():
            node_ids = _collect_node_ids(g)
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    tgt = e.get("target")
                    if isinstance(tgt, str) and tgt not in node_ids:
                        dangling.append((cid, e.get("id", "?"), tgt))
        assert not dangling, f"{len(dangling)} dangling targets: {dangling[:15]}"

    def test_no_self_loops(self):
        _skip_if_no_graphs()
        loops = []
        for cid, g in _valid_graphs():
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    src, tgt = e.get("source"), e.get("target")
                    if isinstance(src, str) and src == tgt:
                        loops.append((cid, e.get("id", "?"), src))
        assert not loops, f"{len(loops)} self-loops: {loops[:15]}"

    def test_no_duplicate_edge_ids(self):
        _skip_if_no_graphs()
        dups = []
        for cid, g in _valid_graphs():
            seen = set()
            for e in g.get("edges", []):
                if isinstance(e, dict) and isinstance(e.get("id"), str):
                    if e["id"] in seen:
                        dups.append((cid, e["id"]))
                    seen.add(e["id"])
        assert not dups, f"{len(dups)} duplicate edge IDs: {dups[:15]}"

    def test_edge_relations_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    rel = e.get("relation")
                    if isinstance(rel, str) and rel not in EDGE_RELATIONS:
                        bad.append((cid, e.get("id", "?"), rel))
        assert not bad, f"{len(bad)} invalid relations: {bad[:15]}"

    def test_edge_confidence_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    conf = e.get("confidence")
                    if conf is not None and conf not in CONFIDENCE_LEVELS:
                        bad.append((cid, e.get("id", "?"), conf))
        assert not bad, f"{len(bad)} invalid confidences: {bad[:15]}"

    def test_edge_strength_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    st = e.get("strength")
                    if st is not None and st not in STRENGTH_VALUES:
                        bad.append((cid, e.get("id", "?"), st))
        assert not bad, f"{len(bad)} invalid strengths: {bad[:15]}"

    def test_edges_have_required_fields(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, e in enumerate(g.get("edges", [])):
                if not isinstance(e, dict):
                    bad.append((cid, i, "not a dict")); continue
                for req in ("id", "source", "target", "relation"):
                    if not e.get(req):
                        bad.append((cid, e.get("id", f"idx:{i}"), f"missing {req}"))
        assert not bad, f"{len(bad)} edges missing required fields: {bad[:15]}"

    def test_support_group_ids_are_lists(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    sgids = e.get("support_group_ids")
                    if sgids is not None and not isinstance(sgids, list):
                        bad.append((cid, e.get("id", "?"), type(sgids).__name__))
        assert not bad, f"{len(bad)} non-list support_group_ids: {bad[:10]}"


class TestJustificationSets:
    """Justification sets must be well-formed and properly wired."""

    def test_js_targets_exist(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            node_ids = _collect_node_ids(g)
            for js in g.get("justification_sets", []):
                if isinstance(js, dict):
                    tid = js.get("target_id")
                    if isinstance(tid, str) and tid not in node_ids:
                        bad.append((cid, js.get("id", "?"), tid))
        assert not bad, f"{len(bad)} JS with missing targets: {bad[:15]}"

    def test_js_logic_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for js in g.get("justification_sets", []):
                if isinstance(js, dict):
                    logic = js.get("logic")
                    if logic not in JUSTIFICATION_LOGIC:
                        bad.append((cid, js.get("id", "?"), logic))
        assert not bad, f"{len(bad)} invalid JS logic: {bad[:15]}"

    def test_support_group_ids_reference_valid_js(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            js_ids = {js.get("id") for js in g.get("justification_sets", [])
                      if isinstance(js, dict) and isinstance(js.get("id"), str)}
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    for sg in (e.get("support_group_ids") or []):
                        if isinstance(sg, str) and sg not in js_ids:
                            bad.append((cid, e.get("id", "?"), sg))
        assert not bad, f"{len(bad)} edges referencing missing JS: {bad[:15]}"

    def test_js_have_required_fields(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, js in enumerate(g.get("justification_sets", [])):
                if not isinstance(js, dict):
                    bad.append((cid, i, "not a dict")); continue
                if not _is_nonempty_string(js.get("id")):
                    bad.append((cid, i, "missing id"))
                if not _is_nonempty_string(js.get("target_id")):
                    bad.append((cid, i, "missing target_id"))
                if js.get("logic") not in JUSTIFICATION_LOGIC:
                    bad.append((cid, i, f"bad logic: {js.get('logic')}"))
        assert not bad, f"{len(bad)} JS field errors: {bad[:15]}"


class TestOutcomeWiring:

    def test_outcome_has_valid_disposition(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            outcome = g.get("outcome")
            if isinstance(outcome, dict):
                if outcome.get("disposition") not in DISPOSITIONS:
                    bad.append((cid, outcome.get("disposition")))
        assert not bad, f"{len(bad)} invalid dispositions: {bad[:10]}"

    def test_outcome_has_valid_binary(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            outcome = g.get("outcome")
            if isinstance(outcome, dict):
                if outcome.get("binary") not in OUTCOME_BINARY:
                    bad.append((cid, outcome.get("binary")))
        assert not bad, f"{len(bad)} invalid binary: {bad[:10]}"

    def test_disposition_binary_consistency(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            outcome = g.get("outcome")
            if isinstance(outcome, dict):
                disp = outcome.get("disposition")
                binary = outcome.get("binary")
                if disp in DISPOSITION_TO_BINARY and binary in OUTCOME_BINARY:
                    if binary != DISPOSITION_TO_BINARY[disp]:
                        bad.append((cid, disp, binary))
        n_outcomes = sum(1 for _, g in _valid_graphs() if isinstance(g.get("outcome"), dict))
        if n_outcomes > 0 and bad:
            pct = len(bad) / n_outcomes * 100
            assert pct < 15, f"{len(bad)}/{n_outcomes} ({pct:.0f}%) disposition↔binary mismatches: {bad[:10]}"

    def test_outcome_has_determines_edge(self):
        _skip_if_no_graphs()
        missing = []
        for cid, g in _valid_graphs():
            outcome = g.get("outcome")
            if not isinstance(outcome, dict):
                continue
            oid = outcome.get("id", "outcome")
            has_det = any(
                isinstance(e, dict) and e.get("target") == oid and e.get("relation") == "determines"
                for e in g.get("edges", [])
            )
            if not has_det:
                missing.append(cid)
        n_with = sum(1 for _, g in _valid_graphs() if isinstance(g.get("outcome"), dict))
        if n_with > 0:
            pct = len(missing) / n_with * 100
            assert pct < 50, f"{len(missing)}/{n_with} ({pct:.0f}%) outcomes lack DETERMINES edge"


class TestReasoningChains:

    def test_chain_node_references_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            node_ids = _collect_node_ids(g)
            for rc in g.get("reasoning_chains", []):
                if not isinstance(rc, dict):
                    continue
                rcid = rc.get("id", "?")
                for ref_field in ("fact_ids", "concept_ids", "argument_ids", "critical_nodes"):
                    for ref in (rc.get(ref_field) or []):
                        if isinstance(ref, str) and ref not in node_ids:
                            bad.append((cid, rcid, ref_field, ref))
                for scalar in ("issue_id", "holding_id"):
                    val = rc.get(scalar)
                    if isinstance(val, str) and val not in node_ids:
                        bad.append((cid, rcid, scalar, val))
        assert not bad, f"{len(bad)} broken chain node refs: {bad[:15]}"

    def test_chain_edge_references_valid(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            edge_ids = {e.get("id") for e in g.get("edges", [])
                        if isinstance(e, dict) and isinstance(e.get("id"), str)}
            for rc in g.get("reasoning_chains", []):
                if not isinstance(rc, dict):
                    continue
                for ref in (rc.get("edge_ids") or []):
                    if isinstance(ref, str) and ref not in edge_ids:
                        bad.append((cid, rc.get("id", "?"), ref))
        assert not bad, f"{len(bad)} broken chain edge refs: {bad[:15]}"

    def test_chain_list_fields_are_lists(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for rc in g.get("reasoning_chains", []):
                if not isinstance(rc, dict):
                    continue
                for lf in ("fact_ids", "concept_ids", "argument_ids", "edge_ids", "critical_nodes"):
                    val = rc.get(lf)
                    if val is not None and not isinstance(val, list):
                        bad.append((cid, rc.get("id", "?"), lf, type(val).__name__))
        assert not bad, f"{len(bad)} chain fields not lists: {bad[:15]}"

    def test_chain_justification_set_references(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            js_ids = {js.get("id") for js in (g.get("justification_sets") or [])
                      if isinstance(js, dict) and isinstance(js.get("id"), str)}
            for rc in (g.get("reasoning_chains") or []):
                if not isinstance(rc, dict):
                    continue
                js_ref = rc.get("justification_set_id")
                if isinstance(js_ref, str) and js_ref not in js_ids:
                    bad.append((cid, rc.get("id", "?"), js_ref))
        assert not bad, f"{len(bad)} broken chain→JS refs: {bad[:15]}"

    def test_chains_have_ids(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for i, rc in enumerate(g.get("reasoning_chains") or []):
                if isinstance(rc, dict) and not _is_nonempty_string(rc.get("id")):
                    bad.append((cid, i))
        assert not bad, f"{len(bad)} chains missing id: {bad[:15]}"


class TestAnchorValidity:

    def test_no_negative_start_chars(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for anchor, path in _iter_anchors(g):
                sc = anchor.get("start_char")
                if isinstance(sc, (int, float)) and sc < 0:
                    bad.append((cid, path, sc))
        assert not bad, f"{len(bad)} negative start_chars: {bad[:15]}"

    def test_end_char_gte_start_char(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for anchor, path in _iter_anchors(g):
                sc = anchor.get("start_char")
                ec = anchor.get("end_char")
                if isinstance(sc, (int, float)) and isinstance(ec, (int, float)) and ec < sc:
                    bad.append((cid, path, sc, ec))
        assert not bad, f"{len(bad)} end_char < start_char: {bad[:15]}"

    def test_no_unreasonably_large_spans(self):
        _skip_if_no_graphs()
        big = []
        total = 0
        for cid, g in _valid_graphs():
            for anchor, path in _iter_anchors(g):
                total += 1
                sc = anchor.get("start_char")
                ec = anchor.get("end_char")
                if isinstance(sc, (int, float)) and isinstance(ec, (int, float)) and (ec - sc) > 50000:
                    big.append((cid, path, ec - sc))
        if total > 0:
            pct = len(big) / total * 100
            assert pct < 5, f"{len(big)}/{total} ({pct:.1f}%) anchors > 50K chars"

    def test_anchors_have_doc_id(self):
        _skip_if_no_graphs()
        bad = []
        total = 0
        for cid, g in _valid_graphs():
            for anchor, path in _iter_anchors(g):
                total += 1
                did = anchor.get("doc_id")
                if did is None or not isinstance(did, str):
                    bad.append((cid, path))
        if total > 0 and bad:
            pct = len(bad) / total * 100
            assert pct < 10, f"{pct:.0f}% anchors missing doc_id: {bad[:10]}"

    def test_anchor_offsets_are_integers(self):
        """Char offsets should be ints, not floats or strings."""
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for anchor, path in _iter_anchors(g):
                for field in ("start_char", "end_char"):
                    val = anchor.get(field)
                    if val is not None and not isinstance(val, int):
                        bad.append((cid, path, field, type(val).__name__))
        if bad:
            total = sum(1 for _, g in _valid_graphs() for _ in _iter_anchors(g)) * 2
            pct = len(bad) / max(total, 1) * 100
            assert pct < 5, f"{len(bad)} non-int offsets ({pct:.1f}%): {bad[:10]}"


class TestAggregateStats:

    def test_majority_pass_validation(self):
        _skip_if_no_graphs()
        agg = _get_agg()
        rate = agg.validation_passed / max(agg.total_graphs, 1) * 100
        assert rate > 50, f"Only {rate:.0f}% ({agg.validation_passed}/{agg.total_graphs}) pass"

    def test_average_facts_reasonable(self):
        _skip_if_no_graphs()
        vals = _get_agg().stat_distributions.get("n_facts", [])
        if len(vals) < 5:
            return
        mean = statistics.mean(vals)
        assert 1.0 <= mean <= 100, f"Average facts/graph = {mean:.1f}, expected [1, 100]"

    def test_average_edges_reasonable(self):
        _skip_if_no_graphs()
        vals = _get_agg().stat_distributions.get("n_edges", [])
        if len(vals) < 5:
            return
        assert statistics.mean(vals) >= 1.0, f"Average edges/graph = {statistics.mean(vals):.1f}"

    def test_average_holdings_nonzero(self):
        _skip_if_no_graphs()
        vals = _get_agg().stat_distributions.get("n_holdings", [])
        if len(vals) < 5:
            return
        assert statistics.mean(vals) >= 0.5, f"Average holdings/graph = {statistics.mean(vals):.1f}"

    def test_not_all_reject_tier(self):
        _skip_if_no_graphs()
        agg = _get_agg()
        non_reject = sum(v for k, v in agg.quality_tier_dist.items() if k != "reject")
        assert non_reject > 0, "All graphs are quality_tier='reject'"

    def test_outcome_extraction_rate(self):
        _skip_if_no_graphs()
        n_valid = sum(1 for _ in _valid_graphs())
        n_with = sum(1 for _, g in _valid_graphs() if isinstance(g.get("outcome"), dict))
        if n_valid > 0:
            assert n_with / n_valid > 0.5, f"Only {n_with}/{n_valid} graphs have outcome"

    def test_report_anomalies(self):
        """Informational: prints z-score anomalies. Never fails — for human review only."""
        _skip_if_no_graphs()
        agg = _get_agg()
        if agg.anomalies:
            print(f"\n⚠ {len(agg.anomalies)} anomalies:")
            for a in agg.anomalies[:25]:
                print(f"  • {a}")


class TestSemanticConstraints:

    def test_every_argument_has_valid_actor(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for a in g.get("arguments", []):
                if isinstance(a, dict) and a.get("actor") not in ACTOR_TYPES:
                    bad.append((cid, a.get("id", "?"), a.get("actor")))
        assert not bad, f"{len(bad)} arguments with bad actor: {bad[:10]}"

    def test_holdings_resolve_existing_issues(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            issue_ids = {iss.get("id") for iss in g.get("issues", [])
                         if isinstance(iss, dict) and isinstance(iss.get("id"), str)}
            for h in g.get("holdings", []):
                if isinstance(h, dict):
                    ri = h.get("resolves_issue")
                    if isinstance(ri, str) and ri not in issue_ids:
                        bad.append((cid, h.get("id", "?"), ri))
        assert not bad, f"{len(bad)} holdings→nonexistent issues: {bad[:10]}"

    def test_edge_type_compatibility(self):
        """Target type constraints: DETERMINES→outcome, RESOLVES→issue, etc."""
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            id_type = _build_id_type_map(g)
            for e in g.get("edges", []):
                if not isinstance(e, dict):
                    continue
                rel = e.get("relation")
                tgt_type = id_type.get(e.get("target"))
                if not isinstance(rel, str) or not tgt_type:
                    continue
                expected = EDGE_TARGET_CONSTRAINTS.get(rel)
                if expected and tgt_type not in expected:
                    bad.append((cid, e.get("id", "?"), rel, f"→{tgt_type}"))
        if bad:
            total_edges = sum(len(g.get("edges", [])) for _, g in _valid_graphs())
            pct = len(bad) / max(total_edges, 1) * 100
            assert pct < 10, f"{len(bad)} target-type violations ({pct:.1f}%): {bad[:10]}"

    def test_edge_source_type_compatibility(self):
        """Source type constraints: DETERMINES←holding, RESOLVES←holding, etc."""
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            id_type = _build_id_type_map(g)
            for e in g.get("edges", []):
                if not isinstance(e, dict):
                    continue
                rel = e.get("relation")
                src_type = id_type.get(e.get("source"))
                if not isinstance(rel, str) or not src_type:
                    continue
                expected = EDGE_SOURCE_CONSTRAINTS.get(rel)
                if expected and src_type not in expected:
                    bad.append((cid, e.get("id", "?"), rel, f"{src_type}→"))
        if bad:
            total_edges = sum(len(g.get("edges", [])) for _, g in _valid_graphs())
            pct = len(bad) / max(total_edges, 1) * 100
            assert pct < 10, f"{len(bad)} source-type violations ({pct:.1f}%): {bad[:10]}"

    def test_concepts_have_concept_id(self):
        _skip_if_no_graphs()
        missing = []
        for cid, g in _valid_graphs():
            for c in g.get("concepts", []):
                if isinstance(c, dict) and not _is_nonempty_string(c.get("concept_id")):
                    missing.append((cid, c.get("id", "?")))
        n = sum(len(g.get("concepts", [])) for _, g in _valid_graphs())
        if n > 0:
            pct = len(missing) / n * 100
            assert pct < 20, f"{pct:.0f}% concepts lack concept_id: {missing[:10]}"

    def test_precedents_have_citation(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            for p in g.get("precedents", []):
                if isinstance(p, dict) and not _is_nonempty_string(p.get("citation")):
                    bad.append((cid, p.get("id", "?")))
        assert not bad, f"{len(bad)} precedents without citation: {bad[:10]}"


class TestGraphConnectivity:

    def test_graphs_with_nodes_have_edges(self):
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            nn = len(_collect_node_ids(g))
            ne = len(g.get("edges", []))
            if nn > 3 and ne == 0:
                bad.append((cid, nn))
        n_valid = sum(1 for _ in _valid_graphs())
        if bad:
            pct = len(bad) / max(n_valid, 1) * 100
            assert pct < 10, f"{len(bad)} ({pct:.0f}%) graphs: nodes but 0 edges: {bad[:10]}"

    def test_orphan_rate_acceptable(self):
        _skip_if_no_graphs()
        total_nodes = 0
        total_orphans = 0
        for cid, g in _valid_graphs():
            node_ids = _collect_node_ids(g)
            connected = set()
            for e in g.get("edges", []):
                if isinstance(e, dict):
                    s, t = e.get("source"), e.get("target")
                    if isinstance(s, str):
                        connected.add(s)
                    if isinstance(t, str):
                        connected.add(t)
            id_type = _build_id_type_map(g)
            for nid in node_ids:
                if id_type.get(nid) in ("outcome", "justification_set"):
                    continue
                total_nodes += 1
                if nid not in connected:
                    total_orphans += 1
        if total_nodes > 0:
            pct = total_orphans / total_nodes * 100
            assert pct < 30, f"{pct:.0f}% orphans ({total_orphans}/{total_nodes})"

    def test_no_completely_empty_graphs(self):
        _skip_if_no_graphs()
        empty = []
        for cid, g in _valid_graphs():
            total = sum(len(g.get(k, [])) for k in
                        ("facts", "concepts", "issues", "arguments", "holdings", "precedents"))
            if total == 0:
                empty.append(cid)
        assert not empty, f"{len(empty)} completely empty graphs: {empty[:10]}"

    def test_holdings_imply_edges(self):
        """If a graph has holdings, it must have edges connecting them."""
        _skip_if_no_graphs()
        bad = []
        for cid, g in _valid_graphs():
            n_holdings = len(g.get("holdings") or [])
            n_edges = len(g.get("edges") or [])
            if n_holdings > 0 and n_edges == 0:
                bad.append((cid, n_holdings))
        if bad:
            n_valid = sum(1 for _ in _valid_graphs())
            pct = len(bad) / max(n_valid, 1) * 100
            assert pct < 10, f"{len(bad)} graphs with holdings but 0 edges: {bad[:10]}"


# =============================================================================
# CLI REPORT GENERATOR
# =============================================================================

def generate_report(agg: AggregateResult) -> dict:
    stat_summary = {}
    for key, vals in agg.stat_distributions.items():
        if vals:
            stat_summary[key] = {
                "min": min(vals), "max": max(vals),
                "mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
            }

    failures = {}
    for cid, r in agg.per_graph.items():
        if not r.passed:
            failures[cid] = {"errors": r.errors[:20], "n_errors": len(r.errors), "n_warnings": len(r.warnings)}

    return {
        "summary": {
            "total_graphs": agg.total_graphs, "parsed_ok": agg.parsed_ok,
            "parse_failures": agg.parse_failures,
            "validation_passed": agg.validation_passed, "validation_failed": agg.validation_failed,
            "pass_rate_pct": round(agg.validation_passed / max(agg.total_graphs, 1) * 100, 1),
            "total_errors": agg.total_errors, "total_warnings": agg.total_warnings,
        },
        "quality_distribution": dict(agg.quality_tier_dist),
        "disposition_distribution": dict(agg.disposition_dist),
        "stat_summary": stat_summary,
        "top_error_patterns": dict(agg.error_frequency.most_common(25)),
        "top_edge_relations": dict(agg.edge_relation_dist.most_common(15)),
        "top_argument_schemes": dict(agg.argument_scheme_dist.most_common(15)),
        "fact_type_distribution": dict(agg.fact_type_dist),
        "confidence_distribution": dict(agg.confidence_dist),
        "anomalies": agg.anomalies[:50],
        "duplicate_case_ids": [(c, f) for c, f in agg.duplicate_case_ids],
        "failures": failures,
    }


def print_report(report: dict):
    s = report["summary"]
    print("\n" + "=" * 72)
    print("  IL-TUR LEGAL REASONING GRAPH — VALIDATION REPORT")
    print("=" * 72)
    print(f"\n  Total graphs:       {s['total_graphs']}")
    print(f"  Parsed OK:          {s['parsed_ok']}")
    print(f"  Parse failures:     {s['parse_failures']}")
    print(f"  Validation passed:  {s['validation_passed']}")
    print(f"  Validation failed:  {s['validation_failed']}")
    print(f"  Pass rate:          {s['pass_rate_pct']}%")
    print(f"  Total errors:       {s['total_errors']}")
    print(f"  Total warnings:     {s['total_warnings']}")

    print(f"\n{'─'*72}\n  QUALITY DISTRIBUTION\n{'─'*72}")
    for tier in ("gold", "silver", "bronze", "reject"):
        count = report["quality_distribution"].get(tier, 0)
        print(f"    {tier:8s}: {count:4d}  {'█' * (count // 2)}")

    print(f"\n{'─'*72}\n  DISPOSITION DISTRIBUTION\n{'─'*72}")
    for disp, count in sorted(report["disposition_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {disp:18s}: {count:4d}  {'█' * (count // 2)}")

    print(f"\n{'─'*72}\n  NODE/EDGE STATISTICS (per graph)\n{'─'*72}")
    for key, vals in sorted(report["stat_summary"].items()):
        print(f"    {key:28s}: μ={vals['mean']:7.1f}  σ={vals['stdev']:6.1f}  "
              f"[{vals['min']:.0f}, {vals['median']:.0f}, {vals['max']:.0f}]")

    if report["top_error_patterns"]:
        print(f"\n{'─'*72}\n  TOP ERROR PATTERNS\n{'─'*72}")
        for pattern, count in list(report["top_error_patterns"].items())[:15]:
            print(f"    {count:4d}× {pattern}")

    if report.get("duplicate_case_ids"):
        print(f"\n{'─'*72}\n  DUPLICATE CASE IDS\n{'─'*72}")
        for cid, files in report["duplicate_case_ids"]:
            print(f"    '{cid}' in: {files}")

    if report["anomalies"]:
        print(f"\n{'─'*72}\n  STATISTICAL ANOMALIES ({len(report['anomalies'])})\n{'─'*72}")
        for a in report["anomalies"][:15]:
            print(f"    ⚠ {a}")

    if report["failures"]:
        n_show = min(10, len(report["failures"]))
        print(f"\n{'─'*72}\n  SAMPLE FAILURES ({n_show}/{len(report['failures'])})\n{'─'*72}")
        for cid, info in list(report["failures"].items())[:n_show]:
            print(f"\n  [{cid}] ({info['n_errors']} errors, {info['n_warnings']} warnings)")
            for err in info["errors"][:5]:
                print(f"      ✗ {err}")

    print(f"\n{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate IL-TUR Legal Reasoning Graph outputs")
    parser.add_argument("--dir", default="iltur_graphs", help="Graph directory")
    parser.add_argument("--strict", action="store_true", help="Warnings become errors")
    parser.add_argument("--report", default=None, help="Save JSON report")
    parser.add_argument("--quiet", action="store_true", help="Summary only")
    args = parser.parse_args()

    print(f"Loading graphs from {args.dir}/ ...")
    try:
        graphs = load_graphs(args.dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loaded {len(graphs)} graph files.")
    agg = run_aggregate_validation(graphs, strict=args.strict)
    report = generate_report(agg)

    if not args.quiet:
        print_report(report)
    else:
        s = report["summary"]
        print(f"✓ {s['validation_passed']}/{s['total_graphs']} passed "
              f"({s['pass_rate_pct']}%), {s['total_errors']} errors, {s['total_warnings']} warnings")

    if args.report:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.report}")

    sys.exit(0 if report["summary"]["pass_rate_pct"] >= 80 else 1)


if __name__ == "__main__":
    main()
