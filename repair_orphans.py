#!/usr/bin/env python3
"""
repair_orphans.py

Post-processing pass that reduces orphan nodes in extracted legal reasoning graphs.
Reads each JSON from the output directory, identifies orphan nodes, and uses
heuristic matching (keyword overlap + anchor proximity) to create plausible edges.

Design Philosophy:
==================
- NOT all nodes should be connected. Arguments dismissed without consideration,
  background facts, and "mentioned" concepts may legitimately be orphans.
- Only connect when there is genuine textual evidence of a relationship.
- All repair edges are marked confidence=inferred, strength=weak, with provenance.
- Each orphan gets at most 2 new edges (best + runner-up if strong enough).
- Anchor proximity in the source document is a strong signal of relatedness.

Usage:
    python repair_orphans.py [--dir iltur_graphs] [--dry-run] [--min-score 2]
"""

import json
import re
import os
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from copy import deepcopy
from datetime import datetime, timezone

# =============================================================================
# VALID EDGE RELATIONS (mirrored from extractor.py)
# =============================================================================

VALID_EDGE_RELATIONS: Dict[Tuple[str, str], Set[str]] = {
    ("fact", "concept"): {"triggers", "negates", "partially_satisfies", "satisfies", "claims_satisfies"},
    ("fact", "argument"): {"supports", "grounds", "rebuts", "undercuts"},
    ("fact", "holding"): {"supports", "grounds"},
    ("fact", "issue"): {"triggers", "supports", "addresses"},
    ("concept", "concept"): {"requires", "excludes", "specializes", "conflicts_with"},
    ("concept", "argument"): {"supports", "grounds", "rebuts", "undercuts"},
    ("concept", "holding"): {"grounds", "constrains", "supports", "enables"},
    ("concept", "issue"): {"requires", "addresses"},
    ("argument", "issue"): {"addresses", "concedes"},
    ("argument", "argument"): {"attacks", "supports_arg", "responds_to"},
    ("argument", "holding"): {"supports", "grounds", "rebuts", "undercuts"},
    ("argument", "concept"): {"supports", "grounds", "rebuts", "undercuts", "claims_satisfies"},
    ("holding", "issue"): {"resolves", "partially_resolves", "addresses"},
    ("holding", "outcome"): {"determines", "contributes_to"},
    ("holding", "precedent"): {"follows", "applies", "distinguishes", "overrules", "doubts", "explains"},
    ("holding", "concept"): {"supports", "grounds", "constrains", "undercuts", "negates"},
    ("holding", "holding"): {"supports", "conflicts_with", "specializes", "constrains", "undercuts"},
    ("precedent", "concept"): {"supports", "grounds", "establishes"},
    ("precedent", "holding"): {"supports"},
    ("precedent", "argument"): {"supports"},
    ("precedent", "issue"): {"addresses", "supports"},
    ("issue", "concept"): {"requires", "addresses"},
    ("issue", "holding"): {"addresses", "requires"},
    ("issue", "argument"): {"addresses", "requires"},
    ("issue", "precedent"): {"addresses"},
    ("issue", "issue"): {"specializes", "conflicts_with", "requires"},
}

# Default relation to use for each (source_type, target_type) pair
DEFAULT_RELATIONS: Dict[Tuple[str, str], str] = {
    ("fact", "argument"): "supports",
    ("fact", "holding"): "supports",
    ("fact", "issue"): "triggers",
    ("fact", "concept"): "triggers",
    ("concept", "issue"): "addresses",
    ("concept", "holding"): "grounds",
    ("concept", "argument"): "supports",
    ("concept", "concept"): "requires",
    ("argument", "issue"): "addresses",
    ("argument", "holding"): "supports",
    ("argument", "argument"): "responds_to",
    ("argument", "concept"): "supports",
    ("precedent", "argument"): "supports",
    ("precedent", "holding"): "supports",
    ("precedent", "concept"): "supports",
    ("precedent", "issue"): "supports",
    ("holding", "issue"): "addresses",
    ("holding", "holding"): "supports",
    ("holding", "concept"): "grounds",
    ("holding", "precedent"): "follows",
    ("issue", "issue"): "specializes",
    ("issue", "concept"): "requires",
    ("issue", "holding"): "addresses",
    ("issue", "argument"): "addresses",
    ("issue", "precedent"): "addresses",
}

# =============================================================================
# TEXT PROCESSING
# =============================================================================

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "by", "with",
    "without", "is", "was", "were", "are", "be", "been", "being", "as", "at",
    "from", "that", "this", "it", "its", "their", "his", "her", "they", "them",
    "he", "she", "we", "our", "you", "not", "no", "yes", "shall", "may", "must",
    "can", "could", "would", "should", "have", "has", "had", "been", "will",
    "also", "such", "under", "over", "before", "after", "into", "upon", "said",
    "above", "between", "through", "during", "about", "against", "same", "other",
    "than", "more", "any", "all", "each", "every", "some", "these", "those",
    "been", "being", "both", "but", "case", "court", "held", "therefore",
    "whereas", "wherein", "herein", "thereof", "whether", "which", "whom",
    "whose", "section", "article", "order", "rule", "act", "provision",
    "learned", "counsel", "submitted", "contended", "argued", "stated",
    "matter", "question", "point", "respect", "regard", "view", "opinion",
    "finding", "found", "observed", "noted", "considered", "considering",
}

# Extra legal stopwords that appear very frequently but add no discriminating power
_LEGAL_NOISE = {
    "appellant", "respondent", "petitioner", "complainant", "accused",
    "prosecution", "defence", "trial", "appeal", "judgment", "decree",
    "lower", "high", "supreme", "bench", "division", "civil", "criminal",
    "writ", "petition", "application", "revision", "reference",
}

ALL_STOPS = _STOPWORDS | _LEGAL_NOISE


def tokenize(text: str) -> List[str]:
    """Extract lowercase word tokens from text."""
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if t]


def keyword_set(text: str, min_len: int = 4) -> Set[str]:
    """Extract meaningful keywords from text, filtering stopwords."""
    toks = tokenize(text)
    return {t for t in toks if len(t) >= min_len and t not in ALL_STOPS}


def keyword_overlap(text_a: str, text_b: str) -> int:
    """Count overlapping meaningful keywords between two texts."""
    return len(keyword_set(text_a) & keyword_set(text_b))


# =============================================================================
# NODE TYPE INFERENCE
# =============================================================================

def get_node_type_from_id(node_id: str) -> str:
    """Infer node type from ID prefix (mirrors extractor.py)."""
    if node_id == "outcome":
        return "outcome"
    prefix_map = {
        "f": "fact",
        "c": "concept",
        "i": "issue",
        "a": "argument",
        "h": "holding",
        "p": "precedent",
        "js": "justification_set",
    }
    for prefix, ntype in prefix_map.items():
        if node_id.startswith(prefix) and (
            len(node_id) == len(prefix) + 1 or node_id[len(prefix):].isdigit()
        ):
            return ntype
    return "unknown"


def build_type_map(g: dict) -> Dict[str, str]:
    """Build node_id -> type mapping from a graph dict."""
    m = {}
    for lst_name, ntype in [
        ("facts", "fact"), ("concepts", "concept"), ("issues", "issue"),
        ("arguments", "argument"), ("holdings", "holding"),
        ("precedents", "precedent"), ("justification_sets", "justification_set"),
    ]:
        for node in g.get(lst_name, []):
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                m[node["id"]] = ntype
    outcome = g.get("outcome")
    if isinstance(outcome, dict):
        m[outcome.get("id", "outcome")] = "outcome"
    return m


# =============================================================================
# NODE TEXT EXTRACTION
# =============================================================================

def get_node_text(node: dict, ntype: str) -> str:
    """Extract the best available text from a node for matching purposes."""
    parts = []
    if ntype == "fact":
        parts.append(node.get("text", ""))
    elif ntype == "concept":
        parts.append(node.get("concept_id", ""))
        parts.append(node.get("interpretation", "") or "")
        parts.append(node.get("unlisted_label", "") or "")
        parts.append(node.get("unlisted_description", "") or "")
    elif ntype == "issue":
        parts.append(node.get("text", ""))
    elif ntype == "argument":
        parts.append(node.get("claim", ""))
        parts.append(node.get("court_reasoning", "") or "")
    elif ntype == "holding":
        parts.append(node.get("text", ""))
        parts.append(node.get("reasoning_summary", "") or "")
    elif ntype == "precedent":
        parts.append(node.get("citation", ""))
        parts.append(node.get("cited_proposition", "") or "")
        parts.append(node.get("cited_holding", "") or "")
        parts.append(node.get("case_name", "") or "")
    return " ".join(p for p in parts if p)


def get_anchor_midpoint(node: dict) -> Optional[int]:
    """Get the midpoint char offset of a node's anchor, or None."""
    anchor = node.get("anchor")
    if not isinstance(anchor, dict):
        return None
    start = anchor.get("start_char")
    end = anchor.get("end_char")
    if isinstance(start, int) and isinstance(end, int) and end > start >= 0:
        return (start + end) // 2
    return None


# =============================================================================
# LEGITIMATE ORPHAN DETECTION
# =============================================================================

def is_legitimate_orphan(node: dict, ntype: str) -> bool:
    """Return True if this node is plausibly orphaned by design.

    In legal reasoning, many nodes *should* be disconnected:
    - Rejected / not_decided facts don't ground anything
    - Background facts are context, not reasoning steps
    - Arguments the court rejected or didn't address
    - Concepts merely "mentioned" or in obiter dicta
    - Precedents the court only cited in passing
    """
    if ntype == "fact":
        # Rejected or undecided facts don't feed into holdings
        if node.get("court_finding") in ("rejected", "not_decided"):
            return True
        # Background facts are scene-setting, not reasoning
        if node.get("fact_type") == "background":
            return True

    elif ntype == "argument":
        # Arguments the court rejected or ignored are legitimately isolated
        if node.get("court_response") in ("rejected", "not_addressed"):
            return True

    elif ntype == "concept":
        # Mentioned/obiter concepts are referenced but not load-bearing
        if node.get("relevance") in ("mentioned", "obiter"):
            return True

    elif ntype == "precedent":
        # Merely cited precedents (no substantive treatment) can stand alone
        if node.get("treatment") == "cited":
            return True

    return False



def find_orphans(g: dict) -> Tuple[Set[str], Set[str]]:
    """Find orphan node IDs (not source or target of any edge).

    Returns:
        (orphan_ids, connected_ids)
    """
    all_node_ids = set()
    for lst_name in ("facts", "concepts", "issues", "arguments",
                     "holdings", "precedents"):
        for node in g.get(lst_name, []):
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                all_node_ids.add(node["id"])

    connected = set()
    for e in g.get("edges", []):
        if isinstance(e, dict):
            s = e.get("source")
            t = e.get("target")
            if isinstance(s, str):
                connected.add(s)
            if isinstance(t, str):
                connected.add(t)

    orphans = all_node_ids - connected
    return orphans, connected


# =============================================================================
# CANDIDATE SCORING
# =============================================================================

def score_candidate_edge(
    orphan_node: dict,
    orphan_type: str,
    candidate_node: dict,
    candidate_type: str,
    anchor_proximity_bonus: bool = True,
) -> Tuple[float, str]:
    """Score a potential edge between an orphan and a candidate node.

    Returns (score, relation) where score=0 means no valid connection.
    """
    # Check if this type pair has valid relations
    pair = (orphan_type, candidate_type)
    reverse_pair = (candidate_type, orphan_type)

    # Determine direction: orphan can be source or target
    if pair in VALID_EDGE_RELATIONS:
        source_type, target_type = orphan_type, candidate_type
        relation = DEFAULT_RELATIONS.get(pair)
        if not relation:
            relation = sorted(VALID_EDGE_RELATIONS[pair])[0]
    elif reverse_pair in VALID_EDGE_RELATIONS:
        source_type, target_type = candidate_type, orphan_type
        relation = DEFAULT_RELATIONS.get(reverse_pair)
        if not relation:
            relation = sorted(VALID_EDGE_RELATIONS[reverse_pair])[0]
    else:
        return 0.0, ""

    # Compute keyword overlap
    orphan_text = get_node_text(orphan_node, orphan_type)
    candidate_text = get_node_text(candidate_node, candidate_type)
    overlap = keyword_overlap(orphan_text, candidate_text)

    if overlap == 0:
        return 0.0, relation

    score = float(overlap)

    # Anchor proximity bonus: nodes close together in the document
    # are more likely to be related
    if anchor_proximity_bonus:
        orphan_mid = get_anchor_midpoint(orphan_node)
        cand_mid = get_anchor_midpoint(candidate_node)
        if orphan_mid is not None and cand_mid is not None:
            distance = abs(orphan_mid - cand_mid)
            if distance < 300:
                score += 2.0  # Strong proximity
            elif distance < 800:
                score += 1.0  # Moderate proximity
            elif distance < 2000:
                score += 0.5  # Mild proximity

    # Type-specific bonuses

    # Arguments that explicitly address an issue (strong signal)
    if orphan_type == "argument" and candidate_type == "issue":
        # Check if the argument's claim directly references the issue text
        issue_kw = keyword_set(candidate_text)
        claim_kw = keyword_set(orphan_node.get("claim", ""))
        if len(issue_kw & claim_kw) >= 3:
            score += 1.5

    # Facts with court_finding="accepted" are more likely to ground holdings
    if orphan_type == "fact" and candidate_type == "holding":
        if orphan_node.get("court_finding") == "accepted":
            score += 1.0

    # Precedents with explicit treatment are more likely to connect
    if orphan_type == "precedent":
        treatment = orphan_node.get("treatment")
        if treatment in ("followed", "applied", "distinguished"):
            score += 1.0

    # Holdings that resolve the same issue as a concept addresses
    if orphan_type == "concept" and candidate_type == "holding":
        resolves_issue = candidate_node.get("resolves_issue")
        if resolves_issue:
            score += 0.5  # Mild bonus for holdings tied to an issue

    return score, relation


# =============================================================================
# EDGE REPAIR ENGINE
# =============================================================================

def repair_graph(
    g: dict,
    min_score: float = 2.0,
    max_edges_per_orphan: int = 2,
    runner_up_threshold: float = 3.5,
) -> Tuple[dict, int, int]:
    """Repair orphan nodes in a graph by adding heuristic edges.

    Args:
        g: The graph dict (will be deepcopied, not mutated)
        min_score: Minimum score to accept an edge
        max_edges_per_orphan: Maximum edges to add per orphan (1 or 2)
        runner_up_threshold: Score threshold for accepting a second edge

    Returns:
        (repaired_graph, n_orphans_before, n_orphans_after)
    """
    g = deepcopy(g)

    type_map = build_type_map(g)
    orphan_ids, connected_ids = find_orphans(g)

    # Exclude types that the test ignores
    orphan_ids = {
        nid for nid in orphan_ids
        if type_map.get(nid) not in ("outcome", "justification_set")
    }

    if not orphan_ids:
        countable = {
            nid for nid, ntype in type_map.items()
            if ntype not in ("outcome", "justification_set")
        }
        return g, 0, 0

    # Build node index: id -> node dict
    node_index: Dict[str, dict] = {}
    for lst_name in ("facts", "concepts", "issues", "arguments",
                     "holdings", "precedents"):
        for node in g.get(lst_name, []):
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                node_index[node["id"]] = node

    # Collect existing edge signatures to avoid duplicates
    existing_sigs = set()
    # Also track undirected connections: if A→B exists, don't add B→A
    connected_pairs = set()
    for e in g.get("edges", []):
        if isinstance(e, dict):
            sig = (e.get("source"), e.get("target"), e.get("relation"))
            existing_sigs.add(sig)
            s, t = e.get("source"), e.get("target")
            if s and t:
                connected_pairs.add(frozenset((s, t)))

    # Existing edge IDs to avoid collisions
    existing_edge_ids = {
        e.get("id") for e in g.get("edges", [])
        if isinstance(e, dict)
    }

    new_edges = []
    repaired_orphans = set()
    edge_counter = 0

    # Strategy: for each orphan, find the best connected (or other orphan)
    # candidate to link to. Prefer connecting to already-connected nodes
    # since that integrates the orphan into the main graph.

    # Define target type priorities per orphan type
    # (which node types should we try to connect an orphan to?)
    target_priorities = {
        "fact": ["argument", "holding", "issue", "concept"],
        "concept": ["issue", "holding", "argument", "concept"],
        "argument": ["issue", "holding", "argument", "concept"],
        "precedent": ["argument", "holding", "concept", "issue"],
        "issue": ["holding", "concept", "argument", "issue", "precedent"],
        "holding": ["issue", "holding", "concept", "precedent"],
    }

    for orphan_id in sorted(orphan_ids):
        orphan_type = type_map.get(orphan_id)
        if not orphan_type or orphan_type in ("outcome", "justification_set"):
            continue

        orphan_node = node_index.get(orphan_id)
        if not orphan_node:
            continue

        # Skip nodes with very little text (likely extraction artifacts)
        orphan_text = get_node_text(orphan_node, orphan_type)
        if len(orphan_text.strip()) < 10:
            continue

        # Skip nodes that are legitimately orphaned by their semantics
        # (rejected facts, dismissed arguments, mentioned-only concepts, etc.)
        if is_legitimate_orphan(orphan_node, orphan_type):
            continue

        # Score all candidate nodes
        candidates: List[Tuple[float, str, str, str]] = []  # (score, relation, src, tgt)

        priority_types = target_priorities.get(orphan_type, [])
        for cand_id, cand_node in node_index.items():
            if cand_id == orphan_id:
                continue
            cand_type = type_map.get(cand_id)
            if not cand_type or cand_type in ("outcome", "justification_set"):
                continue

            # Prefer connecting to target priority types first
            if cand_type not in priority_types:
                continue

            score, relation = score_candidate_edge(
                orphan_node, orphan_type, cand_node, cand_type
            )

            if score < min_score:
                continue

            # Determine direction
            pair = (orphan_type, cand_type)
            reverse_pair = (cand_type, orphan_type)
            if pair in VALID_EDGE_RELATIONS:
                src, tgt = orphan_id, cand_id
            elif reverse_pair in VALID_EDGE_RELATIONS:
                src, tgt = cand_id, orphan_id
            else:
                continue

            # Boost score for already-connected candidates (integrates orphan better)
            if cand_id in connected_ids:
                score += 0.5

            candidates.append((score, relation, src, tgt))

        if not candidates:
            continue

        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])

        # Add best edge
        added = 0
        for score, relation, src, tgt in candidates:
            if added >= max_edges_per_orphan:
                break
            if added >= 1 and score < runner_up_threshold:
                break

            sig = (src, tgt, relation)
            if sig in existing_sigs:
                continue
            # Avoid bidirectional edges (A→B and B→A)
            pair_key = frozenset((src, tgt))
            if pair_key in connected_pairs:
                continue

            edge_counter += 1
            edge_id = f"e_repair_{edge_counter}"
            while edge_id in existing_edge_ids:
                edge_counter += 1
                edge_id = f"e_repair_{edge_counter}"

            new_edge = {
                "id": edge_id,
                "source": src,
                "target": tgt,
                "relation": relation,
                "anchor": None,
                "explanation": f"Orphan repair: score={score:.1f}, "
                               f"{get_node_type_from_id(src)}->{get_node_type_from_id(tgt)}",
                "confidence": "inferred",
                "strength": "weak" if score < 4.0 else "moderate",
                "support_group_ids": [],
                "is_critical": False,
                "provenance": {
                    "extraction_method": "inference",
                    "model_id": None,
                    "prompt_id": "repair_orphans_v1",
                    "run_id": None,
                    "temperature": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            new_edges.append(new_edge)
            existing_sigs.add(sig)
            existing_edge_ids.add(edge_id)
            connected_pairs.add(pair_key)
            repaired_orphans.add(orphan_id)
            connected_ids.add(src)
            connected_ids.add(tgt)
            added += 1

    # Append new edges to graph
    if new_edges:
        edges_list = g.get("edges", [])
        edges_list.extend(new_edges)
        g["edges"] = edges_list

        # Update validation warnings in _meta
        meta = g.get("_meta", {})
        warnings = meta.get("validation_warnings", [])
        warnings.append(
            f"repair_orphans: added {len(new_edges)} edges for {len(repaired_orphans)} orphan nodes"
        )
        meta["validation_warnings"] = warnings
        g["_meta"] = meta

    # Count orphans after repair
    countable = {
        nid for nid, ntype in type_map.items()
        if ntype not in ("outcome", "justification_set")
    }
    post_connected = set()
    for e in g.get("edges", []):
        if isinstance(e, dict):
            s, t = e.get("source"), e.get("target")
            if isinstance(s, str):
                post_connected.add(s)
            if isinstance(t, str):
                post_connected.add(t)
    orphans_after = len(countable - post_connected)

    return g, len(orphan_ids), orphans_after


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(
    graph_dir: str,
    min_score: float = 2.0,
    dry_run: bool = False,
    max_edges_per_orphan: int = 2,
) -> Dict[str, Any]:
    """Process all JSON graphs in a directory.

    Args:
        graph_dir: Path to directory containing .json graph files
        min_score: Minimum keyword overlap score to accept an edge
        dry_run: If True, don't write files, just report
        max_edges_per_orphan: Max edges to add per orphan node

    Returns:
        Summary statistics dict
    """
    graph_path = Path(graph_dir)
    if not graph_path.exists():
        print(f"Error: directory '{graph_dir}' not found")
        return {}

    json_files = sorted([
        f for f in graph_path.glob("*.json")
        if f.name != "checkpoint.json"
    ])

    if not json_files:
        print(f"No JSON graph files found in '{graph_dir}'")
        return {}

    print(f"Found {len(json_files)} graph files in '{graph_dir}'")
    print(f"Settings: min_score={min_score}, max_edges_per_orphan={max_edges_per_orphan}")
    print(f"Mode: {'DRY RUN' if dry_run else 'WRITE'}")
    print("-" * 70)

    total_orphans_before = 0
    total_orphans_after = 0
    total_nodes = 0
    total_edges_added = 0
    files_modified = 0
    per_file_stats = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                g = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  SKIP {json_file.name}: {e}")
            continue

        if not isinstance(g, dict):
            continue

        # Count countable nodes
        type_map = build_type_map(g)
        countable = sum(
            1 for ntype in type_map.values()
            if ntype not in ("outcome", "justification_set")
        )

        repaired_g, orphans_before, orphans_after = repair_graph(
            g,
            min_score=min_score,
            max_edges_per_orphan=max_edges_per_orphan,
        )

        edges_added = len(repaired_g.get("edges", [])) - len(g.get("edges", []))

        total_nodes += countable
        total_orphans_before += orphans_before
        total_orphans_after += orphans_after
        total_edges_added += edges_added

        per_file_stats.append({
            "file": json_file.name,
            "nodes": countable,
            "orphans_before": orphans_before,
            "orphans_after": orphans_after,
            "edges_added": edges_added,
        })

        if edges_added > 0:
            files_modified += 1
            pct_before = (orphans_before / countable * 100) if countable else 0
            pct_after = (orphans_after / countable * 100) if countable else 0
            print(
                f"  {json_file.stem}: {orphans_before}->{orphans_after} orphans "
                f"({pct_before:.0f}%->{pct_after:.0f}%), +{edges_added} edges"
            )

            if not dry_run:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(repaired_g, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 70)
    print("ORPHAN REPAIR SUMMARY")
    print("=" * 70)

    if total_nodes > 0:
        pct_before = total_orphans_before / total_nodes * 100
        pct_after = total_orphans_after / total_nodes * 100
        print(f"Total nodes:       {total_nodes}")
        print(f"Orphans before:    {total_orphans_before} ({pct_before:.1f}%)")
        print(f"Orphans after:     {total_orphans_after} ({pct_after:.1f}%)")
        print(f"Edges added:       {total_edges_added}")
        print(f"Files modified:    {files_modified}/{len(json_files)}")
        print(f"Target:            <30%")
        print(f"Status:            {'PASS' if pct_after < 30 else 'STILL ABOVE 30%'}")

        if pct_after >= 30:
            print(f"\nTo further reduce orphans, try:")
            print(f"  --min-score 1.5   (lower threshold, more aggressive)")
            print(f"  --max-edges 3     (allow more edges per orphan)")
    else:
        print("No countable nodes found.")

    if dry_run:
        print(f"\n[DRY RUN] No files were modified. Remove --dry-run to apply.")

    return {
        "total_nodes": total_nodes,
        "orphans_before": total_orphans_before,
        "orphans_after": total_orphans_after,
        "pct_before": pct_before if total_nodes else 0,
        "pct_after": pct_after if total_nodes else 0,
        "edges_added": total_edges_added,
        "files_modified": files_modified,
        "per_file": per_file_stats,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Repair orphan nodes in legal reasoning graphs"
    )
    parser.add_argument(
        "--dir", type=str, default="iltur_graphttest", #ill keep it changed to graphttest for a wile
        help="Directory containing .json graph files"
    )
    parser.add_argument(
        "--min-score", type=float, default=2.0,
        help="Minimum keyword overlap score to accept an edge (default: 2.0)"
    )
    parser.add_argument(
        "--max-edges", type=int, default=2,
        help="Maximum edges to add per orphan node (default: 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report changes without writing files"
    )

    args = parser.parse_args()
    process_directory(
        graph_dir=args.dir,
        min_score=args.min_score,
        dry_run=args.dry_run,
        max_edges_per_orphan=args.max_edges,
    )


if __name__ == "__main__":
    main()
