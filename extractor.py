#!/usr/bin/env python3
"""
extractor_v3.py

Legal Reasoning Graph Extractor v3.0 - Production Grade

Key Improvements over v2.1:
===========================
1. CONSISTENT RETRY LOGIC - All passes have retry with error feedback
2. ONTOLOGY INTEGRATION - Loads and uses ontology for concept normalization
3. EDGE RELATION VALIDATION - Validates source/target type compatibility
4. LONG-DISTANCE LINK DISCOVERY - New pass to find implicit reasoning chains
5. BETTER DOCUMENT HANDLING - Chunking for long documents
6. IMPROVED PROMPTS - More precise extraction instructions
7. COMPREHENSIVE LOGGING - Track extraction quality metrics
8. BATCH-READY - Async with semaphore control

Architecture:
=============
    Raw Judgment Text
           │
           ▼
    ┌──────────────────┐
    │  Text Segmenter  │  ← Split into paragraphs with char offsets
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Pass 1: Facts   │  ← Material, procedural, background facts
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Pass 2: Concepts │  ← Ontology-constrained legal concepts
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Pass 3: Issues  │  ← Legal questions framed by court
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Pass 4: Arguments│  ← Party submissions + court reasoning
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Pass 5: Holdings │  ← Court's legal determinations
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Pass 6: Precedents│ ← Cited cases + treatment
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Pass 7: Outcome  │  ← Final disposition
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Pass 8: Edges   │  ← Direct reasoning connections
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Pass 9: LinkDisc │  ← Long-distance link discovery (NEW)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Pass 10: Justify │  ← Justification sets for counterfactual
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   Validation     │  ← Schema compliance + consistency
    └────────┬─────────┘
             │
             ▼
      LegalReasoningGraph
"""

import json
import hashlib
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Literal
from datetime import datetime
from enum import Enum
from pathlib import Path
import asyncio

# Import schema
from schema_v2_1 import (
    SCHEMA_VERSION,
    LegalReasoningGraph,
    FactNode, ConceptNode, IssueNode, ArgumentNode, HoldingNode,
    PrecedentNode, OutcomeNode, JustificationSetNode, Edge,
    ReasoningChain,
    Anchor, Provenance,
    NodeType, ActorType, ConceptKind, FactType, ArgumentScheme,
    EdgeRelation, JustificationLogic, Confidence, Relevance, Disposition,
    ExtractionMethod, PrecedentTreatment
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LegalExtractor")

# =============================================================================
# ANCHOR VALIDITY CONSTANTS
# =============================================================================

# SHA256 hash of empty string - indicates invalid anchor
EMPTY_ANCHOR_HASH = "e3b0c44298fc1c14"


def is_anchor_valid(anchor: Optional[Any]) -> bool:
    """Check if an anchor is semantically valid (not empty/corrupt).

    An anchor is invalid if:
    - It's None
    - text_hash equals empty string hash
    - surface_text is empty or None
    - start_char/end_char are invalid (negative or end <= start)
    """
    if anchor is None:
        return False
    if hasattr(anchor, 'text_hash') and anchor.text_hash == EMPTY_ANCHOR_HASH:
        return False
    if hasattr(anchor, 'surface_text') and not anchor.surface_text:
        return False
    if hasattr(anchor, 'start_char') and hasattr(anchor, 'end_char'):
        if anchor.start_char < 0 or anchor.end_char <= anchor.start_char:
            return False
    return True


# =============================================================================
# EDGE RELATION NORMALIZATION + VALIDATION RULES
# =============================================================================

# The IL-TUR corpus (and LLMs in general) can emit slightly different relation/scheme
# strings (e.g., "claim_satisfies" vs "claims_satisfies"). We normalize first, then validate.

_EDGE_RELATION_ALIASES: Dict[str, str] = {
    # Satisfies / requirement relations
    "claim_satisfies": "claims_satisfies",
    "claims_satisfy": "claims_satisfies",
    "claims_satisfies_requirement": "claims_satisfies",
    "satisfy": "satisfies",
    "satisfies_requirement": "satisfies",
    "satisfies_requirements": "satisfies",
    "satisfies_req": "satisfies",
    "partial_satisfies": "partially_satisfies",
    "partially_satisfies_requirement": "partially_satisfies",

    # Establish / enable relations
    "establish": "establishes",
    "established": "establishes",
    "establishes_doctrine": "establishes",
    "enable": "enables",
    "enabling": "enables",
    "permits": "enables",
    "allows": "enables",
    "facilitates": "enables",

    # Common wording variants
    "conflicts": "conflicts_with",
    "conflict": "conflicts_with",
    "conflict_with": "conflicts_with",
    "respond_to": "responds_to",
    "responds": "responds_to",
    "support": "supports",
    "supports_argument": "supports_arg",
    "supportsarg": "supports_arg",
    "attack": "attacks",
    "undercut": "undercuts",
    "rebut": "rebuts",
    "ground": "grounds",
    "address": "addresses",
    "require": "requires",
}


def normalize_edge_relation(relation: Any) -> str:
    """Normalize relation string to canonical schema values."""
    if relation is None:
        return relation
    r = str(relation).strip().lower()
    r = r.replace("-", "_").replace(" ", "_")
    r = re.sub(r"_+", "_", r)
    return _EDGE_RELATION_ALIASES.get(r, r)


def coerce_edge_relation(relation: Any) -> str:
    """Coerce an arbitrary relation-like string into a valid EdgeRelation value.

    This is a safety net for noisy IL-TUR judgments and occasional LLM drift.
    """
    r = normalize_edge_relation(relation)
    valid_values = {er.value for er in EdgeRelation}
    if r in valid_values:
        return r

    # Heuristic fallbacks
    if "satisf" in r:
        return "satisfies" if "partial" not in r else "partially_satisfies"
    if "enable" in r or "permit" in r or "allow" in r:
        return "enables"
    if "establish" in r:
        return "establishes"
    if "conflict" in r or "contradict" in r or "inconsist" in r:
        return "conflicts_with"
    if "require" in r or "necess" in r:
        return "requires"
    if "resolv" in r or "answer" in r:
        return "resolves"
    if "determin" in r:
        return "determines"
    if "contribut" in r or "cause" in r or "lead" in r or "result" in r:
        return "contributes_to"
    if "attack" in r:
        return "attacks"
    if "rebut" in r:
        return "rebuts"
    if "undercut" in r:
        return "undercuts"

    # Default: treat as generic support
    return "supports"


_SCHEME_ALIASES: Dict[str, str] = {
    # Verbose → canonical
    "textual_interpretation": "textual",
    "purposive_interpretation": "purposive",
    "harmonious_construction": "harmonious",
    "procedural_compliance": "procedural",
    "precedent_distinguishing": "precedent_distinction",
    "policy_consequences": "policy_consequence",

    # Natural justice / procedural fairness
    "natural_justice": "natural_justice",
    "naturaljustice": "natural_justice",
    "natural_justice_principle": "natural_justice",
    "procedural_fairness": "natural_justice",
    "audi_alteram_partem": "natural_justice",
}


def normalize_argument_scheme(scheme: Any) -> str:
    """Normalize a scheme label to the ArgumentScheme enum values."""
    if scheme is None:
        return scheme
    s = str(scheme).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return _SCHEME_ALIASES.get(s, s)


# =============================================================================
# ACTOR TYPE NORMALIZATION + COERCION
# =============================================================================

# Alias mapping for common LLM outputs that don't match our ActorType enum
_ACTOR_TYPE_ALIASES: Dict[str, str] = {
    # Union of India / Government variants -> respondent
    "union": "respondent",
    "uoi": "respondent",
    "union_of_india": "respondent",
    "government": "respondent",
    "govt": "respondent",
    "state": "respondent",
    "states": "respondent",
    "state_government": "respondent",
    "central_government": "respondent",
    "central_govt": "respondent",
    "authority": "respondent",
    "authorities": "respondent",
    "department": "respondent",
    "ministry": "respondent",
    "corporation": "respondent",
    "public_authority": "respondent",

    # Appellant/Petitioner variants
    "appellant": "appellant",
    "petitioner": "petitioner",
    "applicant": "petitioner",
    "plaintiff": "petitioner",
    "claimant": "petitioner",
    "writ_petitioner": "petitioner",

    # Respondent variants
    "respondent": "respondent",
    "defendant": "respondent",
    "opposite_party": "respondent",
    "opp_party": "respondent",

    # Criminal case actors
    "accused": "accused",
    "convict": "accused",
    "prisoner": "accused",
    "prosecution": "prosecution",
    "public_prosecutor": "prosecution",
    "pp": "prosecution",
    "complainant": "complainant",
    "informant": "complainant",

    # Court variants
    "court": "court",
    "bench": "court",
    "judge": "court",
    "tribunal": "court",
    "lower_court": "lower_court",
    "high_court": "lower_court",
    "trial_court": "lower_court",
    "sessions_court": "lower_court",
    "magistrate": "lower_court",
    "appellate_authority": "lower_court",

    # Third parties
    "amicus": "amicus",
    "amicus_curiae": "amicus",
    "intervenor": "third_party",
    "intervener": "third_party",
    "third_party": "third_party",
    "witness": "third_party",
    "expert": "third_party",
}


def normalize_actor_type(actor: Any) -> Optional[str]:
    """Normalize an actor string to a canonical form.

    Returns None if actor is None/empty.
    """
    if actor is None:
        return None
    a = str(actor).strip().lower()
    if not a:
        return None
    a = a.replace("-", "_").replace(" ", "_")
    a = re.sub(r"_+", "_", a)
    return _ACTOR_TYPE_ALIASES.get(a, a)


def coerce_actor_type(actor: Any, default: Optional[str] = None) -> Optional[ActorType]:
    """Coerce an arbitrary actor-like string into a valid ActorType enum value.

    This is a safety net for noisy IL-TUR judgments and occasional LLM drift.
    Instead of raising ValueError and dropping nodes, we map unknown actors
    to a valid enum value (typically 'third_party' or 'respondent').

    Args:
        actor: The raw actor value from LLM extraction
        default: Default ActorType value if actor is None/empty (as string)

    Returns:
        ActorType enum value, or None if actor is None and no default provided
    """
    if actor is None:
        if default is not None:
            try:
                return ActorType(default)
            except ValueError:
                return None
        return None

    # First normalize
    normalized = normalize_actor_type(actor)
    if normalized is None:
        if default is not None:
            try:
                return ActorType(default)
            except ValueError:
                return None
        return None

    # Check if it's already a valid enum value
    valid_values = {at.value for at in ActorType}
    if normalized in valid_values:
        return ActorType(normalized)

    # Heuristic fallbacks for unmapped values
    a = normalized.lower()

    # Government/authority patterns -> respondent
    if any(pat in a for pat in ["gov", "union", "state", "ministry", "department",
                                "authority", "board", "commission", "corporation",
                                "municipal", "council", "committee"]):
        logger.debug(f"Coerced actor '{actor}' -> respondent (government pattern)")
        return ActorType.RESPONDENT

    # Petitioner patterns
    if any(pat in a for pat in ["petition", "applic", "plaintiff", "claim", "writ"]):
        logger.debug(f"Coerced actor '{actor}' -> petitioner (petitioner pattern)")
        return ActorType.PETITIONER

    # Appellant patterns
    if "appell" in a:
        logger.debug(f"Coerced actor '{actor}' -> appellant (appellant pattern)")
        return ActorType.APPELLANT

    # Respondent patterns
    if any(pat in a for pat in ["respond", "defend", "opposite"]):
        logger.debug(f"Coerced actor '{actor}' -> respondent (respondent pattern)")
        return ActorType.RESPONDENT

    # Criminal patterns
    if any(pat in a for pat in ["accuse", "convict", "prisoner"]):
        logger.debug(f"Coerced actor '{actor}' -> accused (accused pattern)")
        return ActorType.ACCUSED
    if any(pat in a for pat in ["prosecu", "public_prosecutor"]):
        logger.debug(f"Coerced actor '{actor}' -> prosecution (prosecution pattern)")
        return ActorType.PROSECUTION
    if any(pat in a for pat in ["complain", "inform"]):
        logger.debug(f"Coerced actor '{actor}' -> complainant (complainant pattern)")
        return ActorType.COMPLAINANT

    # Court patterns
    if any(pat in a for pat in ["court", "bench", "judge", "tribunal"]):
        if any(pat in a for pat in ["lower", "trial", "session", "magistrat", "high_court"]):
            logger.debug(f"Coerced actor '{actor}' -> lower_court (lower court pattern)")
            return ActorType.LOWER_COURT
        logger.debug(f"Coerced actor '{actor}' -> court (court pattern)")
        return ActorType.COURT

    # Amicus patterns
    if "amicus" in a or "friend_of_court" in a:
        logger.debug(f"Coerced actor '{actor}' -> amicus (amicus pattern)")
        return ActorType.AMICUS

    # Default fallback: third_party (safest generic option)
    logger.debug(f"Coerced actor '{actor}' -> third_party (fallback)")
    return ActorType.THIRD_PARTY


# Maps (source_type, target_type) -> allowed relations
# NOTE: We keep this relatively permissive to avoid dropping edges on IL-TUR.
VALID_EDGE_RELATIONS: Dict[Tuple[str, str], Set[str]] = {
    # Fact -> X
    ("fact", "concept"): {"triggers", "negates", "partially_satisfies", "satisfies", "claims_satisfies"},
    ("fact", "argument"): {"supports", "grounds", "rebuts", "undercuts"},
    ("fact", "holding"): {"supports", "grounds"},
    ("fact", "issue"): {"triggers", "supports", "addresses"},

    # Concept -> X
    ("concept", "concept"): {"requires", "excludes", "specializes", "conflicts_with"},
    ("concept", "argument"): {"supports", "grounds", "rebuts", "undercuts"},
    ("concept", "holding"): {"grounds", "constrains", "supports", "enables"},
    ("concept", "issue"): {"requires", "addresses"},

    # Argument -> X
    ("argument", "issue"): {"addresses", "concedes"},
    ("argument", "argument"): {"attacks", "supports_arg", "responds_to"},
    ("argument", "holding"): {"supports", "grounds", "rebuts", "undercuts"},
    ("argument", "concept"): {"supports", "grounds", "rebuts", "undercuts", "claims_satisfies"},

    # Holding -> X
    ("holding", "issue"): {"resolves", "partially_resolves", "addresses"},
    ("holding", "outcome"): {"determines", "contributes_to"},
    ("holding", "precedent"): {"follows", "applies", "distinguishes", "overrules", "doubts", "explains"},
    ("holding", "concept"): {"supports", "grounds", "constrains", "undercuts", "negates"},
    ("holding", "holding"): {"supports", "conflicts_with", "specializes", "constrains", "undercuts"},

    # Precedent -> X
    ("precedent", "concept"): {"supports", "grounds", "establishes"},
    ("precedent", "holding"): {"supports"},
    ("precedent", "argument"): {"supports"},
    ("precedent", "issue"): {"addresses", "supports"},

    # Issue -> X
    ("issue", "concept"): {"requires", "addresses"},
    ("issue", "holding"): {"addresses", "requires"},
    ("issue", "argument"): {"addresses", "requires"},
    ("issue", "precedent"): {"addresses"},
    ("issue", "issue"): {"specializes", "conflicts_with", "requires"},
}


def get_node_type_from_id(node_id: str) -> str:
    """Infer node type from ID prefix."""
    if node_id == "outcome":
        return "outcome"
    prefix_map = {
        "f": "fact",
        "c": "concept",
        "i": "issue",
        "a": "argument",
        "h": "holding",
        "p": "precedent",
        "js": "justification_set"
    }
    for prefix, ntype in prefix_map.items():
        if node_id.startswith(prefix) and (len(node_id) == len(prefix) + 1 or node_id[len(prefix):].isdigit()):
            return ntype
    return "unknown"


def validate_edge_relation(source_id: str, target_id: str, relation: str) -> Tuple[bool, str]:
    """Validate that an edge relation is valid for the source/target types."""
    source_type = get_node_type_from_id(source_id)
    target_type = get_node_type_from_id(target_id)

    rel_norm = normalize_edge_relation(relation)

    key = (source_type, target_type)
    if key not in VALID_EDGE_RELATIONS:
        return False, f"No valid relations defined for {source_type} -> {target_type}"

    if rel_norm not in VALID_EDGE_RELATIONS[key]:
        valid = ", ".join(sorted(VALID_EDGE_RELATIONS[key]))
        if rel_norm != relation:
            return False, f"'{relation}' (→ '{rel_norm}') not valid for {source_type} -> {target_type}. Valid: {valid}"
        return False, f"'{relation}' not valid for {source_type} -> {target_type}. Valid: {valid}"

    return True, ""


def repair_edge_relation(source_id: str, target_id: str, relation: str) -> Tuple[
    Optional[str], Optional[str], Optional[str], str]:
    """Attempt to repair an edge relation (and sometimes direction) to satisfy VALID_EDGE_RELATIONS.

    Returns (new_source, new_target, new_relation, note). If no repair is possible,
    returns (None, None, None, note).

    Design goals:
    - Prefer keeping direction when possible.
    - If a source→target type-pair is unsupported but the reverse is supported, flip the edge.
    - Map common drifted relations (e.g., explains → addresses) to the closest permitted relation.
    """
    src_type = get_node_type_from_id(source_id)
    tgt_type = get_node_type_from_id(target_id)
    rel = normalize_edge_relation(relation)

    # If pair isn't supported but reverse is, flip.
    key = (src_type, tgt_type)
    flipped = False
    if key not in VALID_EDGE_RELATIONS:
        rev = (tgt_type, src_type)
        if rev in VALID_EDGE_RELATIONS:
            source_id, target_id = target_id, source_id
            src_type, tgt_type = tgt_type, src_type
            key = rev
            flipped = True
        else:
            return None, None, None, f"no relation matrix for {src_type}->{tgt_type}"

    allowed = VALID_EDGE_RELATIONS[key]
    if rel in allowed:
        return source_id, target_id, rel, ("flipped direction" if flipped else "ok")

    # Common relation drift maps (pair-aware when it matters)
    # General mappings (only apply if the mapped value is allowed)
    general_map = [
        ("explains", "addresses"),
        ("supports_arg", "addresses"),
        ("contributes_to", "addresses"),
        ("partially_resolves", "addresses"),
        ("resolves", "addresses"),
        ("enables", "requires"),
        ("distinguishes", "specializes"),
        ("partially_satisfies", "grounds"),
        ("satisfies", "supports"),
        ("establishes", "supports"),
    ]
    for bad, good in general_map:
        if rel == bad and good in allowed:
            note = f"{'flipped; ' if flipped else ''}{bad}->{good}"
            return source_id, target_id, good, note

    # Pair-specific fallbacks
    if key == ("fact", "concept"):
        # Facts typically *trigger* or *satisfy* concepts
        if "triggers" in allowed:
            return source_id, target_id, "triggers", f"{'flipped; ' if flipped else ''}{rel}->triggers"
        if "satisfies" in allowed:
            return source_id, target_id, "satisfies", f"{'flipped; ' if flipped else ''}{rel}->satisfies"

    if key == ("holding", "issue") and "addresses" in allowed:
        return source_id, target_id, "addresses", f"{'flipped; ' if flipped else ''}{rel}->addresses"

    if key == ("concept", "issue") and "addresses" in allowed:
        return source_id, target_id, "addresses", f"{'flipped; ' if flipped else ''}{rel}->addresses"

    if key == ("issue", "issue") and "specializes" in allowed:
        return source_id, target_id, "specializes", f"{'flipped; ' if flipped else ''}{rel}->specializes"

    if key == ("precedent", "concept") and "grounds" in allowed:
        return source_id, target_id, "grounds", f"{'flipped; ' if flipped else ''}{rel}->grounds"

    # Generic fallback: prefer grounds/supports/addresses when available, else first allowed.
    for pref in ["grounds", "supports", "addresses", "requires", "specializes", "triggers"]:
        if pref in allowed:
            return source_id, target_id, pref, f"{'flipped; ' if flipped else ''}{rel}->{pref}"

    # As a last resort, pick an arbitrary allowed relation deterministically.
    try:
        chosen = sorted(list(allowed))[0]
        return source_id, target_id, chosen, f"{'flipped; ' if flipped else ''}{rel}->{chosen}"
    except Exception:
        return None, None, None, "repair failed"


# =============================================================================
# CONFIGURATION

# =============================================================================
# QUOTE → OFFSET ALIGNMENT (for evidence anchoring)
# =============================================================================


def _normalize_with_mapping(raw: str) -> Tuple[str, List[int]]:
    """Normalize text by collapsing whitespace and return a mapping to original indices.

    Returns:
      normalized_text, index_map

    where index_map[i] gives the original character index for normalized_text[i].
    """
    norm_chars: List[str] = []
    idx_map: List[int] = []
    in_ws = False
    for i, ch in enumerate(raw):
        if ch.isspace():
            if not in_ws:
                norm_chars.append(' ')
                idx_map.append(i)
                in_ws = True
            continue
        norm_chars.append(ch)
        idx_map.append(i)
        in_ws = False
    return ''.join(norm_chars), idx_map


def align_quote_to_span(doc_text: str, quote: str) -> Optional[Tuple[int, int]]:
    """Find (start,end) offsets in doc_text for a quoted snippet.

    Tries exact match after whitespace normalization, then case-insensitive match.
    Returns None if not found.
    """
    if not quote:
        return None
    q = quote.strip()
    if not q:
        return None

    norm_doc, doc_map = _normalize_with_mapping(doc_text)
    norm_q, _ = _normalize_with_mapping(q)

    pos = norm_doc.find(norm_q)
    if pos == -1:
        pos = norm_doc.lower().find(norm_q.lower())
    if pos == -1:
        return None

    start = doc_map[pos]
    end = doc_map[pos + len(norm_q) - 1] + 1
    if start < 0 or end <= start or end > len(doc_text):
        return None
    return start, end


# =============================================================================

def build_cluster_edge_whitelist(node_types_present: Set[str], exclude_structural: bool = True) -> str:
    """Build a per-type whitelist of allowed edge relations for a cluster.

    This creates a prompt-friendly table of which relations are valid for each
    source->target type pair present in the cluster.

    Args:
        node_types_present: Set of node types present in the cluster (e.g., {"fact", "concept", "holding"})
        exclude_structural: If True, exclude resolves/determines (added deterministically later)

    Returns:
        A formatted string for inclusion in prompts
    """
    # Relations that are added deterministically, not by LLM
    structural_relations = {"resolves", "partially_resolves", "determines", "contributes_to"}

    lines = ["ALLOWED RELATIONS (use ONLY these for the specified type pairs):"]
    lines.append("")

    for (src_type, tgt_type), relations in sorted(VALID_EDGE_RELATIONS.items()):
        # Only include if both types are present in the cluster
        if src_type not in node_types_present or tgt_type not in node_types_present:
            continue

        # Filter out structural relations if requested
        if exclude_structural:
            relations = relations - structural_relations

        if not relations:
            continue

        rel_str = " | ".join(sorted(relations))
        lines.append(f"  {src_type} → {tgt_type}: {rel_str}")

    lines.append("")
    lines.append("If a type pair is NOT listed above, do NOT create an edge for that pair.")

    if exclude_structural:
        lines.append(
            "DO NOT emit 'resolves', 'partially_resolves', 'determines', or 'contributes_to' edges - these are added automatically.")

    return "\n".join(lines)


def get_node_types_in_cluster(cluster: 'ConceptCluster') -> Set[str]:
    """Get the set of node types present in a cluster."""
    types = set()
    if cluster.facts:
        types.add("fact")
    if cluster.concepts:
        types.add("concept")
    if cluster.issues:
        types.add("issue")
    if cluster.arguments:
        types.add("argument")
    if cluster.holdings:
        types.add("holding")
    if cluster.precedents:
        types.add("precedent")
    return types


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""

    # Model settings
    model_id: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1
    max_retries: int = 3

    # Extraction settings
    min_facts: int = 3
    max_facts: int = 50
    min_concepts: int = 1
    max_concepts: int = 30
    min_holdings: int = 1

    # Document handling
    max_doc_chars: int = 80000  # Max chars to process
    chunk_overlap: int = 500  # Overlap between chunks

    # Quality thresholds
    gold_threshold: float = 0.9
    silver_threshold: float = 0.7
    bronze_threshold: float = 0.5

    # Ontology
    ontology_path: Optional[str] = None
    ontology_data: Optional[Dict] = None

    # Pipeline
    pipeline_version: Literal["v3", "v4"] = "v4"
    enable_link_discovery: bool = False  # v3 pass 9
    enable_llm_justification_sets: bool = False  # v3 pass 10

    # Run identification
    run_id: Optional[str] = None

    def load_ontology(self) -> Dict:
        """Load ontology from file or return cached."""
        if self.ontology_data:
            return self.ontology_data

        if self.ontology_path and Path(self.ontology_path).exists():
            with open(self.ontology_path, 'r') as f:
                self.ontology_data = json.load(f)
                logger.info(f"Loaded ontology with {len(self.ontology_data.get('concepts', {}))} concepts")
                return self.ontology_data

        return {}


# =============================================================================
# v4: ONTOLOGY-DRIVEN CONCEPT CLUSTERING
# =============================================================================


@dataclass
class ConceptCluster:
    """A concept-centric grouping of nodes.

    In v4, clusters are the *primary index* for edge extraction.
    """

    concept_id: str
    concept_label: str

    # Ontology logic
    logic: str = "and"  # "and" | "or"
    requires: List[str] = field(default_factory=list)
    defeaters: List[str] = field(default_factory=list)

    # Grouped node IDs
    facts: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    precedents: List[str] = field(default_factory=list)
    arguments: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    holdings: List[str] = field(default_factory=list)

    # requirement text -> fact_id that appears to satisfy it (best-effort)
    satisfied_requirements: Dict[str, Optional[str]] = field(default_factory=dict)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "by", "with", "without",
    "is", "was", "were", "are", "be", "been", "being", "as", "at", "from", "that", "this",
    "it", "its", "their", "his", "her", "they", "them", "he", "she", "we", "our", "you",
    "not", "no", "yes", "shall", "may", "must", "can", "could", "would", "should",
}


def parse_key_phrases(raw: str) -> List[str]:
    """Parse a key_phrases string from the compiled ontology into a list.

    The compiled file often stores phrases like:
        '"foo", "bar", "baz"'
    """
    if not raw:
        return []

    # Prefer quoted phrases when present.
    quoted = re.findall(r'"(.*?)"', raw)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]

    # Fallback: split on commas
    parts = [p.strip().strip('"\'') for p in raw.split(',')]
    return [p for p in parts if p]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if t]


def _keyword_set(text: str) -> Set[str]:
    toks = _tokenize(text)
    return {t for t in toks if len(t) >= 4 and t not in _STOPWORDS}


def _contains_phrase(haystack: str, phrase: str) -> bool:
    if not haystack or not phrase:
        return False
    return phrase.lower() in haystack.lower()


def _best_ontology_match_for_concept_node(
        concept_node: ConceptNode,
        ontology_concepts: Dict[str, Dict]
) -> Optional[str]:
    """Map an extracted ConceptNode to the most plausible ontology concept_id.

    Matching strategy (best-effort):
      1) exact concept_id
      2) prefix/substring match on ontology IDs
      3) keyword overlap between node surface_text / interpretation and ontology fields
    """

    if not concept_node or not concept_node.concept_id:
        return None

    cid = concept_node.concept_id.strip()
    if cid in ontology_concepts:
        return cid

    # Prefix/substring heuristic (helps mapping DOCTRINE_NATURAL_JUSTICE -> ..._AUDIO...)
    candidates = [k for k in ontology_concepts.keys() if cid in k]
    if candidates:
        # Choose the most specific (longest) match.
        return max(candidates, key=len)

    # Avoid over-mapping highly structured IDs (e.g., CONST_ART21, STATUTE_*) to
    # unrelated ontology doctrines via generic keyword overlap.
    raw_id = cid.upper()
    if raw_id.startswith(("CONST_", "STATUTE_", "PROCEDURE_", "REMEDY_", "RIGHT_")):
        return None

    # Keyword overlap
    node_text = " ".join(filter(None, [
        concept_node.concept_id,
        concept_node.unlisted_label,
        concept_node.interpretation,
        concept_node.anchor.surface_text if concept_node.anchor else None
    ]))
    node_kw = _keyword_set(node_text)
    if not node_kw:
        return None

    best = None
    best_score = 0
    for ont_id, ont_def in ontology_concepts.items():
        ont_text = " ".join(filter(None, [
            ont_id,
            str(ont_def.get("label", "")),
            " ".join(ont_def.get("requires") or []),
            " ".join(parse_key_phrases(ont_def.get("key_phrases", ""))),
        ]))
        ont_kw = _keyword_set(ont_text)
        if not ont_kw:
            continue
        score = len(node_kw.intersection(ont_kw))
        if score > best_score:
            best_score = score
            best = ont_id

    # Require at least a small amount of overlap to avoid random clustering
    if best and best_score >= 2:
        return best
    return None


def _node_text_for_matching(node: Any) -> str:
    """Return a best-effort text representation for matching."""
    if node is None:
        return ""
    if isinstance(node, FactNode):
        return node.text or ""
    if isinstance(node, IssueNode):
        return node.text or ""
    if isinstance(node, HoldingNode):
        return node.text or ""
    if isinstance(node, ArgumentNode):
        return " ".join(filter(None, [node.claim, node.court_reasoning]))
    if isinstance(node, PrecedentNode):
        return " ".join(filter(None, [node.case_name, node.citation, node.cited_proposition]))
    if isinstance(node, ConceptNode):
        return " ".join(filter(None, [node.concept_id, node.unlisted_label, node.interpretation]))
    return str(node)


def _concept_match_score(node_text: str, concept_def: Dict) -> int:
    """Crude relevance score between a node text and an ontology concept.

    This is intentionally lightweight (no embeddings) but tries to capture the
    high-signal hooks in the compiled ontology:
      - key_phrases (strong)
      - establishing_cases (strong)
      - requirement keywords (medium)
      - defeater keywords (weak)
    """

    if not node_text or not concept_def:
        return 0

    # NOTE: _normalize_with_mapping returns (normalized_text, index_map).
    # We only need the normalized text here.
    txt_norm, _ = _normalize_with_mapping(node_text)
    txt_norm_l = txt_norm.lower()

    phrases = parse_key_phrases(concept_def.get("key_phrases", ""))

    raw_requires = concept_def.get("requires") or []
    requires = list(raw_requires)
    if requires and isinstance(requires[0], str) and requires[0].startswith("["):
        requires = requires[1:]

    defeaters = list(concept_def.get("defeaters") or [])
    label = str(concept_def.get("label", ""))

    score = 0

    # 1) Phrase hits are high signal
    for ph in phrases:
        if ph and _contains_phrase(node_text, ph):
            score += 5

    # 2) Establishing cases are high signal, but in this ontology they are often
    #    stored as a comma-separated string.
    establishing_cases = concept_def.get("establishing_cases")
    if isinstance(establishing_cases, str):
        case_names = [c.strip() for c in re.split(r"[\n;]+|,", establishing_cases) if c.strip()]
    elif isinstance(establishing_cases, list):
        case_names = [str(c).strip() for c in establishing_cases if str(c).strip()]
    else:
        case_names = []

    for case_name in case_names:
        cn, _ = _normalize_with_mapping(case_name)
        cn = cn.strip()
        if len(cn) >= 8 and cn.lower() in txt_norm_l:
            score += 8
            break

    # 3) Typical fact patterns (if present) can be a moderate signal
    typical_fact_patterns = concept_def.get("typical_fact_patterns")
    if isinstance(typical_fact_patterns, str):
        patterns = [p.strip() for p in re.split(r"[\n;]+|,", typical_fact_patterns) if p.strip()]
    elif isinstance(typical_fact_patterns, list):
        patterns = [str(p).strip() for p in typical_fact_patterns if str(p).strip()]
    else:
        patterns = []

    for pat in patterns:
        if pat and _contains_phrase(node_text, pat):
            score += 3
            break

    # 4) Keyword overlap (capped)
    node_kw = _keyword_set(node_text)
    concept_kw = _keyword_set(" ".join([label] + requires + defeaters + phrases))
    score += min(6, len(node_kw.intersection(concept_kw)))

    # 5) Extra weight for requirement keyword overlap (capped)
    req_kw = _keyword_set(" ".join(requires))
    score += min(6, len(node_kw.intersection(req_kw)))

    return score


def _looks_negative(text: str) -> bool:
    """Heuristic: detect negation patterns for requirement satisfaction."""
    t = (text or "").lower()
    return any(pat in t for pat in ["without ", "no ", "not ", "denied", "refused", "failed to"])


def normalize_ontology_requires(requires_raw: Any) -> Tuple[str, List[str]]:
    """Normalize the 'requires' field from ontology into (logic, requirements_list).

    The ontology can have 'requires' as:
    - A list like ["[AND]", "req1", "req2"] or ["[OR]", "req1", "req2"]
    - A plain list like ["req1", "req2"] (defaults to AND)
    - A string like "1. req1\n2. req2" or "req1; req2"
    - None or empty

    Returns:
        (logic, requirements_list) where logic is "and" or "or"
    """
    if requires_raw is None:
        return "and", []

    # Case 1: Already a list
    if isinstance(requires_raw, list):
        if not requires_raw:
            return "and", []

        # Check for logic marker in first element
        logic = "and"
        start_idx = 0
        first = str(requires_raw[0]).strip().upper()
        if first.startswith("["):
            if "OR" in first:
                logic = "or"
            elif "AND" in first:
                logic = "and"
            start_idx = 1

        # Extract requirements, filtering out empty strings
        requirements = [
            str(r).strip() for r in requires_raw[start_idx:]
            if str(r).strip() and not str(r).strip().upper().startswith("[")
        ]
        return logic, requirements

    # Case 2: String - need to parse
    if isinstance(requires_raw, str):
        text = requires_raw.strip()
        if not text:
            return "and", []

        # Check for logic marker at start
        logic = "and"
        if text.upper().startswith("[OR"):
            logic = "or"
            # Remove the marker
            text = re.sub(r'^\s*\[OR\]?\s*', '', text, flags=re.IGNORECASE)
        elif text.upper().startswith("[AND"):
            logic = "and"
            text = re.sub(r'^\s*\[AND\]?\s*', '', text, flags=re.IGNORECASE)

        # Split on common delimiters: newlines, semicolons, or numbered bullets
        # First try numbered bullets (1. 2. etc)
        if re.search(r'^\s*\d+[\.\)]\s*', text, re.MULTILINE):
            parts = re.split(r'\d+[\.\)]\s*', text)
        else:
            # Split on newlines or semicolons
            parts = re.split(r'[\n;]+', text)

        # Clean up and filter
        requirements = [p.strip() for p in parts if p.strip()]
        return logic, requirements

    # Fallback
    return "and", []


def normalize_ontology_defeaters(defeaters_raw: Any) -> List[str]:
    """Normalize the 'defeaters' field from ontology into a list.

    Similar to requires, but without logic markers.
    """
    if defeaters_raw is None:
        return []

    if isinstance(defeaters_raw, list):
        return [str(d).strip() for d in defeaters_raw if str(d).strip()]

    if isinstance(defeaters_raw, str):
        text = defeaters_raw.strip()
        if not text:
            return []

        # Split on common delimiters
        if re.search(r'^\s*\d+[\.\)]\s*', text, re.MULTILINE):
            parts = re.split(r'\d+[\.\)]\s*', text)
        else:
            parts = re.split(r'[\n;]+', text)

        return [p.strip() for p in parts if p.strip()]

    return []


def cluster_nodes(
        graph: LegalReasoningGraph,
        ontology: Dict
) -> Tuple[Dict[str, ConceptCluster], Dict[str, List[str]]]:
    """Group extracted nodes into concept-centric clusters.

    Returns:
      clusters: concept_id -> ConceptCluster
      node_cluster_membership: node_id -> [concept_id, ...]

    Notes:
      - Uses the compiled ontology as the primary clustering index when possible.
      - Falls back to creating pseudo-clusters for concept_ids not present in the ontology.
    """

    ontology_concepts = (ontology or {}).get("concepts", {}) or {}

    clusters: Dict[str, ConceptCluster] = {}
    node_membership: Dict[str, List[str]] = {}

    # 1) Seed clusters from ontology
    for concept_id, concept_def in ontology_concepts.items():
        # Use normalized parsing to handle string vs list requires
        logic, requires = normalize_ontology_requires(concept_def.get("requires"))
        defeaters = normalize_ontology_defeaters(concept_def.get("defeaters"))

        clusters[concept_id] = ConceptCluster(
            concept_id=concept_id,
            concept_label=str(concept_def.get("label") or concept_id),
            logic=logic,
            requires=requires,
            defeaters=defeaters,
            satisfied_requirements={req: None for req in requires}
        )

    # Helper to create a pseudo cluster when we don't have ontology support
    def _ensure_pseudo_cluster(pseudo_id: str, label: Optional[str] = None) -> ConceptCluster:
        if pseudo_id not in clusters:
            clusters[pseudo_id] = ConceptCluster(
                concept_id=pseudo_id,
                concept_label=label or pseudo_id,
                logic="and",
                requires=[],
                defeaters=[],
                satisfied_requirements={}
            )
        return clusters[pseudo_id]

    # 2) Map ConceptNodes to ontology clusters (or pseudo clusters)
    concept_node_to_cluster_ids: Dict[str, List[str]] = {}
    for c in graph.concepts:
        best = _best_ontology_match_for_concept_node(c, ontology_concepts)
        if best:
            concept_node_to_cluster_ids[c.id] = [best]
        else:
            # Fallback pseudo cluster keyed by concept_id (stable across runs)
            pseudo = c.concept_id or f"UNLISTED_{c.id}"
            _ensure_pseudo_cluster(pseudo, label=c.unlisted_label or pseudo)
            concept_node_to_cluster_ids[c.id] = [pseudo]

        # Add the concept node itself to each cluster
        for cluster_id in concept_node_to_cluster_ids[c.id]:
            clusters[cluster_id].concepts.append(c.id)
            node_membership.setdefault(c.id, []).append(cluster_id)

    # 3) Assign Issues based on primary_concepts (concept node IDs)
    for issue in graph.issues:
        assigned = False
        for concept_node_id in (issue.primary_concepts or []):
            for cluster_id in concept_node_to_cluster_ids.get(concept_node_id, []):
                clusters[cluster_id].issues.append(issue.id)
                node_membership.setdefault(issue.id, []).append(cluster_id)
                assigned = True

        if not assigned:
            # Fallback: keyword match issue text against ontology
            best_cluster = None
            best_score = 0
            for ont_id, ont_def in ontology_concepts.items():
                score = _concept_match_score(issue.text or "", ont_def)
                if score > best_score:
                    best_score = score
                    best_cluster = ont_id
            if best_cluster and best_score >= 3:
                clusters[best_cluster].issues.append(issue.id)
                node_membership.setdefault(issue.id, []).append(best_cluster)

    # 4) Assign Holdings based on resolves_issue → issue clusters
    issue_to_clusters = {i.id: node_membership.get(i.id, []) for i in graph.issues}
    for holding in graph.holdings:
        assigned = False
        if holding.resolves_issue and holding.resolves_issue in issue_to_clusters:
            for cluster_id in issue_to_clusters[holding.resolves_issue]:
                clusters[cluster_id].holdings.append(holding.id)
                node_membership.setdefault(holding.id, []).append(cluster_id)
                assigned = True

        if not assigned:
            # Fallback match on holding text
            best_cluster = None
            best_score = 0
            for ont_id, ont_def in ontology_concepts.items():
                score = _concept_match_score(holding.text or "", ont_def)
                if score > best_score:
                    best_score = score
                    best_cluster = ont_id
            if best_cluster and best_score >= 3:
                clusters[best_cluster].holdings.append(holding.id)
                node_membership.setdefault(holding.id, []).append(best_cluster)

    # 5) Assign Facts / Arguments / Precedents via scoring against ontology + already-seeded clusters
    def _assign_by_score(node_obj: Any, min_score: int = 3):
        txt = _node_text_for_matching(node_obj)
        # Prefer clusters that already have issues/holdings to reduce noise
        candidate_cluster_ids = [
            cid for cid, cl in clusters.items()
            if (cl.issues or cl.holdings or cl.concepts)
        ]
        best_cluster = None
        best_score = 0
        for cid in candidate_cluster_ids:
            concept_def = ontology_concepts.get(cid)
            if concept_def:
                score = _concept_match_score(txt, concept_def)
            else:
                # Pseudo cluster: approximate "concept definition" from the cluster's
                # own concept nodes (anchor snippets) + label/id.
                pseudo_kw: Set[str] = set()
                pseudo_kw |= _keyword_set(cid)
                pseudo_kw |= _keyword_set(clusters[cid].concept_label)
                for concept_node_id in clusters[cid].concepts:
                    cn = graph.get_node(concept_node_id)
                    if cn:
                        pseudo_kw |= _keyword_set(_node_text_for_matching(cn))
                score = len(_keyword_set(txt).intersection(pseudo_kw))
            if score > best_score:
                best_score = score
                best_cluster = cid
        if best_cluster and best_score >= min_score:
            node_membership.setdefault(node_obj.id, []).append(best_cluster)
            return best_cluster
        return None

    # Facts
    for f in graph.facts:
        cluster_id = _assign_by_score(f, min_score=2)
        if cluster_id:
            clusters[cluster_id].facts.append(f.id)

    # Arguments
    for a in graph.arguments:
        cluster_id = _assign_by_score(a, min_score=2)
        if cluster_id:
            clusters[cluster_id].arguments.append(a.id)

    # Precedents: match on establishing_cases first
    for p in graph.precedents:
        assigned = False
        for cid, cdef in ontology_concepts.items():
            establishing = str(cdef.get("establishing_cases") or "").lower()
            if p.case_name and p.case_name.lower() in establishing:
                clusters[cid].precedents.append(p.id)
                node_membership.setdefault(p.id, []).append(cid)
                assigned = True
        if not assigned:
            cluster_id = _assign_by_score(p, min_score=2)
            if cluster_id:
                clusters[cluster_id].precedents.append(p.id)

    # 6) Best-effort requirement satisfaction (only within ontology-backed clusters)
    for cid, cl in clusters.items():
        if not cl.requires or not cl.facts:
            continue

        fact_map = {f.id: f for f in graph.facts}
        for req in cl.requires:
            best_fact = None
            best_score = 0
            for fid in cl.facts:
                fnode = fact_map.get(fid)
                if not fnode:
                    continue
                # Keyword overlap between requirement and fact text
                req_kw = _keyword_set(req)
                f_kw = _keyword_set(fnode.text)
                score = len(req_kw.intersection(f_kw))

                # If the fact looks explicitly negative ("without hearing"),
                # treat it as *not* satisfying the requirement.
                if _looks_negative(fnode.text):
                    score = max(0, score - 2)

                if score > best_score:
                    best_score = score
                    best_fact = fid

            # Require at least one non-trivial hit
            if best_fact and best_score >= 1:
                cl.satisfied_requirements[req] = best_fact

    # 7) Prune truly empty clusters
    clusters = {
        cid: cl for cid, cl in clusters.items()
        if (cl.facts or cl.concepts or cl.issues or cl.arguments or cl.holdings or cl.precedents)
    }

    return clusters, node_membership


# =============================================================================
# TEXT SEGMENTATION
# =============================================================================

@dataclass
class TextSegment:
    """A segment of text with stable offsets."""
    text: str
    start_char: int
    end_char: int
    para_index: int
    sent_index: Optional[int] = None

    @property
    def display_location(self) -> str:
        if self.sent_index is not None:
            return f"{self.para_index}:{self.sent_index}"
        return str(self.para_index)


@dataclass
class SegmentedDocument:
    """A document split into paragraphs and sentences with offsets."""
    doc_id: str
    full_text: str
    paragraphs: List[TextSegment]
    sentences: List[TextSegment]

    char_count: int = 0
    para_count: int = 0
    sent_count: int = 0

    def get_segment_at(self, start: int, end: int) -> Optional[TextSegment]:
        """Find the segment containing a char range."""
        for seg in self.sentences:
            if seg.start_char <= start and seg.end_char >= end:
                return seg
        for seg in self.paragraphs:
            if seg.start_char <= start and seg.end_char >= end:
                return seg
        return None

    def text_at(self, start: int, end: int) -> str:
        """Get text at char offsets."""
        return self.full_text[start:end]

    def compute_text_hash(self, start: int, end: int) -> str:
        """Compute hash for text span."""
        text = self.full_text[start:end]
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def get_context_window(self, center_char: int, window_size: int = 2000) -> Tuple[int, int, str]:
        """Get a context window around a character position."""
        start = max(0, center_char - window_size // 2)
        end = min(len(self.full_text), center_char + window_size // 2)
        return start, end, self.full_text[start:end]


def segment_document(text: str, doc_id: str) -> SegmentedDocument:
    """Segment a document into paragraphs and sentences with stable char offsets."""
    paragraphs = []
    sentences = []

    # Split into paragraphs
    para_pattern = re.compile(r'\n\s*\n|\n(?=\d+\.?\s)|(?<=\.)\s*\n')

    para_starts = [0]
    for match in para_pattern.finditer(text):
        para_starts.append(match.end())
    para_starts.append(len(text))

    sent_global_idx = 0

    for i, (start, end) in enumerate(zip(para_starts[:-1], para_starts[1:])):
        raw_para_text = text[start:end]
        para_text = raw_para_text.strip()

        if para_text:
            leading_ws = len(raw_para_text) - len(raw_para_text.lstrip())
            adjusted_start = start + leading_ws
            adjusted_end = adjusted_start + len(para_text)

            para_seg = TextSegment(
                text=para_text,
                start_char=adjusted_start,
                end_char=adjusted_end,
                para_index=len(paragraphs)
            )
            paragraphs.append(para_seg)

            # Split paragraph into sentences
            sent_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
            sent_starts = [0]
            for match in sent_pattern.finditer(para_text):
                sent_starts.append(match.end())
            sent_starts.append(len(para_text))

            for j, (s_start, s_end) in enumerate(zip(sent_starts[:-1], sent_starts[1:])):
                sent_text = para_text[s_start:s_end].strip()
                if sent_text:
                    sent_seg = TextSegment(
                        text=sent_text,
                        start_char=adjusted_start + s_start,
                        end_char=adjusted_start + s_end,
                        para_index=para_seg.para_index,
                        sent_index=j
                    )
                    sentences.append(sent_seg)
                    sent_global_idx += 1

    doc = SegmentedDocument(
        doc_id=doc_id,
        full_text=text,
        paragraphs=paragraphs,
        sentences=sentences,
        char_count=len(text),
        para_count=len(paragraphs),
        sent_count=len(sentences)
    )

    return doc


# =============================================================================
# LLM CLIENT INTERFACE
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
            self,
            prompt: str,
            system: str,
            temperature: float = 0.1,
            max_tokens: int = 4096,
            json_mode: bool = True
    ) -> str:
        pass


class MockLLMClient(LLMClient):
    """Mock LLM client for offline testing.

    Records calls in `call_log` and returns an empty JSON object by default.
    Test suites can subclass and override `complete`.
    """

    def __init__(self):
        self.call_log: List[Dict[str, str]] = []

    async def complete(
            self,
            prompt: str,
            system: str,
            temperature: float = 0.1,
            max_tokens: int = 4096,
            json_mode: bool = True
    ) -> str:
        # Store a small snippet for debugging
        self.call_log.append({"prompt": prompt[:500], "system": system[:200]})
        return "{}"


class AnthropicClient(LLMClient):
    """Anthropic Claude client implementation."""

    def __init__(self, api_key: str, model_id: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model_id = model_id
        self._client = None

    async def complete(
            self,
            prompt: str,
            system: str,
            temperature: float = 0.1,
            max_tokens: int = 4096,
            json_mode: bool = True
    ) -> str:
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

        if json_mode:
            system = system + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, no ```json blocks."

        response = await self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


class GrokClient(LLMClient):
    """X.AI Grok client implementation."""

    def __init__(self, api_key: str, model_id: str = "grok-4-1-fast-reasoning"):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://api.x.ai/v1"

    async def complete(
            self,
            prompt: str,
            system: str,
            temperature: float = 0.1,
            max_tokens: int = 4096,
            json_mode: bool = True
    ) -> str:
        import httpx

        if json_mode:
            system = system + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation."

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


# =============================================================================
# EXTRACTION PROMPTS
# =============================================================================

SYSTEM_BASE = """You are a legal reasoning extraction system specialized in Indian court judgments.

CRITICAL RULES:
1. ANCHORS ARE MANDATORY - Every extraction must include exact character offsets (start_char, end_char)
2. USE ONLY WHAT'S IN THE TEXT - Do not infer or hallucinate information not present
3. CONFIDENCE MUST BE HONEST - Mark as "inferred" only if deriving from context
4. ACTORS MATTER - Distinguish between petitioner claims, respondent claims, and court findings
5. CITE SURFACE TEXT - Include first 150 characters of relevant text span
6. BE EXHAUSTIVE - Extract ALL relevant instances, not just obvious ones

OUTPUT: Valid JSON only. No markdown code blocks, no explanations."""

FACTS_PROMPT = """Extract ALL FACTS from this legal judgment.

Facts are factual assertions - what happened, when, to whom. They are the foundation of legal reasoning.

FACT TYPES:
- material: Facts essential to the legal issues (MOST IMPORTANT)
- procedural: Procedural history (filed on, heard by, appeals, etc.)
- background: Context facts, not directly relevant to issues
- disputed: Facts contested by parties
- admitted: Facts agreed by both parties
- judicial_notice: Facts court accepts without proof

EXTRACTION STRATEGY:
1. First scan for material facts that trigger legal issues
2. Then procedural history (dates, filings, lower court decisions)
3. Then background context
4. Note which facts are disputed or admitted

For each fact, provide:
{{
  "id": "f1", "f2", etc.,
  "text": "Clear statement of the fact",
  "start_char": <int>,
  "end_char": <int>,
  "surface_text": "First 150 chars of source text",
  "fact_type": "material" | "procedural" | "background" | "disputed" | "admitted" | "judicial_notice",
  "actor_source": "petitioner" | "respondent" | "court" | "lower_court" | null,
  "date": "YYYY-MM-DD" or null,
  "date_approximate": true/false,
  "disputed_by": "petitioner" | "respondent" | null,
  "court_finding": "accepted" | "rejected" | "not_decided" | null,
  "confidence": "high" | "medium" | "low"
}}

DOCUMENT:
{document}

PARAGRAPH OFFSET REFERENCE:
{offsets}

Extract between {min_facts} and {max_facts} facts. Prioritize MATERIAL facts.

Respond with JSON: {{"facts": [...]}}"""

CONCEPTS_PROMPT = """Extract ALL LEGAL CONCEPTS applied in this judgment.

Concepts are the legal rules, doctrines, statutes, and principles the court discusses or applies.

CONCEPT TYPES (kind):
- statute_article: Constitutional articles (Art. 14, Art. 21, Art. 226)
- statute_section: Statutory sections (Section 482 CrPC, Section 34 IPC)
- order_rule: Procedural rules (Order 41 Rule 27 CPC)
- doctrine: Judge-made principles (natural justice, res judicata, estoppel)
- test: Multi-factor tests (Wednesbury, proportionality, rational nexus)
- standard: Standards of proof/review (beyond reasonable doubt, preponderance)
- right: Fundamental or statutory rights (right to livelihood, right to privacy)
- definition: Statutory definitions

ONTOLOGY IDs - Use these when the concept matches:
{ontology_excerpt}

For UNLISTED concepts, use format: UNLISTED_<short_name>

RELEVANCE levels:
- central: Core to the court's reasoning and holding
- supporting: Used to support main reasoning
- mentioned: Referenced but not applied
- obiter: Said in passing, not binding

For each concept, provide:
{{
  "id": "c1", "c2", etc.,
  "concept_id": "CONST_ART21" or "UNLISTED_xxx",
  "start_char": <int>,
  "end_char": <int>,
  "surface_text": "First 150 chars of source text",
  "relevance": "central" | "supporting" | "mentioned" | "obiter",
  "kind": "statute_article" | "statute_section" | "order_rule" | "doctrine" | "test" | "standard" | "right" | "definition",
  "interpretation": "How the court interprets/applies this concept" or null,
  "interpretation_start_char": <int> or null,
  "interpretation_end_char": <int> or null,
  "unlisted_label": "Name if not in ontology" or null,
  "unlisted_description": "Description if not in ontology" or null,
  "confidence": "high" | "medium" | "low"
}}

EXTRACTED FACTS (for context):
{facts_json}

DOCUMENT:
{document}

Respond with JSON: {{"concepts": [...]}}"""

ISSUES_PROMPT = """Extract ALL LEGAL ISSUES framed in this judgment.

Issues are the legal questions the court must decide. Courts often state these explicitly:
- "The question before us is..."
- "The issue for determination is..."
- "We need to consider whether..."
- "The points for consideration are..."

Also identify IMPLICIT issues from the court's analysis structure.

For each issue, provide:
{{
  "id": "i1", "i2", etc.,
  "text": "The legal question as framed",
  "start_char": <int>,
  "end_char": <int>,
  "surface_text": "First 150 chars of source text",
  "issue_number": <int> or null (if court numbers issues),
  "framed_by": "court" | "petitioner" | "respondent",
  "primary_concepts": ["c1", "c2", ...] (concept node IDs that relate to this issue),
  "answer": "yes" | "no" | "partly" | "not_decided" | null,
  "confidence": "high" | "medium" | "low"
}}

EXTRACTED CONCEPTS (for linking):
{concepts_json}

DOCUMENT:
{document}

Respond with JSON: {{"issues": [...]}}"""

ARGUMENTS_PROMPT = """Extract ALL ARGUMENTS made by parties and the court.

Arguments have:
- A claim (the proposition being argued)
- An actor (who's making it)
- Argument schemes (how it's being argued)
- Court's response (how court treated it)

ARGUMENT SCHEMES:
- rule_application: Applying statute/rule to facts
- rule_exception: Arguing an exception applies
- precedent_following: Following binding precedent
- precedent_analogy: Analogizing to similar case
- precedent_distinction: Distinguishing a precedent
- textual: Plain meaning interpretation
- purposive: Legislative intent argument
- harmonious: Reading provisions together
- proportionality: Proportionality analysis
- balancing: Weighing competing interests
- evidence_sufficiency: Evidence meets required standard
- evidence_credibility: Witness/document reliability
- procedural: Procedure compliance argument
- jurisdiction: Jurisdictional argument
- limitation: Time-bar argument
- policy_consequence: Policy implications
- public_interest: Public interest consideration
- natural_justice: Natural justice / procedural fairness
- other: Fallback when no scheme fits

For each argument, provide:
{{
  "id": "a1", "a2", etc.,
  "claim": "The argument's main claim",
  "start_char": <int>,
  "end_char": <int>,
  "surface_text": "First 150 chars of source text",
  "actor": "petitioner" | "respondent" | "court" | "amicus" | "lower_court",
  "schemes": ["rule_application", ...] (can be multiple),
  "qualifiers": "Hedging language" or null,
  "court_response": "accepted" | "rejected" | "partly_accepted" | "not_addressed" | null,
  "court_response_start_char": <int> or null,
  "court_response_end_char": <int> or null,
  "court_reasoning": "Brief explanation of why court accepted/rejected" or null,
  "confidence": "high" | "medium" | "low"
}}

CONTEXT (facts, concepts, issues):
{context_json}

DOCUMENT:
{document}

Respond with JSON: {{"arguments": [...]}}"""

HOLDINGS_PROMPT = """Extract ALL HOLDINGS (legal determinations) made by the court.

Holdings are the court's conclusions on legal issues. They:
- Answer the framed legal questions
- Apply legal concepts to facts
- Become precedent for future cases

Distinguish RATIO DECIDENDI (binding holdings) from OBITER DICTA (non-binding observations).

For each holding, provide:
{{
  "id": "h1", "h2", etc.,
  "text": "The holding statement",
  "start_char": <int>,
  "end_char": <int>,
  "surface_text": "First 150 chars of source text",
  "resolves_issue": "i1" or null (which issue this holding answers),
  "is_ratio": true (binding ratio) | false (obiter dicta),
  "novel": true (new legal principle) | false (applying existing law),
  "reasoning_summary": "Brief explanation of court's reasoning path",
  "schemes": ["rule_application", "precedent_following", ...] (argument schemes used),
  "confidence": "high" | "medium" | "low"
}}

CONTEXT (issues, concepts):
{context_json}

DOCUMENT:
{document}

Respond with JSON: {{"holdings": [...]}}"""

PRECEDENTS_PROMPT = """Extract ALL PRECEDENT citations from this judgment.

For each cited case, capture:
- The citation (exact as written)
- What proposition it's cited for
- How the court treats it

TREATMENT TYPES:
- followed: Court follows the precedent's ratio
- applied: Court applies precedent to similar facts
- distinguished: Court distinguishes (facts/law different)
- overruled: Court overrules the precedent
- doubted: Court expresses doubt about precedent
- explained: Court explains/interprets the precedent
- cited: Mentioned without specific treatment

For each precedent, provide:
{{
  "id": "p1", "p2", etc.,
  "citation": "Full citation string as written",
  "start_char": <int>,
  "end_char": <int>,
  "surface_text": "First 150 chars of source text",
  "case_name": "Normalized case name" or null,
  "case_year": <int> or null,
  "cited_case_id": "Unique ID if known" or null,
  "cited_proposition": "The legal principle being cited for",
  "cited_holding": "Specific holding referenced" or null,
  "treatment": "followed" | "applied" | "distinguished" | "overruled" | "doubted" | "explained" | "cited",
  "relevance": "central" | "supporting" | "mentioned" | "obiter",
  "confidence": "high" | "medium" | "low"
}}

DOCUMENT:
{document}

Respond with JSON: {{"precedents": [...]}}"""

OUTCOME_PROMPT = """Extract the OUTCOME (final disposition) of this case.

Look for the operative part, typically at the end:
- "Accordingly, the appeal is allowed..."
- "The petition is dismissed..."
- "The matter is remanded..."

DISPOSITION TYPES:
- allowed: Petition/appeal allowed (petitioner wins)
- dismissed: Petition/appeal dismissed (respondent wins)
- partly_allowed: Partial relief granted
- remanded: Sent back to lower court/authority
- modified: Lower court order modified
- set_aside: Lower court order set aside

Provide:
{{
  "outcome": {{
    "disposition": "allowed" | "dismissed" | "partly_allowed" | "remanded" | "modified" | "set_aside",
    "start_char": <int>,
    "end_char": <int>,
    "surface_text": "First 150 chars of source text",
    "binary": "accepted" (petitioner wins) | "rejected" (petitioner loses),
    "relief_summary": "What relief was granted/denied",
    "costs": "petitioner" | "respondent" | "none" | "shared" | null,
    "directions": ["Any directions to parties/lower courts", ...]
  }}
}}

DOCUMENT:
{document}

Respond with JSON: {{"outcome": {{...}} }}"""

EDGES_PROMPT = """Extract REASONING EDGES connecting the extracted nodes.

Edges represent the logical moves in legal reasoning:
- WHY does this fact matter? (fact TRIGGERS concept)
- HOW does this concept apply? (concept GROUNDS holding)  
- WHAT determines outcome? (holding DETERMINES outcome)

VALID EDGE RELATIONS by source->target type:

Fact → Concept: triggers, negates, partially_satisfies
Fact/Concept → Argument: supports, rebuts, undercuts
Concept → Concept: requires, excludes, specializes, conflicts_with
Concept → Holding: grounds, constrains
Argument → Issue: addresses, concedes
Argument → Argument: attacks, supports_arg, responds_to
Holding → Issue: resolves, partially_resolves
Holding → Outcome: determines, contributes_to
Holding → Precedent: follows, applies, distinguishes, overrules, doubts, explains
Precedent → Holding/Argument: supports

DIRECTION CONVENTION:
- Source is the "grounding" or "supporting" element
- Target is what is being affected/supported/grounded
- Example: f1 TRIGGERS c1 means fact f1 makes concept c1 applicable

For each edge, provide:
{{
  "id": "e1", "e2", etc.,
  "source": "f1" | "c1" | "h1" | etc.,
  "target": "c1" | "h1" | "outcome" | etc.,
  "relation": "triggers" | "grounds" | "supports" | etc.,
  "start_char": <int> or null (where connection is explicit in text),
  "end_char": <int> or null,
  "explanation": "Why this connection exists",
  "confidence": "high" | "medium" | "low" | "inferred",
  "strength": "strong" | "moderate" | "weak",
  "is_critical": true (removing breaks reasoning) | false
}}

RULES:
- If confidence is HIGH or MEDIUM, provide start_char/end_char
- If confidence is INFERRED, provide explanation
- Ensure relation is VALID for source->target type pair

EXTRACTED NODES:
{nodes_json}

DOCUMENT (focus on reasoning sections):
{document}

Respond with JSON: {{"edges": [...]}}"""

# =============================================================================
# v4: INTRA-CLUSTER EDGE EXTRACTION
# =============================================================================

INTRA_CLUSTER_EDGES_PROMPT = """Extract INTRA-CLUSTER REASONING EDGES within a SINGLE legal concept cluster.

IMPORTANT: All edges MUST connect ONLY nodes listed in this cluster. Do NOT invent node IDs.

CLUSTER CONCEPT:
- concept_id: {concept_id}
- label: {concept_label}

ONTOLOGY GUIDANCE:
- typical_edge_pattern: {typical_edge_pattern}

REQUIREMENTS (logic={logic}):
{requires_block}

DEFEATERS (what can undercut/break this doctrine):
{defeaters_block}

{edge_whitelist}

NODES IN THIS CLUSTER:
{cluster_context}

EDGE FORMAT:
{{
  "id": "e1",  
  "source": "node_id",
  "target": "node_id",
  "relation": "<relation from whitelist above>",
  "start_char": <int> or null,
  "end_char": <int> or null,
  "explanation": "why this connection exists",
  "confidence": "high" | "medium" | "low" | "inferred",
  "strength": "strong" | "moderate" | "weak",
  "is_critical": true | false
}}

RULES:
- If confidence is HIGH or MEDIUM, provide start_char/end_char (where the connection is explicit).
- If confidence is INFERRED, provide an explanation.
- Use ONLY valid relations for the source->target node types.

Respond with JSON: {{"edges": [...]}}"""

LINK_DISCOVERY_PROMPT = """Analyze the extracted graph for MISSING REASONING LINKS.

You have a partially extracted legal reasoning graph. Your task is to find IMPLICIT connections 
that weren't captured in the initial extraction, especially LONG-DISTANCE links.

WHAT TO LOOK FOR:

1. FACT → HOLDING gaps: Facts mentioned early that support holdings stated late
2. CONCEPT chains: Concept A requires Concept B, which grounds Holding H
3. PRECEDENT support: Precedents that establish doctrines used in holdings
4. ARGUMENT dependencies: Arguments that depend on earlier facts/concepts
5. COUNTER-ARGUMENT links: Arguments that directly respond to other arguments

ANALYSIS STRATEGY:
For each HOLDING, ask: "What facts and concepts MUST be true for this holding to stand?"
For each FACT, ask: "What legal concepts does this fact trigger or negate?"
For each PRECEDENT, ask: "Which holding relies on this precedent's ratio?"

EXTRACTED GRAPH:
{graph_summary}

HOLDINGS requiring support analysis:
{holdings_json}

FACTS that might have unexplored connections:
{facts_json}

DOCUMENT (for verification):
{document}

For each discovered link, provide:
{{
  "id": "ed1", "ed2", etc.,
  "source": "<node_id>",
  "target": "<node_id>",
  "relation": "<valid_relation>",
  "explanation": "Detailed reasoning why this link exists",
  "evidence_quote": "Relevant text from document supporting this link",
  "confidence": "medium" | "inferred",
  "discovery_type": "fact_to_holding" | "concept_chain" | "precedent_support" | "argument_dependency" | "counter_argument"
}}

Respond with JSON: {{"discovered_edges": [...]}}"""

JUSTIFICATION_SETS_PROMPT = """Extract JUSTIFICATION SETS for holdings and outcome.

A justification set groups supporting elements that TOGETHER justify a conclusion.
This enables COUNTERFACTUAL reasoning: "What if we removed fact F1?"

LOGIC TYPES:
- and: ALL members required (removing ANY breaks justification)
- or: ANY member sufficient (removing one doesn't break justification)

A holding may have MULTIPLE justification sets (alternative reasoning paths).

For each justification set, provide:
{{
  "id": "js1", "js2", etc.,
  "target_id": "h1" | "h2" | "outcome" (what this set justifies),
  "logic": "and" | "or",
  "member_edge_ids": ["e1", "e2", ...] (edges that belong to this set),
  "label": "Human-readable description of this justification path",
  "is_primary": true (main reasoning) | false (alternative),
  "confidence": "high" | "medium" | "low"
}}

EXAMPLE:
If holding H1 is justified by (Fact F1 + Concept C1) OR (Precedent P1 alone):
- JS1: target=h1, logic=and, members=[e_f1_h1, e_c1_h1], is_primary=true
- JS2: target=h1, logic=and, members=[e_p1_h1], is_primary=false

HOLDINGS & EDGES:
{context_json}

DOCUMENT:
{document}

Respond with JSON: {{"justification_sets": [...]}}"""


# =============================================================================
# EXTRACTION RESULT AND BASE PASS
# =============================================================================

@dataclass
class ExtractionResult:
    """Result of a single extraction pass."""
    success: bool
    data: Any
    raw_response: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0


class ExtractionPass(ABC):
    """Base class for extraction passes with consistent retry logic."""

    def __init__(
            self,
            client: LLMClient,
            config: ExtractionConfig,
            doc: SegmentedDocument
    ):
        self.client = client
        self.config = config
        self.doc = doc
        self.provenance = Provenance(
            extraction_method=ExtractionMethod.LLM,
            model_id=config.model_id,
            run_id=config.run_id,
            temperature=config.temperature,
            timestamp=datetime.utcnow().isoformat()
        )

    def make_anchor(
            self,
            start_char: int,
            end_char: int,
            surface_text: Optional[str] = None,
            quote_for_alignment: Optional[str] = None
    ) -> Optional[Anchor]:
        """Create an Anchor from char offsets with validation and repair.

        Args:
            start_char: Starting character offset
            end_char: Ending character offset
            surface_text: Optional surface text (first ~150 chars)
            quote_for_alignment: Optional verbatim quote to use for offset repair

        Returns:
            Anchor if valid, None if completely invalid and unrepairable
        """
        doc_len = self.doc.char_count

        # Validate basic offset sanity
        offsets_valid = (
                start_char is not None and
                end_char is not None and
                0 <= start_char < end_char <= doc_len
        )

        # If offsets invalid, try to repair via quote alignment
        if not offsets_valid:
            repair_quote = quote_for_alignment or surface_text
            if repair_quote:
                repaired = align_quote_to_span(self.doc.full_text, repair_quote)
                if repaired:
                    start_char, end_char = repaired
                    offsets_valid = True
                    logger.debug(f"Repaired anchor via quote alignment: {start_char}-{end_char}")

        # If still invalid, return None
        if not offsets_valid:
            logger.debug(f"Invalid anchor offsets: {start_char}-{end_char} (doc_len={doc_len})")
            return None

        # Get text at these offsets
        actual_text = self.doc.text_at(start_char, end_char)
        text_hash = self.doc.compute_text_hash(start_char, end_char)

        # Check for empty hash (indicates empty text extraction)
        if text_hash == EMPTY_ANCHOR_HASH or not actual_text.strip():
            # Try repair via quote if available
            repair_quote = quote_for_alignment or surface_text
            if repair_quote:
                repaired = align_quote_to_span(self.doc.full_text, repair_quote)
                if repaired:
                    start_char, end_char = repaired
                    actual_text = self.doc.text_at(start_char, end_char)
                    text_hash = self.doc.compute_text_hash(start_char, end_char)
                    logger.debug(f"Repaired empty anchor via quote alignment: {start_char}-{end_char}")

        # Final check - if still empty, return None
        if text_hash == EMPTY_ANCHOR_HASH or not actual_text.strip():
            logger.debug(f"Anchor produces empty text at {start_char}-{end_char}")
            return None

        seg = self.doc.get_segment_at(start_char, end_char)
        display = seg.display_location if seg else None

        if surface_text is None:
            surface_text = actual_text[:150]

        return Anchor(
            doc_id=self.doc.doc_id,
            start_char=start_char,
            end_char=end_char,
            text_hash=text_hash,
            display_location=display,
            surface_text=surface_text
        )

    def make_anchor_from_quote(self, quote: str) -> Optional[Anchor]:
        """Create an Anchor by finding a verbatim quote in the document.

        This is the preferred method for anchoring - more reliable than LLM offsets.

        Args:
            quote: Verbatim text to find in document

        Returns:
            Anchor if quote found, None otherwise
        """
        if not quote or not quote.strip():
            return None

        result = align_quote_to_span(self.doc.full_text, quote)
        if result is None:
            return None

        start_char, end_char = result
        return self.make_anchor(start_char, end_char, surface_text=quote[:150])

    def get_offsets_reference(self, max_paras: int = 30) -> str:
        """Get paragraph offset reference for prompts."""
        lines = []
        for para in self.doc.paragraphs[:max_paras]:
            preview = para.text[:80].replace('\n', ' ')
            lines.append(f"[{para.start_char}-{para.end_char}] {preview}...")
        return "\n".join(lines)

    def parse_json_response(self, response: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Parse JSON from LLM response, handling common issues."""
        # Remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"

    async def extract_with_retry(
            self,
            prompt: str,
            context: Dict = None,
            max_tokens: int = 4096
    ) -> ExtractionResult:
        """Execute extraction with retry logic."""
        last_response = ""

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.complete(
                    prompt=prompt,
                    system=SYSTEM_BASE,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens
                )
                last_response = response

                data, parse_error = self.parse_json_response(response)
                if parse_error:
                    if attempt < self.config.max_retries - 1:
                        prompt += f"\n\nPREVIOUS ERROR: {parse_error}\nPlease respond with valid JSON only."
                        continue
                    return ExtractionResult(
                        success=False,
                        data=None,
                        raw_response=response,
                        errors=[parse_error],
                        retry_count=attempt
                    )

                valid, errors, warnings = self.validate(data, context)

                if valid or attempt == self.config.max_retries - 1:
                    return ExtractionResult(
                        success=valid,
                        data=data,
                        raw_response=response,
                        errors=errors,
                        warnings=warnings,
                        retry_count=attempt
                    )

                # Add errors to prompt for retry
                prompt += f"\n\nVALIDATION ERRORS (please fix):\n" + "\n".join(errors)

            except Exception as e:
                logger.error(f"Extraction error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    return ExtractionResult(
                        success=False,
                        data=None,
                        raw_response=last_response,
                        errors=[f"Exception: {str(e)}"],
                        retry_count=attempt
                    )

        return ExtractionResult(
            success=False,
            data=None,
            raw_response=last_response,
            errors=["Max retries exceeded"]
        )

    @abstractmethod
    async def extract(self, context: Dict = None) -> ExtractionResult:
        """Run the extraction pass."""
        pass

    @abstractmethod
    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        """Validate extracted data. Returns (valid, errors, warnings)."""
        pass


# =============================================================================
# EXTRACTION PASSES
# =============================================================================

class FactsExtractionPass(ExtractionPass):
    """Extract facts from the document."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = FACTS_PROMPT.format(
            document=self.doc.full_text[:self.config.max_doc_chars],
            offsets=self.get_offsets_reference(),
            min_facts=self.config.min_facts,
            max_facts=self.config.max_facts
        )
        return await self.extract_with_retry(prompt, context)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "facts" not in data:
            errors.append("Missing 'facts' key")
            return False, errors, warnings

        facts = data["facts"]

        if len(facts) < self.config.min_facts:
            warnings.append(f"Only {len(facts)} facts (min: {self.config.min_facts})")

        seen_ids = set()
        for i, f in enumerate(facts):
            # Required fields
            for field in ["id", "text", "start_char", "end_char", "fact_type"]:
                if field not in f:
                    errors.append(f"Fact {i}: missing '{field}'")

            # Duplicate ID check
            if f.get("id") in seen_ids:
                errors.append(f"Duplicate fact ID: {f.get('id')}")
            seen_ids.add(f.get("id"))

            # Offset validation
            if "start_char" in f and "end_char" in f:
                if f["start_char"] >= f["end_char"]:
                    errors.append(f"Fact {f.get('id', i)}: start_char >= end_char")
                if f["end_char"] > self.doc.char_count:
                    warnings.append(f"Fact {f.get('id', i)}: end_char exceeds doc length")

            # Enum validation
            if f.get("fact_type") and f["fact_type"] not in [ft.value for ft in FactType]:
                errors.append(f"Fact {f.get('id', i)}: invalid fact_type '{f['fact_type']}'")

        return len(errors) == 0, errors, warnings

    def to_nodes(self, data: Dict) -> List[FactNode]:
        """Convert extracted data to FactNode objects."""
        nodes = []
        for f in data.get("facts", []):
            try:
                # Try to create anchor, using surface_text as repair quote
                anchor = self.make_anchor(
                    f.get("start_char", -1),
                    f.get("end_char", -1),
                    surface_text=f.get("surface_text"),
                    quote_for_alignment=f.get("surface_text")
                )

                # Determine confidence - downgrade if anchor is invalid
                conf = f.get("confidence") or "high"
                if anchor is None or not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"

                node = FactNode(
                    id=f["id"],
                    text=f["text"],
                    anchor=anchor,
                    fact_type=FactType(f["fact_type"]),
                    actor_source=coerce_actor_type(f.get("actor_source")),
                    date=f.get("date"),
                    date_approximate=bool(f.get("date_approximate", False)),
                    disputed_by=coerce_actor_type(f.get("disputed_by")),
                    court_finding=f.get("court_finding"),
                    confidence=Confidence(conf),
                    provenance=self.provenance
                )
                nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not create FactNode: {e}")
        return nodes


class ConceptsExtractionPass(ExtractionPass):
    """Extract legal concepts from the document."""

    def __init__(self, client: LLMClient, config: ExtractionConfig, doc: SegmentedDocument):
        super().__init__(client, config, doc)
        self.ontology = config.load_ontology()

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = CONCEPTS_PROMPT.format(
            document=self.doc.full_text[:self.config.max_doc_chars],
            ontology_excerpt=self._get_ontology_excerpt(),
            facts_json=json.dumps(context.get("facts", []) if context else [], indent=2)
        )
        return await self.extract_with_retry(prompt, context)

    def _get_ontology_excerpt(self) -> str:
        """Get ontology IDs for prompt."""
        # Default Indian legal concepts
        common = [
            "CONST_ART14 - Article 14 (Equality before Law)",
            "CONST_ART19 - Article 19 (Freedom of Speech etc.)",
            "CONST_ART21 - Article 21 (Right to Life & Personal Liberty)",
            "CONST_ART32 - Article 32 (Constitutional Remedies)",
            "CONST_ART136 - Article 136 (Special Leave Petition)",
            "CONST_ART226 - Article 226 (High Court Writ Jurisdiction)",
            "CONST_ART227 - Article 227 (HC Superintendence)",
            "DOCTRINE_NATURAL_JUSTICE - Principles of Natural Justice",
            "DOCTRINE_RES_JUDICATA - Res Judicata",
            "DOCTRINE_ESTOPPEL - Estoppel",
            "DOCTRINE_PROPORTIONALITY - Proportionality Principle",
            "DOCTRINE_LEGITIMATE_EXPECTATION - Legitimate Expectation",
            "TEST_WEDNESBURY - Wednesbury Unreasonableness",
            "TEST_RATIONAL_NEXUS - Rational Nexus Test",
            "CPC_S151 - Section 151 CPC (Inherent Powers)",
            "CRPC_S482 - Section 482 CrPC (Inherent Powers)",
            "CRPC_S438 - Section 438 CrPC (Anticipatory Bail)",
            "IPC_S302 - Section 302 IPC (Murder)",
            "IPC_S34 - Section 34 IPC (Common Intention)",
            "EVIDENCE_S3 - Section 3 Evidence Act (Definitions)",
        ]

        # Add from loaded ontology
        if self.ontology and "concepts" in self.ontology:
            for cid, info in list(self.ontology["concepts"].items())[:30]:
                label = info.get("label", cid)
                common.append(f"{cid} - {label}")

        return "\n".join(common)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "concepts" not in data:
            errors.append("Missing 'concepts' key")
            return False, errors, warnings

        concepts = data["concepts"]

        if len(concepts) < self.config.min_concepts:
            warnings.append(f"Only {len(concepts)} concepts (min: {self.config.min_concepts})")

        seen_ids = set()
        for i, c in enumerate(concepts):
            for field in ["id", "concept_id", "start_char", "end_char", "relevance"]:
                if field not in c:
                    errors.append(f"Concept {i}: missing '{field}'")

            if c.get("id") in seen_ids:
                errors.append(f"Duplicate concept ID: {c.get('id')}")
            seen_ids.add(c.get("id"))

            if c.get("relevance") and c["relevance"] not in [r.value for r in Relevance]:
                errors.append(f"Concept {c.get('id', i)}: invalid relevance '{c['relevance']}'")

        return len(errors) == 0, errors, warnings

    def to_nodes(self, data: Dict) -> List[ConceptNode]:
        """Convert extracted data to ConceptNode objects."""
        nodes = []
        for c in data.get("concepts", []):
            try:
                # Handle interpretation anchor
                interp_anchor = None
                if c.get("interpretation_start_char") is not None and c.get("interpretation_end_char") is not None:
                    interp_anchor = self.make_anchor(
                        c["interpretation_start_char"],
                        c["interpretation_end_char"]
                    )

                # Create main anchor with repair support
                anchor = self.make_anchor(
                    c.get("start_char", -1),
                    c.get("end_char", -1),
                    surface_text=c.get("surface_text"),
                    quote_for_alignment=c.get("surface_text")
                )

                # Determine confidence - downgrade if anchor is invalid
                conf = c.get("confidence") or "high"
                if anchor is None or not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"

                node = ConceptNode(
                    id=c["id"],
                    concept_id=c["concept_id"],
                    anchor=anchor,
                    relevance=Relevance(c["relevance"]),
                    kind=ConceptKind(c["kind"]) if c.get("kind") else None,
                    interpretation=c.get("interpretation"),
                    interpretation_anchor=interp_anchor,
                    unlisted_label=c.get("unlisted_label"),
                    unlisted_description=c.get("unlisted_description"),
                    confidence=Confidence(conf),
                    provenance=self.provenance
                )
                nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not create ConceptNode: {e}")
        return nodes


class IssuesExtractionPass(ExtractionPass):
    """Extract legal issues from the document."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = ISSUES_PROMPT.format(
            document=self.doc.full_text[:self.config.max_doc_chars],
            concepts_json=json.dumps(context.get("concepts", []) if context else [], indent=2)
        )
        return await self.extract_with_retry(prompt, context)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "issues" not in data:
            errors.append("Missing 'issues' key")
            return False, errors, warnings

        for i, issue in enumerate(data["issues"]):
            for field in ["id", "text", "start_char", "end_char"]:
                if field not in issue:
                    errors.append(f"Issue {i}: missing '{field}'")

        return len(errors) == 0, errors, warnings

    def to_nodes(self, data: Dict) -> List[IssueNode]:
        nodes = []
        for iss in data.get("issues", []):
            try:
                # Create anchor with repair support
                anchor = self.make_anchor(
                    iss.get("start_char", -1),
                    iss.get("end_char", -1),
                    surface_text=iss.get("surface_text"),
                    quote_for_alignment=iss.get("surface_text")
                )

                # Determine confidence - downgrade if anchor is invalid
                conf = iss.get("confidence") or "high"
                if anchor is None or not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"

                node = IssueNode(
                    id=iss["id"],
                    text=iss["text"],
                    anchor=anchor,
                    issue_number=iss.get("issue_number"),
                    framed_by=coerce_actor_type(iss.get("framed_by"), default="court"),
                    primary_concepts=iss.get("primary_concepts", []),
                    answer=iss.get("answer"),
                    confidence=Confidence(conf),
                    provenance=self.provenance
                )
                nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not create IssueNode: {e}")
        return nodes


class ArgumentsExtractionPass(ExtractionPass):
    """Extract arguments from the document."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = ARGUMENTS_PROMPT.format(
            document=self.doc.full_text[:self.config.max_doc_chars],
            context_json=json.dumps(context, indent=2) if context else "{}"
        )
        return await self.extract_with_retry(prompt, context, max_tokens=6000)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "arguments" not in data:
            errors.append("Missing 'arguments' key")
            return False, errors, warnings

        for i, arg in enumerate(data["arguments"]):
            for field in ["id", "claim", "start_char", "end_char", "actor", "schemes"]:
                if field not in arg:
                    errors.append(f"Argument {i}: missing '{field}'")

            # Validate schemes (normalize; unknown schemes are mapped to OTHER downstream)
            if arg.get("schemes"):
                for scheme in arg["schemes"]:
                    _ = normalize_argument_scheme(scheme)

        return len(errors) == 0, errors, warnings

    def to_nodes(self, data: Dict) -> List[ArgumentNode]:
        nodes = []
        for arg in data.get("arguments", []):
            try:
                court_anchor = None
                if arg.get("court_response_start_char") is not None and arg.get("court_response_end_char") is not None:
                    court_anchor = self.make_anchor(
                        arg["court_response_start_char"],
                        arg["court_response_end_char"]
                    )

                # Parse schemes (normalize for IL-TUR; keep unmapped info)
                raw_schemes = arg.get("schemes") or ["rule_application"]
                schemes: List[ArgumentScheme] = []
                unmapped_schemes: List[str] = []
                for s in raw_schemes:
                    norm_s = normalize_argument_scheme(s)
                    try:
                        schemes.append(ArgumentScheme(norm_s))
                    except ValueError:
                        unmapped_schemes.append(str(s))
                        schemes.append(ArgumentScheme.OTHER)

                # De-duplicate while preserving order
                seen = set()
                schemes = [x for x in schemes if not (x in seen or seen.add(x))]
                if not schemes:
                    schemes = [ArgumentScheme.RULE_APPLICATION]

                qualifiers = arg.get("qualifiers")
                if unmapped_schemes:
                    extra = "unmapped_schemes:" + ",".join(unmapped_schemes)
                    qualifiers = (qualifiers + "; " if qualifiers else "") + extra

                # Create anchor with repair support
                anchor = self.make_anchor(
                    arg.get("start_char", -1),
                    arg.get("end_char", -1),
                    surface_text=arg.get("surface_text"),
                    quote_for_alignment=arg.get("surface_text")
                )

                # Determine confidence - downgrade if anchor is invalid
                conf = arg.get("confidence") or "high"
                if anchor is None or not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"

                node = ArgumentNode(
                    id=arg["id"],
                    claim=arg["claim"],
                    anchor=anchor,
                    actor=coerce_actor_type(arg.get("actor"), default="petitioner"),
                    schemes=schemes,
                    qualifiers=qualifiers,
                    court_response=arg.get("court_response"),
                    court_response_anchor=court_anchor,
                    court_reasoning=arg.get("court_reasoning"),
                    confidence=Confidence(conf),
                    provenance=self.provenance
                )
                nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not create ArgumentNode: {e}")
        return nodes


class HoldingsExtractionPass(ExtractionPass):
    """Extract holdings from the document."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = HOLDINGS_PROMPT.format(
            document=self.doc.full_text[:self.config.max_doc_chars],
            context_json=json.dumps(context, indent=2) if context else "{}"
        )
        return await self.extract_with_retry(prompt, context)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "holdings" not in data:
            errors.append("Missing 'holdings' key")
            return False, errors, warnings

        if len(data["holdings"]) < self.config.min_holdings:
            warnings.append(f"Only {len(data['holdings'])} holdings (min: {self.config.min_holdings})")

        for i, h in enumerate(data["holdings"]):
            for field in ["id", "text", "start_char", "end_char"]:
                if field not in h:
                    errors.append(f"Holding {i}: missing '{field}'")

        return len(errors) == 0, errors, warnings

    def to_nodes(self, data: Dict) -> List[HoldingNode]:
        nodes = []
        for h in data.get("holdings", []):
            try:
                schemes: List[ArgumentScheme] = []
                raw_schemes = h.get("schemes") or ["rule_application"]
                for s in raw_schemes:
                    norm_s = normalize_argument_scheme(s)
                    try:
                        schemes.append(ArgumentScheme(norm_s))
                    except ValueError:
                        schemes.append(ArgumentScheme.OTHER)

                # De-duplicate while preserving order
                seen = set()
                schemes = [x for x in schemes if not (x in seen or seen.add(x))]
                if not schemes:
                    schemes = [ArgumentScheme.RULE_APPLICATION]

                # Create anchor with repair support
                anchor = self.make_anchor(
                    h.get("start_char", -1),
                    h.get("end_char", -1),
                    surface_text=h.get("surface_text"),
                    quote_for_alignment=h.get("surface_text")
                )

                # Determine confidence - downgrade if anchor is invalid
                conf = h.get("confidence") or "high"
                if anchor is None or not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"

                node = HoldingNode(
                    id=h["id"],
                    text=h["text"],
                    anchor=anchor,
                    resolves_issue=h.get("resolves_issue"),
                    is_ratio=h.get("is_ratio", True),
                    novel=h.get("novel", False),
                    reasoning_summary=h.get("reasoning_summary"),
                    schemes=schemes,
                    confidence=Confidence(conf),
                    provenance=self.provenance
                )
                nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not create HoldingNode: {e}")
        return nodes


class PrecedentsExtractionPass(ExtractionPass):
    """Extract precedent citations from the document."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = PRECEDENTS_PROMPT.format(
            document=self.doc.full_text[:self.config.max_doc_chars]
        )
        return await self.extract_with_retry(prompt, context)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "precedents" not in data:
            errors.append("Missing 'precedents' key")
            return False, errors, warnings

        for i, p in enumerate(data["precedents"]):
            for field in ["id", "citation", "start_char", "end_char"]:
                if field not in p:
                    errors.append(f"Precedent {i}: missing '{field}'")

        return len(errors) == 0, errors, warnings

    def to_nodes(self, data: Dict) -> List[PrecedentNode]:
        nodes = []
        for p in data.get("precedents", []):
            try:
                treatment = None
                if p.get("treatment"):
                    try:
                        treatment = PrecedentTreatment(p["treatment"])
                    except ValueError:
                        logger.warning(f"Unknown treatment: {p['treatment']}")

                # Create anchor with repair support
                anchor = self.make_anchor(
                    p.get("start_char", -1),
                    p.get("end_char", -1),
                    surface_text=p.get("surface_text"),
                    quote_for_alignment=p.get("surface_text")
                )

                # Determine confidence - downgrade if anchor is invalid
                conf = p.get("confidence") or "high"
                if anchor is None or not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"

                node = PrecedentNode(
                    id=p["id"],
                    citation=p["citation"],
                    anchor=anchor,
                    case_name=p.get("case_name"),
                    case_year=p.get("case_year"),
                    cited_case_id=p.get("cited_case_id"),
                    cited_proposition=p.get("cited_proposition"),
                    cited_holding=p.get("cited_holding"),
                    treatment=treatment,
                    relevance=Relevance(p.get("relevance") or "supporting"),
                    confidence=Confidence(conf),
                    provenance=self.provenance
                )
                nodes.append(node)
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not create PrecedentNode: {e}")
        return nodes


class OutcomeExtractionPass(ExtractionPass):
    """Extract case outcome from the document."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = OUTCOME_PROMPT.format(
            document=self.doc.full_text[:self.config.max_doc_chars]
        )
        return await self.extract_with_retry(prompt, context)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "outcome" not in data or data["outcome"] is None:
            errors.append("Missing 'outcome'")
            return False, errors, warnings

        outcome = data["outcome"]
        for field in ["disposition", "start_char", "end_char", "binary"]:
            if field not in outcome:
                errors.append(f"Outcome: missing '{field}'")

        return len(errors) == 0, errors, warnings

    def to_node(self, data: Dict) -> Optional[OutcomeNode]:
        outcome = data.get("outcome")
        if not outcome:
            return None

        try:
            # Create anchor with repair support
            anchor = self.make_anchor(
                outcome.get("start_char", -1),
                outcome.get("end_char", -1),
                surface_text=outcome.get("surface_text"),
                quote_for_alignment=outcome.get("surface_text")
            )

            return OutcomeNode(
                disposition=Disposition(outcome["disposition"]),
                anchor=anchor,
                binary=outcome["binary"],
                relief_summary=outcome.get("relief_summary"),
                costs=outcome.get("costs"),
                directions=outcome.get("directions", []),
                provenance=self.provenance
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not create OutcomeNode: {e}")
            return None


# =============================================================================
# v4: INTRA-CLUSTER EDGE EXTRACTION PASS
# =============================================================================


def _format_bullets(items: List[str], max_items: int = 20) -> str:
    if not items:
        return "- (none)"
    lines = []
    for it in items[:max_items]:
        lines.append(f"- {it}")
    if len(items) > max_items:
        lines.append(f"- ... ({len(items) - max_items} more)")
    return "\n".join(lines)


def build_cluster_context(cluster: ConceptCluster, graph: LegalReasoningGraph) -> str:
    """Build a compact, anchor-aware listing of cluster nodes for prompting."""

    def _anchor_span(n: Any) -> str:
        a = getattr(n, "anchor", None)
        if not a:
            return ""
        return f"[{a.start_char}-{a.end_char}]"

    def _lines(title: str, ids: List[str]) -> List[str]:
        out = [f"{title} ({len(ids)}):"]
        for nid in ids:
            node = graph.get_node(nid)
            if not node:
                continue
            txt = _node_text_for_matching(node).replace("\n", " ")
            if len(txt) > 180:
                txt = txt[:180] + "..."
            out.append(f"  - {nid} {_anchor_span(node)}: {txt}")
        return out

    parts: List[str] = []
    parts.extend(_lines("FACTS", cluster.facts))
    parts.append("")
    parts.extend(_lines("CONCEPTS", cluster.concepts))
    parts.append("")
    parts.extend(_lines("ISSUES", cluster.issues))
    parts.append("")
    parts.extend(_lines("ARGUMENTS", cluster.arguments))
    parts.append("")
    parts.extend(_lines("PRECEDENTS", cluster.precedents))
    parts.append("")
    parts.extend(_lines("HOLDINGS", cluster.holdings))
    return "\n".join(parts).strip()


class IntraClusterEdgesExtractionPass(ExtractionPass):
    """Extract edges within a single concept cluster (v4)."""

    def __init__(
            self,
            client: LLMClient,
            config: ExtractionConfig,
            doc: SegmentedDocument,
            cluster: ConceptCluster,
            graph: LegalReasoningGraph,
            ontology_concept: Optional[Dict] = None
    ):
        super().__init__(client, config, doc)
        self.cluster = cluster
        self.graph = graph
        self.ontology_concept = ontology_concept or {}

    async def extract(self, context: Dict = None) -> ExtractionResult:
        requires_block = _format_bullets(self.cluster.requires)
        defeaters_block = _format_bullets(self.cluster.defeaters)
        cluster_context = build_cluster_context(self.cluster, self.graph)

        # Build per-type edge whitelist for this cluster
        node_types = get_node_types_in_cluster(self.cluster)
        edge_whitelist = build_cluster_edge_whitelist(node_types, exclude_structural=True)

        prompt = INTRA_CLUSTER_EDGES_PROMPT.format(
            concept_id=self.cluster.concept_id,
            concept_label=self.cluster.concept_label,
            typical_edge_pattern=str(self.ontology_concept.get("typical_edge_pattern", "")),
            logic=self.cluster.logic,
            requires_block=requires_block,
            defeaters_block=defeaters_block,
            edge_whitelist=edge_whitelist,
            cluster_context=cluster_context
        )

        return await self.extract_with_retry(prompt, context, max_tokens=6000)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if not isinstance(data, dict) or "edges" not in data:
            errors.append("Missing 'edges' key")
            return False, errors, warnings

        allowed_ids = set(
            self.cluster.facts
            + self.cluster.concepts
            + self.cluster.issues
            + self.cluster.arguments
            + self.cluster.precedents
            + self.cluster.holdings
        )

        seen_ids = set()
        for i, e in enumerate(data.get("edges", [])):
            for field in ["source", "target", "relation"]:
                if field not in e:
                    errors.append(f"Edge {i}: missing '{field}'")

            # Duplicate check (within this cluster response)
            if e.get("id") and e.get("id") in seen_ids:
                warnings.append(f"Duplicate edge ID in cluster response: {e.get('id')}")
            if e.get("id"):
                seen_ids.add(e.get("id"))

            # Enforce intra-cluster locality
            if e.get("source") and e["source"] not in allowed_ids:
                errors.append(f"Edge {e.get('id', i)}: source '{e['source']}' not in this cluster")
            if e.get("target") and e["target"] not in allowed_ids:
                errors.append(f"Edge {e.get('id', i)}: target '{e['target']}' not in this cluster")

            # Relation validation
            if e.get("source") and e.get("target") and e.get("relation"):
                valid, msg = validate_edge_relation(e["source"], e["target"], coerce_edge_relation(e.get("relation")))
                if not valid:
                    warnings.append(f"Edge {e.get('id', i)}: {msg}")

            # Anchor rules
            # If the key exists but is null, .get() returns None; treat that as default.
            conf = e.get("confidence") or "high"
            # If anchor is missing, we downgrade confidence during to_edges (no warning needed here).
            if conf == "inferred" and not e.get("explanation"):
                errors.append(f"Edge {e.get('id', i)}: INFERRED requires explanation")

        return len(errors) == 0, errors, warnings

    def to_edges(self, data: Dict, edge_id_prefix: str) -> List[Edge]:
        edges: List[Edge] = []
        warnings = []
        for idx, e in enumerate(data.get("edges", [])):
            try:
                # Normalize confidence/anchor consistency
                # If the key exists but is null, .get() returns None; treat that as default.
                conf = e.get("confidence") or "high"
                start = e.get("start_char")
                end = e.get("end_char")
                explanation = e.get("explanation")

                if conf in ["high", "medium"] and (start is None or end is None):
                    # Downgrade rather than emit invalid HIGH/MEDIUM without anchor
                    conf = "inferred"
                    if not explanation:
                        explanation = "Inferred within concept cluster context"

                anchor = None
                if start is not None and end is not None:
                    anchor = self.make_anchor(start, end)

                # Fix 4: Downgrade confidence if anchor is invalid (empty hash, etc.)
                if anchor is not None and not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"
                        if not explanation:
                            explanation = "Anchor validation failed; inferred from context"
                    anchor = None  # Don't keep invalid anchors

                # Repair relation/direction to satisfy VALID_EDGE_RELATIONS
                src = e["source"]
                tgt = e["target"]
                rel_raw = coerce_edge_relation(e.get("relation"))
                ok_rel, _rel_msg = validate_edge_relation(src, tgt, rel_raw)
                repair_note = ""
                if not ok_rel:
                    new_src, new_tgt, new_rel, repair_note = repair_edge_relation(src, tgt, rel_raw)
                    if new_src is None:
                        warnings.append(
                            f"Edge {e.get('id', '(no-id)')}: dropped (unrepairable relation {rel_raw} for {get_node_type_from_id(src)} -> {get_node_type_from_id(tgt)})"
                        )
                        continue
                    src, tgt, rel_raw = new_src, new_tgt, new_rel
                    # If we repaired direction or relation, downgrade confidence to avoid overstating certainty
                    if conf in ("high", "medium"):
                        conf = "inferred"
                if repair_note:
                    explanation = (
                        explanation + f" [REPAIRED: {repair_note}]" if explanation else f"[REPAIRED: {repair_note}]")
                edge = Edge(
                    id=f"{edge_id_prefix}{idx + 1}",
                    source=src,
                    target=tgt,
                    relation=EdgeRelation(rel_raw),
                    anchor=anchor,
                    explanation=explanation,
                    confidence=Confidence(conf),
                    strength=e.get("strength", "strong"),
                    support_group_ids=e.get("support_group_ids", []),
                    is_critical=e.get("is_critical", False),
                    provenance=self.provenance
                )
                edges.append(edge)
            except (KeyError, ValueError) as ex:
                logger.warning(f"Could not create intra-cluster Edge: {ex}")
        return edges


class EdgesExtractionPass(ExtractionPass):
    """Extract reasoning edges connecting nodes."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = EDGES_PROMPT.format(
            document=self.doc.full_text[:60000],  # More text for edge context
            nodes_json=json.dumps(context, indent=2) if context else "{}"
        )
        return await self.extract_with_retry(prompt, context, max_tokens=8000)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "edges" not in data:
            errors.append("Missing 'edges' key")
            return False, errors, warnings

        # Collect all node IDs
        all_node_ids = set()
        if context:
            for key in ["facts", "concepts", "issues", "arguments", "holdings", "precedents"]:
                for node in context.get(key, []):
                    all_node_ids.add(node.get("id"))
            if context.get("outcome"):
                all_node_ids.add("outcome")

        seen_ids = set()
        for i, e in enumerate(data["edges"]):
            # Required fields
            for field in ["id", "source", "target", "relation"]:
                if field not in e:
                    errors.append(f"Edge {i}: missing '{field}'")

            # Duplicate check
            if e.get("id") in seen_ids:
                errors.append(f"Duplicate edge ID: {e.get('id')}")
            seen_ids.add(e.get("id"))

            # Endpoint validation
            if e.get("source") and e["source"] not in all_node_ids:
                warnings.append(f"Edge {e.get('id', i)}: source '{e['source']}' not found")
            if e.get("target") and e["target"] not in all_node_ids:
                warnings.append(f"Edge {e.get('id', i)}: target '{e['target']}' not found")

            # Relation validation
            if e.get("source") and e.get("target") and e.get("relation"):
                valid, msg = validate_edge_relation(e["source"], e["target"], coerce_edge_relation(e.get("relation")))
                if not valid:
                    warnings.append(f"Edge {e.get('id', i)}: {msg}")

            # Anchor rules
            conf = e.get("confidence") or "high"
            # If anchor is missing, we downgrade confidence during to_edges (no warning needed here).
            if conf == "inferred":
                if not e.get("explanation"):
                    errors.append(f"Edge {e.get('id', i)}: INFERRED requires explanation")

        return len(errors) == 0, errors, warnings

    def to_edges(self, data: Dict) -> List[Edge]:
        edges = []
        warnings = []
        for e in data.get("edges", []):
            try:
                anchor = None
                if e.get("start_char") is not None and e.get("end_char") is not None:
                    anchor = self.make_anchor(e["start_char"], e["end_char"])

                # If there is no explicit evidence span, downgrade confidence.
                conf = e.get("confidence") or "high"
                explanation = e.get("explanation")
                if conf in ["high", "medium"] and anchor is None:
                    conf = "inferred"
                    if not explanation:
                        explanation = "Implicit connection inferred (no explicit anchor span)."

                # Fix 4: Downgrade confidence if anchor is invalid (empty hash, etc.)
                if anchor is not None and not is_anchor_valid(anchor):
                    if conf in ("high", "medium"):
                        conf = "inferred"
                        if not explanation:
                            explanation = "Anchor validation failed; inferred from context"
                    anchor = None  # Don't keep invalid anchors

                # Repair relation/direction to satisfy VALID_EDGE_RELATIONS
                src = e["source"]
                tgt = e["target"]
                rel_raw = coerce_edge_relation(e.get("relation"))
                ok_rel, _rel_msg = validate_edge_relation(src, tgt, rel_raw)
                repair_note = ""
                if not ok_rel:
                    new_src, new_tgt, new_rel, repair_note = repair_edge_relation(src, tgt, rel_raw)
                    if new_src is None:
                        warnings.append(
                            f"Edge {e.get('id', '(no-id)')}: dropped (unrepairable relation {rel_raw} for {get_node_type_from_id(src)} -> {get_node_type_from_id(tgt)})"
                        )
                        continue
                    src, tgt, rel_raw = new_src, new_tgt, new_rel
                    # If we repaired direction or relation, downgrade confidence to avoid overstating certainty
                    if conf in ("high", "medium"):
                        conf = "inferred"
                if repair_note:
                    explanation = (
                        explanation + f" [REPAIRED: {repair_note}]" if explanation else f"[REPAIRED: {repair_note}]")
                edge = Edge(
                    id=e["id"],
                    source=src,
                    target=tgt,
                    relation=EdgeRelation(rel_raw),
                    anchor=anchor,
                    explanation=explanation,
                    confidence=Confidence(conf),
                    strength=e.get("strength", "strong"),
                    support_group_ids=e.get("support_group_ids", []),
                    is_critical=e.get("is_critical", False),
                    provenance=self.provenance
                )
                edges.append(edge)
            except (KeyError, ValueError) as ex:
                logger.warning(f"Could not create Edge: {ex}")
        return edges


class LinkDiscoveryPass(ExtractionPass):
    """Discover long-distance and implicit reasoning links."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        # Build graph summary
        graph_summary = self._build_graph_summary(context)

        prompt = LINK_DISCOVERY_PROMPT.format(
            document=self.doc.full_text[:50000],
            graph_summary=graph_summary,
            holdings_json=json.dumps(context.get("holdings", []), indent=2) if context else "[]",
            facts_json=json.dumps(context.get("facts", []), indent=2) if context else "[]"
        )
        return await self.extract_with_retry(prompt, context, max_tokens=6000)

    def _build_graph_summary(self, context: Dict) -> str:
        """Build a summary of the current graph for the prompt."""
        if not context:
            return "No nodes extracted yet."

        lines = []
        lines.append(f"FACTS ({len(context.get('facts', []))}):")
        for f in context.get("facts", [])[:10]:
            lines.append(f"  {f['id']}: {f['text'][:80]}...")

        lines.append(f"\nCONCEPTS ({len(context.get('concepts', []))}):")
        for c in context.get("concepts", [])[:10]:
            lines.append(f"  {c['id']}: {c['concept_id']}")

        lines.append(f"\nHOLDINGS ({len(context.get('holdings', []))}):")
        for h in context.get("holdings", []):
            lines.append(f"  {h['id']}: {h['text'][:80]}...")

        lines.append(f"\nEXISTING EDGES ({len(context.get('edges', []))}):")
        for e in context.get("edges", [])[:15]:
            lines.append(f"  {e.get('source')} --{e.get('relation')}--> {e.get('target')}")

        return "\n".join(lines)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "discovered_edges" not in data:
            # Optional pass - no edges is okay
            return True, errors, warnings

        for i, e in enumerate(data["discovered_edges"]):
            for field in ["id", "source", "target", "relation", "explanation"]:
                if field not in e:
                    errors.append(f"Discovered edge {i}: missing '{field}'")

        return len(errors) == 0, errors, warnings

    def to_edges(self, data: Dict, existing_edge_ids: Set[str], node_anchors: Optional[Dict[str, Anchor]] = None) -> \
    List[Edge]:
        """Convert discovered edges, avoiding duplicates and anchoring evidence when possible."""
        edges: List[Edge] = []

        for e in data.get("discovered_edges", []):
            try:
                # Generate unique ID if conflicts
                edge_id = e.get("id")
                if not edge_id:
                    continue
                if edge_id in existing_edge_ids:
                    edge_id = f"ld_{edge_id}"

                source = e.get("source")
                target = e.get("target")
                rel_str = e.get("relation")
                if not source or not target or not rel_str:
                    continue

                # Validate relation compatibility (skip invalid)
                ok, msg = validate_edge_relation(source, target, rel_str)
                if not ok:
                    logger.warning(f"Skipping discovered edge {edge_id}: {msg}")
                    continue

                # Evidence anchoring from quote (best-effort)
                evidence_quote = e.get("evidence_quote")
                anchor = None
                if evidence_quote:
                    span = align_quote_to_span(self.doc.full_text, evidence_quote)
                    if span:
                        anchor = self.make_anchor(span[0], span[1], surface_text=evidence_quote[:150])

                        # Attach source/target spans as secondary spans (if available)
                        if node_anchors:
                            sec: List[Tuple[int, int]] = []
                            sa = node_anchors.get(source)
                            ta = node_anchors.get(target)
                            if sa:
                                sec.append((sa.start_char, sa.end_char))
                            if ta:
                                sec.append((ta.start_char, ta.end_char))
                            if sec:
                                anchor.secondary_spans = sec

                conf_str = e.get("confidence") or "inferred"
                # Avoid schema warnings: MEDIUM without anchor → downgrade to inferred
                if anchor is None and conf_str in ("high", "medium"):
                    conf_str = "inferred"

                explanation = e.get("explanation")

                edge = Edge(
                    id=edge_id,
                    source=source,
                    target=target,
                    relation=EdgeRelation(rel_str),
                    anchor=anchor,
                    explanation=explanation,
                    confidence=Confidence(conf_str),
                    strength=e.get("strength", "moderate"),
                    support_group_ids=[],
                    is_critical=False,
                    provenance=self.provenance
                )
                edges.append(edge)

            except (KeyError, ValueError) as ex:
                logger.warning(f"Could not create discovered Edge: {ex}")

        return edges


class JustificationSetsExtractionPass(ExtractionPass):
    """Extract justification sets for counterfactual reasoning."""

    async def extract(self, context: Dict = None) -> ExtractionResult:
        prompt = JUSTIFICATION_SETS_PROMPT.format(
            document=self.doc.full_text[:40000],
            context_json=json.dumps(context, indent=2) if context else "{}"
        )
        return await self.extract_with_retry(prompt, context)

    def validate(self, data: Any, context: Dict = None) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []

        if "justification_sets" not in data:
            warnings.append("No justification_sets extracted")
            return True, errors, warnings  # Optional

        for i, js in enumerate(data["justification_sets"]):
            for field in ["id", "target_id", "logic", "member_edge_ids"]:
                if field not in js:
                    errors.append(f"JustificationSet {i}: missing '{field}'")

        return len(errors) == 0, errors, warnings

    def to_nodes(self, data: Dict, edges: List[Edge]) -> List[JustificationSetNode]:
        """Convert to JustificationSetNode and update edge support_group_ids."""
        nodes = []
        edge_by_id = {e.id: e for e in edges}

        for js in data.get("justification_sets", []):
            try:
                node = JustificationSetNode(
                    id=js["id"],
                    target_id=js["target_id"],
                    logic=JustificationLogic(js["logic"]),
                    label=js.get("label"),
                    is_primary=js.get("is_primary", False),
                    confidence=Confidence(js.get("confidence") or "high"),
                    provenance=self.provenance
                )
                nodes.append(node)

                # Update edges with JS membership
                for edge_id in js.get("member_edge_ids", []):
                    if edge_id in edge_by_id:
                        edge_by_id[edge_id].support_group_ids.append(js["id"])

            except (KeyError, ValueError) as e:
                logger.warning(f"Could not create JustificationSetNode: {e}")

        return nodes


# =============================================================================
# v4: DETERMINISTIC STRUCTURE (cross-cluster edges, JS, chains)
# =============================================================================


def dedupe_edges(edges: List[Edge]) -> List[Edge]:
    """Dedupe edges by (source, target, relation), keeping the highest-confidence instance."""
    if not edges:
        return []

    rank = {
        Confidence.HIGH: 4,
        Confidence.MEDIUM: 3,
        Confidence.LOW: 2,
        Confidence.INFERRED: 1,
    }

    best_by_sig: Dict[Tuple[str, str, EdgeRelation], Edge] = {}
    for e in edges:
        sig = (e.source, e.target, e.relation)
        existing = best_by_sig.get(sig)
        if not existing:
            best_by_sig[sig] = e
            continue
        if rank.get(e.confidence, 0) > rank.get(existing.confidence, 0):
            best_by_sig[sig] = e

    # Preserve deterministic order: sort by source/target/relation string
    return sorted(best_by_sig.values(), key=lambda x: (x.source, x.target, x.relation.value, x.id))


def extract_cross_cluster_edges(graph: LegalReasoningGraph) -> List[Edge]:
    """Create structurally constrained edges that don't require LLM reasoning."""

    edges: List[Edge] = []

    # 1) Holding → Issue (resolves)
    for h in graph.holdings:
        if not h.resolves_issue:
            continue
        edges.append(Edge(
            id=f"e_{h.id}_resolves_{h.resolves_issue}",
            source=h.id,
            target=h.resolves_issue,
            relation=EdgeRelation.RESOLVES,
            anchor=h.anchor,
            confidence=Confidence.HIGH if h.anchor else Confidence.INFERRED,
            explanation=None if h.anchor else "Inferred from holding.resolves_issue field",
            strength="strong",
            is_critical=False,
            provenance=None
        ))

    # 2) Holding → Outcome (determines / contributes_to)
    if graph.outcome:
        outcome_text = " ".join(filter(None, [
            graph.outcome.relief_summary,
            graph.outcome.anchor.surface_text if graph.outcome.anchor else None
        ]))
        outcome_kw = _keyword_set(outcome_text)

        # Score each ratio holding for overlap with outcome relief
        scored: List[Tuple[int, HoldingNode]] = []
        for h in graph.holdings:
            if not getattr(h, "is_ratio", False):
                continue
            hold_kw = _keyword_set(h.text or "")
            overlap = len(hold_kw.intersection(outcome_kw))
            scored.append((overlap, h))

        # Ensure at least one determines edge exists when outcome exists
        if scored:
            best_overlap = max(scored, key=lambda t: t[0])[0]
            if best_overlap == 0:
                # Fall back to the latest (closest-to-disposition) ratio holding by anchor
                scored_sorted = sorted(
                    scored,
                    key=lambda t: (t[1].anchor.start_char if t[1].anchor else -1)
                )
                best_holding = scored_sorted[-1][1]
                scored = [(1 if h.id == best_holding.id else 0, h) for _, h in scored]

        for overlap, h in scored:
            if overlap >= 1:
                rel = EdgeRelation.DETERMINES
                critical = True
            else:
                rel = EdgeRelation.CONTRIBUTES_TO
                critical = False

            edges.append(Edge(
                id=f"e_{h.id}_{rel.value}_outcome",
                source=h.id,
                target=graph.outcome.id,
                relation=rel,
                anchor=graph.outcome.anchor or h.anchor,
                confidence=Confidence.HIGH if (graph.outcome.anchor or h.anchor) else Confidence.INFERRED,
                explanation=None if (graph.outcome.anchor or h.anchor) else "Inferred outcome linkage",
                strength="strong" if rel == EdgeRelation.DETERMINES else "moderate",
                is_critical=critical,
                provenance=None
            ))

    return edges


def build_justification_sets_v4(
        graph: LegalReasoningGraph,
        clusters: Dict[str, ConceptCluster]
) -> List[JustificationSetNode]:
    """Build justification sets deterministically, using cluster logic where available."""

    # holding_id -> best cluster
    holding_clusters: Dict[str, List[str]] = {}
    for cid, cl in clusters.items():
        for hid in cl.holdings:
            holding_clusters.setdefault(hid, []).append(cid)

    def _pick_cluster_for_holding(hid: str) -> Optional[str]:
        cids = holding_clusters.get(hid, [])
        if not cids:
            return None
        # Prefer ontology-backed clusters (has requires) and bigger clusters
        return max(
            cids,
            key=lambda cid: (
                1 if clusters[cid].requires else 0,
                len(clusters[cid].facts) + len(clusters[cid].concepts) + len(clusters[cid].precedents)
            )
        )

    js_list: List[JustificationSetNode] = []
    counter = 1

    for h in graph.holdings:
        # Support edges into this holding
        support_edges = [
            e for e in graph.edges
            if e.target == h.id and e.relation in {EdgeRelation.SUPPORTS, EdgeRelation.GROUNDS}
        ]
        if not support_edges:
            continue

        cid = _pick_cluster_for_holding(h.id)
        logic = "and"
        label = None
        if cid:
            logic = clusters[cid].logic
            label = f"{clusters[cid].concept_label} support"

        js = JustificationSetNode(
            id=f"js{counter}",
            target_id=h.id,
            logic=JustificationLogic.AND if logic == "and" else JustificationLogic.OR,
            label=label,
            is_primary=True,
            confidence=Confidence.HIGH,
            provenance=None
        )
        counter += 1

        for e in support_edges:
            if js.id not in e.support_group_ids:
                e.support_group_ids.append(js.id)

        js_list.append(js)

        # Optional defeater JS if we have undercut edges into the holding
        defeater_edges = [
            e for e in graph.edges
            if e.target == h.id and e.relation == EdgeRelation.UNDERCUTS
        ]
        if defeater_edges:
            defeater_js = JustificationSetNode(
                id=f"js{counter}",
                target_id=h.id,
                logic=JustificationLogic.OR,
                label=f"{label or h.id} defeaters",
                is_primary=False,
                confidence=Confidence.MEDIUM,
                provenance=None
            )
            counter += 1
            for e in defeater_edges:
                if defeater_js.id not in e.support_group_ids:
                    e.support_group_ids.append(defeater_js.id)
            js_list.append(defeater_js)

    return js_list


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def synthesize_reasoning_chains_v4(graph: LegalReasoningGraph) -> List[ReasoningChain]:
    """Create deterministic reasoning chains via graph traversal."""

    chains: List[ReasoningChain] = []

    # Quick index for primary JS per holding
    primary_js_for: Dict[str, str] = {}
    for js in graph.justification_sets:
        if js.is_primary and js.target_id and js.id:
            primary_js_for[js.target_id] = js.id

    edges_to: Dict[str, List[Edge]] = {}
    for e in graph.edges:
        edges_to.setdefault(e.target, []).append(e)

    for issue in graph.issues:
        resolving = [h for h in graph.holdings if h.resolves_issue == issue.id]
        for holding in resolving:
            chain = ReasoningChain(
                id=f"rc_{issue.id}_{holding.id}",
                issue_id=issue.id,
                holding_id=holding.id,
                fact_ids=[],
                concept_ids=[],
                argument_ids=[],
                edge_ids=[],
                justification_set_id=primary_js_for.get(holding.id),
                critical_nodes=[],
                narrative=None,
            )

            visited: Set[str] = set()
            queue: List[str] = [holding.id]

            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)

                for e in edges_to.get(nid, []):
                    # Skip structural edges when traversing support structure
                    if e.relation in {EdgeRelation.RESOLVES, EdgeRelation.DETERMINES, EdgeRelation.CONTRIBUTES_TO}:
                        continue

                    chain.edge_ids.append(e.id)
                    if e.is_critical:
                        chain.critical_nodes.append(e.source)

                    src = graph.get_node(e.source)
                    if isinstance(src, FactNode):
                        chain.fact_ids.append(src.id)
                    elif isinstance(src, ConceptNode):
                        chain.concept_ids.append(src.id)
                    elif isinstance(src, ArgumentNode):
                        chain.argument_ids.append(src.id)

                    if e.source not in visited:
                        queue.append(e.source)

            # De-dupe in stable order
            chain.fact_ids = _unique_preserve_order(chain.fact_ids)
            chain.concept_ids = _unique_preserve_order(chain.concept_ids)
            chain.argument_ids = _unique_preserve_order(chain.argument_ids)
            chain.edge_ids = _unique_preserve_order(chain.edge_ids)
            chain.critical_nodes = _unique_preserve_order(chain.critical_nodes)

            # Narrative (lightweight, deterministic)
            parts = [f"Issue {issue.id} resolved by holding {holding.id}."]
            if chain.fact_ids:
                parts.append(f"Facts: {', '.join(chain.fact_ids)}.")
            if chain.concept_ids:
                parts.append(f"Concepts: {', '.join(chain.concept_ids)}.")
            if chain.argument_ids:
                parts.append(f"Arguments: {', '.join(chain.argument_ids)}.")
            chain.narrative = " ".join(parts)

            chains.append(chain)

    return chains


def counterfactual_remove_node_v4(
        graph: LegalReasoningGraph,
        node_id: str,
        clusters: Dict[str, ConceptCluster]
) -> Dict[str, Any]:
    """Best-effort counterfactual analysis for removing ANY node (v4).

    This complements (does not replace) `LegalReasoningGraph.counterfactual_remove_concept`,
    which only supports concept-node removal.
    """

    results = {
        "removed_node": node_id,
        "affected_holdings": [],
        "unaffected_holdings": [],
        "broken_requirements": [],
        "outcome_affected": False,
    }

    # 1) Ontology requirement check
    for cid, cl in clusters.items():
        for req, satisfying_node in (cl.satisfied_requirements or {}).items():
            if satisfying_node == node_id:
                results["broken_requirements"].append({
                    "concept": cid,
                    "requirement": req,
                })

                if cl.logic == "and":
                    for hid in cl.holdings:
                        results["affected_holdings"].append({
                            "holding_id": hid,
                            "reason": f"Removed node satisfied required element '{req}' for {cl.concept_label}",
                            "concept": cid,
                        })

    affected_holding_ids = {a["holding_id"] for a in results["affected_holdings"]}

    # 2) Justification-set based dependency check
    for h in graph.holdings:
        if h.id in affected_holding_ids:
            continue

        support = graph.get_holding_support(h.id)
        js_list = support.get("justification_sets") or []

        if not js_list:
            # Fallback: if node is in the traced upstream support, mark affected
            if node_id in set(support.get("all_concepts", []) + support.get("all_facts", [])):
                results["affected_holdings"].append({
                    "holding_id": h.id,
                    "reason": "Node appears in support trace (no explicit justification sets)",
                })
                continue
            results["unaffected_holdings"].append(h.id)
            continue

        has_surviving_path = False
        for js in js_list:
            js_id = js.get("id")
            logic = js.get("logic")
            members = graph.get_justification_members(js_id) if js_id else []

            if node_id not in members:
                has_surviving_path = True
                continue

            if logic == "or":
                remaining = [m for m in members if m != node_id]
                if remaining:
                    has_surviving_path = True

        if not has_surviving_path:
            results["affected_holdings"].append({
                "holding_id": h.id,
                "reason": "All justification paths break when node is removed",
            })
        else:
            results["unaffected_holdings"].append(h.id)

    affected_holding_ids = {a["holding_id"] for a in results["affected_holdings"]}

    # 3) Outcome effect check (only via determines)
    results["outcome_affected"] = any(
        (e.source in affected_holding_ids and e.relation == EdgeRelation.DETERMINES)
        for e in graph.edges
    )

    return results


# =============================================================================
# MAIN EXTRACTOR
# =============================================================================

class LegalReasoningExtractor:
    """Main extraction orchestrator."""

    def __init__(self, client: LLMClient, config: ExtractionConfig = None):
        self.client = client
        self.config = config or ExtractionConfig()

    async def extract(
            self,
            text: str,
            case_id: str,
            case_name: Optional[str] = None,
            case_year: Optional[int] = None,
            court: Optional[str] = None,
            judges: Optional[List[str]] = None
    ) -> LegalReasoningGraph:
        """Extract a complete legal reasoning graph from judgment text."""

        logger.info(f"Starting extraction for case: {case_id}")

        # Segment document
        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        doc = segment_document(text, f"sha256:{doc_hash}")
        logger.info(f"Document segmented: {doc.para_count} paragraphs, {doc.sent_count} sentences")

        # Initialize graph
        graph = LegalReasoningGraph(
            case_id=case_id,
            case_name=case_name,
            case_year=case_year,
            court=court,
            judges=judges or [],
            extraction_model=self.config.model_id,
            extraction_timestamp=datetime.utcnow().isoformat()
        )

        context = {}
        all_warnings = []

        # Pass 1: Facts
        logger.info("Pass 1: Extracting facts...")
        facts_pass = FactsExtractionPass(self.client, self.config, doc)
        facts_result = await facts_pass.extract()
        if facts_result.success:
            graph.facts = facts_pass.to_nodes(facts_result.data)
            context["facts"] = facts_result.data.get("facts", [])
            logger.info(f"  → {len(graph.facts)} facts extracted")
        else:
            logger.warning(f"  ⚠ Facts extraction failed: {facts_result.errors}")
            all_warnings.extend(facts_result.errors)
        all_warnings.extend(facts_result.warnings)

        # Pass 2: Concepts
        logger.info("Pass 2: Extracting concepts...")
        concepts_pass = ConceptsExtractionPass(self.client, self.config, doc)
        concepts_result = await concepts_pass.extract(context)
        if concepts_result.success:
            graph.concepts = concepts_pass.to_nodes(concepts_result.data)
            context["concepts"] = concepts_result.data.get("concepts", [])
            logger.info(f"  → {len(graph.concepts)} concepts extracted")
        else:
            logger.warning(f"  ⚠ Concepts extraction failed: {concepts_result.errors}")
            all_warnings.extend(concepts_result.errors)
        all_warnings.extend(concepts_result.warnings)

        # Pass 3: Issues
        logger.info("Pass 3: Extracting issues...")
        issues_pass = IssuesExtractionPass(self.client, self.config, doc)
        issues_result = await issues_pass.extract(context)
        if issues_result.success:
            graph.issues = issues_pass.to_nodes(issues_result.data)
            context["issues"] = issues_result.data.get("issues", [])
            logger.info(f"  → {len(graph.issues)} issues extracted")
        else:
            logger.warning(f"  ⚠ Issues extraction failed: {issues_result.errors}")
            all_warnings.extend(issues_result.errors)
        all_warnings.extend(issues_result.warnings)

        # Pass 4: Arguments
        logger.info("Pass 4: Extracting arguments...")
        args_pass = ArgumentsExtractionPass(self.client, self.config, doc)
        args_result = await args_pass.extract(context)
        if args_result.success:
            graph.arguments = args_pass.to_nodes(args_result.data)
            context["arguments"] = args_result.data.get("arguments", [])
            logger.info(f"  → {len(graph.arguments)} arguments extracted")
        else:
            logger.warning(f"  ⚠ Arguments extraction failed: {args_result.errors}")
            all_warnings.extend(args_result.errors)
        all_warnings.extend(args_result.warnings)

        # Pass 5: Holdings
        logger.info("Pass 5: Extracting holdings...")
        holdings_pass = HoldingsExtractionPass(self.client, self.config, doc)
        holdings_result = await holdings_pass.extract(context)
        if holdings_result.success:
            graph.holdings = holdings_pass.to_nodes(holdings_result.data)
            context["holdings"] = holdings_result.data.get("holdings", [])
            logger.info(f"  → {len(graph.holdings)} holdings extracted")
        else:
            logger.warning(f"  ⚠ Holdings extraction failed: {holdings_result.errors}")
            all_warnings.extend(holdings_result.errors)
        all_warnings.extend(holdings_result.warnings)

        # Pass 6: Precedents
        logger.info("Pass 6: Extracting precedents...")
        prec_pass = PrecedentsExtractionPass(self.client, self.config, doc)
        prec_result = await prec_pass.extract()
        if prec_result.success:
            graph.precedents = prec_pass.to_nodes(prec_result.data)
            context["precedents"] = prec_result.data.get("precedents", [])
            logger.info(f"  → {len(graph.precedents)} precedents extracted")
        else:
            logger.warning(f"  ⚠ Precedents extraction failed: {prec_result.errors}")
            all_warnings.extend(prec_result.errors)
        all_warnings.extend(prec_result.warnings)

        # Pass 7: Outcome
        logger.info("Pass 7: Extracting outcome...")
        outcome_pass = OutcomeExtractionPass(self.client, self.config, doc)
        outcome_result = await outcome_pass.extract()
        if outcome_result.success:
            graph.outcome = outcome_pass.to_node(outcome_result.data)
            context["outcome"] = outcome_result.data.get("outcome")
            logger.info(f"  → Outcome: {graph.outcome.disposition.value if graph.outcome else 'None'}")
        else:
            logger.warning(f"  ⚠ Outcome extraction failed: {outcome_result.errors}")
            all_warnings.extend(outcome_result.errors)
        all_warnings.extend(outcome_result.warnings)

        # ---------------------------------------------------------------------
        # Phase C/D: Edges, Justification Sets, Reasoning Chains
        # ---------------------------------------------------------------------

        if self.config.pipeline_version == "v3":
            # Pass 8: Global Edges (v3)
            logger.info("Pass 8: Extracting edges (v3 global)...")
            edges_pass = EdgesExtractionPass(self.client, self.config, doc)
            edges_result = await edges_pass.extract(context)
            if edges_result.success:
                graph.edges = edges_pass.to_edges(edges_result.data)
                context["edges"] = [e.to_dict() for e in graph.edges]
                logger.info(f"  → {len(graph.edges)} edges extracted")
            else:
                logger.warning(f"  ⚠ Edges extraction failed: {edges_result.errors}")
                all_warnings.extend(edges_result.errors)
            all_warnings.extend(edges_result.warnings)

            # Pass 9: Link Discovery (v3) - optional
            if self.config.enable_link_discovery:
                logger.info("Pass 9: Discovering long-distance links (v3)...")
                link_pass = LinkDiscoveryPass(self.client, self.config, doc)
                link_result = await link_pass.extract(context)
                if link_result.success or link_result.data:
                    existing_ids = {e.id for e in graph.edges}
                    # Build node anchor map for secondary span attachment
                    node_anchors = {}
                    for n in graph.facts: node_anchors[n.id] = n.anchor
                    for n in graph.concepts: node_anchors[n.id] = n.anchor
                    for n in graph.issues: node_anchors[n.id] = n.anchor
                    for n in graph.arguments: node_anchors[n.id] = n.anchor
                    for n in graph.holdings: node_anchors[n.id] = n.anchor
                    for n in graph.precedents: node_anchors[n.id] = n.anchor
                    if graph.outcome: node_anchors[graph.outcome.id] = graph.outcome.anchor

                    discovered = link_pass.to_edges(link_result.data or {}, existing_ids, node_anchors=node_anchors)
                    graph.edges.extend(discovered)
                    logger.info(f"  → {len(discovered)} additional links discovered")
                all_warnings.extend(link_result.warnings)

            # Pass 10: Justification Sets (v3) - optional
            if self.config.enable_llm_justification_sets:
                logger.info("Pass 10: Extracting justification sets (v3)...")
                js_context = {
                    "holdings": context.get("holdings", []),
                    "edges": [e.to_dict() for e in graph.edges]
                }
                js_pass = JustificationSetsExtractionPass(self.client, self.config, doc)
                js_result = await js_pass.extract(js_context)
                if js_result.success or js_result.data:
                    graph.justification_sets = js_pass.to_nodes(js_result.data or {}, graph.edges)
                    logger.info(f"  → {len(graph.justification_sets)} justification sets extracted")
                all_warnings.extend(js_result.warnings)

        else:
            # =============================
            # v4: Ontology-driven clustering
            # =============================

            # Pass 7.5: Clustering
            logger.info("Pass 7.5: Clustering nodes by ontology concept...")
            ontology = self.config.load_ontology()
            clusters, node_membership = cluster_nodes(graph, ontology)
            logger.info(f"  → {len(clusters)} non-empty clusters")

            # Fix 8: Store cluster membership for debugging
            graph.cluster_membership = node_membership
            # Build cluster summary: concept_id -> {facts: [...], concepts: [...], ...}
            graph.cluster_summary = {}
            for concept_id, cluster in clusters.items():
                # Only include non-empty clusters
                if any([cluster.facts, cluster.concepts, cluster.issues,
                        cluster.arguments, cluster.holdings, cluster.precedents]):
                    graph.cluster_summary[concept_id] = {
                        "label": cluster.concept_label,
                        "logic": cluster.logic,
                        "facts": cluster.facts,
                        "concepts": cluster.concepts,
                        "issues": cluster.issues,
                        "arguments": cluster.arguments,
                        "holdings": cluster.holdings,
                        "precedents": cluster.precedents,
                    }

            # Pass 8: Intra-cluster edges (LLM-local)
            logger.info("Pass 8: Extracting intra-cluster edges (v4)...")
            intra_edges: List[Edge] = []
            concept_defs = (ontology or {}).get("concepts", {}) or {}

            # Only run the LLM on clusters that contain at least one holding or issue.
            for concept_id, cluster in clusters.items():
                node_count = sum(len(x) for x in [
                    cluster.facts, cluster.concepts, cluster.issues,
                    cluster.arguments, cluster.precedents, cluster.holdings
                ])
                if node_count < 2:
                    continue
                if not (cluster.holdings or cluster.issues):
                    continue

                ont_def = concept_defs.get(concept_id)
                intra_pass = IntraClusterEdgesExtractionPass(
                    self.client, self.config, doc,
                    cluster=cluster,
                    graph=graph,
                    ontology_concept=ont_def
                )

                intra_result = await intra_pass.extract()
                if intra_result.success:
                    # Unique, deterministic edge IDs per cluster
                    tag = hashlib.sha1(concept_id.encode("utf-8")).hexdigest()[:8]
                    prefix = f"e_{tag}_"
                    intra_edges.extend(intra_pass.to_edges(intra_result.data, edge_id_prefix=prefix))
                else:
                    logger.warning(f"  ⚠ Intra-cluster edges failed for {concept_id}: {intra_result.errors}")
                    all_warnings.extend(intra_result.errors)
                all_warnings.extend(intra_result.warnings)

            graph.edges = intra_edges
            logger.info(f"  → {len(graph.edges)} intra-cluster edges extracted")

            # Pass 8.5: Cross-cluster edges (deterministic)
            logger.info("Pass 8.5: Adding cross-cluster structural edges...")
            cross_edges = extract_cross_cluster_edges(graph)
            graph.edges = dedupe_edges(graph.edges + cross_edges)
            logger.info(f"  → {len(cross_edges)} cross-cluster edges added ({len(graph.edges)} total)")

            # Pass 9: Ontology-driven justification sets (deterministic)
            logger.info("Pass 9: Building justification sets (v4 deterministic)...")
            graph.justification_sets = build_justification_sets_v4(graph, clusters)
            logger.info(f"  → {len(graph.justification_sets)} justification sets built")

            # Pass 10: Deterministic reasoning chains
            logger.info("Pass 10: Synthesizing reasoning chains (v4)...")
            graph.reasoning_chains = synthesize_reasoning_chains_v4(graph)
            logger.info(f"  → {len(graph.reasoning_chains)} reasoning chains created")

        # Final validation
        logger.info("Running final validation...")
        validation_warnings = graph.validate()
        all_warnings.extend(validation_warnings)
        graph.validation_warnings = all_warnings

        # Determine quality tier
        error_patterns = [
            "error", "missing", "not found", "duplicate", "requires anchor",
            "doesn't match", "invalid", "failed", "exceeds"
        ]
        error_count = len([
            w for w in all_warnings
            if any(pattern in w.lower() for pattern in error_patterns)
        ])
        warning_count = len(all_warnings) - error_count

        if error_count == 0 and warning_count <= 3:
            graph.quality_tier = "gold"
        elif error_count <= 2 and warning_count <= 10:
            graph.quality_tier = "silver"
        elif error_count <= 5:
            graph.quality_tier = "bronze"
        else:
            graph.quality_tier = "reject"

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Extraction complete: {graph.quality_tier.upper()}")
        logger.info(f"  Facts: {len(graph.facts)}")
        logger.info(f"  Concepts: {len(graph.concepts)}")
        logger.info(f"  Issues: {len(graph.issues)}")
        logger.info(f"  Arguments: {len(graph.arguments)}")
        logger.info(f"  Holdings: {len(graph.holdings)}")
        logger.info(f"  Precedents: {len(graph.precedents)}")
        logger.info(f"  Edges: {len(graph.edges)}")
        logger.info(f"  Justification Sets: {len(graph.justification_sets)}")
        logger.info(f"  Warnings: {len(all_warnings)}")
        logger.info(f"{'=' * 60}")

        return graph


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """CLI entry point."""
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python extractor_v3.py <judgment.txt> [case_id] [--api anthropic|grok]")
        print("\nEnvironment variables:")
        print("  ANTHROPIC_API_KEY - for Anthropic Claude")
        print("  XAI_API_KEY - for X.AI Grok")
        return

    filepath = sys.argv[1]
    case_id = sys.argv[2] if len(sys.argv) > 2 else Path(filepath).stem

    # Determine API
    api = "anthropic"
    if "--api" in sys.argv:
        api_idx = sys.argv.index("--api")
        if api_idx + 1 < len(sys.argv):
            api = sys.argv[api_idx + 1]

    # Create client
    if api == "grok":
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            print("Error: XAI_API_KEY not set")
            return
        model_id = "grok-4-1-fast-reasoning"
        client = GrokClient(api_key, model_id=model_id)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set")
            return
        model_id = "claude-sonnet-4-20250514"
        client = AnthropicClient(api_key, model_id=model_id)

    # Load document
    with open(filepath, 'r') as f:
        text = f.read()

    # Extract
    config = ExtractionConfig(model_id=model_id)
    extractor = LegalReasoningExtractor(client, config)

    graph = await extractor.extract(
        text=text,
        case_id=case_id
    )

    # Save output
    output_path = filepath.replace(".txt", "_graph_v3.json")
    with open(output_path, 'w') as f:
        f.write(graph.to_json())

    print(f"\n✅ Graph saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())