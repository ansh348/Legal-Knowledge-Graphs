#!/usr/bin/env python3
"""
schema_v2_1.py

Legal Reasoning Graph Schema v2.1

Changes from v2.0:
==================
1. SUPPORT GROUPS - Edges now have support_group_id for AND/OR justification sets
2. STABLE ANCHORS - Char offsets + text_hash instead of brittle "para:sent" strings
3. CANONICAL EDGES - Removed redundant lists from ArgumentNode (Toulmin fields derived)
4. DIRECTION CONVENTION - Source is always the "active" party in relation
5. MULTI-LABEL SCHEMES - Arguments/Holdings can have multiple schemes
6. ANCHOR VALIDATION - HIGH/MEDIUM confidence requires anchor; INFERRED requires explanation
7. PROVENANCE TRACKING - extraction_method, model_id, run_id on nodes/edges
8. SCHEMA VERSIONING - _meta.schema_version for migrations

Design Philosophy:
==================
1. EVERYTHING IS A NODE - Facts, concepts, arguments, holdings are all nodes
2. REASONING IS IN THE EDGES - The "why" lives in connections, not just components
3. ANCHORS ARE MANDATORY - Every claim must be traceable to source text
4. CONFIDENCE IS EXPLICIT - We know what we know and what we're guessing
5. ACTORS MATTER - Who said what affects how it's weighted
6. TIME IS FIRST-CLASS - Legal reasoning is inherently temporal
7. SUPPORT GROUPS ENABLE COUNTERFACTUAL - Multiple justification paths per conclusion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Literal, Union, Tuple
from enum import Enum
import json
import hashlib

SCHEMA_VERSION = "2.1.2"


# =============================================================================
# ENUMS - Constrained vocabularies for consistency
# =============================================================================

class NodeType(str, Enum):
    """Types of nodes in the reasoning graph."""
    FACT = "fact"
    CONCEPT = "concept"
    ISSUE = "issue"
    ARGUMENT = "argument"
    HOLDING = "holding"
    PRECEDENT = "precedent"
    OUTCOME = "outcome"
    JUSTIFICATION_SET = "justification_set"


class ActorType(str, Enum):
    """Who made a claim or argument."""
    PETITIONER = "petitioner"
    RESPONDENT = "respondent"
    COURT = "court"
    LOWER_COURT = "lower_court"
    AMICUS = "amicus"
    THIRD_PARTY = "third_party"
    APPELLANT = "appellant"
    COMPLAINANT = "complainant"
    ACCUSED = "accused"
    PROSECUTION = "prosecution"


class ConceptKind(str, Enum):
    """Types of legal concepts."""
    STATUTE_ARTICLE = "statute_article"
    STATUTE_SECTION = "statute_section"
    ORDER_RULE = "order_rule"
    DOCTRINE = "doctrine"
    TEST = "test"
    STANDARD = "standard"
    RIGHT = "right"
    DEFINITION = "definition"


class FactType(str, Enum):
    """Categories of facts."""
    MATERIAL = "material"
    PROCEDURAL = "procedural"
    BACKGROUND = "background"
    DISPUTED = "disputed"
    ADMITTED = "admitted"
    JUDICIAL_NOTICE = "judicial_notice"


class ArgumentScheme(str, Enum):
    """Types of legal argument patterns."""
    RULE_APPLICATION = "rule_application"
    RULE_EXCEPTION = "rule_exception"
    PRECEDENT_FOLLOWING = "precedent_following"
    PRECEDENT_ANALOGY = "precedent_analogy"
    PRECEDENT_DISTINCTION = "precedent_distinction"
    TEXTUAL_INTERPRETATION = "textual"
    PURPOSIVE_INTERPRETATION = "purposive"
    HARMONIOUS_CONSTRUCTION = "harmonious"
    PROPORTIONALITY = "proportionality"
    BALANCING = "balancing"
    EVIDENCE_SUFFICIENCY = "evidence_sufficiency"
    EVIDENCE_CREDIBILITY = "evidence_credibility"
    PROCEDURAL_COMPLIANCE = "procedural"
    JURISDICTION = "jurisdiction"
    LIMITATION = "limitation"
    POLICY_CONSEQUENCE = "policy_consequence"
    PUBLIC_INTEREST = "public_interest"
    NATURAL_JUSTICE = "natural_justice"
    OTHER = "other"


class EdgeRelation(str, Enum):
    """Types of relations between nodes."""
    TRIGGERS = "triggers"
    NEGATES = "negates"
    PARTIALLY_SATISFIES = "partially_satisfies"
    SATISFIES = "satisfies"
    CLAIMS_SATISFIES = "claims_satisfies"
    SUPPORTS = "supports"
    REBUTS = "rebuts"
    UNDERCUTS = "undercuts"
    GROUNDS = "grounds"
    ESTABLISHES = "establishes"
    ENABLES = "enables"
    CONSTRAINS = "constrains"
    REQUIRES = "requires"
    EXCLUDES = "excludes"
    SPECIALIZES = "specializes"
    CONFLICTS_WITH = "conflicts_with"
    ADDRESSES = "addresses"
    CONCEDES = "concedes"
    ATTACKS = "attacks"
    SUPPORTS_ARG = "supports_arg"
    RESPONDS_TO = "responds_to"
    RESOLVES = "resolves"
    PARTIALLY_RESOLVES = "partially_resolves"
    DETERMINES = "determines"
    CONTRIBUTES_TO = "contributes_to"
    FOLLOWS = "follows"
    APPLIES = "applies"
    DISTINGUISHES = "distinguishes"
    OVERRULES = "overrules"
    DOUBTS = "doubts"
    EXPLAINS = "explains"
    MEMBER_OF = "member_of"


class JustificationLogic(str, Enum):
    """How elements in a justification set combine."""
    AND = "and"
    OR = "or"


class Confidence(str, Enum):
    """Confidence level for extracted information."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFERRED = "inferred"


class Relevance(str, Enum):
    """How central a concept/fact is to the decision."""
    CENTRAL = "central"
    SUPPORTING = "supporting"
    MENTIONED = "mentioned"
    OBITER = "obiter"


class PrecedentTreatment(str, Enum):
    """How a precedent is treated by the court."""
    FOLLOWED = "followed"
    APPLIED = "applied"
    DISTINGUISHED = "distinguished"
    OVERRULED = "overruled"
    DOUBTED = "doubted"
    EXPLAINED = "explained"
    CITED = "cited"


class Disposition(str, Enum):
    """Case outcome types."""
    ALLOWED = "allowed"
    DISMISSED = "dismissed"
    PARTLY_ALLOWED = "partly_allowed"
    REMANDED = "remanded"
    MODIFIED = "modified"
    SET_ASIDE = "set_aside"


class ExtractionMethod(str, Enum):
    """How this node/edge was extracted."""
    REGEX = "regex"
    LLM = "llm"
    RULE = "rule"
    INFERENCE = "inference"
    MANUAL = "manual"


# =============================================================================
# ANCHOR - Stable text spans
# =============================================================================

@dataclass
class Anchor:
    """Links extraction to source text with STABLE references."""
    doc_id: str
    start_char: int
    end_char: int
    text_hash: Optional[str] = None
    display_location: Optional[str] = None
    secondary_spans: List[Tuple[int, int]] = field(default_factory=list)
    surface_text: Optional[str] = None

    @staticmethod
    def compute_text_hash(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text_hash": self.text_hash,
            "display_location": self.display_location,
            # Keep this JSON-friendly and type-consistent (always a list, never null).
            # Tuples are converted to lists for stable downstream parsing.
            "secondary_spans": [list(span) for span in self.secondary_spans],
            "surface_text": self.surface_text
        }


# =============================================================================
# PROVENANCE
# =============================================================================

@dataclass
class Provenance:
    """Tracks extraction origin for debugging and evaluation."""
    extraction_method: ExtractionMethod
    model_id: Optional[str] = None
    prompt_id: Optional[str] = None
    run_id: Optional[str] = None
    temperature: Optional[float] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "extraction_method": self.extraction_method.value,
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "run_id": self.run_id,
            "temperature": self.temperature,
            "timestamp": self.timestamp
        }


# =============================================================================
# JUSTIFICATION SET
# =============================================================================

@dataclass
class JustificationSetNode:
    """A group of supporting elements that together justify a conclusion."""
    id: str
    target_id: str
    logic: JustificationLogic
    label: Optional[str] = None
    is_primary: bool = False
    confidence: Confidence = Confidence.HIGH
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.JUSTIFICATION_SET.value,
            "target_id": self.target_id,
            "logic": self.logic.value,
            "label": self.label,
            "is_primary": self.is_primary,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


# =============================================================================
# NODES
# =============================================================================

@dataclass
class FactNode:
    """A factual assertion in the case."""
    id: str
    text: str
    anchor: Anchor
    fact_type: FactType
    actor_source: Optional[ActorType] = None
    date: Optional[str] = None
    date_approximate: bool = False
    disputed_by: Optional[ActorType] = None
    court_finding: Optional[Literal["accepted", "rejected", "not_decided"]] = None
    confidence: Confidence = Confidence.HIGH
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.FACT.value,
            "text": self.text,
            "anchor": self.anchor.to_dict(),
            "fact_type": self.fact_type.value,
            "actor_source": self.actor_source.value if self.actor_source else None,
            "date": self.date,
            "date_approximate": self.date_approximate,
            "disputed_by": self.disputed_by.value if self.disputed_by else None,
            "court_finding": self.court_finding,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class ConceptNode:
    """A legal concept applied in the case."""
    id: str
    concept_id: str
    anchor: Anchor
    relevance: Relevance
    kind: Optional[ConceptKind] = None
    interpretation: Optional[str] = None
    interpretation_anchor: Optional[Anchor] = None
    unlisted_label: Optional[str] = None
    unlisted_description: Optional[str] = None
    confidence: Confidence = Confidence.HIGH
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.CONCEPT.value,
            "concept_id": self.concept_id,
            "anchor": self.anchor.to_dict(),
            "relevance": self.relevance.value,
            "kind": self.kind.value if self.kind else None,
            "interpretation": self.interpretation,
            "interpretation_anchor": self.interpretation_anchor.to_dict() if self.interpretation_anchor else None,
            "unlisted_label": self.unlisted_label,
            "unlisted_description": self.unlisted_description,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class IssueNode:
    """A legal question the court addresses."""
    id: str
    text: str
    anchor: Anchor
    issue_number: Optional[int] = None
    framed_by: ActorType = ActorType.COURT
    primary_concepts: List[str] = field(default_factory=list)
    answer: Optional[Literal["yes", "no", "partly", "not_decided"]] = None
    confidence: Confidence = Confidence.HIGH
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.ISSUE.value,
            "text": self.text,
            "anchor": self.anchor.to_dict(),
            "issue_number": self.issue_number,
            "framed_by": self.framed_by.value,
            "primary_concepts": self.primary_concepts,
            "answer": self.answer,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class ArgumentNode:
    """An argument made by a party or the court."""
    id: str
    claim: str
    anchor: Anchor
    actor: ActorType
    schemes: List[ArgumentScheme]
    qualifiers: Optional[str] = None
    court_response: Optional[Literal["accepted", "rejected", "partly_accepted", "not_addressed"]] = None
    court_response_anchor: Optional[Anchor] = None
    court_reasoning: Optional[str] = None
    confidence: Confidence = Confidence.HIGH
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.ARGUMENT.value,
            "claim": self.claim,
            "anchor": self.anchor.to_dict(),
            "actor": self.actor.value,
            "schemes": [s.value for s in self.schemes],
            "qualifiers": self.qualifiers,
            "court_response": self.court_response,
            "court_response_anchor": self.court_response_anchor.to_dict() if self.court_response_anchor else None,
            "court_reasoning": self.court_reasoning,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class HoldingNode:
    """A legal determination by the court."""
    id: str
    text: str
    anchor: Anchor
    resolves_issue: Optional[str] = None
    is_ratio: bool = True
    novel: bool = False
    reasoning_summary: Optional[str] = None
    schemes: List[ArgumentScheme] = field(default_factory=list)
    confidence: Confidence = Confidence.HIGH
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.HOLDING.value,
            "text": self.text,
            "anchor": self.anchor.to_dict(),
            "resolves_issue": self.resolves_issue,
            "is_ratio": self.is_ratio,
            "novel": self.novel,
            "reasoning_summary": self.reasoning_summary,
            "schemes": [s.value for s in self.schemes],
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class PrecedentNode:
    """A cited precedent case."""
    id: str
    citation: str
    anchor: Anchor
    case_name: Optional[str] = None
    case_year: Optional[int] = None
    cited_case_id: Optional[str] = None
    cited_proposition: Optional[str] = None
    cited_holding: Optional[str] = None
    treatment: Optional[PrecedentTreatment] = None
    relevance: Relevance = Relevance.SUPPORTING
    confidence: Confidence = Confidence.HIGH
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.PRECEDENT.value,
            "citation": self.citation,
            "anchor": self.anchor.to_dict(),
            "case_name": self.case_name,
            "case_year": self.case_year,
            "cited_case_id": self.cited_case_id,
            "cited_proposition": self.cited_proposition,
            "cited_holding": self.cited_holding,
            "treatment": self.treatment.value if self.treatment else None,
            "relevance": self.relevance.value,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class OutcomeNode:
    """The final outcome/disposition of the case."""
    disposition: Disposition
    anchor: Anchor
    binary: Literal["accepted", "rejected"]
    id: str = "outcome"
    relief_summary: Optional[str] = None
    costs: Optional[Literal["petitioner", "respondent", "none", "shared"]] = None
    directions: List[str] = field(default_factory=list)
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": NodeType.OUTCOME.value,
            "disposition": self.disposition.value,
            "anchor": self.anchor.to_dict(),
            "binary": self.binary,
            "relief_summary": self.relief_summary,
            "costs": self.costs,
            "directions": self.directions,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


# =============================================================================
# EDGES
# =============================================================================

@dataclass
class Edge:
    """A directed edge in the reasoning graph."""
    id: str
    source: str
    target: str
    relation: EdgeRelation
    anchor: Optional[Anchor] = None
    explanation: Optional[str] = None
    confidence: Confidence = Confidence.HIGH
    strength: Literal["strong", "moderate", "weak"] = "strong"
    support_group_ids: List[str] = field(default_factory=list)
    is_critical: bool = False
    provenance: Optional[Provenance] = None

    def validate(self) -> List[str]:
        """Validate edge according to v2.1 rules."""
        warnings = []
        if self.confidence in [Confidence.HIGH, Confidence.MEDIUM]:
            if self.anchor is None:
                warnings.append(f"Edge {self.id}: HIGH/MEDIUM confidence requires anchor")
        if self.confidence == Confidence.INFERRED:
            if not self.explanation:
                warnings.append(f"Edge {self.id}: INFERRED confidence requires explanation")
        return warnings

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "anchor": self.anchor.to_dict() if self.anchor else None,
            "explanation": self.explanation,
            "confidence": self.confidence.value,
            "strength": self.strength,
            "support_group_ids": self.support_group_ids if self.support_group_ids else [],
            "is_critical": self.is_critical,
            "provenance": self.provenance.to_dict() if self.provenance else None
        }


# =============================================================================
# REASONING CHAIN
# =============================================================================

@dataclass
class ReasoningChain:
    """A complete reasoning path from facts to outcome."""
    id: str
    issue_id: Optional[str] = None
    fact_ids: List[str] = field(default_factory=list)
    concept_ids: List[str] = field(default_factory=list)
    argument_ids: List[str] = field(default_factory=list)
    holding_id: Optional[str] = None
    edge_ids: List[str] = field(default_factory=list)
    justification_set_id: Optional[str] = None
    critical_nodes: List[str] = field(default_factory=list)
    narrative: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "issue_id": self.issue_id,
            "fact_ids": self.fact_ids,
            "concept_ids": self.concept_ids,
            "argument_ids": self.argument_ids,
            "holding_id": self.holding_id,
            "edge_ids": self.edge_ids,
            "justification_set_id": self.justification_set_id,
            "critical_nodes": self.critical_nodes,
            "narrative": self.narrative
        }


# =============================================================================
# THE COMPLETE GRAPH
# =============================================================================

@dataclass
class LegalReasoningGraph:
    """The complete legal reasoning graph for a case."""

    case_id: str
    case_name: Optional[str] = None
    case_year: Optional[int] = None
    court: Optional[str] = None
    judges: List[str] = field(default_factory=list)

    facts: List[FactNode] = field(default_factory=list)
    concepts: List[ConceptNode] = field(default_factory=list)
    issues: List[IssueNode] = field(default_factory=list)
    arguments: List[ArgumentNode] = field(default_factory=list)
    holdings: List[HoldingNode] = field(default_factory=list)
    precedents: List[PrecedentNode] = field(default_factory=list)
    justification_sets: List[JustificationSetNode] = field(default_factory=list)
    outcome: Optional[OutcomeNode] = None

    edges: List[Edge] = field(default_factory=list)
    reasoning_chains: List[ReasoningChain] = field(default_factory=list)

    quality_tier: Literal["gold", "silver", "bronze", "reject"] = "bronze"
    extraction_model: Optional[str] = None
    extraction_timestamp: Optional[str] = None
    retry_attempts: int = 0
    validation_warnings: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate the graph according to v2.1.1 rules."""
        warnings = []

        all_node_ids = set()
        for f in self.facts:
            all_node_ids.add(f.id)
        for c in self.concepts:
            all_node_ids.add(c.id)
        for i in self.issues:
            all_node_ids.add(i.id)
        for a in self.arguments:
            all_node_ids.add(a.id)
        for h in self.holdings:
            all_node_ids.add(h.id)
        for p in self.precedents:
            all_node_ids.add(p.id)
        for js in self.justification_sets:
            all_node_ids.add(js.id)
        if self.outcome:
            all_node_ids.add(self.outcome.id)

        js_ids = {js.id for js in self.justification_sets}
        js_targets = {js.id: js.target_id for js in self.justification_sets}

        for e in self.edges:
            if e.source not in all_node_ids:
                warnings.append(f"Edge {e.id}: source '{e.source}' not found")
            if e.target not in all_node_ids:
                warnings.append(f"Edge {e.id}: target '{e.target}' not found")

            for sg_id in e.support_group_ids:
                if sg_id not in js_ids:
                    warnings.append(f"Edge {e.id}: support_group_id '{sg_id}' not found")
                elif e.target != js_targets[sg_id]:
                    warnings.append(
                        f"Edge {e.id}: target '{e.target}' doesn't match "
                        f"justification set '{sg_id}' target '{js_targets[sg_id]}'"
                    )

            warnings.extend(e.validate())

        for js in self.justification_sets:
            if js.target_id not in all_node_ids:
                warnings.append(f"JustificationSet {js.id}: target '{js.target_id}' not found")
            members = self.get_justification_members(js.id)
            if not members:
                warnings.append(f"JustificationSet {js.id}: no edges belong to this set")

        seen_ids = set()
        for node_list in [self.facts, self.concepts, self.issues, self.arguments,
                          self.holdings, self.precedents, self.justification_sets]:
            for node in node_list:
                if node.id in seen_ids:
                    warnings.append(f"Duplicate node ID: '{node.id}'")
                seen_ids.add(node.id)

        edge_ids = set()
        for e in self.edges:
            if e.id in edge_ids:
                warnings.append(f"Duplicate edge ID: '{e.id}'")
            edge_ids.add(e.id)

        for node_list in [self.facts, self.concepts, self.issues, self.arguments,
                          self.holdings, self.precedents]:
            for node in node_list:
                if hasattr(node, 'anchor') and node.anchor:
                    a = node.anchor
                    if a.start_char < 0:
                        warnings.append(f"Node {node.id}: anchor start_char < 0")
                    if a.end_char <= a.start_char:
                        warnings.append(f"Node {node.id}: anchor end_char <= start_char")

        if self.outcome:
            has_determining = any(
                e.target == self.outcome.id and e.relation == EdgeRelation.DETERMINES
                for e in self.edges
            )
            if not has_determining:
                warnings.append("Outcome exists but no DETERMINES edge points to it")

        return warnings

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "case_year": self.case_year,
            "court": self.court,
            "judges": self.judges,
            "facts": [f.to_dict() for f in self.facts],
            "concepts": [c.to_dict() for c in self.concepts],
            "issues": [i.to_dict() for i in self.issues],
            "arguments": [a.to_dict() for a in self.arguments],
            "holdings": [h.to_dict() for h in self.holdings],
            "precedents": [p.to_dict() for p in self.precedents],
            "justification_sets": [js.to_dict() for js in self.justification_sets],
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "edges": [e.to_dict() for e in self.edges],
            "reasoning_chains": [rc.to_dict() for rc in self.reasoning_chains],
            "_meta": {
                "schema_version": SCHEMA_VERSION,
                "quality_tier": self.quality_tier,
                "extraction_model": self.extraction_model,
                "extraction_timestamp": self.extraction_timestamp,
                "retry_attempts": self.retry_attempts,
                "validation_warnings": self.validation_warnings
            }
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_node(self, node_id: str) -> Optional[Union[
        FactNode, ConceptNode, IssueNode, ArgumentNode, HoldingNode, PrecedentNode, JustificationSetNode, OutcomeNode]]:
        """Get any node by ID."""
        for f in self.facts:
            if f.id == node_id:
                return f
        for c in self.concepts:
            if c.id == node_id:
                return c
        for i in self.issues:
            if i.id == node_id:
                return i
        for a in self.arguments:
            if a.id == node_id:
                return a
        for h in self.holdings:
            if h.id == node_id:
                return h
        for p in self.precedents:
            if p.id == node_id:
                return p
        for js in self.justification_sets:
            if js.id == node_id:
                return js
        if self.outcome and self.outcome.id == node_id:
            return self.outcome
        return None

    def get_edges_from(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges if e.target == node_id]

    def get_concept_ids(self) -> Set[str]:
        return {c.concept_id for c in self.concepts}

    def get_central_concepts(self) -> Set[str]:
        return {c.concept_id for c in self.concepts if c.relevance == Relevance.CENTRAL}

    def get_justification_members(self, js_id: str) -> List[str]:
        return [e.source for e in self.edges if js_id in e.support_group_ids]

    def get_toulmin_structure(self, argument_id: str) -> Dict:
        grounds = []
        warrants = []
        backing = []
        rebuttals = []

        for e in self.edges:
            if e.target == argument_id:
                if e.relation == EdgeRelation.SUPPORTS:
                    source_node = self.get_node(e.source)
                    if isinstance(source_node, FactNode):
                        grounds.append(e.source)
                    elif isinstance(source_node, ConceptNode):
                        warrants.append(e.source)
                    elif isinstance(source_node, PrecedentNode):
                        backing.append(e.source)
                elif e.relation in [EdgeRelation.ATTACKS, EdgeRelation.REBUTS, EdgeRelation.UNDERCUTS]:
                    rebuttals.append(e.source)

        return {
            "grounds": grounds,
            "warrants": warrants,
            "backing": backing,
            "rebuttals": rebuttals
        }

    def get_holding_support(self, holding_id: str) -> Dict:
        concepts = []
        facts = []

        for e in self.edges:
            if e.target == holding_id:
                if e.relation == EdgeRelation.GROUNDS:
                    concepts.append(e.source)
                elif e.relation == EdgeRelation.SUPPORTS:
                    source = self.get_node(e.source)
                    if isinstance(source, FactNode):
                        facts.append(e.source)

        js_list = [
            {
                "id": js.id,
                "logic": js.logic.value,
                "members": self.get_justification_members(js.id),
                "is_primary": js.is_primary
            }
            for js in self.justification_sets
            if js.target_id == holding_id
        ]

        return {
            "all_concepts": concepts,
            "all_facts": facts,
            "justification_sets": js_list
        }

    def counterfactual_remove_concept(self, concept_node_id: str) -> Dict:
        affected_holdings = []
        unaffected_holdings = []

        for h in self.holdings:
            support = self.get_holding_support(h.id)

            if not support["justification_sets"]:
                if concept_node_id in support["all_concepts"]:
                    affected_holdings.append({
                        "holding_id": h.id,
                        "reason": "concept directly grounds holding (no justification sets defined)"
                    })
                else:
                    unaffected_holdings.append(h.id)
                continue

            surviving_paths = []
            broken_paths = []

            for js in support["justification_sets"]:
                if concept_node_id in js["members"]:
                    if js["logic"] == "and":
                        broken_paths.append(js["id"])
                    else:
                        remaining = [m for m in js["members"] if m != concept_node_id]
                        if remaining:
                            surviving_paths.append(js["id"])
                        else:
                            broken_paths.append(js["id"])
                else:
                    surviving_paths.append(js["id"])

            if surviving_paths:
                unaffected_holdings.append(h.id)
            else:
                affected_holdings.append({
                    "holding_id": h.id,
                    "broken_paths": broken_paths,
                    "reason": "all justification paths broken"
                })

        return {
            "removed_concept": concept_node_id,
            "affected_holdings": affected_holdings,
            "unaffected_holdings": unaffected_holdings,
            "outcome_affected": any(
                e.source in [h["holding_id"] for h in affected_holdings]
                and e.relation == EdgeRelation.DETERMINES
                for e in self.edges
            ) if affected_holdings else False
        }

    def trace_reasoning_path(self, start_id: str, end_id: str) -> List[str]:
        from collections import deque

        adj = {}
        for e in self.edges:
            if e.source not in adj:
                adj[e.source] = []
            adj[e.source].append(e.target)

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            node, path = queue.popleft()
            if node == end_id:
                return path
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []