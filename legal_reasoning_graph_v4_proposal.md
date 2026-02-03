# Legal Reasoning Graph Extractor v4: Ontology-Driven Clustering Architecture

## Proposal Document
**Date:** February 2, 2026  
**Author:** Ansuman Mullick + Claude  
**Status:** Design Proposal

---

## Executive Summary

This document proposes a fundamental architectural change to the Legal Reasoning Graph extraction system. Instead of fighting the LLM's locality bias with increasingly complex "link discovery" passes, we eliminate the concept of "distance" entirely by clustering nodes around their legal concepts before edge extraction.

**Core Insight:** The ontology isn't just for concept normalization—it's the clustering index.

**Result:** Three previously "open problems" in computational legal reasoning become tractable:
1. Long-distance linking → eliminated (no distance within clusters)
2. Counterfactual analysis → trivial (ontology encodes AND/OR structure)
3. Reasoning chain synthesis → deterministic (graph traversal, not LLM generation)

---

## Part 1: Problem Statement

### 1.1 The Current Architecture (v3)

```
Pass 1: Facts        → extract from linear document
Pass 2: Concepts     → extract from linear document
Pass 3: Issues       → extract from linear document
Pass 4: Arguments    → extract from linear document
Pass 5: Holdings     → extract from linear document
Pass 6: Precedents   → extract from linear document
Pass 7: Outcome      → extract from linear document
Pass 8: Edges        → LLM finds connections (locality-biased)
Pass 9: Link Discovery → LLM searches for "missed" far connections (band-aid)
Pass 10: Justification Sets → LLM groups support (often wrong AND/OR logic)
```

### 1.2 The Fundamental Problem: Locality Bias

LLMs have a strong locality bias. When asked to find relationships, they preferentially connect things that appear near each other in the context window. This is a feature for most tasks, but legal reasoning doesn't work that way:

```
Document Structure (Linear):
┌─────────────────────────────────────────────────────────────────┐
│ Para 3: Factory operated without hearing [FACT f3]              │
│ ...                                                             │
│ Para 16: Natural justice requires hearing [CONCEPT c2]          │
│ ...                                                             │
│ Para 22: Maneka Gandhi established due process [PRECEDENT p1]   │
│ ...                                                             │
│ Para 31: Closure order violated NJ [HOLDING h2]                 │
└─────────────────────────────────────────────────────────────────┘

Legal Reasoning Structure (Non-Linear):
     f3 ──────────────────────┐
                              │
     c2 ──────────────────────┼──► h2 ──► outcome
                              │
     p1 ──────────────────────┘
```

The fact (f3), concept (c2), precedent (p1), and holding (h2) are legally tightly coupled but documentally scattered across ~3000 characters.

### 1.3 Why Pass 9 (Link Discovery) is a Band-Aid

The current approach:
1. Extract nodes (Passes 1-7)
2. Extract "obvious" edges (Pass 8) — misses far connections
3. Dump all nodes into context, ask LLM to find "missing links" (Pass 9)

Problems:
- **Combinatorial explosion:** 15 facts × 8 concepts × 5 holdings = hundreds of candidate edges
- **Context noise:** By the time you list all nodes, the LLM pattern-matches rather than reasons
- **No structural guidance:** Which far connections are legally meaningful vs coincidental?
- **Inconsistent results:** Different runs find different "missing" links

---

## Part 2: The Key Insight

### 2.1 The Question That Changes Everything

> "Why does it have to be far? Why can't we cluster similar things together and attach a dimensional score so there is no far-reaching data anymore?"

This reframes the problem. We're not asking "how do we find far links better?" We're asking "why does 'far' exist at all?"

### 2.2 The Answer: Cluster by Concept

The document is linear. Legal reasoning isn't. We're fighting the LLM's locality bias when we should eliminate the locality problem entirely.

**Before (document-order):**
```
[f1 para 3] ----2000 chars---- [c2 para 16] ----1500 chars---- [h2 para 31]
```

**After (concept-clustered):**
```
Cluster: DOCTRINE_NATURAL_JUSTICE
├── f3  (no hearing given)
├── c2  (audi alteram partem)
├── p1  (Maneka Gandhi)
├── a1  (appellant's NJ argument)
├── h2  (violation finding)
└── All edges are LOCAL now
```

### 2.3 The Ontology IS the Clustering Index

The compiled ontology already contains everything needed:

```json
"DOCTRINE_NATURAL_JUSTICE_AUDI_ALTERAM_PARTEM": {
  "label": "Natural Justice - Audi Alteram Partem",
  "requires": [
    "[AND]",
    "Notice of proposed action",
    "Reasonable opportunity to be heard",
    "Fair hearing before adverse decision",
    "Speaking order/reasoned decision"
  ],
  "defeaters": [
    "Urgency/emergency",
    "Statutory exclusion",
    "Purely administrative actions"
  ],
  "key_phrases": "\"no one shall be condemned unheard\", \"fair opportunity\"...",
  "typical_edge_pattern": "(Adverse_Action) --[requires]--> (Notice + Hearing) --[if omitted]--> (Order_Void)"
}
```

This gives us:
- **Clustering keys:** `key_phrases` for matching nodes to concepts
- **AND/OR logic:** `[AND]` / `[OR]` markers in `requires`
- **Defeaters:** What breaks the doctrine (maps to `undercuts` edges)
- **Edge templates:** `typical_edge_pattern` guides extraction
- **Justification structure:** `requires` elements = JS members

---

## Part 3: Proposed Architecture (v4)

### 3.1 New Pipeline

```
Phase A: Extraction (unchanged)
├── Pass 1: Facts
├── Pass 2: Concepts
├── Pass 3: Issues
├── Pass 4: Arguments
├── Pass 5: Holdings
├── Pass 6: Precedents
└── Pass 7: Outcome

Phase B: Clustering (NEW)
└── Pass 7.5: Cluster nodes by concept_id
    ├── Match nodes to concepts via key_phrases
    ├── Group into concept-centric clusters
    └── Flag multi-concept nodes (appear in multiple clusters)

Phase C: Edge Extraction (restructured)
├── Pass 8: Intra-cluster edges (local, easy)
│   ├── Per cluster, use ontology's typical_edge_pattern
│   ├── Facts → triggers/satisfies concept requirements
│   ├── Precedents → supports/establishes doctrine
│   └── Defeaters → undercuts edges
│
└── Pass 8.5: Cross-cluster edges (small, constrained)
    ├── holding → issue (resolves)
    ├── holding → outcome (determines)
    └── concept → concept (requires/specializes) — from ontology hierarchy

Phase D: Justification & Chains (deterministic)
├── Pass 9: Justification Sets (ontology-driven)
│   ├── Logic from concept's requires field ([AND]/[OR])
│   ├── Members from intra-cluster edges
│   └── Defeaters create alternative/non-primary JS
│
└── Pass 10: Reasoning Chains (graph traversal)
    ├── For each issue → find resolving holding
    ├── For each holding → traverse upstream (facts, concepts, precedents)
    └── Output deterministic chain (no LLM needed)
```

### 3.2 Visual Comparison

**Current (v3):**
```
┌──────────────────────────────────────────────────────────┐
│                    LINEAR DOCUMENT                        │
│  f1...f2...f3...c1...c2...i1...i2...a1...h1...h2...p1... │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  LLM: "Find all edges" │  ← locality-biased
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ LLM: "Find far links"  │  ← band-aid pass
              └────────────────────────┘
```

**Proposed (v4):**
```
┌──────────────────────────────────────────────────────────┐
│                    LINEAR DOCUMENT                        │
│  f1...f2...f3...c1...c2...i1...i2...a1...h1...h2...p1... │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Cluster: ART_21 │  │ Cluster: NAT_J  │  │ Cluster: EPA    │
│ f1, c1, p2, p3  │  │ f3, c2, p1, a1  │  │ c4, a2, h3      │
│ i1, h1          │  │ i2, h2          │  │ i3              │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   Local edges          Local edges          Local edges
   (trivial)            (trivial)            (trivial)
         │                    │                    │
         └──────────┬─────────┴──────────┬─────────┘
                    │                    │
                    ▼                    ▼
           Cross-cluster edges    Structural edges
           (concept hierarchy)    (h→i, h→outcome)
```

---

## Part 4: Detailed Implementation

### 4.1 Pass 7.5: Clustering

```python
@dataclass
class ConceptCluster:
    concept_id: str
    concept_label: str
    logic: str  # "and" | "or" from ontology
    requires: List[str]  # from ontology
    defeaters: List[str]  # from ontology
    
    # Grouped nodes
    facts: List[str]  # node IDs
    concepts: List[str]
    precedents: List[str]
    arguments: List[str]
    issues: List[str]
    holdings: List[str]
    
    # Which requirements are satisfied
    satisfied_requirements: Dict[str, str]  # requirement → fact_id that satisfies it


def cluster_nodes(
    graph: LegalReasoningGraph,
    ontology: Dict
) -> Dict[str, ConceptCluster]:
    """
    Group all extracted nodes by their associated concept_id.
    
    Matching strategy:
    1. ConceptNodes: direct match via concept_id field
    2. Other nodes: match via key_phrases from ontology against node text
    3. Holdings: inherit concepts from the issues they resolve
    4. Arguments: inherit concepts from holdings they support
    """
    clusters = {}
    
    for concept_id, concept_def in ontology["concepts"].items():
        # Extract logic from requires field
        logic = "and"  # default
        requires = concept_def.get("requires", [])
        if requires and isinstance(requires[0], str):
            if "[OR" in requires[0].upper():
                logic = "or"
            requires = requires[1:]  # skip the logic marker
        
        clusters[concept_id] = ConceptCluster(
            concept_id=concept_id,
            concept_label=concept_def.get("label", concept_id),
            logic=logic,
            requires=requires,
            defeaters=concept_def.get("defeaters", []),
            facts=[], concepts=[], precedents=[],
            arguments=[], issues=[], holdings=[],
            satisfied_requirements={}
        )
    
    # Phase 1: Assign ConceptNodes (direct match)
    for c in graph.concepts:
        if c.concept_id in clusters:
            clusters[c.concept_id].concepts.append(c.id)
    
    # Phase 2: Assign other nodes via key_phrase matching
    for concept_id, concept_def in ontology["concepts"].items():
        key_phrases = parse_key_phrases(concept_def.get("key_phrases", ""))
        
        for f in graph.facts:
            if matches_any_phrase(f.text, key_phrases):
                clusters[concept_id].facts.append(f.id)
                # Check if this fact satisfies any requirement
                for req in clusters[concept_id].requires:
                    if satisfies_requirement(f.text, req):
                        clusters[concept_id].satisfied_requirements[req] = f.id
        
        for p in graph.precedents:
            if matches_any_phrase(p.case_name, key_phrases) or \
               matches_any_phrase(p.proposition, key_phrases):
                clusters[concept_id].precedents.append(p.id)
        
        for a in graph.arguments:
            if matches_any_phrase(a.text, key_phrases):
                clusters[concept_id].arguments.append(a.id)
    
    # Phase 3: Assign Issues to clusters of their primary_concepts
    for i in graph.issues:
        for pc in i.primary_concepts:
            if pc in clusters:
                clusters[pc].issues.append(i.id)
    
    # Phase 4: Assign Holdings to clusters of issues they resolve
    for h in graph.holdings:
        if h.resolves_issue:
            issue = graph.get_node(h.resolves_issue)
            if issue:
                for pc in issue.primary_concepts:
                    if pc in clusters:
                        clusters[pc].holdings.append(h.id)
    
    # Remove empty clusters
    return {k: v for k, v in clusters.items() 
            if v.facts or v.concepts or v.holdings or v.precedents}
```

### 4.2 Pass 8: Intra-Cluster Edge Extraction

```python
async def extract_intra_cluster_edges(
    cluster: ConceptCluster,
    graph: LegalReasoningGraph,
    ontology_concept: Dict,
    client: LLMClient
) -> List[Edge]:
    """
    Extract edges WITHIN a single cluster.
    
    This is now a LOCAL operation - all nodes are semantically related.
    The LLM's locality bias becomes a feature, not a bug.
    """
    
    # Gather node texts for this cluster only
    cluster_context = build_cluster_context(cluster, graph)
    
    # Get edge template from ontology
    edge_template = ontology_concept.get("typical_edge_pattern", "")
    
    prompt = f"""
You are extracting reasoning edges within a single legal concept cluster.

CONCEPT: {cluster.concept_label}
TYPICAL PATTERN: {edge_template}

REQUIREMENTS FOR THIS CONCEPT:
{format_requirements(cluster.requires, cluster.logic)}

DEFEATERS (what would break this):
{format_list(cluster.defeaters)}

NODES IN THIS CLUSTER:
{cluster_context}

Extract edges between these nodes. All nodes are related to {cluster.concept_label}.
Focus on:
1. Which facts TRIGGER or SATISFY the concept requirements
2. Which precedents SUPPORT or ESTABLISH the doctrine  
3. Which arguments ADDRESS the related issues
4. Which holdings RESOLVE the related issues
5. Any DEFEATER relationships (fact/argument that UNDERCUTS the concept)

Output JSON:
{{
  "edges": [
    {{
      "source": "node_id",
      "target": "node_id", 
      "relation": "triggers|supports|grounds|undercuts|resolves|...",
      "satisfies_requirement": "requirement text if applicable",
      "is_defeater": true/false,
      "explanation": "why this connection exists"
    }}
  ]
}}
"""
    
    response = await client.complete(prompt, system=EDGE_SYSTEM_PROMPT)
    return parse_edges(response, cluster.concept_id)
```

### 4.3 Pass 8.5: Cross-Cluster Edges

```python
def extract_cross_cluster_edges(
    clusters: Dict[str, ConceptCluster],
    graph: LegalReasoningGraph,
    ontology: Dict
) -> List[Edge]:
    """
    Extract edges BETWEEN clusters.
    
    These are structurally constrained - we know what types to look for:
    1. holding → issue (resolves)
    2. holding → outcome (determines)
    3. concept → concept (from ontology hierarchy)
    """
    edges = []
    
    # 1. Holdings → Issues (resolves)
    for h in graph.holdings:
        if h.resolves_issue:
            edges.append(Edge(
                id=f"e_h{h.id}_resolves_{h.resolves_issue}",
                source=h.id,
                target=h.resolves_issue,
                relation=EdgeRelation.RESOLVES,
                confidence=Confidence.HIGH
            ))
    
    # 2. Holdings → Outcome (determines)
    if graph.outcome:
        for h in graph.holdings:
            if h.is_ratio:  # Only ratio holdings determine outcome
                edges.append(Edge(
                    id=f"e_{h.id}_determines_outcome",
                    source=h.id,
                    target=graph.outcome.id,
                    relation=EdgeRelation.DETERMINES,
                    confidence=Confidence.HIGH,
                    is_critical=True
                ))
    
    # 3. Concept → Concept (from ontology parent/requires relationships)
    for concept_id, concept_def in ontology["concepts"].items():
        parent = concept_def.get("parent")
        if parent and parent in ontology["concepts"]:
            # Find the concept nodes
            source_nodes = [c for c in graph.concepts if c.concept_id == concept_id]
            target_nodes = [c for c in graph.concepts if c.concept_id == parent]
            for s in source_nodes:
                for t in target_nodes:
                    edges.append(Edge(
                        id=f"e_{s.id}_specializes_{t.id}",
                        source=s.id,
                        target=t.id,
                        relation=EdgeRelation.SPECIALIZES,
                        confidence=Confidence.HIGH
                    ))
    
    return edges
```

### 4.4 Pass 9: Ontology-Driven Justification Sets

```python
def build_justification_sets(
    clusters: Dict[str, ConceptCluster],
    graph: LegalReasoningGraph
) -> List[JustificationSetNode]:
    """
    Build justification sets using ontology's AND/OR structure.
    
    This is now DETERMINISTIC - no LLM needed.
    The ontology tells us exactly what the logic should be.
    """
    js_list = []
    js_counter = 1
    
    for concept_id, cluster in clusters.items():
        if not cluster.holdings:
            continue
            
        for holding_id in cluster.holdings:
            # Get all supporting edges into this holding
            support_edges = [
                e for e in graph.edges 
                if e.target == holding_id and 
                   e.relation in [EdgeRelation.SUPPORTS, EdgeRelation.GROUNDS]
            ]
            
            if not support_edges:
                continue
            
            # PRIMARY JS: Uses ontology's logic
            primary_js = JustificationSetNode(
                id=f"js{js_counter}",
                target_id=holding_id,
                logic=JustificationLogic.AND if cluster.logic == "and" else JustificationLogic.OR,
                is_primary=True,
                explanation=f"Based on {cluster.concept_label} requirements"
            )
            
            # Assign edges to this JS
            for edge in support_edges:
                edge.support_group_ids.append(primary_js.id)
            
            js_list.append(primary_js)
            js_counter += 1
            
            # DEFEATER JS: If any defeater edges exist, create non-primary OR set
            defeater_edges = [
                e for e in graph.edges
                if e.target == holding_id and 
                   e.relation == EdgeRelation.UNDERCUTS
            ]
            
            if defeater_edges:
                defeater_js = JustificationSetNode(
                    id=f"js{js_counter}",
                    target_id=holding_id,
                    logic=JustificationLogic.OR,  # Any defeater breaks it
                    is_primary=False,
                    explanation=f"Defeaters for {cluster.concept_label}"
                )
                for edge in defeater_edges:
                    edge.support_group_ids.append(defeater_js.id)
                js_list.append(defeater_js)
                js_counter += 1
    
    return js_list
```

### 4.5 Pass 10: Deterministic Reasoning Chains

```python
def synthesize_reasoning_chains(
    graph: LegalReasoningGraph,
    clusters: Dict[str, ConceptCluster]
) -> List[ReasoningChain]:
    """
    Build reasoning chains via graph traversal.
    
    This is DETERMINISTIC - no LLM needed.
    We traverse upstream from each holding to collect the support structure.
    """
    chains = []
    
    for issue in graph.issues:
        # Find holdings that resolve this issue
        resolving_holdings = [
            h for h in graph.holdings 
            if h.resolves_issue == issue.id
        ]
        
        for holding in resolving_holdings:
            chain = ReasoningChain(
                id=f"rc_{issue.id}_{holding.id}",
                issue_id=issue.id,
                holding_id=holding.id,
                fact_ids=[],
                concept_ids=[],
                precedent_ids=[],
                argument_ids=[],
                edge_ids=[],
                critical_nodes=[]
            )
            
            # BFS upstream from holding
            visited = set()
            queue = [holding.id]
            
            while queue:
                node_id = queue.pop(0)
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                # Get all edges pointing TO this node
                incoming = graph.get_edges_to(node_id)
                
                for edge in incoming:
                    chain.edge_ids.append(edge.id)
                    
                    if edge.is_critical:
                        chain.critical_nodes.append(edge.source)
                    
                    # Categorize source node
                    source = graph.get_node(edge.source)
                    if isinstance(source, FactNode):
                        chain.fact_ids.append(edge.source)
                    elif isinstance(source, ConceptNode):
                        chain.concept_ids.append(edge.source)
                    elif isinstance(source, PrecedentNode):
                        chain.precedent_ids.append(edge.source)
                    elif isinstance(source, ArgumentNode):
                        chain.argument_ids.append(edge.source)
                    
                    # Continue traversal
                    queue.append(edge.source)
            
            chains.append(chain)
    
    return chains
```

### 4.6 Counterfactual Analysis (Enhanced)

```python
def counterfactual_remove_node(
    graph: LegalReasoningGraph,
    node_id: str,
    clusters: Dict[str, ConceptCluster]
) -> Dict:
    """
    Enhanced counterfactual analysis using ontology structure.
    
    Key insight: We check against the ontology's requirements,
    not just direct edge membership.
    """
    results = {
        "removed_node": node_id,
        "affected_holdings": [],
        "unaffected_holdings": [],
        "broken_requirements": [],
        "outcome_affected": False
    }
    
    removed_node = graph.get_node(node_id)
    
    for concept_id, cluster in clusters.items():
        # Check if removed node was satisfying any requirement
        for req, satisfying_node in cluster.satisfied_requirements.items():
            if satisfying_node == node_id:
                results["broken_requirements"].append({
                    "concept": concept_id,
                    "requirement": req
                })
                
                # If this concept uses AND logic, all holdings in cluster are affected
                if cluster.logic == "and":
                    for h_id in cluster.holdings:
                        results["affected_holdings"].append({
                            "holding_id": h_id,
                            "reason": f"Required element '{req}' for {cluster.concept_label} is now unsatisfied",
                            "concept": concept_id
                        })
    
    # Check direct edge dependencies (existing logic)
    for h in graph.holdings:
        if h.id in [a["holding_id"] for a in results["affected_holdings"]]:
            continue  # Already marked affected
            
        support = graph.get_holding_support(h.id)
        
        # Check justification sets
        has_surviving_path = False
        for js in support["justification_sets"]:
            members = graph.get_justification_members(js["id"])
            if node_id not in members:
                has_surviving_path = True
            elif js["logic"] == "or":
                remaining = [m for m in members if m != node_id]
                if remaining:
                    has_surviving_path = True
        
        if not has_surviving_path and support["justification_sets"]:
            results["affected_holdings"].append({
                "holding_id": h.id,
                "reason": "All justification paths broken"
            })
        else:
            results["unaffected_holdings"].append(h.id)
    
    # Check if outcome is affected
    affected_holding_ids = [a["holding_id"] for a in results["affected_holdings"]]
    results["outcome_affected"] = any(
        e.source in affected_holding_ids and e.relation == EdgeRelation.DETERMINES
        for e in graph.edges
    )
    
    return results
```

---

## Part 5: How This Solves the Open Problems

### 5.1 Long-Distance Linking → ELIMINATED

**Before:** Facts in paragraph 3 must somehow connect to holdings in paragraph 31.

**After:** Both are in the same cluster (DOCTRINE_NATURAL_JUSTICE). Edge extraction is local. The LLM sees:
```
Cluster: Natural Justice
- f3: "closure order passed without hearing"
- c2: "audi alteram partem doctrine"
- p1: "Maneka Gandhi precedent"
- h2: "order violated natural justice"

Find edges between these.
```

There is no "far" anymore. Everything legally related is contextually adjacent.

### 5.2 Counterfactual Analysis → TRIVIAL

**Before:** Check if removing a node breaks any direct edges to holdings. Miss multi-hop dependencies. AND/OR logic often wrong.

**After:** The ontology explicitly encodes:
- `requires: [AND]` → ALL elements must be satisfied
- `requires: [OR]` → ANY element suffices
- `defeaters` → what breaks the doctrine

Counterfactual becomes: "Does removing this node break any [AND] requirement in any cluster containing the holding?"

```python
# Pseudocode
if cluster.logic == "and":
    if removed_node satisfied any requirement:
        holding is BROKEN
elif cluster.logic == "or":
    if removed_node was the ONLY satisfying element:
        holding is BROKEN
```

### 5.3 Reasoning Chains → DETERMINISTIC

**Before:** Ask LLM to generate chains. Get inconsistent, often wrong results. `reasoning_chains: []` in output.

**After:** Pure graph traversal:
1. Start at Issue
2. Find Holdings that resolve it
3. BFS upstream: collect all Facts, Concepts, Precedents, Arguments
4. Mark critical nodes (on `is_critical` edges or sole [AND] members)
5. Output chain

No LLM needed. 100% reproducible. Chains reflect actual extracted graph structure.

---

## Part 6: Migration Path

### Phase 1: Integrate Compiled Ontology (Week 1)
- [x] Compile MD → JSON (done: `ontology_compiled_full.json`)
- [ ] Update `ExtractionConfig.load_ontology()` to use new format
- [ ] Verify concept extraction uses `key_phrases` for matching

### Phase 2: Implement Clustering Pass (Week 2)
- [ ] Add `ConceptCluster` dataclass
- [ ] Implement `cluster_nodes()` function
- [ ] Add Pass 7.5 to pipeline
- [ ] Unit tests for clustering

### Phase 3: Restructure Edge Extraction (Week 3)
- [ ] Modify Pass 8 to iterate per-cluster
- [ ] Implement `extract_intra_cluster_edges()`
- [ ] Add Pass 8.5 for cross-cluster edges
- [ ] Remove or minimize Pass 9 (Link Discovery)

### Phase 4: Ontology-Driven JS & Chains (Week 4)
- [ ] Rewrite `build_justification_sets()` to use ontology logic
- [ ] Implement `synthesize_reasoning_chains()` as graph traversal
- [ ] Enhanced `counterfactual_remove_node()` with requirement checking

### Phase 5: Testing & Validation (Week 5)
- [ ] Regression tests on sample judgments
- [ ] Assertions:
  - At least 1 reasoning chain per issue
  - Removal of f3 breaks h2 in sample
  - AND/OR logic matches ontology
- [ ] Compare v3 vs v4 output quality

---

## Part 7: Summary of Conversation

### The Journey

1. **Starting Point:** Anshu shared his Legal Reasoning Graph extractor (v3) - a sophisticated 10-pass system for extracting structured reasoning from Indian Supreme Court judgments.

2. **The Problem:** Pass 9 (Link Discovery) exists because LLMs miss "far" connections - facts in early paragraphs that relate to holdings in later paragraphs. The current solution is to dump all nodes into context and ask the LLM to find missing links. This is a band-aid.

3. **The Question:** Anshu asked what seemed like a "stupid question" - why does distance exist at all? Why can't we cluster by concept so there's no far-reaching data?

4. **The Insight:** This reframes the problem. We're not asking "how do we find far links better?" We're asking "why does 'far' exist?" The answer: it doesn't have to. Cluster by legal concept, and everything legally related becomes locally adjacent.

5. **The Realization:** The ontology Anshu already has isn't just for concept normalization - it's the clustering index. It already encodes AND/OR logic, requirements, defeaters, and edge patterns.

6. **The Architecture:** A new pipeline where nodes are clustered by concept_id after extraction, edge extraction happens per-cluster (local, easy), and justification sets / reasoning chains become deterministic operations on the graph.

### Key Takeaways

| Problem | Old Approach | New Approach |
|---------|--------------|--------------|
| Long-distance linking | LLM searches for "missing" links | Eliminated - cluster by concept |
| AND/OR logic | LLM guesses | Ontology explicitly encodes |
| Counterfactuals | Check direct edges only | Check ontology requirements |
| Reasoning chains | LLM generates (often empty) | Deterministic graph traversal |
| Edge extraction | Fight locality bias | Leverage locality bias |

### The Meta-Lesson

Sometimes the best solution isn't to solve the problem better - it's to make the problem disappear. Anshu's "stupid question" was actually the smart question: instead of building a better bridge across the distance, ask why the distance exists at all.

---

## Appendix A: File Changes Required

```
extractor.py (or extractor_v4.py)
├── Add: ConceptCluster dataclass
├── Add: cluster_nodes() function
├── Add: extract_intra_cluster_edges() 
├── Add: extract_cross_cluster_edges()
├── Modify: build_justification_sets() - use ontology logic
├── Add: synthesize_reasoning_chains() - graph traversal
├── Modify: counterfactual_remove_node() - check requirements
└── Modify: LegalReasoningExtractor.extract() - new pass order

schema_v2_1.py
├── No changes required (v2.1.1 is sufficient)
└── Optional: Add cluster_id field to nodes for debugging

test_extraction.py
├── Add: test_clustering()
├── Add: test_intra_cluster_edges()
├── Add: test_reasoning_chains_exist()
├── Add: test_counterfactual_breaks_holding()
└── Add: test_and_or_logic_from_ontology()
```

---

## Appendix B: Sample Cluster Output

For the test judgment (XYZ Corporation v. Union of India):

```json
{
  "clusters": {
    "DOCTRINE_NATURAL_JUSTICE_AUDI_ALTERAM_PARTEM": {
      "concept_id": "DOCTRINE_NATURAL_JUSTICE_AUDI_ALTERAM_PARTEM",
      "concept_label": "Natural Justice - Audi Alteram Partem",
      "logic": "and",
      "requires": [
        "Notice of proposed action",
        "Reasonable opportunity to be heard",
        "Fair hearing before adverse decision",
        "Speaking order/reasoned decision"
      ],
      "facts": ["f3", "f4"],
      "concepts": ["c2"],
      "precedents": ["p1"],
      "arguments": ["a1"],
      "issues": ["i2"],
      "holdings": ["h2"],
      "satisfied_requirements": {
        "Notice of proposed action": "f4",
        "Reasonable opportunity to be heard": null,  // NOT SATISFIED - this is the violation!
        "Fair hearing before adverse decision": null
      }
    },
    "CONST_ART21_RIGHT_TO_LIVELIHOOD": {
      "concept_id": "CONST_ART21",
      "concept_label": "Article 21 - Right to Life/Livelihood",
      "logic": "and",
      "facts": ["f1"],
      "concepts": ["c1"],
      "precedents": ["p2", "p3"],
      "arguments": [],
      "issues": ["i1"],
      "holdings": ["h1"]
    }
  }
}
```

Notice how the cluster structure immediately reveals the legal reasoning:
- f3 (no hearing) + unsatisfied "opportunity to be heard" requirement → violation → h2
- This is exactly what the judgment concludes, now encoded structurally.
