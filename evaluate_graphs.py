#!/usr/bin/env python3
"""
Legal Reasoning Graph Evaluation Script

Evaluates 50 legal reasoning graph JSON files in iltur_graphs/ against quality criteria.
Outputs individual results and summary statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import Counter, defaultdict
import re

# User-specified STRICTER edge validation rules
VALID_EDGE_RELATIONS: Dict[Tuple[str, str], Set[str]] = {
    # Fact -> X
    ("fact", "argument"): {"grounds", "supports", "rebuts", "undercuts"},
    ("fact", "concept"): {"triggers", "negates", "partially_satisfies", "satisfies", "claims_satisfies"},
    ("fact", "holding"): {"supports", "grounds"},
    ("fact", "issue"): {"triggers", "supports", "addresses"},

    # Concept -> X
    ("concept", "holding"): {"grounds", "supports"},  # STRICTER: removed constrains, enables
    ("concept", "concept"): {"requires", "excludes", "specializes", "conflicts_with"},
    ("concept", "argument"): {"supports", "grounds", "rebuts", "undercuts"},
    ("concept", "issue"): {"requires", "addresses"},

    # Argument -> X
    ("argument", "holding"): {"supports", "contributes_to"},  # STRICTER: removed grounds, rebuts, undercuts
    ("argument", "issue"): {"addresses", "concedes"},
    ("argument", "argument"): {"attacks", "supports_arg", "responds_to"},
    ("argument", "concept"): {"supports", "grounds", "rebuts", "undercuts", "claims_satisfies"},

    # Holding -> X
    ("holding", "outcome"): {"determines", "contributes_to"},
    ("holding", "issue"): {"resolves", "partially_resolves", "addresses"},
    ("holding", "precedent"): {"follows", "applies", "distinguishes", "overrules", "doubts", "explains"},
    ("holding", "concept"): {"supports", "grounds", "constrains", "undercuts", "negates"},
    ("holding", "holding"): {"supports", "conflicts_with", "specializes", "constrains", "undercuts"},

    # Precedent -> X
    ("precedent", "argument"): {"supports"},  # STRICTER: supports only
    ("precedent", "concept"): {"supports", "grounds", "establishes"},
    ("precedent", "holding"): {"supports"},
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


def load_ontology(ontology_path: str) -> Set[str]:
    """Load valid concept IDs from ontology."""
    valid_concept_ids = set()
    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
        if 'concepts' in ontology:
            valid_concept_ids = set(ontology['concepts'].keys())
    except Exception as e:
        print(f"Warning: Could not load ontology: {e}")
    return valid_concept_ids


def evaluate_edge_validity(edges: List[Dict], verbose: bool = False) -> Tuple[float, List[Dict]]:
    """
    Check edge validity against user-specified rules.
    Returns (validity_rate, list of invalid edges with details).
    """
    if not edges:
        return 1.0, []

    invalid_edges = []
    valid_count = 0

    for edge in edges:
        source_id = edge.get('source', '')
        target_id = edge.get('target', '')
        relation = edge.get('relation', '')

        source_type = get_node_type_from_id(source_id)
        target_type = get_node_type_from_id(target_id)

        key = (source_type, target_type)

        if key in VALID_EDGE_RELATIONS:
            if relation in VALID_EDGE_RELATIONS[key]:
                valid_count += 1
            else:
                invalid_edges.append({
                    'edge_id': edge.get('id', 'unknown'),
                    'source': source_id,
                    'target': target_id,
                    'source_type': source_type,
                    'target_type': target_type,
                    'relation': relation,
                    'allowed': list(VALID_EDGE_RELATIONS[key]),
                    'pattern': f"{source_type[0]}->{target_type[0]}:{relation}"
                })
        else:
            # No valid relations defined for this source->target pair
            invalid_edges.append({
                'edge_id': edge.get('id', 'unknown'),
                'source': source_id,
                'target': target_id,
                'source_type': source_type,
                'target_type': target_type,
                'relation': relation,
                'allowed': [],
                'pattern': f"{source_type[0]}->{target_type[0]}:{relation}"
            })

    total = len(edges)
    return valid_count / total if total > 0 else 1.0, invalid_edges


def evaluate_connectivity(graph_data: Dict) -> Tuple[float, int, List[str]]:
    """
    Check which nodes appear in edges.
    Returns (connectivity_rate, orphan_count, orphan_node_ids).
    """
    # Collect all node IDs
    all_node_ids = set()

    for section in ['facts', 'concepts', 'issues', 'arguments', 'holdings', 'precedents']:
        nodes = graph_data.get(section, [])
        for node in nodes:
            if 'id' in node:
                all_node_ids.add(node['id'])

    # Add outcome if present
    if graph_data.get('outcome') and isinstance(graph_data['outcome'], dict):
        outcome_id = graph_data['outcome'].get('id', 'outcome')
        all_node_ids.add(outcome_id)

    # Collect nodes appearing in edges
    connected_nodes = set()
    edges = graph_data.get('edges', [])
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source:
            connected_nodes.add(source)
        if target:
            connected_nodes.add(target)

    # Also check holdings for resolves_issue references
    for holding in graph_data.get('holdings', []):
        resolves = holding.get('resolves_issue')
        if resolves:
            connected_nodes.add(holding['id'])
            connected_nodes.add(resolves)

    orphan_nodes = all_node_ids - connected_nodes
    total_nodes = len(all_node_ids)
    connected_count = len(all_node_ids & connected_nodes)

    connectivity_rate = connected_count / total_nodes if total_nodes > 0 else 1.0
    return connectivity_rate, len(orphan_nodes), list(orphan_nodes)


def evaluate_reasoning_completeness(graph_data: Dict) -> Tuple[float, float, List[str]]:
    """
    Check issue resolution and holding-outcome connectivity.
    Returns (issues_resolved_rate, holdings_connected_rate, unresolved_issues).
    """
    issues = graph_data.get('issues', [])
    holdings = graph_data.get('holdings', [])
    edges = graph_data.get('edges', [])

    # Check which issues are resolved by holdings
    issues_with_holdings = set()
    for holding in holdings:
        resolves = holding.get('resolves_issue')
        if resolves:
            issues_with_holdings.add(resolves)

    issue_ids = {i['id'] for i in issues if 'id' in i}
    resolved_issues = issue_ids & issues_with_holdings
    unresolved = list(issue_ids - issues_with_holdings)

    issues_resolved_rate = len(resolved_issues) / len(issue_ids) if issue_ids else 1.0

    # Check holdings connected to outcome
    holding_ids = {h['id'] for h in holdings if 'id' in h}
    holdings_to_outcome = set()
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in holding_ids and target == 'outcome':
            holdings_to_outcome.add(source)

    holdings_connected_rate = len(holdings_to_outcome) / len(holding_ids) if holding_ids else 1.0

    return issues_resolved_rate, holdings_connected_rate, unresolved


def evaluate_anchor_quality(graph_data: Dict) -> Tuple[int, List[str]]:
    """
    Check if high/medium confidence nodes have anchors.
    Returns (missing_anchor_count, nodes_missing_anchors).
    """
    missing_anchors = []

    for section in ['facts', 'concepts', 'issues', 'arguments', 'holdings', 'precedents']:
        nodes = graph_data.get(section, [])
        for node in nodes:
            confidence = node.get('confidence', 'low')
            if confidence in ('high', 'medium'):
                anchor = node.get('anchor')
                if not anchor or not isinstance(anchor, dict) or 'start_char' not in anchor:
                    missing_anchors.append(node.get('id', 'unknown'))

    return len(missing_anchors), missing_anchors


def evaluate_semantic_sanity(graph_data: Dict, valid_concept_ids: Set[str]) -> Dict:
    """
    Check for duplicates, actor distribution, concept validity.
    Returns dict with warnings and stats.
    """
    warnings = []

    # Check for duplicate text (first 100 chars normalized)
    seen_texts = defaultdict(list)
    for section in ['facts', 'concepts', 'issues', 'arguments', 'holdings']:
        nodes = graph_data.get(section, [])
        for node in nodes:
            text = node.get('text', '')
            normalized = re.sub(r'\s+', ' ', text[:100].lower().strip())
            if normalized:
                seen_texts[normalized].append((section, node.get('id', 'unknown')))

    duplicates = {k: v for k, v in seen_texts.items() if len(v) > 1}
    if duplicates:
        for text, nodes in duplicates.items():
            warnings.append(f"Duplicate text found: {nodes}")

    # Actor distribution from arguments
    actors = Counter()
    for arg in graph_data.get('arguments', []):
        actor = arg.get('actor')
        if actor:
            actors[actor] += 1

    # Validate concept_ids
    invalid_concepts = []
    for concept in graph_data.get('concepts', []):
        concept_id = concept.get('concept_id')
        if concept_id:
            # Allow UNLISTED_ prefix
            if not concept_id.startswith('UNLISTED_') and concept_id not in valid_concept_ids:
                invalid_concepts.append(concept_id)
                warnings.append(f"Invalid concept_id: {concept_id}")

    return {
        'warnings_count': len(warnings),
        'warnings': warnings[:10],  # Limit to first 10
        'duplicate_count': len(duplicates),
        'actor_distribution': dict(actors),
        'invalid_concept_count': len(invalid_concepts)
    }


def calculate_quality_score(
    edge_validity_rate: float,
    connectivity_rate: float,
    issues_resolved_rate: float,
    warnings_count: int,
    missing_anchors: int
) -> float:
    """Calculate quality score (1-10) based on metrics."""
    score = 10.0
    score -= (1 - edge_validity_rate) * 3      # max -3
    score -= (1 - connectivity_rate) * 2        # max -2
    score -= (1 - issues_resolved_rate) * 2     # max -2
    score -= min(warnings_count * 0.1, 2)       # max -2
    score -= min(missing_anchors * 0.1, 1)      # max -1
    return max(1.0, round(score, 2))


def evaluate_single_graph(file_path: str, valid_concept_ids: Set[str]) -> Dict:
    """Evaluate a single graph JSON file."""
    try:
        # Try UTF-8 first, then fallback to latin-1 for problematic files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                graph_data = json.load(f)
    except Exception as e:
        return {
            'case_id': Path(file_path).stem,
            'error': str(e),
            'quality_score': 1,
            'is_error': True
        }

    case_id = graph_data.get('case_id', Path(file_path).stem)

    # 1. Edge validity
    edges = graph_data.get('edges', [])
    edge_validity_rate, invalid_edges = evaluate_edge_validity(edges)

    # 2. Connectivity
    connectivity_rate, orphan_count, orphan_nodes = evaluate_connectivity(graph_data)

    # 3. Reasoning completeness
    issues_resolved_rate, holdings_connected_rate, unresolved_issues = evaluate_reasoning_completeness(graph_data)

    # 4. Anchor quality
    missing_anchor_count, nodes_missing_anchors = evaluate_anchor_quality(graph_data)

    # 5. Semantic sanity
    semantic_results = evaluate_semantic_sanity(graph_data, valid_concept_ids)

    # 6. Calculate quality score
    quality_score = calculate_quality_score(
        edge_validity_rate,
        connectivity_rate,
        issues_resolved_rate,
        semantic_results['warnings_count'],
        missing_anchor_count
    )

    # Count various elements
    node_counts = {
        'facts': len(graph_data.get('facts', [])),
        'concepts': len(graph_data.get('concepts', [])),
        'issues': len(graph_data.get('issues', [])),
        'arguments': len(graph_data.get('arguments', [])),
        'holdings': len(graph_data.get('holdings', [])),
        'precedents': len(graph_data.get('precedents', [])),
        'edges': len(edges),
        'reasoning_chains': len(graph_data.get('reasoning_chains', []))
    }

    # Collect invalid edge patterns for aggregation
    invalid_patterns = [e['pattern'] for e in invalid_edges]

    return {
        'case_id': case_id,
        'edge_validity_rate': round(edge_validity_rate, 4),
        'invalid_edge_count': len(invalid_edges),
        'invalid_edge_patterns': invalid_patterns[:5],  # Top 5
        'connectivity_rate': round(connectivity_rate, 4),
        'orphan_count': orphan_count,
        'orphan_nodes': orphan_nodes[:5],  # Top 5
        'issues_resolved_rate': round(issues_resolved_rate, 4),
        'holdings_connected_rate': round(holdings_connected_rate, 4),
        'unresolved_issues': unresolved_issues,
        'missing_anchor_count': missing_anchor_count,
        'warnings_count': semantic_results['warnings_count'],
        'warnings': semantic_results['warnings'],
        'actor_distribution': semantic_results['actor_distribution'],
        'node_counts': node_counts,
        'quality_score': quality_score
    }


def main():
    """Main evaluation function."""
    # Paths
    base_dir = Path(__file__).parent
    graphs_dir = base_dir / 'iltur_graphs'
    output_dir = base_dir / 'evaluation_outputs'
    individual_dir = output_dir / 'individual_results'
    ontology_path = base_dir / 'ontology_compiled.json'

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    individual_dir.mkdir(exist_ok=True)

    # Load ontology for concept validation
    valid_concept_ids = load_ontology(str(ontology_path))
    print(f"Loaded {len(valid_concept_ids)} valid concept IDs from ontology")

    # Find all JSON files (exclude checkpoint.json)
    json_files = sorted([
        f for f in graphs_dir.glob('*.json')
        if f.name != 'checkpoint.json'
    ])

    print(f"Found {len(json_files)} graph files to evaluate")

    # Evaluate all graphs
    all_results = []
    all_invalid_patterns = []

    for i, file_path in enumerate(json_files, 1):
        print(f"Evaluating {i}/{len(json_files)}: {file_path.name}")
        result = evaluate_single_graph(str(file_path), valid_concept_ids)
        all_results.append(result)

        # Save individual result
        individual_path = individual_dir / f"{result['case_id']}_result.json"
        with open(individual_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        # Collect invalid patterns
        all_invalid_patterns.extend(result.get('invalid_edge_patterns', []))

    # Filter out error cases for aggregation
    valid_results = [r for r in all_results if not r.get('is_error')]
    error_results = [r for r in all_results if r.get('is_error')]

    if error_results:
        print(f"\nWarning: {len(error_results)} cases had errors:")
        for r in error_results:
            print(f"  {r['case_id']}: {r.get('error', 'unknown error')}")

    # Aggregate statistics
    quality_scores = [r['quality_score'] for r in valid_results]
    quality_distribution = Counter([int(s) for s in quality_scores])

    if valid_results:
        avg_metrics = {
            'edge_validity_rate': sum(r['edge_validity_rate'] for r in valid_results) / len(valid_results),
            'connectivity_rate': sum(r['connectivity_rate'] for r in valid_results) / len(valid_results),
            'issues_resolved_rate': sum(r['issues_resolved_rate'] for r in valid_results) / len(valid_results),
            'holdings_connected_rate': sum(r['holdings_connected_rate'] for r in valid_results) / len(valid_results),
            'quality_score': sum(quality_scores) / len(quality_scores)
        }
    else:
        avg_metrics = {
            'edge_validity_rate': 0,
            'connectivity_rate': 0,
            'issues_resolved_rate': 0,
            'holdings_connected_rate': 0,
            'quality_score': 0
        }

    # Most common invalid edge patterns
    pattern_counts = Counter(all_invalid_patterns)
    common_patterns = [
        {'pattern': p, 'count': c}
        for p, c in pattern_counts.most_common(10)
    ]

    # Cases needing attention (score < 7)
    problem_cases = sorted(
        [r for r in valid_results if r['quality_score'] < 7],
        key=lambda x: x['quality_score']
    )
    # Also add error cases to problem cases
    problem_cases.extend(error_results)

    # Generate recommendations
    recommendations = []
    for pattern_info in common_patterns[:5]:
        pattern = pattern_info['pattern']
        count = pattern_info['count']
        if count > 5:
            recommendations.append(
                f"Review edge relation validation for pattern '{pattern}' ({count} occurrences)"
            )

    if avg_metrics['connectivity_rate'] < 0.85:
        recommendations.append("Many graphs have orphan nodes - review node extraction")

    if avg_metrics['issues_resolved_rate'] < 0.8:
        recommendations.append("Issues often lack resolving holdings - improve holding-issue linkage")

    # Summary report
    summary = {
        'total_cases': len(all_results),
        'valid_cases': len(valid_results),
        'error_cases': len(error_results),
        'quality_score_distribution': {str(k): v for k, v in sorted(quality_distribution.items())},
        'average_metrics': {k: round(v, 4) for k, v in avg_metrics.items()},
        'most_common_invalid_edge_patterns': common_patterns,
        'cases_needing_attention': [r['case_id'] for r in problem_cases[:10]],
        'recommendations': recommendations
    }

    # Save summary
    summary_path = output_dir / 'summary_report.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary report to {summary_path}")

    # Save problem cases
    problem_cases_list = []
    for r in problem_cases:
        if r.get('is_error'):
            problem_cases_list.append({
                'case_id': r['case_id'],
                'quality_score': r['quality_score'],
                'error': r.get('error', 'unknown')
            })
        else:
            problem_cases_list.append({
                'case_id': r['case_id'],
                'quality_score': r['quality_score'],
                'edge_validity_rate': r['edge_validity_rate'],
                'connectivity_rate': r['connectivity_rate'],
                'issues_resolved_rate': r['issues_resolved_rate'],
                'invalid_edge_count': r['invalid_edge_count'],
                'orphan_count': r['orphan_count'],
                'warnings_count': r['warnings_count']
            })

    problem_cases_data = {
        'count': len(problem_cases),
        'threshold': 7,
        'cases': problem_cases_list
    }

    problem_path = output_dir / 'problem_cases.json'
    with open(problem_path, 'w', encoding='utf-8') as f:
        json.dump(problem_cases_data, f, indent=2)
    print(f"Saved problem cases to {problem_path}")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total cases evaluated: {len(all_results)}")
    print(f"Average quality score: {avg_metrics['quality_score']:.2f}/10")
    print(f"Average edge validity: {avg_metrics['edge_validity_rate']*100:.1f}%")
    print(f"Average connectivity: {avg_metrics['connectivity_rate']*100:.1f}%")
    print(f"Average issues resolved: {avg_metrics['issues_resolved_rate']*100:.1f}%")
    print(f"\nQuality score distribution:")
    for score in range(10, 0, -1):
        count = quality_distribution.get(score, 0)
        bar = '#' * count
        print(f"  {score:2d}: {bar} ({count})")
    print(f"\nCases needing attention (score < 7): {len(problem_cases)}")
    if common_patterns:
        print(f"\nTop invalid edge patterns:")
        for p in common_patterns[:5]:
            print(f"  {p['pattern']}: {p['count']} occurrences")
    print("="*60)


if __name__ == '__main__':
    main()
