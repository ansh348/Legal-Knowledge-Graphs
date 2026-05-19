#!/usr/bin/env python3
"""
rerun_stages_2_4.py

Rerun Stages 2-4 (ontology clustering, intra-cluster edge inference,
cross-cluster edges, justification sets, reasoning chains) on existing
extracted graphs using a new/updated ontology file.

Preserves all Stage 1 outputs (facts, concepts, issues, arguments,
holdings, precedents, outcome) from the original graph files.

Usage:
    python rerun_stages_2_4.py \
        --input_dir iltur_graphs \
        --output_dir graphs_enriched_v2 \
        --ontology ontology_final.json \
        --concurrent 20 \
        --n 2517

Requirements:
    pip install datasets python-dotenv httpx
    XAI_API_KEY in .env file
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

# -- Project imports ----------------------------------------------------------
from schema_v2_1 import (
    LegalReasoningGraph, Edge, Confidence, SCHEMA_VERSION,
)
from extractor import (
    ExtractionConfig,
    GrokClient,
    ConceptCluster,
    IntraClusterEdgesExtractionPass,
    cluster_nodes,
    extract_cross_cluster_edges,
    build_justification_sets_v4,
    synthesize_reasoning_chains_v4,
    dedupe_edges,
    segment_document,
)

logger = logging.getLogger("rerun_stages_2_4")

# -- Clustering calibration (Indian jurisdiction, from run_iltur.py:306-328) --
CLUSTERING_CALIBRATION_IN = {
    "cluster_min_keyword_overlap": 2,
    "cluster_phrase_weight": 5,
    "cluster_case_name_weight": 4,
    "cluster_keyword_weight": 1,
    "cluster_min_score_for_assignment": 3,
}


# =============================================================================
# Helpers
# =============================================================================

def _sanitize_case_id(case_id: str) -> str:
    """Make a safe filename stem from an arbitrary case id."""
    case_id = str(case_id or "").strip()
    if not case_id:
        return "case"
    case_id = case_id.replace("/", "_").replace("\\", "_")
    case_id = re.sub(r"[^0-9A-Za-z._-]+", "_", case_id)
    case_id = re.sub(r"_+", "_", case_id).strip("_")
    return case_id or "case"


def build_text_lookup(hf_dataset: str, hf_config: str, split: str) -> Dict[str, str]:
    """Load HuggingFace dataset and build case_id -> text mapping."""
    from datasets import load_dataset
    logger.info(f"Loading HuggingFace dataset {hf_dataset}/{hf_config} split={split} ...")
    ds = load_dataset(hf_dataset, hf_config)
    # Pick split
    if split in ds:
        data = ds[split]
    elif "single_train" in ds:
        data = ds["single_train"]
    elif "train" in ds:
        data = ds["train"]
    else:
        data = ds[list(ds.keys())[0]]
    lookup = {}
    for row in data:
        cid = _sanitize_case_id(str(row.get("id", "")))
        text = row.get("text", "")
        if cid and text:
            lookup[cid] = text
    logger.info(f"  Loaded {len(lookup)} cases from HuggingFace")
    return lookup


def determine_quality_tier(graph: LegalReasoningGraph) -> str:
    """Replicate quality tier logic from extractor.py:4832-4862."""
    error_patterns = [
        "error", "missing", "not found", "duplicate", "requires anchor",
        "doesn't match", "invalid", "failed", "exceeds"
    ]
    cosmetic_patterns = ["repaired", "coerced", "normalized", "flipped"]
    warnings = graph.validation_warnings or []

    error_count = len([
        w for w in warnings
        if any(p in w.lower() for p in error_patterns)
    ])
    substantive_warning_count = len([
        w for w in warnings
        if not any(p in w.lower() for p in error_patterns)
           and not any(cp in w.lower() for cp in cosmetic_patterns)
    ])

    has_holdings = len(graph.holdings) >= 1
    has_outcome = graph.outcome is not None
    has_chains = len(graph.reasoning_chains) >= 1

    if error_count == 0 and substantive_warning_count <= 15 and has_holdings and has_outcome and has_chains:
        return "gold"
    elif error_count <= 2 and substantive_warning_count <= 30 and has_holdings and has_outcome:
        return "silver"
    elif error_count <= 5:
        return "bronze"
    else:
        return "reject"


# =============================================================================
# Core rerun logic
# =============================================================================

async def rerun_one_case(
    graph: LegalReasoningGraph,
    text: str,
    ontology: Dict,
    config: ExtractionConfig,
    client: GrokClient,
    output_dir: Path,
) -> Dict:
    """Rerun Stages 2-4 for a single case graph."""
    case_id = graph.case_id
    t0 = time.time()

    try:
        # 1. Segment document (needed for IntraClusterEdgesExtractionPass)
        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        doc = segment_document(text, f"sha256:{doc_hash}",
                               section_headers=config.section_headers or None)

        # 2. Clear old Stage 2-4 outputs (keep Stage 1 nodes intact)
        graph.edges = []
        graph.justification_sets = []
        graph.reasoning_chains = []
        graph.cluster_membership = {}
        graph.cluster_summary = {}

        all_warnings = []

        # 3. Pass 7.5: Clustering with new ontology
        clusters, node_membership = cluster_nodes(
            graph, ontology,
            min_keyword_overlap=config.cluster_min_keyword_overlap,
            phrase_weight=config.cluster_phrase_weight,
            case_name_weight=config.cluster_case_name_weight,
            keyword_weight=config.cluster_keyword_weight,
            min_score_for_assignment=config.cluster_min_score_for_assignment,
            jurisdiction=config.jurisdiction,
        )

        # Store cluster membership on graph
        graph.cluster_membership = node_membership
        graph.cluster_summary = {}
        for concept_id, cluster in clusters.items():
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

        # 4. Pass 8: Intra-cluster edges (LLM calls)
        intra_edges: List[Edge] = []
        concept_defs = (ontology or {}).get("concepts", {}) or {}
        llm_calls = 0

        for concept_id, cluster in clusters.items():
            node_count = sum(len(x) for x in [
                cluster.facts, cluster.concepts, cluster.issues,
                cluster.arguments, cluster.precedents, cluster.holdings
            ])
            if node_count < 2:
                continue
            if not (cluster.holdings or cluster.issues or len(cluster.arguments) >= 2):
                continue

            ont_def = concept_defs.get(concept_id)
            try:
                intra_pass = IntraClusterEdgesExtractionPass(
                    client, config, doc,
                    cluster=cluster,
                    graph=graph,
                    ontology_concept=ont_def,
                )
                intra_result = await intra_pass.extract()
                llm_calls += 1
                if intra_result.success:
                    tag = hashlib.sha1(concept_id.encode("utf-8")).hexdigest()[:8]
                    prefix = f"e_{tag}_"
                    intra_edges.extend(intra_pass.to_edges(intra_result.data, edge_id_prefix=prefix))
                else:
                    all_warnings.extend(intra_result.errors)
                all_warnings.extend(intra_result.warnings)
            except Exception as e:
                all_warnings.append(f"Intra-cluster crash for {concept_id}: {type(e).__name__}: {e}")

        graph.edges = intra_edges

        # 5. Pass 8.5: Cross-cluster edges (deterministic)
        cross_edges = extract_cross_cluster_edges(graph)
        graph.edges = dedupe_edges(graph.edges + cross_edges)

        # 6. Pass 9: Justification sets (deterministic)
        try:
            graph.justification_sets = build_justification_sets_v4(graph, clusters)
        except Exception as e:
            all_warnings.append(f"Justification sets crash: {type(e).__name__}: {e}")
            graph.justification_sets = []

        # 7. Pass 10: Reasoning chains (deterministic)
        try:
            graph.reasoning_chains = synthesize_reasoning_chains_v4(graph)
        except Exception as e:
            all_warnings.append(f"Reasoning chains crash: {type(e).__name__}: {e}")
            graph.reasoning_chains = []

        # 8. Validation + quality tier
        try:
            validation_warnings = graph.validate()
            all_warnings.extend(validation_warnings)
        except Exception as e:
            all_warnings.append(f"Validation crash: {type(e).__name__}: {e}")

        graph.validation_warnings = all_warnings
        graph.quality_tier = determine_quality_tier(graph)

        # Update meta timestamps
        graph.extraction_timestamp = datetime.now(timezone.utc).isoformat()

        # 9. Save atomically
        output_path = output_dir / f"{case_id}.json"
        tmp_path = output_dir / f"{case_id}.json.tmp"
        try:
            json_str = graph.to_json()
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            os.replace(tmp_path, output_path)
        except Exception as ser_err:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return {
                "case_id": case_id, "success": False,
                "error": f"Serialization error: {str(ser_err)[:150]}",
            }

        elapsed = time.time() - t0
        return {
            "case_id": case_id,
            "success": True,
            "quality_tier": graph.quality_tier,
            "clusters": len(graph.cluster_summary),
            "edges": len(graph.edges),
            "chains": len(graph.reasoning_chains),
            "js": len(graph.justification_sets),
            "llm_calls": llm_calls,
            "elapsed": round(elapsed, 1),
        }

    except Exception as e:
        return {
            "case_id": case_id, "success": False,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


async def rerun_one_case_limited(
    graph: LegalReasoningGraph,
    text: str,
    ontology: Dict,
    config: ExtractionConfig,
    client: GrokClient,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Wrapper with concurrency limiter and per-case timeout."""
    async with semaphore:
        try:
            return await asyncio.wait_for(
                rerun_one_case(graph, text, ontology, config, client, output_dir),
                timeout=600,  # 10 min per case
            )
        except asyncio.TimeoutError:
            return {
                "case_id": graph.case_id, "success": False,
                "error": "Timeout (600s)",
            }


# =============================================================================
# Checkpoint
# =============================================================================

def load_completed(output_dir: Path) -> Set[str]:
    """Scan output directory for already-processed case IDs."""
    completed = set()
    if output_dir.exists():
        for json_file in output_dir.glob("*.json"):
            if json_file.name == "checkpoint.json":
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json.load(f)  # validate JSON
                completed.add(json_file.stem)
            except Exception:
                try:
                    json_file.unlink()
                except OSError:
                    pass
    return completed


def save_checkpoint(stats: Dict, output_dir: Path):
    """Save progress checkpoint."""
    cp = output_dir / "checkpoint.json"
    stats["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Rerun Stages 2-4 with a new ontology on existing graphs"
    )
    parser.add_argument("--input_dir", type=str, default="iltur_graphs",
                        help="Directory containing existing extracted graphs")
    parser.add_argument("--output_dir", type=str, default="graphs_enriched_v2",
                        help="Output directory for re-processed graphs")
    parser.add_argument("--ontology", type=str, required=True,
                        help="Path to the new ontology JSON file")
    parser.add_argument("--n", type=int, default=0,
                        help="Max cases to process (0 = all)")
    parser.add_argument("--concurrent", type=int, default=20,
                        help="Max concurrent extractions")
    parser.add_argument("--hf_dataset", type=str, default="Exploration-Lab/IL-TUR",
                        help="HuggingFace dataset for original text")
    parser.add_argument("--hf_config", type=str, default="cjpe",
                        help="HuggingFace dataset config")
    parser.add_argument("--hf_split", type=str, default="single_train",
                        help="HuggingFace dataset split")
    parser.add_argument("--jurisdiction", type=str, default="in",
                        help="Jurisdiction code (in, echr, tr)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Count clusters and qualifying LLM calls without making API calls")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("rerun_stages_2_4.log", encoding="utf-8"),
        ],
    )

    # Validate inputs
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ontology_path = Path(args.ontology)

    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' not found")
        return
    if not ontology_path.exists():
        print(f"Error: ontology file '{ontology_path}' not found")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    # Load ontology
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    n_concepts = len((ontology or {}).get("concepts", {}))
    print(f"Ontology: {ontology_path} ({n_concepts} concepts)")

    # Load original text from HuggingFace
    text_lookup = build_text_lookup(args.hf_dataset, args.hf_config, args.hf_split)

    # Scan input graphs
    graph_files = sorted([
        f for f in input_dir.glob("*.json")
        if f.name != "checkpoint.json"
    ])
    print(f"Found {len(graph_files)} graph files in '{input_dir}'")

    # Check already completed in output
    completed = load_completed(output_dir)
    print(f"Already completed in output: {len(completed)}")

    # Filter to unprocessed
    to_process = []
    skipped_no_text = 0
    for gf in graph_files:
        case_id = gf.stem
        if case_id in completed:
            continue
        if case_id not in text_lookup:
            skipped_no_text += 1
            continue
        to_process.append(gf)

    if args.n > 0:
        to_process = to_process[:args.n]

    print(f"Cases to process: {len(to_process)} (skipped no-text: {skipped_no_text})")

    if not to_process:
        print("Nothing to process!")
        return

    # --- Dry run mode ---
    if args.dry_run:
        print("\n=== DRY RUN — estimating LLM calls ===")
        cal = CLUSTERING_CALIBRATION_IN
        total_clusters = 0
        total_qualifying = 0
        for gf in to_process:
            with open(gf, "r", encoding="utf-8") as f:
                d = json.load(f)
            graph = LegalReasoningGraph.from_dict(d)
            clusters, _ = cluster_nodes(
                graph, ontology,
                min_keyword_overlap=cal["cluster_min_keyword_overlap"],
                phrase_weight=cal["cluster_phrase_weight"],
                case_name_weight=cal["cluster_case_name_weight"],
                keyword_weight=cal["cluster_keyword_weight"],
                min_score_for_assignment=cal["cluster_min_score_for_assignment"],
                jurisdiction=args.jurisdiction,
            )
            qualifying = 0
            for cid, cl in clusters.items():
                nc = sum(len(x) for x in [cl.facts, cl.concepts, cl.issues,
                                           cl.arguments, cl.precedents, cl.holdings])
                if nc < 2:
                    continue
                if not (cl.holdings or cl.issues or len(cl.arguments) >= 2):
                    continue
                qualifying += 1
            total_clusters += len(clusters)
            total_qualifying += qualifying
        avg_clusters = total_clusters / len(to_process) if to_process else 0
        avg_qualifying = total_qualifying / len(to_process) if to_process else 0
        print(f"  Cases:              {len(to_process)}")
        print(f"  Total clusters:     {total_clusters} (avg {avg_clusters:.1f}/case)")
        print(f"  Qualifying (LLM):   {total_qualifying} (avg {avg_qualifying:.1f}/case)")
        print(f"  Est. input tokens:  {total_qualifying * 3000 / 1e6:.1f}M")
        print(f"  Est. output tokens: {total_qualifying * 2000 / 1e6:.1f}M")
        print(f"  Est. cost (Grok):   ${total_qualifying * 3000 * 0.20 / 1e6 + total_qualifying * 2000 * 0.60 / 1e6:.2f}")
        return

    # --- Real run ---
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not set in .env")
        return

    model_id = "grok-4-1-fast-reasoning"
    client = GrokClient(api_key, model_id=model_id)

    cal = CLUSTERING_CALIBRATION_IN
    config = ExtractionConfig(
        model_id=model_id,
        pipeline_version="v4",
        ontology_path=str(ontology_path),
        ontology_data=ontology,
        jurisdiction=args.jurisdiction,
        cluster_min_keyword_overlap=cal["cluster_min_keyword_overlap"],
        cluster_phrase_weight=cal["cluster_phrase_weight"],
        cluster_case_name_weight=cal["cluster_case_name_weight"],
        cluster_keyword_weight=cal["cluster_keyword_weight"],
        cluster_min_score_for_assignment=cal["cluster_min_score_for_assignment"],
    )

    semaphore = asyncio.Semaphore(args.concurrent)

    print(f"\nPipeline: v4 (Stages 2-4 only)")
    print(f"Model: {model_id}")
    print(f"Ontology: {ontology_path}")
    print(f"Concurrency: {args.concurrent}")
    print(f"Output: {output_dir}")
    print()

    stats = {
        "success": 0, "errors": 0,
        "total_edges": 0, "total_chains": 0,
        "total_llm_calls": 0,
        "quality_gold": 0, "quality_silver": 0,
        "quality_bronze": 0, "quality_reject": 0,
    }

    # Process in batches of 50 for checkpoint saving
    batch_size = 200
    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start:batch_start + batch_size]

        # Load graphs and pair with text
        tasks = []
        for gf in batch:
            case_id = gf.stem
            with open(gf, "r", encoding="utf-8") as f:
                d = json.load(f)
            graph = LegalReasoningGraph.from_dict(d)
            text = text_lookup[case_id]
            tasks.append(
                rerun_one_case_limited(
                    graph, text, ontology, config, client, output_dir, semaphore
                )
            )

        results = await asyncio.gather(*tasks)

        for r in results:
            if r["success"]:
                stats["success"] += 1
                stats["total_edges"] += r.get("edges", 0)
                stats["total_chains"] += r.get("chains", 0)
                stats["total_llm_calls"] += r.get("llm_calls", 0)
                tier = r.get("quality_tier", "bronze")
                stats[f"quality_{tier}"] = stats.get(f"quality_{tier}", 0) + 1
                logger.info(
                    f"  OK {r['case_id']}: {r['quality_tier']} | "
                    f"{r['clusters']} clusters, {r['edges']} edges, "
                    f"{r['chains']} chains, {r['llm_calls']} LLM calls, "
                    f"{r['elapsed']}s"
                )
            else:
                stats["errors"] += 1
                logger.error(f"  FAIL {r['case_id']}: {r.get('error', '?')}")

        processed = stats["success"] + stats["errors"]
        print(
            f"[{processed}/{len(to_process)}] "
            f"OK={stats['success']} ERR={stats['errors']} "
            f"Gold={stats['quality_gold']} Silver={stats['quality_silver']} "
            f"LLM calls={stats['total_llm_calls']}"
        )
        save_checkpoint(stats, output_dir)

    # Final summary
    await client.close()
    print(f"\n{'=' * 60}")
    print(f"DONE — Rerun Stages 2-4 with {ontology_path}")
    print(f"  Success:     {stats['success']}")
    print(f"  Errors:      {stats['errors']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Total chains:{stats['total_chains']}")
    print(f"  LLM calls:   {stats['total_llm_calls']}")
    print(f"  Gold:   {stats['quality_gold']}")
    print(f"  Silver: {stats['quality_silver']}")
    print(f"  Bronze: {stats['quality_bronze']}")
    print(f"  Reject: {stats['quality_reject']}")
    print(f"  Grok token usage: {client.total_prompt_tokens:,} prompt + {client.total_completion_tokens:,} completion")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
